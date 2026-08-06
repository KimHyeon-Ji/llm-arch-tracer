"""Entry point. Usage: python run.py --profile models/<id>.yaml --out out/
Add --check-repro to actually run twice and verify C13 (off by default since it
doubles runtime -- see 01-main.md C13 and develop/04-verification-plan.md "통과 기준").

Wires together provenance -> loader -> introspect -> inputs -> tracer (via the
adaptive Tier 0/1 loop) -> normalize -> build_table -> validate -> report.
See 01-main.md for the step-by-step spec this implements."""
import argparse
import json
import os
import sys
import yaml

sys.path.insert(0, os.path.dirname(__file__))

import provenance
import loader
import introspect
import inputs as input_builder
import normalize
import build_table
import validate
import research
import summarize
import symbolic_shape
import symbolic_dims
from scope import ScopeLabeler
from tracer import OpGraphTracer
from adapt import trace_adaptive


class RunContext:
    """Mutable execution state the adaptive loop (adapt.py) can rewrite between
    retries: backend (meta/fake), attn_implementation, seq_len, cache."""

    def __init__(self, cfg, model_id, revision):
        self.cfg = cfg
        self.model_id = model_id
        self.revision = revision
        self.backend = "meta"
        self.attn = getattr(cfg, "attn_implementation", None)
        # prefer native transformers impl; only use remote code when no native support exists
        # (stale remote modeling breaks on new transformers -- see provenance.needs_remote_code)
        self.trust_remote = provenance.needs_remote_code(cfg)
        # collision-free seq_len so T (and T*k, T+1, ...) never equals a config dim value,
        # keeping symbolic shapes unambiguous (symbolic_shape.py). Still >= the derived min,
        # so C14 holds.
        self.seq_len = symbolic_shape.resolve_seq_len(cfg, introspect.derive_min_seq_len(cfg))
        self.dtype = None  # parameter dtype for the load; a remedy can bump it to bf16 (see use_bf16)
        self.model = None
        self.last_past_key_values = None

    def _load(self):
        if self.attn is not None:
            self.cfg.attn_implementation = self.attn
        if self.backend == "fake":
            self.model, self._fake_mode = loader.load_fake(
                self.cfg, trust_remote_code=self.trust_remote, dtype=self.dtype)
        else:
            self.model = loader.load_meta(
                self.cfg, trust_remote_code=self.trust_remote, dtype=self.dtype)
        return self.model

    def run_once(self, phase: str):
        model = self._load()
        scope = ScopeLabeler(model)
        tracer = OpGraphTracer(model, scope)
        kwargs = input_builder.build_inputs(
            model, self.cfg, phase, self.seq_len, past=self.last_past_key_values
        )
        import torch

        with torch.no_grad(), tracer:
            out = model(**kwargs)
        scope.remove()
        if phase == "prefill" and hasattr(out, "past_key_values"):
            self.last_past_key_values = out.past_key_values
        return tracer.rows

    # -- remedies the adaptive loop can invoke by name, see rules/error_remedies.yaml --
    def switch_backend(self, name: str = "fake"):
        self.backend = name

    def meta_to_fake(self):
        self.backend = "fake"

    def set_attn(self, name: str):
        self.attn = name

    def attn_sdpa(self):
        self.attn = "sdpa"

    def attn_eager(self):
        self.attn = "eager"

    def use_bf16(self):
        # some kernels assert a specific param dtype (e.g. gpt-oss MoE grouped-matmul wants BF16);
        # shapes are dtype-independent, so reloading in bf16 is safe and only unblocks the trace.
        import torch
        self.dtype = torch.bfloat16

    def bump_seq(self, factor: int = 2):
        self.seq_len *= factor

    def bump_seq_len(self):
        self.bump_seq(2)

    def rebuild_cache(self):
        self.last_past_key_values = None


def _extract(profile: dict, cfg):
    """One full extraction pass (all phases). Returns (ctx, all_rows, adaptation_log)."""
    ctx = RunContext(cfg, profile["model_id"], profile.get("revision"))
    all_rows = {}
    adaptation_log = []
    for phase in profile.get("phases", ["prefill", "decode"]):
        rows, applied = trace_adaptive(ctx, phase)
        rows = normalize.normalize_rows(rows)
        # stamp phase here (part of the canonical schema) so it is identical across runs.
        # Otherwise build_table.write_outputs() adds it to run 1's rows as a side-effect
        # before the C13 comparison, making run 1 (stamped) != run 2 (unstamped) -- a false
        # reproducibility mismatch even though the trace itself is deterministic.
        for r in rows:
            r["phase"] = phase
        all_rows[phase] = rows
        adaptation_log.extend(applied)
    return ctx, all_rows, adaptation_log


def run(profile_path: str, out_dir: str, check_repro: bool = False):
    with open(profile_path, encoding="utf-8") as f:
        profile = yaml.safe_load(f)

    model_id = profile["model_id"]
    cfg, prov = provenance.snapshot(
        model_id, profile.get("revision"), config_overrides=profile.get("config_overrides"))
    provenance.support_gate(cfg)

    model_dir = os.path.join(out_dir, model_id.replace("/", "__"))
    # top level holds only the "headline" files: major-op <phase>.csv/.jsonl, structure.yaml,
    # model_summary.md. Everything else (full trace, provenance, report) goes under full/.
    full_dir = os.path.join(model_dir, build_table.FULL_SUBDIR)
    os.makedirs(full_dir, exist_ok=True)

    ctx, all_rows, adaptation_log = _extract(profile, cfg)
    prov["adaptation_log"].extend(adaptation_log)

    # shapes are written symbolically; the resolver maps concrete dims -> B/T/d_model/E/...
    # (symbolic_shape.py). Its .table (symbol -> concrete value) goes into provenance so the
    # numbers stay recoverable (01-main.md P1 / section 10).
    resolver = symbolic_shape.build_resolver(cfg, ctx.seq_len)
    # per-module width expressions read off a tagged build (src/symbolic_dims.py); they outrank
    # value matching wherever they name every dimension they use
    probe = symbolic_dims.probe(model_id, profile.get("revision"), profile.get("config_overrides"))
    tags = probe.get("expressions") or {}
    param_axes = probe.get("param_axes") or {}
    # each phase writes its own csv / trace.raw.jsonl -- see build_table.py
    for phase, rows in all_rows.items():
        build_table.write_outputs(model_dir, phase, rows, resolver, tags, param_axes)

    prov["capture_backend"] = ctx.backend
    prov["seq_len_used"] = ctx.seq_len
    prov["attn_implementation_used"] = ctx.attn
    prov["symbol_table"] = resolver.table

    # entrypoint discovery (01-main.md Step 3). Actually tracing a discovered entrypoint
    # needs an architecture-specific input builder; the profile can supply one via
    # `entrypoints:` (see USAGE.md). Anything discovered but not listed there is reported,
    # not silently dropped (P8) -- see C15 below.
    discovered = {name for name, _cls in introspect.find_extra_entrypoints(ctx.model)}
    declared = {e["module_path"] for e in profile.get("entrypoints", []) if isinstance(e, dict)}
    traced_entrypoints = declared & discovered  # TODO: actually trace these once a generic
    # cross-architecture input-builder exists; for now this stays empty unless wired manually.

    provenance.write_provenance(os.path.join(full_dir, "provenance.json"), prov)

    named_params = list(ctx.model.named_parameters())
    param_names = {n for n, _ in named_params}
    # param scale for model_summary (meta tensors: numel is just the shape product, no weights).
    total_params = sum(p.numel() for _, p in named_params)
    expert_numel = sum(p.numel() for n, p in named_params if ".expert" in n)
    _E = getattr(cfg, "num_experts", None) or getattr(cfg, "n_routed_experts", None) \
        or getattr(cfg, "num_local_experts", None)
    _k = getattr(cfg, "num_experts_per_tok", None) or getattr(cfg, "moe_topk", None)
    active_params = (int(total_params - expert_numel * (1 - _k / _E))
                     if expert_numel and _E and _k else total_params)
    scale = {"total_params": total_params, "active_params": active_params, "expert_params": expert_numel}
    min_seq = introspect.derive_min_seq_len(cfg)
    prefill_rows = all_rows.get("prefill", [])
    decode_rows = all_rows.get("decode", [])

    checks = {
        "C1": validate.c1_layer_count(prefill_rows, cfg),
        "C2": validate.c2_layer_clustering(prefill_rows, cfg),
        "C3": validate.c3_dag_integrity(prefill_rows),
        "C4": validate.c4_reachability(prefill_rows),
        "C5": validate.c5_connection(prefill_rows, cfg),
        "C6": validate.c6_hidden_head_consistency(prefill_rows, cfg),
        "C7": validate.c7_gqa(cfg),
        "C8": validate.c8_moe(prefill_rows, cfg),
        "C9": validate.c9_embed_lm_head(prefill_rows, cfg),
        "C10": validate.c10_coverage(prefill_rows, param_names),
        "C11": validate.c11_decode_consistency(decode_rows),
        "C14": validate.c14_seq_len(ctx.seq_len, min_seq),
        "C15": validate.c15_entrypoint_coverage(traced_entrypoints, discovered, cfg),
        "C16": validate.c16_unmapped(prefill_rows),
    }

    if check_repro:
        _, all_rows_2, _ = _extract(profile, cfg)
        checks["C13"] = validate.c13_reproducibility(all_rows, all_rows_2)
    else:
        checks["C13"] = ("SKIP", "pass --check-repro to actually run twice and verify")

    structure = summarize.build_structure(prefill_rows, cfg, model_id, prov["revision_resolved"])
    # fixed, config-derived dims the symbolizer left as literals (e.g. MLA kv_b_proj = 32768) --
    # recorded so a reader can tell a "mystery number" from a bug (resolver symbolizes concrete rows).
    literals = summarize.find_literal_dims(prefill_rows, structure["symbols"], resolver,
                                           cfg=cfg, seq_len=resolver.table.get("T"))
    structure["literal_dims"] = literals
    # Which config fields this architecture uses that rules/symbols.yaml does not know about.
    # A separate throwaway build so a labelling experiment can never perturb the trace above.
    structure["unregistered_fields"] = probe.get("unregistered", [])
    structure["label_provenance"] = summarize.label_provenance(resolver)
    # Phase 0 onboarding gate -- runs even when everything else passed, which is the whole point
    # (see 02-new-module-handling.md: DeepSeek-V4 passed C1-C16 with 5 undocumented modules).
    checks["C17"] = validate.c17_module_onboarding(
        literals, structure["symbols"],
        structures_dir=os.path.join(os.path.dirname(__file__), "..", "rules", "structures"),
        model_type=getattr(cfg, "model_type", None), model_id=model_id)
    summarize.write_structure(model_dir, structure, fmt=profile.get("structure_format", "yaml"))
    # Which axes this model could not settle on its own, and which source answers each
    # (02-new-module-handling.md Tier 2). Written next to the summary so the decision
    # "this needs architecture research" is produced by the tool, not by whoever reads it.
    research.build(model_dir, model_id, getattr(cfg, "model_type", None), structure)


    sources = []
    sources_path = profile.get("sources_file")  # optional: agent-supplied Tier 2 findings
    if sources_path and os.path.exists(sources_path):
        with open(sources_path, encoding="utf-8") as f:
            sources = yaml.safe_load(f) or []
    summary_md = summarize.render_model_summary(
        model_id, prov, structure, cfg=cfg, rows=prefill_rows, scale=scale,
        checks=checks, sources=sources, literals=literals)
    summarize.write_model_summary(model_dir, summary_md)

    # report.md only (human-readable). No report.json: the same check results are also embedded
    # in model_summary.md's 검증 로그 table, and report.md is line-parseable if a machine needs it.
    report_path = os.path.join(full_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Extraction Report -- {model_id} @ {prov['revision_resolved']}\n\n")
        for cid in sorted(checks, key=lambda c: int(c[1:])):
            status, detail = checks[cid]
            f.write(f"{cid:4s} {status:6s} {detail}\n")
        if discovered - declared:
            f.write(f"\nnote: discovered but not declared in profile entrypoints: {sorted(discovered - declared)}\n")
    print(f"wrote {report_path}")

    # Verification layer 3 (free-form review) -- ALWAYS generated, never a manual afterthought.
    # Layers 1/2 (rules + first-principles checks) can only catch failure modes we already met;
    # every serious labelling bug in this project was found by layer 3 and missed by the gate.
    # Emitting the packet with the run guarantees it exists and matches THIS trace. Reading it
    # is still a reviewer's job (02-new-module-handling.md), but a stale packet is worse than
    # none -- the ones in develop/review/ predated the anchoring change entirely.
    print(f"wrote {_write_review_packet(model_dir)}")


def _write_review_packet(model_dir: str) -> str:
    """develop/ is tooling, src/ is the pipeline, so this import is deliberately local and
    non-fatal: a packet-generation failure must never lose a completed trace."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "develop"))
        import make_review_packet
        return make_review_packet.write_packet(model_dir)
    except Exception as e:                       # noqa: BLE001 -- reporting, not control flow
        return f"(review packet SKIPPED: {type(e).__name__}: {str(e)[:120]})"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--out", default="out")
    ap.add_argument("--check-repro", action="store_true",
                     help="run extraction twice and verify C13 reproducibility (doubles runtime)")
    args = ap.parse_args()
    run(args.profile, args.out, check_repro=args.check_repro)
