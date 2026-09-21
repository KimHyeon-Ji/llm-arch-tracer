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
import review_request
import source_check
import summarize
import symbolic_shape
import symbolic_dims
from scope import ScopeLabeler
from tracer import OpGraphTracer
import semantic_events
from adapt import trace_adaptive


class RunContext:
    """Mutable execution state the adaptive loop (adapt.py) can rewrite between
    retries: backend (meta/fake), attn_implementation, seq_len, cache."""

    def __init__(self, cfg, model_id, revision, seq_len=None, batch=None,
                 seq_len_multiple=None):
        self.cfg = cfg
        # 아키텍처가 거는 길이 제약(예: KDA 청크 스캔의 `T % 64 == 0`). 프로파일이 선언한다.
        self.seq_len_multiple = seq_len_multiple

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
        # A profile may pin the base seq_len. USAGE documented it but nothing read it, so an
        # architecture with a hard length requirement had no way to state it: KDA's chunked scan
        # asserts `T % 64 == 0` (fla/ops/kda/naive.py:112) and the derived minimum is 16, so
        # Kimi-K3 could never run (2026-08-10). Collision avoidance still applies on top, so the
        # symbolic shapes stay unambiguous either way.
        _base = seq_len if isinstance(seq_len, int) else introspect.derive_min_seq_len(cfg)
        # **배치도 함께 고른다.** B=1 이면 배치·decode query 길이·방송 싱글턴이 수치적으로
        # 같고 `B*T == T`, `B*n_h == n_h` 라 접힌 배치가 라벨에서 통째로 사라진다 -- Llama-4
        # 에서 그런 축이 3,962개였다(외부 검토 2026-09-13). 이 저장소는 같은 문제를 T 에
        # 대해 이미 풀어놨는데(`resolve_seq_len`) 배치만 1 로 박혀 있었다.
        #
        # `batch` 를 명시로 주면 그것을 쓴다 -- 다른 배치로 라벨을 재평가하는 검증 트레이스용.
        if batch is None:
            self.batch, self.seq_len = symbolic_shape.resolve_capture_sizes(
                cfg, _base, multiple=self.seq_len_multiple)
        else:
            self.batch = batch
            self.seq_len = symbolic_shape.resolve_seq_len(cfg, _base)
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
        tracer = OpGraphTracer(model, scope, phase=phase)
        kwargs = input_builder.build_inputs(
            model, self.cfg, phase, self.seq_len, past=self.last_past_key_values,
            batch=self.batch,
        )
        import torch

        # ATen 에 안 보이는 의미 경계를 함께 기록한다 -- `repeat_kv(n_rep=1)` 은 op 자체가
        # 발생하지 않아 provenance 만으로는 n_kv/n_h 가 갈리지 않고, `Cache.update` 의
        # key/value 인자는 파이썬 레벨이라 concat 순서로 역산할 수밖에 없다.
        # 기록만 한다 -- 라벨 결정에는 쓰지 않는다(외부 검토 2026-09-09 의 3단계).
        # 모델 호출 인자를 `graph_input` 으로 등록한다. 이게 없으면 `input_ids` 같은 것이
        # `unknown_external` 로 남아 "모른다" 와 구분되지 않는다.
        tracer.register_graph_inputs(kwargs)
        with torch.no_grad(), semantic_events.SemanticWrappers(tracer, model), tracer:
            out = model(**kwargs)
        self.last_semantic_events = semantic_events.events()
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




def _extract(profile: dict, cfg, batch=None):
    """One full extraction pass (all phases). Returns (ctx, all_rows, adaptation_log).

    `batch` 를 주면 발행 배치 대신 그것으로 잡는다 -- **진단용**이다. 같은 규칙으로 B 만
    바꿔 뜨면 전환 diff 에서 "규칙이 바뀌어서 생긴 차이"와 "배치가 바뀌어서 생긴 차이"를
    갈라 볼 수 있다. 섞여 있으면 어느 쪽이 회귀인지 판단할 수 없다 (2026-09-17).
    """
    _sl = profile.get("seq_len")
    ctx = RunContext(cfg, profile["model_id"], profile.get("revision"),
                     batch=batch,
                     seq_len=_sl if isinstance(_sl, int) else None,
                     seq_len_multiple=profile.get("seq_len_multiple"))
    all_rows = {}
    sem_events = {}
    adaptation_log = []
    for phase in profile.get("phases", ["prefill", "decode"]):
        rows, applied = trace_adaptive(ctx, phase)
        rows = normalize.normalize_rows(rows)
        # 의미 이벤트는 행이 아니라 phase 단위다. 여기서 받아 두고 아래에서 사이드카로 쓴다.
        sem_events[phase] = list(getattr(ctx, "last_semantic_events", None) or [])
        # stamp phase here (part of the canonical schema) so it is identical across runs.
        # Otherwise build_table.write_outputs() adds it to run 1's rows as a side-effect
        # before the C13 comparison, making run 1 (stamped) != run 2 (unstamped) -- a false
        # reproducibility mismatch even though the trace itself is deterministic.
        for r in rows:
            r["phase"] = phase
        all_rows[phase] = rows
        adaptation_log.extend(applied)
    # loader.load_meta/load_fake stash their own (non-retry) remedies -- KDA torch-reference
    # install, MoE expert-cap -- on the model object, since those aren't triggered through the
    # error-retry path trace_adaptive covers. Merge them in here so provenance and the C10 check
    # both see them; dedup because every phase reload produces the same entry again.
    for e in getattr(ctx.model, "_adaptation_extra", None) or []:
        if e not in adaptation_log:
            adaptation_log.append(e)
    return ctx, all_rows, adaptation_log, sem_events


def run(profile_path: str, out_dir: str, check_repro: bool = False, batch=None):
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

    ctx, all_rows, adaptation_log, sem_events = _extract(profile, cfg, batch=batch)
    prov["adaptation_log"].extend(adaptation_log)

    # shapes are written symbolically; the resolver maps concrete dims -> B/T/d_model/E/...
    # (symbolic_shape.py). Its .table (symbol -> concrete value) goes into provenance so the
    # numbers stay recoverable (01-main.md P1 / section 10).
    resolver = symbolic_shape.build_resolver(cfg, ctx.seq_len, batch=ctx.batch)
    # per-module width expressions read off a tagged build (src/symbolic_dims.py); they outrank
    # value matching wherever they name every dimension they use
    probe = symbolic_dims.probe(model_id, profile.get("revision"), profile.get("config_overrides"))
    tags = probe.get("expressions") or {}
    param_axes = probe.get("param_axes") or {}
    # each phase writes its own csv / trace.raw.jsonl -- see build_table.py
    for phase, rows in all_rows.items():
        # 의미 이벤트를 **인자로** 넘긴다. 사이드카는 아래에서 쓰지만, write_outputs 가 그
        # 파일을 읽게 두면 이전 실행의 것을 읽는다(실행 순서상 아직 안 쓰였다).
        build_table.write_outputs(model_dir, phase, rows, resolver, tags, param_axes,
                                  semantic_events=sem_events.get(phase) or [],
                                  adaptation_log=prov.get("adaptation_log") or [])
        # ATen 에 안 보이는 의미 경계(`repeat_kv` 의 no-op, `Cache.update` 의 key/value 인자).
        # 관측층이라 라벨과 무관하고, 축 계보를 세울 때 barrier 와 역할 근거로 쓴다.
        _ev = sem_events.get(phase) or []
        _sp = os.path.join(model_dir, "full", f"{phase}.semantic.jsonl")
        with open(_sp, "w", encoding="utf-8") as _f:
            for _e in _ev:
                _f.write(json.dumps(_e, ensure_ascii=False) + chr(10))

    prov["capture_backend"] = ctx.backend
    prov["seq_len_used"] = ctx.seq_len
    prov["capture_batch"] = ctx.batch
    prov["attn_implementation_used"] = ctx.attn
    prov["symbol_table"] = resolver.table

    # entrypoint discovery (01-main.md Step 3). Actually tracing a discovered entrypoint
    # needs an architecture-specific input builder; the profile can supply one via
    # `entrypoints:` (see README, "새 모델 뽑기"). Anything discovered but not listed there is reported,
    # not silently dropped (P8) -- see C15 below.
    discovered = {name for name, _cls in introspect.find_extra_entrypoints(ctx.model)}
    declared = {e["module_path"] for e in profile.get("entrypoints", []) if isinstance(e, dict)}
    traced_entrypoints = declared & discovered  # TODO: actually trace these once a generic
    # cross-architecture input-builder exists; for now this stays empty unless wired manually.

    provenance.write_provenance(os.path.join(full_dir, "provenance.json"), prov)
    provenance.write_stamp(model_dir)

    # module path -> source class, so the label check can ask the SOURCE whether a module may
    # carry a given config field (see introspect.module_classes / source_check.membership_gaps).
    with open(os.path.join(full_dir, "module_classes.json"), "w", encoding="utf-8") as f:
        json.dump(introspect.module_classes(ctx.model), f, ensure_ascii=False, indent=1)

    named_params = list(ctx.model.named_parameters())
    param_names = {n for n, _ in named_params}
    # param scale for model_summary (meta tensors: numel is just the shape product, no weights).
    total_params = sum(p.numel() for _, p in named_params)
    expert_numel = sum(p.numel() for n, p in named_params if ".expert" in n)
    _E = getattr(cfg, "num_experts", None) or getattr(cfg, "n_routed_experts", None) \
        or getattr(cfg, "num_local_experts", None)
    _k = getattr(cfg, "num_experts_per_tok", None) or getattr(cfg, "moe_topk", None)

    def _per_layer_scalar(v):
        """A config field that may be stated PER LAYER. Hunyuan-A13B writes `moe_topk: [8, 8, ...]`
        and `num_experts` the same way, one entry per decoder layer, which made the active-parameter
        arithmetic divide a list by an int and abort the whole run. A single number is what this
        estimate needs; the layers agree in every checkpoint seen so far, and if they ever disagree
        the mean is the honest summary of "how many experts a token activates"."""
        if isinstance(v, (list, tuple)):
            nums = [x for x in v if isinstance(x, int) and not isinstance(x, bool)]
            return sum(nums) / len(nums) if nums else None
        return v

    _E, _k = _per_layer_scalar(_E), _per_layer_scalar(_k)
    active_params = (int(total_params - expert_numel * (1 - _k / _E))
                     if expert_numel and _E and _k else total_params)
    scale = {"total_params": total_params, "active_params": active_params, "expert_params": expert_numel}
    min_seq = introspect.derive_min_seq_len(cfg)
    prefill_rows = all_rows.get("prefill", [])
    decode_rows = all_rows.get("decode", [])

    # C10 exception: params for experts the moe_infer_even_split remedy (src/kda_shim.py)
    # deliberately didn't run. Not a coverage miss -- the same structure, unexecuted -- so
    # it's excluded from FAIL and reported separately, the same way C8 treats routed-token
    # counts as expected-missing rather than a defect.
    def _expert_cap_gap(log, names):
        import re
        gap = set()
        for e in log:
            if e.get("remedy") != "moe_infer_even_split" or not e.get("expert_cap"):
                continue
            cap = e["expert_cap"]
            gap |= {p for p in names
                    if (m := re.search(r"\.experts\.(\d+)\.", p)) and int(m.group(1)) >= cap}
        return gap

    # C10 exception: Kimi-Linear/Kimi-K3's block-residual mechanism (KimiDecoderLayer, gated by
    # config.attn_res_block_size) seeds `block_residual` with a hardcoded 0-length "num_blocks"
    # axis on every forward call (`hidden_states.new_zeros(..., 0, ...)` in the model's own
    # source) and only calls self_attention_res_proj/self_attention_res_norm when
    # `block_residual.shape[1] > 0`. Layer 0 is always the first to see block_residual, so its
    # OWN copies of those two params can never receive an op for ANY input -- this is a fact
    # about the model's control flow, not a property of this particular trace, and later layers'
    # copies of the same params are used normally once the buffer has grown. Not a coverage miss.
    def _attn_res_layer0_gap(cfg, names):
        import re
        if not getattr(cfg, "attn_res_block_size", None):
            return set()
        return {p for p in names
                if re.search(r"^model\.layers\.0\.self_attention_res_(norm|proj)\.weight$", p)}

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
        "C10": validate.c10_coverage(
            prefill_rows, param_names,
            expected_gap=_expert_cap_gap(adaptation_log, param_names)
            | _attn_res_layer0_gap(cfg, param_names)),
        "C11": validate.c11_decode_consistency(decode_rows),
        "C14": validate.c14_seq_len(ctx.seq_len, min_seq),
        "C15": validate.c15_entrypoint_coverage(traced_entrypoints, discovered, cfg),
        "C16": validate.c16_unmapped(prefill_rows),
    }

    if check_repro:
        _, all_rows_2, _, _ = _extract(profile, cfg)
        checks["C13"] = validate.c13_reproducibility(all_rows, all_rows_2)
    else:
        checks["C13"] = ("SKIP", "pass --check-repro to actually run twice and verify")

    structure = summarize.build_structure(prefill_rows, cfg, model_id, prov["revision_resolved"],
                                          seq_len=ctx.seq_len, batch=ctx.batch,
                                          resolver=resolver)
    # fixed, config-derived dims the symbolizer left as literals (e.g. MLA kv_b_proj = 32768) --
    # recorded so a reader can tell a "mystery number" from a bug (resolver symbolizes concrete rows).
    literals = summarize.find_literal_dims(prefill_rows, structure["symbols"], resolver,
                                           cfg=cfg, seq_len=resolver.table.get("T"))
    # A value already written up in references.yaml's irreducible_literals (e.g. a shim artifact
    # that must NOT get a name, since naming it would misrepresent an approximation as an
    # architecture dimension -- Kimi-K3's MoE-cap-shim value 1280, review/06-open-renames.md A60)
    # gets its `expr` filled in here, once, so every downstream reader (C17, model_summary.md,
    # structure.yaml itself) sees it as explained rather than re-deciding "is this documented" on
    # its own -- see develop/verify/literals.py's module docstring for why three places used to
    # disagree about this.
    literals = _annotate_documented_literals(literals, os.path.basename(model_dir))
    structure["literal_dims"] = literals
    # Which config fields this architecture uses that rules/symbols.yaml does not know about.
    # A separate throwaway build so a labelling experiment can never perturb the trace above.
    structure["unregistered_fields"] = probe.get("unregistered", [])
    structure["label_provenance"] = summarize.label_provenance(resolver, model_dir)
    # Phase 0 onboarding gate -- runs even when everything else passed, which is the whole point
    # (see 02-new-module-handling.md: DeepSeek-V4 passed C1-C16 with 5 undocumented modules).
    checks["C17"] = validate.c17_module_onboarding(
        literals, structure["symbols"],
        structures_dir=os.path.join(os.path.dirname(__file__), "..", "rules", "structures"),
        model_type=getattr(cfg, "model_type", None), model_id=model_id)
    summarize.write_structure(model_dir, structure, fmt=profile.get("structure_format", "yaml"))
    # Everything a label review needs, assembled here so a freshly traced model arrives ready to
    # review rather than waiting for the next regeneration: download this architecture's real
    # modeling/configuration source, cross-check what is mechanically decidable, and write the
    # hand-off naming only what is left (see review/).
    _mt = getattr(cfg, "model_type", None)
    _fields = summarize.resolved_fields(cfg)
    _sc = source_check.run(model_dir, model_id, _mt, _fields, source_check.square_labels(model_dir),
                           alias_map=summarize.alias_fields())
    review_request.build(model_dir, model_id, _mt, structure, _sc, _fields)


    sources = []
    sources_path = profile.get("sources_file")  # optional: agent-supplied Tier 2 findings
    if sources_path and os.path.exists(sources_path):
        with open(sources_path, encoding="utf-8") as f:
            sources = yaml.safe_load(f) or []
    summary_md = summarize.render_model_summary(
        model_id, prov, structure, cfg=cfg, rows=prefill_rows, scale=scale,
        checks=checks, sources=sources, literals=literals, model_dir=model_dir)
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


def _annotate_documented_literals(literals: list, model_dir_name: str) -> list:
    """develop/ is tooling, src/ is the pipeline (see _write_review_packet) -- same local,
    non-fatal import. A lookup failure must never turn a documented value into a false C17 WARN
    NOR silently drop a real one, so it degrades to "return literals unchanged" rather than
    raising."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "develop"))
        from verify import literals as _lit
        return _lit.annotate(literals, model_dir_name)
    except Exception:                             # noqa: BLE001 -- reporting, not control flow
        return literals


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--out", default="out")
    ap.add_argument("--check-repro", action="store_true",
                     help="run extraction twice and verify C13 reproducibility (doubles runtime)")
    ap.add_argument("--batch", type=int, default=None,
                     help="발행 배치 대신 이 배치로 잡는다 (진단용 -- 규칙 변화와 배치 변화를 "
                          "갈라 보려고 같은 규칙으로 B 만 바꿔 뜰 때 쓴다)")
    args = ap.parse_args()
    run(args.profile, args.out, check_repro=args.check_repro, batch=args.batch)
