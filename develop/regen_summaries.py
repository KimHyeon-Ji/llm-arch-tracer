"""Regenerate model_summary.md for validated models in ../models/ WITHOUT re-tracing.

Reads the existing develop/out/<dir> artifacts (structure.yaml, prefill.trace.raw.jsonl,
report.md, provenance.json) + a fresh config snapshot (for hf_created_at) + a meta model
load (for param scale), then re-renders the summary with the current summarize.py (gallery
card etc.). Tracing is the expensive step and is intentionally skipped -- the trace is
deterministic and unchanged; only model_summary.md content is being updated.

Run:   .venv\\Scripts\\python.exe develop\\regen_summaries.py [substring-filter]
"""
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import yaml

import provenance
import loader
import summarize
import validate
import build_table
import symbolic_shape

# published outputs live in top-level models/; profiles are kept in develop/models/.
OUT = os.path.join(os.path.dirname(__file__), "..", "models")
MODELS = os.path.join(os.path.dirname(__file__), "models")

_CHECK_RE = re.compile(r"^(C\d+)\s+(PASS|FAIL|WARN|SKIP|INFO)\s+(.*)$")


def _parse_report_md(path: str) -> dict:
    """checks {cid: (status, detail)} parsed from report.md (report.json is no longer emitted)."""
    checks = {}
    for line in open(path, encoding="utf-8"):
        m = _CHECK_RE.match(line.rstrip("\n"))
        if m:
            checks[m.group(1)] = (m.group(2), m.group(3).strip())
    return checks


def _rewrite_c17_in_report(path: str, result):
    """Keep full/report.md in step with the recomputed C17.

    Without this the two deliverables disagree: model_summary.md would say PASS (recomputed after
    registering a new structure doc) while report.md still carried the WARN from trace time.
    Only the C17 line is touched -- every other check reflects the trace and must not be edited
    here, since regeneration does not re-trace."""
    if not os.path.exists(path):
        return
    status, detail = result
    lines = open(path, encoding="utf-8").read().splitlines()
    new_line = f"{'C17':4s} {status:6s} {detail}"
    for i, ln in enumerate(lines):
        if re.match(r"^C17\s", ln):
            lines[i] = new_line
            break
    else:  # models traced before C17 existed
        insert_at = max((i for i, ln in enumerate(lines) if re.match(r"^C\d+\s", ln)), default=len(lines) - 1)
        lines.insert(insert_at + 1, new_line)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def compute_scale(model, cfg) -> dict:
    named = list(model.named_parameters())
    total = sum(p.numel() for _, p in named)
    expert = sum(p.numel() for n, p in named if ".expert" in n)
    E = (getattr(cfg, "num_experts", None) or getattr(cfg, "n_routed_experts", None)
         or getattr(cfg, "num_local_experts", None))
    kk = getattr(cfg, "num_experts_per_tok", None) or getattr(cfg, "moe_topk", None)
    active = int(total - expert * (1 - kk / E)) if (expert and E and kk) else total
    return {"total_params": total, "active_params": active, "expert_params": expert}


def regen(profile_path: str):
    prof = yaml.safe_load(open(profile_path, encoding="utf-8"))
    mid = prof["model_id"]
    d = os.path.join(OUT, mid.replace("/", "__"))
    if not os.path.isdir(d):
        print("skip (no output dir):", mid)
        return

    # fresh snapshot -> cfg + hf_created_at; keep the run-specific fields from the existing prov
    cfg, prov = provenance.snapshot(mid, prof.get("revision"),
                                    config_overrides=prof.get("config_overrides"))
    full = os.path.join(d, "full")
    old = json.load(open(os.path.join(full, "provenance.json"), encoding="utf-8"))
    for kf in ("capture_backend", "seq_len_used", "attn_implementation_used",
               "symbol_table", "adaptation_log"):
        prov[kf] = old.get(kf)

    # reuse the param scale stored from the original run (avoids re-loading huge models on meta
    # just to recount params); fall back to a fresh meta load only if it isn't there.
    scale = old.get("param_scale")
    if not scale:
        model = loader.load_meta(cfg, trust_remote_code=prov["trust_remote_code"])
        scale = compute_scale(model, cfg)

    structure = yaml.safe_load(open(os.path.join(d, "structure.yaml"), encoding="utf-8"))
    # Re-resolve symbols from the current rules/symbols.yaml. The stored table was built with
    # whatever symbol set existed at trace time, so a newly registered symbol (a new architecture
    # group) would otherwise stay invisible here and its derived dims would look unexplained.
    # Layer/op data is untouched -- only the symbol lookup is refreshed.
    structure["symbols"] = summarize.resolve_symbols(cfg)
    rows = [json.loads(l) for l in open(os.path.join(full, "prefill.trace.raw.jsonl"), encoding="utf-8")]
    checks = _parse_report_md(os.path.join(full, "report.md"))

    # If the concrete-shape sidecar exists, re-render the whole trace with the CURRENT symbolizer.
    # Without it we can only re-read shapes that were symbolized by whatever rules were in force at
    # trace time, so a symbolizer fix would need a full re-trace (see build_table docstring).
    seq_len = prov.get("seq_len_used")
    resolver = None
    if seq_len and build_table.load_concrete(d, "prefill"):
        resolver = symbolic_shape.build_resolver(cfg, seq_len)
        for phase in ("prefill", "decode"):
            conc = build_table.load_concrete(d, phase)
            raw_path = os.path.join(full, f"{phase}.trace.raw.jsonl")
            if not conc or not os.path.exists(raw_path):
                continue
            phase_rows = [json.loads(l) for l in open(raw_path, encoding="utf-8")]
            for r in phase_rows:  # swap symbolic shapes back to the recorded concrete ones
                c = conc.get(r.get("op_id"))
                if c:
                    r["input_shape"] = c["input_shape"]
                    r["weight_shape"] = c["weight_shape"]
                    r["output_shape"] = c["output_shape"]
            build_table.write_outputs(d, phase, phase_rows, resolver)
            if phase == "prefill":
                rows = phase_rows  # concrete now; find_literal_dims gets the resolver below
        prov["symbol_table"] = resolver.table

    # literal (non-symbol) dims. With the sidecar `rows` are concrete and need the resolver;
    # without it they are already symbolic and must NOT be resolved again.
    literals = summarize.find_literal_dims(rows, structure["symbols"], resolver,
                                           cfg=cfg, seq_len=seq_len)
    structure["literal_dims"] = literals
    summarize.write_structure(d, structure)

    # C17 is recomputed here rather than read from the stored report: it grades the *research*
    # (derived_dims + structure library), which changes without re-tracing.
    checks["C17"] = validate.c17_module_onboarding(
        literals, structure["symbols"],
        structures_dir=os.path.join(os.path.dirname(__file__), "..", "rules", "structures"),
        model_type=getattr(cfg, "model_type", None), model_id=mid)
    _rewrite_c17_in_report(os.path.join(full, "report.md"), checks["C17"])

    sources = []
    sf = prof.get("sources_file")
    if sf and os.path.exists(sf):
        sources = yaml.safe_load(open(sf, encoding="utf-8")) or []

    md = summarize.render_model_summary(mid, prov, structure, cfg=cfg, rows=rows,
                                        scale=scale, checks=checks, sources=sources, literals=literals)
    summarize.write_model_summary(d, md)

    # persist hf_created_at + scale back into provenance.json (so it's not lost)
    old["hf_created_at"] = prov.get("hf_created_at")
    old["param_scale"] = scale
    json.dump(old, open(os.path.join(full, "provenance.json"), "w", encoding="utf-8"),
              indent=2, default=str)
    print(f"regenerated: {mid:45s} | {summarize._hnum(scale['total_params'])} total, "
          f"{summarize._hnum(scale['active_params'])} active | date {prov.get('hf_created_at')}")


if __name__ == "__main__":
    filt = sys.argv[1] if len(sys.argv) > 1 else ""
    for p in sorted(glob.glob(os.path.join(MODELS, "*.yaml"))):
        if filt and filt.lower() not in os.path.basename(p).lower():
            continue
        try:
            regen(p)
        except Exception as e:
            print("ERROR", os.path.basename(p), type(e).__name__, str(e)[:160])
