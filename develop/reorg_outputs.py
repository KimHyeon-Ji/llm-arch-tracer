"""Migrate every model dir in develop/out/ to the new two-view layout WITHOUT re-tracing.

Old layout (flat):
    <dir>/prefill.csv, decode.csv, prefill.trace.raw.jsonl, decode.trace.raw.jsonl,
          provenance.json, report.md, structure.yaml, model_summary.md
New layout:
    <dir>/prefill.csv, decode.csv                       (MAJOR operators only)
    <dir>/prefill.jsonl, decode.jsonl                   (MAJOR operators only)
    <dir>/structure.yaml, model_summary.md              (unchanged, stay on top)
    <dir>/full/prefill.csv, decode.csv, prefill.trace.raw.jsonl, decode.trace.raw.jsonl,
              provenance.json, report.md

The full trace already lives in each existing top-level *.trace.raw.jsonl (shapes already
symbolic), so we read it back and re-emit via build_table.write_outputs (identity resolver),
which now writes the full view under full/ and derives the major view on top. Then provenance
and report are moved into full/, and the stale top-level full jsonl is removed.

Run:  .venv\\Scripts\\python.exe develop\\reorg_outputs.py [substring-filter]
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import build_table

OUT = os.path.join(os.path.dirname(__file__), "out")


def _identity(shape):
    return shape  # rows read from an existing jsonl are already symbolic


def reorg_dir(d: str):
    name = os.path.basename(d)
    full_dir = os.path.join(d, build_table.FULL_SUBDIR)
    os.makedirs(full_dir, exist_ok=True)

    phases = []
    for phase in ("prefill", "decode"):
        jl = os.path.join(d, f"{phase}.trace.raw.jsonl")
        if not os.path.exists(jl):
            continue
        rows = [json.loads(line) for line in open(jl, encoding="utf-8")]
        build_table.write_outputs(d, phase, rows, _identity)  # -> full/ + major on top
        phases.append(phase)

    # move run metadata under full/
    for fn in ("provenance.json", "report.md"):
        src = os.path.join(d, fn)
        if os.path.exists(src):
            os.replace(src, os.path.join(full_dir, fn))

    # drop the stale top-level full traces (their content now lives in full/ + the major view)
    for phase in phases:
        stale = os.path.join(d, f"{phase}.trace.raw.jsonl")
        if os.path.exists(stale):
            os.remove(stale)

    print(f"reorged: {name:45s} phases={phases}")


if __name__ == "__main__":
    filt = sys.argv[1] if len(sys.argv) > 1 else ""
    for d in sorted(glob.glob(os.path.join(OUT, "*"))):
        if os.path.isdir(d) and (not filt or filt.lower() in os.path.basename(d).lower()):
            try:
                reorg_dir(d)
            except Exception as e:
                print("ERROR", os.path.basename(d), type(e).__name__, str(e)[:160])
