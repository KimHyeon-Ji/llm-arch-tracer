"""Rewrite the CSV/JSONL deliverables for every model in develop/out/ with the CURRENT
build_table logic (column order + full/major split) -- WITHOUT re-tracing. The full trace
already lives in each full/<phase>.trace.raw.jsonl (shapes already symbolic), so we read it
back and re-emit via build_table.write_outputs, which rewrites full/<phase>.{csv,trace.raw.jsonl}
and re-derives the top-level major <phase>.{csv,jsonl}. The resolver is identity here because
shapes are already symbolic (build_table's resolver is idempotent on symbolic shapes anyway).

Run:  .venv\\Scripts\\python.exe develop\\regen_tables.py [substring-filter]
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import build_table

# published outputs live in the top-level models/ (develop/out/ is only the working area for
# not-yet-promoted runs); point regen at models/ so format updates rewrite the deliverables.
OUT = os.path.join(os.path.dirname(__file__), "..", "models")


def _identity(shape):
    return shape  # shapes read from an existing jsonl are already symbolic


def regen_dir(d: str):
    name = os.path.basename(d)
    did = False
    for phase in ("prefill", "decode"):
        jl = os.path.join(d, build_table.FULL_SUBDIR, f"{phase}.trace.raw.jsonl")
        if not os.path.exists(jl):
            continue
        rows = [json.loads(line) for line in open(jl, encoding="utf-8")]
        build_table.write_outputs(d, phase, rows, _identity)
        did = True
    print(("rewrote " if did else "skip (no jsonl): ") + name)


if __name__ == "__main__":
    filt = sys.argv[1] if len(sys.argv) > 1 else ""
    for d in sorted(glob.glob(os.path.join(OUT, "*"))):
        if os.path.isdir(d) and (not filt or filt.lower() in os.path.basename(d).lower()):
            try:
                regen_dir(d)
            except Exception as e:
                print("ERROR", os.path.basename(d), type(e).__name__, str(e)[:160])
