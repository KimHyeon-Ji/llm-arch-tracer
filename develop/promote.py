"""Promote a validated extraction from the working area (develop/out/) to the deliverables
folder (top-level models/). The gate: full/report.md must have NO C-check in FAIL state
(WARN/SKIP/INFO are fine). This is the develop -> models/ hand-off (README / models/README.md).

Run:  .venv\\Scripts\\python.exe develop\\promote.py [substring-filter]
      (no filter = consider every dir under develop/out/)
"""
import glob
import os
import re
import shutil
import sys

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "out")                    # working area
MODELS = os.path.join(HERE, "..", "models")        # deliverables
_CHECK = re.compile(r"^(C\d+)\s+(PASS|FAIL|WARN|SKIP|INFO)\b\s*(.*)$")


def _fails(report_path: str) -> list[str]:
    if not os.path.exists(report_path):
        return ["<no report.md>"]
    fails = []
    for line in open(report_path, encoding="utf-8"):
        m = _CHECK.match(line.rstrip("\n"))
        if m and m.group(2) == "FAIL":
            fails.append(f"{m.group(1)}: {m.group(3)[:60]}")
    return fails


def promote(d: str):
    name = os.path.basename(d)
    fails = _fails(os.path.join(d, "full", "report.md"))
    if fails:
        print(f"SKIP  {name}  (FAIL: {'; '.join(fails)})")
        return
    dest = os.path.join(MODELS, name)
    if os.path.exists(dest):
        shutil.rmtree(dest)          # replace an older promoted copy
    shutil.move(d, dest)
    print(f"PROMOTED  {name}  ->  models/")


if __name__ == "__main__":
    filt = sys.argv[1] if len(sys.argv) > 1 else ""
    dirs = [d for d in sorted(glob.glob(os.path.join(OUT, "*")))
            if os.path.isdir(d) and (not filt or filt.lower() in os.path.basename(d).lower())]
    if not dirs:
        print("nothing to promote in develop/out/" + (f" matching '{filt}'" if filt else ""))
    for d in dirs:
        promote(d)
