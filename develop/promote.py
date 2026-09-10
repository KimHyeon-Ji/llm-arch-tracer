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


# **트레이스가 만들지 않는 산출물.** 승격은 모델 디렉터리를 통째로 갈아끼우므로, 여기 적힌
# 것을 먼저 들어내지 않으면 재트레이스마다 사라진다. 실제로 47/47 모델에서 ③층의
# `review_findings.json` 이 없어져 있었고, 복구 커밋 다음 재트레이스에서 또 날아갔다
# (2026-09-10). 교정 적용에 재트레이스가 필수이므로 반복할 때마다 증거를 잃는 구조였다.
#
# `.md` 는 승계하지 않는다 -- JSON 에서 다시 그린다. 두 파일이 어긋날 수 없게 하기 위해서다
# (`src/review_notes.render_md` 의 계약).
CARRY_OVER = ("review_findings.json", "research_agenda.md")
REGENERATED_FROM_JSON = ("review_findings.md",)


def _carry(dest: str, name: str) -> list:
    """옛 산출물에서 살려야 할 파일을 읽어 둔다. 디렉터리를 지우기 **전에** 부른다."""
    kept = []
    for f in CARRY_OVER:
        p = os.path.join(dest, f)
        if os.path.exists(p):
            with open(p, "rb") as fh:
                kept.append((f, fh.read()))
    return kept


def _restore(dest: str, kept: list, name: str):
    """새 산출물 위에 되돌려 놓고, 파생 문서는 다시 그린다."""
    for f, blob in kept:
        with open(os.path.join(dest, f), "wb") as fh:
            fh.write(blob)
    if any(f == "review_findings.json" for f, _ in kept):
        sys.path.insert(0, os.path.join(HERE, "..", "src"))
        try:
            import review_notes
            review_notes.render_md(dest, name)
        except Exception as e:                      # 승격 자체를 막지는 않는다
            print(f"      review_findings.md 재생성 실패: {e}")
    if kept:
        print(f"      승계 {len(kept)}개: {', '.join(f for f, _ in kept)}")


def promote(d: str):
    name = os.path.basename(d)
    fails = _fails(os.path.join(d, "full", "report.md"))
    if fails:
        print(f"SKIP  {name}  (FAIL: {'; '.join(fails)})")
        return
    dest = os.path.join(MODELS, name)
    kept = []
    if os.path.exists(dest):
        kept = _carry(dest, name)
        shutil.rmtree(dest)          # replace an older promoted copy
    shutil.move(d, dest)
    _restore(dest, kept, name)
    print(f"PROMOTED  {name}  ->  models/")


if __name__ == "__main__":
    filt = sys.argv[1] if len(sys.argv) > 1 else ""
    dirs = [d for d in sorted(glob.glob(os.path.join(OUT, "*")))
            if os.path.isdir(d) and (not filt or filt.lower() in os.path.basename(d).lower())]
    if not dirs:
        print("nothing to promote in develop/out/" + (f" matching '{filt}'" if filt else ""))
    for d in dirs:
        promote(d)
