r"""bundle 가드가 **실제로 발화하는가.** 전부 0 으로 보고되므로 증명이 필요하다.

`0 == 0` 은 통과가 아니다. 각 검사에 대해 걸려야 하는 입력을 하나씩 만들어 넣는다.

실행:
    .venv\Scripts\python.exe develop\test_bundle_guards.py
"""
import io
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(PROJ, "src"))
if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import _buildguard                                               # noqa: E402
import build_review_bundle as B                                  # noqa: E402

OK, FAIL = [], []


def check(name, cond):
    (OK if cond else FAIL).append(name)
    print(("  OK   " if cond else "  FAIL ") + name)


def unit(**kw):
    u = {"decision_unit_id": "u1", "model": "m", "_phase": "prefill",
         "published_cells": [[1, "i", 0, 0]], "concrete_value": 4,
         "concrete_shapes": {"i": [[1, 4]]},
         "affects_published_cells": 7, "represents_raw_sites": 9,
         "question_family_id": "f1", "grade": "heuristic",
         "signature": {"block_type": "b", "layer_cohort_id": "c", "module": "mm",
                       "op_type": "linear", "raw_op": "aten.mm",
                       "old_expr": "d_head", "candidates": ["d_head", "d_qk"]}}
    u.update(kw)
    return u


print("1) _has_private 가 금지 키를 깊이 상관없이 잡는가")
check("원본 단위는 금지 키를 갖는다", B._has_private(unit()))
check("candidates 가 중첩돼 있어도 잡는다",
      B._has_private({"a": [{"b": {"candidates": []}}]}))
check("grade 를 잡는다", B._has_private({"x": {"grade": "heuristic"}}))
check("깨끗한 dict 은 통과한다",
      not B._has_private({"decision_unit_id": "u", "signature": {"module": "m"}}))

print("2) _public 이 민감 필드를 실제로 제거하는가")
v = B._public(unit())
check("public DTO 에 금지 키가 없다", not B._has_private(v))
check("old_expr 가 사라졌다", "old_expr" not in v["signature"])
check("candidates 가 사라졌다", "candidates" not in v["signature"])
check("grade 가 사라졌다", "grade" not in v)
check("question_family_id 가 사라졌다", "question_family_id" not in v)
check("stage1 에서는 발행 영향 수도 사라진다",
      "affects_published_cells" not in B._public(unit(), stage1=True))
check("모집단에서는 발행 영향 수가 남는다",
      "affects_published_cells" in B._public(unit(), stage1=False))

print("3) lineage 는 provenance 만 받고, 포트가 없으면 실패하는가")
for mode in ("legacy", "migration", "hybrid", "none", None):
    try:
        B._lineage("openai__gpt-oss-20b", "prefill", mode)
        check(f"mode={mode!r} 을 거부한다", False)
    except ValueError:
        check(f"mode={mode!r} 을 거부한다", True)
    except SystemExit:
        check(f"mode={mode!r} 을 ValueError 로 거부한다", False)
try:
    B._lineage("openai__gpt-oss-20b", "prefill", "provenance")
    check("포트 coverage 0 에서 실패한다", False)
except SystemExit as e:
    check("포트 coverage 0 에서 SystemExit(4)", e.code == 4)

print("4) port_coverage 가 빈 포트를 0 으로 보고하는가")
cv = B.port_coverage("openai__gpt-oss-20b", "prefill")
check("ports 0 행", cv["ports_lines"] == 0)
check("raw 행은 세어진다", cv["raw_lines"] > 0)
check("coverage 는 0 또는 None", not cv["coverage"])

print("5) swap_dir 이 실패 시 기존 것을 되돌리는가")
tmp = tempfile.mkdtemp()
try:
    dest = os.path.join(tmp, "d")
    os.makedirs(dest)
    io.open(os.path.join(dest, "old.txt"), "w").write("old")
    src = os.path.join(tmp, "d.tmp")
    os.makedirs(src)
    io.open(os.path.join(src, "new.txt"), "w").write("new")
    _buildguard.swap_dir(src, dest)
    check("정상 교체 -- 새 파일이 있다", os.path.exists(os.path.join(dest, "new.txt")))
    check("정상 교체 -- 옛 파일은 없다", not os.path.exists(os.path.join(dest, "old.txt")))
    check("정상 교체 -- .bak 이 남지 않는다", not os.path.isdir(dest + ".bak"))
    # 이제 실패하게 만든다: tmp 가 없으면 os.replace 가 터진다
    missing = os.path.join(tmp, "nope")
    try:
        _buildguard.swap_dir(missing, dest)
        check("교체 실패가 예외를 낸다", False)
    except OSError:
        check("교체 실패가 예외를 낸다", True)
    check("**실패해도 기존 디렉터리가 살아 있다**",
          os.path.isdir(dest) and os.path.exists(os.path.join(dest, "new.txt")))
    check("실패 후 .bak 이 남지 않는다", not os.path.isdir(dest + ".bak"))
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("6) input_worktrees 가 출력 경로를 세지 않는가")
iw = _buildguard.input_worktrees([
    ("tracer", PROJ, ("models/", "src/", "develop/", "rules/"))])
check("HEAD 를 읽는다", len(iw["tracer"]["head"]) == 40)
check("입력 prefix 를 기록한다", "src/" in iw["tracer"]["input_prefixes"])
none_scope = _buildguard.input_worktrees([("t", PROJ, ("존재하지않는경로/",))])
check("범위 밖은 세지 않는다", none_scope["t"]["dirty_input_paths"] == [])

print("6-b) 후보 문구 검사가 발화하는가")
HDR = ("# shard 001" + chr(10) + "`rejected_candidates` 는 답 형식이고, "
       "후보 목록도 주지 않습니다." + chr(10))
clean = HDR + chr(10) + "### u1" + chr(10) + "* 모듈: `mm`" + chr(10)
check("정상 문서에서는 걸리지 않는다", B.candidate_text_hits(clean) == [])
check("머리글의 '후보' 는 걸리지 않는다",
      B.candidate_text_hits(HDR + chr(10) + "### u1" + chr(10) + "* x") == [])
dirty = HDR + chr(10) + "### u2" + chr(10) + "* 가능한 후보: d_head, d_qk" + chr(10)
check("**단위 블록의 후보 목록은 걸린다**", B.candidate_text_hits(dirty) == ["u2"])
other = HDR + chr(10) + "### u3" + chr(10) + "#### 이름 후보군" + chr(10)
check("제목 형식이 달라도 걸린다", B.candidate_text_hits(other) == ["u3"])
sym = HDR + chr(10) + "### u4" + chr(10) + "* x" + chr(10) + chr(10)     + "## 심볼 정의와 config 값" + chr(10) + "후보" + chr(10)
check("심볼표 뒤는 보지 않는다", B.candidate_text_hits(sym) == [])
check("단위 블록을 정확히 센다", len(B.unit_blocks(dirty)) == 1)

print("7) 생성된 shard 문서에 실제로 비밀이 없는가")
import json
import re
LAB = B.LAB
for kind, d in (("모집단", B.BUNDLE), ("1 단계", B.PRIORITY_BUNDLE)):
    sd = os.path.join(d, "shards")
    if not os.path.isdir(sd):
        check(f"{kind} bundle 이 있다", False)
        continue
    check(f"{kind} bundle 에 X_linked 가 없다",
          not any("X_linked" in io.open(os.path.join(sd, f), encoding="utf-8").read()
                  for f in os.listdir(sd)))
    man = json.load(io.open(os.path.join(d, "_manifest.json"), encoding="utf-8"))
    check(f"{kind} manifest 의 검사가 전부 0",
          all(v == 0 for k, v in man["checks"].items() if k != "shard_unit_total"))
    check(f"{kind} lineage_mode 가 none", man["lineage_mode"] == "none")
    check(f"{kind} X_linked 미사용 기록",
          man["lineage_placeholder_x_linked_used"] is False)
for k in ("manifest", "stage1_unit_ids", "duplicate_assignment", "duplicate_ratio"):
    man = json.load(io.open(os.path.join(B.PRIORITY_BUNDLE, "_manifest.json"),
                            encoding="utf-8"))
    check(f"1 단계 bundle manifest 에 `{k}` 가 없다", k not in man)
ap = os.path.join(LAB, "priority", "_assignment_manifest.json")
check("배정 manifest 는 bundle 밖에 있다", os.path.exists(ap))
am = json.load(io.open(ap, encoding="utf-8"))
check("배정 manifest 에 중복 표가 있다", len(am["duplicate_assignment"]) > 0)
check("배정 manifest 에 단위 목록이 있다", len(am["stage1_unit_ids"]) == 783)
loc = {}
for f in sorted(os.listdir(os.path.join(B.PRIORITY_BUNDLE, "shards"))):
    t = io.open(os.path.join(B.PRIORITY_BUNDLE, "shards", f),
                encoding="utf-8").read()
    for uid in re.findall(r"### (\S+)", t.split("## 심볼 정의와 config 값")[0]):
        loc.setdefault(uid, []).append(f)
check("중복 단위는 **서로 다른 shard** 에 있다",
      all(len(set(v)) == len(v) for v in loc.values()))
check("중복 단위 수가 배정 manifest 와 맞는다",
      sum(1 for v in loc.values() if len(v) > 1) == len(am["duplicate_assignment"]))

print(f"\n{len(OK)}/{len(OK) + len(FAIL)} 통과")
if FAIL:
    print("실패: " + ", ".join(FAIL))
sys.exit(1 if FAIL else 0)
