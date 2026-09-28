# -*- coding: utf-8 -*-
"""R3d 음성 대조 -- 고친 fail-open 경로가 **실제로 발화**하는지 본다.

0 == 0 은 통과가 아니다. "fail-closed 로 고쳤다" 는 말만으로는 지난 라운드와 같다.
실제 모델 데이터 위에서 crosswalk 을 일부러 망가뜨려 각 경로가 발화하는지 본다.
실행: .venv/Scripts/python.exe develop/plus_at_negctl.py
  1  reshape: crosswalk 이 없는 raw op 을 가리키면 not_evaluated 인가
  2  axis:    crosswalk 이 없는 raw slot / 범위 밖 축을 가리키면 not_evaluated 인가
  3  crosswalk 이 발행본 셀 집합을 정확히 안 덮으면 not_evaluated 인가
  4  port:    raw op-id 집합과 ports op-id 집합이 다르면 FAIL 인가
"""
import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ, "develop"))
sys.path.insert(0, os.path.join(PROJ, "src"))
import plus_at_canon as C          # noqa: E402
import plus_at_v9 as V9            # noqa: E402

MD = os.path.join(PROJ, "models", "moonshotai__Kimi-K3")
PA = os.path.join(MD, "plus_at")
import yaml                        # noqa: E402
ov = yaml.safe_load(io.open(os.path.join(
    PROJ, "develop", "plus_at", "overlay-moonshotai__Kimi-K3.yaml"),
    encoding="utf-8"))
actual = [json.loads(l) for l in io.open(
    os.path.join(PA, "actual_footprint.jsonl"), encoding="utf-8") if l.strip()]

ctx = {"proj": PROJ, "model_dir": MD, "overlay": ov, "actual": actual,
       "orig_rows": {}, "derived_rows": {}, "orig_cells": {},
       "derived_cells": {}, "env": {}, "sidecar_records": None}
for ph in ("prefill", "decode"):
    ctx["orig_rows"][ph] = C.read_jsonl_rows(os.path.join(MD, f"{ph}.jsonl"))
    ctx["derived_rows"][ph] = C.read_jsonl_rows(os.path.join(PA, f"{ph}.jsonl"))
    ctx["orig_cells"][ph] = C.csv_cells(os.path.join(MD, f"{ph}.csv"), ph)
    ctx["derived_cells"][ph] = C.csv_cells(os.path.join(PA, f"{ph}.csv"), ph)

real_cw = V9._crosswalk

# **바뀐 셀의 키만 게이트가 읽는다.** crosswalk 전체에 주입하면 안 읽히는 자리에 넣게 된다
# (첫 probe 가 그래서 조용히 통과했다).
CHANGED = {(r["phase"], r["op_id"], r["field"], r["shape_index"], r["axis"])
           for r in actual}


def _victim(d, phase):
    for k in sorted(x for x in d if x[0] == phase and x in CHANGED):
        return k
    raise SystemExit(f"{phase}: 주입할 바뀐 셀이 crosswalk 에 없다")


def show(tag, want, got):
    ok = "OK  " if got[1] == want else "**발화 안 함**"
    print(f"  {ok} {tag}: 기대 {want!r} 실제 {got[1]!r} -- {str(got[2])[:110]}")
    return got[1] == want


res = []

# --- 1/2  crosswalk 이 없는 raw op 을 가리킨다
def cw_bad_op(proj, model, phase):
    p, d = real_cw(proj, model, phase)
    if d is None:
        return p, d
    d = dict(d)
    if any(x[0] == phase and x in CHANGED for x in d):
        k = _victim(d, phase)
        e, sites = d[k]
        d[k] = (e, [(10 ** 9, "i", 0, 0)] + list(sites))
    return p, d


V9._crosswalk = cw_bad_op
res.append(show("reshape / 없는 raw op", "not_evaluated", V9.g_reshape_derivation(ctx)))
res.append(show("axis / 없는 raw op", "not_evaluated",
                V9.g_axis_class_consistency(ctx)))

# --- 2b  존재하는 op 이지만 축이 범위 밖
def cw_bad_axis(proj, model, phase):
    p, d = real_cw(proj, model, phase)
    if d is None:
        return p, d
    d = dict(d)
    if any(x[0] == phase and x in CHANGED for x in d):
        k = _victim(d, phase)
        e, sites = d[k]
        d[k] = (e, [(int(sites[0][0]), "i", 0, 99)] + list(sites))
    return p, d


V9._crosswalk = cw_bad_axis
res.append(show("axis / 범위 밖 축", "not_evaluated", V9.g_axis_class_consistency(ctx)))

# --- 3  crosswalk 이 발행본 전체를 안 덮는다
# 앞서 시도한 "발행본 member 없는 class" 대조는 **원리적으로 무효**였다. rev 를 같은
# crosswalk 에서 만들므로 자리를 주입하는 순간 그 자리는 발행본 키로 되돌아온다. 즉
# `n_ck == len(roots)` 는 지금 구조에서 깨질 수 없는 항등식이다 (아래 주석 참고).
# 실제로 falsifiable 한 것은 **집합 커버리지**다 -- 그래서 그것을 대조한다.
def cw_extra(proj, model, phase):
    p, d = real_cw(proj, model, phase)
    if d is None:
        return p, d
    d = dict(d)
    d[(phase, 10 ** 9, "input_shape", 0, 0)] = ("X", [(0, "i", 0, 0)])
    return p, d


V9._crosswalk = cw_extra
res.append(show("reshape / crosswalk 커버리지", "not_evaluated",
                V9.g_reshape_derivation(ctx)))
res.append(show("axis / crosswalk 커버리지", "not_evaluated",
                V9.g_axis_class_consistency(ctx)))

V9._crosswalk = real_cw

# --- 4  ports 집합 불일치
import shutil                       # noqa: E402
import tempfile                     # noqa: E402
tmp = tempfile.mkdtemp()
shutil.copytree(os.path.join(MD, "full"), os.path.join(tmp, "full"),
                dirs_exist_ok=True)
for f in ("prefill.csv", "decode.csv", "prefill.jsonl", "decode.jsonl"):
    shutil.copy2(os.path.join(MD, f), os.path.join(tmp, f))
pp = os.path.join(tmp, "full", "prefill.ports.jsonl")
lines = io.open(pp, encoding="utf-8").read().splitlines()
io.open(pp, "w", encoding="utf-8", newline=chr(10)).write(
    chr(10).join(lines[:-1]) + chr(10))             # 포트 한 줄을 뺀다
ctx2 = dict(ctx, model_dir=tmp)
res.append(show("port / raw!=ports 집합", False, V9.g_port_coverage_inherited(ctx2)))
shutil.rmtree(tmp, ignore_errors=True)

print()
print(f"음성 대조 {sum(res)}/{len(res)} 발화")
sys.exit(0 if all(res) else 1)
