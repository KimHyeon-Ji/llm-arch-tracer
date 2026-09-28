# -*- coding: utf-8 -*-
"""R3d 음성 대조 -- 고친 fail-open 경로가 **실제로 발화**하는지 본다.

0 == 0 은 통과가 아니다. "fail-closed 로 고쳤다" 는 말만으로는 지난 라운드와 같다.
실제 모델 데이터 위에서 crosswalk 을 일부러 망가뜨려 각 경로가 발화하는지 본다.
실행: .venv/Scripts/python.exe develop/plus_at_negctl.py
  1  reshape: crosswalk 이 없는 raw op 을 가리키면 not_evaluated 인가
  2  axis:    crosswalk 이 없는 raw slot / 범위 밖 축을 가리키면 not_evaluated 인가
  3  crosswalk 이 발행본 셀 집합을 정확히 안 덮으면 not_evaluated 인가
  4  port:    raw op-id 집합과 ports op-id 집합이 다르면 FAIL 인가
  5  reshape: **축이 범위 밖**이면 not_evaluated 인가 (R3d 지적. 예전엔 조용히 넘어갔다)
  6  cycle:   `x: {expr: x}` 같은 직접 자기 참조를 잡는가
  7  sidecar: 미등록 식별자 / registry 불일치 / 값 불일치 / 레코드 없음을 잡는가
  8  sidecar: formula 를 다른 셀로 **재배치**해도 phase 검사가 잡는가
  9  zero:    허용 자리를 선언했는데 실제 `0` 이 없으면 FAIL 인가
 10  base:    본표에 미등록 식별자가 있으면 FAIL 인가
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

# =========================================================== R3e 에서 더한 대조
# 아래는 R3d 판정이 probe 로 깨 보인 자리다. 같은 방식으로 여기서 먼저 깬다.

# --- 5  reshape / 범위 밖 축 (R3d 가 직접 재현한 것)
V9._crosswalk = cw_bad_axis
res.append(show("reshape / 범위 밖 축", "not_evaluated", V9.g_reshape_derivation(ctx)))
V9._crosswalk = real_cw

# --- 6  직접 자기 참조
import copy                          # noqa: E402
_ov2 = copy.deepcopy(ov)
_ov2["symbols"]["x"] = {"kind": "trace_artifact", "expr": "x", "value": 1}
res.append(show("cycle / x: {expr: x}", False,
                V9.g_expression_no_cycle(dict(ctx, overlay=_ov2))))

# --- 7  사이드카 무결성 네 가지
import yaml as _y                    # noqa: E402
_sc = _y.safe_load(io.open(os.path.join(PA, "expressions.yaml"),
                          encoding="utf-8"))["records"]


def _sctx(recs, overlay=None):
    return dict(ctx, sidecar_records=recs, overlay=overlay or ov)


_r = copy.deepcopy(_sc)
_r[0]["formula"] = "mystery"
_r[0]["expr"] = "mystery(l)"
res.append(show("sidecar / 미등록 formula", False,
                V9.g_sidecar_expression_integrity(_sctx(_r))))

_r = copy.deepcopy(_sc)
_r[0]["expr"] = "ceil(l/R_res)+99"          # registry 와 다른 식
res.append(show("sidecar / registry 불일치", False,
                V9.g_sidecar_expression_integrity(_sctx(_r))))

_r = copy.deepcopy(_sc)
_r[0]["value"] = _r[0]["value"] + 1         # 식값과 어긋난다
res.append(show("sidecar / 식값 != value", False,
                V9.g_sidecar_expression_integrity(_sctx(_r))))

res.append(show("sidecar / 선언했는데 레코드 없음", "not_evaluated",
                V9.g_sidecar_expression_integrity(_sctx([]))))

# --- 8  formula 재배치 (수와 분포는 그대로 두고 자리만 바꾼다)
_r = copy.deepcopy(_sc)
_pf = [i for i, x in enumerate(_r) if x["phase"] == "prefill"]
_i = next(i for i in _pf if _r[i]["formula"] != _r[_pf[0]]["formula"])
for _a, _b in (("formula", "formula"), ("expr", "expr"), ("value", "value"),
               ("stage", "stage")):
    _r[_pf[0]][_a], _r[_i][_b] = _r[_i][_b], _r[_pf[0]][_a]
res.append(show("sidecar / formula 재배치", False,
                V9.g_sidecar_phase_consistency(_sctx(_r))))

# --- 9  허용 자리를 선언했는데 실제 `0` 이 없다
_ov3 = copy.deepcopy(ov)
_ov3["not_substituted"][0].setdefault("allowed_zero_cells", []).append(
    ["prefill", 999999, "input_shape", 0, 0])
res.append(show("zero / 선언했는데 없는 자리", False,
                V9.g_zero_axis_only_initial_residual(dict(ctx, overlay=_ov3))))

# --- 10  본표에 미등록 식별자
_cells = {ph: dict(c) for ph, c in ctx["derived_cells"].items()}
_k = next(iter(_cells["prefill"]))
_cells["prefill"][_k] = "d_unregistered"
res.append(show("base / 미등록 식별자", False,
                V9.g_base_symbol_coverage(dict(ctx, derived_cells=_cells))))

print()
print(f"음성 대조 {sum(res)}/{len(res)} 발화")
sys.exit(0 if all(res) else 1)

# =========================================================== 적용기 쪽 대조 (수동)
# 아래 둘은 게이트가 아니라 적용기의 control flow 라 이 스크립트에 넣지 않았다.
# 재현 방법을 적어 둔다 -- 둘 다 실제로 발화하는 것을 확인했다(2026-09-28).
#
#  G  expected_footprint.sha256 없으면 중단하는가
#     overlay 사본에서 sha256 줄을 지우고
#       .venv/Scripts/python.exe develop/plus_at_apply.py moonshotai__Kimi-K3 #           --overlay <사본>
#     -> "**overlay 에 expected_footprint.sha256 이 없다 -- 고정되지 않았다**" 로 중단
#
#  H  고정 revision 에 근거 파일이 없으면 다른 snapshot 으로 넘어가지 않는가
#     full/provenance.json 의 revision_resolved 를 가짜로 바꾸고 --publish
#     -> "근거 파일을 못 찾았다: modeling_kimi_linear.py ... publish 하지 않는다" 로 중단
#     -> 발행본 MANIFEST 의 md5 가 그대로다(교체는 근거 해석 뒤에 일어난다)
#     끝나면 git checkout -- models/<m>/full/provenance.json
