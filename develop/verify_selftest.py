"""게이트 자체를 검증한다 — 폴트 인젝션.

`verify_all.py`는 산출물이 맞는지 판정한다. 그럼 **게이트가 맞는지는 누가 판정하나?**
"FAIL 0"은 두 가지 이유로 나올 수 있다:
  (a) 산출물에 결함이 없다
  (b) 검사가 결함을 못 잡는다
이 둘을 구분하지 않으면 "26/26 통과"는 아무 의미가 없다. 실제로 이 프로젝트에서
그 일이 일어났다 — 2026-07-30에 attention 레이어 수를 이름 기반으로 바꿨다가
falcon(32→0)과 Nemotron(4→0)이 조용히 attention-free로 뒤집혔는데 게이트는 통과시켰다.

그래서 각 검사마다 **그 검사가 잡기로 되어 있는 결함을 일부러 주입**하고, 게이트가
실제로 그걸 잡는지 확인한다. 잡으면 그 검사는 살아 있는 것이고, 못 잡으면 그 검사는
장식이다. 정상 산출물에서 0이 나오는지(오탐 없음)도 같이 본다.

실행:  .venv\\Scripts\\python.exe develop\\verify_selftest.py [--verbose]
종료코드 0 = 모든 검사가 자기 결함을 잡아냄.
"""
import argparse
import copy
import io
import json
import os
import re
import shutil
import sys
import tempfile

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(PROJ, "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import verify_all as V  # noqa: E402

MODELS = V.MODELS


# ------------------------------------------------------------------ 주입 도구
def _edit_jsonl(path, fn, limit=None):
    """jsonl의 각 행에 fn(row)를 적용. fn이 True를 반환한 행만 센다."""
    rows, hit = [], 0
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if (limit is None or hit < limit) and fn(r):
            hit += 1
        rows.append(r)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return hit


def _first_shape_axis(r, field, pred):
    for sh in (r.get(field) or []):
        if isinstance(sh, list):
            for i, v in enumerate(sh):
                if pred(v):
                    return sh, i
    return None, None


# ------------------------------------------------------------------ 결함 정의
# 각 항목: (지표 이름, 사람이 읽는 설명, 어떤 모델에 주입할지, 주입 함수)
# 주입 함수는 모델 폴더 경로를 받아 결함을 심고, 심은 개수를 반환한다.

def inj_weight_T(d):
    """가중치 축에 T — 정적 파라미터가 시퀀스 길이에 의존한다는 물리적 불가능.

    weight_shape는 input/output_shape와 달리 **평면 리스트**(`["V","d_model"]`)다."""
    p = os.path.join(d, "full", "prefill.trace.raw.jsonl")

    def f(r):
        ws = r.get("weight_shape")
        if isinstance(ws, list) and ws:
            ws[0] = "T"
            return True
        return False
    return _edit_jsonl(p, f, limit=1)


def inj_head_excl(d):
    """한 shape에 n_h와 n_kv 동시 등장 — head 개수 축은 둘 중 하나뿐이다."""
    p = os.path.join(d, "full", "prefill.trace.raw.jsonl")

    def f(r):
        for sh in (r.get("output_shape") or []):
            if isinstance(sh, list) and len(sh) >= 3:
                sh[-2], sh[-1] = "n_h", "n_kv"
                return True
        return False
    return _edit_jsonl(p, f, limit=1)


def inj_batch_excl(d):
    """한 shape에 B가 두 번 — 텐서의 배치 축은 하나뿐이다."""
    p = os.path.join(d, "full", "prefill.trace.raw.jsonl")

    def f(r):
        for sh in (r.get("output_shape") or []):
            if isinstance(sh, list) and len(sh) >= 3:
                sh[0], sh[-1] = "B", "B"
                return True
        return False
    return _edit_jsonl(p, f, limit=1)


def inj_batch_weight(d):
    """가중치 축에 B — 정적 파라미터는 배치 차원을 가질 수 없다."""
    p = os.path.join(d, "full", "prefill.trace.raw.jsonl")

    def f(r):
        ws = r.get("weight_shape")
        if isinstance(ws, list) and ws and not isinstance(ws[0], list):
            ws[0] = "B"
            return True
        return False
    return _edit_jsonl(p, f, limit=1)


def inj_resid_norm(d):
    """레이어 직속 LayerNorm의 활성 폭이 d_model이 아님 — 잔차 스트림 폭 위반."""
    p = os.path.join(d, "full", "prefill.trace.raw.jsonl")

    def f(r):
        if V._RESID_NORM.match(r.get("module_path") or ""):
            for sh in (r.get("output_shape") or []):
                if isinstance(sh, list) and sh:
                    sh[-1] = "n_h*d_head"
                    return True
        return False
    return _edit_jsonl(p, f, limit=1)


def inj_ident_elementwise(d):
    """elementwise op 의 입력과 출력이 같은 텐서인데 라벨이 다름 — 한 연산 안의 두 이름.

    복사 계열만 보던 기존 ident 검사가 놓치던 자리다(입력이 둘이라 제외됐다). 실제로
    DeepSeek-V4-Pro 의 mHC 혼합이 `elementwise_add([B,1,4,d_model], [B,1,4,d_model]) ->
    [B,1,n_hc,d_model]` 로 나오고 있었고 아무 검사도 보지 못했다(2026-08-10).
    """
    p = os.path.join(d, "full", "prefill.trace.raw.jsonl")

    def f(r):
        if r.get("op_type") not in ("elementwise_add", "elementwise_mul"):
            return False
        outs, ins = r.get("output_shape") or [], r.get("input_shape") or []
        if len(outs) != 1 or not isinstance(outs[0], list) or not outs[0]:
            return False
        for op in ins:
            if isinstance(op, list) and [str(x) for x in op] == [str(x) for x in outs[0]]:
                op[-1] = "n_h*d_head"      # 같은 텐서인데 마지막 축만 다른 이름으로
                return True
        return False
    return _edit_jsonl(p, f, limit=1)


def inj_label_false(d):
    """산술적으로 거짓인 라벨 — 심볼표 대입값이 실측 구체값과 다름."""
    p = os.path.join(d, "full", "prefill.trace.raw.jsonl")

    def f(r):
        sh, i = _first_shape_axis(r, "output_shape", lambda v: str(v) not in ("B", "1"))
        if sh is not None:
            sh[i] = "V"          # vocab size를 아무 축에나 붙이면 거의 항상 거짓
            return True
        return False
    return _edit_jsonl(p, f, limit=1)


def inj_param_incons(d):
    """같은 파라미터가 op마다 다르게 라벨링됨.

    게이트는 `params`가 정확히 1개인 행만 파라미터별로 모으므로(verify_all._label_checks),
    같은 파라미터를 두 번째로 만났을 때 weight_shape를 뒤집어 두 가지 라벨을 만든다."""
    p = os.path.join(d, "full", "prefill.trace.raw.jsonl")
    seen = set()

    def f(r):
        ps, ws = (r.get("params") or []), r.get("weight_shape")
        if len(ps) != 1 or not isinstance(ws, list) or len(ws) < 2 or ws[0] == ws[1]:
            return False
        if ps[0] not in seen:
            seen.add(ps[0])
            return False
        ws[0], ws[1] = ws[1], ws[0]
        return True
    return _edit_jsonl(p, f, limit=1)


def inj_flow_wrong(d):
    """데이터플로우 불일치 — 같은 텐서인데 한쪽만 정수로 방치."""
    p = os.path.join(d, "full", "prefill.trace.raw.jsonl")
    conc = {json.loads(l)["op_id"]: json.loads(l)
            for l in open(os.path.join(d, "full", "prefill.shapes.concrete.jsonl"),
                          encoding="utf-8")}

    def f(r):
        if not (r.get("depends_on") and r.get("input_shape")):
            return False
        c = conc.get(r["op_id"]) or {}
        for sh, csh in zip(r["input_shape"], c.get("input_shape") or []):
            if isinstance(sh, list) and isinstance(csh, list):
                for i, (v, cv) in enumerate(zip(sh, csh)):
                    if not str(v).isdigit() and str(v) != "B" and isinstance(cv, int) and cv > 1:
                        sh[i] = str(cv)      # 이름을 지우고 정수로 되돌린다
                        return True
        return False
    return _edit_jsonl(p, f, limit=1)


def inj_self_contra(d):
    """심볼표가 자기 트레이스와 모순 — 트레이스에 expert가 있는데 E=0."""
    p = os.path.join(d, "structure.yaml")
    y = yaml.safe_load(open(p, encoding="utf-8")) or {}
    if "E" not in (y.get("symbols") or {}):
        return 0
    y["symbols"]["E"] = 0
    yaml.safe_dump(y, open(p, "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
    return 1


def inj_c_fail(d):
    """C 체크 FAIL — report.md에 실패가 있는데 게이트가 못 보면 안 된다."""
    p = os.path.join(d, "full", "report.md")
    t = open(p, encoding="utf-8").read()
    open(p, "w", encoding="utf-8").write(t.replace("C5   PASS", "C5   FAIL", 1))
    return 1


def inj_c17(d):
    """C17 미통과 — 신규 모듈 온보딩이 안 끝난 상태."""
    p = os.path.join(d, "full", "report.md")
    t = open(p, encoding="utf-8").read()
    open(p, "w", encoding="utf-8").write(t.replace("C17  PASS", "C17  WARN", 1))
    return 1


def inj_unresolved(d):
    """미해결 유도 상수가 남은 채로 통과하면 안 된다."""
    p = os.path.join(d, "model_summary.md")
    t = open(p, encoding="utf-8").read()
    open(p, "w", encoding="utf-8").write(t + "\n미해결 — 아래 Tier 3\n")
    return 1


def inj_membership(d):
    """가중치 축이 그 모듈이 읽지도 않는 config 필드의 이름을 달게 만든다.

    `membership.json` 은 재생성 때 계산돼 저장되고 게이트는 그걸 읽는다. 주입은 그 산출물에
    위반 한 건을 넣는 것으로 충분하다 — 게이트가 실제로 이 파일을 보고 실패하는지가 이 시험의
    질문이다. 이 검사가 없던 동안 DeepSeek-V4 의 indexer 는 자기 것이 아닌 `n_h`/`d_head` 를
    2,000축 넘게 달고 있었고 값이 전부 일치해서 어떤 지표도 움직이지 않았다(2026-08-11).
    """
    p = os.path.join(d, "full", "membership.json")
    if not os.path.exists(p):
        return 0
    j = json.load(open(p, encoding="utf-8"))
    j["gaps"] = (j.get("gaps") or []) + [{
        "module": "model.layers.*.self_attn.q_proj", "label": "E", "symbol": "E",
        "field": "num_experts", "owner": "model.layers.*.self_attn", "axes": 1}]
    json.dump(j, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return 1


def inj_membership_notrun(d):
    """소속 검사가 수행되지 않은 상태를 '깨끗함'으로 읽으면 안 된다."""
    p = os.path.join(d, "full", "membership.json")
    if not os.path.exists(p):
        return 0
    json.dump({"ran": False, "gaps": [], "unknown_classes": []},
              open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return 1


def inj_weight_operand(d):
    """같은 가중치가 한 행 안에서 두 이름 — `weight_shape` 만 바꾸고 피연산자를 방치한다.

    외부 검토(2026-08-12)가 찾아낸 부류다. Llama-3.1-405B/70B 의 q/k/v_proj 가 피연산자 쪽만
    `d_model` 로 고쳐지고 저장 형태는 `n_h*d_head` 로 남아 있었는데, 두 값이 같아서 어떤 값
    검사도 못 봤고 파라미터 일관성 검사는 op **사이**만 비교해서 못 봤다."""
    p = os.path.join(d, "full", "prefill.trace.raw.jsonl")

    def f(r):
        wp, ws = r.get("weight_pos"), r.get("weight_shape")
        if not (isinstance(wp, int) and wp >= 0 and isinstance(ws, list) and len(ws) >= 2):
            return False
        if isinstance(ws[0], list):
            return False
        ins = r.get("input_shape") or []
        if wp >= len(ins) or not isinstance(ins[wp], list):
            return False
        # 검사가 실제로 발동하는 행에만 심는다 -- 길이가 맞고, 저장 형태와 피연산자가 그대로든
        # 전치든 대응되는 행. 그렇지 않은 행에 심으면 결함이 아니라 잡음이다.
        op, st = [str(x) for x in ins[wp]], [str(x) for x in ws]
        if len(op) != len(st) or len(st) < 2:
            return False
        if op != st and op != st[:-2] + st[-2:][::-1]:
            return False
        # 이미 그 이름이면 주입이 무의미하다 -- 첫 후보가 embedding 이라 st[0] 이 이미 'V' 였고,
        # 그래서 첫 시도는 아무것도 심지 않은 채 '못 잡음'으로 보고됐다.
        ws[0] = "d_ff" if st[0] != "d_ff" else "V"
        return True
    return _edit_jsonl(p, f, limit=1)


def inj_unanswered(d):
    """의뢰서가 질문을 냈는데 판정이 하나도 없는 상태 — 검토가 배정된 일을 안 한 경우."""
    import io as _io
    req = os.path.join(d, "review_request.md")
    fnd = os.path.join(d, "review_findings.json")
    if not os.path.exists(req):
        return 0
    t = _io.open(req, encoding="utf-8").read()
    if "## 판단이 필요한 것" not in t:
        return 0
    # 검사가 **항목 대조**로 바뀌었으므로(src/review_ledger.unanswered_items) 개수만 늘려서는
    # 재현되지 않는다. 아무도 답하지 않은 의뢰 항목을 실제로 한 줄 심는다.
    item = ("\n- `zz_nonexistent` in `model.layers.*.nowhere` — "
            "값 7 를 두고 후보가 2개, 1축\n")
    t2 = t.replace("## 판단이 필요한 것", "## 판단이 필요한 것" + item, 1)
    _io.open(req, "w", encoding="utf-8").write(t2)
    json.dump({"model_id": "x", "findings": []}, open(fnd, "w", encoding="utf-8"))
    return 1


def inj_uncited_confirm(d):
    """근거 없는 확인 기록 — 그 주장이 축을 인계 목록에서 영구히 빼므로 인용이 필요하다."""
    p = os.path.join(d, 'full', 'label_confirmed.json')
    json.dump([{"id": "injected", "module": "x$", "label": "y", "expect": 1,
                "source": "봤는데 맞다", "matched": 3}],
              open(p, 'w', encoding='utf-8'), ensure_ascii=False)
    return 1


def inj_dead_confirm(d):
    """더 이상 맞지 않는 확인 기록 — "예전에 맞았다"가 "지금 맞다"로 통과하면 안 된다.

    `rules/label_confirmed.yaml` 은 ④층이 "봤고 이 이름이 맞다"고 적는 자리다. 그 뒤 규칙이
    바뀌어 축 이름이 달라지면 그 확인은 거짓이 되고, 인계 목록에서 그 축을 계속 빼 버린다 --
    즉 아무도 다시 안 보게 된다. 낡은 확인은 낡은 교정만큼 위험하다.
    """
    p = os.path.join(d, 'full', 'label_confirmed.json')
    json.dump([{"id": "injected", "module": "x$", "label": "zz_gone", "expect": 1,
                "source": "주입", "matched": 0}],
              open(p, 'w', encoding='utf-8'), ensure_ascii=False)
    return 1


def inj_bad_stub(d):
    """인계 초안을 지목 불가능하게 만든다 — 못 쓰는 초안을 조용히 배포하면 안 된다.

    이 검사가 왜 있나: 초안의 유일성 검증을 두 번 틀렸다. 처음에는 접은 뒤에 세어
    아무것도 검증하지 못했고(외부 검토가 짚었다), 고친 뒤에도 모듈을 문자열로 비교해
    적용기의 정규식 매칭과 달랐다. 검사가 적용기와 어긋나면 그 검사는 장식이다.
    """
    p = os.path.join(d, 'full', 'prefill.unsettled.json')
    if not os.path.exists(p):
        return 0
    data = json.load(open(p, encoding='utf-8'))
    items = data.get('items') or []
    if not items:
        return 0
    items[0]['stub_ambiguous'] = '주입: 이 초안은 등가류 여럿을 잡는다'
    json.dump(data, open(p, 'w', encoding='utf-8'), ensure_ascii=False)
    return 1


def inj_axis_conflict(d):
    """한 등가류 안에 두 이름을 심는다 — 같은 텐서의 같은 축인데 자리마다 이름이 다른 상태.

    `axis_conflict` 는 2026-08-14 에 0 이 됐다(등가류 통일). 0 불변식이 됐으니 여기로 옮긴다 --
    그 전에는 정상값이 6,033 이라 "결함을 심으면 0 -> N" 패턴에 맞지 않아 CASES 밖에 있었다.
    소비자의 피연산자 이름 하나만 바꾼다: 생산자와 어긋나면 그 등가류가 두 이름을 갖는다.
    """
    p = os.path.join(d, "full", "prefill.trace.raw.jsonl")

    def f(r):
        if not r.get("depends_on"):
            return False
        for sh in (r.get("input_shape") or []):
            if isinstance(sh, list) and sh and not str(sh[-1]).isdigit() and str(sh[-1]) != "B":
                sh[-1] = "V"          # 그 자리에 있을 수 없는 이름
                return True
        return False
    return _edit_jsonl(p, f, limit=1)


def inj_soft_undetermined(d):
    """URL 없이 '확인 못함'으로 남긴 판정 — HF 소스만 보고 포기한 것은 확인 못함이 아니다."""
    p = os.path.join(d, 'review_findings.json')
    if not os.path.exists(p):
        return 0
    data = json.load(open(p, encoding='utf-8'))
    finds = data.get('findings') or []
    if not finds:
        finds = [{}]
        data['findings'] = finds
    finds[0]['verdict'] = 'undetermined'
    finds[0]['evidence'] = '값이 겹쳐서 무엇인지 알 수 없다'
    json.dump(data, open(p, 'w', encoding='utf-8'), ensure_ascii=False)
    return 1


def inj_claim_only(d):
    """방법 서술만 바꾸고 판정은 그대로 둔다 — '다르게 봤다'는 검증 가능한 주장이다."""
    p = os.path.join(d, 'review_findings.json')
    if not os.path.exists(p):
        return 0
    data = json.load(open(p, encoding='utf-8'))
    data['angle'] = '이번에는 완전히 새로운 각도로, 아주 꼼꼼하게 다시 봤다'
    json.dump(data, open(p, 'w', encoding='utf-8'), ensure_ascii=False)
    return 1


def inj_uncited(d):
    """교정 주장에서 소스 인용을 지운다 — '근거 없는 판정'이 통과하면 안 된다."""
    p = os.path.join(d, 'review_findings.json')
    if not os.path.exists(p):
        return 0
    data = json.load(open(p, encoding='utf-8'))
    finds = data.get('findings') or []
    if not finds:
        finds = [{}]
        data['findings'] = finds
    finds[0]['verdict'] = 'should_be_renamed'
    finds[0]['evidence'] = '근거 없이 이름만 바꿨다고 주장한다'
    json.dump(data, open(p, 'w', encoding='utf-8'), ensure_ascii=False)
    return 1


def inj_attn_layers(d):
    """attention 레이어 수 오판 — 이름 기반 규칙이 falcon/Nemotron을 뒤집었던 그 사고."""
    p = os.path.join(d, "full", "prefill.trace.raw.jsonl")

    def f(r):
        mp = r.get("module_path") or ""
        if mp.endswith(("q_proj", "query_key_value", "c_attn", "qkv_proj", "Wqkv")):
            r["module_path"] = mp.rsplit(".", 1)[0] + ".xxx_proj"
            return True
        return False
    return _edit_jsonl(p, f)


def inj_kv_card(d):
    """KV cache 카드 수치가 공개 갤러리와 어긋남."""
    p = os.path.join(d, "model_summary.md")
    t = open(p, encoding="utf-8").read()
    import re as _re
    t2 = _re.sub(r"(KV CACHE / TOKEN \(BF16\) \| )[\d.]+( KiB)", r"\g<1>999.9\2", t, count=1)
    open(p, "w", encoding="utf-8").write(t2)
    return 1 if t2 != t else 0


# 지표 이름 -> (설명, 대상 모델, 주입 함수). 대상 모델은 그 결함이 성립하는 모델로 고른다.
CASES = [
    ("weight_T",     "가중치 축에 T (물리적 불가능)",           "meta-llama__Llama-3.1-8B",  inj_weight_T),
    ("head_excl",    "한 shape에 n_h + n_kv 동시",              "meta-llama__Llama-3.1-8B",  inj_head_excl),
    ("batch_excl",   "한 shape에 B가 2번 (배치 축은 하나)",     "meta-llama__Llama-3.1-8B",  inj_batch_excl),
    ("batch_excl",   "가중치 축에 B (배치 없는 정적 파라미터)", "meta-llama__Llama-3.1-8B",  inj_batch_weight),
    ("resid_norm",   "레이어 직속 LayerNorm 폭 != d_model",     "meta-llama__Llama-3.1-8B",  inj_resid_norm),
    ("label_false",  "산술적으로 거짓인 라벨",                   "google__gemma-2-2b",        inj_label_false),
    ("param_incons", "같은 파라미터의 라벨 불일치",              "google__gemma-2-2b",        inj_param_incons),
    ("flow_wrong",   "데이터플로우 라벨 불일치(이름->정수)",     "Qwen__Qwen2.5-0.5B",        inj_flow_wrong),
    ("ident_incons", "elementwise 입력/출력이 같은 텐서인데 라벨 다름", "Qwen__Qwen2.5-0.5B",  inj_ident_elementwise),
    ("self_contra",  "심볼표가 자기 트레이스와 모순 (E=0)",      "Qwen__Qwen3-30B-A3B",       inj_self_contra),
    ("c_fail",       "C 체크 FAIL",                              "Qwen__Qwen2.5-0.5B",        inj_c_fail),
    ("c17",          "C17 미통과",                               "Qwen__Qwen2.5-0.5B",        inj_c17),
    ("unresolved",   "미해결 유도 상수 잔존",                    "Qwen__Qwen2.5-0.5B",        inj_unresolved),
    ("membership",   "가중치 축이 그 모듈이 안 읽는 필드의 이름", "Qwen__Qwen2.5-0.5B",       inj_membership),
    ("membership_notrun", "소속 검사 미수행을 통과로 읽지 않는가",   "Qwen__Qwen2.5-0.5B",       inj_membership_notrun),
    ("weight_operand", "같은 가중치가 한 행 안에서 두 이름",        "meta-llama__Llama-3.1-8B",  inj_weight_operand),
    ("unanswered",     "의뢰서 질문에 판정이 하나도 없음",          "Qwen__Qwen2.5-0.5B",       inj_unanswered),
    ("uncited",       "소스 인용 없는 교정 주장",                 "Qwen__Qwen2.5-0.5B",       inj_uncited),
    ("claim_only",    "방법 서술만 바뀌고 판정은 동일",           "Qwen__Qwen2.5-0.5B",       inj_claim_only),
    ("soft_undet",    "밖을 안 찾아보고 확인 못함 처리",          "Qwen__Qwen2.5-0.5B",       inj_soft_undetermined),
    ("axis_conflict", "한 등가류에 두 이름",                     "Qwen__Qwen2.5-0.5B",       inj_axis_conflict),
    ("bad_stub",      "지목 불가능한 인계 초안",                  "deepseek-ai__DeepSeek-V4-Pro", inj_bad_stub),
    ("dead_confirm",  "더 이상 맞지 않는 확인 기록",              "Qwen__Qwen2.5-0.5B",       inj_dead_confirm),
    ("uncited_confirm", "근거 없는 확인 기록",                    "Qwen__Qwen2.5-0.5B",       inj_uncited_confirm),
]

# 외부 대조 검사는 scan_model 지표가 아니라 별도 함수라 따로 돌린다.
EXTERNAL_CASES = [
    ("attention_layers", "attention 레이어 수 오판", "tiiuae__falcon-7b", inj_attn_layers),
    ("kv_cache",         "KV cache 카드 수치 오류",  "openai__gpt-oss-20b", inj_kv_card),
]


# ------------------------------------------------- STATIC / BASELINE 검사 자기검증
# 이 둘은 모델 폴더가 아니라 rules/ 와 baseline.json을 읽으므로 샌드박스 방식이 다르다.
# rules/를 실제로 망가뜨리는 건 위험하므로(2026-07-29에 PowerShell 정규식이 symbols.yaml을
# 손상시킨 전력이 있다) **검사 로직에 직접 결함 입력을 먹인다**.

def _static_cases():
    import summarize
    out = []

    def run_static_with(symbols):
        orig = summarize.load_symbols
        summarize.load_symbols = lambda *a, **k: symbols
        before = len(V.failures)
        buf, real_out = io.StringIO(), sys.stdout
        try:
            sys.stdout = buf
            V.check_static()
        finally:
            sys.stdout = real_out
            summarize.load_symbols = orig
        n = len(V.failures) - before
        del V.failures[before:]
        return n

    good = {"d_model": {"aliases": ["hidden_size"], "dim": True, "priority": 1}}
    base_noise = run_static_with(good)          # 정상 입력에서 나오는 FAIL 수(0이어야 함)

    out.append(("static:priority중복", "dim priority가 겹치면 렌더 순서가 비결정적",
                run_static_with({
                    "a": {"aliases": ["x"], "dim": True, "priority": 3},
                    "b": {"aliases": ["y"], "dim": True, "priority": 3}}) > base_noise))
    out.append(("static:priority누락", "dim인데 priority가 없음",
                run_static_with({"a": {"aliases": ["x"], "dim": True}}) > base_noise))
    out.append(("static:scope정규식", "scope 정규식이 컴파일 불가",
                run_static_with({"a": {"aliases": ["x"], "scope": "(("}}) > base_noise))
    out.append(("static:출처없음", "aliases도 from도 없는 심볼",
                run_static_with({"a": {"meaning": "x"}}) > base_noise))

    # YAML 중복 키: safe_load는 조용히 마지막 것만 남긴다 -> 규칙이 소리없이 사라진다.
    dup = "a: 1\nb: 2\na: 3\n"
    caught = False
    try:
        yaml.load(io.StringIO(dup), V._DupCheckLoader)
    except Exception:
        caught = True
    out.append(("static:YAML중복키", "같은 키 두 번 = 앞 규칙이 조용히 증발", caught))
    out += _propagate_cases()
    out += _square_cases()
    return out


def _square_cases():
    """정사각 활성 탐지(source_check.square_labels)가 **실제 크기**를 보는지 확인한다.

    이 검사는 "두 축의 폭이 같아 값으로는 못 가리는 자리"를 소스에 물어보라고 뽑아낸다.
    2026-08-15까지는 렌더된 **이름**이 같은지를 봤는데, 그건 묻고 있는 질문의 답을 이미
    안다고 가정하는 것이다 — Falcon-H1 은 `torch.ones(chunk_size, chunk_size)` 를 짓는데
    이름이 `d_state`/`d_chunk` 로 갈려 있어서 검사가 그냥 지나쳤다.
    """
    import csv as _csv
    import tempfile as _tf
    import source_check as SC

    def run(concrete, labels, weight="", weight_concrete=None):
        d = _tf.mkdtemp()
        os.makedirs(os.path.join(d, "full"))
        with open(os.path.join(d, "full", "prefill.csv"), "w",
                  newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            w.writerow(["op_id", "op_type", "input_shape", "weight_shape", "output_shape"])
            w.writerow(["0", "ones", "", weight, "[" + ", ".join(labels) + "]"])
        with open(os.path.join(d, "full", "prefill.shapes.concrete.jsonl"), "w",
                  encoding="utf-8") as f:
            f.write(json.dumps({"op_id": 0, "input_shape": [],
                                "weight_shape": weight_concrete,
                                "output_shape": [concrete]}) + "\n")
        try:
            return SC.square_labels(d)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    return [
        ("square:이름갈림", "실제로 정사각인데 이름이 갈리면 예전 검사는 눈을 감았다",
         run([256, 256], ["d_state", "d_chunk"]) == {"d_state", "d_chunk"}),
        ("square:값다름", "이름이 같아도 실제 폭이 다르면 정사각이 아니다",
         run([256, 128], ["d_state", "d_state"]) == set()),
        ("square:가중치제외", "전치돼 들어온 가중치는 질문이 아니다 (nn.Linear(d,d))",
         run([896, 896], ["d_model", "n_h*d_head"],
             weight="[n_h*d_head, d_model]", weight_concrete=[896, 896]) == set()),
    ]


def _propagate_cases():
    """계열 판정 승격(propagate_verdicts.py)이 **아무 데나 옮기지 않는지** 확인한다.

    이 도구는 한 모델에서 내린 판정을 같은 아키텍처의 다른 모델에 복사한다. 잘못 옮기면
    근거 없는 이름이 조용히 퍼진다 — 이 저장소가 계속 잡아온 실패 그대로다. 그래서
    "옮기지 말아야 할 때 안 옮기는가"를 결함 주입으로 확인한다.

    **양성 대조가 먼저다.** 아무것도 안 옮기는 도구는 음성 시험을 전부 통과하므로,
    정상 입력에서 실제로 1건이 나오는 것을 먼저 확인하지 않으면 나머지가 무의미하다.
    """
    import propagate_verdicts as P

    SIG = frozenset(["DecoderLayer", "Attention"])
    ANCHOR = {"module": "self_attn$", "op_type": "split_with_sizes", "nth": 2,
              "field": "o", "shape_index": 1, "axis": 3}
    verdict = dict(ANCHOR, model="A", **{"from": "d_nope", "to": "d_v"},
                   shape=["B", "n_h", "T", "d_nope"], expect=128, source="src")
    item = {"current_label": "d_nope", "size": 128, "candidates": ["d_nope", "d_v"],
            "override_stub": dict(ANCHOR, shape=["B", "n_h", "T", "d_nope"])}

    def n(ents=None, groups=None, items=None):
        return len(P.candidates("override", ents=[dict(verdict, **(ents or {}))],
                                groups=groups or {SIG: ["A", "B"]},
                                unsettled=lambda m: [] if m == "A" else
                                          [dict(item, **(items or {}))]))

    return [
        ("propagate:양성대조", "같은 계열의 같은 자리면 실제로 1건이 나와야 한다", n() == 1),
        ("propagate:이름불일치", "형제의 현재 이름이 다르면 같은 자리가 아니다",
         n(items={"current_label": "d_head"}) == 0),
        ("propagate:자리불일치", "앵커 하나(nth)만 달라도 다른 코드 줄이다",
         n(items={"override_stub": dict(ANCHOR, nth=0,
                                        shape=["B", "n_h", "T", "d_nope"])}) == 0),
        ("propagate:계열분리", "아키텍처가 다르면 소스가 다르므로 옮기면 안 된다",
         n(groups={SIG: ["A"], frozenset(["Other"]): ["B"]}) == 0),
        ("propagate:모호초안", "그 모델에서 초안이 한 자리를 못 짚으면 옮기면 안 된다",
         n(items={"stub_ambiguous": True}) == 0),
        ("propagate:선택자없음", "구조 선택자 없는 옛 판정은 값만 보고 옮기면 안 된다",
         n(ents={"op_type": None}) == 0),
        ("propagate:후보밖이름", "그 모델이 후보로 두지도 않은 이름은 옮기면 안 된다",
         n(items={"candidates": ["d_nope", "d_head"]}) == 0),
    ]


def _baseline_case():
    """퇴행 검사: 지표가 나빠졌는데 통과시키면 안 된다."""
    import tempfile as _tf
    # check_baseline이 읽는 지표를 전부 갖춘 스텁(하나라도 빠지면 KeyError로 조용히 못 돈다).
    stub = {"c_fail": 0, "c17": "PASS", "unresolved": 0, "bare": 10, "bare_pct": 0.0,
            "unknown_syms": 0, "kv_card": None, "weight_T": 0, "self_contra": 0,
            "label_false": 0, "param_incons": 0, "flow_wrong": 0, "flow_ambig": 0,
            "head_excl": 0, "resid_norm": 0, "batch_excl": 0,
            "heur": 0, "ident_incons": 0, "reshape_incons": 0,
            "matmul_compose": 0}
    fleet_before = {"m": dict(stub)}
    fleet_after = {"m": dict(stub, bare=99)}                # bare 악화
    fd = os.path.join(_tf.mkdtemp(), "baseline.json")
    json.dump(fleet_before, open(fd, "w"))
    real, V.BASELINE = V.BASELINE, fd
    before = len(V.failures)
    buf, real_out = io.StringIO(), sys.stdout
    try:
        sys.stdout = buf
        V.check_baseline(fleet_after, False)
    finally:
        sys.stdout = real_out
        V.BASELINE = real
    n = len(V.failures) - before
    del V.failures[before:]
    return n > 0


def _sandbox(name):
    tmp = tempfile.mkdtemp(prefix="vst_")
    shutil.copytree(os.path.join(MODELS, name), os.path.join(tmp, name))
    return tmp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    print("게이트 자기검증 — 각 검사에 그 검사가 잡아야 할 결함을 주입한다\n")
    ok = bad = 0

    print(f"   {'검사':16s} {'모델':38s} {'정상':>6} {'주입후':>7}  판정")
    for metric, desc, model, inject in CASES:
        if not os.path.isdir(os.path.join(MODELS, model)):
            print(f"   {metric:16s} {model:38s}  SKIP (모델 없음)")
            continue
        clean = V.scan_model(model)[metric]
        clean_n = 0 if clean in (0, "", "PASS") else 1
        tmp = _sandbox(model)
        try:
            n = inject(os.path.join(tmp, model))
            real, V.MODELS = V.MODELS, tmp
            dirty = V.scan_model(model)[metric]
            V.MODELS = real
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        dirty_n = 0 if dirty in (0, "", "PASS") else 1
        caught = (clean_n == 0) and (dirty_n == 1) and n > 0
        print(f"   {metric:16s} {model:38s} {str(clean):>6} {str(dirty):>7}  "
              f"{'OK — 잡음' if caught else 'FAIL — 못 잡음'}")
        if args.verbose:
            print(f"        {desc} (주입 {n}건)")
        ok, bad = (ok + 1, bad) if caught else (ok, bad + 1)

    print()
    for metric, desc, model, inject in EXTERNAL_CASES:
        if not os.path.isdir(os.path.join(MODELS, model)):
            print(f"   {metric:16s} {model:38s}  SKIP (모델 없음)")
            continue
        tmp = _sandbox(model)
        try:
            n = inject(os.path.join(tmp, model))
            real, V.MODELS = V.MODELS, tmp
            before = len(V.failures)
            # 샌드박스엔 모델이 하나뿐이라 나머지는 전부 "트레이스 없음" WARN이 된다.
            # 그 소음은 삼키고 FAIL 발생 여부만 본다.
            buf, real_out = io.StringIO(), sys.stdout
            try:
                sys.stdout = buf
                V.check_external({model: V.scan_model(model)})
            finally:
                sys.stdout = real_out
            after = len(V.failures)
            V.MODELS = real
            del V.failures[before:]           # 주입 실패를 최종 집계에 남기지 않는다
            del V.warnings[:]
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        caught = after > before and n > 0
        print(f"   {metric:16s} {model:38s} {'':>6} {'':>7}  "
              f"{'OK — 잡음' if caught else 'FAIL — 못 잡음'}")
        if args.verbose:
            print(f"        {desc} (주입 {n}건)")
        ok, bad = (ok + 1, bad) if caught else (ok, bad + 1)

    print()
    for name, desc, caught in _static_cases():
        print(f"   {name:16s} {'(검사 로직에 직접 입력)':38s} {'':>6} {'':>7}  "
              f"{'OK — 잡음' if caught else 'FAIL — 못 잡음'}")
        if args.verbose:
            print(f"        {desc}")
        ok, bad = (ok + 1, bad) if caught else (ok, bad + 1)

    caught = _baseline_case()
    print(f"   {'baseline':16s} {'(지표 악화 시뮬레이션)':38s} {'':>6} {'':>7}  "
          f"{'OK — 잡음' if caught else 'FAIL — 못 잡음'}")
    if args.verbose:
        print("        이전 기준보다 지표가 나빠졌는데 통과시키면 안 된다")
    ok, bad = (ok + 1, bad) if caught else (ok, bad + 1)

    print("\n" + "=" * 72)
    print(f"살아 있는 검사 {ok}개 / 죽은 검사 {bad}개")
    if bad:
        print("죽은 검사가 있다 — 그 검사의 'FAIL 0'은 아무것도 보장하지 않는다.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
