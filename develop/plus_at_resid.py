r"""residual 누적 축의 **stage 를 op lineage 로** 판별한다. 값으로 고르지 않는다.

왜 lineage 인가
---------------
외부 검토(R3, 2026-09-27): "여러 식이 같은 값을 내는 경우가 있어 셀별 semantic stage 까지
확정한 것은 아닙니다. 따라서 값으로 식을 고르지 말고 op lineage 로 stage 를 판별해야
합니다 -- buffer_pre / mix_pre / boundary_append / buffer_post / mix_post / buffer_final /
mix_final."

실측이 그 경고를 확인했다. 1,262 셀 중 식이 **유일하게** 결정되는 것은 216 셀뿐이고
950 셀이 두 식, 96 셀이 세 식에 맞는다. 값으로 고르면 절반 이상을 찍는 것이다.

식 (R1 blind 도출, 2026-09-27)
------------------------------
`R = attn_res_block_size`, `L = num_hidden_layers`, `l = layer_idx` (0-based):

    b_in(l)    = ceil(l/R)           레이어 진입 시 저장된 residual stream 수
    b_after(l) = ceil((l+1)/R)       경계 처리(l % R == 0 에서 추가) 후 저장 수
    c_pre(l)   = ceil(l/R) + 1       pre-attention 혼합 후보 수   (l > 0)
    c_post(l)  = ceil((l+1)/R) + 1   post-attention 혼합 후보 수
    b_final    = ceil(L/R)           전 층 종료 후 저장 수
    c_final    = ceil(L/R) + 1       최종 output 혼합 후보 수

근거: `modeling_kimi_linear.py:1188-1192`(저장소가 너비 0 으로 시작),
`:995-998`(l % R == 0 일 때 stream 하나 추가),
`:1075-1087`(`_apply_attn_res` 가 저장소와 현재 prefix 를 concat 하고 그 축으로
score/softmax/BMM 을 수행).

관측한 구조
-----------
    boundary_append   concat  [tok,a,d_model] + [tok,1,d_model] -> [tok,a+1,d_model]
                      출력이 **다른 concat 의 피연산자 0** 으로 쓰인다 (버퍼를 키운다)
    mix 그룹 (6 op)   concat -> elementwise_mul -> elementwise_mul -> sum -> softmax
                      -> batched_matmul.  concat 출력이 elementwise_mul 로 간다.

층 안의 mix 그룹은 **뒤에서부터** 배정한다 -- 마지막이 post, 그 앞이 pre. 층 0 은 버퍼가
비어 pre-mix 가 없으므로(R1 의 "c_pre 는 l>0") 그룹이 하나뿐이다. 앞에서부터 세면 그것을
pre 로 잘못 붙여 12 건이 어긋난다(실측).

**독립성의 한계 (정직하게 적는다)**
-----------------------------------
계열 A/B 는 적용기와 독립 matcher 가 규칙을 **따로** 구현해 대조했다. 계열 C 는 이 모듈
하나를 둘이 공유한다 -- lineage 분류기를 두 번 쓰는 것이 실질적 독립이 아니라고 판단했다.
대신 방어를 이렇게 둔다:

    1. 식 자체를 R1(blind)이 **독립 도출**했다. 내 판정과 별개로 같은 식에 도달했다.
    2. 적용기가 셀마다 **접힌 행의 모든 층**에서 식이 성립하는지 산술로 재확인한다
       (선택이 아니라 주장의 검증 -- 실패 방식이 다르다).
    3. 후보 완전성: shape 로 뽑은 residual 후보 전부가 footprint 에 있어야 한다.
       분류기가 한 셀이라도 놓치면 적용기가 잡는다.
    4. fixture 에 stage 별 사례를 둔다.

이 한계는 R3 재요청에 그대로 싣는다.
"""
import collections
import math

import plus_at_canon as C

TOKENS = {"prefill": "B*T", "decode": "B"}
MIX_OPS = ("elementwise_mul", "sum", "softmax", "batched_matmul")

# stage 와 자리 종류 -> 식 이름
FORMULA = {
    ("boundary_append", "buf"): "b_in",
    ("boundary_append", "mix"): "b_after",
    ("mix_pre", "buf"): "b_in",
    ("mix_pre", "mix"): "c_pre",
    ("mix_post", "buf"): "b_after",
    ("mix_post", "mix"): "c_post",
    ("mix_final", "buf"): "b_final",
    ("mix_final", "mix"): "c_final",
}

# 식 이름 -> 표에 쓰는 토큰. `l` 은 그 행의 layer_idx 다(symbols.yaml 에 선언한다).
TOKEN = {
    "b_in": "ceil(l/R_res)",
    "b_after": "ceil((l+1)/R_res)",
    "c_pre": "ceil(l/R_res)+1",
    "c_post": "ceil((l+1)/R_res)+1",
    "b_final": "ceil(L_layers/R_res)",
    "c_final": "ceil(L_layers/R_res)+1",
}


def value(name, l, R, L):
    return {"b_in": math.ceil(l / R),
            "b_after": math.ceil((l + 1) / R),
            "c_pre": math.ceil(l / R) + 1,
            "c_post": math.ceil((l + 1) / R) + 1,
            "b_final": math.ceil(L / R),
            "c_final": math.ceil(L / R) + 1}[name]


def _residual_module(mp):
    return mp == "model" or (mp.startswith("model.layers.") and mp.count(".") == 2)


def candidates(phase, rows):
    """shape 만 보고 뽑은 residual 후보 셀. **완전성 검사용**이다.

    축 0 이 tokens 이고 값이 2..9 인 정수 축. stage 는 여기서 정하지 않는다.
    """
    tk = TOKENS[phase]
    out = set()
    for r in rows:
        if not _residual_module(r.get("module_path") or ""):
            continue
        for field in ("input_shape", "output_shape"):
            for si, sh in enumerate(C.parse_jsonl_shape(r.get(field), field)):
                t = [str(x) for x in sh]
                if not t or t[0] != tk:
                    continue
                for ax, v in enumerate(t):
                    if v.isdigit() and 2 <= int(v) <= 9:
                        out.add((phase, int(r["op_id"]), field, si, ax))
    return out


def stages(phase, rows):
    """op_id -> stage. lineage 로만 정한다."""
    tk = TOKENS[phase]
    consumers = collections.defaultdict(list)
    for r in rows:
        for d in (r.get("depends_on") or []):
            consumers[int(d)].append(r)

    def sh(r, f):
        return C.parse_jsonl_shape(r.get(f), f)

    appends, mixes = [], []
    for r in rows:
        if r.get("op_type") != "concat":
            continue
        if not _residual_module(r.get("module_path") or ""):
            continue
        ins, outs = sh(r, "input_shape"), sh(r, "output_shape")
        if len(ins) != 2 or not outs or len(ins[0]) != 3:
            continue
        if str(ins[0][0]) != tk or str(ins[1][1]) != "1":
            continue
        cons = consumers[int(r["op_id"])]
        # **혼합 체인 op 로 가면 mix, 그 밖(다른 concat 등)이면 append.**
        # 처음에는 elementwise_mul 만 봤는데, 그건 실제 트레이스의 op 순서에 기댄 것이라
        # 체인이 짧은 경우를 놓친다(fixture 가 잡았다). 혼합 체인 전체를 본다.
        to_mix = any(c.get("op_type") in MIX_OPS for c in cons)
        (mixes if to_mix else appends).append(r)

    stage_of = {}
    per = collections.defaultdict(list)
    for r in mixes:
        mp = r.get("module_path") or ""
        per["final" if mp == "model" else mp].append(r)
    for key, grp in per.items():
        grp.sort(key=lambda x: int(x["op_id"]))
        for i, r in enumerate(grp):
            st = ("mix_final" if key == "final"
                  else "mix_post" if i == len(grp) - 1 else "mix_pre")
            stage_of[int(r["op_id"])] = st
            seen, cur = set(), [int(r["op_id"])]
            while cur:
                nxt = []
                for o in cur:
                    for c in consumers[o]:
                        ci = int(c["op_id"])
                        if ci in seen or c.get("op_type") not in MIX_OPS:
                            continue
                        seen.add(ci)
                        stage_of[ci] = st
                        nxt.append(ci)
                cur = nxt
    for r in appends:
        stage_of[int(r["op_id"])] = "boundary_append"
    return stage_of


def cells(phase, rows, R, L):
    """(key, before, after_token, stage, formula) 목록. 식이 안 맞으면 예외."""
    tk = TOKENS[phase]
    stage_of = stages(phase, rows)
    out = []
    for r in rows:
        oid = int(r["op_id"])
        st = stage_of.get(oid)
        layers = C.expand_layers(r.get("layers")) or [L]
        for field in ("input_shape", "output_shape"):
            for si, sh in enumerate(C.parse_jsonl_shape(r.get(field), field)):
                t = [str(x) for x in sh]
                if not t or t[0] != tk:
                    continue
                for ax, v in enumerate(t):
                    if not (v.isdigit() and 2 <= int(v) <= 9):
                        continue
                    if st is None:
                        raise ValueError(
                            f"{phase} op{oid} {field}[{si}] ax{ax} 값 {v}: "
                            f"stage 를 lineage 로 정할 수 없다")
                    which = "buf" if (r.get("op_type") == "concat" and
                                      field == "input_shape" and si == 0) else "mix"
                    name = FORMULA[(st, which)]
                    if not all(value(name, l, R, L) == int(v) for l in layers):
                        raise ValueError(
                            f"{phase} op{oid} {field}[{si}] ax{ax}: stage {st} 의 식 "
                            f"{name} 이 층 {layers[:4]} 에서 {v} 를 내지 않는다")
                    out.append(((phase, oid, field, si, ax), v, TOKEN[name],
                                st, name))
    return out
