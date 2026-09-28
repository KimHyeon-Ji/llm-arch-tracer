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

stage 를 무엇으로 가르는가 (2026-09-28 에 바꿨다)
---------------------------------------------
처음에는 **op 순서**로 갈랐다 -- 층 안의 마지막 mix 그룹이 post, 그 앞이 pre. 현재 데이터에서
정확했지만 외부 검토(R2/R3)가 깨질 방식을 짚었다: 중간에 view/cast 가 끼면 mix 를 append 로
오인하고, mix 가 셋 이상이면 조용히 `pre, pre, ..., post` 로 배정되며, 의미가 op 순서에
의존한다.

그래서 **파라미터 lineage** 로 바꿨다. 각 mix 그룹의 `elementwise_mul` 이 `[d_model]` 폭의
norm 가중치를 소비하는데, 그 가중치 이름이 stage 를 직접 말한다:

    self_attention_res_norm  -> mix_pre     (pre-attention 혼합)
    mlp_res_norm             -> mix_post    (post-attention, MLP 앞)
    output_attn_res_norm     -> mix_final   (전 층 종료 후)

실측: prefill/decode 각각 self_attention_res_norm 23, mlp_res_norm 24,
output_attn_res_norm 1. 외부 검토가 독립으로 센 수와 같고, 옛 op-순서 판정과도 일치한다
(pre 가 24 가 아니라 23 인 것은 층 0 에 pre-mix 가 없기 때문이다 -- R1 의 "c_pre 는 l>0").

**cardinality 를 강제한다.** 추측하지 않고 실패시킨다:

    층 0            pre 0, post 1
    그 밖의 층      pre 1, post 1
    boundary_append l % R_res == 0 인 층에만
    final           정확히 1

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

# **투명 op.** 텐서를 바꾸지 않고 통과시키는 것들. concat 과 혼합 체인 사이에 이런 op 이
# 끼면 직접 consumer 만 보는 판별은 mix 를 append 로 오인한다 -- 외부 검토(R3b)가 짚었다.
# 그래서 consumer 를 볼 때 이것들을 **건너뛴다**.
TRANSPARENT_OPS = ("view", "_unsafe_view", "reshape", "clone", "contiguous",
                   "_to_copy", "detach", "alias", "squeeze", "unsqueeze",
                   "expand", "permute", "transpose", "slice", "copy_")

# **stage 를 말하는 norm 가중치.** 그룹의 op 가 이 파라미터를 소비한다.
NORM_STAGE = {
    "self_attention_res_norm": "mix_pre",
    "mlp_res_norm": "mix_post",
    "output_attn_res_norm": "mix_final",
}

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


def _reaches_mix(oid, consumers, by_op, depth=6):
    """`oid` 의 하류가 **투명 op 을 건너뛰어** 혼합 체인 op 에 닿는가.

    닿으면 mix 그룹의 머리이고, 안 닿으면(다른 concat 으로만 가거나 소비자가 없으면)
    버퍼를 키우는 boundary_append 다.
    """
    seen, cur = {oid}, [oid]
    for _ in range(depth):
        nxt = []
        for o in cur:
            for c in consumers.get(o, ()):
                ci = int(c["op_id"])
                ot = c.get("op_type")
                if ot in MIX_OPS:
                    return True
                if ci in seen or ot not in TRANSPARENT_OPS:
                    continue
                seen.add(ci)
                nxt.append(ci)
        if not nxt:
            break
        cur = nxt
    return False


def stages(phase, rows, strict=True, R_res=None):
    """op_id -> stage. lineage 로만 정한다.

    `strict` 는 cardinality 강제(층별 pre/post 수, final 1 개). 실제 모델에는 켠다.
    fixture 의 단일 사례처럼 **일부만 든 입력**에는 끈다 -- 그때는 stage 배정만 본다.
    (cardinality 가 실제로 발화하는지는 fixture 의 전용 사례가 시험한다.)
    """
    tk = TOKENS[phase]
    consumers = collections.defaultdict(list)
    by_op = {int(r["op_id"]): r for r in rows}
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
        # **투명 op 을 건너뛰며** 혼합 체인에 닿는지 본다. 직접 consumer 만 보면
        # concat 뒤에 view/cast 가 끼는 경우를 append 로 오인한다(외부 검토 R3b).
        to_mix = _reaches_mix(int(r["op_id"]), consumers, by_op)
        (mixes if to_mix else appends).append(r)

    by_id = {int(r["op_id"]): r for r in rows}
    stage_of = {}
    per_layer = collections.defaultdict(collections.Counter)
    per_cover = collections.defaultdict(set)
    for r in mixes:
        oid = int(r["op_id"])
        # 그룹 구성원을 먼저 모은다
        members, seen, cur = [oid], {oid}, [oid]
        while cur:
            nxt = []
            for o in cur:
                for c in consumers[o]:
                    ci = int(c["op_id"])
                    ot = c.get("op_type")
                    if ci in seen or ot not in (MIX_OPS + TRANSPARENT_OPS):
                        continue
                    seen.add(ci)
                    if ot in MIX_OPS:
                        members.append(ci)   # 투명 op 은 통과만 시키고 구성원은 아니다
                    nxt.append(ci)
            cur = nxt
        # **파라미터 lineage.** 그룹의 op 가 소비하는 norm 가중치 이름이 stage 를 말한다.
        norms = set()
        for m in members:
            for d in (by_id[m].get("depends_on") or []):
                for p in (by_id.get(int(d), {}).get("params") or []):
                    for tag, stg in NORM_STAGE.items():
                        if f".{tag}." in p:
                            norms.add(stg)
        if len(norms) != 1:
            raise ValueError(
                f"{phase} op{oid}: stage 를 파라미터로 정할 수 없다 (찾은 것 {norms}). "
                f"norm 가중치가 없거나 둘 이상이다 -- 추측하지 않는다")
        st = norms.pop()
        mp = r.get("module_path") or ""
        per_layer[mp][st] += 1
        per_cover[mp] |= set(C.expand_layers(r.get("layers")))
        for m in members:
            stage_of[m] = st
    for r in appends:
        stage_of[int(r["op_id"])] = "boundary_append"
        per_layer[r.get("module_path") or ""]["boundary_append"] += 1
        per_cover[r.get("module_path") or ""] |= set(C.expand_layers(r.get("layers")))

    # **cardinality 강제.** 추측하지 않고 실패시킨다.
    if not strict:
        return stage_of
    if R_res is None:
        raise ValueError("strict cardinality 검사는 R_res 가 필요하다")
    seen_layers, covered = set(), set()
    # **층별 검사를 먼저 한다.** 전역 final 수를 먼저 보면 구체적인 위반(층 안의 mix 수,
    # 경계 아닌 append)이 가려진다 -- 진단이 덜 쓸모 있어진다.
    for mp, c in per_layer.items():
        if mp == "model":
            continue
        li = int(mp.rsplit(".", 1)[-1]) if mp.rsplit(".", 1)[-1].isdigit() else None
        if li is None:
            continue
        want_pre = 0 if li == 0 else 1
        if c.get("mix_pre", 0) != want_pre:
            raise ValueError(f"{phase} 층 {li}: mix_pre 가 {c.get('mix_pre', 0)} 개다 "
                             f"({want_pre} 이어야 한다)")
        if c.get("mix_post", 0) != 1:
            raise ValueError(f"{phase} 층 {li}: mix_post 가 {c.get('mix_post', 0)} 개다 "
                             f"(1 이어야 한다)")
        # **append 는 l % R_res == 0 인 층에만** (외부 검토 R3b).
        want_app = 1 if li % R_res == 0 else 0
        got_app = c.get("boundary_append", 0)
        if got_app != want_app:
            raise ValueError(
                f"{phase} 층 {li}: boundary_append 가 {got_app} 개다 "
                f"({want_app} 이어야 한다 -- l % R_res == {li % R_res})")
        seen_layers.add(li)
        covered |= per_cover.get(mp, {li})
    # **기대한 층이 통째로 누락됐는지** 도 본다.
    # 발행본은 접혀 있어 대표 층만 module_path 를 가진다 -- `layers` 를 펼쳐 합집합으로
    # 본다. 처음에는 층 인덱스마다 group 이 있어야 한다고 썼는데, 접힘을 잊은 것이라
    # 64 개 층이 거짓으로 누락돼 보였다.
    if covered:
        want = set(range(max(covered) + 1))
        missing = sorted(want - covered)
        if missing:
            raise ValueError(f"{phase}: residual group 이 덮지 않는 층 {missing[:6]} "
                             f"(총 {len(missing)}) -- 통째로 누락됐다")
    n_final = sum(c.get("mix_final", 0) for c in per_layer.values())
    if n_final != 1:
        raise ValueError(f"{phase}: mix_final 이 {n_final} 개다 (1 이어야 한다)")
    return stage_of


def cells(phase, rows, R, L, strict=True):
    """(key, before, after_token, stage, formula) 목록. 식이 안 맞으면 예외."""
    tk = TOKENS[phase]
    stage_of = stages(phase, rows, strict=strict, R_res=R)
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
