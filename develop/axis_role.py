"""텐서의 **역할**(q/k/v, hidden/B/C …)을 재추적 없이 지금 트레이스에서 되짚는다. 읽기 전용.

WHY THIS EXISTS
---------------
위치만으로는 역할을 못 가린다. `[B, n_h_lin_v, T, d_head_lin_k]` 의 마지막 축은 그 텐서가
query/key 면 K, value 면 V 인데 **shape 은 셋 다 똑같다**. 실제로 그 착각이 출하됐다 --
Qwen3-Next 의 `transpose nth 2` 를 value 로 읽은 판정이 query 를 뒤집어 216축을 거꾸로
고치고 있었다(2026-08-16).

역할을 크기로 되짚으려는 시도는 **이미 실패했다.** 전치 순열을 크기로 역산해 축 등가류
간선을 놓아 봤더니(92.7% 유일 역산, 모호 0건) Zamba2 에서 물렸다: `repeat_kv` 뒤 key 축은
소스상 `n_h` 인데 등가류가 그 경계를 넘어 `n_kv` 를 밀어 넣었고, n_h == n_kv == 32 라
HF 가 `if n_rep == 1: return hidden_states` 로 빠져나가 **끊을 op 이 트레이스에 없었다**
(`axis_classes.py` docstring 참고).

그래서 이 패스는 **크기도, 기존 축 등가류도 역할 근거로 쓰지 않는다.** 쓰는 것은 넷뿐:

    params        `*.q_proj.weight` / `*.k_proj.weight` / `*.v_proj.weight` -- 실제로 쓴 파라미터
    depends_on    부모가 **정확히 하나**인 unary/복사 경로만 따라간다
    output_index  split 출력은 그 자리 자체가 신원이다
    module_path   leaf 모듈 이름

그리고 **막아야 할 자리에서 멈춘다**: matmul/concat/브로드캐스트/복제(expand·repeat)는
부모의 역할을 물려받으면 안 된다. 특히 복제는 `n_kv -> n_h` 처럼 **역할이 바뀌는 경계**다.

실행:
    .venv\\Scripts\\python.exe develop\\axis_role.py --model Qwen3-Next
    .venv\\Scripts\\python.exe develop\\axis_role.py           # 전 모델 커버리지
"""
import argparse
import collections
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
MODELS = os.path.join(PROJ, "models")
sys.path.insert(0, os.path.join(PROJ, "src"))

import axis_classes as AC          # noqa: E402  (op_ordinals / module_key_of 만 쓴다)
import build_table as BT           # noqa: E402  (구체 shape 사이드카 로더)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# 파라미터 이름 -> 역할. 이름이 곧 저자의 의도다: `k_proj.weight` 를 쓴 matmul 의 출력은 key 다.
_PARAM_ROLE = [
    (re.compile(r"\.q_proj\.(weight|bias)$"), "q"),
    (re.compile(r"\.k_proj\.(weight|bias)$"), "k"),
    (re.compile(r"\.v_proj\.(weight|bias)$"), "v"),
    (re.compile(r"\.o_proj\.(weight|bias)$"), "o"),
    (re.compile(r"\.in_proj_qkvz\.(weight|bias)$"), "qkvz"),
    (re.compile(r"\.in_proj_ba\.(weight|bias)$"), "ba"),
    (re.compile(r"\.in_proj\.(weight|bias)$"), "in_proj"),
    (re.compile(r"\.out_proj\.(weight|bias)$"), "out_proj"),
    (re.compile(r"\.conv1d\.(weight|bias)$"), "conv"),
    (re.compile(r"\.dt_proj\.(weight|bias)$"), "dt"),
    (re.compile(r"\.x_proj\.(weight|bias)$"), "x_proj"),
]

# 부모의 역할을 그대로 물려받아도 되는 op. **모양만 바꾸거나 복사하는 것**뿐이다.
# `expand`/`repeat_interleave`/`repeat` 는 일부러 뺐다 -- 복제는 `n_kv -> n_h` 처럼 역할이
# 바뀌는 경계이고, 그 경계를 넘은 것이 전치 간선 실험을 무너뜨린 원인이다.
_PASS = frozenset({
    "view", "reshape", "_unsafe_view", "clone", "contiguous", "_to_copy", "alias",
    "detach", "transpose", "permute", "slice", "select", "unsqueeze", "squeeze",
    "flatten", "t", "silu", "gelu", "relu", "sigmoid", "tanh", "neg", "exp",
    "softplus", "rsqrt", "sqrt", "abs", "log", "cumsum", "tril", "triu",
})
# 여기서는 멈춘다. 부모가 여럿이거나(어느 쪽 역할인지 모른다) 역할이 새로 생기는 자리다.
_STOP = frozenset({
    "matmul", "batched_matmul", "linear", "mm", "bmm", "addmm", "concat", "cat",
    "stack", "expand", "repeat", "repeat_interleave", "broadcast_to", "index",
    "index_put_", "scatter", "gather", "einsum",
})


def roles(model: str, phase: str):
    """{(op_id, output_index): 역할}. 근거는 params / depends_on / split 자리뿐이다."""
    p = os.path.join(MODELS, model, "full", f"{phase}.trace.raw.jsonl")
    if not os.path.exists(p):
        return None, None
    rows = [json.loads(l) for l in io.open(p, encoding="utf-8")]
    conc = BT.load_concrete(os.path.join(MODELS, model), phase) or {}
    byid = {r["op_id"]: r for r in rows}
    ordn = AC.op_ordinals(rows)
    role, why = {}, {}

    for r in rows:                                    # 1) 파라미터로 씨를 뿌린다
        for nm in (r.get("params") or []):
            for rx, tag in _PARAM_ROLE:
                if rx.search(nm):
                    role[(r["op_id"], 0)] = tag
                    why[(r["op_id"], 0)] = f"param {nm.rsplit('.', 2)[-2]}"
                    break

    for r in rows:                                    # 2) split 출력은 그 자리가 신원이다
        if r.get("op_type") in ("split_with_sizes", "split", "chunk", "unbind"):
            n = len(r.get("output_shape") or [])
            for i in range(n):
                key = (r["op_id"], i)
                if key not in role:
                    # **`nth` 를 태그에 넣어야 한다.** 한 모듈에 split 이 여럿이면 출력 인덱스만
                    # 으로는 서로 다른 텐서가 같은 이름을 갖는다 -- Qwen3-Next 의 linear_attn 은
                    # qkvz split(nth0, 출력 4개)과 conv 뒤 qkv split(nth2, 출력 3개)이 둘 다
                    # 있어서 `out2` 가 각각 "qkvz 의 value 조각"과 "post-conv value" 를 가리킨다
                    # (폭도 256 vs 4096 으로 다르다). 태그가 겹치면 역할 대조가 조용히 틀린다.
                    role[key] = f"split{ordn.get(r['op_id'])}#{i}"
                    why[key] = (f"{AC.module_key_of(r.get('module_path') or '')}"
                                f"/{r.get('op_type')}/nth{ordn.get(r['op_id'])}/out{i}")

    # 3) **부모가 정확히 하나인 모양-보존 op 만** 따라간다. op_id 순이라 한 번에 수렴한다.
    for r in rows:
        oid, ot = r["op_id"], r.get("op_type")
        if (oid, 0) in role or ot in _STOP or ot not in _PASS:
            continue
        deps = r.get("depends_on") or []
        if len(deps) != 1:
            continue                                  # 부모가 여럿이면 어느 역할인지 모른다
        par = byid.get(deps[0])
        if par is None:
            continue
        outs = (conc.get(par["op_id"]) or {}).get("output_shape") or []
        oi = 0
        if len(outs) != 1:
            # 부모의 출력이 여럿이면 어느 조각을 받았는지 신원으로는 알 수 없다. 다만 조각들의
            # 구체 크기가 **서로 다르고** 그중 정확히 하나가 이 op 의 피연산자와 같다면 복원이
            # 된다 -- 크기를 **역할의 근거로** 쓰는 게 아니라 **어느 출력인지 고르는 데만**
            # 쓴다(같은 크기가 둘 이상이면 포기한다). 축 등가류의 간선 (1) 과 같은 판정이다.
            mine = (conc.get(oid) or {}).get("input_shape") or []
            if len(mine) != 1:
                continue
            hits = [k for k, x in enumerate(outs) if x == mine[0]]
            if len(hits) != 1 or sum(1 for x in outs if x == mine[0]) != 1:
                continue
            oi = hits[0]
        if (par["op_id"], oi) in role:
            role[(oid, 0)] = role[(par["op_id"], oi)]
            why[(oid, 0)] = why[(par["op_id"], oi)]
    return rows, (role, why)


def cover(model: str):
    """인계 항목마다 그 앵커 자리의 역할을 붙여 본다."""
    out = collections.Counter()
    detail = []
    for phase in ("prefill", "decode"):
        rows, rw = roles(model, phase)
        if not rows:
            continue
        role, why = rw
        ordn = AC.op_ordinals(rows)
        up = os.path.join(MODELS, model, "full", f"{phase}.unsettled.json")
        if not os.path.exists(up):
            continue
        items = (json.load(io.open(up, encoding="utf-8")) or {}).get("items") or []
        for it in items:
            st = it["override_stub"]
            rx = re.compile(st["module"])
            fld = "input_shape" if st["field"] == "i" else "output_shape"
            got = collections.Counter()
            for r in rows:
                if not rx.search(AC.module_key_of(r.get("module_path") or "") or ""):
                    continue
                if r.get("op_type") != st["op_type"] or ordn.get(r["op_id"]) != st["nth"]:
                    continue
                shs = r.get(fld) or []
                si = st["shape_index"]
                if si >= len(shs) or not isinstance(shs[si], list):
                    continue
                if [str(x) for x in shs[si]] != [str(x) for x in st["shape"]]:
                    continue
                # 출력 자리면 그 op 의 역할, 입력 자리면 그 부모의 역할
                if st["field"] == "o":
                    tag = role.get((r["op_id"], si))
                    ev = why.get((r["op_id"], si))
                else:
                    deps = r.get("depends_on") or []
                    tag = role.get((deps[0], 0)) if len(deps) == 1 else None
                    ev = why.get((deps[0], 0)) if len(deps) == 1 else None
                got[(tag, ev)] += 1
            tags = {t for (t, _e) in got if t}
            if not tags:
                out["역할없음"] += 1
            elif len(tags) > 1:
                out["역할혼재"] += 1
            else:
                out["역할확정"] += 1
                if len(detail) < 12:
                    (t, ev), _ = got.most_common(1)[0]
                    detail.append((st["op_type"], st["axis"], it["current_label"],
                                   t, ev, it["anchor_shape"]))
    return out, detail


def main():
    ap = argparse.ArgumentParser(description="텐서 역할을 지금 트레이스에서 되짚는다 (읽기 전용)")
    ap.add_argument("--model", default="")
    a = ap.parse_args()
    MIX = re.compile(r"mamba|mixer|ssm|linear_attn")
    grand = collections.Counter()
    for m in sorted(os.listdir(MODELS)):
        if not os.path.isdir(os.path.join(MODELS, m)):
            continue
        if a.model and a.model.lower() not in m.lower():
            continue
        c, detail = cover(m)
        if not c:
            continue
        grand.update(c)
        print(f"{m.split('__')[-1][:34]:<36} {dict(c)}")
        if a.model:
            for op, ax, cur, tag, ev, sh in detail:
                print(f"    {op:<16}ax{ax} {cur:<14} 역할={tag:<10} ({ev})  {sh}")
    tot = sum(grand.values())
    ok = grand["역할확정"]
    print(f"\n전체 {tot}건 중 역할 확정 {ok}건 ({100*ok//max(tot,1)}%), "
          f"혼재 {grand['역할혼재']}, 없음 {grand['역할없음']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
