r"""op 정의가 **보증하는 축 관계**를 선언적으로 계산한다. 계보 구현을 호출하지 않는다.

왜 별도 파일인가
----------------
`develop/provenance_oracle.py` 의 structural sentinel 이 `axis_classes.lineage_edges()` 를
불러 기대값을 만들면 **자기 구현을 자기가 검사하는 셈**이 된다. 규칙에 버그가 있어도 기대값이
같이 틀려서 통과한다. 그래서 기대 관계를 여기서 **독립적으로** 계산한다 -- 구체 shape 과
ATen 스칼라 인자만 읽고, `src/axis_classes.py` 는 import 하지 않는다.

두 종류를 낸다(외부 검토 2026-09-10):

  positive  같은 UF 여야 하는 축 쌍   -- 끊겨 있으면 계보가 못 이은 것
  negative  같은 UF 이면 **안 되는** 축 쌍 -- 이어져 있으면 계보가 잘못 이은 것

negative 가 없으면 "많이 잇는 것"이 항상 좋아 보인다. 실제로 이 프로젝트의 결함은 대부분
**잘못 이은 것**이었다.
"""
import json
import os

# ---- positive: op 정의가 "같은 축"이라고 보증하는 자리 ----------------------


def _expand_pairs(r, cins, couts):
    """`expand` 의 **크기가 변하지 않은** 축은 같은 축이다."""
    if r.get("op_type") not in ("expand", "broadcast_to", "expand_as"):
        return [], []
    if not cins or len(couts) != 1:
        return [], []
    ci, co = cins[0], couts[0]
    if not (isinstance(ci, list) and isinstance(co, list) and len(ci) == len(co)):
        return [], []
    pos, neg = [], []
    for a in range(len(ci)):
        if ci[a] == co[a] and ci[a] != 1:
            pos.append(((r["op_id"], "i", 0, a), (r["op_id"], "o", 0, a)))
        elif ci[a] == 1 and co[a] != 1:
            # 방송 축: 펼쳐진 축은 원래 축과 **같은 축이 아니다**(derived).
            neg.append(((r["op_id"], "i", 0, a), (r["op_id"], "o", 0, a)))
    return pos, neg


def _perm_of(op, args, rank):
    if not args:
        return None
    p = args.get("pos") or []
    if op == "transpose" and len(p) >= 2 and all(isinstance(x, int) for x in p[:2]):
        d0, d1 = p[0] % rank, p[1] % rank
        perm = list(range(rank))
        perm[d0], perm[d1] = perm[d1], perm[d0]
        return perm
    if op == "permute" and p and isinstance(p[0], list) and len(p[0]) == rank:
        return [d % rank for d in p[0]]
    return None


def _permute_pairs(r, cins, couts):
    """`transpose`/`permute` 는 축을 재배열만 한다 -- 순열이 대응을 **정의**한다."""
    op = r.get("op_type")
    if op not in ("transpose", "permute"):
        return [], []
    if len(cins) != 1 or len(couts) != 1:
        return [], []
    ci, co = cins[0], couts[0]
    if not (isinstance(ci, list) and isinstance(co, list) and len(ci) == len(co)):
        return [], []
    perm = _perm_of(op, r.get("scalar_args"), len(ci))
    if not perm or not all(ci[perm[i]] == co[i] for i in range(len(co))):
        return [], []
    pos = [((r["op_id"], "i", 0, perm[i]), (r["op_id"], "o", 0, i))
           for i in range(len(co)) if ci[perm[i]] != 1]      # 크기-1 은 정보가 없다
    return pos, []


def _split_pairs(r, cins, couts):
    """`split` 의 **비분할 축**은 그대로다. 분할 축은 **같은 축이 아니다**."""
    if r.get("op_type") not in ("split_with_sizes", "split", "chunk"):
        return [], []
    if not cins or len(couts) < 2:
        return [], []
    ci = cins[0]
    if not isinstance(ci, list):
        return [], []
    cand = [a for a in range(len(ci))
            if all(isinstance(c, list) and len(c) == len(ci) and c[a] != ci[a] for c in couts)]
    if len(cand) != 1:
        return [], []
    ax = cand[0]
    for c in couts:
        for j in range(len(ci)):
            if j != ax and c[j] != ci[j]:
                return [], []
    pos, neg = [], []
    for oi in range(len(couts)):
        for j in range(len(ci)):
            if j == ax:
                neg.append(((r["op_id"], "i", 0, j), (r["op_id"], "o", oi, j)))
            elif ci[j] != 1:
                pos.append(((r["op_id"], "i", 0, j), (r["op_id"], "o", oi, j)))
    return pos, neg


_RULES = {
    "expand": _expand_pairs,
    "permute": _permute_pairs,
    "split": _split_pairs,
}


def contracts(rows, conc):
    """{rule: {"pos": [...], "neg": [...]}}"""
    out = {k: {"pos": [], "neg": []} for k in _RULES}
    for r in rows:
        c = conc.get(r.get("op_id")) or {}
        cins = c.get("input_shape") or []
        couts = c.get("output_shape") or []
        for name, fn in _RULES.items():
            p, n = fn(r, cins, couts)
            out[name]["pos"].extend(p)
            out[name]["neg"].extend(n)
    return out


def barrier_contracts(sem):
    """`repeat_kv` 가 역할을 바꾼 축은 barrier 전후가 **같은 축이 아니다**.

    지금은 텐서 id 와 축 번호만 낸다 -- 노드 표현이 바뀌면(#4) 여기서 자리 쌍을 만든다.
    """
    return [{"in": e.get("in_tensor_id"), "out": e.get("out_tensor_id"),
             "axis": e.get("axis"), "noop": e.get("noop"), "at_op_id": e.get("at_op_id")}
            for e in (sem or []) if e.get("kind") == "repeat_kv"]
