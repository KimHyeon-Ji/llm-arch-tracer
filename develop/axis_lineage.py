r"""축 계보를 **기존 등가류와 독립적으로** 다시 만든다. 지금은 세기만 한다.

왜 새로 만드는가
----------------
`src/axis_classes.py` 는 간선의 근거가 **정수값**이다("피연산자 중 구체 shape 이 정확히
하나만 일치"). 그래서 값이 겹치는 두 축을 묶는다. 이것이 남은 결함 전부의 뿌리다.

그리고 기존 `develop/axis_origin.py` 로는 그 결함을 잴 수 없다 -- 기존 UF 로 클래스를 만든
뒤 그 클래스의 최초 출력을 origin 으로 고르는 **순환 구조**이기 때문이다(외부 검토 2026-09-09).
그래서 여기서는 `full/<phase>.ports.jsonl` 의 정확한 포트와 `<phase>.semantic.jsonl` 의 의미
경계만 읽어 계보를 세운다. 기존 UF 는 **대조 대상**으로만 쓴다.

세 층을 분리한다(외부 검토의 권고):

    관측  ports.jsonl 의 (생산자 op, 출력 슬롯), 텐서 id, ATen 스칼라 인자
    계보  축 사이의 same / derived / barrier 관계  <- union 은 `same` 에만
    의미  n_h / d_head / branch / config 근거      <- 이름 선택과 충돌 검증

지금 단계에서 답해야 할 것은 두 가지다. **양쪽 다 센다** -- 한쪽만 보면 결함을 다른 결함으로
바꾸고도 좋아졌다고 착각한다.

  * `wrong_union`  : 기존이 이었는데 `same` 근거가 없는 축 쌍
  * `missing_same` : `same` 이 확정인데 기존이 잇지 않은 축 쌍

실행:
    .venv\Scripts\python.exe develop\axis_lineage.py            # 전 모델 요약
    .venv\Scripts\python.exe develop\axis_lineage.py --model X  # 한 모델 상세
"""
import argparse
import collections
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJ, "src"))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import noderef
import axis_classes as ac      # noqa: E402  -- 대조용으로만 쓴다

MODELS = os.path.join(PROJ, "models")

# 축을 **그대로 통과시키는** op. 입력 축 i 와 출력 축 i 가 같은 축이라는 것이 op 의 정의다.
IDENTITY = frozenset({
    "_to_copy", "clone", "contiguous", "detach", "alias", "to", "elementwise_add",
    "elementwise_mul", "elementwise_sub", "elementwise_div", "silu", "gelu", "relu",
    "sigmoid", "tanh", "exp", "neg", "pow", "rsqrt", "sqrt", "abs", "cos", "sin",
    "masked_fill", "masked_fill_", "where", "clamp", "clamp_", "clamp_min", "ge", "gt",
    "lt", "le", "eq", "ne", "bitwise_not", "logical_not", "zeros_like", "ones_like",
    "empty_like", "full_like", "rand_like", "dropout", "add_", "mul_", "div_", "copy_",
})


# `expand` 는 여기 넣지 않는다. 크기-1 이 N 으로 늘어나므로 "비단위 축이 그대로"라는
# 정렬이 성립하지 않고, 실제로 그렇게 뒀더니 oracle 314건이 하나도 안 이어졌다(실측).
# expand 는 아래 (7) 에서 따로 다룬다: **크기가 그대로인 축은 same, 펼쳐진 축은 derived**.
VIEWY = frozenset({"view", "reshape", "_unsafe_view", "squeeze", "unsqueeze", "alias",
                   "flatten"})


def _load(model, phase):
    d = os.path.join(MODELS, model, "full")
    raw = os.path.join(d, f"{phase}.trace.raw.jsonl")
    ports = os.path.join(d, f"{phase}.ports.jsonl")
    con = os.path.join(d, f"{phase}.shapes.concrete.jsonl")
    if not (os.path.exists(raw) and os.path.exists(ports) and os.path.exists(con)):
        return None
    rows = [json.loads(l) for l in open(raw, encoding="utf-8")]
    port = {p["op_id"]: p for p in (json.loads(l) for l in open(ports, encoding="utf-8"))}
    conc = {c["op_id"]: c for c in (json.loads(l) for l in open(con, encoding="utf-8"))}
    sem = []
    sp = os.path.join(d, f"{phase}.semantic.jsonl")
    if os.path.exists(sp):
        sem = [json.loads(l) for l in open(sp, encoding="utf-8")]
    return rows, port, conc, sem


def _align_nonunit(a, b):
    """크기-1 축만 늘고 준 두 shape 의 축 대응. 못 맞추면 None.

    `[T, d]` 와 `[B, T, d]` 는 같은 텐서의 두 표기다 -- 크기-1 축을 빼면 남는 것이 같다.
    크기-1 축 자체는 잇지 않는다: 정보가 없어서 `B` 와 리터럴 `1` 이 서로 덮어쓴다
    (`_squeeze_view_keeps_names` 가 같은 이유로 크기-1 을 건너뛴다).
    """
    ia = [i for i, v in enumerate(a) if v != 1]
    ib = [i for i, v in enumerate(b) if v != 1]
    if len(ia) != len(ib) or [a[i] for i in ia] != [b[i] for i in ib]:
        return None
    return list(zip(ia, ib))


def _split_dim(cins, couts):
    """split 이 가른 축. 나머지 축은 전부 그대로다."""
    if not cins or len(couts) < 2:
        return None
    ci = cins[0]
    if not isinstance(ci, list):
        return None
    cand = [ax for ax in range(len(ci))
            if all(isinstance(co, list) and len(co) == len(ci) and co[ax] != ci[ax]
                   for co in couts)]
    if len(cand) != 1:
        return None
    # 나머지 축은 모든 조각에서 입력과 같아야 한다 -- 아니면 이 op 를 split 으로 못 읽는다.
    ax = cand[0]
    for co in couts:
        for j in range(len(ci)):
            if j != ax and co[j] != ci[j]:
                return None
    return ax


def _concat_dim(cins, couts):
    if len(couts) != 1 or len(cins) < 2:
        return None
    co = couts[0]
    if not isinstance(co, list):
        return None
    ok = [c for c in cins if isinstance(c, list) and len(c) == len(co)]
    if len(ok) != len(cins):
        return None
    cand = [ax for ax in range(len(co)) if any(c[ax] != co[ax] for c in ok)]
    if len(cand) != 1:
        return None
    ax = cand[0]
    if sum(c[ax] for c in ok) != co[ax]:
        return None
    return ax


def _perm_from_args(op, args, rank):
    """`scalar_args` 에서 순열을 그대로 읽는다. 구체 shape 으로 역산하지 않는다."""
    if not args:
        return None
    pos = args.get("pos") or []
    if op == "transpose" and len(pos) >= 2 and all(isinstance(x, int) for x in pos[:2]):
        d0, d1 = pos[0] % rank, pos[1] % rank
        perm = list(range(rank))
        perm[d0], perm[d1] = perm[d1], perm[d0]
        return perm
    if op == "permute" and pos and isinstance(pos[0], list) and len(pos[0]) == rank:
        return [d % rank for d in pos[0]]
    return None


def _same_edges(rows, port, conc, rules=("port", "identity"), sem=None):
    """op 정의가 **확정**하는 `same` 간선만 만든다. 값 일치는 근거로 쓰지 않는다.

    지금 켜는 규칙은 둘뿐이다. 나머지(split/concat/matmul/transpose)는 다음 단계에서
    하나씩 켠다 -- 한꺼번에 켜면 회귀 원인을 못 가린다.

      1. **exact port + 랭크 동일 + 크기 동일**: 소비자의 입력 슬롯 s 가 생산자 op 의 출력
         슬롯 t 에서 왔고 두 shape 이 완전히 같으면, 축 i 끼리 같은 축이다. 텐서가 그대로
         전달된 것이므로 값 매칭이 아니라 **동일성**이다.
      2. **unary identity**: 위 IDENTITY 집합의 op 는 입력 축 i 와 출력 축 i 가 같은 축이다.
    """
    edges = set()
    # `repeat_kv(n_rep=1)` 의 시간 경계. 같은 텐서라 "잇지 마라"로 표현할 수 없고,
    # **이 시점 이후의 소비자는 다른 역할**로 갈라야 한다. 전치 규칙에서만 쓴다.
    noop_bar = sorted(e.get("at_op_id") or 0 for e in (sem or [])
                      if e.get("kind") == "repeat_kv" and e.get("noop"))
    for r in rows:
        oid = r.get("op_id")
        p = port.get(oid) or {}
        c = conc.get(oid) or {}
        cins = c.get("input_shape") or []
        couts = c.get("output_shape") or []
        srcs = p.get("input_sources") or []

        # (1) 정확한 포트로 이어진, 모양이 같은 텐서
        for si, src in enumerate(srcs) if "port" in rules else ():
            if not src or si >= len(cins):
                continue
            osrc = noderef.op_source(src) if hasattr(src, "node") else src
            if osrc is None:
                continue
            pop, pslot = osrc
            pc = conc.get(pop) or {}
            pouts = pc.get("output_shape") or []
            if pslot >= len(pouts):
                continue
            a, b = pouts[pslot], cins[si]
            if not (isinstance(a, list) and isinstance(b, list)) or a != b:
                continue
            for ax in range(len(a)):
                edges.add(((pop, "o", pslot, ax), (oid, "i", si, ax)))

        # (2) 축을 그대로 두는 단항 op
        if "identity" in rules and r.get("op_type") in IDENTITY and len(couts) == 1 and cins:
            co = couts[0]
            for si, ci in enumerate(cins):
                if isinstance(ci, list) and isinstance(co, list) and ci == co:
                    for ax in range(len(ci)):
                        edges.add(((oid, "i", si, ax), (oid, "o", 0, ax)))
        op = r.get("op_type")

        # (3) 크기-1 축만 달라지는 view/reshape/squeeze/unsqueeze
        if "view" in rules and op in VIEWY and len(couts) == 1 and len(cins) >= 1:
            ci, co = cins[0], couts[0]
            if isinstance(ci, list) and isinstance(co, list):
                m = _align_nonunit(ci, co)
                if m:
                    for ia, ib in m:
                        edges.add(((oid, "i", 0, ia), (oid, "o", 0, ib)))

        # (4) split / concat -- **가르는 축만 빼고** 전부 그대로다
        if "splitcat" in rules and op in ("split_with_sizes", "split", "chunk"):
            ax = _split_dim(cins, couts)
            if ax is not None:
                for oi in range(len(couts)):
                    for j in range(len(cins[0])):
                        if j != ax:
                            edges.add(((oid, "i", 0, j), (oid, "o", oi, j)))
        if "splitcat" in rules and op in ("concat", "cat", "stack"):
            ax = _concat_dim(cins, couts)
            if ax is not None:
                for si2 in range(len(cins)):
                    for j in range(len(couts[0])):
                        if j != ax:
                            edges.add(((oid, "i", si2, j), (oid, "o", 0, j)))

        # (5) matmul -- 배치 축과 M/N 은 그대로, 수축 축은 derived 라 잇지 않는다
        if "matmul" in rules and op in ("matmul", "batched_matmul", "bmm", "mm")                 and len(cins) >= 2 and len(couts) == 1:
            a, b, co = cins[0], cins[1], couts[0]
            if all(isinstance(x, list) for x in (a, b, co)) and len(a) >= 2 and len(b) >= 2                     and len(co) == len(a) == len(b):
                nb = len(co) - 2                      # 배치 축 개수
                for j in range(nb):
                    if a[j] == co[j]:
                        edges.add(((oid, "i", 0, j), (oid, "o", 0, j)))
                    if b[j] == co[j]:
                        edges.add(((oid, "i", 1, j), (oid, "o", 0, j)))
                if a[-2] == co[-2]:
                    edges.add(((oid, "i", 0, len(a) - 2), (oid, "o", 0, len(co) - 2)))
                if b[-1] == co[-1]:
                    edges.add(((oid, "i", 1, len(b) - 1), (oid, "o", 0, len(co) - 1)))

        # (7) expand -- 크기가 그대로인 축은 `same`, 1 -> N 으로 펼쳐진 축은 `derived` 라
        #     잇지 않는다. 펼쳐진 축의 이름은 expand 가 정하는 것이 아니라 upstream origin 이
        #     정한다(외부 검토 2026-09-09). oracle 314건이 정확히 이 규칙을 기다리고 있었다.
        if "expand" in rules and op in ("expand", "broadcast_to", "expand_as")                 and len(cins) >= 1 and len(couts) == 1:
            ci, co = cins[0], couts[0]
            if isinstance(ci, list) and isinstance(co, list) and len(ci) == len(co):
                for j in range(len(ci)):
                    if ci[j] == co[j] and ci[j] != 1:
                        edges.add(((oid, "i", 0, j), (oid, "o", 0, j)))

        # (6) transpose / permute -- **맨 마지막에 켠다.** 순열은 scalar_args 에서 그대로
        #     읽는다(구체 shape 역산이 아니다). `repeat_kv(n_rep=1)` 경계를 지난 op 는
        #     제외한다 -- 그 자리는 같은 텐서인데 역할이 바뀐 곳이라, 이으면 n_kv 와 n_h 가
        #     한 클래스가 된다(2026-09-05 에 실제로 그렇게 깨졌다).
        if "transpose" in rules and op in ("transpose", "permute")                 and len(cins) == 1 and len(couts) == 1:
            ci, co = cins[0], couts[0]
            if isinstance(ci, list) and isinstance(co, list) and len(ci) == len(co):
                perm = _perm_from_args(op, r.get("scalar_args") or p.get("scalar_args"), len(ci))
                if perm and all(ci[perm[i]] == co[i] for i in range(len(co))):
                    crossed = any(b <= oid for b in noop_bar) and any(b >= oid for b in noop_bar)
                    if not crossed:
                        for i in range(len(co)):
                            edges.add(((oid, "i", 0, perm[i]), (oid, "o", 0, i)))
    return edges


def _barriers(sem):
    """의미 경계. 이 텐서 쌍의 그 축은 **이어서는 안 된다**."""
    out = []
    for e in sem:
        if e.get("kind") == "repeat_kv":
            out.append({"in": e.get("in_tensor_id"), "out": e.get("out_tensor_id"),
                        "axis": e.get("axis"), "noop": e.get("noop"),
                        "before": e.get("role_before"), "after": e.get("role_after")})
    return out


class _UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def analyse(model, rules=("port", "identity")):
    res = {"model": model, "phases": {}}
    for phase in ("prefill", "decode"):
        got = _load(model, phase)
        if not got:
            continue
        rows, port, conc, sem = got
        edges = _same_edges(rows, port, conc, rules=rules, sem=sem)
        uf = _UF()
        for a, b in edges:
            uf.union(a, b)

        # 기존 등가류(값 기반)와 대조한다.
        old = ac.build(rows, conc)
        sites = set()
        for a, b in edges:
            sites.add(a)
            sites.add(b)
        # 두 그룹핑이 같은 자리들에 대해 얼마나 다른가
        pair_old = collections.Counter()
        pair_new = collections.Counter()
        for s in sites:
            pair_old[old.find(s)] += 1
            pair_new[uf.find(s)] += 1
        # ---- 5·6단계: **양쪽을 센다** -------------------------------------
        # missing_same : `same` 이 확정인데 기존 UF 가 잇지 않은 쌍. 지금 등가류가 놓친 것.
        # unexplained  : 기존 UF 가 이었는데 지금까지 켠 규칙으로는 설명이 안 되는 병합.
        #                아직 split/concat/matmul/transpose 를 안 켰으므로 "틀렸다"가 아니라
        #                "아직 설명 못 함" 이다. 규칙을 하나씩 켤 때마다 이 수가 줄어야 한다.
        missing = sum(1 for a, b in edges if old.find(a) != old.find(b))
        by_old = collections.defaultdict(set)
        for s_ in sites:
            by_old[old.find(s_)].add(uf.find(s_))
        unexplained = sum(len(v) - 1 for v in by_old.values() if len(v) > 1)

        res["phases"][phase] = {
            "rows": len(rows),
            "ports_rows": len(port),
            "same_edges": len(edges),
            "sites_covered": len(sites),
            "new_classes": len(pair_new),
            "old_classes_over_same_sites": len(pair_old),
            "barriers": len(_barriers(sem)),
            "barriers_noop": sum(1 for b in _barriers(sem) if b.get("noop")),
            "missing_same": missing,
            "unexplained_merges": unexplained,
        }
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", help="한 모델만")
    ap.add_argument("--rules", default="port,identity",
                    help="켤 규칙: port,identity,view,splitcat,matmul,transpose "
                         "expand (하나씩 켜면서 unexplained_merges 가 줄어드는지 본다)")
    a = ap.parse_args()
    rules = tuple(x.strip() for x in a.rules.split(',') if x.strip())
    names = [a.model] if a.model else sorted(
        d for d in os.listdir(MODELS) if os.path.isdir(os.path.join(MODELS, d)))
    tot = collections.Counter()
    missing = []
    for m in names:
        r = analyse(m, rules=rules)
        if not r["phases"]:
            missing.append(m)
            continue
        for ph, v in r["phases"].items():
            for k in ("same_edges", "sites_covered", "barriers", "barriers_noop",
                      "missing_same", "unexplained_merges"):
                tot[k] += v[k]
        if a.model:
            print(json.dumps(r, ensure_ascii=False, indent=1))
    if missing:
        print(f"관측 사이드카 없음 {len(missing)}개 (재트레이스 필요): "
              f"{', '.join(missing[:4])}{' ...' if len(missing) > 4 else ''}")
    print(f"규칙: {','.join(rules)}")
    print(f"모델 {len(names) - len(missing)}개 | same 간선 {tot['same_edges']:,} | "
          f"덮은 축 자리 {tot['sites_covered']:,} | 경계 {tot['barriers']} "
          f"(그중 no-op {tot['barriers_noop']})")
    print(f"  missing_same        {tot['missing_same']:>12,}  "
          f"(same 이 확정인데 기존 등가류가 안 이은 쌍)")
    print(f"  unexplained_merges  {tot['unexplained_merges']:>12,}  "
          f"(기존이 이었는데 지금 켠 규칙으로는 설명 안 되는 병합)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
