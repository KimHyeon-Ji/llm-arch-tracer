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


def _same_edges(rows, port, conc):
    """op 정의가 **확정**하는 `same` 간선만 만든다. 값 일치는 근거로 쓰지 않는다.

    지금 켜는 규칙은 둘뿐이다. 나머지(split/concat/matmul/transpose)는 다음 단계에서
    하나씩 켠다 -- 한꺼번에 켜면 회귀 원인을 못 가린다.

      1. **exact port + 랭크 동일 + 크기 동일**: 소비자의 입력 슬롯 s 가 생산자 op 의 출력
         슬롯 t 에서 왔고 두 shape 이 완전히 같으면, 축 i 끼리 같은 축이다. 텐서가 그대로
         전달된 것이므로 값 매칭이 아니라 **동일성**이다.
      2. **unary identity**: 위 IDENTITY 집합의 op 는 입력 축 i 와 출력 축 i 가 같은 축이다.
    """
    edges = set()
    for r in rows:
        oid = r.get("op_id")
        p = port.get(oid) or {}
        c = conc.get(oid) or {}
        cins = c.get("input_shape") or []
        couts = c.get("output_shape") or []
        srcs = p.get("input_sources") or []

        # (1) 정확한 포트로 이어진, 모양이 같은 텐서
        for si, src in enumerate(srcs):
            if not src or si >= len(cins):
                continue
            pop, pslot = src
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
        if r.get("op_type") in IDENTITY and len(couts) == 1 and cins:
            co = couts[0]
            for si, ci in enumerate(cins):
                if isinstance(ci, list) and isinstance(co, list) and ci == co:
                    for ax in range(len(ci)):
                        edges.add(((oid, "i", si, ax), (oid, "o", 0, ax)))
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


def analyse(model):
    res = {"model": model, "phases": {}}
    for phase in ("prefill", "decode"):
        got = _load(model, phase)
        if not got:
            continue
        rows, port, conc, sem = got
        edges = _same_edges(rows, port, conc)
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
    a = ap.parse_args()
    names = [a.model] if a.model else sorted(
        d for d in os.listdir(MODELS) if os.path.isdir(os.path.join(MODELS, d)))
    tot = collections.Counter()
    missing = []
    for m in names:
        r = analyse(m)
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
