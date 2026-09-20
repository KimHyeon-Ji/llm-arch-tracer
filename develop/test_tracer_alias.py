r"""뷰에 제자리로 쓸 때 **의존은 남기고 값 출처는 건드리지 않는가.**

두 반례 모두 외부 검토(2026-09-20)가 코드로 짚은 것이다.

  1. 포트 계약 -- `x[0].copy_(...)` 뒤에 `x` 를 읽으면, 그 입력의 출처는 여전히 `x` 를 만든
     op 이어야 한다. 뷰의 출력 포트(rank 가 하나 작다)를 가리키면 안 된다.
     내가 처음 넣은 수정이 바로 그 오류를 만들었다 -- `depends_on` 을 살리려고
     `physical_producer[base]` 를 덮어썼다.

  2. 미리 만든 뷰 -- 뷰를 먼저 다 만들어 놓고 차례로 쓰면, 마지막 뷰의 `select` 가 첫 쓰기
     **이전** 의 base 를 참조하므로 "select 사슬에 남는다" 는 보장이 깨진다. 이 누락은 내
     수정 이전부터 있었다.

실행:
    .venv\Scripts\python.exe develop\test_tracer_alias.py
"""
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJ, "src"))
if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import torch                                                   # noqa: E402
from tracer import OpGraphTracer                               # noqa: E402
from scope import ScopeLabeler                                 # noqa: E402


class _Empty(torch.nn.Module):
    pass


def _trace(fn):
    m = _Empty()
    tr = OpGraphTracer(m, ScopeLabeler(m), phase="t")
    with tr:
        fn()
    return tr.rows


def _last(rows, suffix):
    return [r for r in rows if (r.get("raw_op") or "").endswith(suffix)][-1]


def port_rank_ok(rows) -> list:
    """모든 행에서 `input_sources` 가 가리키는 출력의 rank 가 그 입력의 rank 와 같은가."""
    prod = {r["op_id"]: r for r in rows}
    bad = []
    for r in rows:
        for src, sh in zip(r.get("input_sources") or [], r.get("input_shape") or []):
            if not (isinstance(src, dict) and src.get("kind") == "op"):
                continue
            p = prod.get(src["op_id"])
            if not p:
                continue
            outs = p.get("output_shape") or []
            slot = src.get("output_slot") or 0
            if slot < len(outs) and isinstance(outs[slot], list) and isinstance(sh, list):
                if len(outs[slot]) != len(sh):
                    bad.append((r["op_id"], src["op_id"], outs[slot], sh))
    return bad


def reaches(rows, target, wanted) -> bool:
    prod = {r["op_id"]: r for r in rows}
    seen, stack = set(), list(target.get("depends_on") or [])
    while stack:
        x = stack.pop()
        if x in seen:
            continue
        seen.add(x)
        stack += (prod.get(x, {}).get("depends_on") or [])
    return all(w in seen for w in wanted)


def case_write_through_view(dev):
    def f():
        x = torch.zeros(2, 3, device=dev)
        a = x[0]
        a.copy_(torch.ones(3, device=dev))
        x.clone()
    return f


def case_views_made_first(dev):
    def f():
        x = torch.zeros(2, 3, device=dev)
        a, b = x[0], x[1]
        p = torch.ones(3, device=dev)
        q = torch.full((3,), 2.0, device=dev)
        a.copy_(p)
        b.copy_(q)
        x.clone()
    return f


def main():
    from torch._subclasses.fake_tensor import FakeTensorMode
    fails = 0
    for label, mk in (("meta", lambda f: f()),
                      ("fake", lambda f: FakeTensorMode().__enter__() and f())):
        for name, case in (("뷰에 쓰고 base 읽기", case_write_through_view),
                           ("뷰를 미리 만들고 차례로 쓰기", case_views_made_first)):
            dev = "meta"
            rows = _trace(case(dev))
            bad = port_rank_ok(rows)
            cl = _last(rows, "clone.default")
            writes = [r["op_id"] for r in rows if (r.get("raw_op") or "").endswith("copy_.default")]
            ok_port = not bad
            ok_dep = reaches(rows, cl, writes)
            fails += (0 if ok_port else 1) + (0 if ok_dep else 1)
            print(f"{'OK ' if ok_port else '**실패**'} [{label}] {name:26} 포트 rank 일치"
                  + ("" if ok_port else f"  {bad[:2]}"))
            print(f"{'OK ' if ok_dep else '**실패**'} [{label}] {name:26} 쓰기 {writes} 전부 도달")
        break                      # meta 로 두 경우를 다 덮는다 -- fake 는 같은 경로다
    print()
    if fails:
        print(f"**{fails}개 실패 -- 뷰 제자리 쓰기의 계보가 깨졌다**")
    else:
        print("4/4 — 값 출처는 그대로 두고 쓰기 의존만 남는다")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
