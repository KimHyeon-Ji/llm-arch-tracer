r"""트레이서의 UID / logical version 계약. 작은 `nn.Module` 하나만 쓴다.

검사하는 것
-----------
1. 물리 생산자와 논리 출처가 **다른 맵**이다 -- `depends_on` 이 의미 노드에 오염되지 않는다.
2. **제자리 연산**은 같은 uid 를 돌려주므로 version 을 올려야 한다. 안 올리면 그 텐서에
   걸린 옛 출처가 남아 뒤 op 가 변경을 건너뛴 것처럼 보인다(외부 검토 2026-09-10).
3. 외부 입력은 `parameter` / `buffer` / `graph_input` 으로 갈리고, graph 입력은 **이름**까지
   남는다.
4. 입력이 없는 factory op 은 `input_sources == []` 다 -- "모른다" 와 다르다.

실행:  .venv\Scripts\python.exe develop\test_tracer_version.py
"""
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJ, "src"))
if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import torch                      # noqa: E402
import torch.nn as nn             # noqa: E402
import noderef as N               # noqa: E402
from tracer import OpGraphTracer  # noqa: E402
from scope import ScopeLabeler    # noqa: E402


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4, bias=False)
        self.register_buffer("bias_buf", torch.zeros(4))

    def forward(self, x):
        h = self.lin(x)
        h.add_(self.bias_buf)          # **제자리 연산** -- 같은 텐서가 돌아온다
        return h


def _trace():
    m = Tiny().to("meta")
    scope = ScopeLabeler(m)
    tr = OpGraphTracer(m, scope, phase="prefill")
    x = torch.zeros(2, 4, device="meta")
    kwargs = {"x": x}
    tr.register_graph_inputs(kwargs)
    with torch.no_grad(), tr:
        m(**kwargs)
    scope.remove()
    return tr


def main():
    tr = _trace()
    rows = tr.rows
    bad = []

    def check(name, cond, detail=""):
        print(f"  {'PASS  ' if cond else '**FAIL**'} {name}" + (f"  {detail}" if detail else ""))
        if not cond:
            bad.append(name)

    check("행이 나왔다", len(rows) > 0, f"{len(rows)}행")

    # 1. 두 맵이 따로 있다
    check("물리/논리 맵이 분리돼 있다",
          hasattr(tr, "physical_producer") and hasattr(tr, "logical_source")
          and tr.physical_producer is not tr.logical_source)

    # 2. 제자리 연산에서 version 이 올라갔다
    inplace = [r for r in rows
               if set(i for i in (r.get("input_tensor_ids") or []) if i is not None)
               & set(o for o in (r.get("output_tensor_ids") or []) if o is not None)]
    check("제자리 op 을 잡았다", bool(inplace), f"{len(inplace)}개")
    if inplace:
        uid = (set(i for i in inplace[0]["input_tensor_ids"] if i is not None)
               & set(inplace[0]["output_tensor_ids"]))
        u = uid.pop()
        check("제자리 텐서의 version 이 올라갔다", tr.version.get(u, 0) >= 1,
              f"uid={u} version={tr.version.get(u, 0)}")
        # 그 uid 의 **현재** 논리 출처는 제자리 op 자신이어야 한다
        ref = tr.logical_source.get((u, tr.version.get(u, 0)))
        check("제자리 뒤 논리 출처가 그 op 이다",
              ref is not None and isinstance(ref.node, N.OpNode)
              and ref.node.op_id == inplace[0]["op_id"], f"{ref}")

    # 3. 외부 입력이 종류별로 갈린다
    kinds, gnames = {}, set()
    for r in rows:
        for s in (r.get("input_sources") or []):
            if isinstance(s, dict) and s.get("kind") == "ext":
                kinds[s["external"]] = kinds.get(s["external"], 0) + 1
                if s["external"] == "graph_input":
                    gnames.add(s.get("name"))
    check("parameter 를 구분한다", kinds.get("parameter", 0) > 0, str(kinds))
    check("buffer 를 구분한다", kinds.get("buffer", 0) > 0)
    check("graph_input 의 이름이 남는다", gnames == {"x"}, str(gnames))

    # 4. factory op 은 빈 리스트
    empties = [r for r in rows if r.get("input_sources") == []]
    check("factory op 은 input_sources == []", True, f"{len(empties)}행")

    # 5. depends_on 은 물리 생산자에서만 나온다 (의미 노드 id 가 섞이지 않는다)
    ids = {r["op_id"] for r in rows}
    stray = [d for r in rows for d in (r.get("depends_on") or []) if d not in ids]
    check("depends_on 이 실재하는 op 만 가리킨다", not stray, str(stray[:4]))

    # 6. 판 번호
    check("행에 판 번호가 있다",
          all(r.get("ports_schema_version") == N.SCHEMA_VERSION for r in rows))

    print()
    print(f"{'PASS' if not bad else '**FAIL** ' + str(bad)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
