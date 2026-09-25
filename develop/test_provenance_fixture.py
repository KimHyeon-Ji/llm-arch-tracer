r"""provenance 경로의 **행동 검사.** 배선됐다는 문자열 확인만으로는 부족하다.

`test_bundle_guards.py` 의 provenance 항목은 `inspect.getsource()` 로 호출이 있는지 보는
것이어서, 실제로 포트가 채워진 입력을 통과시키는 증명이 아니다(외부 검토 2026-09-25).
여기서는 작은 synthetic raw/ports/semantic fixture 를 만들어
`attach_ports -> build(mode="provenance") -> root 연결` 을 실제로 확인한다.

발행 모델의 `ports.jsonl` 은 전부 0 바이트이므로 fixture 없이는 이 경로를 한 번도 실행해
볼 수 없다. **그래서 fixture 가 필요하다.**

실행:
    .venv\Scripts\python.exe develop\test_provenance_fixture.py
"""
import io
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(PROJ, "src"))
if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import axis_classes as AC                                        # noqa: E402
import noderef                                                   # noqa: E402

OK, FAIL = [], []
NL = chr(10)


def check(name, cond):
    (OK if cond else FAIL).append(name)
    print(("  OK   " if cond else "  FAIL ") + name)


def write(p, records):
    with io.open(p, "w", encoding="utf-8", newline=NL) as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + NL)


def fixture(tmp, *, dup_port=False, partial=False, barrier=False):
    """세 op 사슬. `op1 -> op2 -> op3`, 모두 shape `[4, 8]`.

    op3 는 `view` 로 `[4, 8] -> [32]` 를 해서 포트 간선과 view 간선이 섞인 경우도 본다.
    """
    md = os.path.join(tmp, "FIXTURE")
    full = os.path.join(md, "full")
    os.makedirs(full, exist_ok=True)

    def op(oid):
        return noderef.encode(noderef.SourceRef(noderef.OpNode(oid), 0))

    ext = noderef.encode(noderef.SourceRef(
        noderef.ExtNode("graph_input", "hidden_states"), 0))

    rows = [{"op_id": 1, "op_type": "linear"},
            {"op_id": 2, "op_type": "elementwise_add"},
            {"op_id": 3, "op_type": "view"}]
    # `ports_schema_version` 이 없으면 **v1 로 읽힌다**(`noderef.schema_of`). v2 레코드를
    # 쓰면서 판 표시를 빼면 `PortsSchemaError` 가 난다 -- fixture 를 만들다 실제로 겪었다.
    V = noderef.SCHEMA_VERSION
    ports = [{"op_id": 1, "ports_schema_version": V, "input_sources": [ext]},
             {"op_id": 2, "ports_schema_version": V, "input_sources": [op(1)]},
             {"op_id": 3, "ports_schema_version": V, "input_sources": [op(2)]}]
    if dup_port:
        ports.append({"op_id": 2, "ports_schema_version": V,     # 같은 op_id 두 번
                      "input_sources": [ext]})
    if partial:
        ports = ports[:2]                                        # op3 의 기록이 없다
    write(os.path.join(full, "prefill.ports.jsonl"), ports)

    sem = []
    if barrier:
        sem.append({"kind": "repeat_kv", "noop": True, "at_op_id": 2})
    write(os.path.join(full, "prefill.semantic.jsonl"), sem)
    # raw 는 coverage 계산용으로 행 수만 맞춘다
    write(os.path.join(full, "prefill.trace.raw.jsonl"), rows)

    conc = {1: {"input_shape": [[4, 8]], "output_shape": [[4, 8]]},
            2: {"input_shape": [[4, 8]], "output_shape": [[4, 8]]},
            3: {"input_shape": [[4, 8]], "output_shape": [[32]]}}
    write(os.path.join(full, "prefill.shapes.concrete.jsonl"),
          [{"op_id": k, **v} for k, v in conc.items()])
    return md, [dict(r) for r in rows], conc


tmp = tempfile.mkdtemp()
try:
    import build_review_bundle as B                              # noqa: E402

    print("1) attach_ports 없이는 포트 간선이 **하나도** 생기지 않는다")
    md, rows, conc = fixture(tmp)
    check("행에 input_sources 가 없다", AC.missing_port_records(rows) == 3)
    uf0 = AC.build(rows, conc, mode="provenance")
    check("**op1.o 와 op2.i 가 이어지지 않는다** (조용한 퇴화)",
          uf0.find((1, "o", 0, 0)) != uf0.find((2, "i", 0, 0)))
    check("build 가 예외를 내지 않는다 -- 그래서 조용하다", True)

    print("2) attach_ports 를 부르면 실제로 이어진다")
    n = AC.attach_ports(md, "prefill", rows)
    check("세 행 모두에 붙었다", n == 3)
    check("빠진 기록 0", AC.missing_port_records(rows) == 0)
    uf = AC.build(rows, conc, noop_barriers=AC.noop_barriers_of(md, "prefill"),
                  mode="provenance")
    check("**op1.o[0] == op2.i[0]**",
          uf.find((1, "o", 0, 0)) == uf.find((2, "i", 0, 0)))
    check("**op1.o[1] == op2.i[1]**",
          uf.find((1, "o", 0, 1)) == uf.find((2, "i", 0, 1)))
    check("op2 는 elementwise 라 입력·출력도 이어진다",
          uf.find((2, "i", 0, 0)) == uf.find((2, "o", 0, 0)))
    check("사슬이 op1.o -> op3.i 까지 닿는다",
          uf.find((1, "o", 0, 0)) == uf.find((3, "i", 0, 0)))
    check("축 0 과 축 1 은 **섞이지 않는다**",
          uf.find((1, "o", 0, 0)) != uf.find((1, "o", 0, 1)))
    check("외부 입력은 간선을 내지 않는다 (op1.i 는 따로)",
          uf.find((1, "i", 0, 0)) != uf.find((1, "o", 0, 0)))

    print("3) 부분 provenance 는 거부된다")
    md2, rows2, conc2 = fixture(os.path.join(tmp, "b"), partial=True)
    AC.attach_ports(md2, "prefill", rows2)
    check("한 행이 비었다", AC.missing_port_records(rows2) == 1)
    try:
        AC.build(rows2, conc2, mode="provenance")
        check("**부분 provenance 를 거부한다**", False)
    except Exception as e:
        check("**부분 provenance 를 거부한다**",
              type(e).__name__ == "PartialPortTrace")

    print("4) port_coverage 가 채워진 사이드카를 제대로 읽는다")
    _M = B.MODELS
    try:
        B.MODELS = tmp
        cv = B.port_coverage("FIXTURE", "prefill")
        check("coverage 1.0", cv["coverage"] == 1.0)
        check("고유 op_id 3", cv["ports_unique_op_ids"] == 3)
        check("중복 0", cv["ports_duplicate_op_ids"] == 0)

        md3, _, _ = fixture(os.path.join(tmp, "c"), dup_port=True)
        B.MODELS = os.path.join(tmp, "c")
        cd = B.port_coverage("FIXTURE", "prefill")
        check("**중복 op_id 를 센다**", cd["ports_duplicate_op_ids"] == 1)
        check("행 수만 보면 coverage 가 1 을 넘어 보인다", cd["coverage"] > 1.0)
    finally:
        B.MODELS = _M

    print("5) semantic barrier 가 실제로 전달되고 효과가 있다")
    md4, rows4, conc4 = fixture(os.path.join(tmp, "d"), barrier=True)
    AC.attach_ports(md4, "prefill", rows4)
    bars = AC.noop_barriers_of(md4, "prefill")
    check("barrier 를 읽는다", list(bars) == [2])
    check("barrier 없는 fixture 는 빈 목록",
          list(AC.noop_barriers_of(md, "prefill")) == [])
    ufb = AC.build(rows4, conc4, noop_barriers=bars, mode="provenance")
    check("barrier 를 넘겨도 build 가 돈다",
          ufb.find((1, "o", 0, 0)) == ufb.find((2, "i", 0, 0)))

    print("6) ports 판이 섞이면 거부된다")
    md5, rows5, _ = fixture(os.path.join(tmp, "e"))
    p = os.path.join(md5, "full", "prefill.ports.jsonl")
    recs = [json.loads(l) for l in io.open(p, encoding="utf-8")]
    recs[0].pop("ports_schema_version")                           # 이 줄만 v1
    write(p, recs)
    try:
        AC.attach_ports(md5, "prefill", rows5)
        check("**섞인 판을 거부한다**", False)
    except noderef.PortsSchemaError:
        check("**섞인 판을 거부한다**", True)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print(NL + f"{len(OK)}/{len(OK) + len(FAIL)} 통과 — "
      "포트가 붙어야만 계보가 생긴다")
if FAIL:
    print("실패: " + ", ".join(FAIL))
sys.exit(1 if FAIL else 0)
