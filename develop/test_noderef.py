r"""`src/noderef.py` 계약과 트레이서의 UID/version 규칙. 모델을 안 띄운다.

왜 있는가
---------
포트 사이드카는 47개 모델 209MB 이고 계보 전체가 그 위에 선다. 형식이 조용히 어긋나면
"생산자를 모름" 으로 위장돼 계보가 끊긴 것을 못 본다. 외부 검토(2026-09-10)가 전용 시험이
없다는 점과 세 가지 구멍을 짚었다: 미래 판을 v2 로 읽음, `encode(None)` 이 null 을 냄,
graph input 의 이름이 안 남음.

실행:  .venv\Scripts\python.exe develop\test_noderef.py
"""
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJ, "src"))
if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import noderef as N            # noqa: E402


def case_roundtrip():
    """v2 는 넣은 그대로 돌아온다."""
    for ref in (N.SourceRef(N.OpNode(242), 0),
                N.SourceRef(N.OpNode(7), 3),
                N.SourceRef(N.SemNode("prefill", 5), 0),
                N.SourceRef(N.ExtNode("parameter", "m.q_proj.weight"), 0),
                N.SourceRef(N.ExtNode("graph_input", "input_ids"), 0),
                N.SourceRef(N.ExtNode("unknown_external", None), 0)):
        back = N.decode_one(N.encode(ref), 2)
        assert back == ref, f"{ref} -> {back}"


def case_v1_reads():
    """v1 의 `[op, slot]` 과 `null` 을 정확히 읽는다."""
    assert N.decode_one([12, 1], 1) == N.SourceRef(N.OpNode(12), 1)
    got = N.decode_one(None, 1)
    assert got.node.external == "unknown_external", got
    # **v1 의 null 을 parameter 로 추측하면 안 된다.**
    assert got.node.name is None, got


def case_v2_rejects_v1_shapes():
    """v2 는 배열도 null 도 안 받는다. 망가진 레코드가 위장하지 못하게."""
    for bad in ([0, 1], None, "op:0:0", 3):
        try:
            N.decode_one(bad, 2)
        except N.PortsSchemaError:
            continue
        raise AssertionError(f"v2 가 {bad!r} 를 받아버렸다")


def case_unknown_schema():
    """모르는 판은 거부한다. v2 처럼 읽으면 새 필드를 조용히 흘린다."""
    for v in (0, 3, 99):
        try:
            N.decode_one({"kind": "op", "op_id": 1, "output_slot": 0}, v)
        except N.PortsSchemaError:
            continue
        raise AssertionError(f"판 {v} 를 받아버렸다")


def case_encode_none():
    """`encode(None)` 은 null 을 내지 않는다 -- v2 계약이 쓰는 쪽에서 깨진다."""
    try:
        N.encode(None)
    except N.PortsSchemaError:
        return
    raise AssertionError("encode(None) 이 통과했다")


def case_external_kinds():
    """external 은 넷 중 하나뿐이다."""
    assert set(N.EXTERNAL_KINDS) == {"parameter", "buffer", "graph_input", "unknown_external"}
    try:
        N.decode_one({"kind": "ext", "external": "weight"}, 2)
    except N.PortsSchemaError:
        return
    raise AssertionError("모르는 external 이 통과했다")


def case_op_source_projection():
    """계보 규칙이 쓰는 축약형. ATen op 이 아니면 `None` 이라 규칙이 건너뛴다."""
    assert N.op_source(N.SourceRef(N.OpNode(9), 2)) == (9, 2)
    assert N.op_source(N.SourceRef(N.ExtNode("parameter", "w"), 0)) is None
    assert N.op_source(N.SourceRef(N.SemNode("prefill", 1), 0)) is None
    assert N.op_source(None) is None


def case_v1_projection_equivalence():
    """v1 과 v2 가 **같은 계보**를 낸다 -- op 출처로 투영했을 때."""
    v1 = [[3, 0], None, [4, 1]]
    v2 = [N.encode(N.SourceRef(N.OpNode(3), 0)),
          N.encode(N.SourceRef(N.ExtNode("unknown_external", None), 0)),
          N.encode(N.SourceRef(N.OpNode(4), 1))]
    a = [N.op_source(x) for x in N.decode_list(v1, 1)]
    b = [N.op_source(x) for x in N.decode_list(v2, 2)]
    assert a == b == [(3, 0), None, (4, 1)], (a, b)


def case_empty_is_not_unknown():
    """입력이 **없는** factory op 과 "생산자를 모름" 은 다르다."""
    assert N.decode_list([], 2) == []
    assert N.decode_list(None, 2) is None


def case_semantic_guard():
    """의미 노드가 필요한 실행에서 v1 포트를 만나면 멈춘다."""
    N.require_semantic_capable(2, "테스트")
    try:
        N.require_semantic_capable(1, "테스트")
    except N.PortsSchemaError as e:
        assert "semantic_port_unavailable" in str(e), e
        return
    raise AssertionError("v1 포트로 통과했다")


CASES = [case_roundtrip, case_v1_reads, case_v2_rejects_v1_shapes, case_unknown_schema,
         case_encode_none, case_external_kinds, case_op_source_projection,
         case_v1_projection_equivalence, case_empty_is_not_unknown, case_semantic_guard]


def main():
    bad = []
    for fn in CASES:
        try:
            fn()
            print(f"  PASS      {fn.__name__:32} {(fn.__doc__ or '').splitlines()[0][:48]}")
        except AssertionError as e:
            bad.append(fn.__name__)
            print(f"  **FAIL**  {fn.__name__:32} {str(e).splitlines()[0][:60]}")
    print()
    print(f"{len(CASES) - len(bad)}/{len(CASES)} 통과" + (f"  실패: {bad}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
