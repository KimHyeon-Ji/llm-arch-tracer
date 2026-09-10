r"""계보 그래프의 **노드 이름표**와 그 직렬화.

왜 노드와 출력 슬롯을 나누는가
------------------------------
예전 `input_sources` 는 `[op_id, slot]` 이었다. 여기서 `slot` 은 **생산자의 출력 번호**인데,
축 자리(`AxisSite`)의 `slot` 은 **소비자의 입력 번호**다. 둘을 같은 이름의 한 필드에 담으면
역할이 겹쳐 읽는 쪽이 헷갈린다(외부 검토 2026-09-10). 그래서 노드는 노드대로 두고,
출력 번호는 `SourceRef` 가 따로 든다.

    OpNode(op_id)                 ATen op 하나
    SemNode(phase, event_id)      ATen 에 안 보이는 의미 노드 (repeat_kv 등)
    ExtNode(external, name)       그래프 밖에서 들어온 것
    SourceRef(node, output_slot)  "그 노드의 몇 번째 출력"

`ExtNode.external` 은 넷 중 하나다. `null` 하나로 뭉쳐 두면 "모른다" 와 "가중치다" 가 같아
보이고, 그 상태로는 `missing_port_records` 를 하드 게이트로 올릴 수 없다.

    parameter          named_parameters()
    buffer             named_buffers()
    graph_input        모델 호출 인자
    unknown_external   위 어디에도 안 잡힌 것. **이게 0 에 가까워야 한다.**

사이드카 형식
-------------
`ports_schema_version` 로 판을 가른다. **한 파일 안에 두 판이 섞이면 거부한다** -- 섞인 것을
조용히 읽으면 어느 쪽 규칙으로 해석했는지 알 수 없다.

    v1 (2026-09-10 이전)   [op_id, slot] | null | []
    v2                     {"kind":"op","op_id":242,"output_slot":0} | []

v2 는 **배열도 null 도 안 받는다.** 그래야 망가진 레코드가 "생산자를 모름" 으로 위장하지
못한다. v1 의 `null` 은 `unknown_external` 로만 읽는다 -- parameter 로 추측하지 않는다.

실측(2026-09-10): v2 객체 형식은 사이드카를 1.57배로 키운다(함대 209MB -> 328MB).
짧은 키나 태그 문자열로 줄일 수 있지만 읽는 사람이 있는 형식을 택했다.
"""
from typing import NamedTuple, Optional

SCHEMA_VERSION = 2

EXTERNAL_KINDS = ("parameter", "buffer", "graph_input", "unknown_external")


class PortsSchemaError(ValueError):
    """사이드카 형식이 섞였거나 망가졌다. 조용히 넘어가면 안 되는 것들."""


class OpNode(NamedTuple):
    op_id: int

    kind = "op"


class SemNode(NamedTuple):
    phase: str
    event_id: int

    kind = "sem"


class ExtNode(NamedTuple):
    external: str
    name: Optional[str] = None

    kind = "ext"


class SourceRef(NamedTuple):
    node: object
    output_slot: int = 0


def encode(ref) -> dict:
    """`SourceRef` -> v2 레코드.

    **`None` 을 받지 않는다.** v2 는 null 을 안 쓰기로 했는데 여기서 null 을 내보내면 그
    계약이 쓰는 쪽에서 깨진다. 생산자를 모르는 입력은 `ExtNode("unknown_external")` 이지
    `None` 이 아니다.
    """
    if ref is None:
        raise PortsSchemaError(
            "v2 는 null 출처를 쓰지 않는다 -- 모르는 입력은 ExtNode('unknown_external') 이다")
    n = ref.node
    if isinstance(n, OpNode):
        return {"kind": "op", "op_id": n.op_id, "output_slot": ref.output_slot}
    if isinstance(n, SemNode):
        return {"kind": "sem", "phase": n.phase, "event_id": n.event_id,
                "output_slot": ref.output_slot}
    if isinstance(n, ExtNode):
        return {"kind": "ext", "external": n.external, "name": n.name}
    raise PortsSchemaError(f"알 수 없는 노드: {n!r}")


def decode_one(rec, schema: int) -> SourceRef:
    """레코드 하나 -> `SourceRef`. 못 읽으면 예외를 던진다(빈손으로 넘어가지 않는다)."""
    if schema not in (1, SCHEMA_VERSION):
        # 미래 판을 v2 로 읽으면 새 필드를 조용히 흘린다. 모르는 판은 거부한다.
        raise PortsSchemaError(
            f"모르는 ports 판 {schema} (아는 것: 1, {SCHEMA_VERSION}) -- 도구를 갱신해라")
    if schema == 1:
        if rec is None:
            # v1 은 "가중치" 와 "모름" 을 구분하지 못했다. 추측하지 않는다.
            return SourceRef(ExtNode("unknown_external"), 0)
        if isinstance(rec, (list, tuple)) and len(rec) == 2 \
                and all(isinstance(x, int) for x in rec):
            return SourceRef(OpNode(rec[0]), rec[1])
        raise PortsSchemaError(f"v1 레코드가 아니다: {rec!r}")
    if not isinstance(rec, dict):
        raise PortsSchemaError(
            f"v2 는 객체만 받는다(배열/null 금지): {rec!r}")
    k = rec.get("kind")
    if k == "op":
        return SourceRef(OpNode(int(rec["op_id"])), int(rec.get("output_slot", 0)))
    if k == "sem":
        return SourceRef(SemNode(str(rec["phase"]), int(rec["event_id"])),
                         int(rec.get("output_slot", 0)))
    if k == "ext":
        ext = rec.get("external")
        if ext not in EXTERNAL_KINDS:
            raise PortsSchemaError(f"알 수 없는 external: {ext!r}")
        return SourceRef(ExtNode(ext, rec.get("name")), 0)
    raise PortsSchemaError(f"알 수 없는 kind: {k!r}")


def decode_list(seq, schema: int):
    """입력 슬롯별 `SourceRef` 목록. 빈 리스트는 **입력이 없는 것**이라 그대로 둔다."""
    if seq is None:
        return None
    return [decode_one(x, schema) for x in seq]


def schema_of(rec: dict) -> int:
    """레코드 하나의 판. 없으면 v1 이다."""
    v = rec.get("ports_schema_version")
    return int(v) if v is not None else 1


def require_semantic_capable(schema: int, where: str = ""):
    """의미 노드가 필요한 실행에서 v1 포트를 만나면 **멈춘다**.

    v1 사이드카에는 `SemNode` 를 적을 자리가 없다. 그대로 진행하면 의미 경계가 없는 계보를
    "경계가 없다" 로 읽어 조용히 틀린 답을 낸다(외부 검토 2026-09-10).
    """
    if schema != SCHEMA_VERSION:
        raise PortsSchemaError(
            f"semantic_port_unavailable: {where or '이 실행'} 은 v{SCHEMA_VERSION} 포트가 "
            f"필요한데 v{schema} 다 -- 재트레이스해라")


def op_source(ref) -> Optional[tuple]:
    """계보 규칙이 쓰는 축약형 `(op_id, output_slot)`. ATen op 이 아니면 `None`."""
    if ref is None or not isinstance(ref.node, OpNode):
        return None
    return (ref.node.op_id, ref.output_slot)


def has_semantic(sources) -> bool:
    return any(s is not None and isinstance(s.node, SemNode) for s in (sources or []))
