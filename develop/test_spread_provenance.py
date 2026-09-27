r"""`spread: provenance_class` 계약. 모델을 안 띄운다.

왜 있는가
---------
`spread: class` 는 `axis_classes.build` 를 mode 없이 부르므로 값 간선으로 등가류를 세운다.
폭이 같은 두 이름(MLA 의 `d_nope` / `d_v`, 둘 다 128)은 값으로 안 갈려서 클래스가 끊기고,
교정이 사슬 중간에서 멈춘다 -- Kimi-K3 에서 reshape 자체 유도와 24 건 어긋났다.
`provenance_class` 는 같은 일을 **포트 계보** 위에서 한다.

이 시험이 지켜야 하는 것 네 가지. 전부 실제 결함에서 나왔다.

  1. 포트가 없으면 **거부한다.** 포트가 하나도 없으면 build 는 조용히 단일 원소 클래스만
     내놓고, 그건 "퍼뜨릴 곳이 없다" 와 구별되지 않는다. 조용한 폴백은 근거를 위조한다.
  2. 값으로는 못 가르는 자리를 **계보로는 가른다.** legacy 가 못 닿는 자리에 닿아야 한다.
  3. 계보가 **다른** 텐서는 건드리지 않는다. 폭이 같아도 다른 포트면 다른 클래스다.
  4. 알 수 없는 `spread` 값은 거부한다. 예전엔 조용히 무시돼서 오타가 "퍼뜨리지 않는 교정"
     으로 통과했고, 발화 수는 앵커 한 자리만 세니 게이트도 넘어갔다.

실행:  .venv\Scripts\python.exe develop\test_spread_provenance.py
"""
import io
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJ, "src"))
if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import label_overrides as LO       # noqa: E402
import noderef as NR              # noqa: E402
import yaml                       # noqa: E402

MODEL = "fixture__mla-like"
W = 128                           # d_nope == d_v == W. 값으로는 절대 안 갈린다.


def _fixture(with_ports=True, encoded=False):
    r"""MLA value 사슬의 최소 모형.

        op0  split_with_sizes  [B,H,T,2W] -> [B,H,T,W](k_nope), [B,H,T,W](value)
        op1  unsqueeze         op0 출력 1 -> [B,H,T,W,1]
        op2  clone             [B,H,T,W,1] -> [B,H,T,W,1]
        op3  concat            op0 출력 0 + [B,H,1,W] -> [B,H,T+1,W]   (key 경로, 별개 텐서)

    op1/op2 는 value 이고 op3 는 k_nope 다. **구체 shape 이 전부 같다** -- 값 간선으로는
    한 덩어리로 묶이거나 아예 안 묶인다. 포트만이 둘을 가른다.
    """
    # `attach_ports` 는 사이드카를 **디코드해서** 행에 붙인다. 메모리 위의 행은 인코딩된
    # dict 가 아니라 SourceRef 를 들고 있으므로 fixture 도 그래야 한다.
    def ref(op, slot):
        return NR.SourceRef(NR.OpNode(op), slot)

    def ext(name):
        return NR.SourceRef(NR.ExtNode("graph_input", name), 0)

    # v2 에서 "생산자를 모름" 은 raw null 이 아니라 명시된 외부 노드다. `encode(None)` 이
    # null 을 내던 것이 이미 한 번 막힌 구멍이라(develop/test_noderef.case_encode_none),
    # fixture 도 실제 형식을 써야 한다.
    unknown = NR.SourceRef(NR.ExtNode("unknown_external", None), 0)

    rows = [
        {"op_id": 0, "op_type": "split_with_sizes", "module_path": "model.layers.0.self_attn",
         "input_shape": [[2, 4, 8, 2 * W]], "weight_shape": None,
         "output_shape": [[2, 4, 8, W], [2, 4, 8, W]]},
        {"op_id": 1, "op_type": "unsqueeze", "module_path": "model.layers.0.self_attn",
         "input_shape": [[2, 4, 8, W]], "weight_shape": None,
         "output_shape": [[2, 4, 8, W, 1]]},
        {"op_id": 2, "op_type": "clone", "module_path": "model.layers.0.self_attn",
         "input_shape": [[2, 4, 8, W, 1]], "weight_shape": None,
         "output_shape": [[2, 4, 8, W, 1]]},
        {"op_id": 3, "op_type": "concat", "module_path": "model.layers.0.self_attn",
         "input_shape": [[2, 4, 8, W], [2, 4, 1, W]], "weight_shape": None,
         "output_shape": [[2, 4, 9, W]]},
    ]
    if with_ports:
        rows[0]["input_sources"] = [unknown, ext("kv")]
        rows[1]["input_sources"] = [ref(0, 1)]      # value = split 출력 1
        rows[2]["input_sources"] = [ref(1, 0)]
        rows[3]["input_sources"] = [ref(0, 0), ext("cache")]  # k_nope = split 출력 0
        for r in rows:
            r["input_tensor_ids"] = []
            r["output_tensor_ids"] = []
            r["ports_schema_version"] = NR.SCHEMA_VERSION
            if encoded:
                # 트레이스 시점의 표현. `build_table.write_ports` 가 이대로 json 으로 쓴다.
                r["input_sources"] = [NR.encode(x) for x in r["input_sources"]]
    ordered = [
        {"op_id": 0, "op_type": "split_with_sizes", "module_path": "model.layers.0.self_attn",
         "input_shape": [["B", "H", "T", "d_nope+d_v"]], "weight_shape": None,
         "output_shape": [["B", "H", "T", "d_nope"], ["B", "H", "T", "d_v"]]},
        {"op_id": 1, "op_type": "unsqueeze", "module_path": "model.layers.0.self_attn",
         "input_shape": [["B", "H", "T", "d_nope"]], "weight_shape": None,
         "output_shape": [["B", "H", "T", "d_nope", "1"]]},
        {"op_id": 2, "op_type": "clone", "module_path": "model.layers.0.self_attn",
         "input_shape": [["B", "H", "T", "d_nope", "1"]], "weight_shape": None,
         "output_shape": [["B", "H", "T", "d_nope", "1"]]},
        {"op_id": 3, "op_type": "concat", "module_path": "model.layers.0.self_attn",
         "input_shape": [["B", "H", "T", "d_nope"], ["B", "H", "1", "d_nope"]],
         "weight_shape": None, "output_shape": [["B", "H", "T+1", "d_nope"]]},
    ]
    return rows, ordered


def _rules(spread, **extra):
    spec = {"overrides": [dict({
        "model": MODEL, "module": "self_attn$", "shape": ["B", "H", "T", "d_nope"],
        "axis": 3, "field": "i", "shape_index": 0, "op_type": "unsqueeze",
        "from": "d_nope", "to": "d_v", "expect": W,
        "source": "fixture.py:1 -- 시험용"}, **extra)]}
    if spread is not None:
        spec["overrides"][0]["spread"] = spread
    p = os.path.join(tempfile.gettempdir(), f"test_spread_{spread}_{len(extra)}.yaml")
    with io.open(p, "w", encoding="utf-8", newline=chr(10)) as f:
        yaml.safe_dump(spec, f, allow_unicode=True, sort_keys=False)
    return p


def _run(spread, with_ports=True, encoded=False, **extra):
    rows, ordered = _fixture(with_ports, encoded)
    LO._CACHE = None
    rep = LO.apply(rows, ordered, MODEL, path=_rules(spread, **extra))
    return rep, ordered


def _labels(ordered):
    """(op_id, i|o, shape_index, axis) -> 라벨, 폭 W 인 자리만."""
    out = {}
    for r in ordered:
        for key, tag in (("input_shape", "i"), ("output_shape", "o")):
            for si, sh in enumerate(r.get(key) or []):
                for ax, v in enumerate(sh):
                    out[(r["op_id"], tag, si, ax)] = str(v)
    return out


def case_refuses_without_ports():
    """포트가 없으면 조용히 값 간선으로 돌지 않고 거부한다."""
    try:
        _run("provenance_class", with_ports=False)
    except ValueError as e:
        assert "provenance_class" in str(e), str(e)
        assert "input_sources" in str(e), str(e)
        return
    raise AssertionError("포트 없이 통과했다 -- 조용한 폴백")


def case_accepts_trace_time_encoding():
    """트레이스 시점의 행은 `input_sources` 를 **인코딩된 dict** 로 들고 온다.

    `attach_ports` 경로는 이미 디코드된 `SourceRef` 를 붙이므로 두 표현이 다르다.
    맞춰 주지 않으면 `'dict' object has no attribute 'node'` 로 죽는다 -- 2026-09-27 에
    Kimi-K3 재트레이스가 13 분 뒤 정확히 이 자리에서 멈췄다. 첫 슬롯이 None 인 것도
    첫 슬롯이 op 가 아닌 외부 노드인 것도 같이 시험한다.
    """
    _, dec = _run("provenance_class", encoded=False)
    _, enc = _run("provenance_class", encoded=True)
    assert _labels(enc) == _labels(dec), "인코딩된 입력이 다른 결과를 냈다"
    assert _labels(enc)[(2, "o", 0, 3)] == "d_v", "인코딩 경로가 사슬을 못 닫았다"


def case_reaches_where_legacy_cannot():
    """계보는 legacy 가 못 닿는 사슬 끝까지 닿는다."""
    _, leg = _run("class")
    _, prov = _run("provenance_class")
    lab_l, lab_p = _labels(leg), _labels(prov)
    tail = (2, "o", 0, 3)                    # clone 출력 -- 사슬의 끝
    assert lab_p[tail] == "d_v", f"provenance 가 사슬 끝을 못 닿았다: {lab_p[tail]}"
    assert lab_l[tail] != "d_v", (
        "legacy 가 이미 사슬 끝에 닿는다 -- 이 fixture 는 두 모드를 가르지 못한다 "
        f"({lab_l[tail]})")


def case_leaves_other_lineage_alone():
    """포트가 다른 텐서(k_nope)는 폭이 같아도 건드리지 않는다."""
    _, prov = _run("provenance_class")
    lab = _labels(prov)
    for slot in ((0, "o", 0, 3), (3, "i", 0, 3), (3, "i", 1, 3), (3, "o", 0, 3)):
        assert lab[slot] == "d_nope", f"k_nope 자리 {slot} 가 {lab[slot]} 로 덮였다"


def case_anchor_class_is_the_value_lineage():
    """value 쪽 자리는 전부 바뀐다 -- 하나라도 남으면 사슬이 열려 있다."""
    _, prov = _run("provenance_class")
    lab = _labels(prov)
    for slot in ((1, "i", 0, 3), (1, "o", 0, 3), (2, "i", 0, 3), (2, "o", 0, 3)):
        assert lab[slot] == "d_v", f"value 자리 {slot} 가 {lab[slot]} 로 남았다"


def case_applied_count_is_real():
    """발화 수는 실제로 쓴 자리 수다. 앵커만 세지 않는다."""
    rep, _ = _run("provenance_class")
    assert len(rep) == 1, rep
    # 앵커(op1 i) + op1 o + op2 i + op2 o = 4
    assert rep[0]["applied"] == 4, rep[0]["applied"]


def case_rejects_unknown_spread():
    """오타 난 spread 값은 조용히 무시하지 않고 거부한다."""
    try:
        _run("provenance")          # 정답은 provenance_class
    except ValueError as e:
        assert "unknown spread" in str(e), str(e)
        return
    raise AssertionError("알 수 없는 spread 로 통과했다")


def case_legacy_unchanged():
    """`spread: class` 의 동작은 그대로다 -- 함대 전체가 이걸 쓴다."""
    rep, leg = _run("class")
    lab = _labels(leg)
    assert lab[(1, "i", 0, 3)] == "d_v", "앵커 자리가 안 바뀌었다"
    for slot in ((0, "o", 0, 3), (3, "o", 0, 3)):
        assert lab[slot] == "d_nope", f"legacy 가 {slot} 까지 넘어갔다: {lab[slot]}"
    assert rep[0]["applied"] >= 1, rep[0]["applied"]


CASES = [case_refuses_without_ports, case_accepts_trace_time_encoding,
         case_reaches_where_legacy_cannot,
         case_leaves_other_lineage_alone, case_anchor_class_is_the_value_lineage,
         case_applied_count_is_real, case_rejects_unknown_spread,
         case_legacy_unchanged]


def main():
    bad = []
    for fn in CASES:
        try:
            fn()
            print(f"  PASS      {fn.__name__:38} {(fn.__doc__ or '').splitlines()[0][:44]}")
        except AssertionError as e:
            bad.append(fn.__name__)
            print(f"  **FAIL**  {fn.__name__:38} {str(e).splitlines()[0][:56]}")
    print()
    print(f"{len(CASES) - len(bad)}/{len(CASES)} 통과" + (f"  실패: {bad}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
