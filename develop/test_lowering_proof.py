r"""`lowering_proof` 가 **거절할 수 있는지** 증명한다.

통과만 보여주는 검사는 검사가 아니다. 이전 분류기는 Codex 의 `cat([a,a])` 반례를
승인했다. 그래서 여기서는 합성 trace 를 직접 만들어, 진짜 lowering 차이는 통과하고
아래 위조는 전부 떨어지는지 본다.

    1. 전치된 bmm (같은 contraction 의 다른 lowering)          -> 통과해야 한다
    2. 같은 조각을 두 번 쓰기 (`cat([a, a])`)                   -> 떨어져야 한다
    3. 경계 입력을 다른 텐서로 바꾸기 (shape 은 같다)            -> 떨어져야 한다
    4. 배치를 섞기 (b=0 결과가 다른 배치에 의존)                 -> 떨어져야 한다
    5. 수축 축을 바꾸기 (전치를 잘못 되돌리기)                   -> 떨어져야 한다
    6. **첫 배치만 맞게 하기** (b=0 은 옳고 b>=1 은 0)          -> 떨어져야 한다
       외부 검토(2026-09-19)가 반례로 짚었다. `b=0` 만 대조하면 통과한다 --
       `compare_components` 가 모든 배치 조각을 도는 이유다.

실행:
    .venv\Scripts\python.exe develop\test_lowering_proof.py
"""
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJ, "src"))
sys.path.insert(0, HERE)
if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import lowering_proof as LP                       # noqa: E402

B, H, C, D = 3, 4, 5, 6                           # batch, head, chunk, width


class Trace:
    """합성 trace 한 벌. `ports` 와 `raw` 를 같이 만든다.

    경계 텐서에도 **생산자 op** 를 달아야 앵커가 선다. 실제 trace 에서는 경계 입력이
    늘 다른 op 의 출력이거나 파라미터이기 때문이다."""

    def __init__(self):
        self.ports, self.raw, self.nid, self.oid = {}, {}, 100, 0

    def tensor(self):
        self.nid += 1
        return self.nid

    def add(self, raw_op, in_ids, pos, out_shapes, in_shapes, kw=None):
        oid = self.oid
        self.oid += 1
        outs = [self.tensor() for _ in out_shapes]
        self.ports[oid] = {"srcs": [None] * len(in_ids), "in_ids": list(in_ids),
                           "out_ids": outs, "pos": list(pos), "kw": dict(kw or {}),
                           "schema": 2}
        self.raw[oid] = {"op_id": oid, "raw_op": raw_op, "module_path": "m.self_attn",
                         "input_shape": [list(s) for s in in_shapes],
                         "output_shape": [list(s) for s in out_shapes]}
        return outs

    def graph(self):
        return LP.Graph(self.ports)


def attach_consumer(to, tn, co, cn, amap):
    """성분 출력에 **외부 소비자**를 단다. 실제 trace 에서는 늘 누군가 쓰고,
    검증기는 그 소비자를 앵커로 쓴다(소비자 없는 출력은 승인하지 않는다)."""
    go, gn = to.graph(), tn.graph()
    # 아직 소비자가 없으므로 `boundary` 는 이 출력을 경계로 내지 않는다(관측 불가능한 값은
    # 경계가 아니다). 마지막 op 의 출력을 직접 집는다.
    lo_t, ln_t = to.ports[max(co)]["out_ids"][0], tn.ports[max(cn)]["out_ids"][0]
    lo = LP._out_label(lo_t, co, go, to.raw)
    ln = LP._out_label(ln_t, cn, gn, tn.raw)
    to.add("aten.mul.Tensor", [lo_t], [1.0], [lo], [lo])
    tn.add("aten.mul.Tensor", [ln_t], [1.0], [ln], [ln])
    amap[to.oid - 1] = tn.oid - 1


def _cmp(to, tn, co, cn, ns_old, ns_newB, amap):
    return LP.compare_components(co, cn, to.graph(), tn.graph(), to.raw, tn.raw,
                                 ns_old, ns_newB, B, amap)


def build(kind):
    """옛(B=1)/새(B=3) 한 쌍. `kind` 가 위조 종류다.

    경계 입력은 **성분 밖의 생산자 op** 가 만든다. 그 생산자 쌍이 `amap` 으로 이어져야
    앵커가 서고, 앵커가 서야 경계 대응이 만들어진다.
    """
    ns_o = {"B": 1, "H": H, "C": C, "D": D}
    ns_n = {"B": B, "H": H, "C": C, "D": D}

    to = Trace()
    x_o, = to.add("aten.zeros.default", [], [[H, 1, D]], [["H", "B", "D"]], [])   # op 0
    w_o, = to.add("aten.zeros.default", [], [[H, D, C]], [["H", "D", "C"]], [])   # op 1
    to.add("aten.bmm.default", [x_o, w_o], [],
           [["H", "B", "C"]], [["H", "B", "D"], ["H", "D", "C"]])                 # op 2
    co = [2]

    tn = Trace()
    x_n, = tn.add("aten.zeros.default", [], [[B * H, B, D]], [["B*H", "B", "D"]], [])
    w_n, = tn.add("aten.zeros.default", [], [[B * H, D, C]], [["B*H", "D", "C"]], [])
    amap = {0: 0, 1: 1}

    if kind == "genuine":
        # 같은 contraction 이 피연산자를 바꾼 전치 bmm 으로 내려간다
        xt, = tn.add("aten.permute.default", [x_n], [[0, 2, 1]],
                     [["B*H", "D", "B"]], [["B*H", "B", "D"]])
        wt, = tn.add("aten.permute.default", [w_n], [[0, 2, 1]],
                     [["B*H", "C", "D"]], [["B*H", "D", "C"]])
        z, = tn.add("aten.bmm.default", [wt, xt], [],
                    [["B*H", "C", "B"]], [["B*H", "C", "D"], ["B*H", "D", "B"]])
        tn.add("aten.permute.default", [z], [[0, 2, 1]],
               [["B*H", "B", "C"]], [["B*H", "C", "B"]])
        cn = [2, 3, 4, 5]
    elif kind == "dup_slice":
        # 같은 조각을 두 번 쓴다 -- shape 은 맞지만 값이 다르다
        half, = tn.add("aten.slice.Tensor", [x_n], [2, 0, D // 2, 1],
                       [[B * H, B, D // 2]], [["B*H", "B", "D"]])
        x2, = tn.add("aten.cat.default", [half, half], [2],
                     [["B*H", "B", "D"]], [[B * H, B, D // 2], [B * H, B, D // 2]])
        tn.add("aten.bmm.default", [x2, w_n], [],
               [["B*H", "B", "C"]], [["B*H", "B", "D"], ["B*H", "D", "C"]])
        cn = [2, 3, 4]
    elif kind == "wrong_input":
        # 경계 입력 자리에 **shape 이 똑같은 딴 텐서**를 쓴다. 개수·순서·shape 이 전부
        # 같으므로 위치로 짝지으면 구성상 통과한다. 생산자 앵커만이 이걸 가른다.
        j_n, = tn.add("aten.zeros.default", [], [[B * H, B, D]], [["B*H", "B", "D"]], [])
        tn.add("aten.bmm.default", [j_n, w_n], [],
               [["B*H", "B", "C"]], [["B*H", "B", "D"], ["B*H", "D", "C"]])
        cn = [3]
    elif kind == "batch_mix":
        # 배치를 섞는다 -- b=0 결과가 다른 배치에 의존하게 만든다
        rolled, = tn.add("aten.roll.default", [x_n], [[1], [1]],
                         [["B*H", "B", "D"]], [["B*H", "B", "D"]])
        tn.add("aten.bmm.default", [rolled, w_n], [],
               [["B*H", "B", "C"]], [["B*H", "B", "D"], ["B*H", "D", "C"]])
        cn = [2, 3]
    elif kind == "first_batch_only":
        # b=0 만 옳고 나머지 배치를 0 으로 만든다. `b=0` 만 보는 검사는 이걸 승인한다.
        mask, = tn.add("aten.zeros.default", [], [[B * H, B, D]], [["B*H", "B", "D"]], [])
        m0, = tn.add("aten.slice.Tensor", [mask], [1, 0, 1, 1],
                     [["B*H", "1", "D"]], [["B*H", "B", "D"]])
        one, = tn.add("aten.add.Tensor", [m0], [1.0], [["B*H", "1", "D"]], [["B*H", "1", "D"]])
        keep, = tn.add("aten.slice.Tensor", [x_n], [1, 0, 1, 1],
                       [["B*H", "1", "D"]], [["B*H", "B", "D"]])
        kept, = tn.add("aten.mul.Tensor", [keep, one],
                       [], [["B*H", "1", "D"]], [["B*H", "1", "D"], ["B*H", "1", "D"]])
        rest, = tn.add("aten.slice.Tensor", [mask], [1, 1, B, 1],
                       [["B*H", "2", "D"]], [["B*H", "B", "D"]])
        full, = tn.add("aten.cat.default", [kept, rest], [1],
                       [["B*H", "B", "D"]], [["B*H", "1", "D"], ["B*H", "2", "D"]])
        tn.add("aten.bmm.default", [full, w_n], [],
               [["B*H", "B", "C"]], [["B*H", "B", "D"], ["B*H", "D", "C"]])
        cn = [2, 3, 4, 5, 6, 7, 8, 9]
    elif kind == "bad_restore":
        # 전치는 맞게 했는데 출력 복원을 빠뜨린다
        xt, = tn.add("aten.permute.default", [x_n], [[0, 2, 1]],
                     [["B*H", "D", "B"]], [["B*H", "B", "D"]])
        wt, = tn.add("aten.permute.default", [w_n], [[0, 2, 1]],
                     [["B*H", "C", "D"]], [["B*H", "D", "C"]])
        tn.add("aten.bmm.default", [wt, xt], [],
               [["B*H", "C", "B"]], [["B*H", "C", "D"], ["B*H", "D", "B"]])
        cn = [2, 3, 4]
    else:
        raise ValueError(kind)
    attach_consumer(to, tn, co, cn, amap)
    return to, tn, co, cn, ns_o, ns_n, amap


def main():
    want = {"genuine": True, "dup_slice": False, "wrong_input": False,
            "batch_mix": False, "bad_restore": False, "first_batch_only": False}
    fails = 0
    for kind, expect in want.items():
        to, tn, co, cn, ns_o, ns_n, amap = build(kind)
        ok, why = _cmp(to, tn, co, cn, ns_o, ns_n, amap)
        good = (ok == expect)
        fails += 0 if good else 1
        mark = "OK " if good else "**검사 실패**"
        print(f"{mark} {kind:14} 판정 {'통과' if ok else '거절'} "
              f"(기대 {'통과' if expect else '거절'})   {why[:70]}")
    print()
    if fails:
        print(f"**{fails}개 항목이 기대와 다르다 — 검증기를 믿을 수 없다**")
    else:
        print("6/6 — 진짜 lowering 은 통과하고 위조 5종은 전부 거절한다")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
