"""축 등가류 — "같은 축"인 자리들을 하나로 묶는다.

WHY THIS EXISTS
---------------
이름이 **텐서가 아니라 칸에 붙어 있다.** 한 텐서는 표에 여러 번 나타난다 -- 만든 op 의
`output_shape` 에 한 번, 그것을 쓰는 모든 op 의 `input_shape` 에 또 한 번씩 -- 그런데 그
자리들은 서로 연결돼 있지 않고, 각각 독립적으로 값 매칭으로 이름을 얻는다.
Qwen2.5-0.5B 하나만 봐도 텐서 2,673개에 라벨 자리는 19,016개다.

그래서 소스로 확정한 이름을 한 자리에 넣으면 옆자리가 옛 이름으로 남고, 데이터플로우 검사가
"같은 텐서에 두 이름"을 잡는다. 지금까지 그것을 사후 조정 패스 13개로 꿰매 왔고, 그 패스들은
서로 싸운다(`_squeeze_view_keeps_names` 가 붙인 이름을 `_unname_refilled_operands` 가 떼는 것을
2026-08-13 에 확인).

이 모듈은 그 순서를 뒤집기 위한 첫 걸음이다: **먼저 "어느 자리들이 같은 축인가"를 정하고**,
이름은 그 등가류마다 한 번만 결정한다. 지금은 감사 전용 -- 라벨을 바꾸지 않고 무엇이 어긋나
있는지만 보고한다.

무엇이 간선인가
--------------
보수적으로만 잇는다. 잘못 이으면 한 번에 대량 오라벨이 나므로, 애매하면 **잇지 않는다**.

  (1) 생산자 → 소비자: op B 가 op A 에 의존하고, A 의 어떤 출력과 **정확히 하나의** B 피연산자가
      같은 구체 shape 을 가질 때, 축 단위로 잇는다. 두 피연산자가 같은 shape 이면 어느 쪽이
      그 텐서인지 모르므로 잇지 않는다 -- 함대 전체에서 간선의 15.9% 가 여기 해당한다.
      (트레이서는 트레이스 시점에 진짜 텐서 신원을 갖고 있다. 행에 남기지 않을 뿐이다.
       텐서 id 를 남기면 이 15.9% 가 사라지지만, 재추적이 필요하므로 별도 단계로 둔다.)

  (2) shape 보존 elementwise/복사: 출력 축은 입력 축과 같은 축이다. 텐서로는 새것이지만
      **축의 뜻**은 같다 -- 게이트의 `ident_incons` 가 이미 그것을 전제로 검사한다.

간선을 더 넣을 수 있는 자리(전치의 축 순열, view 의 앞뒤 정렬, matmul 의 수축 축)는 지금
build_table 의 패스들이 사후에 하고 있는 일이다. 그것들을 여기 간선으로 옮기는 것이 다음
단계이며, 옮기고 나면 그 패스들은 "패치"가 아니라 "선언"이 된다.
"""
import collections
import json
import os

_RESHAPE_OPS = frozenset({"view", "reshape", "_unsafe_view", "unsqueeze", "squeeze", "alias", "clone", "contiguous", "flatten"})

_IDENT_OPS = frozenset({
    "elementwise_add", "elementwise_mul", "elementwise_sub", "elementwise_div",
    "silu", "gelu", "relu", "sigmoid", "tanh", "softmax", "rsqrt", "pow", "neg",
    "_to_copy", "clone", "alias", "contiguous", "detach", "clamp", "masked_fill_",
    "dropout", "abs", "exp", "log",
})


class _UF:
    __slots__ = ("p",)

    def __init__(self):
        self.p = {}

    def find(self, x):
        p = self.p
        p.setdefault(x, x)
        root = x
        while p[root] != root:
            root = p[root]
        while p[x] != root:      # path compression
            p[x], x = root, p[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def _slots(shape, key):
    return [(key[0], key[1], key[2], ax) for ax in range(len(shape))]


def build(rows: list, concrete: dict) -> _UF:
    """축 슬롯 `(op_id, 'i'|'o', shape_index, axis)` 들의 등가류.

    `concrete` 는 op_id -> 구체 shape 행. 구체값으로만 잇는다 -- 렌더된 이름으로 이으면
    이름이 틀린 곳끼리 묶여 틀림을 확정해 버린다.
    """
    uf = _UF()
    for r in rows:
        oid = r.get("op_id")
        c = concrete.get(oid) or {}
        ins = c.get("input_shape") or []
        outs = c.get("output_shape") or []

        # (1) 생산자 -> 소비자
        for dep in (r.get("depends_on") or []):
            douts = (concrete.get(dep) or {}).get("output_shape") or []
            for oi, po in enumerate(douts):
                if not isinstance(po, list) or not po:
                    continue
                # 생산자 안에서도 모호할 수 있다: `split_with_sizes` 는 출력이 여럿이고 그중
                # 둘이 같은 shape 이면(Nemotron 의 in_proj 분할은 빈 조각 `[1,24,0]` 이 둘)
                # 소비자가 어느 조각을 받았는지 알 수 없다. 그때는 잇지 않는다.
                if sum(1 for x in douts if x == po) != 1:
                    continue
                match = [k for k, x in enumerate(ins) if x == po]
                if len(match) != 1:
                    continue                      # 소비자 쪽이 모호하면 잇지 않는다
                i = match[0]
                # **출력 인덱스를 그대로 쓴다.** 여기서 0 을 쓰면 다출력 op 의 조각들이 전부
                # 한 등가류로 뭉친다 -- Nemotron 의 5-way split 이 `0 / d_inner /
                # d_inner+2*n_g*d_state / d_state` 를 한 축으로 묶어 거짓 충돌 138건을
                # 만들었다(감사 1차, 2026-08-14). 라벨을 건드리기 전에 잡은 자기 버그다.
                for ax in range(len(po)):
                    uf.union((dep, "o", oi, ax), (oid, "i", i, ax))

        # (2) shape 을 보존하는 elementwise/복사: 입력 축 == 출력 축
        if r.get("op_type") in _IDENT_OPS and len(outs) == 1 and isinstance(outs[0], list):
            for i, sh in enumerate(ins):
                if isinstance(sh, list) and sh == outs[0]:
                    for ax in range(len(sh)):
                        uf.union((oid, "i", i, ax), (oid, "o", 0, ax))

        # (3) reshape 의 **오른쪽 정렬**: 뒤에서부터 크기가 같은 동안 그 축들은 같은 축이다.
        #     `[B, T, n_h_lin_v, d_head_lin_v] -> [n_h_lin_v*T, d_head_lin_v]` 에서 마지막 축은
        #     같은 축이고, 그 앞은 뭉쳐졌으므로 다르다. 첫 불일치에서 멈춘다 -- 뭉치거나 쪼개진
        #     축에 대해서는 아무 말도 하지 않는 것이 맞다.
        #
        #     왜 필요한가: 이 간선이 없으면 등가류가 랭크 경계에서 끊겨 **너무 작아진다**.
        #     Qwen3-Next 의 gated norm 을 소스대로 `d_head_lin_v` 로 고정했더니 그 norm 을 먹이는
        #     view 의 입력이 여전히 `d_head_lin_k` 여서 reshape 자체 유도와 어긋났다
        #     (288 -> 504, 2026-08-14). 클래스가 작으면 교정이 거기까지 못 간다.
        #
        #     안전한 이유: 크기가 같은 자리만 잇고 첫 불일치에서 끊으므로, 뭉침/쪼갬을 가로질러
        #     잇는 일이 없다. `[2,3,4] -> [6,4]` 는 마지막 축만, `[4,4,4] -> [4,16]` 은 아무것도.
        if r.get("op_type") in _RESHAPE_OPS and len(outs) == 1 and isinstance(outs[0], list):
            for i, sh in enumerate(ins):
                if not isinstance(sh, list):
                    continue
                dst = outs[0]
                for k in range(1, min(len(sh), len(dst)) + 1):
                    if sh[-k] != dst[-k]:
                        break
                    # 크기-1 축은 **잇지 않는다.** 정보가 없어서 아무 축과도 크기가 맞고,
                    # 이으면 배치 축 `B`(크기 1)와 브로드캐스트용 리터럴 `1` 이 한 등가류가
                    # 된다 -- 처음 넣었을 때 그 하나로 충돌이 1,557 -> 46,530 이 됐고 그중
                    # 44,469 가 `1 vs B` 였다(2026-08-14). 정렬은 계속하되 결합만 건너뛴다.
                    if sh[-k] == 1:
                        continue
                    uf.union((oid, "i", i, len(sh) - k), (oid, "o", 0, len(dst) - k))
    return uf


def name_conflicts(rows: list, concrete: dict, uf: _UF | None = None) -> dict:
    """등가류마다 지금 붙어 있는 이름들. {root: {"names": {...}, "sites": [...]}}"""
    uf = build(rows, concrete) if uf is None else uf
    out = collections.defaultdict(lambda: {"names": set(), "sites": []})
    for r in rows:
        oid = r.get("op_id")
        for si, sh in enumerate(r.get("output_shape") or []):
            if not isinstance(sh, list):
                continue
            for ax, v in enumerate(sh):
                e = out[uf.find((oid, "o", si, ax))]
                e["names"].add(str(v))
                e["sites"].append((oid, "o", si, ax, r.get("op_type"), r.get("module_path")))
        for si, sh in enumerate(r.get("input_shape") or []):
            if not isinstance(sh, list):
                continue
            for ax, v in enumerate(sh):
                e = out[uf.find((oid, "i", si, ax))]
                e["names"].add(str(v))
                e["sites"].append((oid, "i", si, ax, r.get("op_type"), r.get("module_path")))
    return out


def audit(model_dir: str, phases=("prefill", "decode")) -> dict:
    """{phase: {"classes": n, "conflicts": n, "detail": [...]}} — 라벨은 건드리지 않는다."""
    res = {}
    for ph in phases:
        raw = os.path.join(model_dir, "full", f"{ph}.trace.raw.jsonl")
        con = os.path.join(model_dir, "full", f"{ph}.shapes.concrete.jsonl")
        if not (os.path.exists(raw) and os.path.exists(con)):
            continue
        rows = [json.loads(l) for l in open(raw, encoding="utf-8")]
        concrete = {}
        with open(con, encoding="utf-8") as f:
            for l in f:
                c = json.loads(l)
                concrete[c["op_id"]] = c
        cls = name_conflicts(rows, concrete)
        bad = [(sorted(v["names"]), v["sites"]) for v in cls.values() if len(v["names"]) > 1]
        res[ph] = {"classes": len(cls), "conflicts": len(bad), "detail": bad}
    return res


def write_audit(model_dir: str, phase: str) -> int:
    """방금 발행된 `full/<phase>.trace.raw.jsonl` 을 읽어 감사 결과를 옆에 쓴다.

    **발행물을 읽는다** -- 메모리 안의 중간 상태가 아니라. label_provenance 가 후처리 이전
    라벨을 세고 있던 것을 2026-08-13 에 발견했고, 같은 실수를 반복하지 않는다.
    """
    r = audit(model_dir, phases=(phase,)).get(phase)
    if not r:
        return 0
    out = {"phase": phase, "classes": r["classes"], "conflicts": r["conflicts"],
           "note": "한 등가류(같은 축) 안에서 이름이 둘 이상인 자리. 0 이어야 옳다.",
           "detail": [{"names": names,
                       "sites": [{"op_id": s[0], "field": s[1], "shape_index": s[2],
                                  "axis": s[3], "op_type": s[4], "module": s[5]}
                                 for s in sites[:12]],
                       "n_sites": len(sites)}
                      for names, sites in r["detail"][:200]]}
    with open(os.path.join(model_dir, "full", f"{phase}.axis_classes.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    return r["conflicts"]


def conflicts(model_dir: str) -> int:
    """게이트용 — 발행된 **트레이스에서 다시 세어** 반환한다 (양쪽 phase).

    옆에 써 둔 `*.axis_classes.json` 을 읽지 않는다. 요약 파일은 산출물이지 증거가 아니고,
    그것을 믿으면 게이트가 검사하는 대상이 트레이스가 아니라 우리가 쓴 숫자가 된다.
    이 프로젝트는 같은 실수를 이미 두 번 했다(prefill 만 읽던 게이트, 후처리 이전 라벨을 세던
    label_provenance). 다시 세는 비용은 모델당 수십 ms 다.
    """
    r = audit(model_dir)
    return sum(v["conflicts"] for v in r.values())
