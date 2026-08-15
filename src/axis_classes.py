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

_MATMUL_OPS = frozenset({"matmul", "linear", "mm", "bmm", "batched_matmul"})

_RESHAPE_OPS = frozenset({"view", "reshape", "_unsafe_view", "unsqueeze", "squeeze", "alias", "clone", "contiguous", "flatten"})

# shape 을 보존하는(=축의 뜻이 그대로인) op. 목록은 **추측이 아니라 전수 측정**으로 만들었다:
# 44개 모델의 모든 행에서 "모든 피연산자의 구체 shape 이 출력과 같은가"를 세어, 항상 그런 op 을
# 뽑았다(2026-08-14). 부분적으로만 그런 op(`sub`/`expand`/`ge`/`masked_fill` -- 브로드캐스트가
# 섞인다)도 넣어 두는데, 간선 (2) 가 **피연산자마다** `shape == 출력` 을 다시 확인하므로
# 브로드캐스트한 자리에서는 잇지 않는다.
#
# `t`/`transpose` 는 **절대 넣지 않는다**: 정사각이면 입력과 출력 shape 이 같지만 축은 바뀐다.
# 그 자리는 `build_table._transpose_swaps_names` 가 순열로 처리한다.
_IDENT_OPS = frozenset({
    "elementwise_add", "elementwise_mul", "elementwise_sub", "elementwise_div",
    "silu", "gelu", "relu", "sigmoid", "tanh", "softmax", "rsqrt", "pow", "neg",
    "_to_copy", "clone", "alias", "contiguous", "detach", "clamp", "masked_fill_",
    "dropout", "abs", "exp", "log",
    # 아래는 2026-08-14 전수 측정으로 추가. `copy_` 하나가 Qwen3.6-27B 의 남은
    # 데이터플로우 불일치 96건이었다.
    "copy_", "sub", "add", "mul", "div", "rsub", "tril", "triu", "cumsum",
    "floor_divide", "zeros_like", "ones_like", "empty_like", "full_like",
    "bitwise_not", "clamp_", "index_put_", "maximum", "minimum",
    "ge", "gt", "le", "lt", "eq", "ne", "logical_and", "logical_or", "logical_not",
    "masked_fill", "where", "expand", "softplus", "sqrt", "erf", "sign",
    "floor", "ceil", "round", "isnan", "reciprocal", "logsumexp",
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
                    # 보통은 어느 피연산자가 그 텐서인지 몰라 잇지 않는다. 예외가 하나 있다:
                    # **shape 을 보존하는 elementwise** 는 축 i 가 모든 피연산자와 출력에서
                    # 같은 뜻이므로(간선 (2) 가 이미 그것들을 한 등가류로 묶는다) 어느 쪽에
                    # 이어도 결과가 같다. 그래서 여기서는 모호함이 해가 없다.
                    #
                    # 이걸 안 열면 `elementwise_add(x, y)` 의 두 피연산자가 같은 shape 이라는
                    # 이유만으로 생산자와 끊기고, 등가류는 각각 내부적으로 일관되면서 서로
                    # 다른 이름을 갖는다 -- 충돌 0 인데 flow_ambig 288 이라는 모순된 그림이
                    # 나왔다(Qwen3.6-27B, 2026-08-14).
                    if not (r.get("op_type") in _IDENT_OPS and len(outs) == 1
                            and isinstance(outs[0], list) and po == outs[0] and match):
                        continue
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
                        # 크기-1 축은 제외한다 -- (3)(5) 와 같은 이유이고, 이 간선에만 빠져
                        # 있었다. "shape 이 같다"는 판정이 **브로드캐스트를 같은 축으로 읽는다**:
                        # Falcon-H1 은 MuP 벡터 `[1, 1, N]` 을 활성에 곱하는데, decode 는 B=1
                        # 이라 두 shape 이 우연히 같아져 버퍼의 축 0 과 배치 축이 한 등가류로
                        # 묶였다(prefill 은 T=17 이라 안 묶인다 -- decode 에서만 나는 결함이었다).
                        # 브로드캐스트되는 축은 상대와 같은 축이 아니라 상대에 **펼쳐지는** 축이다.
                        if sh[ax] == 1:
                            continue
                        uf.union((oid, "i", i, ax), (oid, "o", 0, ax))

        # (5) concat: **이어붙이는 축 말고는 전부 통과한다.** 그 축들은 같은 축이다.
        #
        #     이 간선이 없으면 등가류가 concat 에서 끊긴다. Kimi 의 MLA 는 value_states 를
        #     빈 캐시와 이어붙이는데(`concat([0], [B,n_h,T,d_v])`), 그 자리를 소스 판정으로
        #     `d_v` 로 고쳤더니 **같은 행 안에서 입력은 d_nope, 출력은 d_v** 가 됐다 --
        #     축 3 은 concat 축이 아니라 그냥 통과하는 축인데도(2026-08-14, 외부 검토가 낸
        #     교정을 검증하다 발견). 한쪽만 고치는 수정이 되어 버린다.
        #
        #     안전한 판정: **같은 랭크의 모든 피연산자가 그 축에서 출력과 크기가 같을 때만**
        #     잇는다. 이어붙이는 축은 피연산자 크기의 합이 출력이라 하나라도 다르므로 자동으로
        #     빠진다(`[128]+[64] -> [192]` 의 축 3). 크기-1 축은 늘 그렇듯 제외한다.
        if r.get("op_type") in ("concat", "cat", "stack") and len(outs) == 1                 and isinstance(outs[0], list):
            co = outs[0]
            same = [k for k, x in enumerate(ins) if isinstance(x, list) and len(x) == len(co)]
            if same:
                for ax in range(len(co)):
                    if co[ax] == 1:
                        continue
                    if all(ins[k][ax] == co[ax] for k in same):
                        for k in same:
                            uf.union((oid, "i", k, ax), (oid, "o", 0, ax))

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
        # (4) 행렬곱 합성: `[.., m, k] @ [.., k, n] -> [.., m, n]` 의 세 쌍은 같은 축이다.
        #     이것도 "같은 축"이라는 선언이므로 별도 패스가 아니라 간선이어야 한다 -- 패스로
        #     두면 통일과 서로 밀어낸다(2026-08-14 에 실제로 그랬다: 통일을 넣자 compose 가
        #     0 -> 96 으로 되살아났다). 간선으로 옮기면 합성은 통일의 결과로 보장된다.
        #     구체 shape 이 실제로 합성될 때만.
        if r.get("op_type") in _MATMUL_OPS and len(outs) == 1 and isinstance(outs[0], list):
            two = [k for k, x in enumerate(ins) if isinstance(x, list) and len(x) >= 2]
            co = outs[0]
            if len(two) >= 2 and len(co) >= 2:
                ia, ib = two[-2], two[-1]
                ca, cb = ins[ia], ins[ib]
                if ca[-1] == cb[-2] and ca[-2] == co[-2] and cb[-1] == co[-1]:
                    if ca[-1] != 1:
                        uf.union((oid, "i", ia, len(ca) - 1), (oid, "i", ib, len(cb) - 2))
                    if ca[-2] != 1:
                        uf.union((oid, "i", ia, len(ca) - 2), (oid, "o", 0, len(co) - 2))
                    if cb[-1] != 1:
                        uf.union((oid, "i", ib, len(cb) - 1), (oid, "o", 0, len(co) - 1))

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


# ---------------------------------------------------------------- 미결 축 인계
#
# 규칙으로 **끝까지 갈 수 없는 축**이 있다. 두 config 필드가 이 체크포인트에서 같은 값이면
# 값으로는 영원히 못 가르고, 정규식과 우선순위를 더 비트는 것은 다른 모델을 깨뜨리는 길이다
# (2026-08-13~14 에 그 길로 세 번 갔다가 세 번 되돌렸다).
#
# 그래서 **그런 축은 규칙이 스스로 "여기까지"라고 선언하고 ④층(LLM이 소스와 대조)으로 넘긴다.**
# 넘길 때 필요한 것을 전부 같이 준다 -- 어느 등가류인지, 실측 크기가 얼마인지, 후보가 무엇인지,
# 그 축이 어느 모듈들을 지나는지, 그리고 **붙여넣으면 되는 override 초안**. 답이 나오면
# `spread: class` 로 그 축 전체가 한 번에 바뀐다(모듈 경계에서 멈추지 않는다).
#
# 무엇을 미결로 보는가 -- 세 가지뿐이고, 전부 "규칙이 근거 없이 골랐다"는 뜻이다:
#   tie   두 개 이상의 심볼이 같은 값을 가져 전역 우선순위(=관례)로 이긴 축
#   heur  등록된 규칙이 없어 산술로 지어낸 이름 (`4*n_h_lin_k` 같은)
#   bare  이름이 없는데 크기가 커서(>= 64) 진짜 차원일 가능성이 있는 축
#
# 크기-1 축과 작은 정수(루프 카운터·피연산자 개수)는 **이름이 없는 게 정답**이므로 넣지 않는다.
# 2026-08-14 전수 분석: 이름 없는 축의 62.3% 가 크기-1, 35.2% 가 루프 카운터였다.
_BARE_MIN = 64


def unsettled(rows: list, concrete: dict, ties=None, weak=None, uf: _UF | None = None,
              settled=None) -> list:
    """규칙이 끝내지 못한 축 등가류. ④층이 그대로 집어들 수 있는 형태로 돌려준다."""
    uf = build(rows, concrete) if uf is None else uf
    tie_at, heur_at = {}, {}
    for (mp, val, cands), _ in (ties or {}).items():
        tie_at.setdefault((module_key_of(mp), val), tuple(cands))
    for (rule, mp, label), _ in (weak or {}).items():
        if str(rule).startswith("heur"):
            heur_at.setdefault((module_key_of(mp), str(label)), str(rule))

    ordinals = op_ordinals(rows)
    all_mods = {module_key_of(r.get('module_path') or '') for r in rows}
    groups = collections.defaultdict(list)
    for r in rows:
        oid = r.get("op_id")
        crow = concrete.get(oid) or {}
        for fld, tag in (("output_shape", "o"), ("input_shape", "i")):
            cvals = crow.get(fld) or []
            for si, sh in enumerate(r.get(fld) or []):
                if not isinstance(sh, list):
                    continue
                csh = cvals[si] if si < len(cvals) and isinstance(cvals[si], list) else None
                for ax, lab in enumerate(sh):
                    size = csh[ax] if csh is not None and ax < len(csh) else None
                    if not isinstance(size, int) or size <= 1:
                        continue
                    groups[uf.find((oid, tag, si, ax))].append(
                        (oid, tag, si, ax, str(lab), size, r.get("module_path") or "",
                         r.get("op_type"), tuple(str(x) for x in sh)))

    # **질문 단위로 접는다.** 등가류는 레이어마다 하나씩 생기므로 접지 않으면 같은 질문이
    # 수백 번 반복된다(V4-Pro 3,577건, Qwen3.6-27B 3,904건 -- 읽을 수 없는 목록이다).
    # 검토자가 답해야 할 것은 "이 모듈의 이 크기 축은 무엇인가" 하나다.
    folded = {}
    for root, sites in groups.items():
        label, size = sites[0][4], sites[0][5]
        mods = sorted({module_key_of(s[6]) for s in sites if s[6]})
        why = cands = None
        for mk in mods:
            if (mk, size) in tie_at:
                why, cands = "tie", list(tie_at[(mk, size)])
                break
            if (mk, label) in heur_at:
                why, cands = "heur", []
                break
        if why is None and label.isdigit() and size >= _BARE_MIN:
            why, cands = "bare", []
        if why is None:
            continue
        # **이미 소스 판정으로 닫힌 축은 다시 묻지 않는다.** override 가 손댄 슬롯이 이 등가류
        # 안에 하나라도 있으면 종결된 것이다. 이게 없으면 답한 질문이 재생성마다 되살아나고,
        # 검토자가 정답을 다시 넣으면 `applied: 0` 인 죽은 교정이 된다(Codex 2026-08-14).
        if settled and any((s[0], s[1], s[2], s[3]) in settled for s in sites):
            continue
        # 축 **위치**까지 키에 넣는다. 값이 같은 심볼이 넷이면(Kimi: d_head == d_rope ==
        # n_h == n_kv == 64) 값으로는 영원히 못 가르지만 `[B, n_h, T, d_head]` 처럼 위치가
        # 말해 준다. 위치를 빼고 접으면 head 개수 축과 head 폭 축이 한 질문으로 뭉쳐
        # **답할 수 없는 질문**이 된다 -- 인계는 답할 수 있는 형태여야 한다(2026-08-14).
        # 앵커 = 그 축을 가장 먼저 만든 자리. stub 은 **이 앵커를 유일하게 지목**해야 한다.
        #
        # module/from/expect 만으로는 못 지목한다: Kimi 의 `self_attn` 에는 크기 64 짜리
        # `n_h` 축이 **여섯 개의 서로 다른 등가류**에 나뉘어 있고, 그중 일부는 진짜 head 수,
        # 일부는 RoPE 폭이다. 같은 stub 을 주면 어느 하나를 고치는 순간 나머지가 망가진다.
        # 외부 검토(Codex, 2026-08-14)가 정확히 그 이유로 "이 stub 으로는 안전한 override 를
        # 만들 수 없다"며 작업을 거부했고, 그 판단이 옳았다.
        #
        # 그래서 앵커의 **렌더된 shape 전체**와 축 위치를 stub 에 싣는다. `spread: class` 가
        # 나머지 자리로 옮기므로 앵커 하나만 정확히 짚으면 된다.
        anc = min(sites)
        a_shape, a_ax = list(anc[8]), anc[3]
        a_field, a_si, a_op, a_oid = anc[1], anc[2], anc[7], anc[0]
        # 모듈도 **앵커의 것**을 쓴다. 한 등가류가 여러 모듈을 지나면 `mods[0]`(알파벳순 첫
        # 모듈)과 앵커의 모듈이 달라져, 초안의 조건과 검증의 조건이 어긋난다 -- 그러면 검증이
        # 아무것도 못 찾고 전부 "모호"로 나온다(2026-08-14 에 실제로 그랬다).
        a_mod = module_key_of(anc[6]) if anc[6] else "(root)"
        pos = "%d/%d" % (a_ax, len(a_shape))
        key = (a_mod, size, label, why, tuple(cands or ()),
               pos, tuple(a_shape), a_field, a_si, a_op, ordinals.get(a_oid))
        e = folded.setdefault(key, {"classes": 0, "sites": 0, "modules": set(), "ops": set(),
                                    "shapes": collections.Counter()})
        e["classes"] += 1
        e["sites"] += len(sites)
        e["modules"].update(mods)
        e["ops"].update(s[7] for s in sites if s[7])
        for st in sites:
            e["shapes"]["[" + ", ".join(st[8]) + "]  (축 %d)" % st[3]] += 1
        e["anchor"] = ("[" + ", ".join(a_shape) + "]", a_ax)

    out = []
    for (a_mod, size, label, why, cands, pos, a_shape,
         a_field, a_si, a_op, a_nth), e in folded.items():
        mod = a_mod
        out.append({
            "module": mod,
            "size": size,
            "current_label": label,
            "why": why,
            "candidates": list(cands),
            "classes": e["classes"],
            "axes": e["sites"],
            "also_in": sorted(e["modules"] - {mod})[:5],
            "op_types": sorted(e["ops"])[:6],
            "axis_pos": pos,
            "anchor_module": a_mod,
            "anchor_shape": "[" + ", ".join(a_shape) + "]",
            "anchor_axis": int(pos.split("/")[0]),
            # 한 등가류가 여러 op 를 지나면 축 위치가 shape 마다 다르다. 앵커의 것을 먼저 싣고
            # 나머지는 참고용이다 -- 예전에는 둘이 섞여 "위치 2/3 인데 표본은 축 3" 처럼
            # 서로 어긋나 보였다.
            "other_shapes": [k for k, _ in e["shapes"].most_common(3)
                             if not k.startswith("[" + ", ".join(a_shape) + "]")][:2],
            "override_stub": {
                "module": _stub_regex(a_mod, all_mods),
                "spread": "class",
                "shape": a_shape,          # 앵커를 유일하게 지목한다 (아래 검증 참고)
                "axis": int(pos.split("/")[0]),
                "field": a_field,
                "shape_index": a_si,
                "op_type": a_op,
                "nth": a_nth,
                "from": label,
                "to": "<소스가 말하는 이름>",
                "expect": size,
                "source": "<modeling_*.py:줄 인용>",
            },
        })
    # **stub 유일성 검증 — 접힌 항목끼리 세지 않고, 실제 등가류를 센다.**
    #
    # 첫 판은 folded 항목끼리 중복을 셌다. 그건 아무것도 검증하지 못한다: 여러 등가류가 같은
    # (shape, axis) 로 **먼저 접히고** 나면 항목은 하나뿐이라 중복이 0으로 보인다. 실제 적용기는
    # 접힘을 모르고 조건에 맞는 **모든 행**을 잡는다. 외부 검토(Codex, 2026-08-14)가 실측으로
    # 짚었다 -- `[B, n_h, T, d_nope]` 축 3 stub 하나가 976 자리 / **366 등가류**를 잡는다.
    #
    # 그래서 stub 을 실제로 돌려 본다: 그 조건에 맞는 자리들이 몇 개의 서로 다른 등가류에
    # 속하는가. 둘 이상이면 그 초안은 쓸 수 없다.
    # **검사는 적용기를 그대로 모사해야 한다.**
    #
    # 두 번 틀렸다. 처음에는 접은 뒤에 셌고(그래서 아무것도 검증 못 했다 -- 외부 검토가 짚었다),
    # 고친 뒤에도 모듈을 **문자열 일치**로 비교했는데 적용기는 **정규식**으로 매치한다.
    # `_stub_regex` 가 만든 짧은 꼬리는 다른 모듈까지 잡는다: `conv$` 는 `...conv` 와
    # `...conv.conv` 를, `mamba$` 는 `...mamba` 와 `...mamba_decoder.mamba` 를 함께 잡는다
    # (2026-08-14, 폴트 인젝션을 넣으려다 발견). 검사가 적용기와 다르면 그 검사는 장식이다.
    #
    # 그래서 여기서는 stub 의 정규식을 실제로 돌린다. 매치된 등가류를 **레이어별로** 세고,
    # 한 레이어 안에서 둘 이상이면 그 초안은 쓸 수 없다(레이어마다 하나씩인 것은 정상 --
    # override 는 모든 레이어에 걸려야 한다).
    import re as _re
    _LI = _re.compile(r"\.(?:layers|h|blocks|block|layer)\.(\d+)(?:\.|$)")
    site_idx = []
    for root, sites in groups.items():
        for oid, tag, si, ax, lab, size, mp, op, sh in sites:
            m = _LI.search(mp or "")
            site_idx.append((module_key_of(mp), lab, size, ax, tuple(sh), tag, si, op,
                             ordinals.get(oid), m.group(1) if m else "-", root))
    for it in out:
        st = it["override_stub"]
        if not st["module"]:
            it["stub_ambiguous"] = "모듈 정규식을 만들 수 없어 지목이 불가능하다."
            continue
        rx = _re.compile(st["module"])
        per = collections.defaultdict(set)
        for mk, lab, size, ax, sh, tag, si, op, nth, lay, root in site_idx:
            if (lab == st["from"] and size == st["expect"] and ax == st["axis"]
                    and sh == tuple(st["shape"]) and tag == st["field"]
                    and si == st["shape_index"] and op == st["op_type"]
                    and nth == st["nth"] and rx.search(mk)):
                per[lay].add(root)
        worst = max((len(v) for v in per.values()), default=0)
        it["stub_layers"] = len(per)
        it["stub_classes_per_layer"] = worst
        if worst != 1:
            it["stub_ambiguous"] = (
                "이 초안은 한 레이어 안에서 등가류 %d개를 동시에 잡는다 — 그대로 쓰면 "
                "나머지가 망가진다. 이 축은 위치 선택자로 지목할 수 없으니 `open` 으로 "
                "남길 것." % worst)
    out.sort(key=lambda x: -x["axes"])
    return out


def _stub_regex(module_key: str, all_keys=None) -> str:
    """override 의 `module` 에 넣을 정규식. **반드시 그 module_key 에 매치해야 한다.**

    module_key 를 그대로 쓰면 안 된다 -- `model.layers.*.self_attn` 을 정규식으로 읽으면
    `\.*` 가 "점 0개 이상"이 되어 엉뚱한 것에 매치한다. 레이어 와일드카드(`*`)를 건너뛰고
    **뒤에서부터 최대 세 마디**만 이스케이프해 쓴다(`label_overrides` 는 `search` 라 접미사면
    충분하다). 첫 판에서 `model.layers.*.self_attn` 이 `layers\.self_attn$` 로 나와 아무것도
    매치하지 않았다 -- 쓸 수 없는 초안을 주는 것은 안 주느니만 못하므로 아래에서 검증한다.
    """
    import re as _re
    parts = module_key.split(".")
    tail = []
    for seg in reversed(parts):
        if seg == "*" or not seg:
            break
        tail.insert(0, seg)
        if len(tail) == 3:
            break
    if not tail:
        # 모듈이 와일드카드로 끝나면(Zamba2 의 `...adapter_list.*.*`) 꼬리를 못 만든다.
        # 그때는 경로 전체를 이스케이프해 못 박는다 -- `*` 도 이스케이프되므로 정확히
        # 그 module_key 에만 맞는다. 빈 정규식을 돌려주면 초안이 통째로 쓸모없어진다.
        rx = "^" + _re.escape(module_key) + "$"
        return rx if _re.search(rx, module_key) else ""
    # 꼬리가 짧으면 **다른 모듈까지 잡는다**: `conv$` 는 `...conv` 와 `...conv.conv` 를,
    # `mamba$` 는 `...mamba` 와 `...mamba_decoder.mamba` 를 함께 잡는다. 그래서 뒤에서부터
    # 한 마디씩 늘려 가며 **앵커 모듈 하나에만 맞는 가장 짧은 꼬리**를 고르고, 끝까지 가도
    # 유일하지 않으면 경로 전체를 이스케이프해 못 박는다(`*` 도 그대로 이스케이프된다).
    # 2026-08-14: 검사를 적용기와 같게 만들자 이 과매칭이 드러났다.
    cands = [".".join(tail[i:]) for i in range(len(tail) - 1, -1, -1)]
    for c in cands:
        rx = _re.escape(c) + "$"
        if not _re.search(rx, module_key):
            continue
        if all_keys is None or sum(1 for k in all_keys if _re.search(rx, k)) == 1:
            return rx
    rx = "^" + _re.escape(module_key) + "$"
    return rx if _re.search(rx, module_key) else ""


def module_key_of(module_path):
    import re as _re
    if not module_path:
        return "(root)"
    return _re.sub(r"\.(\d+)(?=\.|$)", ".*", module_path)


def write_unsettled(model_dir: str, phase: str, rows: list, concrete: dict,
                    ties=None, weak=None, settled=None) -> int:
    items = unsettled(rows, concrete, ties, weak, settled=settled)
    with open(os.path.join(model_dir, "full", f"{phase}.unsettled.json"), "w",
              encoding="utf-8") as f:
        json.dump({"phase": phase, "count": len(items),
                   "note": ("규칙으로 끝낼 수 없어 ④층(소스 대조)으로 넘기는 축. "
                            "답이 나오면 override_stub 을 rules/label_overrides.yaml 에 채워 넣는다 "
                            "— `spread: class` 라 그 축 전체가 한 번에 바뀐다."),
                   "items": items[:300]}, f, ensure_ascii=False, indent=1)
    return len(items)


def unsettled_count(model_dir: str) -> int:
    n = 0
    for ph in ("prefill", "decode"):
        p = os.path.join(model_dir, "full", f"{ph}.unsettled.json")
        if not os.path.exists(p):
            continue
        try:
            with open(p, encoding="utf-8") as f:
                n += int((json.load(f) or {}).get("count") or 0)
        except (ValueError, OSError):
            pass
    return n


def op_ordinals(rows: list) -> dict:
    """{op_id: 그 모듈 안에서 같은 op_type 의 몇 번째인가}.

    한 모듈에 같은 종류의 op 가 여러 번 나오면(MLA 의 `self_attn` 에는 `split_with_sizes` 가
    q 용과 kv 용 둘) `op_type` 만으로는 못 가른다. 레이어 인덱스를 포함한 **모듈 인스턴스**
    안에서 순서를 센다 -- 레이어마다 같은 구조가 반복되므로 이 서수는 레이어를 가로질러
    안정적이다. 생성기와 적용기가 같은 순서(op_id 오름차순)로 세므로 결과가 일치한다.
    """
    seen, out = collections.Counter(), {}
    for r in sorted(rows, key=lambda x: x.get("op_id") or 0):
        k = (r.get("module_path") or "", r.get("op_type") or "")
        out[r.get("op_id")] = seen[k]
        seen[k] += 1
    return out


def bad_stub_count(model_dir: str) -> int:
    """지목이 불가능해 쓸 수 없는 초안 수(발행된 파일에서 다시 센다)."""
    n = 0
    for ph in ("prefill", "decode"):
        p = os.path.join(model_dir, "full", f"{ph}.unsettled.json")
        if not os.path.exists(p):
            continue
        try:
            with open(p, encoding="utf-8") as f:
                n += sum(1 for it in ((json.load(f) or {}).get("items") or [])
                         if it.get("stub_ambiguous"))
        except (ValueError, OSError):
            pass
    return n


def dead_confirm_count(model_dir: str) -> int:
    """더 이상 맞지 않는(=이름이 바뀐) 확인 기록 수. 낡은 확인은 낡은 교정만큼 위험하다."""
    p = os.path.join(model_dir, "full", "label_confirmed.json")
    if not os.path.exists(p):
        return 0
    try:
        with open(p, encoding="utf-8") as f:
            return sum(1 for o in (json.load(f) or []) if not o.get("matched"))
    except (ValueError, OSError):
        return 0


# 확인에도 근거를 요구한다. `should_be_renamed` 판정에 인용을 강제하는 것과 같은 이유다 --
# "봤고 맞다"는 **소스에 대한 주장**이고, 그 주장이 그 축을 인계 목록에서 영구히 빼 버린다.
# 근거 없이 뺀 축은 아무도 다시 안 본다. 2026-08-14 에 확인 경로를 만들 때 이 검사를 같이
# 넣지 않아, 외부 검토가 성실했던 덕에 16건이 전부 인용을 갖췄을 뿐이었다.
_CONFIRM_CITE = __import__("re").compile(r"(modeling_\w+\.py|configuration_\w+\.py|\.py:\d+|https?://)")


def uncited_confirm_count(model_dir: str) -> int:
    """소스 인용 없는 확인 기록 수."""
    p = os.path.join(model_dir, "full", "label_confirmed.json")
    if not os.path.exists(p):
        return 0
    try:
        with open(p, encoding="utf-8") as f:
            return sum(1 for o in (json.load(f) or [])
                       if not _CONFIRM_CITE.search(str(o.get("source") or "")))
    except (ValueError, OSError):
        return 0
