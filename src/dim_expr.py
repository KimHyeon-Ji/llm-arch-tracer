r"""렌더된 축 라벨을 **수치로 평가**한다. 흩어져 있던 같은 기능을 한곳에 모은 것이다.

왜 필요한가
-----------
라벨은 식이다(`n_h*d_head`, `T+1`, `E*B*T`). 그 식이 맞는지 보려면 심볼에 값을 넣어
실제 shape 과 비교해야 한다. 그 계산이 네 군데에 흩어져 있었다 --
`develop/verify_all.py` 의 라벨 검사, `summarize` 의 유도식, `anchors` 의 앵커 식,
`symbolic_dims.normalize`. 새 게이트를 만들면서 다섯 번째를 만들지 않으려고 여기로 모은다
(외부 검토 2026-09-13).

**배치 차수**가 이 모듈의 존재 이유다. B=1 로 트레이스하면 `T` 와 `B*T` 가 같은 값이라
라벨이 접힌 배치를 빠뜨려도 산술적으로 참이다. 같은 라벨을 **B=2 심볼표로도** 평가해
실제 B=2 shape 과 비교하면 그 누락이 드러난다.
"""
import json
import math
import os
import re

_FLOOR = re.compile(r"(?<![/*])/(?![/*])")     # 우리 식의 `/` 는 floor division 이다
_IDENT = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")


def namespace(prov: dict, batch: int = 1, seq_len: int | None = None) -> dict:
    """`provenance.json` 에서 심볼 -> 값. `batch` 로 `B` 를 바꿔 끼운다.

    B 만 바꾸고 나머지는 그대로 두는 것이 핵심이다 -- 같은 라벨을 두 환경에서 평가해야
    하므로 라벨을 다시 만들면 안 된다(다시 만들면 두 번 독립으로 오판할 수 있다).
    """
    import summarize
    ns = dict(prov.get("symbol_table") or {})
    ns["B"] = batch

    class _C:
        def __init__(s, dd):
            for k, v in dd.items():
                setattr(s, k, v)

    ns.update(summarize._derived_vars(_C(prov.get("config") or {}),
                                      summarize.load_derived_dims()))
    if ns.get("n_h_ssm") and ns.get("d_head_ssm"):
        ns["d_inner"] = ns["n_h_ssm"] * ns["d_head_ssm"]
    if ns.get("n_g_ssm"):
        ns["n_g"] = ns["n_g_ssm"]
    for a, b in (("n_k", "n_h_lin_k"), ("d_k", "d_head_lin_k"), ("n_v", "n_h_lin_v")):
        if ns.get(b):
            ns[a] = ns[b]
    ns["B"] = batch          # 유도식이 덮어썼을 수 있다
    # **T 도 바꿔 끼울 수 있다.** 전환 diff 는 새 라벨을 옛 판의 `(B=1, 옛 T)` 로 평가해야
    # 옛 라벨과 같은 좌표가 된다 -- 발행점이 `(B, T)` 를 함께 고르므로 T 가 달라질 수 있고,
    # 그러면 T 를 품은 모든 shape 의 서명이 어긋나 전부 미짝으로 떨어진다(V4-Pro 에서
    # 2048 -> 2049 로 바뀌어 74,856건이 그렇게 됐다, 2026-09-14).
    if seq_len is not None:
        ns["T"] = seq_len
    ns.update(ceil=math.ceil, round=round, min=min, max=max,
              roundup=lambda a, b: math.ceil(a / b) * b)
    return ns


# 같은 식을 같은 namespace 에서 수백만 번 다시 계산하지 않는다.
#
# 라벨은 몇십 종류뿐인데 축 자리는 수백만이다 -- Kimi-K3 는 prefill+decode 합쳐 557만
# 자리이고, 전환 diff 는 자리마다 여러 번 `evaluate` 를 부른다. 캐시 없이 돌리면 감사
# 하나가 여덟 시간을 넘겼다(2026-09-16). namespace 는 실행당 두세 개뿐이라 그것을 키에
# 넣고, `id()` 가 재활용되지 않도록 참조를 붙들어 둔다.
_MEMO: dict = {}
_NS_KEEP: list = []


def evaluate(expr, ns: dict):
    """식의 값. 못 구하면 `None` -- 예외를 밖으로 내보내지 않는다."""
    if expr is None:
        return None
    if isinstance(expr, int):
        return expr
    key = (id(ns), str(expr))
    if key in _MEMO:
        return _MEMO[key]
    if len(_NS_KEEP) < 64 and not any(n is ns for n in _NS_KEEP):
        _NS_KEEP.append(ns)                       # id 재활용 방지
    e = str(expr).replace("·", "*").replace("−", "-")
    e = _FLOOR.sub("//", e)
    try:
        v = eval(e, {"__builtins__": {}}, ns)     # noqa: S307 -- 우리가 렌더한 라벨이다
    except Exception:
        v = None
    v = v if isinstance(v, int) else None
    _MEMO[key] = v
    return v


def free_symbols(expr) -> set:
    """식에 나오는 이름들(함수 이름 제외)."""
    if expr is None:
        return set()
    return {m for m in _IDENT.findall(str(expr))
            if m not in ("ceil", "round", "min", "max", "roundup")}


def batch_degree(expr, ns: dict):
    """이 식이 배치에 **몇 제곱으로** 비례하는가. 못 정하면 `None`.

    기호 해석 대신 **두 번 평가해 비율을 본다.** 식의 형태가 무엇이든(`B*T`, `E*B*T`,
    `roundup(B*T, 8)`) 통하고, 라벨을 다시 만들지 않아도 된다.
    """
    ns1 = dict(ns); ns1["B"] = 1
    ns2 = dict(ns); ns2["B"] = 2
    a, b = evaluate(expr, ns1), evaluate(expr, ns2)
    if not a or not b or a <= 0 or b <= 0 or b % a:
        return None
    r = b // a
    if r == 1:
        return 0
    d = 0
    while r % 2 == 0:
        r //= 2
        d += 1
    return d if r == 1 else None


def shape_batch_degree(shape, ns: dict):
    """한 shape 전체의 배치 차수 합. 텐서 하나에 배치 축은 **하나**뿐이어야 한다."""
    tot = 0
    for lab in (shape or []):
        d = batch_degree(lab, ns)
        if d is None:
            return None
        tot += d
    return tot


def load_provenance(model_dir: str) -> dict:
    p = os.path.join(model_dir, "full", "provenance.json")
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f) or {}
