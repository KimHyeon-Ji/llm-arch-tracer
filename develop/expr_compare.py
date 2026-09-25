r"""축 식 두 개가 **의미상 같은가.** 0-c 의 판정 비교 규칙(승인된 Q4).

세 층으로 본다. 위에서 결론이 나면 아래로 내려가지 않는다.

    1  문자열 정규화      `·`->`*`, `−`->`-`, 공백, 겉 괄호
    2  다항식 정규형      심볼은 **원자로 둔다.** 값을 대입하지 않는다
    3  선언된 alias·유도   `rules/label_aliases.yaml`(모델·phase 범위) + `symbols_label_only`

**값이 같은 것은 근거가 아니다.** `d_model == d_moe` 가 우연히 같은 모델이 있으므로
concrete 값으로 동일 판정을 내리면 안 된다. 그래서 이 모듈은 namespace 를 받지 않는다.

## 2 층은 **완전한** 다항식 정규형이다

처음에는 곱·합을 정렬된 다중집합으로만 만들었다. 그래서 분배법칙을 못 넘어 의미상 같은
식을 `different` 로 판정했다(외부 검토 2026-09-25 가 반례를 실행해 보였다).

    2*(d_model+n_h)  vs  2*d_model+2*n_h     -> different  (틀렸다)
    (a+b)*c          vs  a*c+b*c             -> different  (틀렸다)
    (a+b)**2         vs  a*a+2*a*b+b*b       -> different  (틀렸다)

`different` 가 rename 후보로 이어지므로 이것은 단순한 false negative 가 아니라 **같은 식을
변경 대상으로 만드는** 결함이다. 그래서 `+ - *` 와 작은 비음수 정수 거듭제곱을 **정수 계수
다항식**으로 완전히 정규화한다(단항식 -> 계수 사전).

`different` 는 "내 트리가 다르다" 가 아니라 **"지원하는 의미론 안에서 같지 않음이
증명됐다"** 일 때만 낸다. 다항식으로 다루지 못하는 것(나눗셈·나머지·`ceil`·심볼 지수·
**함수 호출**)이 끼어 있고 정규형이 다르면 `cannot_determine` 이다.

## 함수 호출도 불일치를 단정하지 않는다

호출을 원자로만 두면 항등식을 못 넘어 또 틀린 `different` 가 나왔다
(외부 검토 2026-09-25 가 재현했다).

    round(n_h)                    vs  n_h        -> different  (정수 축에서는 같다)
    min(n_h,n_h)                  vs  n_h        -> different  (멱등)
    min(n_h,n_kv)+max(n_h,n_kv)   vs  n_h+n_kv   -> different  (항상 성립)

그리고 keyword 인자를 무시해서 `round(n_h, ndigits=1)` 과 `round(n_h, ndigits=2)` 가
`same` 이 됐다.

그래서 지금은

    함수별 **arity 를 검사한다.** 틀리면 `Undecidable` -- `min(n_h)` 은 이 DSL 의
      유효한 축 식이 아니다
    keyword 인자는 **거부한다** -- 지원하지 않는 것을 조용히 흘리지 않는다
    확실히 건전한 항등식만 정규형에서 처리한다
      `min(x,x) = x`, `max(x,x) = x`   -- 멱등. x 가 무엇이든 성립한다
      `round(x) = x`                   -- **인자가 순수 정수 다항식일 때만.**
                                          `round(n_h**-1)` 은 정수식이 아니다
    그 밖에 호출이 끼고 정규형이 다르면 **`cannot_determine`**

**문자열이 같아도 검증을 건너뛰지 않는다.** 예전에는 `compare()` 가 파싱 전에 문자열
동일성으로 `same` 을 냈다. 그래서 `round(x, ndigits=1)` 끼리, 모르는 심볼끼리, 금지된
호출끼리 비교하면 거부 규칙이 전부 우회됐다(외부 검토 2026-09-25). 빠른 경로를 없앴다 --
같은 문자열이면 정규형도 같으므로 잃는 것이 없다.

`min`·`max` 의 완전한 항등식(교환·결합·분배, `min+max`)은 검증한 뒤 따로 넣는다.

`/` 는 이 저장소의 관례상 **floor division** 이다(`dim_expr._FLOOR`). 그래서
`(n_h*d_head)/n_h` 를 `d_head` 로 줄이는 것은 **나눗셈이 딱 맞을 때만** 참이다.

판정:
    same               정규형이 같다
    alias              선언된 alias·유도를 펼치면 같아진다
    different          다항식으로 **증명된** 불일치 (나눗셈류가 끼지 않았다)
    cannot_determine   모르는 심볼 / 다항식 밖의 연산이 낀 불일치 / 파싱 불가

실행(자기검사는 `test_expr_compare.py`):
    .venv\Scripts\python.exe develop\expr_compare.py "n_h*d_head" "d_head*n_h"
"""
import ast
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)

CALLS = ("ceil", "round", "roundup", "min", "max")
# 함수별 positional 인자 개수 `(최소, 최대)`. `None` 은 상한 없음.
# `len(args)` 를 원자에 넣는 것만으로는 부족하다 -- 유효하지 않은 arity 를 받아들이면
# `min(n_h)` 이 `n_h` 와 same 이 된다(외부 검토 2026-09-25).
CALL_ARITY = {"ceil": (1, 1), "round": (1, 2), "roundup": (2, 2),
              "min": (2, None), "max": (2, None)}
# 다항식으로 다룰 수 없는 연산. 이것이 끼면 불일치를 단정하지 않는다.
# `call` 이 여기 있는 이유: 호출의 항등식을 완전히 처리하지 못한다(위 docstring).
OPAQUE_DIVISION = ("div", "mod", "ceil", "roundup", "negpow", "call")
MAX_POW = 8
_WS = re.compile(r"\s+")


class Undecidable(Exception):
    """단정하면 안 되는 비교. 이유를 담는다."""

    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def normalize_text(s):
    """1 층. 글자만 고른다 -- 구조는 건드리지 않는다."""
    if s is None:
        return None
    t = str(s).replace("·", "*").replace("−", "-").replace("×", "*")
    t = t.replace("⋅", "*").replace("∗", "*")
    t = _WS.sub("", t)
    while t.startswith("(") and t.endswith(")"):
        depth = 0
        for i, c in enumerate(t):
            depth += (c == "(") - (c == ")")
            if depth == 0 and i < len(t) - 1:
                return t                       # 겉 괄호가 한 쌍이 아니다
        t = t[1:-1]
    return t


# ------------------------------------------------------- 다항식 (정수 계수)
# 표현: `{단항식: 계수}`. 단항식은 `((원자, 지수), ...)` 를 정렬한 튜플이고 상수항은 `()`.
# 원자는 심볼 `("sym", 이름)` 또는 다항식으로 못 다루는 노드(`div`, `mod`, `pow`, 호출).
def _p_const(k):
    return {(): k} if k else {}


def _p_atom(a):
    return {((a, 1),): 1}


def _p_add(p, q):
    out = dict(p)
    for m, c in q.items():
        v = out.get(m, 0) + c
        if v:
            out[m] = v
        else:
            out.pop(m, None)
    return out


def _p_neg(p):
    return {m: -c for m, c in p.items()}


def _m_mul(m1, m2):
    d = dict(m1)
    for a, e in m2:
        d[a] = d.get(a, 0) + e
    return tuple(sorted(d.items()))


def _p_mul(p, q):
    out = {}
    for m1, c1 in p.items():
        for m2, c2 in q.items():
            m = _m_mul(m1, m2)
            v = out.get(m, 0) + c1 * c2
            if v:
                out[m] = v
            else:
                out.pop(m, None)
    return out


def _p_pow(p, n):
    out = _p_const(1)
    for _ in range(n):
        out = _p_mul(out, p)
    return out


def _pkey(p):
    """해시 가능한 정규형. 사전 순서에 의존하지 않는다."""
    return tuple(sorted(p.items()))


def _as_int(p):
    """다항식이 정수 상수면 그 값, 아니면 `None`."""
    if not p:
        return 0
    if len(p) == 1 and () in p:
        return p[()]
    return None


# ------------------------------------------------------------------ 변환
def _poly(node, symbols, aliases, seen, opaque):
    if isinstance(node, ast.Expression):
        return _poly(node.body, symbols, aliases, seen, opaque)
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, int) or isinstance(node.value, bool):
            raise Undecidable("정수가 아닌 상수")
        return _p_const(node.value)
    if isinstance(node, ast.Name):
        name = node.id
        if name in aliases and name not in seen:
            # 선언된 유도를 펼친다. 같은 이름을 두 번 펼치지 않는다(순환 방지)
            return _poly(_parse(aliases[name]), symbols, aliases,
                         seen | {name}, opaque)
        if symbols is not None and name not in symbols:
            raise Undecidable(f"모르는 심볼 {name}")
        return _p_atom(("sym", name))
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            return _p_neg(_poly(node.operand, symbols, aliases, seen, opaque))
        if isinstance(node.op, ast.UAdd):
            return _poly(node.operand, symbols, aliases, seen, opaque)
        raise Undecidable(f"모르는 단항 연산 {type(node.op).__name__}")
    if isinstance(node, ast.BinOp):
        L = _poly(node.left, symbols, aliases, seen, opaque)
        R = _poly(node.right, symbols, aliases, seen, opaque)
        op = node.op
        if isinstance(op, ast.Mult):
            return _p_mul(L, R)
        if isinstance(op, ast.Add):
            return _p_add(L, R)
        if isinstance(op, ast.Sub):
            return _p_add(L, _p_neg(R))
        if isinstance(op, (ast.Div, ast.FloorDiv)):
            opaque.add("div")                  # **축약하지 않는다** (floor division)
            return _p_atom(("div", _pkey(L), _pkey(R)))
        if isinstance(op, ast.Mod):
            opaque.add("mod")
            return _p_atom(("mod", _pkey(L), _pkey(R)))
        if isinstance(op, ast.Pow):
            n = _as_int(R)
            if n is not None and 0 <= n <= MAX_POW:
                # `(a+b)**2` 를 `a*a+2*a*b+b*b` 로 펼친다 -- 펼치지 않으면 의미상 같은
                # 식을 different 로 판정한다(외부 검토 2026-09-25).
                return _p_pow(L, n)
            if n is not None and n < 0:
                opaque.add("negpow")           # 1/x 의미가 숨어 있다
            else:
                opaque.add("sympow")           # 심볼 지수 -- 다항식 밖이다
            return _p_atom(("pow", _pkey(L), _pkey(R)))
        raise Undecidable(f"모르는 연산 {type(op).__name__}")
    if isinstance(node, ast.Call):
        fn = getattr(node.func, "id", None)
        if fn not in CALLS:
            raise Undecidable(f"모르는 호출 {fn}")
        # **keyword 인자를 거부한다.** 무시하면 `round(x, ndigits=1)` 과
        # `round(x, ndigits=2)` 가 same 이 된다(외부 검토 2026-09-25).
        if node.keywords:
            raise Undecidable(f"{fn} 의 keyword 인자는 지원하지 않는다")
        if getattr(node.func, "attr", None):
            raise Undecidable("속성 호출은 지원하지 않는다")
        lo, hi = CALL_ARITY[fn]
        n_args = len(node.args)
        if n_args < lo or (hi is not None and n_args > hi):
            raise Undecidable(
                f"{fn} 의 인자 수가 {n_args} 다 (허용 {lo}"
                + (f"~{hi}" if hi != lo else "") + ")")
        # **인자의 불투명 연산을 따로 센다.** `round(x)=x` 는 x 가 순수 정수 다항식일
        # 때만 성립한다. 공용 집합에 섞으면 그 판단을 할 수 없다.
        sub = set()
        polys = [_poly(a, symbols, aliases, seen, sub) for a in node.args]
        opaque |= sub
        args = [_pkey(p) for p in polys]
        # ---- 건전한 항등식만 여기서 줄인다 (arity 검사 뒤에)
        if fn in ("min", "max") and len(set(args)) == 1:
            return polys[0]                    # 멱등: x 가 무엇이든 성립한다
        if fn == "round" and n_args == 1 and not sub:
            return polys[0]                    # 인자가 순수 정수 다항식일 때만
        if fn in ("ceil", "roundup"):
            opaque.add(fn)                     # 나눗셈 의미가 숨어 있다
        else:
            opaque.add("call")
        # min·max 는 인자 순서가 뜻을 바꾸지 않는다
        key = tuple(sorted(args)) if fn in ("min", "max") else tuple(args)
        return _p_atom(("call", fn, len(args)) + key)
    raise Undecidable(f"모르는 노드 {type(node).__name__}")


def _parse(expr):
    t = normalize_text(expr)
    if t in (None, ""):
        raise Undecidable("빈 식")
    try:
        return ast.parse(t, mode="eval")
    except SyntaxError as e:
        raise Undecidable(f"파싱 불가: {e.msg}")


def canonical(expr, symbols=None, aliases=None):
    """`(정규형, 다항식 밖의 연산 집합)`. 심볼은 원자로 남는다."""
    opaque = set()
    p = _poly(_parse(expr), symbols, aliases or {}, frozenset(), opaque)
    return _pkey(p), opaque


def _undecidable(oa, ob):
    """불일치를 단정할 수 없게 만드는 연산이 끼었는가.

    나눗셈·나머지·`ceil`·음수 지수·심볼 지수·**남아 있는 함수 호출**은 다항식이 다루지
    못한다. 정규형이 달라도 실제로는 같을 수 있으므로 `different` 라고 말하지 않는다.

    `min(a,b)` vs `a` 하나만 보면 자유 정수 심볼에서 항등식이 아니므로 `different` 가
    수학적으로 맞다. 그러나 구현이 `min`·`max` 항등식을 **완전히** 처리하지 못하는 동안
    호출이 낀 불일치를 통째로 내리는 것이 맞다(외부 검토 2026-09-25).
    """
    bad = set(OPAQUE_DIVISION) | {"sympow"}
    return bool((oa | ob) & bad)


def compare(a, b, symbols=None, aliases=None):
    """`(판정, 이유)`. 판정은 same / alias / different / cannot_determine.

    **문자열 동일성 빠른 경로를 두지 않는다.** 그것을 두면 keyword 거부·모르는 심볼
    거부·금지 호출 거부가 같은 문자열일 때 전부 우회된다(외부 검토 2026-09-25).
    같은 문자열이면 정규형도 같으므로 잃는 것이 없다.
    """
    first = None
    try:
        fa, oa = canonical(a, symbols, None)
        fb, ob = canonical(b, symbols, None)
    except Undecidable as e:
        first = e.reason                       # alias 를 펼치면 알 수도 있다
    else:
        if fa == fb:
            return "same", "다항식 정규형이 일치"
    if aliases:
        try:
            ga, xa = canonical(a, symbols, aliases)
            gb, xb = canonical(b, symbols, aliases)
        except Undecidable as e:
            return "cannot_determine", first or e.reason
        if ga == gb:
            return "alias", "선언된 alias·유도를 펼치면 일치"
        if _undecidable(xa, xb):
            return "cannot_determine", _why(xa | xb)
        # alias 를 펼쳐 양쪽이 정규화됐다. 첫 패스에서 이름을 몰랐던 것은 더 이상 결론을
        # 막지 않는다 -- 막으면 확실히 다른 것도 cannot_determine 이 된다.
        return "different", "선언된 alias 를 펼쳐도 다항식이 다르다"
    if first:
        return "cannot_determine", first
    if _undecidable(oa, ob):
        return "cannot_determine", _why(oa | ob)
    return "different", "다항식 정규형이 다르다"


def _why(ops):
    names = {"div": "나눗셈", "mod": "나머지", "ceil": "ceil",
             "roundup": "roundup", "negpow": "음수 지수", "sympow": "심볼 지수",
             "call": "함수 호출"}
    got = sorted(names[o] for o in ops if o in names)
    return f"다항식 밖의 연산이 끼었다 ({', '.join(got)}) -- 불일치를 단정하지 않는다"


# --------------------------------------------------------------- alias 규칙
def load_aliases(model=None, phase=None, path=None):
    """선언된 alias·유도. **모델·phase 범위를 지킨다. 범위를 모르면 쓰지 않는다.**

    전역 문자열 맵으로 두면 위험하다 -- 같은 기호가 모델마다 다른 뜻일 수 있다
    (외부 검토 2026-09-25). 그래서 항목마다 `model` 과 `phase` 를 적고, 맞지 않는 항목은
    **쓰지 않는다**.

    **fail-closed 두 가지**(외부 검토 2026-09-25):

      * 호출자가 `model=None` 이면 범위를 판단할 수 없으므로 **아무 항목도 주지 않는다.**
        예전에는 그대로 통과시켜서, 범위가 필요한 규칙이 범위 없이 쓰였다.
      * `phase` 가 적힌 항목은 호출자가 `phase=None` 이면 **쓰지 않는다.**
      * `model: "*"` 도 **거부한다.** "전역 alias 금지" 와 모순되기 때문이다. 전역이
        필요하면 모델을 모두 적어라.

    형식:
        aliases:
          n_chunk:
            expr: T/d_chunk
            model: moonshotai__Kimi-K3       # 필수. 목록도 된다. `*` 는 거부
            phase: [prefill, decode]         # 생략하면 phase 를 가리지 않는다
            source: "modeling_kimi_linear.py:120-138"
    """
    import yaml
    p = path or os.path.join(PROJ, "rules", "label_aliases.yaml")
    if not os.path.exists(p):
        return {}
    d = yaml.safe_load(io.open(p, encoding="utf-8")) or {}
    out = {}
    entries = d.get("aliases") or {}
    if entries and model is None:
        # 범위를 모르면 범위 있는 규칙을 쓸 수 없다. 조용히 통과시키지 않는다.
        return {}
    for k, v in entries.items():
        if not isinstance(v, dict):
            raise ValueError(
                f"label_aliases.yaml: `{k}` 는 model 범위를 적어야 한다 "
                "(전역 alias 는 쓰지 않는다)")
        if "expr" not in v:
            raise ValueError(f"label_aliases.yaml: `{k}` 에 expr 이 없다")
        m = v.get("model")
        if m is None:
            raise ValueError(
                f"label_aliases.yaml: `{k}` 에 model 이 없다 -- 전역 alias 금지")
        ms = m if isinstance(m, list) else [m]
        if "*" in ms:
            raise ValueError(
                f"label_aliases.yaml: `{k}` 의 model 에 `*` 를 쓸 수 없다 -- "
                "전역 alias 금지. 필요한 모델을 모두 적어라")
        if model not in ms:
            continue
        ph = v.get("phase")
        if ph:
            phs = ph if isinstance(ph, list) else [ph]
            if phase is None or phase not in phs:
                continue        # phase 범위가 있는 규칙은 phase 를 알 때만 쓴다
        out[k] = v["expr"]
    return out


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit(2)
    v, why = compare(sys.argv[1], sys.argv[2])
    print(f"{v}  ({why})")
