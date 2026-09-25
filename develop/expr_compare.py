r"""축 식 두 개가 **의미상 같은가.** 0-c 의 판정 비교 규칙(승인된 Q4).

세 층으로 본다. 위에서 결론이 나면 아래로 내려가지 않는다.

    1  문자열 정규화      `·`->`*`, `−`->`-`, 공백, 겉 괄호
    2  AST 대수 비교      심볼은 **원자로 둔다.** 값을 대입하지 않는다
    3  선언된 alias·유도   `rules/label_aliases.yaml` 과 structure 의 `symbols_label_only`

**값이 같은 것은 근거가 아니다.** `d_model == d_moe` 가 우연히 같은 모델이 있으므로
concrete 값으로 동일 판정을 내리면 안 된다. 그래서 이 모듈은 namespace 를 받지 않는다.

`/` 는 이 저장소의 관례상 **floor division** 이다(`dim_expr._FLOOR`). 그래서
`(n_h*d_head)/n_h` 를 `d_head` 로 줄이는 것은 **나눗셈이 딱 맞을 때만** 참이다. 그런
축약이 필요한 비교는 `different` 라고 단정하지 않고 `cannot_determine` 을 낸다.

판정:
    same               정규형이 같다
    alias              선언된 alias·유도를 펼치면 같아진다
    different          같지 않다 (나눗셈 축약이 끼어들지 않았을 때만)
    cannot_determine   모르는 심볼 / 나눗셈 축약 필요 / 파싱 불가

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


# ---------------------------------------------------------------- 정규형
# 정규형은 **해시 가능한 튜플**이다. 곱·합은 정렬된 다중집합으로, 나눗셈과 거듭제곱과
# 호출은 **불투명한 노드**로 남긴다(축약하지 않는다).
def _canon(node, symbols, aliases, seen, has_div):
    if isinstance(node, ast.Expression):
        return _canon(node.body, symbols, aliases, seen, has_div)
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, int) or isinstance(node.value, bool):
            raise Undecidable("정수가 아닌 상수")
        return ("int", node.value)
    if isinstance(node, ast.Name):
        name = node.id
        if name in aliases and name not in seen:
            # 선언된 유도를 펼친다. 같은 이름을 두 번 펼치지 않는다(순환 방지)
            return _canon(_parse(aliases[name]), symbols, aliases,
                          seen | {name}, has_div)
        if symbols is not None and name not in symbols:
            raise Undecidable(f"모르는 심볼 {name}")
        return ("sym", name)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return _mul([("int", -1), _canon(node.operand, symbols, aliases, seen,
                                        has_div)])
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return _canon(node.operand, symbols, aliases, seen, has_div)
    if isinstance(node, ast.BinOp):
        L = _canon(node.left, symbols, aliases, seen, has_div)
        R = _canon(node.right, symbols, aliases, seen, has_div)
        op = node.op
        if isinstance(op, ast.Mult):
            return _mul([L, R])
        if isinstance(op, ast.Add):
            return _add([L, R])
        if isinstance(op, ast.Sub):
            return _add([L, _mul([("int", -1), R])])
        if isinstance(op, (ast.Div, ast.FloorDiv)):
            has_div.append(True)
            return ("div", L, R)               # **축약하지 않는다** (floor division)
        if isinstance(op, ast.Pow):
            # `n_h**2` 와 `n_h*n_h` 는 **같다.** 작은 양의 정수 지수는 펼쳐서 곱으로
            # 만든다 -- 펼치지 않으면 "다르다" 고 틀리게 단정한다(자기검사가 잡았다).
            if R[0] == "int" and 0 <= R[1] <= 8:
                return _mul([L] * R[1]) if R[1] else ("int", 1)
            if R[0] == "int" and R[1] < 0:
                has_div.append(True)           # 1/x 의미가 숨어 있다
            return ("pow", L, R)
        if isinstance(op, ast.Mod):
            has_div.append(True)
            return ("mod", L, R)
        raise Undecidable(f"모르는 연산 {type(op).__name__}")
    if isinstance(node, ast.Call):
        fn = getattr(node.func, "id", None)
        if fn not in CALLS:
            raise Undecidable(f"모르는 호출 {fn}")
        if fn in ("ceil", "roundup"):
            has_div.append(True)               # 나눗셈 의미가 숨어 있다
        args = tuple(_canon(a, symbols, aliases, seen, has_div)
                     for a in node.args)
        # min·max 는 인자 순서가 뜻을 바꾸지 않는다
        return (("call", fn) + (tuple(sorted(args)) if fn in ("min", "max")
                                else args))
    raise Undecidable(f"모르는 노드 {type(node).__name__}")


def _mul(parts):
    flat, k = [], 1
    for p in parts:
        if p[0] == "mul":
            flat.extend(p[1])
            k *= p[2]
        elif p[0] == "int":
            k *= p[1]
        else:
            flat.append(p)
    if k == 0:
        return ("int", 0)
    if not flat:
        return ("int", k)
    if k == 1 and len(flat) == 1:
        return flat[0]
    return ("mul", tuple(sorted(flat)), k)


def _add(parts):
    flat, k = [], 0
    for p in parts:
        if p[0] == "add":
            flat.extend(p[1])
            k += p[2]
        elif p[0] == "int":
            k += p[1]
        else:
            flat.append(p)
    # 같은 항을 모은다: a + a -> 2*a
    bag = {}
    for p in flat:
        if p[0] == "mul" and len(p) == 3:
            bag[p[1]] = bag.get(p[1], 0) + p[2]
        else:
            bag[(p,)] = bag.get((p,), 0) + 1
    terms = []
    for key, c in bag.items():
        if c == 0:
            continue
        base = list(key)
        terms.append(_mul([("int", c)] + base))
    if not terms:
        return ("int", k)
    if k == 0 and len(terms) == 1:
        return terms[0]
    return ("add", tuple(sorted(terms)), k)


def _parse(expr):
    t = normalize_text(expr)
    if t in (None, ""):
        raise Undecidable("빈 식")
    try:
        return ast.parse(t, mode="eval")
    except SyntaxError as e:
        raise Undecidable(f"파싱 불가: {e.msg}")


def canonical(expr, symbols=None, aliases=None):
    """`(정규형, 나눗셈이 끼었는가)`. 심볼은 원자로 남는다."""
    has_div = []
    form = _canon(_parse(expr), symbols, aliases or {}, frozenset(), has_div)
    return form, bool(has_div)


def compare(a, b, symbols=None, aliases=None):
    """`(판정, 이유)`. 판정은 same / alias / different / cannot_determine."""
    if normalize_text(a) == normalize_text(b):
        return "same", "문자열 정규화로 일치"
    try:
        fa, da = canonical(a, symbols, None)
        fb, db = canonical(b, symbols, None)
    except Undecidable as e:
        # alias 를 펼치면 알 수도 있다 -- 아래에서 한 번 더 본다
        fa = fb = None
        da = db = False
        first = e.reason
    else:
        first = None
        if fa == fb:
            return "same", "대수 정규형이 일치"
    if aliases:
        try:
            ga, xa = canonical(a, symbols, aliases)
            gb, xb = canonical(b, symbols, aliases)
        except Undecidable as e:
            return "cannot_determine", first or e.reason
        if ga == gb:
            return "alias", "선언된 alias·유도를 펼치면 일치"
        if xa or xb:
            return "cannot_determine", "나눗셈 축약이 필요할 수 있다 (floor division)"
        # **alias 를 펼쳐 양쪽이 정규화됐다.** 첫 패스에서 이름을 몰랐던 것은 더 이상
        # 결론을 막지 않는다 -- 막으면 확실히 다른 것도 cannot_determine 이 된다.
        return "different", "선언된 alias 를 펼쳐 비교해도 다르다"
    if first:
        return "cannot_determine", first
    if da or db:
        # `/` 가 끼면 "다르다" 고 단정할 수 없다: (n_h*d_head)/n_h 는 나눗셈이 딱
        # 맞을 때만 d_head 다. 나눗셈 가정을 근거로 쓰지 않는다.
        return "cannot_determine", "나눗셈 축약이 필요할 수 있다 (floor division)"
    return "different", "대수 정규형이 다르다"


def load_aliases(path=None):
    """선언된 alias·유도. **없으면 빈 것이다** -- 추측하지 않는다."""
    import yaml
    p = path or os.path.join(PROJ, "rules", "label_aliases.yaml")
    if not os.path.exists(p):
        return {}
    d = yaml.safe_load(io.open(p, encoding="utf-8")) or {}
    out = {}
    for k, v in (d.get("aliases") or {}).items():
        out[k] = v["expr"] if isinstance(v, dict) else v
    return out


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit(2)
    v, why = compare(sys.argv[1], sys.argv[2], aliases=load_aliases())
    print(f"{v}  ({why})")
