r"""`expr_compare` 가 **값을 근거로 쓰지 않고**, 나눗셈 축약을 단정하지 않는가.

승인된 Q4 규칙의 요점 셋을 시험한다.

    1  심볼은 원자다 -- `d_model` 과 `d_moe` 가 값이 같아도 다르다
    2  `/` 는 floor division 이므로 축약이 필요한 비교는 `cannot_determine`
    3  alias 는 **선언된 것만** -- 모르는 이름은 `cannot_determine`

실행:
    .venv\Scripts\python.exe develop\test_expr_compare.py
"""
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import expr_compare as E                                         # noqa: E402

OK, FAIL = [], []
SYMS = {"n_h": 64, "n_kv": 8, "d_head": 128, "d_model": 8192, "d_moe": 8192,
        "T": 640, "d_chunk": 128, "B": 1, "n_exp": 128, "d_ff": 28672}
ALIAS = {"n_chunk": "T/d_chunk", "d_qk": "d_head", "hidden": "d_model"}


def case(a, b, want, note="", symbols=SYMS, aliases=ALIAS):
    got, why = E.compare(a, b, symbols, aliases)
    ok = got == want
    (OK if ok else FAIL).append(f"{a!r} vs {b!r}")
    mark = "  OK   " if ok else "  FAIL "
    extra = f"   <- {want} 이어야 한다" if not ok else ""
    print(f"{mark}{got:<17}{a!r} vs {b!r}  ({why}){extra}"
          + (f"  -- {note}" if note else ""))


print("1) 문자열 정규화")
case("n_h*d_head", "n_h·d_head", "same", "가운뎃점")
case("n_h * d_head", "n_h*d_head", "same", "공백")
case("(n_h*d_head)", "n_h*d_head", "same", "겉 괄호")
case("d_model−1", "d_model-1", "same", "유니코드 마이너스")

print()
print("2) 대수 정규형 -- 심볼은 원자로 둔다")
case("n_h*d_head", "d_head*n_h", "same", "교환")
case("2*d_model", "d_model*2", "same")
case("(n_h*d_head)*2", "2*n_h*d_head", "same", "결합")
case("d_model+d_model", "2*d_model", "same", "같은 항 모으기")
case("d_model*2+1", "1+2*d_model", "same")
case("n_h*d_head", "n_kv*d_head", "different", "머리 수가 다르다")
case("d_model", "d_moe", "different", "**값이 같지만 다른 심볼이다**")
case("d_model", "d_model+1", "different")
case("n_h**2", "n_h*n_h", "same", "정수 거듭제곱은 펼친다")
# `n_kv == 2` 라는 **값 가정**이 있어야 같아진다. 값을 근거로 쓰지 않으므로 다르다.
case("n_h**n_kv", "n_h*n_h", "different", "심볼 지수는 값 가정 없이는 다르다")
case("n_h**n_kv", "n_h**n_kv", "same", "같은 심볼 지수는 같다")
case("n_h**-1", "1/n_h", "cannot_determine", "음수 지수는 나눗셈 의미")
case("min(T,d_chunk)", "min(d_chunk,T)", "same", "min 은 순서 무관")
case("max(T,d_chunk)", "min(T,d_chunk)", "different", "함수가 다르다")

print()
print("3) `/` 는 floor division -- 축약을 단정하지 않는다")
case("(n_h*d_head)/n_h", "d_head", "cannot_determine",
     "나눗셈이 딱 맞을 때만 참이다")
case("d_model/n_h", "d_model/n_h", "same", "같은 식은 같다")
case("d_model/n_h", "d_head", "cannot_determine")
case("T/d_chunk", "T/d_chunk", "same")
case("ceil(T/d_chunk)", "T/d_chunk", "cannot_determine", "ceil 은 나눗셈 의미")
case("d_model%n_h", "0", "cannot_determine", "나머지")
case("d_model/n_h", "d_moe/n_h", "cannot_determine",
     "나눗셈이 끼면 다르다고도 못 한다")

print()
print("4) alias 는 선언된 것만")
case("n_chunk", "T/d_chunk", "alias", "선언된 유도")
case("T/d_chunk", "n_chunk", "alias", "방향 무관")
case("d_qk", "d_head", "alias")
case("d_qk*n_h", "d_head*n_h", "alias", "합성식 안에서도 펼친다")
case("n_chunk", "T/d_model", "cannot_determine", "펼쳐도 다르지만 나눗셈이 끼었다")
case("d_qk", "d_model", "different", "펼쳐서 비교해 확실히 다르다")
case("무명축", "d_head", "cannot_determine", "모르는 심볼")
case("d_head", "d_kv_lora", "cannot_determine", "선언 안 된 이름")
case("n_chunk", "T/d_chunk", "cannot_determine", "alias 를 안 주면 모른다",
     aliases=None)

print()
print("5) 심볼표를 안 주면 이름을 검사하지 않는다")
case("아무거나*2", "2*아무거나", "same", "정규형만 본다", symbols=None,
     aliases=None)
case("x*y", "y*x", "same", symbols=None, aliases=None)

print()
print("6) 망가진 입력")
case("n_h*", "n_h", "cannot_determine", "파싱 불가")
case("", "d_head", "cannot_determine", "빈 식")
case("'문자열'", "d_head", "cannot_determine", "정수가 아닌 상수")
case("open('x')", "d_head", "cannot_determine", "모르는 호출 -- 평가하지 않는다")

print()
print("7) 값을 대입할 통로가 **구조적으로** 없다")
import inspect                                                   # noqa: E402


def sig_has(fn, name):
    return name in inspect.signature(fn).parameters


for fn in (E.compare, E.canonical):
    for bad in ("ns", "namespace", "concrete", "values"):
        cond = not sig_has(fn, bad)
        (OK if cond else FAIL).append(f"{fn.__name__}({bad})")
        print(("  OK   " if cond else "  FAIL ")
              + f"`{fn.__name__}` 이 `{bad}` 를 받지 않는다")
mods = {m for m in dir(E) if inspect.ismodule(getattr(E, m))}
for bad in ("dim_expr", "summarize", "crosswalk"):
    cond = bad not in mods
    (OK if cond else FAIL).append(f"no import {bad}")
    print(("  OK   " if cond else "  FAIL ") + f"`{bad}` 를 import 하지 않는다")
code = inspect.getsource(E)
body = chr(10).join(l for l in code.splitlines()
                    if not l.lstrip().startswith("#"))
body = body.split('"""', 2)[-1]                  # 모듈 docstring 을 뺀다
cond = "eval(" not in body
(OK if cond else FAIL).append("no eval")
print(("  OK   " if cond else "  FAIL ") + "`eval(` 를 쓰지 않는다 (주석·docstring 제외)")

print()
print(f"{len(OK)}/{len(OK) + len(FAIL)} 통과 — 값이 같은 것은 근거가 아니다")
if FAIL:
    print("실패: " + ", ".join(FAIL))
sys.exit(1 if FAIL else 0)
