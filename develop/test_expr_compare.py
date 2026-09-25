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
# 심볼 지수는 다항식 밖이다. `n_kv == 2` 면 같아지므로 **단정하지 않는다** --
# "지원하는 의미론 안에서 증명됐을 때만 different" 라는 원칙(외부 검토 2026-09-25).
case("n_h**n_kv", "n_h*n_h", "cannot_determine", "심볼 지수는 다항식 밖")
case("n_h**n_kv", "n_h**n_kv", "same", "같은 심볼 지수는 같다")
case("n_h**9", "n_h*n_h*n_h*n_h*n_h*n_h*n_h*n_h*n_h", "cannot_determine",
     "지수 9 는 펼치지 않는다 (MAX_POW=8)")
case("n_h**-1", "1/n_h", "cannot_determine", "음수 지수는 나눗셈 의미")
case("min(T,d_chunk)", "min(d_chunk,T)", "same", "min 은 순서 무관")
# 호출이 끼고 정규형이 다르면 **단정하지 않는다.** min/max 의 항등식을 완전히
# 처리하지 못하는 동안은 통째로 내리는 것이 맞다(외부 검토 2026-09-25).
case("max(T,d_chunk)", "min(T,d_chunk)", "cannot_determine", "호출이 끼었다")
case("round(n_h)", "n_h", "same", "정수 축에서 round 는 항등")
case("min(n_h,n_h)", "n_h", "same", "min 멱등")
case("max(n_h,n_h)", "n_h", "same", "max 멱등")
case("min(n_h,n_kv)+max(n_h,n_kv)", "n_h+n_kv", "cannot_determine",
     "항등식이지만 아직 구현하지 않았다 -- 단정하지 않는다")
case("round(n_h, ndigits=1)", "round(n_h, ndigits=2)", "cannot_determine",
     "**keyword 인자를 거부한다** (무시하면 same 이 됐다)")
case("min(n_h,n_kv)", "n_h", "cannot_determine", "호출이 끼었다")
case("ceil(n_h)", "n_h", "cannot_determine", "ceil 은 나눗셈 의미")

print()
print("2-b) **분배법칙** -- 외부 검토가 실행해 보인 반례 (2026-09-25)")
# `different` 가 rename 후보로 이어지므로, 의미상 같은 식을 different 로 판정하면
# 같은 식을 변경 대상으로 만든다. 단순한 false negative 가 아니다.
case("2*(d_model+n_h)", "2*d_model+2*n_h", "same", "상수 분배")
case("(n_h+n_kv)*d_head", "n_h*d_head+n_kv*d_head", "same", "합에 곱 분배")
case("(n_h+n_kv)+(n_h+n_kv)", "2*(n_h+n_kv)", "same", "같은 합 두 번")
case("(n_h+n_kv)**2", "n_h*n_h+2*n_h*n_kv+n_kv*n_kv", "same", "이항 전개")
case("(n_h+n_kv)**3", "n_h**3+3*n_h**2*n_kv+3*n_h*n_kv**2+n_kv**3", "same",
     "삼항 전개")
case("(d_model-n_h)*(d_model+n_h)", "d_model**2-n_h**2", "same", "곱셈 공식")
case("(n_h+1)*(n_h+2)", "n_h**2+3*n_h+2", "same")
case("(n_h+n_kv)**2", "n_h**2+n_kv**2", "different", "교차항이 빠졌다")
case("2*(d_model+n_h)", "2*d_model+n_h", "different", "한쪽만 분배됐다")
case("d_model*(n_h-n_h)", "0", "same", "0 으로 줄어든다")
case("d_model+d_model-d_model", "d_model", "same", "상쇄")

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
print("3-b) 함수 검증 -- 외부 검토가 재현한 false same (2026-09-25)")
# round 단순화는 인자가 **순수 정수 다항식**일 때만. 음수 지수는 정수식이 아니다.
case("round(n_h**-1)", "n_h**-1", "cannot_determine", "round 단순화 범위")
case("round(d_model/n_h)", "d_model/n_h", "cannot_determine", "나눗셈 인자")
case("round(n_h)", "n_h", "same", "순수 정수 다항식이면 항등")
case("round(n_h+d_head)", "n_h+d_head", "same", "합도 정수식이다")
# arity 를 검사한다. min(n_h) 은 이 DSL 의 유효한 축 식이 아니다.
case("min(n_h)", "n_h", "cannot_determine", "min 은 2 개 이상")
case("max(n_h)", "n_h", "cannot_determine", "max 는 2 개 이상")
case("ceil(n_h,2)", "n_h", "cannot_determine", "ceil 은 정확히 1")
case("roundup(n_h)", "n_h", "cannot_determine", "roundup 은 정확히 2")
case("round(n_h,2)", "n_h", "cannot_determine", "두 인자 round 는 줄이지 않는다")
# **문자열이 같아도 검증을 건너뛰지 않는다**
case("round(n_h, ndigits=1)", "round(n_h, ndigits=1)", "cannot_determine",
     "같은 keyword 호출도 거부한다")
case("min(n_h)", "min(n_h)", "cannot_determine", "같은 arity 오류도 거부한다")
case("무명축", "무명축", "cannot_determine", "같은 미등록 심볼도 거부한다")
case("open('x')", "open('x')", "cannot_determine", "같은 금지 호출도 거부한다")
case("n_h*", "n_h*", "cannot_determine", "같은 파싱 오류도 거부한다")
check_no_fast = "normalize_text(a) == normalize_text(b)" not in E.compare.__doc__
(OK if check_no_fast else FAIL).append("빠른 경로 없음")
print(("  OK   " if check_no_fast else "  FAIL ")
      + "compare 에 문자열 빠른 경로가 없다")

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
print("6-b) alias 규칙은 **모델 범위**를 요구한다")
import tempfile                                                  # noqa: E402


def alias_file(text):
    p = os.path.join(tempfile.mkdtemp(), "a.yaml")
    io.open(p, "w", encoding="utf-8", newline=chr(10)).write(text)
    return p


ok = alias_file("aliases:" + chr(10)
                + "  n_chunk:" + chr(10)
                + "    expr: T/d_chunk" + chr(10)
                + "    model: moonshotai__Kimi-K3" + chr(10))
got = E.load_aliases("moonshotai__Kimi-K3", "prefill", ok)
check_eq = got.get("n_chunk") == "T/d_chunk"
(OK if check_eq else FAIL).append("alias 범위 일치")
print(("  OK   " if check_eq else "  FAIL ") + "그 모델에서는 쓰인다")
got2 = E.load_aliases("openai__gpt-oss-20b", "prefill", ok)
cond = "n_chunk" not in got2
(OK if cond else FAIL).append("alias 범위 불일치")
print(("  OK   " if cond else "  FAIL ") + "**다른 모델에서는 쓰이지 않는다**")
for bad, why in ((("aliases:" + chr(10) + "  x: T/d_chunk"), "전역 문자열 맵 거부"),
                 (("aliases:" + chr(10) + "  x:" + chr(10) + "    expr: T"),
                  "model 없는 항목 거부"),
                 (("aliases:" + chr(10) + "  x:" + chr(10)
                   + "    model: m"), "expr 없는 항목 거부")):
    try:
        E.load_aliases("m", None, alias_file(bad))
        cond = False
    except ValueError:
        cond = True
    (OK if cond else FAIL).append(why)
    print(("  OK   " if cond else "  FAIL ") + why)
cond = E.load_aliases("m", None, os.path.join(tempfile.mkdtemp(), "없다.yaml")) == {}
(OK if cond else FAIL).append("파일 없으면 빈 것")
print(("  OK   " if cond else "  FAIL ") + "파일이 없으면 빈 것이다 (추측하지 않는다)")

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
