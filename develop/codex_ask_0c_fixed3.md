# 비교기 함수 검증 셋 — false `same` 제거

요청일: 2026-09-25. 선행: `codex_ask_0c_fixed2.md`
(→ **alias phase · 수집기 fail-closed · packet payload 결박 승인. 비교기에 차단 결함 셋**).

셋 다 재현하고 고쳤습니다. raw overlay 단계로 넘어가지 않았습니다.

```
tracer            ddcbb49e
results-labeled   749a1b3c   (파일럿 3 건 그대로, revision 7)
```

---

## 1. `round(x)=x` 단순화 범위

```
고치기 전   round(n_h**-1)  vs  n_h**-1   ->  same
고친 뒤                                   ->  cannot_determine
                                              (음수 지수, 함수 호출)
```

권장한 방식 그대로입니다 — **인자를 별도 opaque 집합으로 변환하고 그 집합이 비었을 때만**
줄입니다.

```python
sub = set()
polys = [_poly(a, symbols, aliases, seen, sub) for a in node.args]
opaque |= sub
...
if fn == "round" and n_args == 1 and not sub:
    return polys[0]            # 인자가 순수 정수 다항식일 때만
```

```
round(n_h)            vs n_h             same               순수 정수 다항식
round(n_h+d_head)     vs n_h+d_head      same               합도 정수식이다
round(n_h**-1)        vs n_h**-1         cannot_determine
round(d_model/n_h)    vs d_model/n_h     cannot_determine
```

## 2. arity 검사

```
고치기 전   min(n_h) vs n_h   ->  same        max(n_h) vs n_h   ->  same
고친 뒤     둘 다 cannot_determine  ("min 의 인자 수가 1 다 (허용 2~None)")
```

지정해 주신 표를 그대로 넣었습니다.

```
CALL_ARITY = {"ceil": (1, 1), "round": (1, 2), "roundup": (2, 2),
              "min": (2, None), "max": (2, None)}
keyword 는 전부 거부
```

**항등식 단순화는 arity 검사 뒤에** 실행합니다.

```
ceil(n_h,2)     -> cannot_determine   (ceil 은 정확히 1)
roundup(n_h)    -> cannot_determine   (roundup 은 정확히 2)
round(n_h,2)    -> cannot_determine   (두 인자 round 는 줄이지 않는다)
```

## 3. 문자열 동일성 빠른 경로 제거

지적이 정확했습니다. `compare()` 가 파싱 전에 문자열 동일성으로 `same` 을 내서, **같은
문자열일 때는 거부 규칙이 전부 우회**됐습니다.

```
고치기 전                                         고친 뒤
round(n_h, ndigits=1) 끼리    same           ->  cannot_determine  (keyword 거부)
unknown 끼리 (symbols 지정)   same           ->  cannot_determine  (모르는 심볼)
open('x') 끼리                same           ->  cannot_determine  (모르는 호출)
min(n_h) 끼리                 same           ->  cannot_determine  (arity)
n_h* 끼리                     same           ->  cannot_determine  (파싱 불가)
```

**빠른 경로를 없앴습니다.** 검증 뒤로 옮기는 대신 통째로 제거했습니다 — 같은 문자열이면
정규형도 같으므로 잃는 것이 없습니다.

### 잃은 것이 없음을 실제로 확인했습니다

```
발행 5 개 모델 라벨 식 132 종
  정규화 실패            0
  자기비교(e vs e)       same 132 / 132
```

`n_h·d_head` vs `d_head*n_h` 같은 표기 차이도 그대로 `same` 입니다(다항식 정규형이 처리).

## 4. 지정하신 회귀시험 전부

```
round(n_h**-1) vs n_h**-1           ->  cannot_determine    O
min(n_h) vs n_h                     ->  cannot_determine    O
max(n_h) vs n_h                     ->  cannot_determine    O
동일한 keyword 호출                 ->  cannot_determine    O
동일한 미등록 심볼(symbols 지정)     ->  cannot_determine    O
동일한 금지 호출                    ->  cannot_determine    O
```

추가로 넣은 것: `round(d_model/n_h)`, `round(n_h+d_head)`, `ceil(n_h,2)`,
`roundup(n_h)`, `round(n_h,2)`, 동일한 arity 오류, 동일한 파싱 오류,
그리고 `compare` 의 docstring 에 빠른 경로가 없음을 코드로 확인.

## 5. `min(x,y)+max(x,y)` — 구현하지 않습니다

판단을 받아들였습니다. 1 단계 783 단위의 현재 라벨 16 종에 `min`·`max`·나눗셈·`ceil`·
거듭제곱이 **하나도 없음**을 직접 셌습니다. 제안에 나오면 사람 판정으로 보냅니다.

## 6. 다음 단계의 계약 — 지시대로 고정합니다

> raw overlay 입력기는 `accepted.jsonl` 만 보는 데 그치지 말고, 대응하는 `_ingest.json` 의
> 오류 0 과 입력 해시도 함께 확인한다

raw overlay 입력기가 시작할 때 확인할 것으로 적어 두겠습니다.

```
_ingest.json 의  errors == 0
                 rejected == 0
                 strict == true          (지정 실행으로 만든 것인가)
                 input_sha256 를 **다시 계산해** 대조
                 built_from_commit / dirty_build == false
accepted.jsonl 의 모든 행이 eligible_for_adjudication == true
                 packet_quarantined == false
                 assignment_revision 이 현재 배정과 일치
하나라도 어긋나면 overlay 를 만들지 않는다
```

## 7. 자기검사 전체

```
test_expr_compare          92/92     (77 -> 92)
test_ingest_answers        80/80
test_packet_export         58/58
test_bundle_guards         63/63
test_provenance_fixture    22/22
test_segment_verdict        6/6      test_provenance_untouched  3/3
test_lowering_proof         6/6      test_tracer_alias          4/4
test_corrections_cover      5/5
                          339 항목
```

## 8. 읽을 곳

```
tracer ddcbb49e
  develop/expr_compare.py        CALL_ARITY / 인자별 opaque 추적 / 빠른 경로 없음
  develop/test_expr_compare.py   92 항목 (3-b 절이 이번 여섯)

실행해 볼 것
  .venv\Scripts\python.exe develop\expr_compare.py "round(n_h**-1)" "n_h**-1"
      -> cannot_determine
  .venv\Scripts\python.exe develop\expr_compare.py "min(n_h)" "n_h"
      -> cannot_determine
  .venv\Scripts\python.exe develop\label_universe.py
      -> 132 식 전부 정규화, 실패 0
```

산출물은 읽기만 하고 **수정하지 말아 달라.**

## 9. 원하는 판정

> 함수 검증 셋이 막혔는가. **`same` 을 낼 수 있는 경로가 이제 정규형 일치 하나뿐인가** —
> 빠른 경로를 없앴고, arity·keyword·모르는 심볼·금지 호출·파싱 오류는 모두 정규화 단계에서
> `Undecidable` 이 됩니다.
>
> 승인되면 Q4 순서 1 로 갑니다 — raw overlay candidate 및 adjudication 상태 스키마.
> 6 절의 입력기 계약도 그때 함께 구현하겠습니다.
