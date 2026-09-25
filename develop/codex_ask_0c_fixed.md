# 0-c 결함 둘 수정 — false `different` 와 fail-open

요청일: 2026-09-25. 선행: `codex_ask_0c_compare.md`
(→ **앞 두 층 미승인. 결함 두 군데 수정 후 재검토**).

지적 둘을 재현하고 고쳤습니다. overlay 구현으로 넘어가지 않았습니다.

```
tracer            7c8eac1b
results-labeled   2bdd76b1   (파일럿 3 건 그대로, revision 7)
```

---

## 1. `different` 가 보수적이지 않았다 — 반례 넷 전부 재현

```
고치기 전
  2*(d_model+n_h)  vs  2*d_model+2*n_h   ->  different   (틀렸다)
  (a+b)*c          vs  a*c+b*c           ->  different   (틀렸다)
  (a+b)+(a+b)      vs  2*(a+b)           ->  different   (틀렸다)
  (a+b)**2         vs  a*a+2*a*b+b*b     ->  different   (틀렸다)

고친 뒤  넷 다  same  (다항식 정규형이 일치)
```

**권장안 1 을 택했습니다** — `+ - *` 와 작은 비음수 정수 거듭제곱을 **정수 계수 다항식
정규형**으로 완전히 정규화합니다.

```
표현   {단항식: 계수}.  단항식 = ((원자, 지수), ...) 정렬 튜플,  상수항 = ()
원자   ("sym", 이름)  또는  다항식으로 못 다루는 노드(div, mod, pow, 호출)
곱     다항식 곱셈 (분배법칙이 여기서 나온다)
지수   0..8 정수는 반복 곱으로 펼친다
```

`different` 의 의미를 지적대로 바꿨습니다: **"내 트리가 다르다" 가 아니라 "지원하는
의미론 안에서 같지 않음이 증명됐다"**.

### 심볼 지수도 `cannot_determine` 으로 내렸습니다

전에는 `n_h**n_kv` vs `n_h*n_h` 를 `different` 로 냈습니다. `n_kv == 2` 라는 **값 가정**이
있으면 같아지므로 같은 원칙에 따라 단정하지 않게 바꿨습니다.

```
다항식 밖의 연산 = 나눗셈 · 나머지 · ceil · roundup · 음수 지수 · 심볼 지수
  -> 정규형이 달라도 different 라고 말하지 않는다
```

`min`·`max` 는 인자를 정규화하므로 남겨 둡니다 — `min(a,b)` 가 `a` 와 항등적으로 같지는
않으니 `different` 가 건전합니다.

### 회귀시험에 넣었습니다

반례 넷 + 추가 일곱:

```
(n_h+n_kv)**3    vs  n_h**3+3*n_h**2*n_kv+3*n_h*n_kv**2+n_kv**3   same
(d_model-n_h)*(d_model+n_h)  vs  d_model**2-n_h**2                same
(n_h+1)*(n_h+2)  vs  n_h**2+3*n_h+2                               same
d_model*(n_h-n_h) vs 0                                            same
d_model+d_model-d_model  vs  d_model                              same
(n_h+n_kv)**2    vs  n_h**2+n_kv**2         different   (교차항이 빠졌다)
2*(d_model+n_h)  vs  2*d_model+n_h          different   (한쪽만 분배됐다)
n_h**9           vs  n_h 를 아홉 번 곱한 것  cannot_determine  (MAX_POW=8)
```

발행 라벨 132 종은 여전히 **전부 정규화 성공(실패 0)** 입니다.

## 2. 수집기가 fail-open 이었다 — 지적 전부 반영

```
accepted.jsonl    eligible_for_adjudication: true   **overlay 입력기는 이것만 읽는다**
rejected.jsonl    거부된 답 + format_errors + packet_errors
종료 코드          패킷을 **지정**하면 문제가 있으면 exit 1
                  (지정 없이 전체 점검할 때는 "답이 아직 없음" 만으로는 실패시키지 않는다)
```

검사를 이렇게 넓혔습니다.

| 지적 | 반영 |
|---|---|
| `confidence`·`assumptions`·`rejected_candidates` 존재·타입 | 필수 + 목록 타입 검사 |
| `source` 의 `file`·`lines`·`source_sha256`·`claim` 필수 | 넷 다 |
| `trace_or_metamorphic` 의 `artifact`·`claim` 필수 | 둘 다 |
| 필수 필드를 통과한 근거만 종류로 세기 | `good_kinds` 로 분리 — 빈껍데기 두 개로 요건을 못 채운다 |
| `no_name` 에도 근거 요건 | `NEEDS_EVIDENCE = (named, no_name)` |
| `_packet.json` 의 해시를 믿지 말고 실제 파일 재해시 | `verify_packet()` 이 다시 해시하고 **실제값**을 검사에 쓴다 |
| packet id·revision·`unit_order`·manifest 결합 재검증 | 같은 함수에서. `shard.md` 의 단위 순서까지 |
| 입력 해시 기록 | `_ingest.json` 의 `input_sha256` (대장·배정·answers 파일) |
| `submission_sha256` / `answer_revision` | `submission_sha256` 추가 |

`answer_id` 는 지적대로 재제출을 같은 단위로 묶는 키로만 남기고, **제출 내용의 해시**를
따로 붙였습니다.

### 자기검사로 발화를 증명했습니다

`test_ingest_answers.py` **28 → 59 항목**.

```
형식 문제를 하나씩 넣어 21 종이 실제로 잡히는가
fail-closed        거부된 답 1 + 유효한 답 1 -> accepted 1 / rejected 1 로 갈린다
종료 코드          문제 있으면 exit 1, 온전하면 exit 0, accepted·rejected 파일 생성
패킷 재검증        **source 를 한 줄 고치면 잡는다** (그 패킷의 답 전부 거부)
                  shard.md 순서를 뒤바꾸면 잡는다
                  _packet.json 의 packet_id·units 를 바꾸면 잡는다
```

## 3. alias 규칙을 배선했습니다 — 지적대로 범위를 강제합니다

`universe()` 가 `expr_compare.load_aliases()` 를 부르지 않아 **규칙 파일을 만들어도
수집 비교에 반영되지 않는** 상태였습니다. 맞습니다. 이제 합칩니다.

```
structure 의 symbols_label_only.expr   +   rules/label_aliases.yaml (모델·phase 범위)
```

`rules/label_aliases.yaml` 을 만들었고 **항목은 비워 뒀습니다**(`aliases: {}`) — 미리
채우면 그것이 곧 후보 목록이 되어 블라인드가 약해지기 때문입니다.

**전역 alias 를 금지했습니다.** 항목마다 `model`(필요하면 `phase`)이 없으면
`load_aliases()` 가 `ValueError` 로 거부합니다.

```
아래 셋은 전부 ValueError
  x: T/d_chunk                    (전역 문자열 맵)
  x: {expr: T}                    (model 없음)
  x: {model: m}                   (expr 없음)
파일이 없으면 빈 것이다 -- 추측하지 않는다
```

같은 이름이 모델마다 다른 뜻일 수 있다는 지적을 시험으로 고정했습니다: 같은 `n_chunk`
항목이 Kimi 에서는 쓰이고 gpt-oss 에서는 **쓰이지 않습니다.**

## 4. Q1–Q4 결정 반영 상태

| 결정 | 반영 |
|---|---|
| Q1 모든 등급에 **같은** 블라인드 질문지 | 지금 그렇습니다. grade 는 bundle 에 없고, 이후 우선순위에만 씁니다 |
| Q2 1 차 `no_name` 만으로 verdict 없음 | overlay 를 아직 안 만들었습니다. `pending_no_name_adjudication` 상태로만 둘 계획입니다. 자동 `demote` 없음 |
| Q3 제안은 버리지 않고 큐로, alias 는 근거 확인 후 등록 | 규칙 파일과 범위 강제까지 준비했고 **항목은 비어 있습니다** |
| Q4 순서 0(결함 수정) | **이 제출이 0 입니다.** 1 부터는 승인 후 |

## 5. 자기검사 전체

```
test_expr_compare          70/70     (52 -> 70)
test_ingest_answers        59/59     (28 -> 59)
test_packet_export         58/58
test_bundle_guards         63/63
test_provenance_fixture    22/22
test_segment_verdict        6/6      test_provenance_untouched  3/3
test_lowering_proof         6/6      test_tracer_alias          4/4
test_corrections_cover      5/5
                          296 항목
```

## 6. 읽을 곳

```
tracer 7c8eac1b
  develop/expr_compare.py          다항식 정규형 / 다항식 밖 연산 목록 / alias 범위
  develop/ingest_answers.py        verify_packet / accepted·rejected / 종료 코드
  develop/label_universe.py        aliases() 가 규칙 파일을 합친다
  rules/label_aliases.yaml         비어 있음 + 형식 문서
  develop/test_expr_compare.py     70 항목 (2-b 절이 반례 넷)
  develop/test_ingest_answers.py   59 항목 (4-b·4-c·4-d 절이 fail-closed)

실행해 볼 것
  .venv\Scripts\python.exe develop\expr_compare.py "2*(d_model+n_h)" "2*d_model+2*n_h"
      -> same
  .venv\Scripts\python.exe develop\expr_compare.py "n_h**n_kv" "n_h*n_h"
      -> cannot_determine
  .venv\Scripts\python.exe develop\label_universe.py
      -> 132 식 전부 정규화, 실패 0
```

산출물은 읽기만 하고 **수정하지 말아 달라.**

## 7. 원하는 판정

> 두 결함이 막혔는가. `different` 가 이제 **증명된 불일치**만 내는가 — 다항식으로 다루지
> 못하는 것을 빠뜨리지 않았는지 봐 주십시오(지금 목록: 나눗셈, 나머지, `ceil`, `roundup`,
> 음수 지수, 심볼 지수).
>
> 그리고 하나 확인받고 싶습니다. **`min`·`max` 는 `different` 를 허용하고 있습니다**
> (인자를 정렬해 정규화하므로 `min(a,b)` vs `min(b,a)` 는 `same`, `min(a,b)` vs `a` 는
> `different`). `min(a,b)` 가 `a` 와 항등적으로 같지 않으니 건전하다고 판단했는데,
> 축 라벨에서 `min`·`max` 가 쓰이는 방식을 보면 보수적으로 내려야 합니까?
>
> 승인되면 Q4 순서 1(raw overlay candidate 및 adjudication 상태 스키마 확정)로 갑니다.
