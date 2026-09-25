# 0-c 앞쪽 두 층: 식 동일성 비교와 답 수집 — overlay 전에 검토

요청일: 2026-09-25. 선행: `codex_ask_session_1to1.md`
(→ **1 차 파일럿 시작 승인 / 남은 차단 사유 없음 / 0-c 계속 승인**).

0-c 를 셋으로 나눠 **앞의 둘**을 만들었습니다. overlay 형식·파생·적용은 아직입니다 —
비교 규칙이 틀리면 overlay 전체가 틀리므로 먼저 검토받고 싶습니다.

```
tracer            7a9b0ed9
results-labeled   2bdd76b1   (파일럿 3 건 그대로, revision 7)
```

**아직 하지 않은 것:** overlay 형식, 파생·적용 스크립트, 실제 판정 세션, verdict 적용.

---

## 1. 승인된 Q4 규칙을 세 층으로 구현했습니다

`develop/expr_compare.py`

```
1 층  문자열 정규화     ·→* , −→- , × ⋅ ∗ →* , 공백, 겉 괄호
2 층  AST 대수 비교      심볼은 **원자**. 곱·합은 정렬된 다중집합, 나눗셈·거듭제곱·
                        호출은 **불투명 노드로 남긴다**(축약하지 않는다)
3 층  선언된 alias·유도   structure 의 `symbols_label_only.expr` + rules/label_aliases.yaml
```

판정은 `same` / `alias` / `different` / `cannot_determine` + 이유.

### 값을 근거로 쓸 통로를 **구조적으로** 없앴습니다

지정해 주신 "concrete 값만 같은 것은 동일 판정 근거가 아님" 을 주석이 아니라 형태로
지켰습니다.

```
compare(a, b, symbols, aliases)     <- namespace 를 받지 않는다
canonical(expr, symbols, aliases)   <- 같다
symbols 는 **이름 집합**일 뿐 값이 아니다
dim_expr 를 import 하지 않는다
```

자기검사가 `ns`/`namespace`/`concrete`/`values` 파라미터 부재와 import 부재를 확인합니다.
실증: `d_model` vs `d_moe` → **`different`** (이 모델에서 둘 다 8192 입니다).

### `/` 는 floor division 이므로 축약을 단정하지 않습니다

```
(n_h*d_head)/n_h  vs  d_head     ->  cannot_determine   나눗셈이 딱 맞을 때만 참이다
ceil(T/d_chunk)   vs  T/d_chunk  ->  cannot_determine
d_model%n_h       vs  0          ->  cannot_determine
n_h**-1           vs  1/n_h      ->  cannot_determine
d_model/n_h       vs  d_moe/n_h  ->  cannot_determine   나눗셈이 끼면 **다르다고도** 못 한다
```

### 자기검사가 제 오류 둘을 잡았습니다

```
n_h**2  vs  n_h*n_h    different 라고 했다 -> 틀렸다. 작은 정수 지수는 펼쳐야 한다
d_qk    vs  d_model    alias 를 펼쳐 양쪽이 정규화됐는데도 첫 패스의 "모르는 심볼"
                       때문에 cannot_determine 을 냈다 -- 확실히 다른 것까지 막혔다
```

반대로 `n_h**n_kv` vs `n_h*n_h` 는 **`different`** 로 둡니다 — 같아지려면 `n_kv == 2`
라는 **값 가정**이 필요하므로.

`test_expr_compare.py` **52 항목**.

## 2. 실제 발행 라벨로 검증했습니다

`develop/label_universe.py` — 심볼 **이름만** 모읍니다(값 격리).

```
모델                          식   정규화  실패   심볼  유도
DeepSeek-V4-Pro              40     40     0     59     0
Llama-4-Maverick-17B-128E    17     17     0     58     0
Kimi-K3                      35     35     0     59     1
gpt-oss-120b                 20     20     0     58     0
gpt-oss-20b                  20     20     0     58     0
```

**발행 라벨 식 132 종 전부 정규화 성공(실패 0).**

처음에는 49/132 가 "모르는 심볼" 이었습니다 — `B` 와 `T` 는 `structure.yaml` 의 `symbols`
가 아니라 **scope**(`batch`, `prefill_len`)에서 옵니다. 전체에 돌려 보고서야 알았습니다.
단위 시험만 했다면 못 봤을 종류입니다.

### 보수성의 실제 비용을 재 봤습니다 — 그리고 제 첫 수치가 오해를 줬습니다

모델 안에서 발행 라벨 식끼리 **모든 쌍**을 비교하면:

```
쌍 1,891   same 0   alias 0   different 1,483   cannot_determine 408  (21.6%)
식 132 종 중 나눗셈·ceil 이 든 것 12 종 (DeepSeek 의 T/m_csa, Kimi 의 T/d_chunk 등)
```

21.6% 는 커 보이지만 **실전 수치가 아닙니다.** 실제 비교는 "제안 vs 그 자리의 현재 라벨"
이므로, 판정 단위 2,898 개의 현재 라벨을 직접 셌습니다.

```
현재 라벨에 나눗셈이 든 판정 단위:  0 / 2,898
  1 단계 783    전부 순수 곱·합
  나머지 2,115  전부 순수 곱·합
```

즉 **floor division 보수성은 현재 판정 대상에서 비용이 0** 입니다. 검토자가 나눗셈이 든
식을 **제안할 때만** 발동합니다.

1 단계 783 단위의 현재 라벨은 16 종뿐입니다.

```
d_head_kda 243   d_chunk 180   n_h 80   n_h_kda 65   n_hc 52   T 40
d_nope 32   d_head 32   n_h_I 24   c_I 12   d_model 10   m_hca 4   d_moe 4   d_g 2
등급  scope_inferred 351   heuristic 248   open_tie 184
```

## 3. 답 수집 — 검증하고 지표만 냅니다

`develop/ingest_answers.py`. **적용하지 않습니다**(산출물은 `work/answers/` 뿐).

검사하는 것:

```
스키마          decision_unit_id·proposal 필수 / 모르는 proposal·confidence /
               named 인데 proposed_expr 없음 / 그 반대 / evidence 가 목록인가
근거            named 는 **두 종류 이상**(source + trace_or_metamorphic)
source          패킷에 있는 파일인가 / source_sha256 가 패킷과 일치하는가 / claim 이 있는가
정합            패킷에 없는 단위 / 답이 없는 단위 / 같은 단위 두 번 / JSON 이 아닌 줄
비교            expr_compare 로 same·alias·different·cannot_determine
```

지시하신 pilot 지표를 **위치별로** 냅니다(대장의 `unit_order` 로 위치를 압니다).

```
위치  답  named  no_name  cannot_det  형식오류  평균 근거
```

`answer_id = HMAC(salt, "answer:" + packet_id + ":" + unit_id)` — **답 내용을 넣지
않습니다.** 고쳐 내도 같은 id 라서 재제출을 추적할 수 있습니다.

### 자기검사가 잡은 결함

`named` 인데 `proposed_expr` 가 없는 **바로 그 오류 케이스**에서 수집기가 `KeyError` 로
죽었습니다. 검사해야 할 입력에 검사기가 죽으면 안 됩니다.

`test_ingest_answers.py` **28 항목** — 합성 답에 형식 문제를 **하나씩** 넣어 12 종이
실제로 잡히는지, 온전한 답은 문제 0 인지 확인합니다.

## 4. 자기검사 전체

```
test_expr_compare          52/52     <- 새로 만듦
test_ingest_answers        28/28     <- 새로 만듦
test_packet_export         58/58
test_bundle_guards         63/63
test_provenance_fixture    22/22
test_segment_verdict        6/6      test_provenance_untouched  3/3
test_lowering_proof         6/6      test_tracer_alias          4/4
test_corrections_cover      5/5
                          247 항목
```

## 5. 확인받고 싶은 것

### Q1. `scope_inferred` 351 개가 1 단계에 있습니다

1 단계 783 중 `scope_inferred` 가 351 개입니다(`heuristic` 248 + `open_tie` 184 = 432 는
위험 등급으로 전부 포함, 나머지는 family·cohort·영향 기준으로 들어왔습니다).
`scope_inferred` 는 모듈 정규식으로만 정한 것이어서 `heuristic` 보다는 낫지만
`confirmed` 는 아닙니다.

검토자에게는 등급을 알려 주지 않으므로 구분 없이 같은 질문을 받습니다. **이대로 맞습니까?**
아니면 `scope_inferred` 는 "현재 라벨을 확인해 달라" 는 성격이 달라서 다른 질문지가
필요합니까?

### Q2. `no_name` 을 어떻게 받아들일지

승인된 규칙은 `no_name` → `no_name_exists_candidate` → **반드시 2 차 반증 + 사람 판정**
입니다. 그런데 1 단계 대상은 전부 **이미 이름이 붙어 있는** 자리입니다(현재 라벨 16 종).
검토자가 `no_name` 을 내면 "지금 붙은 이름이 근거 없다" 는 주장이 됩니다.

이때 overlay 의 `verdict` 를 무엇으로 두어야 합니까?

* `demote`(라벨을 유지하되 grade 를 `unresolved` 로 내린다)
* `strip`(라벨을 지우고 정수로 되돌린다)
* 아니면 verdict 를 만들지 않고 사람 판정 대기 목록에만 넣는다

저는 세 번째가 맞다고 보는데(발행물을 되돌리는 것은 되돌리기 어려운 변경), 확인받고
싶습니다.

### Q3. 비교 판정이 `cannot_determine` 일 때

제안이 나눗셈을 담으면(`T/d_chunk` 같은) 현재 라벨과 비교가 `cannot_determine` 이 됩니다.
이때:

* 제안을 **버리지 않고** 사람 판정으로 올린다 (제 계획)
* 선언된 유도로 등록해 달라고 요구한 뒤 다시 비교한다
* 둘 다

`rules/label_aliases.yaml` 은 아직 **만들지 않았습니다**(없으면 빈 것으로 읽습니다).
검토자가 `n_chunk` 같은 유도명을 제안하면 그때 등록하는 흐름이 맞습니까, 아니면 미리
채워 두어야 합니까? 미리 채우면 그것이 곧 후보 목록이 되어 블라인드가 약해집니다.

### Q4. 다음 순서

이 순서로 갈 계획입니다.

```
1  raw_overlay 형식 (원장 사이트 튜플 키) + 스키마 검사
2  published_overlay 파생 -- crosswalk 의 raw_sites 가 전부
   (verdict, resulting_expr, grade_after, answer_id) 에서 일치해야 한다. 불일치는 conflict
3  적용 스크립트 -- **하드 게이트**. 조정 완료 표식이 없으면 dry-run 만
4  Q5 중복 조정 (일치 / 불일치→2차 / 3차 / 사람 판정, family·grade·model 별 불일치율)
```

맞습니까? 그리고 3 의 하드 게이트를 무엇으로 두는 것이 좋습니까 — 사람이 서명한 파일
(`work/adjudication/APPROVED.json` 에 revision·해시·판정 수)을 요구하는 방식을 생각했습니다.

## 6. 읽을 곳

```
tracer 7a9b0ed9
  develop/expr_compare.py        3 층 비교. namespace 를 받지 않는다
  develop/label_universe.py      심볼 이름만 (값 격리). B·T 는 scope 에서 온다
  develop/ingest_answers.py      스키마·근거·정합 검사 + 위치별 지표
  develop/test_expr_compare.py   52 항목
  develop/test_ingest_answers.py 28 항목

실행해 볼 것
  .venv\Scripts\python.exe develop\label_universe.py
      -> 132 식 전부 정규화, 실패 0
  .venv\Scripts\python.exe develop\expr_compare.py "(n_h*d_head)/n_h" "d_head"
      -> cannot_determine
```

산출물은 읽기만 하고 **수정하지 말아 달라.**

## 7. 원하는 판정

> 비교 규칙(1·2 절)과 답 수집(3 절)이 승인된 Q4 를 제대로 구현했는가.
> 특히 **floor division 을 축약하지 않는 것**과 **값을 근거로 쓸 통로가 없는 것**이
> 충분한가.
>
> 그리고 Q1(`scope_inferred` 351) · Q2(`no_name` verdict) · Q3(`cannot_determine` 처리와
> alias 등록 시점) · Q4(다음 순서와 적용 게이트)를 정해 주십시오.
>
> Q2·Q3 는 overlay 형식을 결정하므로 다음 작업의 모양을 바꿉니다.
