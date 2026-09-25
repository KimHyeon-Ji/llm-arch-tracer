# 0-b.1 재분할 결과 제출 — 완료 조건 전부 통과

요청일: 2026-09-25. 선행: `codex_ask_0b_units.md` (→ **0-c 보류, 0-b.1 재분할 필요**).

**지적 일곱 개를 전부 재현해 확인했고 수치가 정확히 일치했습니다.** 고친 결과를 제출합니다.
0-c 는 시작하지 않았습니다.

```
built_from_commit    (아래 7절 -- 이번에는 가드로 강제했습니다)
results-labeled      6d6a793a
```

---

## 1. 지적 검증 — 제 산출물로 직접 셌습니다

| 지적 | Codex 수치 | 제가 센 값 |
|---|---:|---:|
| 전체 I/W/O shape 가 단위 안에서 다름 | 26 | **26** (K3 prefill 12 / gpt-oss-120b 3+5 / 20b 2+4) |
| `layer_cohort` 혼합 | "다수" | **336** |
| concrete 문맥 구조 불일치 | 10 | **10** (V4 1+1 / K3 4+4) |
| family 전체 / 발행 영향 / 빠짐 | 79 / 49 / 30 | **79 / 49 / 30** |

## 2. 완료 조건 — 전부 통과

```
full_context_variants_per_unit        1      (variant > 1 인 단위 0)
layer_cohort_variants_per_unit        1      (variant > 1 인 단위 0)
review_packet_context_variants        1
concrete_context_schema_mismatch      0
concrete_context_value_mismatch       0      (평가 실패 자리 0)
blind_id_secret_dependency            0      (ID 가 비밀을 전혀 안 씀)
family registry                      79
published-impact families            49
raw-only backlog families            30
build tree clean                   true      (가드가 강제)
decision_unit_id 충돌                 0
```

추가 검사: **shard 242 개 / 단위 2,898 전부에서 대상 식 노출 0**, bundle 에 금지 자료 0,
후보 목록 0.

## 3. signature 를 보강했습니다

```
phase / block_type / layer_cohort_id / 상대 module path / op_type + raw_op /
field·shape_index·axis / 전체 input·weight·output shape / 정규화 parameter role /
이웃 (직전·직후의 op_type + 상대 모듈)
  + 묶기용으로만: old_expr, candidates, grade  (문서에는 싣지 않음)
```

`layer_cohort_id` 는 `layers` 문자열을 씁니다. absolute `op_occurrence` 는 버린 상태를
유지했습니다.

### 단위 수가 늘었습니다 — 480 → 2,898

| 모델 | phase | 질문 셀 | 0-b 단위 | **0-b.1 단위** |
|---|---|---:|---:|---:|
| Kimi-K3 | prefill | 65,248 | 124 | **1,632** |
| | decode | 416 | 52 | **416** |
| DeepSeek-V4-Pro | prefill / decode | 224 / 216 | 57 / 53 | **224 / 216** |
| gpt-oss-120b | prefill / decode | 107 / 107 | 47 / 53 | **107 / 107** |
| gpt-oss-20b | prefill / decode | 91 / 87 | 40 / 42 | **91 / 87** |
| Llama-4 | prefill / decode | 9 / 9 | 6 / 6 | **9 / 9** |
| **합계** | | 66,514 | 480 | **2,898** |

K3 prefill 외에는 대부분 **셀 1 개당 단위 1 개**가 됐습니다. 즉 이 모델들에서는 질문
셀마다 문맥이 실제로 다릅니다.

**Q1. 이 규모가 의도한 것인가?** shard 당 12 단위면 **242 shard** 이고, 1 차 + 10% 중복
+ 2 차 반증까지 하면 세션이 500 회 규모입니다. 발행 영향 집중도는 이렇습니다:

```
상위   10 단위 ->  3,300 셀 (  5.0%)      상위  200 단위 -> 54,760 셀 ( 82.3%)
상위   50 단위 -> 16,500 셀 ( 24.8%)      상위  400 단위 -> 62,896 셀 ( 94.6%)
상위  100 단위 -> 32,960 셀 ( 49.6%)      상위  800 단위 -> 64,416 셀 ( 96.8%)
```

**상위 200 단위(17 shard)가 발행 셀의 82%, 400 단위(34 shard)가 95% 를 덮습니다.**
1 단계를 영향 순으로 끊어 진행하는 것이 맞는가, 아니면 2,898 전부를 해야 하는가?
끊는다면 "나머지는 미판정" 을 `UNKNOWNS.md` 에 어떻게 적어야 오해가 없는가?

## 4. `decision_unit_id` 를 비밀과 분리했습니다

```
HMAC(private salt, sorted published cell keys)  ->  128 비트 (hex 32)
salt: work/_private/unit_id_salt.txt  -- .gitignore + bundle 제외
```

signature 를 전혀 쓰지 않으므로 후보를 대입해 역산할 것이 없습니다. `question_family_id`
에도 model·phase 를 넣고 충돌 검사했습니다(충돌 0).

## 5. review bundle 을 분리했습니다

`work/review_bundle/` — 1 차에게 주는 것은 **이것만**입니다.

**넣지 않은 것**: `work/units/` · `work/crosswalk/` · `work/_private/`(salt) ·
`_family_registry.jsonl` · `models/*.csv` · `models/*.jsonl` · 생성 코드

**넣은 것**: shard 242 개 / frozen source **15 파일** 사본 + SHA-256 **전체 64 자리** +
줄 수 / shard 에 등장한 모델의 심볼표·config·추적 범위 / `_manifest.json`(unit id, shuffle
seed, shard 별 candidate seed, 상태)

Kimi 는 지적대로 `naive.py` 만이 아니라 **remote `modeling_kimi_linear.py` ·
`configuration_kimi_k3.py` · `config.json` · 실행 shim `src/kda_shim.py` · `gate.py`** 를
함께 담았습니다.

### 1 차는 후보 없이 자유 제안입니다

승인된 운영 방식대로 후보 목록을 **없앴습니다**. 답 형식은 지정해 주신 JSON 스키마를
shard 머리에 넣었습니다(`proposal` / `proposed_expr` / `evidence` 2종 이상 /
`rejected_candidates` / `assumptions` / `confidence`).

### 가림 표기를 `X` 로 바꿨습니다

지적대로 `?tok` 대신 일관된 opaque variable 을 씁니다:

```
shape (가린 형태):
  입력   [B*T, X*d_model], [X*d_model, X]
  가중치  [X_same, X*d_model]
  출력   [B*T, X_same]

같은 자리의 concrete shape:
  입력   [6528, 28672], [28672, 4]
  가중치  [4, 28672]
  출력   [6528, 4]

이 축의 concrete 값: 4
```

**Q2. `X_same` 의 쓰임이 맞는가?** 지금은 "같은 식이 다른 자리에 또 나왔다" 를 뜻합니다.
그런데 지적하신 대로 **lineage 가 입증되지 않은 다른 출현**을 같은 것으로 표시하면
안 됩니다. 현재는 "현재 라벨 문자열이 같다" 만으로 `X_same` 을 붙입니다 — 이걸
별도 placeholder(`Y`, `Z` …)로 갈라야 하는가? 그러면 가림이 더 강해지지만 읽기는
어려워집니다.

## 6. concrete 문맥을 복사하지 않고 **평가해서** 만듭니다

발행 행의 각 식을 검증된 namespace 로 평가해 **발행 행과 같은 구조**로 새로 만듭니다.
합성 norm·canonical weight 에서 raw op 전체 shape 가 발행 행과 다른 문제가 사라졌습니다
(구조 불일치 10 → 0, 평가 실패 자리 0).

## 7. metadata — 같은 실수를 **구조로** 막았습니다

제출문과 파일이 다른 일이 **두 번** 있었습니다(1차 2c88c314↔935284c1, 2차
029a7fef↔e7613998 dirty). 두 번 다 **커밋 전에 생성**한 것이었습니다.

필드로 적는 것으로는 부족하다고 판단해 `develop/_buildguard.py` 를 만들고,
**`src/`·`develop/`·`rules/` 가 깨끗하지 않으면 생성 스크립트가 멈춥니다**(`SystemExit(2)`).
`models/` 변경은 생성 대상이므로 허용합니다. 검사용으로 뚫으려면 환경변수가 필요하고 그
경우 `dirty_build: true` 가 기록됩니다. 생성 스크립트 자신들의 SHA-256 도 함께 남깁니다.

**Q3. 이 가드가 적절한가?** 더 필요한 것이 있는가?

---

## 8. 0-c 로 가기 전 확인받고 싶은 것

### Q4. 1 차 답을 시스템이 비교하는 규칙

지정해 주신 대로 구현할 계획입니다:

* symbolic expression 이 **의미상 동일**할 때만 `current_correct`
* concrete 값만 같은 것은 동일 판정 근거가 아님 (`d_model == d_moe` 같은 우연)
* 다른 이름이면 `rename_candidate`
* 후보 밖 이름이면 자동 rename 금지 → 2차 반증 + 사람 판정
* `no_name` 은 `no_name_exists_candidate` → 반드시 2차 반증 + 사람 판정
* 불충분하면 `cannot_determine`

**"의미상 동일" 을 무엇으로 판정해야 하는가?** 문자열 일치는 너무 좁습니다
(`n_h*d_head` vs `n_h·d_head`, `T/d_chunk` vs `n_chunk`). 심볼 대수적 동일성으로
비교해야 하는가, 아니면 문자열 정규화까지만 하고 나머지는 사람 판정으로 올리는가?

### Q5. 10% 중복 배정의 일관성 측정

지시대로 약 10% 를 다른 1차 세션에 중복 배정할 계획입니다. **불일치가 나왔을 때** 무엇을
해야 하는가 — 둘 다 폐기하고 사람 판정인가, 아니면 3차를 붙이는가?

## 9. 읽을 곳

```
tracer  (7절의 가드 때문에 깨끗한 커밋에서 생성됨 -- _manifest.json 의 built_from_commit 참조)
  develop/_buildguard.py             더러운 트리에서 생성 금지
  develop/build_decision_units.py    signature 보강 + family registry
  develop/build_review_bundle.py     bundle 분리 + 가림 + shard

results-labeled 6d6a793a
  work/units/_units_report.json          단위·family 수
  work/units/_family_registry.jsonl      79 개 전부 (label 포함 -- bundle 밖)
  work/review_bundle/_manifest.json      shard·seed·source 해시·상태
  work/review_bundle/shards/shard001.md  1 차에게 주는 것
  work/review_bundle/source/             frozen source 15 파일
```

`work/review_bundle/` 밖의 자료는 1 차 판정 세션에 주지 않습니다.
산출물은 읽기만 하고 **수정하지 말아 달라.**

## 10. 원하는 판정

> 0-b.1 이 완료 조건을 채웠는가. 0-c 로 가도 되는가.
> 그리고 Q1(2,898 규모를 어떻게 끊을지)·Q2(`X_same`)·Q4(의미상 동일 판정)·Q5(불일치 처리)를
> 정해 달라. Q1 이 다음 작업의 크기를 결정합니다.
