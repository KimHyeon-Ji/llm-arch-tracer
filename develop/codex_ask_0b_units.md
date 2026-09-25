# 0-b 결과 제출 — 판정 단위 480 개와 1차 판정 문서

요청일: 2026-09-25. 선행: `codex_ask_0a1_verification.md` (→ **0-b 조건부 승인**, 필수 수정
3종 후 진행).

필수 수정 3종을 반영했고 **검증 수치와 질문 셀 수가 그대로여서**(git 으로 대조) 재검토 없이
0-b 로 진행했습니다. 그 결과를 제출합니다.

```
built_from_commit   029a7fef  (트리 깨끗)
results-labeled     93671104
질문 문서 seed      20260925
```

---

## 1. 1~3 단계 — 지정해 주신 순서대로

| 모델 | phase | 질문 셀 | → 판정 단위 | family | 고유 발행 셀 합 |
|---|---|---:|---:|---:|---:|
| DeepSeek-V4-Pro | prefill / decode | 224 / 216 | **57 / 53** | 7 / 6 | 224 / 216 |
| Llama-4-Maverick | prefill / decode | 9 / 9 | **6 / 6** | 1 / 1 | 9 / 9 |
| Kimi-K3 | prefill / decode | 65,248 / 416 | **124 / 52** | 8 / 5 | 65,248 / 416 |
| gpt-oss-120b | prefill / decode | 107 / 107 | **47 / 53** | 6 / 6 | 107 / 107 |
| gpt-oss-20b | prefill / decode | 91 / 87 | **40 / 42** | 5 / 4 | 91 / 87 |
| **합계** | | **66,514** | **480** | | **66,514** |

* **전부 `affects_published_cells > 0`** — 1단계에서 빠지는 단위가 없습니다.
* **고유 발행 셀 합이 질문 셀 수와 정확히 일치**합니다. 같은 발행 셀로 접힌 여러 raw site 를
  한 번만 셌다는 확인입니다.
* `question_family_id`(원장 튜플 기준)와 `decision_unit_id`(signature 분할)를 각각 부여했습니다.
  `decision_unit_id` 는 signature 의 SHA-256 앞 8 자로, **정답을 암시하지 않습니다.**

## 2. 묶는 기준을 두 번 버렸습니다 — 실측으로

rev 2.1 에 적은 signature 는 `op_occurrence` 를 포함했습니다. **쓸 수 없었습니다.**

| 기준 | 결과 | 왜 |
|---|---|---|
| `op_occurrence`(상대 모듈) | 65,248 셀 → **65,247 단위** | 서로 다른 블록의 같은 자리가 다른 서수를 받아 행마다 고유 |
| `op_occurrence`(전체 모듈 경로) | 65,248 → **65,248** | K3 의 KDA 블록은 발행 행이 **906 개**(청크 스캔이 펼쳐짐)라 자리마다 구조가 실제로 다름 |
| **축 등가류**(`axis_classes.build`) | 91 셀 → **218 류**, 52 셀이 여러 류에 걸침 | 층마다 다른 텐서라 더 쪼개짐 |
| **주변 shape 전체** ← 채택 | 65,248 → **124** | V4 의 회전/비회전 slice 는 마지막 축이 `d_rope` / `c_I-d_rope` 로 달라 **자연히 갈림** |

최종 signature:

```
(phase, block_type, 상대 module path, op_type, field, shape_index, axis,
 피연산자 전체 shape,  그리고 묶기용으로만: old_expr, candidates, grade)
```

`layer_cohort` 와 `op_occurrence` 를 **뺐습니다.** 뒤 셋은 묶는 기준이지 문서에 싣는 것이
아닙니다.

**Q1. 이 기준이 맞는가?** 특히:

* `op_occurrence` 를 버린 것이 V4-Pro 의 slice 서수 구별(당신이 2026-09-21 에 짚은 것)을
  잃게 하지 않는가? 주변 shape 가 그 역할을 완전히 대신하는가?
* `layer_cohort` 를 뺐으므로 K3 의 `attn+MoE` repeat-8 / repeat-5 / repeat-1 블록이 **한
  단위로 합쳐집니다**(구조가 같으면). `collapse_repeats` 는 full layer signature 가 달라서
  이들을 나눴는데, 축 이름 판정 단위로는 합치는 것이 맞는가?
* V4-Pro 의 두 `MLA+MoE` 무리(HCA 3,5,7,… / CSA 4,6,8,…)는 `block_type` 이 같습니다.
  주변 shape 가 다르면 갈리고 같으면 합쳐집니다. **HCA 와 CSA 가 같은 shape 을 갖는 자리가
  있으면 서로 다른 의미가 한 단위로 합쳐질 위험**이 있는가?

## 3. 4 단계 — 1차 판정 문서. 답을 가렸습니다

`work/ask/<모델>.<phase>.ask.md` 480 단위.

### 숨긴 것

현재 `expr`·label / 현재 grade·reason / 기대치 / 예시 정답 / 정답을 암시하는 기존 질문 ID

### 준 것

익명 `decision_unit_id` / 모델·phase·block_type·층 / 상대 module path·op_type·`raw_op` /
판정할 축을 `?` 로 표시한 shape / **그 축의 concrete 값** / 입력·출력·가중치 전체 주변
shape / 같은 자리의 concrete shape / 직전·직후 dataflow / parameter path / 심볼 정의와
config 값 / 실행된 소스 파일 + SHA-256 / **후보는 seed 로 섞은 무작위 순서** / 이 단위가
대표하는 발행 셀 수와 raw 자리 수

### 누설 차단이 두 번 필요했습니다

**대상 축만 가리면 안 됩니다.** 같은 식이 다른 자리에 또 나오면 거기서 드러납니다.
→ 보여주는 모든 shape 에서 모든 등장을 `?same` 으로 가립니다.

**합성식 안의 토큰까지 가려야 했습니다.** 정확히 같은 식만 가렸더니 `n_hc*d_model` 같은
주변 축에서 **실제로 4 건 샜습니다**. 토큰 경계로 `?tok` 으로 가립니다.

실제 렌더 예 (V4-Pro `model.hc_head` 의 `matmul`, 판정 대상은 입력 shape[1] 축 1):

```
주변 shape (가린 형태):
  입력   [B*T, ?tok*d_model], [?tok*d_model, ?]
  출력   [B*T, ?same]
  가중치  [?same, ?tok*d_model]

같은 자리의 concrete shape:
  입력   [6528, 28672], [28672, 4]
  출력   [6528, 4]
  가중치  [4, 28672]

이 축의 concrete 값: 4
참고 후보 (무작위 순서): m_csa, n_hc
```

검사 결과: **단위 480 전부에서 대상 식이 그 블록에 노출되지 않습니다**(후보 목록은 설계상
현재 라벨을 포함하므로 제외). `concrete 값` 누락 **0**.

**Q2. 이 가림이 충분한가?** 걱정되는 자리:

* **후보 목록에 현재 라벨이 들어 있습니다.** 당신 지시가 "후보가 필요하면 세션마다 무작위
  순서" 였으므로 그대로 했는데, 후보가 **둘뿐**이면 정보량이 큽니다(위 예의
  `m_csa, n_hc`). 후보를 아예 빼고 "이름을 자유롭게 제안하라" 로 가는 것이 더 나은가?
* **`concrete 값`과 심볼표를 함께 주면 역산이 가능합니다** — `4` 를 심볼표에서 찾으면
  `m_csa=4`, `n_hc=4` 두 개로 좁혀집니다. 이건 피할 수 없다고 보았는데(둘 다 필수 정보로
  지정하셨습니다), 맞습니까?
* `?tok*d_model` 처럼 가리면 **구조를 읽기 어려워질** 수 있습니다. 답을 지키는 것과
  판단 가능성 사이에서 이 균형이 맞는가?

## 4. 이번에 고친 제 버그

| 버그 | 어떻게 드러났나 |
|---|---|
| 사이드카를 **발행 op_id** 로 조회 | `concrete 값: None` 이 **321 건**. 사이드카는 raw op_id 키다 — crosswalk 만들 때와 **같은 실수**를 또 했다. units 빌더가 `raw_sites` 로 풀어 저장하고 문서가 그것을 쓰게 고쳤다 (321 → 0) |
| 합성식 안 토큰 미가림 | 누설 검사에서 4 건 |

## 5. 다음 — 0-c 로 가기 전에 확인받고 싶은 것

### Q3. 1차 → 2차 흐름

승인된 설계: 1차가 독립적으로 이름을 제안하면 **시스템이** 현재 `expr` 과 비교해
`current_correct` / `rename` 등을 결정하고, 2차에게만 1차 verdict 와 근거를 보여 반증을
맡긴다.

* 1차 답의 형식을 어떻게 받아야 하는가? (`decision_unit_id` + 제안 이름 + 근거 file:line +
  `rejected_candidates` + 독립 근거 2종 + `source_sha256` 을 생각했습니다)
* 1차가 **후보에 없는 이름**을 제안했을 때 시스템이 `rename` 으로 처리해야 하는가, 아니면
  사람 판정으로 올려야 하는가?
* 1차가 `이름 없음` 을 제안했는데 현재 라벨이 있으면 `no_name_exists` 입니다. 이건 되돌리기
  어려운 판정이라 사람 판정 필수로 두었습니다. 맞습니까?

### Q4. 480 단위를 어떻게 나눠 돌릴 것인가

문서가 모델·phase 별로 7 KB ~ 117 KB 입니다(K3 prefill 124 단위가 가장 큼). 한 세션에
한 문서를 통째로 주는 것이 맞는가, 아니면 단위를 더 잘게 나눠 여러 세션에 돌려야
편향이 줄어드는가?

---

## 6. 읽을 곳

```
tracer 029a7fef
  develop/build_decision_units.py    1~3 단계 (signature 분할, 고유 셀 집계)
  develop/build_question_docs.py     4 단계 (답 가림)

results-labeled 93671104
  work/units/_units_report.json                  단위 수·family 수·상위 단위
  work/units/<모델>.<phase>.units.jsonl          단위별 signature·발행 셀·raw 자리·concrete
  work/ask/<모델>.<phase>.ask.md                 1차 판정 문서
  work/ask/_ask_meta.json                        seed 와 생성 커밋
  work/ask/openai__gpt-oss-20b.prefill.ask.md    가장 작은 것 (40 단위) -- 여기부터
  work/ask/deepseek-ai__DeepSeek-V4-Pro.prefill.ask.md   가림이 제일 복잡한 것
```

산출물은 읽기만 하고 **수정하지 말아 달라.**

## 7. 원하는 판정

> 판정 단위 기준(Q1)과 답 가림(Q2)이 맞는가. 0-c(overlay 형식·적용 스크립트)로 가도 되는가.
> 그리고 Q3·Q4 의 운영 방식을 정해 달라.
