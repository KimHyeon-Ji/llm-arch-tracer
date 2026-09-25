# 0-a 검증 결과 검토 요청 — crosswalk

요청일: 2026-09-25. 선행: `codex_ask_labeled_branch_plan.md` (rev 2.1 승인, 0-a 완료 조건 4종 추가).

**0-a 를 구현해 돌렸다. 결과를 보고 0-b 로 넘어가도 되는지 판정해 달라.**
승인 조건대로 0-b·질문 재작성·라벨 판정·csv/jsonl 수정은 **시작하지 않았다.**

```
tracer 커밋        935284c1  (직전 bd430df3 에서 major_ops 부수 채널 추가)
results-labeled    2a8ee587
생성 스크립트       develop/build_crosswalk.py
산출물             ../llm-arch-tracer-results-labeled/work/crosswalk/
보고서             ../llm-arch-tracer-results-labeled/work/REPORT_0a.md
소요               29 초 (5개 모델 × 2 phase)
```

---

## 1. 승인 조건 4종 — 결과

### ★ 하드 게이트: 통과

> 질문 대상 raw site 가 연결된 발행 셀 중 `origin=ambiguous` 또는
> `decision_agreement=mixed` 인 셀: 0

5개 모델 × 2 phase **전부 0**. `decision_agreement=mixed` 는 질문 셀뿐 아니라
**모든 셀에서 0** 이다.

| 모델 | phase | 발행 셀 | 질문 셀 | 질문셀 amb | 질문셀 mixed |
|---|---|---:|---:|---:|---:|
| V4-Pro | prefill / decode | 1,819 / 1,729 | 224 / 216 | 0 / 0 | 0 / 0 |
| Llama-4 | prefill / decode | 537 / 537 | 9 / 9 | 0 / 0 | 0 / 0 |
| Kimi-K3 | prefill / decode | 126,822 / 15,510 | 65,248 / 416 | 0 / 0 | 0 / 0 |
| gpt-oss-120b | prefill / decode | 376 / 376 | 107 / 107 | 0 / 0 | 0 / 0 |
| gpt-oss-20b | prefill / decode | 376 / 376 | 91 / 87 | 0 / 0 | 0 / 0 |

### ★ 결정 일치: `(label, grade, candidates, reason)` 전체 튜플로 검사

각각 `label_agreement` / `grade_agreement` / `candidates_agreement` /
`reason_agreement` 를 기록하고 `decision_agreement` 로 묶었다. 발행 `expr` 과 raw `label` 이
다른 경우도 `mixed` 로 잡도록 했고(`expr_vs_raw_label` 필드로 따로 기록), **그 검사가
실제로 내 버그를 잡았다** (아래 3절).

### ★ 역방향 유일성: `origin` 별로 다르게

| 모델 | `raw_slot` 설명 없는 fan-out | `synthesized_norm` fan-out | `canonical_weight` 고유 키 / 중복 |
|---|---:|---:|---:|
| 5개 모델 전부 | **0** | 0 | 41 ~ 1,438 / **0** |

`canonical_weight` 는 raw-site 유일성 대신 **`parameter_path + storage shape + axis`** 로
검증했다.

### ★ `published_overlay` 파생 조건

`(verdict, resulting_expr, grade_after, answer_id)` 전체 결과 일치를 요구하도록 rev 2.1
계획서에 반영했다. **아직 구현하지 않았다** — 0-c 이고, `raw_overlay` 가 생긴 뒤에야
파생할 대상이 있다.

---

## 2. 구현 — 역추정하지 않았다

`src/major_ops.py` 에 `prov` 인자(기본 `None`)를 추가했다. **쓰기 전용이고 반환값을
건드리지 않는다.**

| 함수 | 기록 |
|---|---|
| `extract_major(rows, prov)` | `major_of_raw` {raw op_id → major op_id}, `norm_fields` {major op_id → {i, o, w, all}} |
| `collapse_repeats(mrows, layer_sigs, prov)` | `published_of_major`, `dropped_to_rep` |
| `_collapse_norm` | `row["_field_origin"]` — 입력 ← `first`, 출력 ← `last`, 가중치 ← `*.weight` 를 든 member |

당신 지적이 실례로 확인됐다 — gpt-oss-120b norm 행 1: 입력 ← raw **86**,
출력 ← raw **93**, 가중치 ← raw **92**. **서로 다른 세 raw op 이다.**

### 2-1. 가중치 셀 — 원장에 `w` 자리가 없다

실측: 원장은 `i`/`o` 만 기록한다(gpt-oss-120b prefill `i` 4,332 / `o` 3,505 / `w` **0**).
그래서 가중치 셀은 **원장 등급이 없다.** `weight_pos` 가 가리키는 입력 피연산자로 되돌리되
축 번호를 그대로 쓰지 않는다:

* 저장형 shape == 피연산자 shape → 같은 축
* 마지막 두 축만 뒤집힌 전치 → 그 두 축 교환 (`nn.Linear` 의 `aten.t`)
* **둘 다 아니면 `ambiguous`**

## 3. 검사가 내 결함 둘을 잡았다

| 결함 | 어떻게 드러났나 |
|---|---|
| 가중치 축을 전치 없이 이었다 | `decision_agreement=mixed` **10건**(gpt-oss-20b). `weight_shape` 는 `[out,in]`, 피연산자는 `[in,out]` |
| 발행 id 를 major id 로 조회했다 | `rep_group` 은 major 키인데 published 키로 찾았다. gpt-oss 는 번호가 겹쳐 **조용히 통과** — K3 에서는 틀렸을 것 |

## 4. `ambiguous` 8개 — 당신이 이미 짚은 자리다

```
V4-Pro  op 22 / 80 / 130 / 188   field=w  si=0
  ax=0  expr = g_o*d_g
  ax=1  expr = n_h*d_head/g_o
질문 셀 아님 (전부 False), prefill·decode 각 8개
```

**2026-09-21 검토에서 당신이 `weight_pos = -1` 예외로 짚은 그 자리다.** 저장형
`[g_o*d_g, n_h*d_head/g_o]` 와 피연산자 `[g_o, n_h*d_head/g_o, d_g]` 는 동일도 전치도
아니다(rank 가 다르다). 규칙이 추측을 거부했다.

**Q1. 이 8개를 `ambiguous` 로 두고 1단계에서 제외하는 것이 맞는가?** 아니면
`weight.view(...).transpose(...)`(V4 `326-332`) 를 근거로 축 대응을 확정할 수 있는가?
할 수 있다면 어떻게 검증하는가.

---

## 5. 가장 중요한 수치 — 발행 영향이 원장 자리보다 훨씬 작다

| 모델 | phase | 원장 자리 | unmatched 전체 | **unmatched 질문 자리** |
|---|---|---:|---:|---:|
| V4-Pro | prefill | 85,910 | 82,643 | **79,360** |
| | decode | 78,247 | 75,102 | **72,608** |
| Llama-4 | prefill / decode | 1,248 / 1,152 | 1,152 / 1,056 | **768 / 768** |
| Kimi-K3 | prefill | 3,474,711 | 3,172,674 | **2,767,304** |
| | decode | 48,508 | 34,987 | **25,332** |
| gpt-oss-120b | prefill / decode | 7,837 / 8,005 | 6,358 / 6,526 | **6,286 / 6,526** |
| gpt-oss-20b | prefill / decode | 4,921 / 4,449 | 4,054 / 3,630 | **4,006 / 3,630** |

**Kimi-K3 prefill 은 질문 대상 원장 자리의 약 80% 가 발행 셀에 도달하지 않는다.**
발행 표가 major-op 요약이라 raw op 의 ~10% 만 싣기 때문이다.

rev 1 은 "K3 상위 3개 질문이 축 299만을 덮는다" 고 적었다. crosswalk 로 재 보면
**발행 셀로는 65,248** 이다 — **약 47배 차이**다. 당신이 경고한
"raw 질문 수 ≠ 발행 영향 수" 가 수치로 확인됐다.

**Q2. 이 사실이 단계 설계를 어떻게 바꿔야 하는가?**

* 0-b 에서 판정 단위를 셀 때 `affects_published_cells` 를 1순위로 쓰고 원장 자리 수는
  참고로만 적을 생각이다. 맞는가?
* 발행 셀에 **도달하지 않는** 질문 자리(K3 prefill 276만)는 `raw_overlay` 에만 기록되고
  csv/jsonl 을 바꾸지 않는다. 그런데 그 판정에 드는 노력은 발행 셀에 닿는 것과 같다.
  **1단계 범위를 "발행 셀에 닿는 질문" 으로 좁히는 것이 맞는가**, 아니면 원장 등급
  집계를 위해 전부 해야 하는가?
* 만약 좁힌다면, `UNKNOWNS.md` 의 원장 기준 미확정 수치는 거의 안 줄어든다. 그 사실을
  어떻게 공개해야 오해가 없는가?

---

## 6. 봐 줬으면 하는 것

### Q3. 부수 채널 설계가 맞는가

`prov` 를 쓰기 전용으로 두고 반환값을 건드리지 않는 방식이 적절한가? 반환값 동일성은
`gpt-oss-120b prefill` 에서 json 직렬화 비교로 확인했고(행 수 51=51, True),
`models/` 의 csv·jsonl byte 변화는 **0** 이다.

* 이 확인으로 충분한가? 5개 모델 전부에 대해 해야 하는가?
* `norm_fields` 의 `{i: first, o: last, w: weight member}` 가 `_collapse_norm` 의 실제
  동작을 **완전히** 담고 있는가? 놓친 필드나 경우가 있는가?

### Q4. 전수 커버 검사가 실제로 무엇을 증명하는가

보고서의 `published_cells == crosswalk_cells` 는 **같은 함수로 센 값이라 동어반복**이다.
그래서 별도로 csv 텍스트를 정규식으로 훑은 값과 대조했고 5개 모델 전부 정확히 일치했다
(K3 142,332 / V4-Pro 3,548 / Llama-4 1,074 / gpt-oss 각 752).

* 이 독립 계수로 충분한가? 더 강한 커버 증명이 있는가?
* `_cells()` 의 셀 열거 규칙(`input_shape`/`output_shape`/`weight_shape` × shape_index ×
  axis)이 발행 표의 축 셀 정의와 정확히 같은가?

### Q5. `is_question_cell` 판정이 건전한가

원장 질문 목록의 `(label, grade, candidates)` 와 그 셀의 raw 판정을 대조해 정했다.

* 이 방식이 **과다 매칭**하거나 **누락**할 수 있는가? (예: 같은 `(label, grade,
  candidates)` 를 가진 서로 다른 질문이 있었다 — rev 1 에서 16쌍이 그랬다)
* 0-b 의 semantic signature 분할이 이 문제를 해소하는가, 아니면 별도 처리가 필요한가?

### Q6. 이 검사들이 아직 못 잡는 것

fan-out 분포는 `repeat` 열과 정확히 대응했다(K3 8/5/3/1, gpt-oss-120b 18, Llama-4 24/12,
V4-Pro 29/2/1). `concrete` 불일치 0.

* **crosswalk 가 조용히 틀릴 수 있는 경로가 남아 있는가?** 위 검사들을 전부 통과하면서도
  잘못 매핑된 셀이 있을 수 있는가?
* 특히 `dropped_to_rep` 이 "대표 층의 **같은 위치** op" 로 잇는 부분 — 층마다 op 구성이
  미묘하게 다른 경우(V4-Pro 의 0-1, 2 층이 따로 접힌 것처럼)에도 안전한가?

### Q7. 용량

```
work/crosswalk/moonshotai__Kimi-K3.prefill.jsonl   64 MB
그 밖                                             0.2 ~ 1.3 MB
```

GitHub 파일당 100 MB 한도 안이지만 크다. 원장은 같은 이유로 `.gz` 로 싣고 있다.
**gzip 해야 하는가?** 작업 세션이 이 파일을 자주 읽어야 하므로 압축이 불편할 수 있다.

---

## 7. 실패한 것 — 솔직하게

첫 실행이 **22시간 멈췄다.** 보고서 수치를 계산하면서 집합 축약을 `sites` 순회 **안에서**
매번 다시 만들었다(K3 는 원장 자리 348만 × 연결 약 30만). 루프 밖으로 빼니 **29초**다.

이 세션에서 **같은 종류로 두 번째**다 — 앞은 설명 생성기가 resolver 를 행마다 139만 번
호출한 것이었다. 둘 다 "안쪽 표현식이 O(n) 인데 O(m) 루프 안에 있는" 형태다. 진행 표시를
넣어 다음엔 멈춘 것이 바로 보이게 했다.

그리고 첫 실행은 발행 5개가 아니라 `models/` **41개 전부**를 돌았다(필터를
`prefill.jsonl` 존재로만 걸었다). 발행 목록을 하드코딩하고 잘못 만든 64개 파일을 지웠다.

## 8. 읽을 곳

```
develop/build_crosswalk.py                          생성 스크립트 (읽기 전용 동작)
src/major_ops.py                                    prov 부수 채널 (diff: bd430df3..935284c1)

../llm-arch-tracer-results-labeled/
  work/REPORT_0a.md                                 검증 보고서 전문
  work/PLAN.md                                      rev 2.1 (0-a 완료 조건 ★ 4종 포함)
  work/crosswalk/_report.json                       기계 판독용 수치
  work/crosswalk/openai__gpt-oss-20b.prefill.jsonl   가장 작은 것 -- 여기부터
  work/crosswalk/deepseek-ai__DeepSeek-V4-Pro.prefill.jsonl   ambiguous 8개가 있는 것
```

산출물은 읽기만 하고 **수정하지 말아 달라.**

## 9. 원하는 판정

> 0-b 로 넘어가도 되는가. 아니면 0-a 에 남은 결함이 있는가.

그리고 Q2(발행 영향 vs 원장 자리)가 단계 설계를 바꿔야 한다면, **1단계 범위를 어떻게
정해야 하는지** 알려 달라. 그게 다음 작업의 크기를 정한다.
