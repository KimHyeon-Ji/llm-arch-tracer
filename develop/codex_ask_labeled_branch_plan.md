# 계획 검토 요청 — 규칙 대신 LLM 판단으로 라벨을 마무리하는 작업

요청일: 2026-09-24. **이번에는 산출물이 아니라 계획을 봐 달라.**
선행: `codex_answer_fleet5_artifacts.md`(2026-09-21, 5개 모델 산출물 검토).

---

## 1. 무엇을 하려는가

`llm-arch-tracer` 는 규칙 기반(모듈 경로 정규식, 유도식, 값 대조)으로 축 이름을 붙인다.
거기까지 와서 남은 것은 **규칙으로는 원리적으로 못 가리는 자리**다 — 후보들의 값이 같아
트레이스로 갈릴 수 없는 축들이다.

그래서 **도구를 더 키우지 않고**, LLM 이 모델 소스·논문·공식 자료를 직접 읽고 판단해
마무리하기로 했다. 목표는 도구 개선이 아니라 **이 5개 모델의 최종 csv/jsonl** 이다.

새 브랜치 `results-labeled` 를 `results` 에서 갈라 준비했다. 작업할 세션에 넘길 문서를
전부 써 뒀다. **그 문서와 계획이 적절한지 봐 달라.**

```
results          기계가 규칙으로 만든 판 (건드리지 않는 기준선)
results-labeled  LLM 판단으로 마무리한 판
-> 두 브랜치의 diff 가 "사람 판단이 바꾼 것"
```

## 2. 일의 크기 — 측정값

미확정 축은 수백만이지만, 같은 `(등급, 후보, 현재 라벨)` 은 질문 하나이고 **답 하나가
그 축 전부를 확정한다.** 원장(`*.axis_resolution.jsonl.gz`)의 `kind: question` 을 센 값:

```
                 질문(prefill/전체)   질문이 덮는 축(prefill)   전체 축 자리(prefill)
Kimi-K3               15 / 23            3,048,395              5,383,266
DeepSeek-V4-Pro       13 / 25               82,627                358,169
gpt-oss-120b           8 / 16                7,765                 34,663
gpt-oss-20b            7 / 13                4,873                 23,251
Llama-4-Maverick       1 /  2                  864                 37,015
합계                  44 / 79            3,144,524
```

Kimi-K3 의 상위 3개가 축 299만을 덮는다:

| 축 수 | 후보 | 현재 라벨 |
|---:|---|---|
| 1,164,168 | `n_h` \| `n_h_kda` \| `n_kv` | `n_h_kda` |
| 952,062 | `d_chunk` \| `d_rope` | `d_chunk` |
| 871,746 | `d_head_kda` \| `d_nope` \| `d_v` | `d_head_kda` |

## 3. 준비한 것

`results-labeled` 브랜치 (커밋 `90f40b87`):

```
README.md            브랜치 목적, 원칙, 금지사항, 기대치
README.results.md    산출물 자체의 설명 (열 의미·읽는 법·한계) -- 기존 results 문서
work/HANDOFF.md      작업 지시: 무엇을 바꾸고 무엇을 두는가, 답 형식, 판단의 규율 3가지
work/CONTEXT.md      함정 목록 (A~H)
work/SOURCES.md      고정 revision + 실행된 구현체 경로 + 외부 자료 허용 범위
work/questions/*.md  모델별 질문 목록 + 그 축이 실제로 사는 (모듈, 연산, field/si/axis, 예시 shape)
work/answers/        답을 쌓는 곳 (빈 디렉터리)
work/VERIFY.md       자기 검사 8종
work/REVIEW.md       검토 보고서 틀
```

### 3-1. 답의 형식

```yaml
- question_id: Kimi-K3-P-02
  current_label: d_chunk
  candidates: [d_chunk, d_rope]
  verdict: current_correct | rename | no_name_exists | cannot_determine
  new_label: d_chunk
  scope: 'self_attn$'
  source: |
    fla/ops/kda/naive.py:106-118 -- rearrange(x, 'b (n c) h d -> b h n c d', c=BT),
    BT = chunk_size = 64. 그 c 축이 이 자리다. d_rope 는 MLA 의 decoupled RoPE 폭이고
    KDA 경로에는 RoPE 자체가 없다(modeling_kimi_linear.py:504-529).
  confidence: high | medium | low
  affects_axes: 952062
```

### 3-2. 강조한 규율 세 가지

1. **값으로 고르지 말고 소스로 고를 것** -- 후보들이 값이 같아 올라온 질문이므로 산술은
   답을 못 낸다. `source` 에 file:line 필수.
2. **"이름이 없다" 도 정답** -- 과거에 KDA 청크 루프의 prefix 길이(값 10, 37)에
   `d_head-d_rope`, `d_head/2` 를 붙였다가 거둬들였다. 정수로 남는 것이 거짓 이름보다 낫다.
3. **범위를 좁게** -- "`d_chunk` 를 `d_rope` 로" 가 아니라 "`self_attn$` 안에서" 여야 한다.

### 3-3. `CONTEXT.md` 에 적은 함정 (A~H 요약)

| | 내용 |
|---|---|
| A | 등급 4종의 뜻과 `heuristic` 이 가장 위험한 이유 (K3 57,419 / V4-Pro 33,500) |
| B | 모델별 값 충돌 표. **단일 config 값만이 아니라** `c_I - d_rope = 64 = d_rope` 같은 유도식 충돌과 singleton/B 충돌도 있다고 명시 |
| C | 과거 실패 5종: 산술 우연 일치로 지어낸 이름, 스코프 과다 매칭, `d_head=74` 유령값, 배치 접기(`[E, B, d_moe]` 의 축 1 은 진짜 B), 계열 전이 금지(6건 중 1건만 맞았다) |
| D | 불변식 4종 (`n_h`/`n_kv` 공존 금지, 한 축에 이름 둘 금지, transpose 가 이름 못 바꿈, weight 에 B/T 금지) |
| E | 표 읽는 법: major-op 요약이라 7~18% 만 덮음 / `weight_pos` 는 역할 분류가 아님 / `block_type` 이 attention 종류를 숨김 / MoE 표현 3종 / `n_chunk` / `mla_use_nope=true` |
| F | **원장을 발행 csv 에 `op_id` 로 조인할 수 없다** -- 실측 7,837 중 1개만 이어진다(원시 트레이스 번호 vs 접은 뒤 재번호) |
| G | 이미 통과한 검사 목록 (다시 안 해도 됨) + 당신의 2026-09-21 검토 범위를 인용 |
| H | 아직 아무도 안 한 것 = 이 작업 |

## 4. 봐 줬으면 하는 것

### Q1. 이 접근 자체가 맞는가

규칙 엔진을 더 키우지 않고 LLM 판단으로 마무리하는 것이 이 남은 자리들에 적절한가?
아니면 규칙으로 더 갈 수 있는데 포기한 것인가? **포기하면 안 되는 자리**가 있다면 짚어 달라.

### Q2. 인계 문서가 실패하는 경로

작업할 세션이 이 문서를 읽고도 **틀릴 수 있는 경로**가 보이는가. 특히:

* `CONTEXT.md` 의 함정 목록에 **빠진 함정**이 있는가? 당신이 2026-09-21 검토에서 본 것 중
  이 목록에 없는 것이 있으면 그게 가장 중요한 지적이다.
* 답 형식(`verdict` 4종 + `scope` + `source`)이 **잘못된 판단을 담을 수 있는** 구조인가?
  예를 들어 `scope` 를 넓게 써서 맞는 자리까지 바꾸는 것을 이 형식이 막는가?
* `VERIFY.md` 의 검사 8종이 **실제로 무엇을 못 잡는가?**

### Q3. 질문 목록(`work/questions/*.md`)이 충분한 자료인가

각 질문에 `(모듈 경로, 연산, field/shape_index/axis, 예시 shape)` 를 최대 8자리까지
붙였다. 층 번호는 `.layers.*.` 로 묶었다.

* 이걸로 그 축이 **무엇인지 판단할 수 있는가?** 부족하면 무엇을 더 넣어야 하는가?
* 8자리로 자른 것이 **판단을 왜곡할** 위험이 있는가? (예: 잘린 자리에 결정적 단서가 있는 경우)

### Q4. 검증 설계 — 받은 답을 어떻게 믿을 것인가

지금 계획은 (a) `source` 의 file:line 을 사람이 열어 확인, (b) 산술 일치, (c) 불변식
재검사, (d) 같은 질문을 독립된 두 세션에 주고 갈리면 표시 — 넷이다.

* 이 넷으로 충분한가? **자기 확신에 찬 오답**을 무엇이 잡는가?
* `confidence: high` 인데 틀린 답을 걸러낼 방법이 있는가?
* 두 세션 교차 대조가 **같은 편향을 공유**해서 둘 다 같은 오답을 낼 위험은?

### Q5. 이 작업이 끝나면 무엇을 주장할 수 있나

`HANDOFF.md` 에 "대부분 '현재가 맞다' 로 끝날 것이고, 달라지는 것은 '규칙이 추측한
이름' → '소스로 확인된 이름' 이다" 라고 적었다.

* 이 기대치가 맞는가?
* 작업 후 이 산출물에 대해 **정당하게 주장할 수 있는 것**과 **주장하면 안 되는 것**의
  경계는 어디인가? (사용자는 이 표의 심볼에 다른 값을 대입해 roofline·batch sweep 을 할
  계획이다 -- 그 용도에 대해 무엇을 보장할 수 있는가?)

### Q6. 질문 79개 밖의 자리

질문은 **후보가 둘 이상인 자리**만 덮는다. 남는 것:

* `unresolved` (K3 447,947 / V4-Pro 5,777 …) -- 후보가 아예 없어 정수로 둔 자리.
  이름을 새로 만들어야 한다.
* `heuristic` (K3 57,419 / V4-Pro 33,500 …) -- 지어낸 이름. 확인하거나 거둬야 한다.

지금 계획은 **1단계로 질문 79개만** 하고, 결과를 보고 2단계를 정하기로 했다.
이 단계 구분이 맞는가? `heuristic` 을 1단계에 넣어야 하는가?

## 5. 원하는 답의 형태

* **계획의 결함**: 무엇이 어떻게 실패하는가, 어떻게 고치는가
* **빠진 함정**: `CONTEXT.md` 에 없는데 있어야 하는 것 (파일·줄 근거와 함께)
* **문서 문장의 오류**: 내가 사실을 잘못 적은 곳
* 적절하면 **적절하다고** 적어 달라 -- 과다 설계도 비용이다

`results-labeled` 브랜치는 읽기만 하고 **수정하지 말아 달라.** 작업은 별도 세션이 한다.

## 6. 읽을 곳

```
브랜치: results-labeled  (커밋 90f40b87, results@85e33274 에서 분기)
워크트리: ../llm-arch-tracer-results-labeled/

work/HANDOFF.md  work/CONTEXT.md  work/SOURCES.md  work/VERIFY.md  work/REVIEW.md
work/questions/moonshotai__Kimi-K3.md          (가장 큰 것 -- 여기부터)
work/questions/deepseek-ai__DeepSeek-V4-Pro.md
README.md
```
