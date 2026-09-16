# 확실하지 않은 것 -- meta-llama__Llama-4-Maverick-17B-128E

이 파일은 **이 산출물에서 검증되지 않은 부분 전부**를 모은 것이다. 여기 없는 것은 규칙이 결정했고 게이트가 확인했다.

## 1. 축 이름 판정

축 자리 **73,550개** 중 확정 **72,878개 (99.1%)**, 미확정 **672개 (0.9%)**.

| 등급 | 자리 | 무엇을 믿어도 되나 |
|---|---:|---|
| `unresolved` | 672 | 이름 붙일 근거가 없어 정수로 뒀다. 주장을 안 하므로 틀릴 것도 없다 |

### 아직 안 푼 질문

같은 `(등급, 후보, 현재 라벨)` 은 질문 하나다 -- 답 하나가 축 수천 개를 확정시킨다. 답은 `rules/axis_evidence.yaml` 에 인용과 함께 적는다.

질문 합계 **0개**.

없다 -- 모든 자리가 확정이거나 이미 근거가 등록돼 있다.

## 2. 자유 평가(③층) 상태

**`STALE`** -- ③ 자유 평가 이후 산출물이 바뀜 (기록 a369c64426ec7d28 != 현재 f9339c395db9aa99, 검토일 2026-09-12)

즉 규칙이 못 잡는 종류의 오류는 **이 판에서 다시 확인되지 않았다.** 규칙 게이트가 통과했다는 것과는 별개의 이야기다.

## 3. 손 안 댄 검토 지적

없다 (기록된 지적 10건은 전부 처리됨).

### 알고 받아들인 한계

**5건.** 고쳐야 할 결함이 아니라 **요약 표의 범위**다 -- 해당 계산은 원시 trace(`full/`)에는 있고 major-op 표에서 빠진다. 표의 행 수로 연산량을 세면 과소평가된다.

| 모듈 | 무엇이 빠졌나 | 근거 |
|---|---|---|
| `(root)` | E_shared | `num_shared_experts` 같은 config 필드는 없다. shared MLP 모듈이 하나 있다는 구조 사실이므로 축 심볼표가 아니라 구조 메타데이터에 두는 편이 정확하다. Meta 공식 표기도 '128 experts' 이지 129 가 아니다.  [2026-09-11 조치] 이 모델에서 E_shared 는 축 라벨로 쓰이지 않는다(다른 모델에서는 쓰인다). structure.yaml 의 known_ |
| `model.layers.*` | block_type | 레이어 접기 자체는 옳다(3그룹: 0,2,..=dense / 1,5,..=chunked+MoE / 3,7,..=full+NoPE+MoE). 그런데 뒤 두 그룹이 같은 `attn+MoE` 로 찍혀 독자가 왜 같아 보이는 블록이 두 번 나오는지 알 수 없다. chunked/full 과 RoPE/NoPE 차이가 표에 드러나지 않는다. `no_rope_layers` 는 이름과 달리 1 이 RoPE 사용이다.
  [인 |
| `feed_forward` | MoE 결합 | 소스는 `shared_out += routed_sum` 다음에 `residual + combined` 두 단계다(modeling_llama4.py:166-174, 450-458). 지금 표는 add 한 행이 의존성 셋을 물어 두 단계를 하나로 뭉갠다. |
| `(전체)` | 표에 없는 연산 | major 표에 chunked/full 마스크 생성과 score 합산, RoPE, NoPE 층의 temperature tuning, GQA 의 KV head 8->40 반복, router 의 topk/scatter, KV cache update/concat 이 안 보인다. QK-norm 이 없는 것은 누락이 아니다 -- 이 checkpoint 는 `use_qk_norm=false` 다. temperature  |
| `model.layers.*.self_attn` | chunked attention 이 shape 에 안 보임 | 외부 검토(Codex) 2026-09-16, develop/codex_four_models_review_2026-09-16.md. T=17, decode 전체 길이 18 은 chunk_size=8192 안이라 chunked/full 양쪽의 eager attention shape 가 같은 것이 정상이다. 차이는 마스크에 있다. RoPE 와 NoPE 층의 temperature scaling 도 attention  |

## 4. 의뢰서의 판단 필요 항목

없다.

## 5. 더 이상 안 맞는 소스 확인 기록

없다 -- 기록된 확인이 전부 지금 산출물의 축에 맞는다.

## 6. 표의 숫자가 관측값이 아닌 자리

없다 -- 모델 동작을 대체한 remedy 가 없다.
