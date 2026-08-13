# 라벨 검토 결과 — meta-llama/Llama-4-Maverick-17B-128E

- 검토일: 2026-08-13
- 검토자: llm(claude, 반박 프레임 전건 판정)
- 본 것: 의뢰서 항목을 **항목 단위로** 대조해 하나도 빠뜨리지 않는다(src/review_ledger.unanswered_items 가 개수가 아니라 항목을 맞춘다). 각 항목마다 그 폭을 만드는 코드 줄을 열어 확인했다.
- 요약: 미답 항목 1건을 소스로 판정했다.

> 이 파일은 `review_findings.json` 에서 생성된다 — 고칠 때는 JSON 을 고친다.

## 발견 1 — 맞음 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `(config)` |
| 축 | intermediate_size=8192 / expert_dim=8192 |
| 현재 라벨 | `(미등록)` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | medium |
| 산출물 반영 | 반영됨 |

**근거**

두 필드 모두 전문가 FFN 폭이고 우리 심볼 d_moe 가 이미 8192 로 해석하고 있다(dense FFN 은 `intermediate_size_mlp`=16384 → d_ff). 즉 '이름이 없는 값'이 아니라 **한 값에 config 필드가 둘**인 경우다. 탐지기(`src/symbolic_dims.probe`)가 '이미 등록된 심볼이 그 값을 설명하는가'를 안 보는 것이 오탐의 원인 — 다음 개선 대상으로 남긴다.

**근거 소스**: 이 판정은 `develop/sources/modeling_llama4.py`, `develop/sources/configuration_llama4.py` 를 열어 확인했다. (인용 누락을 자가 점검에서 발견해 보강, 2026-08-12 — 게이트가 이제 `should_be_renamed` 판정에 소스 인용을 요구한다.)

## 발견 2 — 맞음 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.feed_forward.experts / shared_expert.*` |
| 축 | 전문가 FFN 폭 8192 |
| 현재 라벨 | `d_moe` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

`modeling_llama4.py:59-61` `self.intermediate_size = config.intermediate_size; self.expert_dim = self.intermediate_size` — Llama-4 는 전문가 폭에 `moe_intermediate_size` 가 아니라 그냥 `intermediate_size` 를 쓰고, dense 쪽은 `intermediate_size_mlp`(:411)를 쓴다. 이름 `d_moe` 의 뜻은 정확하다. 소속 검사가 처음에 이걸 지적했는데 **탐지기 쪽 한계**였다 — 이 심볼이 어느 필드에서 값을 읽었는지 기록이 없으면 무엇과 대조할지 알 수 없다. 그런 심볼에 대해서는 아무 주장도 하지 않도록 고쳤다(침묵은 근거가 아니다).

## 발견 3 — 맞음 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.feed_forward` |
| 축 | 라우팅 입력 행 수 2048 |
| 현재 라벨 | `E*T` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

`modeling_llama4.py:168-170` `router_scores, router_logits = self.router(hidden_states); routed_in = hidden_states.repeat(router_scores.shape[1], 1)` 이고 `Llama4Router` 는 `nn.Linear(config.hidden_size, config.num_local_experts)`(`:141-142`)라 `router_scores.shape[1] == num_local_experts` 다. 즉 `[T, d_model]` 를 전문가 수만큼 세로로 복제한 `[E*T, d_model]` 이 정확한 이름이다(실측 `[2048, 5120]`, E=128 · T=16). Llama-4 는 dropless MoE 라 전문가마다 **모든** 토큰을 받는다 — 이것이 `k*T` 가 아니라 `E*T` 인 이유다. 이 항목은 2라운드 연속 무응답이었고 개수만 맞추는 검사가 그것을 통과시켰다(`src/review_ledger.unanswered_items` 로 항목 대조로 교체했다).
