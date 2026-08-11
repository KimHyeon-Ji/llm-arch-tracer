# 라벨 검토 결과 — nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16

- 검토일: 2026-08-11
- 검토자: llm(claude, 전수 점검 2회차 — 모듈-필드 소속)
- 본 것: 전수 점검 2회차 — 1층(모듈-필드 소속: 가중치 축의 이름이 그 모듈/부모가 실제로 읽는 config 필드에서 나왔는가)을 함대 전건 수행. 값을 보지 않는 검사라 값 충돌이 숨길 수 없다. 1회차의 A절·B절 판정은 유지. C절(모듈별 출력 shape)은 여전히 미수행.
- 요약: 의뢰서 5건 — 2건이 MoE 이름이 Mamba 축으로 새어 든 오라벨이었다.

> 이 파일은 `review_findings.json` 에서 생성된다 — 고칠 때는 JSON 을 고친다.

## 발견 1 — 맞음 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `(config)` |
| 축 | L 이 읽은 num_hidden_layers |
| 현재 라벨 | `num_hidden_layers` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

`configuration_nemotron_h.py:170` 은 num_hidden_layers 를 deprecated 로 두고 `layers_block_type` 길이를 쓴다고 경고한다. 이 체크포인트는 둘 다 42 로 일치하고(실측), structure 의 layers 절도 0~41 을 빠짐없이 덮는다.

## 발견 2 — 이름 없음이 정답 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.mixer` |
| 축 | [B, n_h_ssm, 2, 2] 의 뒤 두 축 |
| 현재 라벨 | `k` |
| 판정 | `no_name_exists` |
| 제안 라벨 | `정수 2` |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

`modeling_nemotron_h.py:320` `decay_chunk = torch.exp(segment_sum(F.pad(A_cumsum[:, :, :, -1], (1, 0)))).transpose(1, 3)` → `[B, n_h_ssm, n_chunks+1, n_chunks+1]`. T=24 < d_chunk=256 이라 n_chunks=1 이고 축은 말 그대로 2 다. 이름 `k`(num_experts_per_tok=2)는 MoE 심볼이 Mamba 모듈로 새어 든 것 — `k` 는 스코프가 있는데도 '스코프가 배제한 심볼' 폴백이 다시 썼다. **일반형 ceil(T/d_chunk)+1 로 등록하지 않았다**: 우리가 트레이스하는 regime 에서 n_chunks 는 항상 1 이라 관측한 적 없는 것을 주장하게 된다.

## 발견 3 — 교정 필요 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.mixer` |
| 축 | [B, T, n_g_ssm, 12, d_state] 의 12 |
| 현재 라벨 | `3*d_conv` |
| 판정 | `should_be_renamed` |
| 제안 라벨 | `n_h_ssm/n_g_ssm` |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

`expand [B,T,n_g_ssm,1,d_state] -> [B,T,n_g_ssm,12,d_state] -> view -> [B,T,n_h_ssm,d_state]` 이고 n_h_ssm/n_g_ssm = 96/8 = 12 다. 3·d_conv(=3·4)와 값이 같아 그쪽으로 지어져 있었다. 규칙 등록 완료.

## 발견 4 — 맞음 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.mixer` |
| 축 | decode KV 폭 T+1 |
| 현재 라벨 | `T+1` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | medium |
| 산출물 반영 | 반영됨 |

**근거**

decode 단계에서 캐시 T 개 + 새 토큰 1 개. 이미 등록된 규칙과 같은 형태이며 값도 일치한다.
