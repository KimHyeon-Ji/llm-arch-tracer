# 라벨 검토 결과 — nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16

- 검토일: 2026-08-12
- 검토자: llm(claude, 접힌 표 전건 순회 — 외부 검토 방법론)
- 본 것: 외부 검토 방법론으로 44개 모델 × 2 phase 를 **행 단위 전건 순회**했다(샘플링 없음). 기존 게이트가 보지 않는 행 안의 조합만 확인: weight_pos 실제 대응, view 축 곱 보존, 전치·view 가 만들어낼 수 없는 새 이름.
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

## 발견 5 — 맞음 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.mixer.{up,down}_proj` |
| 축 | FFN 중간 폭 |
| 현재 라벨 | `d_ff (스코프 밖 폴백)` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

Nemotron-H 는 **모든 블록을 `mixer` 라 부른다** — FFN 블록도 그렇다. 그래서 `mixer.up_proj` 의 가중치 `[d_ff, d_model]` 이 d_ff 스코프의 어떤 철자에도 안 걸렸고, 이름은 맞는데 근거가 '스코프 밖 폴백'이었다. 스코프에 `up_proj|down_proj` 를 추가했다 — expert/router 를 막는 음의 전방탐색은 그대로다.

## 발견 6 — 이름 없음이 정답 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.mixer[.gate]` |
| 축 | Mamba2 chunk-scan decay 행렬의 뒤 두 축 (2) |
| 현재 라벨 | `n_kv / k` |
| 판정 | `no_name_exists` |
| 제안 라벨 | `정수 2` |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

`decay_chunk = torch.exp(segment_sum(F.pad(A_cumsum[:,:,:,-1], (1,0))))` → `[B, n_h_ssm, n_chunks+1, n_chunks+1]` (실측 `[1,128,2,2]`). 우리가 트레이스하는 regime 에서 T < d_chunk 라 n_chunks = 1 이고 그 축은 말 그대로 2 다. Nano 는 k(=2, num_experts_per_tok), Super/Ultra 는 n_kv(=2, GQA KV head 수)가 값이 같아 들어왔다 — 둘 다 Mamba mixer 와 아무 관계가 없다.

**일반형 `ceil(T/d_chunk)+1` 로 등록하지 않는다**: 관측한 적 없는 것을 주장하게 된다. 두 심볼 다 `group` 이 있어 스코프 밖 폴백에서는 배제되므로 재사용·전파 경로로 들어온 것이고, 그 경로를 막는 건 값이 겹치는 축 전반에 영향을 준다. 정수가 정답이라고 판정하고 남긴다.

**산출물에 반영됨(2026-08-12).** 규칙을 고쳐 재추론하는 방식은 사슬이 어긋나 두 번 되돌렸으므로, 렌더가 끝난 뒤 선언된 모듈 아래의 이름을 바꾸는 경로를 만들었다 — `rules/label_overrides.yaml` (근거 인용·기대 크기 필수, 발화 0건이면 게이트 FAIL). 적용 내역은 `full/label_overrides.json`, 절차는 `review/05-overrides.md`.

## 발견 7 — 맞음 (미반영)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.mixer.act_fn` |
| 축 | FFN 중간 폭 |
| 현재 라벨 | `d_ff (스코프 밖 폴백)` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | medium |
| 산출물 반영 | 미반영 |

**근거**

`mixer.act_fn` 은 경로에 `up_proj`/`down_proj` 가 없는 잎 모듈이라 위의 스코프 확장이 닿지 않는다. 이름은 맞고 근거만 약하다. 활성화 함수 잎까지 FFN 스코프로 여는 것은 이득 대비 위험이 커서 하지 않았다.

## 발견 8 — 교정 필요 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.mixer` |
| 축 | Mamba 레이어의 state 축 |
| 현재 라벨 | `d_head` |
| 판정 | `should_be_renamed` |
| 제안 라벨 | `d_state` |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

`layers_block_type` 이 이 스택의 스케줄을 그대로 말해준다 — Nemotron-3-Super 는 linear_attention 40 / moe 40 / full_attention 8 이다. 그런데 **모든 블록이 `mixer` 라는 같은 이름**이라 경로 정규식으로는 못 가른다. 게다가 head_dim == ssm_state_size == mamba_num_heads == 128 이라 값으로도 못 가른다. 그 결과 attention 의 `d_head` 가 Mamba 레이어 축을 대량으로 가져가고 있었다.

**교정**: 심볼이 `not_layer_types` 로 자기가 있을 수 없는 블록 종류를 선언하고 거기서 강등되게 했다(`symbolic_shape._ctx_symbols`). d_head 축이 9,712 → 512 로 줄고 d_state / n_h_ssm / d_chunk 가 그 자리를 받았다(Ultra 는 9,144 → 744).

**허용 목록이 아니라 거부 목록인 이유**: 같은 config 키를 Gemma-2/3·gpt-oss·Llama-4·GLM 은 sliding/full attention 구분에 쓴다 — 전부 attention 이다. 허용 목록으로 짰더니 sliding 레이어에서 head 이름이 통째로 강등돼 gemma-2-2b bare 0 → 1,248, Llama-4 288 → 2,952 로 무너졌다. 모르는 종류에서는 아무것도 하지 않는 쪽으로 바꿨다.

**이 교정에는 어떤 지표도 반응하지 않았다**(퇴행 0 / 개선 0). 값이 전부 맞아떨어지기 때문이다 — 자기모순 추적이 아니었으면 못 봤다.
