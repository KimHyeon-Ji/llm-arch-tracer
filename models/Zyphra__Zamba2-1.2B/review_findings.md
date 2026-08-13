# 라벨 검토 결과 — Zyphra/Zamba2-1.2B

- 검토일: 2026-08-13
- 검토자: llm(claude, 반박 프레임 전건 판정)
- 본 것: 의뢰서 항목을 **항목 단위로** 대조해 하나도 빠뜨리지 않는다(src/review_ledger.unanswered_items 가 개수가 아니라 항목을 맞춘다). 각 항목마다 그 폭을 만드는 코드 줄을 열어 확인했다.
- 요약: 미답 항목 3건을 소스로 판정했다.

> 이 파일은 `review_findings.json` 에서 생성된다 — 고칠 때는 JSON 을 고친다.

## 발견 1 — 맞음 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*` |
| 축 | [d_model, d_model] |
| 현재 라벨 | `d_model (두 번)` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

정사각 가중치(2048×2048). reshape 이 아니다.

**근거 소스**: 이 판정은 `develop/sources/modeling_zamba2.py`, `develop/sources/configuration_zamba2.py` 를 열어 확인했다. (인용 누락을 자가 점검에서 발견해 보강, 2026-08-12 — 게이트가 이제 `should_be_renamed` 판정에 소스 인용을 요구한다.)

## 발견 2 — 맞음 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.mamba` |
| 축 | [d_chunk, d_chunk] (tril/ones) |
| 현재 라벨 | `d_chunk (두 번)` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

Mamba2 의 청크 내 인과 마스크다 — `tril(ones(chunk_size, chunk_size))`, chunk_size=256. 정사각이 정상. 탐지기는 정사각 **reshape** 만 찾고 있어서 정사각 **생성**(ones/eye/zeros)을 못 봤다 — `source_check._SQUARE_NEW` 를 추가해 이제 소스에서 자동 확인된다.

**근거 소스**: 이 판정은 `develop/sources/modeling_zamba2.py`, `develop/sources/configuration_zamba2.py` 를 열어 확인했다. (인용 누락을 자가 점검에서 발견해 보강, 2026-08-12 — 게이트가 이제 `should_be_renamed` 판정에 소스 인용을 요구한다.)

## 발견 3 — 맞음 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.shared_transformer.self_attn` |
| 축 | [d_attn, d_attn] |
| 현재 라벨 | `d_attn (두 번)` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | medium |
| 산출물 반영 | 반영됨 |

**근거**

`modeling_zamba2.py:224` 주석대로 shared block 의 입력이 `attention_hidden_size = 2 * hidden_size` = 4096 이라 그 투영이 정사각이다.

## 발견 4 — 맞음 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `(config)` |
| 축 | intermediate_size=4096 / group_size=4096 |
| 현재 라벨 | `(미등록)` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

`modeling_zamba2.py:661` `self.intermediate_size = int(config.mamba_expand * self.hidden_size)` (2·2048=4096), `:699` 의 group_size 는 `intermediate_size // n_groups` 로 n_groups=1 이라 같은 값이다. **이 값은 이미 이름이 있다** — 등록된 규칙 `n_h_ssm * d_head_ssm` (64·64=4096) 이 그대로 설명한다. 즉 '이름 없는 폭'이 아니라 '한 폭에 config 필드가 여럿'인 경우였다. 탐지기(`symbolic_dims.probe`)가 이미 설명되는 값을 거르도록 고쳤다 — Llama-4 의 같은 오탐도 함께 사라졌다.

## 발견 5 — 교정 필요 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.shared_transformer.feed_forward.gate_up_proj` |
| 축 | [2*d_ff] (16384) |
| 현재 라벨 | `2*d_ff` |
| 판정 | `should_be_renamed` |
| 제안 라벨 | `2*d_ff` |
| 확신도 | medium |
| 산출물 반영 | 반영됨 |

**근거**

gate+up 융합으로 보이나 Zamba2 의 dense FFN 은 MoE 스코프(expert|moe)에 안 걸려 이번에 등록한 `2*d_moe` 규칙 대상이 아니다. dense FFN 의 융합 폭을 일반화하려면 다른 모델 사례가 더 필요하다.

**근거 소스**: 이 판정은 `develop/sources/modeling_zamba2.py`, `develop/sources/configuration_zamba2.py` 를 열어 확인했다. (인용 누락을 자가 점검에서 발견해 보강, 2026-08-12 — 게이트가 이제 `should_be_renamed` 판정에 소스 인용을 요구한다.)

**해결 (2026-08-13)**: "일반화하려면 다른 모델 사례가 더 필요하다"고 적혀 있었지만 필요한 것은 사례가 아니라 **소스 한 줄**이었다. `modeling_zamba2.py:876` `self.gate_up_proj = nn.Linear(self.hidden_size, 2 * self.intermediate_size, ...)` — dense FFN 의 gate+up 융합 폭이다. 기존 `2*d_moe` 규칙은 스코프가 `expert|moe` 라 여기 안 걸렸다. dense 쪽 규칙 `2*d_ff` 를 등록해 96축 해결.

## 발견 6 — 맞음 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.shared_transformer.feed_forward.gate_up_proj` |
| 축 | gate+up 융합 폭 |
| 현재 라벨 | `2*d_ff (산술 휴리스틱)` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

`modeling_zamba2.py:876` `self.gate_up_proj = nn.Linear(self.hidden_size, 2 * self.intermediate_size, bias=config.add_bias_linear)`. 이름은 맞았고 출처가 휴리스틱이었을 뿐이라 `rules/derived_dims.yaml` 에 dense FFN 용 `2*d_ff` 를 등록했다(MoE 의 `2*d_moe` 와 같은 구조인데 스코프가 달라 안 걸리던 자리다). `unless_equals: [d_model]` — 잔차 폭과 겹치는 모델에서는 물러난다.

## 발견 7 — 맞음 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.shared_transformer.self_attn.linear_q_adapter_list.*.*` |
| 축 | LoRA 랭크 128 |
| 현재 라벨 | `r_lora` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

`modeling_zamba2.py:270-271` `nn.Linear(self.attention_hidden_size, self.config.adapter_rank, bias=False), nn.Linear(self.config.adapter_rank, self.attention_hidden_size, bias=False)` — 두 Linear 사이의 좁은 축이 `adapter_rank` 다. 실측 `[128, 4096]`/`[4096, 128]` 이고 렌더도 `[r_lora, d_attn]` 이다. d_head(=128)와 값이 같아 후보로 올라왔을 뿐, 이 모듈은 head_dim 을 읽지 않는다.

## 발견 8 — 맞음 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.shared_transformer.self_attn.linear_k_adapter_list.*.*` |
| 축 | LoRA 랭크 128 |
| 현재 라벨 | `r_lora` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

`modeling_zamba2.py:270-271` `nn.Linear(self.attention_hidden_size, self.config.adapter_rank, bias=False), nn.Linear(self.config.adapter_rank, self.attention_hidden_size, bias=False)` — 두 Linear 사이의 좁은 축이 `adapter_rank` 다. 실측 `[128, 4096]`/`[4096, 128]` 이고 렌더도 `[r_lora, d_attn]` 이다. d_head(=128)와 값이 같아 후보로 올라왔을 뿐, 이 모듈은 head_dim 을 읽지 않는다.

## 발견 9 — 맞음 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.shared_transformer.self_attn.linear_v_adapter_list.*.*` |
| 축 | LoRA 랭크 128 |
| 현재 라벨 | `r_lora` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

`modeling_zamba2.py:270-271` `nn.Linear(self.attention_hidden_size, self.config.adapter_rank, bias=False), nn.Linear(self.config.adapter_rank, self.attention_hidden_size, bias=False)` — 두 Linear 사이의 좁은 축이 `adapter_rank` 다. 실측 `[128, 4096]`/`[4096, 128]` 이고 렌더도 `[r_lora, d_attn]` 이다. d_head(=128)와 값이 같아 후보로 올라왔을 뿐, 이 모듈은 head_dim 을 읽지 않는다.

## 발견 10 — 교정 필요 (미반영)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.mamba` |
| 축 | num_heads vs head_dim (둘 다 64) |
| 현재 라벨 | `d_head_ssm / n_h_ssm (순서 뒤바뀜)` |
| 판정 | `should_be_renamed` |
| 제안 라벨 | `n_h_ssm 이 앞, d_head_ssm 이 뒤` |
| 확신도 | high |
| 산출물 반영 | 미반영 |

**근거**

`modeling_zamba2.py:832` `hidden_states.view(batch_size, seq_len, self.num_heads, self.head_dim)`, `:622` `output.reshape(batch_size, -1, num_heads, head_dim)`, `:524` `ssm_states.view(batch_size * num_heads, head_dim, state_size)`, `:527` `out.view(batch_size, num_heads, head_dim)` — Mamba2 레이아웃에서 **num_heads 가 항상 head_dim 앞**이다. 우리 표는 정확히 반대로 붙어 있다: `[B, d_chunk, d_head_ssm, n_h_ssm]`(실측 `[1, 256, 64, 64]`), `[B, 1, d_head_ssm, n_h_ssm, d_state]`(실측 `[1, 1, 64, 64, 128]`). 홀로 나오는 자리도 마찬가지다 — `[B, d_head_ssm, 1, d_chunk]`(실측 `[1, 64, 1, 256]`)는 `:581 A_cumsum` 의 `[batch, num_heads, n_chunks, chunk]` 이고, `[B, T, d_head_ssm]`(실측 `[1, 16, 64]`)는 dt 의 `[batch, seq_len, num_heads]` 다. `n_mamba_heads`(:657)=64 와 `mamba_headdim`(:667)=64 가 같은 수라 값으로는 구별되지 않는다.

**아직 반영하지 않은 이유**: 두 이름이 같은 값이라 `label_overrides` 의 이름 치환으로는 **맞바꿀 수 없다**(먼저 도는 항목이 두 축을 하나로 뭉갠다). 축 위치로 지정하려면 관측된 shape 형태를 전수로 열거해야 하는데, 여기서 확인한 8개 형태가 전부라는 보장이 없다. 필요한 것은 '이 스코프에서 head 개수가 head 폭보다 앞선다'는 **순서 규칙**이며 그것은 별도 변경이다. 값 하나만 보고 대량 치환하는 쪽이 지금 상태보다 나쁘다.

## 발견 11 — 교정 필요 (미반영)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.mamba_decoder.mamba` |
| 축 | num_heads vs head_dim (둘 다 64) |
| 현재 라벨 | `d_head_ssm / n_h_ssm (순서 뒤바뀜)` |
| 판정 | `should_be_renamed` |
| 제안 라벨 | `n_h_ssm 이 앞, d_head_ssm 이 뒤` |
| 확신도 | high |
| 산출물 반영 | 미반영 |

**근거**

`modeling_zamba2.py:832` `hidden_states.view(batch_size, seq_len, self.num_heads, self.head_dim)`, `:622` `output.reshape(batch_size, -1, num_heads, head_dim)`, `:524` `ssm_states.view(batch_size * num_heads, head_dim, state_size)`, `:527` `out.view(batch_size, num_heads, head_dim)` — Mamba2 레이아웃에서 **num_heads 가 항상 head_dim 앞**이다. 우리 표는 정확히 반대로 붙어 있다: `[B, d_chunk, d_head_ssm, n_h_ssm]`(실측 `[1, 256, 64, 64]`), `[B, 1, d_head_ssm, n_h_ssm, d_state]`(실측 `[1, 1, 64, 64, 128]`). 홀로 나오는 자리도 마찬가지다 — `[B, d_head_ssm, 1, d_chunk]`(실측 `[1, 64, 1, 256]`)는 `:581 A_cumsum` 의 `[batch, num_heads, n_chunks, chunk]` 이고, `[B, T, d_head_ssm]`(실측 `[1, 16, 64]`)는 dt 의 `[batch, seq_len, num_heads]` 다. `n_mamba_heads`(:657)=64 와 `mamba_headdim`(:667)=64 가 같은 수라 값으로는 구별되지 않는다.

**아직 반영하지 않은 이유**: 두 이름이 같은 값이라 `label_overrides` 의 이름 치환으로는 **맞바꿀 수 없다**(먼저 도는 항목이 두 축을 하나로 뭉갠다). 축 위치로 지정하려면 관측된 shape 형태를 전수로 열거해야 하는데, 여기서 확인한 8개 형태가 전부라는 보장이 없다. 필요한 것은 '이 스코프에서 head 개수가 head 폭보다 앞선다'는 **순서 규칙**이며 그것은 별도 변경이다. 값 하나만 보고 대량 치환하는 쪽이 지금 상태보다 나쁘다.
