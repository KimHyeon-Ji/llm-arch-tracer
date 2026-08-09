# 라벨 검토 결과 — Zyphra/Zamba2-1.2B

- 검토일: 2026-08-09
- 검토자: llm(claude, 소스 직접 대조)
- 본 것: 의뢰서 39건 전수 (review/ 절차) — 별칭 접지 / 정사각 축 / 미등록 필드 / 산술로 지은 이름. 39 -> 5건으로 축소 / `develop/sources/` 의 실제 modeling·configuration 소스
- 요약: 의뢰서 7건 — 정사각 3건은 오탐, 미등록 2건과 융합 폭 2건은 근거가 모자라 미확정으로 남긴다.

## 발견 1 — 맞음

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*` |
| 축 | [d_model, d_model] |
| 현재 라벨 | `d_model (두 번)` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |

**근거**

정사각 가중치(2048×2048). reshape 이 아니다.

## 발견 2 — 맞음

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.mamba` |
| 축 | [d_chunk, d_chunk] (tril/ones) |
| 현재 라벨 | `d_chunk (두 번)` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |

**근거**

Mamba2 의 청크 내 인과 마스크다 — `tril(ones(chunk_size, chunk_size))`, chunk_size=256. 정사각이 정상. 탐지기는 정사각 **reshape** 만 찾고 있어서 정사각 **생성**(ones/eye/zeros)을 못 봤다 — `source_check._SQUARE_NEW` 를 추가해 이제 소스에서 자동 확인된다.

## 발견 3 — 맞음

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.shared_transformer.self_attn` |
| 축 | [d_attn, d_attn] |
| 현재 라벨 | `d_attn (두 번)` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | medium |

**근거**

`modeling_zamba2.py:224` 주석대로 shared block 의 입력이 `attention_hidden_size = 2 * hidden_size` = 4096 이라 그 투영이 정사각이다.

## 발견 4 — 맞음

| 항목 | 값 |
|---|---|
| 모듈 | `(config)` |
| 축 | intermediate_size=4096 / group_size=4096 |
| 현재 라벨 | `(미등록)` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |

**근거**

`modeling_zamba2.py:661` `self.intermediate_size = int(config.mamba_expand * self.hidden_size)` (2·2048=4096), `:699` 의 group_size 는 `intermediate_size // n_groups` 로 n_groups=1 이라 같은 값이다. **이 값은 이미 이름이 있다** — 등록된 규칙 `n_h_ssm * d_head_ssm` (64·64=4096) 이 그대로 설명한다. 즉 '이름 없는 폭'이 아니라 '한 폭에 config 필드가 여럿'인 경우였다. 탐지기(`symbolic_dims.probe`)가 이미 설명되는 값을 거르도록 고쳤다 — Llama-4 의 같은 오탐도 함께 사라졌다.

## 발견 5 — 미확정

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.shared_transformer.feed_forward.gate_up_proj` |
| 축 | [2*d_ff] (16384) |
| 현재 라벨 | `2*d_ff` |
| 판정 | `undetermined` |
| 제안 라벨 | — |
| 확신도 | medium |

**근거**

gate+up 융합으로 보이나 Zamba2 의 dense FFN 은 MoE 스코프(expert|moe)에 안 걸려 이번에 등록한 `2*d_moe` 규칙 대상이 아니다. dense FFN 의 융합 폭을 일반화하려면 다른 모델 사례가 더 필요하다.
