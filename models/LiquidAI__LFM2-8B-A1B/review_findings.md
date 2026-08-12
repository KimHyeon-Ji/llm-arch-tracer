# 라벨 검토 결과 — LiquidAI/LFM2-8B-A1B

- 검토일: 2026-08-12
- 검토자: llm(claude, 접힌 표 전건 순회 — 외부 검토 방법론)
- 본 것: 외부 검토 방법론으로 44개 모델 × 2 phase 를 **행 단위 전건 순회**했다(샘플링 없음). 기존 게이트가 보지 않는 행 안의 조합만 확인: weight_pos 실제 대응, view 축 곱 보존, 전치·view 가 만들어낼 수 없는 새 이름.
- 요약: 

> 이 파일은 `review_findings.json` 에서 생성된다 — 고칠 때는 JSON 을 고친다.

## 발견 1 — 교정 필요 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.conv` |
| 축 | short conv 커널 폭 (3) |
| 현재 라벨 | `정수 3` |
| 판정 | `should_be_renamed` |
| 제안 라벨 | `d_conv` |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

LFM2 는 short convolution 블록을 쓰고 커널 크기를 `conv_L_cache` 로 부른다. 커널 폭이라는 역할이 Mamba 의 causal conv1d 와 같으므로 `d_conv` 별칭에 추가하고 스코프에 `conv` 를 넣었다. 아울러 `layer_types` 에 `conv` 블록 종류가 있으므로 attention head 이름들의 `not_layer_types` 에도 `conv` 를 넣어 conv 블록을 배제했다.

## 발견 2 — 교정 필요 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.pos_emb` |
| 축 | rotary 축 (32) |
| 현재 라벨 | `E` |
| 판정 | `should_be_renamed` |
| 제안 라벨 | `d_head/2` |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

실측 `[1, 16, 32]` 이고 바로 옆이 `[1, 16, 64]`(=d_head) 다 — rotary 의 inv_freq 절반 축이다. 전문가 수 E(=32)와 값이 겹쳐 그 이름이 붙었다. `E` 는 `group: moe` 라 스코프 밖 폴백에서는 배제되므로 재사용·전파 경로로 들어온 것이고, 그 경로를 막는 건 값이 겹치는 축 전반에 영향을 준다. 판정만 남긴다.

**산출물에 반영됨(2026-08-12).** 규칙을 고쳐 재추론하는 방식은 사슬이 어긋나 두 번 되돌렸으므로, 렌더가 끝난 뒤 선언된 모듈 아래의 이름을 바꾸는 경로를 만들었다 — `rules/label_overrides.yaml` (근거 인용·기대 크기 필수, 발화 0건이면 게이트 FAIL). 적용 내역은 `full/label_overrides.json`, 절차는 `review/05-overrides.md`.

## 발견 3 — 미확정 (미반영)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.conv` |
| 축 | 설명되지 않는 정수 18·32 |
| 현재 라벨 | `정수` |
| 판정 | `undetermined` |
| 제안 라벨 | — |
| 확신도 | medium |
| 산출물 반영 | 미반영 |

**근거**

`model.layers.*.conv` 안에서만 나타나고 config 어느 필드와도 대응되지 않는다. 커널 폭 3 은 `conv_L_cache` 로 접지했지만 이 둘은 소스에서 근거를 못 찾았다. `develop/verify/references.yaml` 에 사유와 함께 등재했다 — 이름을 지어내지 않는다.

## 발견 4 — 교정 필요 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.pos_emb` |
| 축 | rotary 축이 한 모듈 안에서 두 이름 (E/32) |
| 현재 라벨 | `E 와 정수 32 혼재` |
| 판정 | `should_be_renamed` |
| 제안 라벨 | `d_head/2` |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

`transpose [B, 32, T] -> [B, T, d_head/2]` — **전치는 축 이름을 바꿀 수 없다.** 같은 rotary 축이 어떤 행에서는 `E`(교정 후 `d_head/2`), 어떤 행에서는 정수 `32` 였고, 앞선 교정이 `E` 만 바꾸는 바람에 한 모듈 안에 두 이름이 남았다. **한쪽만 고치는 수정은 그 자체가 결함이다** — 외부 검토가 Llama 의 weight/operand 에서 지적한 것과 같은 부류이며, 이번엔 내가 만든 교정이 그 부류를 새로 만들었다. `rules/label_overrides.yaml` 에 정수 쪽 항목을 추가해 `model.pos_emb` 의 그 축을 `d_head/2` 하나로 통일했다.
