# 라벨 검토 결과 — LiquidAI/LFM2-8B-A1B

- 검토일: 2026-08-12
- 검토자: llm(claude, 양쪽 phase 전건 + 통과군 무작위 표본 감사)
- 본 것: **게이트가 이제 prefill·decode 양쪽을 본다**(그전까지 decode 는 한 번도 검사된 적이 없었다). A·B·C절 전건 + 통과군 무작위 표본 30건 감사. 기준은 review/04-full-inventory.md.
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
