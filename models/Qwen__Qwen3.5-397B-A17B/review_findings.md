# 라벨 검토 결과 — Qwen/Qwen3.5-397B-A17B

- 검토일: 2026-08-12
- 검토자: llm(claude, 반박 프레임 전건 판정)
- 본 것: 의뢰서의 **모든** 질문에 답한다(기계가 개수를 맞춘다). 확인 프레임이 아니라 반박 프레임으로 — 각 라벨에 대해 '틀렸다는 증거'를 먼저 찾고, 못 찾은 것만 맞다고 적었다. 외부 검토가 준 팁 3가지(op 내부 필드 상호 대조 / 요청·응답 개수 diff / 반박 프레임)를 그대로 적용했다.
- 요약: 

> 이 파일은 `review_findings.json` 에서 생성된다 — 고칠 때는 JSON 을 고친다.

## 발견 1 — 맞음 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.linear_attn` |
| 축 | DeltaNet 청크 루프·head 폭 |
| 현재 라벨 | `(Qwen3.5/3.6 계열과 동일)` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

Qwen3-Next 가 만든 Gated DeltaNet 규칙이 그대로 적용됐다 — **전용 규칙 0개**. 남은 reshape 90 / matmul 135 건은 형제 모델들과 같은 원인이다(`linear_key_head_dim == linear_value_head_dim`, 개명 전파 막힘). 그 판정은 Qwen3.5-4B / 3.6-27B / 3.6-35B 의 findings 에 이미 기록돼 있다.

**근거 소스**: 이 판정은 `develop/sources/modeling_qwen3_5_moe.py`, `develop/sources/configuration_qwen3_5_moe.py` 를 열어 확인했다. (인용 누락을 자가 점검에서 발견해 보강, 2026-08-12 — 게이트가 이제 `should_be_renamed` 판정에 소스 인용을 요구한다.)

## 발견 2 — 이름 없음이 정답 (미반영)

| 항목 | 값 |
|---|---|
| 모듈 | `(config)` |
| 축 | d_ff 미확인 |
| 현재 라벨 | `—` |
| 판정 | `no_name_exists` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 미반영 |

**근거**

이 체크포인트는 `intermediate_size` 가 없다 — dense FFN 이 아예 없는 순수 MoE 다. `d_ff` 는 group 이 없어서 '해당 없음' 대신 '미확인'으로 표시된다. 라벨 오류가 아니라 표기 문제이고, d_ff 에 group 을 다는 것은 전 함대에 영향을 주므로 하지 않았다.

**근거 소스**: 이 판정은 `develop/sources/modeling_qwen3_5_moe.py`, `develop/sources/configuration_qwen3_5_moe.py` 를 열어 확인했다. (인용 누락을 자가 점검에서 발견해 보강, 2026-08-12 — 게이트가 이제 `should_be_renamed` 판정에 소스 인용을 요구한다.)

## 발견 3 — 미확정 (미반영)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.linear_attn` |
| 축 | d_head_lin_k vs d_head_lin_v (128) |
| 현재 라벨 | `(값 동률)` |
| 판정 | `undetermined` |
| 제안 라벨 | `판정 불가` |
| 확신도 | high |
| 산출물 반영 | 미반영 |

**근거**

`linear_key_head_dim == linear_value_head_dim == 128` 이다. 소스는 둘을 구별하지만(`torch.split(mixed_qkv, [key_dim, key_dim, value_dim])` 뒤 각각 `head_k_dim`/`head_v_dim` 으로 reshape) **이 체크포인트에서는 값이 같아 트레이스 안에 가를 증거가 없다.** 모듈도 shape 도 같고 갈리는 건 split 의 몇 번째 조각이냐뿐이다.

값으로 우기지 않고 남긴다. 두 값이 다른 체크포인트를 추적하면 규칙이 그대로 작동한다.

**근거 소스**: 이 판정은 `develop/sources/modeling_qwen3_5_moe.py`, `develop/sources/configuration_qwen3_5_moe.py` 를 열어 확인했다. (인용 누락을 자가 점검에서 발견해 보강, 2026-08-12 — 게이트가 이제 `should_be_renamed` 판정에 소스 인용을 요구한다.)

## 발견 4 — 이름 없음이 정답 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.linear_attn` |
| 축 | 청크 스캔 루프 계단 (2*d_conv_lin / 3*d_conv_lin / 3*n_h_lin_k) |
| 현재 라벨 | `산술 휴리스틱` |
| 판정 | `no_name_exists` |
| 제안 라벨 | `정수` |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

`2*d_conv_lin` / `3*d_conv_lin` / `3*n_h_lin_k` — 전부 Gated DeltaNet 청크 스캔의 **언롤된 루프 계단**이다. `linear_attn` 안의 slice/clone 이 반복마다 한 칸씩 넓은 조각을 떼며 3,4,5,…,10 의 연속 사다리를 만들고, 산술 휴리스틱이 그 정수마다 값이 맞는 식을 지어냈다.

**반박 시도**: 아키텍처 차원이라면 config 어딘가에서 나와야 하는데, 같은 자리에서 값이 **반복마다 달라진다**(3,4,5,… 각 값이 같은 횟수씩). 차원은 그렇게 움직이지 않는다. `develop/verify/questions? no` — `develop/verify/references.yaml` 의 irreducible_literals 에 이 모델을 포함해 사유와 함께 등재돼 있다. 정수로 두는 것이 정답이다.

**근거 소스**: 이 판정은 `develop/sources/modeling_qwen3_5_moe.py`, `develop/sources/configuration_qwen3_5_moe.py` 를 열어 확인했다. (인용 누락을 자가 점검에서 발견해 보강, 2026-08-12 — 게이트가 이제 `should_be_renamed` 판정에 소스 인용을 요구한다.)

## 발견 5 — 맞음 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `(config)` |
| 축 | d_head_lin_k 가 읽은 linear_key_head_dim |
| 현재 라벨 | `d_head_lin_k` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

`d_head_lin_k ← linear_key_head_dim` — 값은 로드된 config 에 있는데 config **클래스**가 선언하지 않는다는 안건이다.

**반박 시도**: 코드가 이 필드를 실제로 읽는가? `modeling_qwen3_next.py:518` `self.head_k_dim = config.linear_key_head_dim` — 읽는다. 체크포인트 config.json 이 출처인 것은 정상이며, 클래스가 선언하지 않아도 modeling 이 읽으면 그 값이 권위다(같은 이유로 `optional_config_reads` 가 getattr 패턴을 접지로 인정한다).

## 발견 6 — 미확정 (미반영)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.linear_attn.norm` |
| 축 | d_head_lin_k vs d_head_lin_v (128, norm 쪽) |
| 현재 라벨 | `(값 동률)` |
| 판정 | `undetermined` |
| 제안 라벨 | `판정 불가` |
| 확신도 | high |
| 산출물 반영 | 미반영 |

**근거**

위 `linear_attn` 건과 같은 충돌이 norm 모듈에도 나타난다. 원인·근거 동일하다.

**근거 소스**: 이 판정은 `develop/sources/modeling_qwen3_5_moe.py`, `develop/sources/configuration_qwen3_5_moe.py` 를 열어 확인했다. (인용 누락을 자가 점검에서 발견해 보강, 2026-08-12 — 게이트가 이제 `should_be_renamed` 판정에 소스 인용을 요구한다.)
