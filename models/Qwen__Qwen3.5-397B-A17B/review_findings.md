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

## 발견 3 — 교정 필요 (미반영)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.linear_attn` |
| 축 | d_head_lin_k vs d_head_lin_v (128) |
| 현재 라벨 | `(값 동률)` |
| 판정 | `should_be_renamed` |
| 제안 라벨 | `판정 불가` |
| 확신도 | high |
| 산출물 반영 | 미반영 |

**근거**

`linear_key_head_dim == linear_value_head_dim == 128` 이다. 소스는 둘을 구별하지만(`torch.split(mixed_qkv, [key_dim, key_dim, value_dim])` 뒤 각각 `head_k_dim`/`head_v_dim` 으로 reshape) **이 체크포인트에서는 값이 같아 트레이스 안에 가를 증거가 없다.** 모듈도 shape 도 같고 갈리는 건 split 의 몇 번째 조각이냐뿐이다.

값으로 우기지 않고 남긴다. 두 값이 다른 체크포인트를 추적하면 규칙이 그대로 작동한다.

**근거 소스**: 이 판정은 `develop/sources/modeling_qwen3_5_moe.py`, `develop/sources/configuration_qwen3_5_moe.py` 를 열어 확인했다. (인용 누락을 자가 점검에서 발견해 보강, 2026-08-12 — 게이트가 이제 `should_be_renamed` 판정에 소스 인용을 요구한다.)

**재분류 (2026-08-13)**: 이 판정은 `undetermined` 였다. 잘못된 분류다 — 근거 문장이 "트레이스 안에 가를 증거가 없다"고 적고 있었는데, 그건 *트레이스만으로는* 못 가른다는 말이지 *알 수 없다*는 말이 아니다. **소스는 답을 갖고 있다**(위 인용). 막는 것은 지식이 아니라 표현 수단이다: 두 이름이 같은 값이라 `label_overrides` 의 이름 치환으로는 갈 수 없고, 필요한 것은 권위 있는 이름을 데이터플로우 따라 끌고 가는 메커니즘이다. `review/06-open-renames.md` 의 같은 병이므로 그쪽으로 합친다. **모르는 것과 못 넣는 것은 다르게 적는다.**

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

## 발견 6 — 교정 필요 (미반영)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.linear_attn.norm` |
| 축 | d_head_lin_k vs d_head_lin_v (128, norm 쪽) |
| 현재 라벨 | `(값 동률)` |
| 판정 | `should_be_renamed` |
| 제안 라벨 | `판정 불가` |
| 확신도 | high |
| 산출물 반영 | 미반영 |

**근거**

위 `linear_attn` 건과 같은 충돌이 norm 모듈에도 나타난다. 원인·근거 동일하다.

**근거 소스**: 이 판정은 `develop/sources/modeling_qwen3_5_moe.py`, `develop/sources/configuration_qwen3_5_moe.py` 를 열어 확인했다. (인용 누락을 자가 점검에서 발견해 보강, 2026-08-12 — 게이트가 이제 `should_be_renamed` 판정에 소스 인용을 요구한다.)

**재분류 (2026-08-13)**: 이 판정은 `undetermined` 였다. 잘못된 분류다 — 근거 문장이 "트레이스 안에 가를 증거가 없다"고 적고 있었는데, 그건 *트레이스만으로는* 못 가른다는 말이지 *알 수 없다*는 말이 아니다. **소스는 답을 갖고 있다**(위 인용). 막는 것은 지식이 아니라 표현 수단이다: 두 이름이 같은 값이라 `label_overrides` 의 이름 치환으로는 갈 수 없고, 필요한 것은 권위 있는 이름을 데이터플로우 따라 끌고 가는 메커니즘이다. `review/06-open-renames.md` 의 같은 병이므로 그쪽으로 합친다. **모르는 것과 못 넣는 것은 다르게 적는다.**

## 발견 7 — 교정 필요 (미반영)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.linear_attn` |
| 축 | gated delta rule 청크 길이 64 (chunk_size) |
| 현재 라벨 | `d_rope` |
| 판정 | `should_be_renamed` |
| 제안 라벨 | `d_chunk` |
| 확신도 | high |
| 산출물 반영 | 미반영 |

**근거**

`modeling_qwen3_next.py:381` `def torch_chunk_gated_delta_rule(..., chunk_size=64)` — 청크 길이가 **config 필드가 아니라 커널 fallback 의 기본 인자**다. 같은 리터럴이 `modeling_qwen3_5.py` / `modeling_qwen3_5_moe.py` 에도 있다. 심볼 표는 config 를 읽으므로 코드에만 있는 상수는 구조적으로 유도할 수 없고, 그래서 이 폭이 이름을 못 받거나 엉뚱한 이름을 받는다.

**현재 라벨이 틀렸다는 증거**: 이 블록(`Qwen3NextGatedDeltaNet`)에는 **RoPE 가 아예 없다** — RoPE 는 같은 스택의 `self_attn` 레이어에만 있다. 그런데 `rules/derived_dims.yaml` 의 `round(d_head * pr)` 규칙이 scope `attn|attention|rotary` 로 걸려 있고 `attn` 은 `linear_attn` 안에서도 매치한다. partial_rotary_factor(0.25) x head_dim(256) = 64 이고 chunk_size 도 64 라, 청크 스캔의 `[chunk, chunk]` triu 마스크가 통째로 `d_rope` 로 렌더되고 있다 — Qwen3.6-27B 한 모델에서만 31,440축(2026-08-13 측정).

그 규칙의 주석은 이미 "스코프에서 linear_attn 을 뺀다"고 적어 두었는데 **정규식은 바뀐 적이 없다.** 주석이 코드에 없는 수정을 주장하고 있었다(앵커의 `rank1` 과 같은 부류).

**세 가지 상태를 전부 측정했다(2026-08-13)**:
- (A) 현 상태: `d_rope` 가 120,513축을 차지. 게이트는 통과하지만 **증명 가능하게 틀린 이름**이다.
- (B) `d_rope` 스코프만 교정: 그 자리를 휴리스틱이 채운다 — Qwen3.6-27B heur 4,128 -> 84,000 (`4*n_h_lin_k`). 지어낸 이름이 늘어 더 나쁘다.
- (C) 스코프 교정 + `d_chunk` 상수 등록: 이름은 전부 소스 근거를 얻지만 flow_ambig 이 오른다 (Qwen3-Next 108->288, Qwen3.5-397B 135->360, Qwen3.5-4B 72->192, Qwen3.6-35B 90->240). 원인은 반대편 `batched_matmul` 이 휴리스틱이 지어낸 `2*n_h_lin_v`(=2·32=64)를 들고 있어서다 — 한 텐서에 두 이름.

**아직 반영하지 않은 이유**: (C) 가 옳지만 게이트 퇴행 검사가 막는다. 필요한 것은 '등록된 심볼이 그 값을 설명하는 자리에서는 휴리스틱이 이름을 짓지 않는다'는 규칙, 또는 권위 있는 이름을 데이터플로우 따라 끌고 가는 메커니즘이다. 43건과 같은 병이며 `review/06-open-renames.md` 의 자문 대상이다. 한쪽만 고치는 수정은 하지 않는다.
