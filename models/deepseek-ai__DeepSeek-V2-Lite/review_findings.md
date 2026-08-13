# 라벨 검토 결과 — deepseek-ai/DeepSeek-V2-Lite

- 검토일: 2026-08-12
- 검토자: llm(claude, 행 단위 전건 — 검토자 방식)
- 본 것: 의뢰서에 **행 단위 전건 절**을 앞에 세우고 그 뷰로 다시 봤다. 접힌 표의 고유 행은 모델당 중앙값 62개(최대 136)뿐이라 전부 읽을 수 있다 — A/B/C 절보다 작으면서 한 행 안의 어긋남까지 보인다.
- 요약: 의뢰서 3건 — 산술은 맞지만 이름이 틀렸다. 규칙으로 교정 완료.

> 이 파일은 `review_findings.json` 에서 생성된다 — 고칠 때는 JSON 을 고친다.

## 발견 1 — 교정 필요 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.mlp.shared_experts.{gate,up,down}_proj` |
| 축 | 공유 전문가 FFN 폭 2816 |
| 현재 라벨 | `2*d_moe` |
| 판정 | `should_be_renamed` |
| 제안 라벨 | `E_shared*d_moe` |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

`modeling_deepseek_v2.py:127` `DeepseekV2MLP(config=config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts)`. n_shared_experts=2, d_moe=1408 이라 2·1408 과 값이 같지만 그 2 는 gate+up 의 2 가 아니라 **공유 전문가 수**다. `rules/derived_dims.yaml` 에 `E_shared*d_moe` 를 shared_expert 스코프로 등록했고, gate+up 규칙보다 앞에 두어 순서로 이기게 했다.

## 발견 2 — 이름 없음이 정답 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.self_attn` |
| 축 | [B, 1, 1, d_rope/2, 2] 의 마지막 축 |
| 현재 라벨 | `2 (이름 없음)` |
| 판정 | `no_name_exists` |
| 제안 라벨 | `정수 2` |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

바로 그 축을 `view_as_complex` / `view_as_real` 이 소비한다 — 복소수 한 개의 실수부·허수부 쌍이지 아키텍처 차원이 아니다. RoPE 를 복소수 곱으로 구현하는 표준 형태이고, 이 모델의 `E_shared`(=2)와 값이 같은 것은 우연이다. **정수로 두는 것이 정답이다.** B절이 지목했고 여기서 종결한다.

## 발견 3 — 교정 필요 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.self_attn` |
| 축 | kv_b_proj 출력을 head 별로 접은 폭 (256) |
| 현재 라벨 | `2*d_nope` |
| 판정 | `should_be_renamed` |
| 제안 라벨 | `d_nope+d_v` |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

`modeling_deepseek_v3.py:419` `k_nope, value_states = torch.split(kv_nope, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)` — 조각은 QK 폭과 VALUE 폭이지 같은 것 둘이 아니다. `rules/derived_dims.yaml` 에 `d_nope + d_v` 가 이미 등록돼 있었는데도 `2*d_nope` 로 렌더됐다. 원인은 `build_table._merge_from_split` — '한 축이 n 등분되어 같은 이름 n 개가 되면 그 축은 n 배다' 라는 추론이다. 이 모델들은 qk_nope_head_dim == v_head_dim == 128 이라 두 조각이 **둘 다** `d_nope` 로 렌더됐고, 그 추론이 그걸 근거 삼아 부모 축을 `2*d_nope` 로 덮어썼다. **등록된 규칙이 추론보다 위**가 되도록 가드를 넣었다(`_is_registered`). 6개 모델 554축 교정.

## 발견 4 — 교정 필요 (미반영)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.self_attn` |
| 축 | value 경로 head 폭 (128) — split 둘째 조각부터 o_proj 입력까지 |
| 현재 라벨 | `d_nope` |
| 판정 | `should_be_renamed` |
| 제안 라벨 | `d_v` |
| 확신도 | high |
| 산출물 반영 | 미반영 |

**근거**

같은 split 의 **둘째** 조각이 `value_states` 이고 그 head 폭은 `v_head_dim` 이다(`modeling_deepseek_v3.py:419`). o_proj 가 `nn.Linear(num_heads * v_head_dim, hidden_size)` (:401-402)이므로 합쳐진 폭은 실제로 `n_h*d_v` 로 맞게 렌더된다 — 그래서 `view [B,T,n_h,d_nope] -> [B,T,n_h*d_v]` 한 행 안에서 두 설명이 어긋난다(모델당 61행, 총 195행).

**고치지 못했다. 시도한 것과 결과를 남긴다.** 등록된 `A+B` 의 피연산자 순서가 소스의 split 순서 그대로라는 점을 이용해 조각을 A·B 로 이름 붙이는 규칙을 넣어 봤다(`_split_from_registered_sum`). split 출력은 맞게 바뀌었지만 **그 아래 사슬 전체가 옛 이름을 유지**해서 reshape 불일치가 61 → 122 로, flow_ambig 가 0 → 122 로 늘었다. `_propagate_labels` 는 monotone 이라(빈 정수만 채운다) 이름을 덮어쓰지 않는다. 이건 이 저장소가 이미 두 번 측정한 실패 형태다 — `_carry_reshape_labels` 가 같은 이유로 비활성 상태다. 제대로 고치려면 **권위 있는 개명을 데이터플로우를 따라 끝까지 옮기는** 기계장치가 필요하고, 그건 값이 겹치는 축에서 안전하다는 보장이 아직 없다. 값으로 우기지 않고 남긴다(P1).

## 발견 5 — 교정 필요 (미반영)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.self_attn` |
| 축 | q/k split 둘째 조각 (64) |
| 현재 라벨 | `d_head` |
| 판정 | `should_be_renamed` |
| 제안 라벨 | `d_rope` |
| 확신도 | medium |
| 산출물 반영 | 미반영 |

**근거**

`split_with_sizes [B,n_h,T,d_nope+d_rope] -> [B,n_h,T,d_nope], [B,n_h,T,d_head]` — 둘째 조각은 RoPE 를 받는 부분이므로 `d_rope` 다. 이 모델들은 head_dim == qk_rope_head_dim == 64 라 값이 겹친다. 위와 **정확히 같은 원인·같은 막힘**이라 함께 남긴다.

## 발견 6 — 교정 필요 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.self_attn` |
| 축 | `view_as_real` 이 더하는 마지막 축 (2) |
| 현재 라벨 | `E_shared` |
| 판정 | `should_be_renamed` |
| 제안 라벨 | `정수 2` |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

`view [B,n_h,T,d_head] -> [B,n_h,T,d_rope/2,2]` 를 `view_as_real` 이 소비한다(실측 `[1,16,17,32,2]`). 복소수 하나의 실수부·허수부 쌍이지 아키텍처 차원이 아니다 — RoPE 를 복소수 곱으로 구현하는 표준 형태다. 이 모델은 공유 전문가 수도 2 라 그 이름이 붙었다. **정수로 두는 것이 정답**이고, B절에서 같은 축을 이미 그렇게 판정했는데 산출물에는 아직 `E_shared` 로 남아 있다 — 그때 기록을 과대 기술했다. 여기서 정정한다. `E_shared` 는 `group: moe` 라 스코프 밖 폴백에서는 배제되므로, 재사용 또는 전파로 들어온 경로다. 값으로 우기지 않고 남긴다.

**산출물에 반영됨(2026-08-12).** 규칙을 고쳐 재추론하는 방식은 사슬이 어긋나 두 번 되돌렸으므로, 렌더가 끝난 뒤 선언된 모듈 아래의 이름을 바꾸는 경로를 만들었다 — `rules/label_overrides.yaml` (근거 인용·기대 크기 필수, 발화 0건이면 게이트 FAIL). 적용 내역은 `full/label_overrides.json`, 절차는 `review/05-overrides.md`.

## 발견 7 — 교정 필요 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.self_attn.q_b_proj` |
| 축 | q_b_proj 출력 폭 (MLA Q 업투영) |
| 현재 라벨 | `(n_h+2*n_kv)*d_head` |
| 판정 | `should_be_renamed` |
| 제안 라벨 | `n_h*(d_nope+d_rope)` |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

`modeling_deepseek.py:669` `self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.q_head_dim, bias=False)` — MLA 의 Q 저랭크 업투영이다. 그런데 `(n_h+2*n_kv)*d_head` 라는 **fused QKV 폭**(falcon/GPT-2/Phi-4 의 것)이 붙어 있었다. Kimi-K2.6 은 (64+2·64)·64 = 64·192 = 12288 로 값이 정확히 같다. **MLA 에는 n_kv 라는 개념 자체가 없다.**

이 모델들은 transformers 본체에 `kimi_k2` 파일이 없어 소스 대조가 통째로 '수행되지 않음'이었다 — 저장소를 열자 소속 검사가 즉시 잡았다. 교정: fused QKV 규칙 스코프에서 `q_b_proj` 를 배제하고, MLA 의 `n_h*(d_nope+d_rope)` 를 그 앞으로 옮겼다. `n_h*d_v` 는 그보다도 앞에 둔다(GLM-5.2 는 d_v = d_nope+d_rope = 256).
