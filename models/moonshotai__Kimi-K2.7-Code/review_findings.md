# 라벨 검토 결과 — moonshotai/Kimi-K2.7-Code

- 검토일: 2026-08-12
- 검토자: llm(claude, 외부 검토 지적 반영 + 미답변 4건 판정)
- 본 것: 외부 검토가 지적한 3건(layer_sched 소실 / 검토 미수행 / 가중치-피연산자 불일치)을 재현·반영하고, 새로 배선한 미답변 검사가 잡은 4건에 답했다.
- 요약: 의뢰서가 비어 있었다 — K2.6 과 같은 경로(멀티모달 래퍼의 텍스트 타워가 native DeepSeek-V3 config)로 새 규칙 0개, 휴리스틱 0.00%.

> 이 파일은 `review_findings.json` 에서 생성된다 — 고칠 때는 JSON 을 고친다.

## 발견 1 — 맞음 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `(로딩)` |
| 축 | native 구현 선택 |
| 현재 라벨 | `kimi_k25 래퍼의 텍스트 타워` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

`model_type: kimi_k25` 는 transformers 에 공식 편입돼 있고(2026-07-03), 텍스트 타워는 `kimi_k2` = DeepSeek-V3 아키텍처다. `provenance.needs_remote_code` 가 config 클래스의 출처로 판정하도록 고쳐둔 덕에 native 경로로 그대로 로드된다. 실측 L=61, d=7168, MLA(c_q=1536, c_kv=512, d_nope=128, d_rope=64, d_v=128), MoE E=384 top-8 + shared 1.

## 발견 2 — 교정 필요 (반영됨)

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

## 발견 3 — 교정 필요 (미반영)

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

## 발견 4 — 교정 필요 (미반영)

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
