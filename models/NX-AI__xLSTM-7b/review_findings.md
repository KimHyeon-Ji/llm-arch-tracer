# 라벨 검토 결과 — NX-AI/xLSTM-7b

- 검토일: 2026-08-12
- 검토자: llm(claude, C절 전수 + 소스 대조)
- 본 것: **A·B·C절 전건 수행 완료.** C절은 (모듈, 라벨) 쌍 8,886건을 모집단으로 삼고, 심볼 자신의 scope 가 그 모듈을 덮지 않는 경우를 기계로 선별해(등록 유도식이 그 모듈 스코프로 설명하는 라벨은 제외) 20건을 전건 판정했다. 9건은 규칙 교정으로 닫았고 11건은 판정과 함께 남는다. 모집단·선별 기준은 review/04-full-inventory.md.
- 요약: 의뢰서 3건 전부 오탐이었다. 라벨이 옳다.

> 이 파일은 `review_findings.json` 에서 생성된다 — 고칠 때는 JSON 을 고친다.

## 발견 1 — 맞음 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `(config)` |
| 축 | d_v 가 읽은 v_head_dim |
| 현재 라벨 | `v_head_dim` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

`configuration_xlstm.py:179` `@property def v_head_dim(self): return self.v_dim // self.num_heads`. v_dim_factor=1.0 이라 v_dim=embedding_dim=4096, 4096/8=512 로 실측과 일치.

## 발견 2 — 맞음 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `(config)` |
| 축 | d_head 가 읽은 head_dim |
| 현재 라벨 | `head_dim` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

체크포인트 config.json 이 `head_dim: 512` 를 직접 준다(HF 원본 확인). qk_dim_factor=0.5 라 Q/K 의 head 폭은 256 이고 **그 축들은 d_head 가 아니라 `d_model*qk_f` 로 올바르게 렌더되고 있다**(실측: q/k 가중치 `[d_model*qk_f, d_model]`).

## 발견 3 — 맞음 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.blocks.*.mlstm_layer.v / out_proj` |
| 축 | [d_model, d_model] |
| 현재 라벨 | `d_model (두 번)` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

v_dim_factor=1.0 이라 V 투영이 정사각 `[4096, 4096]` 이고 out_proj 도 마찬가지다. 정사각 reshape 이 아니라 **정사각 가중치**이므로 같은 이름 두 번이 정상 — 탐지기가 reshape 만 찾아서 미확인으로 남겼다.

## 발견 4 — 교정 필요 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `backbone.blocks.*.mlstm_layer.mlstm_backend` |
| 축 | head 폭 (18,752축) |
| 현재 라벨 | `d_head (스코프 밖)` |
| 판정 | `should_be_renamed` |
| 제안 라벨 | `d_head (스코프 확장)` |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

`n_h` 는 스코프에 `mlstm|mixer` 를 갖는데 `d_head` 는 `attn|attention|rotary` 뿐이었다. head 수가 정당한 자리에서는 head 폭도 정당하므로 비대칭이다 — mLSTM 백엔드 안의 head 폭 18,752축이 스코프 밖 이름으로 붙어 있었고, 폴백이 막히면 정수로 떨어질 자리였다. `d_head` 스코프에 `mlstm|mixer` 를 추가했다(게이트: 퇴행 0 / 개선 1). A절 함대 스윕에서 드러났다.

## 발견 5 — 맞음 (미반영)

| 항목 | 값 |
|---|---|
| 모듈 | `backbone` |
| 축 | mLSTM 상태 버퍼의 head 축·head 폭 |
| 현재 라벨 | `n_h / d_head (스코프 밖 폴백)` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 미반영 |

**근거**

xLSTM 은 mLSTM 순환 상태 버퍼를 블록 안이 아니라 **스택 루트**에서 할당한다 — `zeros() -> [B, n_h, d_model*qk_f/n_h, d_head]` (실측 `[1,8,256,512]`). 두 이름 다 정확히 맞고, 근거만 '스코프 밖 폴백'이라 약하다. 스코프에 `backbone` 을 추가해 봤으나 **퇴행했다** — 휴리스틱 0 → 768, flow_ambig 0 → 192. 루트는 온갖 텐서가 지나가는 자리라 거기서 head 이름을 in-scope 로 올리면 다른 축까지 가져간다. 되돌리고 남긴다.
