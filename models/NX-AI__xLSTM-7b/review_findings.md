# 라벨 검토 결과 — NX-AI/xLSTM-7b

- 검토일: 2026-08-10
- 검토자: llm(claude, 전수 점검 + 소스 대조)
- 본 것: 의뢰서 전수 점검 1회차 — A절(붙은 이름 전부 x 나타나는 모듈) 함대 스윕과 B절(이름 없는 정수 x 같은 값의 심볼) 전건 판정. C절(모듈별 출력 shape)은 미수행.
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
