# 라벨 검토 결과 — NX-AI/xLSTM-7b

- 검토일: 2026-08-09
- 검토자: llm(claude, 소스 직접 대조)
- 본 것: 의뢰서 39건 전수 — 별칭 접지 / 정사각 축 / 미등록 필드 / 산술로 지은 이름 / `develop/sources/` 의 실제 modeling·configuration 소스
- 요약: 의뢰서 3건 전부 오탐이었다. 라벨이 옳다.

## 발견 1 — 맞음

| 항목 | 값 |
|---|---|
| 모듈 | `(config)` |
| 축 | d_v 가 읽은 v_head_dim |
| 현재 라벨 | `v_head_dim` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |

**근거**

`configuration_xlstm.py:179` `@property def v_head_dim(self): return self.v_dim // self.num_heads`. v_dim_factor=1.0 이라 v_dim=embedding_dim=4096, 4096/8=512 로 실측과 일치.

## 발견 2 — 맞음

| 항목 | 값 |
|---|---|
| 모듈 | `(config)` |
| 축 | d_head 가 읽은 head_dim |
| 현재 라벨 | `head_dim` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |

**근거**

체크포인트 config.json 이 `head_dim: 512` 를 직접 준다(HF 원본 확인). qk_dim_factor=0.5 라 Q/K 의 head 폭은 256 이고 **그 축들은 d_head 가 아니라 `d_model*qk_f` 로 올바르게 렌더되고 있다**(실측: q/k 가중치 `[d_model*qk_f, d_model]`).

## 발견 3 — 맞음

| 항목 | 값 |
|---|---|
| 모듈 | `model.blocks.*.mlstm_layer.v / out_proj` |
| 축 | [d_model, d_model] |
| 현재 라벨 | `d_model (두 번)` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |

**근거**

v_dim_factor=1.0 이라 V 투영이 정사각 `[4096, 4096]` 이고 out_proj 도 마찬가지다. 정사각 reshape 이 아니라 **정사각 가중치**이므로 같은 이름 두 번이 정상 — 탐지기가 reshape 만 찾아서 미확인으로 남겼다.
