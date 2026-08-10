# 라벨 검토 결과 — deepseek-ai/DeepSeek-V4-Flash-0731

- 검토일: 2026-08-10
- 검토자: llm(claude, 소스 직접 대조)
- 본 것: 신규 온보딩 검토 — 의뢰서 전수 + 소스 대조
- 요약: 의뢰서 3건 — 전부 이름이 있는 축이었고 규칙으로 등록해 해소했다(현재 0건).

> 이 파일은 `review_findings.json` 에서 생성된다 — 고칠 때는 JSON 을 고친다.

## 발견 1 — 교정 필요 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.hc_head` |
| 축 | [B, T, 16384] |
| 현재 라벨 | `4*d_model` |
| 판정 | `should_be_renamed` |
| 제안 라벨 | `n_hc*d_model` |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

n_hc=4, d_model=4096 이라 4·d_model 과 값이 같았지만 어텐션 head 와는 무관한 축이다. mHC 는 잔차를 n_hc 개 병렬 스트림으로 들고 다니고, 그 스택을 한 줄로 편 폭이 n_hc·d_model 이다. 실측이 그대로 보여준다: `matmul [T, 16384] @ [16384, 4]` — hc_head 가 거기서 스트림별 혼합 가중치 4개를 뽑는다. `rules/derived_dims.yaml` 등록 완료.

## 발견 2 — 맞음 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.self_attn` |
| 축 | decode sliding score 폭 129 |
| 현재 라벨 | `w_local+1` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

w_local=128 이고 decode softmax 가 `[B, n_h, 1, 129]` 이다 — window 캐시 + 새 토큰 1. 이름은 맞았고 근거가 산술이라 휴리스틱으로 집계되던 것이라, 규칙으로 등록만 했다(gpt-oss 의 `w_local + n_sink` 와 같은 계열, sink 가 없는 경우).
