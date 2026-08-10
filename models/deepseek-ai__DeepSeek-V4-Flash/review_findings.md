# 라벨 검토 결과 — deepseek-ai/DeepSeek-V4-Flash

- 검토일: 2026-08-09
- 검토자: llm(claude, 소스 직접 대조)
- 본 것: 의뢰서 39건 전수 (review/ 절차) — 별칭 접지 / 정사각 축 / 미등록 필드 / 산술로 지은 이름. 39 -> 5건으로 축소
- 요약: 의뢰서 1건 — OLMoE 와 같은 원인의 오라벨. 같은 경로로 부분 교정했다.

> 이 파일은 `review_findings.json` 에서 생성된다 — 고칠 때는 JSON 을 고친다.

## 발견 1 — 교정 필요 (미반영)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.mlp.experts` |
| 축 | [E, d_model, d_model] 의 가운데 축 (4096) |
| 현재 라벨 | `d_model` |
| 판정 | `should_be_renamed` |
| 제안 라벨 | `2*d_moe` |
| 확신도 | high |
| 산출물 반영 | 미반영 |

**근거**

`modeling_deepseek_v4.py:992` `self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))`. d_moe=2048 이라 2·2048=4096=d_model 로 겹친다. OLMoE 와 같은 경로로 부분 교정됐다 — 활성화 사슬과 down_proj 는 맞고, 융합 가중치의 가운데 축만 남았다(사유 동일).

## 발견 2 — 교정 필요 (미반영)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.self_attn.compressor.indexer` |
| 축 | [B, T/m_csa, 4, c_I] 의 셋째 축 |
| 현재 라벨 | `4 (이름 없음)` |
| 판정 | `should_be_renamed` |
| 제안 라벨 | `m_csa` |
| 확신도 | medium |
| 산출물 반영 | 미반영 |

**근거**

indexer 안의 `[B, T/m_csa, 4, c_I]` 는 압축 엔트리마다 그것이 덮는 원본 토큰 m_csa 개다(m_csa=4). 그런데 `m_csa` 의 스코프가 `compressor(?!\.indexer)` 라 이름이 안 붙고 정수로 남는다. 그 배제는 원래 **m_hca(=128)가 c_I(=128)를 뺏는 것**을 막으려고 넣은 것이라, 값이 겹치지 않는 m_csa 까지 막을 이유가 없다. 배제를 `compressor` 로 여는 것을 시도했으나 되돌렸다 — V4-Pro 의 heur 가 2,131 -> 3,331 로 퇴행한다(indexer 안에서 4 가 다른 축까지 가져가고 128 자리에 T/m_hca 가 밀려든다). 심볼 하나만 스코프를 여는 문법이 없어 그대로 둔다. **이 건은 미결 4범주(별칭·정사각·미등록·휴리스틱) 어디에도 안 걸렸고, 의뢰서에 새로 넣은 전수 점검 B절(이름 없는 정수 x 같은 값의 심볼)이 처음 드러냈다.**
