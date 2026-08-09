# 라벨 검토 결과 — deepseek-ai/DeepSeek-V4-Pro

- 검토일: 2026-08-09
- 검토자: llm(claude, 소스 직접 대조)
- 본 것: 의뢰서 39건 전수 — 별칭 접지 / 정사각 축 / 미등록 필드 / 산술로 지은 이름 / `develop/sources/` 의 실제 modeling·configuration 소스
- 요약: 의뢰서 1건 — 같은 op 의 입력과 출력이 다르게 렌더되던 것을 찾아 교정 완료.

## 발견 1 — 교정 필요

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.self_attn.compressor.kv_norm` |
| 축 | [B, 512, 512] 의 축 순서 |
| 현재 라벨 | `[B, d_head, d_head]` |
| 판정 | `should_be_renamed` |
| 제안 라벨 | `[B, T/m_csa, d_head]` |
| 확신도 | high |

**근거**

`modeling_deepseek_v4.py:382` `self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=...)` — RMSNorm 은 마지막 축을 정규화하므로 **마지막 축이 d_head(512)** 이고 가운데가 압축 KV 길이 `T/m_csa`(마침 512)다. 지금은 같은 elementwise_mul 의 입력이 `[B, d_head, T/m_csa]`, 출력이 `[B, d_head, d_head]` 로 서로 다르게 렌더됐다 — 둘 다 축 순서가 틀렸다. **교정 완료(2026-08-09)**: rank-1 norm 앵커는 가중치를 소비하는 op 에만 걸려서 같은 norm 안의 pow/mean/mul 은 값 매칭으로 떨어지고 있었다. norm 은 정의상 마지막 축을 정규화하므로 앵커를 그 모듈 전체로 확장했다(`src/anchors.py`, rank-1 한정 — rank>=2 앵커는 in/out 을 가릴 파라미터가 필요하다). 지금은 `[B, T, d_head]` / `[B, T/m_hca, d_head]` 로 나온다. **어떤 자동 지표도 이 오류를 보지 못했다** — 고친 뒤에도 게이트 지표는 하나도 움직이지 않았다. ③ 검토가 존재하는 이유가 정확히 이 부류다.
