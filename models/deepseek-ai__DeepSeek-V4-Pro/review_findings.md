# 라벨 검토 결과 — deepseek-ai/DeepSeek-V4-Pro

- 검토일: 2026-08-09
- 검토자: llm(claude, 소스 직접 대조)
- 본 것: 의뢰서 39건 전수 — 별칭 접지 / 정사각 축 / 미등록 필드 / 산술로 지은 이름 / `develop/sources/` 의 실제 modeling·configuration 소스
- 요약: 의뢰서 1건 — 같은 op 의 입력과 출력이 서로 다르게 렌더된 것을 찾았다.

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

`modeling_deepseek_v4.py:382` `self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=...)` — RMSNorm 은 마지막 축을 정규화하므로 **마지막 축이 d_head(512)** 이고 가운데가 압축 KV 길이 `T/m_csa`(마침 512)다. 지금은 같은 elementwise_mul 의 입력이 `[B, d_head, T/m_csa]`, 출력이 `[B, d_head, d_head]` 로 서로 다르게 렌더된다 — 둘 다 축 순서가 틀렸다. 값 충돌(512=512)이라 규칙으로는 못 끊고, norm 모듈의 rank-1 파라미터 폭으로 마지막 축을 고정하는 앵커가 필요하다.
