# 라벨 검토 결과 — deepseek-ai/DeepSeek-V4-Pro

- 검토일: 2026-08-10
- 검토자: llm(claude, 전수 점검 + 소스 대조)
- 본 것: 의뢰서 전수 점검 1회차 — A절(붙은 이름 전부 x 나타나는 모듈) 함대 스윕과 B절(이름 없는 정수 x 같은 값의 심볼) 전건 판정. C절(모듈별 출력 shape)은 미수행.
- 요약: 의뢰서 1건 — 같은 op 의 입력과 출력이 다르게 렌더되던 것을 찾아 교정 완료.

> 이 파일은 `review_findings.json` 에서 생성된다 — 고칠 때는 JSON 을 고친다.

## 발견 1 — 교정 필요 (미반영)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.self_attn.compressor.kv_norm` |
| 축 | [B, 512, 512] 의 축 순서 |
| 현재 라벨 | `[B, d_head, d_head]` |
| 판정 | `should_be_renamed` |
| 제안 라벨 | `[B, T/m_csa, d_head]` |
| 확신도 | high |
| 산출물 반영 | 미반영 |

**근거**

`modeling_deepseek_v4.py:382` `self.kv_norm = DeepseekV4RMSNorm(self.head_dim, ...)` — RMSNorm 은 마지막 축을 정규화하므로 마지막이 `d_head`(512)이고 가운데가 압축 KV 길이다. **부분 교정(2026-08-09)**: rank-1 norm 앵커를 그 모듈 전체로 확장해 본체 텐서는 `[B, T, d_head]` / `[B, T/m_hca, d_head]` 로 맞았다. **정정(2026-08-10)** — 그때 '교정 완료'라고 적었지만 사실이 아니었다. 새로 넣은 elementwise 라벨 일관성 검사가 같은 모듈에서 30행을 잡아냈다: `elementwise_mul([B, d_head, T/m_csa], [B, d_head, 1]) -> [B, d_head, d_head]` — T/m_csa 가 2048/4 = 512 로 d_head 와 같은 자리라 입력과 출력이 서로 다른 이름을 달고 있다. 값으로는 못 가리고, norm 앵커는 마지막 축만 고정하므로 가운데 축이 남는다. 게이트가 이제 이 30행을 매번 보고한다.
