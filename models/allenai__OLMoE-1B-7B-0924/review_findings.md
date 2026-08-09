# 라벨 검토 결과 — allenai/OLMoE-1B-7B-0924

- 검토일: 2026-08-09
- 검토자: llm(claude, 소스 직접 대조)
- 본 것: 의뢰서 39건 전수 — 별칭 접지 / 정사각 축 / 미등록 필드 / 산술로 지은 이름 / `develop/sources/` 의 실제 modeling·configuration 소스
- 요약: 의뢰서 1건 — 전문가 가중치가 확인된 오라벨이다. 값 충돌이라 규칙으로는 못 끊어 미해결로 남겼다.

## 발견 1 — 교정 필요

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.mlp.experts` |
| 축 | [E, d_model, d_model] 의 가운데 축 (2048) |
| 현재 라벨 | `d_model` |
| 판정 | `should_be_renamed` |
| 제안 라벨 | `2*d_moe` |
| 확신도 | high |

**근거**

`modeling_olmoe.py:297` `self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))`. intermediate_size=1024 이므로 2·1024=2048 이고, hidden_size 도 2048 이라 값이 겹쳐 d_model 로 붙었다. **아직 교정되지 않았다**: 같은 모듈 안에 잔차 스트림 `[T, d_model]` 과 gate_up 출력 `[k*T, 2*d_moe]` 이 함께 있어 값·스코프로는 못 가린다(스코프로 이기게 했더니 잔차까지 바뀌어 flow_ambig 32→64). 올바른 해법은 `nn.Parameter` 의 선언 폭을 읽는 앵커 — `src/anchors.py` 는 지금 nn.Linear 만 본다.
