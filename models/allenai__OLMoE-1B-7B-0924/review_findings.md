# 라벨 검토 결과 — allenai/OLMoE-1B-7B-0924

- 검토일: 2026-08-12
- 검토자: llm(claude, 자기모순 추적 + 소스 대조)
- 본 것: A·B·C절 전건 수행 완료 + 게이트가 센 자기모순을 출발점으로 하이브리드 스택 추적. 모집단·선별 기준은 review/04-full-inventory.md.
- 요약: 의뢰서 1건 — 전문가 가중치가 확인된 오라벨. 하류 split 증거로 부분 교정했고, 융합 가중치 축 순서만 남았다.

> 이 파일은 `review_findings.json` 에서 생성된다 — 고칠 때는 JSON 을 고친다.

## 발견 1 — 교정 필요 (미반영)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.mlp.experts` |
| 축 | [E, d_model, d_model] 의 가운데 축 (2048) |
| 현재 라벨 | `d_model` |
| 판정 | `should_be_renamed` |
| 제안 라벨 | `2*d_moe` |
| 확신도 | high |
| 산출물 반영 | 미반영 |

**근거**

`modeling_olmoe.py:297` `self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))`. intermediate_size=1024 이므로 2·1024=2048 이고, hidden_size 도 2048 이라 값이 겹쳐 d_model 로 붙었다. **부분 교정됨(2026-08-09).** 값·스코프로는 못 가려서(스코프로 이기게 했더니 잔차 스트림까지 바뀌어 flow_ambig 32→64) 하류 증거를 쓰는 규칙을 새로 넣었다 — `build_table._merge_from_split`: 한 축이 n 등분되어 같은 이름 n 개가 되면 그 축은 n 배다. 바로 다음 op 이 `split [[k*T, d_model]] -> [[k*T, d_moe], [k*T, d_moe]]` 로 한 행 안에서 자기모순을 드러내고 있었다. 이제 grouped_matmul 출력이 `[k*T, 2*d_moe]`, down_proj 가중치가 `[E, d_model, d_moe]` 로 맞게 나온다. **남은 것**: 융합 가중치 자체의 가운데 축은 여전히 `d_model` 이다 — 뒤 두 축이 둘 다 2048 이라 순서를 가릴 증거가 트레이스 안에 없다(`_weight_out_from_output` 이 이 경우를 의도적으로 건드리지 않는다). 소스는 `[E, 2*d_moe, d_model]` 이라고 말하지만, 그걸 반영하려면 소스에서 nn.Parameter 축 순서를 읽어오는 기계장치가 필요하다.
