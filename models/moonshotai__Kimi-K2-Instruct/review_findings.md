# 라벨 검토 결과 — moonshotai/Kimi-K2-Instruct

- 검토일: 2026-08-11
- 검토자: llm(claude, 전수 점검 2회차 — 모듈-필드 소속)
- 본 것: 전수 점검 2회차 — 1층(모듈-필드 소속: 가중치 축의 이름이 그 모듈/부모가 실제로 읽는 config 필드에서 나왔는가)을 함대 전건 수행. 값을 보지 않는 검사라 값 충돌이 숨길 수 없다. 1회차의 A절·B절 판정은 유지. C절(모듈별 출력 shape)은 여전히 미수행.
- 요약: MLA + MoE 가 전부 등록된 규칙으로 해결됐다 — **새 규칙 0개, 휴리스틱 0.00%, 미등록 config 필드 0, 의뢰서 비어 있음.**

> 이 파일은 `review_findings.json` 에서 생성된다 — 고칠 때는 JSON 을 고친다.

## 발견 1 — 맞음 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `(로딩)` |
| 축 | native 구현 선택 |
| 현재 라벨 | `kimi_k2 (remote code)` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

config 스스로 `architectures: [DeepseekV3ForCausalLM]` 이고, 저장소가 번들한 remote code 는 DeepSeek-V3 modeling 의 사본이다(클래스가 전부 `DeepseekV3*`). 그 사본이 transformers 5.x 에서 import 되지 않아(`is_torch_fx_available` 제거) 유지되는 native 구현으로 로드했다 — 아키텍처를 바꾼 것이 아니라 같은 아키텍처의 다른 구현을 쓴 것이다. 실측 심볼: L=61, d_model=7168, MLA(c_q=1536, c_kv=512, d_nope=128, d_rope=64, d_v=128), MoE E=384 top-8 + shared 1 — DeepSeek-V3 와 같은 형태이고 규모만 다르다. 프로파일에 `config_overrides: {model_type: deepseek_v3, qk_head_dim: 192}` 를 둔다. qk_head_dim 은 native config 가 `__post_init__` 에서 계산하는 값과 같은 식이다(`configuration_deepseek_v3.py:123`, 128+64).
