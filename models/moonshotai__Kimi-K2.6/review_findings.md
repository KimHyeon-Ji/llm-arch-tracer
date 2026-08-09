# 라벨 검토 결과 — moonshotai/Kimi-K2.6

- 검토일: 2026-08-09
- 검토자: llm(claude, 소스 직접 대조)
- 본 것: 신규 온보딩 검토 — 의뢰서 전수 + 소스 대조
- 요약: MLA + MoE 가 전부 등록된 규칙으로 해결됐다 — **새 규칙 0개, 휴리스틱 0.00%, 미등록 config 필드 0, 의뢰서 비어 있음.**

> 이 파일은 `review_findings.json` 에서 생성된다 — 고칠 때는 JSON 을 고친다.

## 발견 1 — 맞음 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `(로딩)` |
| 축 | native 구현 선택 |
| 현재 라벨 | `kimi_k25 래퍼의 텍스트 타워` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

config 스스로 `architectures: [DeepseekV3ForCausalLM]` 이고, 저장소가 번들한 remote code 는 DeepSeek-V3 modeling 의 사본이다(클래스가 전부 `DeepseekV3*`). 그 사본이 transformers 5.x 에서 import 되지 않아(`is_torch_fx_available` 제거) 유지되는 native 구현으로 로드했다 — 아키텍처를 바꾼 것이 아니라 같은 아키텍처의 다른 구현을 쓴 것이다. 실측 심볼: L=61, d_model=7168, MLA(c_q=1536, c_kv=512, d_nope=128, d_rope=64, d_v=128), MoE E=384 top-8 + shared 1 — DeepSeek-V3 와 같은 형태이고 규모만 다르다. K2.6 은 멀티모달 래퍼라 텍스트 타워 config 가 이미 native `DeepseekV3Config` 다. 그런데 `needs_remote_code` 가 `model_type` 문자열(`kimi_k2`)로 판정해 remote code 로 보내고 있었다 — 판정 기준을 **config 클래스의 출처**로 바꿨다(`src/provenance.py`).
