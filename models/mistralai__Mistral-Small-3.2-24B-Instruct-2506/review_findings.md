# 라벨 검토 결과 — mistralai/Mistral-Small-3.2-24B-Instruct-2506

- 검토일: 2026-08-12
- 검토자: llm(claude, 반박 프레임 전건 판정)
- 본 것: 의뢰서의 **모든** 질문에 답한다(기계가 개수를 맞춘다). 확인 프레임이 아니라 반박 프레임으로 — 각 라벨에 대해 '틀렸다는 증거'를 먼저 찾고, 못 찾은 것만 맞다고 적었다. 외부 검토가 준 팁 3가지(op 내부 필드 상호 대조 / 요청·응답 개수 diff / 반박 프레임)를 그대로 적용했다.
- 요약: 외부 검토(Codex) 확인 완료 -- 멀티모달 스코프 지적은 의도된 설계로 판정, 나머지 전부 일치.

> 이 파일은 `review_findings.json` 에서 생성된다 — 고칠 때는 JSON 을 고친다.

## 발견 1 — 맞음 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `(root)` |
| 축 | model_summary.md의 total_params(23.57B)와 모델 타입 표기가 텍스트 백본만 설명한다 |
| 현재 라벨 | `text backbone만 (Mistral3ForConditionalGeneration 멀티모달 wrapper의 Pixtral vision encoder 제외)` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

외부 검토(Codex, 2026-08-31)가 '실제 outer model은 Mistral3ForConditionalGeneration 멀티모달 모델이고 Pixtral vision encoder가 별도로 있는데 결과물은 텍스트 백본(23.57B)만 설명한다'고 지적했다. 이건 결함이 아니라 이 파이프라인의 명시적 설계다 -- src/provenance.py:90-96 'Multimodal/composite configs ... nest the LLM under text_config; we trace the text decoder, so hand back the text sub-config' -- 모든 멀티모달 체크포인트에 일관되게 적용되는 스코프 결정이지 Mistral만의 누락이 아니다. 사용자 확인(2026-08-31): 멀티모달 부분은 이 프로젝트 범위 밖, 텍스트 백본만 다루는 것이 맞는 방향. 핵심 수치(L=40, d_model=5120, n_h/n_kv=32/8, d_head=128, d_ff=32768, dense GQA, 131K context)는 Codex가 전부 일치 판정했다.
