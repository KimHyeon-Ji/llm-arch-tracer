# 라벨 검토 결과 — moonshotai/Kimi-K2.7-Code

- 검토일: 2026-08-10
- 검토자: llm(claude, 전수 점검 + 소스 대조)
- 본 것: 의뢰서 전수 점검 1회차 — A절(붙은 이름 전부 x 나타나는 모듈) 함대 스윕과 B절(이름 없는 정수 x 같은 값의 심볼) 전건 판정. C절(모듈별 출력 shape)은 미수행.
- 요약: 의뢰서가 비어 있었다 — K2.6 과 같은 경로(멀티모달 래퍼의 텍스트 타워가 native DeepSeek-V3 config)로 새 규칙 0개, 휴리스틱 0.00%.

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

`model_type: kimi_k25` 는 transformers 에 공식 편입돼 있고(2026-07-03), 텍스트 타워는 `kimi_k2` = DeepSeek-V3 아키텍처다. `provenance.needs_remote_code` 가 config 클래스의 출처로 판정하도록 고쳐둔 덕에 native 경로로 그대로 로드된다. 실측 L=61, d=7168, MLA(c_q=1536, c_kv=512, d_nope=128, d_rope=64, d_v=128), MoE E=384 top-8 + shared 1.
