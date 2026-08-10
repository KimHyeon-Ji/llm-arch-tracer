# 라벨 검토 결과 — tiiuae/falcon-7b

- 검토일: 2026-08-10
- 검토자: llm(claude, 전수 점검 + 소스 대조)
- 본 것: 의뢰서 전수 점검 1회차 — A절(붙은 이름 전부 x 나타나는 모듈) 함대 스윕과 B절(이름 없는 정수 x 같은 값의 심볼) 전건 판정. C절(모듈별 출력 shape)은 미수행.
- 요약: 의뢰서 3건 중 1건이 실제 오라벨(FFN 폭), 2건은 오탐이었다.

> 이 파일은 `review_findings.json` 에서 생성된다 — 고칠 때는 JSON 을 고친다.

## 발견 1 — 맞음 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `(config)` |
| 축 | d_head 가 읽은 head_dim |
| 현재 라벨 | `head_dim` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

`configuration_falcon.py:102` `@property def head_dim(self): return self.hidden_size // self.num_attention_heads` — 클래스가 프로퍼티로 정의한다. 탐지기가 프로퍼티를 못 읽던 것이 원인이라 `src/source_check.py._config_fields` 에 프로퍼티 수집을 넣었다.

## 발견 2 — 교정 필요 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `transformer.h.*.mlp.dense_h_to_4h / dense_4h_to_h` |
| 축 | FFN 중간 폭 18176 |
| 현재 라벨 | `4*d_model` |
| 판정 | `should_be_renamed` |
| 제안 라벨 | `d_ff` |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

`modeling_falcon.py:531` `self.dense_h_to_4h = FalconLinear(hidden_size, config.ffn_hidden_size, ...)` + `configuration_falcon.py:96` `if self.ffn_hidden_size is None: self.ffn_hidden_size = self.hidden_size * 4`. 폭의 이름은 `ffn_hidden_size` 이고 4배는 그 기본값일 뿐이다. `rules/symbols.yaml` 의 d_ff 별칭에 `ffn_hidden_size` 를 추가했다.
