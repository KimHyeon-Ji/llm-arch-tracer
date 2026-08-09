# 라벨 검토 결과 — tiiuae/falcon-7b

- 검토일: 2026-08-09
- 검토자: llm(claude, 소스 직접 대조)
- 본 것: 의뢰서 39건 전수 (review/ 절차) — 별칭 접지 / 정사각 축 / 미등록 필드 / 산술로 지은 이름. 39 -> 5건으로 축소 / `develop/sources/` 의 실제 modeling·configuration 소스
- 요약: 의뢰서 3건 중 1건이 실제 오라벨(FFN 폭), 2건은 오탐이었다.

## 발견 1 — 맞음

| 항목 | 값 |
|---|---|
| 모듈 | `(config)` |
| 축 | d_head 가 읽은 head_dim |
| 현재 라벨 | `head_dim` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |

**근거**

`configuration_falcon.py:102` `@property def head_dim(self): return self.hidden_size // self.num_attention_heads` — 클래스가 프로퍼티로 정의한다. 탐지기가 프로퍼티를 못 읽던 것이 원인이라 `src/source_check.py._config_fields` 에 프로퍼티 수집을 넣었다.

## 발견 2 — 교정 필요

| 항목 | 값 |
|---|---|
| 모듈 | `transformer.h.*.mlp.dense_h_to_4h / dense_4h_to_h` |
| 축 | FFN 중간 폭 18176 |
| 현재 라벨 | `4*d_model` |
| 판정 | `should_be_renamed` |
| 제안 라벨 | `d_ff` |
| 확신도 | high |

**근거**

`modeling_falcon.py:531` `self.dense_h_to_4h = FalconLinear(hidden_size, config.ffn_hidden_size, ...)` + `configuration_falcon.py:96` `if self.ffn_hidden_size is None: self.ffn_hidden_size = self.hidden_size * 4`. 폭의 이름은 `ffn_hidden_size` 이고 4배는 그 기본값일 뿐이다. `rules/symbols.yaml` 의 d_ff 별칭에 `ffn_hidden_size` 를 추가했다.
