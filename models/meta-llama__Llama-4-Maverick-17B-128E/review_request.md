# 검토 의뢰서 — meta-llama/Llama-4-Maverick-17B-128E

파이썬 파이프라인이 규칙으로 결정할 수 있는 것을 전부 결정하고, **판단이 필요한 것만** 여기 남겼다. 절차와 출력 형식은 `review/` 에 있다.

- transformers 모듈: `llama4_text`
- 판단 필요: **2건**

## 증거 — 이미 받아둔 실제 소스

- `develop/sources/modeling_llama4_text.py` — 있음, 이 파일을 열어서 판정한다
- `develop/sources/configuration_llama4_text.py` — 있음, 이 파일을 열어서 판정한다

그 밖의 재료: `models/<모델>/research_agenda.md`(미해결 축 + 볼 소스 URL), `full/review.md`(리뷰 패킷), `structure.yaml`(심볼 표), `source_check.md`(기계적으로 확인된 것).

## 판단이 필요한 것

### 3. 이름 붙일 근거가 없는 config 필드

모듈 폭으로 쓰이는데 심볼 표에 등록돼 있지 않다. 소스에서 무엇인지 확인하고 `rules/symbols.yaml` 에 등록하면 다음 실행부터 자동으로 잡힌다.

- `{'field': 'intermediate_size', 'value': 8192, 'modules': 24}`
- `{'field': 'expert_dim', 'value': 8192, 'modules': 24}`

## 이 의뢰서를 처리하는 법

`review/prompt.md` 를 LLM 에 넘기고 이 모델을 지정한다. 판정 4종과 근거 요건, 결과를 어디에 어떤 형식으로 쓰는지는 전부 `review/` 안에 있다.
