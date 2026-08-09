# 검토 의뢰서 — deepseek-ai/DeepSeek-V3

파이썬 파이프라인이 규칙으로 결정할 수 있는 것을 전부 결정하고, **판단이 필요한 것만** 여기 남겼다. 절차와 출력 형식은 `review/` 에 있다.

- transformers 모듈: `deepseek_v3`
- 판단 필요: **0건**

## 증거 — 이미 받아둔 실제 소스

- `develop/sources/modeling_deepseek_v3.py` — 있음, 이 파일을 열어서 판정한다
- `develop/sources/configuration_deepseek_v3.py` — 있음, 이 파일을 열어서 판정한다

그 밖의 재료: `models/<모델>/research_agenda.md`(미해결 축 + 볼 소스 URL), `full/review.md`(리뷰 패킷), `structure.yaml`(심볼 표), `source_check.md`(기계적으로 확인된 것).

## 판단이 필요한 것

없다. 이 모델의 축은 전부 등록된 규칙이 이름을 냈고, 소스 대조도 어긋난 곳이 없다.

그래도 검토를 돌린다면 `full/review.md` 의 표본을 보고 규칙 자체가 틀리지 않았는지를 본다 — 그것이 규칙 게이트가 구조적으로 못 보는 부분이다.

## 이 의뢰서를 처리하는 법

`review/prompt.md` 를 LLM 에 넘기고 이 모델을 지정한다. 판정 4종과 근거 요건, 결과를 어디에 어떤 형식으로 쓰는지는 전부 `review/` 안에 있다.
