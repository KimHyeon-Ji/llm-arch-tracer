# 검토 의뢰서 — Qwen/Qwen2.5-0.5B

파이썬 파이프라인이 규칙으로 결정할 수 있는 것을 전부 결정하고, **판단이 필요한 것만** 여기 남겼다. 절차와 출력 형식은 `review/` 에 있다.

- transformers 모듈: `qwen2`
- 판단 필요: **1건**

## 증거 — 이미 받아둔 실제 소스

- `develop/sources/modeling_qwen2.py` — 있음, 이 파일을 열어서 판정한다
- `develop/sources/configuration_qwen2.py` — 있음, 이 파일을 열어서 판정한다

그 밖의 재료: `models/<모델>/research_agenda.md`(미해결 축 + 볼 소스 URL), `full/review.md`(리뷰 패킷), `structure.yaml`(심볼 표), `source_check.md`(기계적으로 확인된 것).

## 판단이 필요한 것

### 2. 이 정사각 축이 정말 같은 이름 두 번인가

`[..., X, X]` 로 렌더됐는데, 그 이름이 읽은 config 필드에서 나온 정사각 reshape 을 modeling 소스에서 찾지 못했다. 두 축 크기가 우연히 같은 것일 수 있다.

- `d_model`

## 이 의뢰서를 처리하는 법

`review/prompt.md` 를 LLM 에 넘기고 이 모델을 지정한다. 판정 4종과 근거 요건, 결과를 어디에 어떤 형식으로 쓰는지는 전부 `review/` 안에 있다.
