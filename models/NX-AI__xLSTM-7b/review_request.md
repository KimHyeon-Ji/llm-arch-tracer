# 검토 의뢰서 — NX-AI/xLSTM-7b

파이썬 파이프라인이 규칙으로 결정할 수 있는 것을 전부 결정하고, **판단이 필요한 것만** 여기 남겼다. 절차와 출력 형식은 `review/` 에 있다.

- transformers 모듈: `xlstm`
- 판단 필요: **1건**

## 증거 — 이미 받아둔 실제 소스

- `develop/sources/modeling_xlstm.py` — 있음, 이 파일을 열어서 판정한다
- `develop/sources/configuration_xlstm.py` — 있음, 이 파일을 열어서 판정한다

그 밖의 재료: `models/<모델>/research_agenda.md`(미해결 축 + 볼 소스 URL), `full/review.md`(리뷰 패킷), `structure.yaml`(심볼 표), `source_check.md`(기계적으로 확인된 것).

## 판단이 필요한 것

### 1. 이 config 필드가 정말 이 뜻인가

값은 로드된 config 에 있지만 이 모델의 config 클래스가 선언한 필드가 아니다 (체크포인트 `config.json` 에서 온 값). 클래스가 뜻을 보증하지 않으므로 modeling 소스에서 이 필드가 실제로 어떻게 쓰이는지 확인해야 한다.

- `d_head ← head_dim`

## 이 의뢰서를 처리하는 법

`review/prompt.md` 를 LLM 에 넘기고 이 모델을 지정한다. 판정 4종과 근거 요건, 결과를 어디에 어떤 형식으로 쓰는지는 전부 `review/` 안에 있다.
