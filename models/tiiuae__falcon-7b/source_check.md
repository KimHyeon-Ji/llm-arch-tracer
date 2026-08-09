# 소스 대조 결과 (자동)

모델의 실제 `modeling_*.py` / `configuration_*.py` 를 받아 라벨을 대조한 결과다. LLM 없이 매 재생성마다 돌며, 받은 소스는 `develop/sources/` 에 남는다.

- transformers 모듈: `falcon`
- configuration 소스: 확보
- modeling 소스: 확보

## A. 심볼이 읽은 config 필드가 실제로 존재하는가

이 모델이 쓴 심볼의 config 필드가 전부 `configuration_*.py` 에 존재한다.

## B. 정사각 축이 소스의 정사각 reshape 과 맞는가

정사각으로 렌더된 축이 없다.

## C. 모듈이 읽는 config 속성

`__init__` 에서 config 를 읽는 클래스 10개를 소스에서 확인했다. 이 목록이 그 모듈의 폭이 가질 수 있는 이름의 전부다 — `src/anchors.py` 가 빌드된 모델에서 읽어오는 값과 같은 출처이며, 서로 어긋나면 그것이 발견이다.
