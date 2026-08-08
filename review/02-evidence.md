# 02. 증거는 어디 있는가

파이프라인은 **판단에 필요한 재료를 다 모아놓고** 끝난다. 검토자가 따로 준비할 것은 없다.

## 이 모델의 실제 소스 (가장 중요)

```
develop/sources/modeling_<model_type>.py
develop/sources/configuration_<model_type>.py
```

`src/source_check.py` 가 매 재생성마다 HuggingFace transformers `main` 에서 받아 캐시한다.
**판정은 이 파일에서 나온다.** 여기 없으면(네트워크 불가, 또는 transformers 본체에 없는
아키텍처) `source_check.md` 가 "미확보"로 표시하고, 그건 **검사를 통과한 것이 아니라 수행되지
않은 것**이다 — 그때는 모델 저장소의 remote code나 논문으로 올라가야 한다.

`<model_type>` 은 `models/<모델>/review_request.md` 첫머리에 적혀 있다.

## 모델별 산출물

| 파일 | 무엇 |
|---|---|
| `models/<모델>/review_request.md` | **판단이 필요한 것만** 추린 의뢰서. 시작점 |
| `models/<모델>/source_check.md` | 기계적으로 확인된 것 — 별칭 접지 / 정사각 reshape / 모듈이 읽는 config 속성 |
| `models/<모델>/research_agenda.md` | 미해결 축 전체 + 판정(코드 조사 필요 / 등록만 / 정수로 두기) + 볼 소스 URL |
| `models/<모델>/full/review.md` | 리뷰 패킷 — prefill/decode 양쪽에서 shape별로 표본 추출한 실제 행 |
| `models/<모델>/structure.yaml` | 심볼 표(이 모델의 `d_model` 이 몇인지), 리터럴 차원, 라벨 출처 통계 |
| `models/<모델>/full/<phase>.csv` | 전체 operator 표. 라벨이 실제로 어떻게 렌더됐는지 |
| `models/<모델>/full/<phase>.trace.raw.jsonl` | 각 행의 원시 aten 근거 |

## 규칙 쪽

| 파일 | 무엇 |
|---|---|
| `rules/symbols.yaml` | 심볼 ↔ config 필드 별칭. 3번 안건(미등록 필드)이 가는 곳 |
| `rules/derived_dims.yaml` | 유도식(`n_h*d_head` 등). 4번 안건이 가는 곳 |
| `rules/structures/` | 모듈 구조 문서. C17이 등재를 확인한다 |
| `rules/references.yaml` | 소스 검증층 — 어떤 상수가 어느 문서에 근거하는지 |

## 자동 검사가 이미 확인한 것 (다시 묻지 말 것)

`source_check.md` 의 A/B/C 절은 결정적으로 확인된 사실이다:

- **A. 별칭 접지** — 심볼이 읽은 config 필드가 그 모델 config 클래스에 실제로 정의돼 있는가
- **B. 정사각 reshape** — `[..., X, X]` 의 이름이 읽은 config 필드가, 소스에서 정사각 reshape을
  만드는 변수와 같은 필드로 이어지는가 (`n_hc → hc → config.hc_mult` 같은 체인을 따라간다)
- **C. 모듈이 읽는 config 속성** — 각 모듈 클래스가 `__init__` 에서 읽는 config 속성 목록.
  그 모듈의 폭이 가질 수 있는 이름은 그것이 전부다

여기서 **확인됨**으로 나온 것은 다시 판정하지 않는다. **미확인**으로 나온 것만 의뢰서에 오른다.
