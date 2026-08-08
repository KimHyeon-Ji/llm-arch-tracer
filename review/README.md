# review/ — LLM이 라벨을 검토하는 자리

`src/` 의 파이썬은 여기 들어오지 않는다. 파이프라인은 규칙으로 결정할 수 있는 것을 전부
결정하고, **판단이 필요한 것만 남긴 채 끝난다.** 그 다음이 이 폴더다.

파이썬 안에서 LLM을 부르려면 API 키가 필요하고 특정 업체에 묶인다. 그래서 호출을 밖으로 뺐다 —
검토는 아무 LLM으로나, 사람이 직접이라도 돌릴 수 있고, 결과는 언제나 같은 형식으로 남는다.

## 쓰는 법

```bash
.venv/Scripts/python.exe develop/regen_summaries.py    # 파이썬은 여기까지
.venv/Scripts/python.exe src/review_ledger.py          # 검토가 필요한 모델 목록
```

그 다음 **`prompt.md` 전문을 LLM에 붙여넣고 모델 이름 하나를 지정한다.** 끝이다.
LLM은 `models/<모델>/review_request.md` 를 읽고, `develop/sources/` 에 이미 받아둔 실제
modeling·configuration 소스를 열어 판정한 뒤, `models/<모델>/review_findings.md` 에 기록한다.

## 파일

| 파일 | 내용 |
|---|---|
| `prompt.md` | 복붙용 한 덩어리. 나머지 세 문서를 읽으라고 지시한다 |
| `01-procedure.md` | 무엇을 어떤 순서로 보고 어떻게 판정하는가 |
| `02-evidence.md` | 증거가 어디 있는가 — 소스 캐시, 안건, 리뷰 패킷 |
| `03-output.md` | 결과를 어떤 형식으로 남기는가 |

## 왜 이 단계가 필요한가

규칙 게이트(`develop/verify_all.py`)는 산출물이 **자기 규칙과 일관되는지**만 판정한다.
규칙 자체가 틀렸거나, 두 config 값이 이 seq_len에서 우연히 같아 값으로 못 가리는 축은
잡지 못한다 — 그런 축에도 그럴듯한 이름이 붙고 모든 지표는 녹색으로 남는다.
그 자리를 메우는 것이 이 검토다.

검토 수행 여부와 만료는 `develop/verify/review_ledger.yaml` 에 남고 게이트가 매번
`최신 / 만료 / 미수행` 을 보고한다. **안 한 것과 깨끗한 것은 구별된다.**
