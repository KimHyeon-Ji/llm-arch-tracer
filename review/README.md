# review/ — LLM이 라벨을 검토하는 자리

`src/` 의 파이썬은 여기 들어오지 않는다. 파이프라인은 규칙으로 결정할 수 있는 것을 전부
결정하고, **판단이 필요한 것만 남긴 채 끝난다.** 그 다음이 이 폴더다.

파이썬 안에서 LLM을 부르려면 API 키가 필요하고 특정 업체에 묶인다. 그래서 호출을 밖으로 뺐다 —
검토는 아무 LLM으로나, 사람이 직접이라도 돌릴 수 있고, 결과는 언제나 같은 형식으로 남는다.

## 쓰는 법

아래 `python` 은 이 저장소의 의존성이 깔린 인터프리터를 뜻한다(`requirements.txt`).
가상환경을 쓰면 먼저 활성화한다 — Windows 는 `.venv\Scripts\activate`, macOS/Linux 는 `source .venv/bin/activate`.

```bash
python develop/regen_summaries.py    # 파이썬은 여기까지
python src/review_ledger.py          # 검토가 필요한 모델 목록
```

그 다음 **`prompt.md` 전문을 LLM에 붙여넣고 모델 이름 하나를 지정한다.** 끝이다.
LLM은 `models/<모델>/review_request.md` 를 읽고, `develop/sources/` 에 이미 받아둔 실제
modeling·configuration 소스를 열어 판정한 뒤, `models/<모델>/review_findings.md` 에 기록한다.

## 파일

| 파일 | 내용 |
|---|---|
| `prompt.md` | 복붙용 한 덩어리. **반박 프레임**(확인이 아니라 틀렸다는 증거를 찾아라)과 나머지 다섯 문서를 지시한다 |
| `01-procedure.md` | 무엇을 어떤 순서로 보고 어떻게 판정하는가 |
| `02-evidence.md` | 증거가 어디 있는가 — 소스 캐시, 안건, 리뷰 패킷 |
| `03-output.md` | 결과를 어떤 형식으로 남기는가 |
| `04-full-inventory.md` | **전수조사 방법론** — 미결 목록에 오르지도 않은 이름까지 확인하는 법 |
| `05-overrides.md` | **판정을 산출물에 반영하는 법** — 규칙으로 도달할 수 없는 축을 표에 넣는 경로 |
| `06-open-renames.md` | **미반영 교정 43건** — 이름은 확정됐는데 반영이 막힌 것들, 측정치와 실패 기록, 그리고 설계 자문용 질문지 |

## 왜 이 단계가 필요한가

규칙 게이트(`develop/verify_all.py`)는 산출물이 **자기 규칙과 일관되는지**만 판정한다.
규칙 자체가 틀렸거나, 두 config 값이 이 seq_len에서 우연히 같아 값으로 못 가리는 축은
잡지 못한다 — 그런 축에도 그럴듯한 이름이 붙고 모든 지표는 녹색으로 남는다.
그 자리를 메우는 것이 이 검토다.

검토 수행 여부와 만료는 `develop/verify/review_ledger.yaml` 에 남고 게이트가 매번
`최신 / 만료 / 미수행` 을 보고한다. **안 한 것과 깨끗한 것은 구별된다.**

그리고 **답했는지도 본다.** 원장은 세 가지를 강제한다:

| 검사 | 무엇을 막는가 |
|---|---|
| `unanswered_items` | 의뢰서 항목마다 그 항목의 라벨·모듈을 언급한 판정이 있는지 **항목 단위로** 맞춘다. 개수만 맞추던 이전 판은 엉뚱한 것에 답해도 통과했다 — Llama-4 의 `E*T` 가 2라운드 연속 그렇게 빠져나갔다 |
| `uncited` | `should_be_renamed` 판정의 근거에 소스 파일이나 URL이 없으면 FAIL. 결론이 맞아도 근거가 지어낸 것일 수 있다 |
| `claim_without_change` | 방법 서술(`angle`)이 바뀌었는데 판정 내용이 그대로면 FAIL. "다르게 봤다"는 검증 가능한 주장이다 |
| `soft_undetermined` | `undetermined` 인데 HF 소스 밖을 찾아본 흔적(URL)이 없으면 FAIL. 전수 재검토에서 13건 중 **10건이 오분류**였다 — "확인 못함"이 아니라 "알지만 못 넣음"이었다 |

셋 다 `develop/verify_selftest.py` 가 결함을 주입해 살아있는지 확인한다 — `claim_without_change` 는
배선 실수로 처음부터 죽어 있었고 그 주입이 잡아냈다.
