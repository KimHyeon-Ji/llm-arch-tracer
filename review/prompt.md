# 복붙용 프롬프트

아래 전문을 LLM에 붙여넣고, 맨 아래 `<모델>` 자리에 검토할 모델 폴더 이름을 적는다
(예: `deepseek-ai__DeepSeek-V4-Pro`). 저장소 파일을 읽을 수 있는 환경이어야 한다.

---

당신은 llm-arch-tracer 가 뽑은 **operator별 tensor shape 라벨**을 검토한다.

이 도구는 HuggingFace 모델을 meta device에서 트레이스해 연산마다 입력·가중치·출력 shape을
기록하고, 각 축에 `d_model`, `n_h`, `d_head` 같은 이름을 붙인다. 이름은 등록된 규칙이 붙이고,
규칙이 자기 일관성을 검사하는 게이트도 이미 통과했다. **당신이 볼 것은 규칙이 구조적으로 볼 수
없는 것이다** — 규칙 자체가 틀렸거나, 두 config 값이 이 seq_len에서 우연히 같아 값으로는
어느 이름인지 가릴 수 없는 축.

## 먼저 읽을 것

1. `review/01-procedure.md` — 판정 절차와 판정 4종
2. `review/02-evidence.md` — 증거가 어디 있는지
3. `review/03-output.md` — 결과를 어떤 형식으로 남기는지
4. `review/04-full-inventory.md` — **전수조사 방법론**: 무엇이 모집단이고, 어느 층까지 기계가
   결론을 냈고, 당신이 볼 것은 그중 무엇이며, 안 본 것을 어떻게 기록하는지
5. `review/05-overrides.md` — 판정을 **산출물에 반영**하는 법(근거 인용 필수)
6. `models/<모델>/review_request.md` — **이 모델에서 판단이 필요한 것 + 전수 점검 목록**

## 반드시 지킬 것

- **소스를 연다.** `develop/sources/modeling_*.py` 에 이 모델의 실제 코드가 이미 받아져 있다.
  문제의 shape을 만드는 **코드 줄**을 찾아라. 구조를 일반론으로 설명하지 말고 그 줄을 인용해라.
- **HF 소스는 출발점이지 울타리가 아니다. 부족하면 필요한 소스를 직접 찾아라.**
  캐시된 파일에 답이 없는 경우는 흔하다 — 연산이 Triton·CUDA 커스텀 커널이라 HF 에는 torch
  fallback 만 있거나(KDA, mamba-ssm, causal-conv1d, flash-attn) 아키텍처가 아직 transformers
  본체에 없어 remote code 로 도는 경우다. **우리 트레이스는 그 fallback 을 돈 것이므로 실제
  커널과 다를 수 있다.** 그럴 땐 알아서 찾아본다: 모델 저장소의 remote code, vLLM / SGLang /
  TensorRT-LLM 독립 구현, 커널 저장소, 논문, 공식 블로그·model card, GitHub 이슈·PR — 무엇이든
  답을 주는 것이면 된다. 아래 순서는 보통 이 순서가 빠르다는 **권고**지 제한이 아니다:

  > 실행 중인 modeling 소스 → config 클래스 docstring → 모델 저장소 remote code →
  > vLLM/SGLang/TRT-LLM → 커널 저장소 → 논문 → model card·블로그 → 검색

  무엇을 봤는지는 근거에 URL 로 남긴다. 어디를 봐도 답이 없으면 **본 곳을 적고**
  `undetermined` 로 남긴다 — 짐작으로 메우지 않는다.
- **우리 자체 산출물을 검증으로 믿지 말 것.** `model_summary.md` 의 체크리스트나
  `review_findings.md` 는 참고다. 실제로 `full/*.csv` 와 `*.jsonl` 을 세어 확인해라 —
  의뢰서가 낸 질문 수와 답한 수가 맞는지, `full/ambiguous.json` 의 `chosen` 이 채워졌는지까지.
  "검토했다"고 적혀 있지만 안 한 경우가 실제로 여러 건 있었다(2026-08-12 외부 검토).
- **접힌 표(`prefill.jsonl` / `decode.jsonl`)는 샘플링 없이 한 줄씩 훑어라.** 모델당 수백 줄이라
  전부 읽을 수 있다. 한 행 안에서 `input_shape` / `weight_shape` / `output_shape` 가 서로
  말이 되는지는 A/B/C 절의 접힌 뷰에서 보이지 않는다 — 그렇게 4,406건이 숨어 있었다.
- **소스가 transformers 본체에 없으면 모델 저장소를 열어라.** `develop/sources/` 에
  `<model_id>__<파일명>.py` 로 이미 받아뒀을 수 있다. 파일 이름은 model_type 과 다를 수 있고
  (Kimi-K2.6 → `modeling_deepseek.py`), 판정에는 실제로 읽은 파일을 인용한다.
- **근거 없는 판정은 판정이 아니다.** 클래스·메서드 이름과 인용한 코드 줄이 있어야 한다.
- **모르면 `undetermined`.** 무엇을 봤고 뭐가 없었는지 적는다. 지어내지 않는다.
- **정수를 억지로 이름 붙이지 않는다.** 루프 인덱스·피연산자 개수는 이름이 없는 게 정답이다.
- **한 번에 한 모델.**

## 결과

`models/<모델>/review_findings.md` 에 `03-output.md` 형식으로 쓰고, 원장에 기록한다.
일반화되는 발견은 `rules/` 에 승격할 것을 제안하되, **직접 고친 뒤에는 반드시**
`develop/verify_all.py` 가 EXIT 0 인지 확인한다.

---

검토할 모델: `<모델>`
