# 02. 증거는 어디 있는가

파이프라인은 **판단에 필요한 재료를 다 모아놓고** 끝난다. 검토자가 따로 준비할 것은 없다.

## 소스를 구하는 순서 — transformers 가 전부가 아니다

1. **`develop/sources/modeling_<model_type>.py`** — transformers `main`. 대부분 여기서 끝난다.
2. **모델 저장소의 remote code** — 본체에 없으면 여기가 진짜 실행 코드다.
   `source_check.fetch_from_repo()` 가 **HF API 로 레포 파일 목록을 먼저 훑고 필요한 것만**
   받아 `develop/sources/<model_id>__<파일명>.py` 로 캐시한다.
   파일 이름은 model_type 과 다를 수 있다 — Kimi-K2.6 의 `model_type` 은 `kimi_k2` 인데
   실제 파일은 `modeling_deepseek.py` 다(갈라져 나온 아키텍처 이름을 따른다).
   **판정에는 실제로 읽은 파일 이름을 인용한다.**
3. 그 아래는 아래 「이 캐시로 부족하면」 절의 사다리.

> 이 2번이 없던 동안 Kimi 2종은 소스 대조가 통째로 "수행되지 않음"이었다. 열자마자 소속 검사가
> MLA 의 `q_b_proj` 에 fused-QKV 폭이 붙어 있는 것을 잡았다(2026-08-12). **"소스가 없다"는
> 대부분 "본체에 없다"는 뜻이지 "구할 수 없다"는 뜻이 아니다.**

## config 만으로는 확정되지 않는 것

아래는 값이 아니라 **구조**라서 반드시 소스를 봐야 한다. config 에 필드가 있다는 것만으로
단정하면 안 된다.

- 공유 전문가가 **실재하는가** — 필드가 있어도 코드가 안 만드는 경우가 있다
  (MiniMax-M2 의 `shared_intermediate_size: 0`, 클래스 미선언·코드 미사용)
- KV 가 **단일 텐서인가** 따로인가 — KV 캐시 폭 계산이 여기서 갈린다
- 융합 파라미터의 **축 순서** — `nn.Parameter` 는 아무것도 선언하지 않는다
- 어떤 폭이 **자기 필드**인가 다른 필드의 배수인가 — Nemotron 의 공유 전문가 폭은
  `moe_shared_expert_intermediate_size` 이고, 우연히 `2 × moe_intermediate_size` 다

## 이 모델의 실제 소스 (가장 중요)

```
develop/sources/modeling_<model_type>.py
develop/sources/configuration_<model_type>.py
```

`src/source_check.py` 가 매 재생성마다 HuggingFace transformers `main` 에서 받아 캐시한다.
**대부분의 판정이 이 파일에서 나온다.** 여기 없으면(네트워크 불가, 또는 transformers 본체에
없는 아키텍처) 의뢰서가 "미확보"로 표시하고, 그건 **검사를 통과한 것이 아니라
수행되지 않은 것**이다.

**이 캐시로 부족하면 거기서 멈추지 말고 필요한 소스를 찾아본다.** 자주 부족해지는 경우:

- **커스텀 커널** — Triton/CUDA 로 도는 연산(KDA, mamba-ssm, causal-conv1d, flash-attn)은
  HF 파일에 torch fallback 만 있다. 우리 트레이스도 그 fallback 을 돈 것이라 **실제 커널과
  shape 이 다를 수 있다.** 커널 저장소나 독립 구현을 봐야 한다.
- **본체에 없는 아키텍처** — 모델 저장소의 remote code(`modeling_*.py`)가 진짜 실행 코드다.
- **코드에 안 적힌 의도** — 왜 그 폭인지는 논문·공식 블로그·model card 에만 있는 경우가 있다.

찾아볼 곳: 모델 저장소 remote code, vLLM / SGLang / TensorRT-LLM 독립 구현, 커널 저장소,
논문, 공식 블로그·model card, GitHub 이슈·PR. **무엇을 봤는지 URL 로 남기면 된다.**

`<model_type>` 은 `models/<모델>/review_request.md` 첫머리에 적혀 있다.

## 모델별 산출물

| 파일 | 무엇 |
|---|---|
| `models/<모델>/review_request.md` | **시작점.** 판단이 필요한 것 + 기계적으로 이미 확인된 것 + 소스 위치가 한 파일에 |
| `models/<모델>/review_findings.json` | 지난 검토의 판정. 같은 질문을 다시 하지 않기 위해 먼저 본다 |
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

의뢰서의 **「기계적으로 이미 확인된 것」** 절은 결정적으로 확인된 사실이다:

- **A. 별칭 접지** — 심볼이 읽은 config 필드가 그 모델 config 클래스에 실제로 정의돼 있는가
- **B. 정사각 reshape** — `[..., X, X]` 의 이름이 읽은 config 필드가, 소스에서 정사각 reshape을
  만드는 변수와 같은 필드로 이어지는가 (`n_hc → hc → config.hc_mult` 같은 체인을 따라간다)
- **C. 모듈이 읽는 config 속성** — 각 모듈 클래스가 `__init__` 에서 읽는 config 속성 목록.
  그 모듈의 폭이 가질 수 있는 이름은 그것이 전부다

여기서 **확인됨**으로 나온 것은 다시 판정하지 않는다. **미확인**으로 나온 것만 의뢰서에 오른다.
