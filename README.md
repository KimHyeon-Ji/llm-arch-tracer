# llm-arch-tracer

HuggingFace LLM을 **가중치 없이** 실행해, 레이어별 **연산(op) · 텐서 shape · 의존관계**를
ATen 레벨에서 뽑아내는 도구다.

가중치를 안 받으므로 GPU도, 디스크 수백 GB도 필요 없다. **1.5조 파라미터 모델도 노트북에서
몇 분이면 구조가 나온다.** shape은 실행값이 아니라 심볼(`B, T, d_model, n_h*d_head …`)로
렌더되므로, 결과는 "이 실행"이 아니라 **아키텍처 자체**를 설명한다.

현재 검증 통과한 모델 **26개**가 `models/` 에 들어 있다 — Llama·Qwen·DeepSeek·gpt-oss·
Gemma·Mamba/SSM 하이브리드·xLSTM 등.

---

## 1. 어떻게 가중치 없이 돌리는가 — meta device

PyTorch의 `meta` device는 **shape과 dtype만 있고 데이터가 없는 텐서**를 만든다.
연산은 정상적으로 "실행"되지만 실제 숫자 계산은 일어나지 않고, 결과 shape만 전파된다.

```python
import torch

a = torch.randn(4096, 4096, device="meta")   # 16M개 값, 메모리 0바이트
b = torch.randn(1, 17, 4096, device="meta")
c = b @ a.T                                   # 곱셈은 안 일어난다
print(c.shape, c.device)                      # torch.Size([1, 17, 4096]) meta
```

모델 전체에 이걸 적용하면 **가중치를 한 바이트도 안 받고 forward를 끝까지 돌릴 수 있다**:

```python
import torch
from transformers import AutoConfig, AutoModelForCausalLM

cfg = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B")   # config.json만 (수 KB)

with torch.device("meta"):                # 이 블록 안의 파라미터는 전부 meta
    model = AutoModelForCausalLM.from_config(cfg)

ids = torch.zeros(1, 17, dtype=torch.long, device="meta")
out = model(ids)                          # 진짜 forward — 값만 없다
print(out.logits.shape)                   # torch.Size([1, 17, 151936])
```

`.safetensors` 를 안 받으므로 **디스크 0, VRAM 0**이다. 받는 건 `config.json` 하나뿐이다.

### 그 forward를 어떻게 가로채는가 — TorchDispatchMode

여기까지는 최종 출력 shape만 안다. 중간 연산을 전부 보려면 PyTorch의 **dispatch 레벨**에
끼어들면 된다. `TorchDispatchMode` 는 모든 ATen 연산이 실제로 실행되기 직전에 호출된다:

```python
from torch.utils._python_dispatch import TorchDispatchMode

class Tracer(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        print(func,                                    # aten.mm.default
              [a.shape for a in args if hasattr(a, "shape")],   # 입력 shape
              getattr(out, "shape", None))             # 출력 shape
        return out

with torch.device("meta"), Tracer():
    model(ids)
```

이 도구가 하는 일이 정확히 이것이다. 여기에 세 가지를 더한다:

| 더하는 것 | 무엇을 위해 |
|---|---|
| **module hook** (`src/scope.py`) | 각 op이 어느 모듈에서 났는지 (`model.layers.7.self_attn.q_proj`) |
| **tensor identity** (`src/tracer.py`) | 어느 파라미터를 썼는지, 어느 op의 출력이 어느 op의 입력인지 |
| **심볼 렌더링** (`src/symbolic_shape.py`) | `896` → `d_model`, `[1,14,17,64]` → `[B, n_h, T, d_head]` |

> **fx graph가 아니다.** `torch.fx`는 코드를 정적으로 추적해 제어 흐름·동적 shape에서
> 깨진다. 여기서는 **실제로 실행된 것**만 기록하므로, MoE 라우팅이든 Mamba 청크 스캔이든
> 실행된 그대로 남는다. (`torch.export`는 교차검증 C12에서만 쓴다.)

### 무엇을 얻는가

모델 하나당 이런 결과가 나온다:

```
models/Qwen__Qwen2.5-0.5B/
  prefill.csv / prefill.jsonl      주요 operator 표 (latency 관점, 반복 레이어 접힘)
  decode.csv  / decode.jsonl       decode 단계 (KV 캐시가 붙은 형태)
  structure.yaml                   심볼 표 · 레이어 스케줄 · 라벨 출처 통계
  model_summary.md                 아키텍처 요약 카드 + 검증 로그 + 라벨 검토 결과
  review_request.md                규칙이 못 정한 축 (있으면)
  review_findings.json / .md       소스와 대조한 판정
  full/
    prefill.csv / decode.csv       전체 트레이스 (모든 ATen 프리미티브)
    *.trace.raw.jsonl              각 행의 원시 근거
    *.shapes.concrete.jsonl        구체 정수 shape (재렌더용)
    provenance.json                commit hash · 버전 · 적용된 조치 이력
    report.md                      C1~C17 검증 체크리스트
    review.md                      리뷰 패킷 (shape별 표본)
```

표는 이렇게 생겼다 (Llama-3.1-8B의 FFN up-projection, 실제 출력):

| op_id | h1 | h2 | op_type | input_shape | weight_shape | weight_pos | output_shape | depends_on |
|---|---|---|---|---|---|---|---|---|
| 122 | mlp | up_proj | matmul | `[[T, d_model], [d_model, d_ff]]` | `[d_ff, d_model]` | 1 | `[[T, d_ff]]` | `[120, 121]` |

숫자가 하나도 없다 — `d_model`이 4096이라는 사실은 `structure.yaml`의 심볼 표에 있고,
표 자체는 **모든 Llama 계열에 그대로 적용되는 구조**를 말한다.

이걸로 답할 수 있는 것: **어떤 연산이 몇 번 도는가 · 각 텐서가 얼마나 큰가 ·
무엇이 무엇에 의존하는가 · prefill과 decode가 어떻게 다른가.**

> **`input_shape`는 activation만이 아니다.** Linear는 `y = x @ W.T`라 피연산자가
> `[activation, W.T]` 둘이고, 같은 weight가 `input_shape` 안에(전치된 모습) 또
> `weight_shape`에(저장 형태) 나온다. 어느 것이 weight인지는 **`weight_pos` 컬럼**이
> 가리킨다 — 자세히는 `01-main.md §6.2`.

---

## 2. 쓰는 법

```bash
pip install -r requirements.txt
```

### 새 모델 뽑기

프로파일 하나만 쓰면 된다. 사람이 채울 값은 **`model_id` 하나뿐**이다:

```yaml
# develop/models/qwen2.5-0.5b.yaml
model_id: Qwen/Qwen2.5-0.5B
revision: null              # commit hash로 고정하려면 지정, null이면 최신
phases: [prefill, decode]
seq_len: auto               # top-k·window·압축 파라미터를 보고 안전한 최소값 자동 계산
overrides:
  attn_implementation: auto # 필요시 "sdpa" | "eager"
extra_entrypoints: auto     # MTP 등 메인 forward 밖 모듈 자동 탐색
```

```bash
python src/run.py --profile develop/models/qwen2.5-0.5b.yaml --out develop/out/
```

`develop/out/<모델>/full/report.md` 가 **FAIL 0**이면 완성품이다:

```bash
python develop/promote.py <id>     # develop/out/ -> models/ 로 승격
```

게이트된 저장소(Meta Llama 등)는 먼저 `huggingface-cli login` + 라이선스 동의가 필요하다.
Triton 커널처럼 meta에서 못 도는 옵션은 프로파일의 `config_overrides:` 로 native 값을 강제한다
(아키텍처를 바꾸지 않는 선에서 — 실제 예시는 `develop/models/*.yaml` 참고).

### 규칙을 고쳤을 때

```bash
python develop/regen_summaries.py   # 재추적 없이 전 모델 산출물 갱신
python develop/verify_all.py        # 단일 게이트 — EXIT 0 이어야 한다
python develop/verify_selftest.py   # 게이트 자체가 살아있는지 (현재 20/20)
```

---

## 3. shape에 이름을 붙이는 3단계

관측(op·shape)은 실행 기록이라 거의 틀리지 않는다. **약점은 "그 축이 무엇인가"** 다.
`[1, 14, 17, 64]` 의 `64`가 `d_head`인지 `d_model/n_h`인지 우연히 같은 다른 값인지는
숫자만 봐서는 모른다. 그래서 3단계로 확정한다.

### ① 코드가 결정한다 (자동)

1. **모듈이 선언한 폭** — `nn.Linear`의 weight는 `[out_features, in_features]`다.
   트레이스는 각 op이 어느 파라미터를 썼는지 텐서 신원으로 이미 알고 있으므로,
   폭은 **추론할 게 아니라 읽으면 된다** (`src/anchors.py`).
2. **등록된 규칙** — `rules/symbols.yaml`(심볼 ↔ config 필드), `rules/derived_dims.yaml`
   (유도식: `n_h*d_head`, `2*d_moe`, `T+T/m_csa` …). 전부 **출처 주석**이 달려 있다.
3. **값 매칭** — 위 둘이 침묵할 때만 쓰는 폴백.

**전체 축의 96.4%가 등록 규칙이 낸 이름이고, 산술적으로 거짓인 라벨은 0이다.**

### ② 게이트가 검사한다 (자동)

`develop/verify_all.py` 한 곳에서 전부 본다 — 규칙 일관성, 제1원리 불변식
(가중치 축은 T에 의존할 수 없다 / `n_h`와 `n_kv`는 한 shape에 공존할 수 없다 /
데이터플로우 일관성 / 심볼 대입값 = 실제 정수), 공개 수치 대조, 퇴행 검사.

그리고 **게이트 자체도 검사한다**(`verify_selftest.py`): 각 검사에 그 검사가 잡아야 할
결함을 주입해 실제로 FAIL이 나는지 확인한다. "FAIL 0"이 결함이 없어서인지 검사가 죽어서인지
구분하기 위해서다 — 실제로 attention 레이어 수 검사가 죽어 falcon이 조용히 attention-free로
뒤집힌 적이 있다.

### ③ LLM이 소스와 대조한다 (판단이 필요한 것만)

게이트는 **산출물이 자기 규칙과 일관되는지**만 판정한다. 규칙 자체가 틀렸거나, 두 config
값이 이 seq_len에서 우연히 같아 값으로 못 가리는 축은 못 잡는다 — 그런 축에도 그럴듯한
이름이 붙고 모든 지표는 녹색으로 남는다. 그 자리를 메우는 단계다.

**파이썬은 판단 직전까지만 간다** (매 실행 자동):

- 그 아키텍처의 **실제 `modeling_*.py` / `configuration_*.py` 를 받아** `develop/sources/` 에 캐시
- 기계적으로 결정 가능한 것을 대조 (심볼이 읽은 config 필드의 실재, 정사각 축이 소스의
  정사각 생성과 이어지는지, 각 모듈이 읽는 config 속성)
- **못 정한 것만** 추린 의뢰서 → `models/<모델>/review_request.md`

그 다음이 사람 손이 가는 유일한 지점이다. **`review/prompt.md` 를 LLM에 붙여넣고 모델
이름 하나를 지정하면 된다.** 파이썬 안에서 LLM을 부르지 않는 이유는 API 키가 필요하고
특정 업체에 묶이기 때문이다 — 이렇게 빼두면 아무 LLM으로나, 사람이 직접이라도 돌릴 수 있고
결과 형식은 같다.

#### HF 코드만으로 안 되는 경우 — 커스텀 커널 · Triton

**이게 실제로 자주 일어난다.** 최신 아키텍처의 핵심 연산은 Triton/CUDA 커널로 구현되고,
transformers에는 느린 **torch fallback만** 들어 있는 경우가 많다:

| 예 | 상황 |
|---|---|
| Mamba2 / SSM | `mamba-ssm`, `causal-conv1d` 커널이 본체. HF엔 `torch_forward` fallback |
| Qwen3-Next linear attention | `fla` (flash-linear-attention) Triton 커널 |
| Kimi-K3 KDA | 93개 레이어 중 69개가 Triton 전용 — meta에서 아예 못 돈다 |
| gpt-oss attention sink | 커널에 융합돼 있고 config 필드가 없다 |

**우리 트레이스는 fallback 경로를 돈 것이므로 실제 커널과 다를 수 있다.** 그래서 검토
프롬프트는 HF 소스를 **출발점이지 울타리가 아니라고** 명시하고, 답이 없으면 알아서 찾으라고
지시한다:

> 실행 중인 modeling 소스 → config 클래스 docstring → 모델 저장소 remote code →
> vLLM / SGLang / TensorRT-LLM 독립 구현 → 커널 저장소 → 논문 → model card·블로그 → 검색

무엇을 봤는지는 근거에 URL로 남긴다. 어디를 봐도 없으면 **본 곳을 적고 `undetermined`** 로
남긴다 — 짐작으로 메우지 않는다.

#### 결과가 어떻게 반영되는가

LLM은 판정 4종(`맞음` / `교정 필요` / `이름 없음이 정답` / `미확정`) 중 하나를 근거 코드
줄과 함께 `review_findings.json` 에 쓴다. 거기서 세 갈래로 흘러간다:

```
review_findings.json
   ├─ review_findings.md                    읽는 기록 (자동 생성)
   ├─ model_summary.md "③ 라벨 검토" 절      표 보는 사람이 바로 만나는 주의사항
   └─ rules/ 승격                            일반화되면 다음 실행부터 자동
```

**규칙으로 못 박는 것도 버리지 않는다.** `status: "open"` 으로 두면 요약 카드에
**"지금 렌더 / 소스가 말하는 것 / 근거"** 가 나란히 실린다. 예를 들어 OLMoE는
`gate_up_proj` 가 `[E, 2*intermediate, hidden]` 인데 `2*1024 == 2048 == hidden` 이라
값으로는 구별이 불가능하다 — 그 사실이 소스 줄과 함께 요약에 남는다.

검토 수행 여부와 만료는 `develop/verify/review_ledger.yaml` 에 기록되고, 게이트가 매번
`최신 / 만료 / 미수행` 을 보고한다. **안 한 것과 깨끗한 것은 구별된다.**

```bash
python src/review_ledger.py       # 검토가 필요한 모델
```

---

## 4. 폴더 구조

```
README.md                      # 이 문서 — 개념 · 사용법 · 검증
01-main.md                     # 파이프라인 전체 스펙 (표 스키마, 검증 C1~C17, 심볼 정의)
02-new-module-handling.md      # 처음 보는 아키텍처를 만났을 때 (Tier 0~3)
review/                        # ③ 라벨 검토 — LLM에 넘기는 프롬프트와 절차
src/                           # 참조 구현 (run.py가 진입점)
rules/                         # 심볼 · 유도식 · 구조 라이브러리 (계속 누적)
models/                        # 검증 통과한 완성 산출물 — 여기 있는 건 다 믿을 수 있다
develop/                       # 작업 공간: 프로파일, 게이트, 회귀 테스트, 소스 캐시
```

`src/` 주요 파일:

| 파일 | 역할 |
|---|---|
| `run.py` | 진입점. 아래 전부를 엮는다 |
| `provenance.py` | revision을 commit hash로 고정, config 스냅샷 |
| `loader.py` | meta device로 가중치 없이 구조만 로드 |
| `introspect.py` | config에서 안전한 seq_len·레이어 스케줄 자동 도출 |
| `tracer.py` | **`TorchDispatchMode`로 op·shape·의존관계 캡처 (핵심)** |
| `scope.py` | module hook 기반 레이어/블록 라벨링 |
| `anchors.py` | 모듈이 선언한 차원을 라벨의 1순위 근거로 |
| `symbolic_shape.py` | 정수 → 심볼 렌더링 |
| `build_table.py` | 트레이스 → 전체 표 + 주요 operator 표 |
| `validate.py` | C1~C17 체크리스트 |
| `summarize.py` | `structure.yaml` + `model_summary.md` |
| `source_check.py` | 실제 modeling/config 소스를 받아 기계적으로 대조 |
| `review_request.py` | 규칙이 못 정한 것만 추려 검토 의뢰서 작성 |
| `review_notes.py` | 검토 판정을 요약 카드로 되돌려 반영 |
| `review_ledger.py` | 검토 수행 여부·만료 기록 |

---

## 5. 지금 상태

| | |
|---|---|
| 검증 통과 모델 | 26개 |
| 전체 축 | 5,297,339 |
| 등록 규칙이 낸 이름 | **96.4%** |
| 산술적으로 거짓인 라벨 | **0** |
| 지어낸 이름(휴리스틱) | 0.26% |
| 게이트 | FAIL 0 · 셀프테스트 20/20 |
| ③ 라벨 검토 | 26/26 수행 |

**범위 밖:** FLOPs · 메모리 대역폭 · latency 추정은 하지 않는다. 이 도구는 **연산과 shape을
정확하게** 내는 데 집중하고, 그 위의 계산은 결과를 받아서 하면 된다.
