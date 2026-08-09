# 라벨링을 "추론"에서 "확인"으로 — 검토 결과와 로드맵

작성 2026-08-04. **아직 아무것도 구현하지 않은 검토 문서다.** 실행 여부는 판단 후 결정한다.

관련: 표 스키마·앵커는 `01-main.md §6.2/§6.3`, 신규 모듈 절차는 `02-new-module-handling.md`.

---

## 0. 문제 한 줄 요약

`symbolic_shape.dim(n, module_path)`는 **정수 하나**를 받아 config 값들 중 같은 값을 찾고,
여러 개가 걸리면 `priority:`로 고른다. 값이 겹치는 순간 **한쪽 문맥에서는 반드시 틀린다.**

| 모델 | 충돌 | 결과 |
|---|---|---|
| Llama-3.1-405B | `d_model == n_h*d_head == 16384` | 잔차 스트림이 `n_h*d_head`로 |
| Llama-4 | `E == d_head == 128` | attention head 차원이 `E`로 |
| gpt-oss | `d_model == d_ff == 2880` | 잔차 스트림이 `d_ff`로 |
| Zamba2 | `q_proj`가 정사각 4096×4096 | weight 축 순서가 뒤집힘 |

지금까지 발견된 라벨 오류는 **전부 이 설계 하나**에서 나왔다.

---

## 1. 이미 반영된 것 (2026-08-04, 게이트 통과)

### 1-1. `weight_pos` 컬럼
`input_shape`는 그 op이 받은 **모든** 텐서다. Linear에서는 weight가 `input_shape`에도
(전치된 모습) `weight_shape`에도(저장 형태) 들어가 같은 텐서가 두 번 나온다. `weight_pos`가
어느 피연산자가 weight인지 가리킨다. 26개 모델 × 8파일 전부 적용, 사이드카에도 보존.

### 1-2. 모듈 선언 차원 앵커 (`src/anchors.py`)
`nn.Linear`의 weight는 `[out_features, in_features]`이고, 트레이스는 각 op이 어떤 파라미터를
썼는지 **텐서 신원으로 이미 확정**한다. 그러니 폭은 추론할 게 아니라 읽으면 된다.

- 어느 축이 `in`인지는 **곱셈 자체에서** 판정 (활성 입력의 마지막 축과 contract되는 축)
- 모듈당 라벨 1회 확정 → 전 레이어·전 op에 동일 적용
- 모듈의 `in`은 생산자의 `out`과 같은 텐서이므로 데이터플로우로 교정
- 앵커가 정한 이름을 그 텐서를 보는 모든 op에 전파

**측정 결과**: 306만 축 중 3,422축 교정, 회귀 0, `flow_ambig`("한 텐서 두 이름") 함대 전체
2,929 → 1,547 (**-47%**).

> 안전장치 4개는 전부 회귀를 실제로 관측하고 나서 추가했다. 파라미터 소비 op으로 범위 한정
> (DeepSeek-V4 compressor의 `new_zeros`가 옳은 라벨 1,440개를 덮음), norm 앵커 불변
> (OLMo-2에서 in/out이 갈라짐), 전파 필수(Zamba2 0→36으로 **악화**), split 규칙 비활성화.

### 1-3. ③ 자유 평가 자동화
`make_review_packet.write_packet()`을 `run.py`·`regen_summaries.py` 끝에서 자동 호출.
26개 모델 전부 `full/review.md` 보유, 산출물과 항상 동기화. (읽는 것은 여전히 사람/LLM 몫.)

---

## 2. 남은 문제 — 이름은 아직 값 매칭이다

앵커는 **"어느 축에 이름이 붙는지"** 를 확정한다. **이름 자체는** 여전히 값 매칭이다.
`5120`을 `d_model`이라 부르는 근거는 아직 "config에 5120인 필드가 있다"이다.

---

## 3. 검토한 해결책 세 가지 (측정 포함)

### 3-A. 소스 파싱 (`ast`) — 기각

`self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim)`을
파싱한다. 사람이 하는 일과 같지만:
- 식이 한 줄에 없다 (`self.head_dim = config.head_dim or hidden // heads`) → 변수 추적 필요
- 소스의 `self.q_proj` → 실행 시 경로 `model.layers.0.self_attn.q_proj` 매핑이 없다
- 모델마다 작성 방식이 다르고 remote code는 더하다

### 3-B. config 섭동 — 열등하지만 교차검증용으로 보존

config 필드 하나를 흔들고 meta에서 재빌드해 어느 차원이 얼마나 움직이는지 본다.
차원은 config 필드의 다중선형 함수라 **편미분이 곧 계수**다.

```
head_dim 128→129 → q_proj.out이 +32 움직임 → 32 = n_h  ⟹ out = n_h × d_head
hidden_size +1   → q_proj.in 이 +1  움직임 → 계수 1     ⟹ in  = d_model
```

검증됨: Llama-3.1-405B(`d_model==n_h*d_head`)와 Llama-4(`E==d_head`) 둘 다 정확히 구분.
프로토타입 `develop/probe_config_provenance.py`.

**단점**: 필드당 재빌드 1회(모델당 7~10회), 복잡한 식(`n_h*(d_nope+d_rope)`)은 못 푼다.

### 3-C. ★ 이름표 전파 — 채택 권장

**config 값에 이름표를 달고 모델을 짓는다.** `config.hidden_size`가 "나는 d_model"이라고
기억하는 `int` 하위 클래스를 돌려주고, 곱셈·덧셈이 그 이름을 이어받게 한다. `nn.Linear`는 받은
값을 그대로 저장하므로 **다 짓고 나서 읽기만 하면 된다.**

```
model.layers.0.self_attn.q_proj.out_features.expr == "n_h*d_head"
model.layers.0.self_attn.q_proj.in_features.expr  == "d_model"      ← 둘 다 4096
```

`int` 하위 클래스라 torch도 transformers도 평범한 정수로 취급한다. **모델 코드는 안 바뀐다.**
프로토타입 `develop/probe_symbolic_dims.py`.

**복잡한 식이 통째로 나온다** (DeepSeek-V2-Lite, MLA):
```
q_proj              out = n_h*(d_nope+d_rope)                in = d_model
kv_a_proj_with_mqa  out = (c_kv+d_rope)                      in = d_model
kv_b_proj           out = n_h*((d_nope+d_rope)-d_rope+d_v)   in = c_kv
o_proj              out = d_model                            in = n_h*d_v
shared_experts      out = d_moe*E_shared                     in = d_model
```

**태그 유지율 실측** (모듈 차원 속성 기준):

| 모델 | 유지율 | 손실 원인 |
|---|---|---|
| Llama-3.1-8B | 100% (584/584) | — |
| DeepSeek-V2-Lite | 100% (866/866) | — |
| Qwen2.5-0.5B | 100% (440/440) | — |
| gpt-oss-20b | 100% (406/406) | — |
| gemma-2-2b | 100% (489/489) | — |
| Llama-4 Maverick | 85.2% | `d_moe` alias 누락 |
| **합계** | **95.7%** | |

`int(x)`는 모델당 수백 번 호출되지만(`modeling_llama.py:__init__` ×384) **실제 손실은 0**이다.
벗겨진 사본이 따로 생길 뿐, 모듈에 저장된 원본은 태그를 유지한다.

Maverick의 손실은 `d_moe` alias 한 줄을 고치자 85.2% → 95.8%가 됐고, 남은 것은 `floor_scale`
(RoPE 상수, 애초에 차원이 아님)뿐이었다. **진짜 차원은 실질 100%.**

---

## 4. alias 누락 — 버그가 아니라 "지식이 잘못된 자리에 있음"

Llama-4는 expert FFN 폭을 `moe_intermediate_size`가 아니라 그냥 `intermediate_size`에 넣는다.
`rules/symbols.yaml`의 `d_moe` alias에는 앞의 것만 있다. **그런데 결과는 멀쩡했다** —
`src/summarize.py:85`에 예외 처리가 박혀 있기 때문이다:

```python
if out.get("d_moe") is None:
    out["d_moe"] = _first_attr(cfg, ["intermediate_size"]) or out.get("d_ff")
```

즉 **같은 지식이 규칙 파일과 파이썬 코드 두 군데에 있고, 태그는 규칙 파일만 읽는다.**

**해결**: 파이썬에 박힌 예외를 alias 목록으로 승격한다.
```yaml
d_moe:  aliases: [moe_intermediate_size, intermediate_size]
```
⚠ MoE 없는 모델에서 `intermediate_size`는 그냥 dense FFN 폭이다. `group: moe` 장치가 걸러줄
것으로 보이나 **26개 전부 돌려 게이트로 확인해야 한다.**

**진짜 소득**: 태그 방식은 "규칙에 없는 지식이 코드에 숨어 있다"를 **자동으로 찾아낸다.**
표가 안 붙은 정수 속성 목록 = 등록해야 할 필드 목록이다. 지금은 값이 맞으니 아무도 모른다.

```
Llama-3.1-8B   표 없음 → layer_idx 뿐                        → 누락 없음
Llama-4        표 없음 → intermediate_size, expert_dim=8192  → d_moe 누락 발견
```

---

## 5. ★ 미해결 핵심 — 모듈이 없는 축 (전체의 98%)

모듈 선언 차원은 전체 306만 축의 **1.8%**뿐이다. 나머지는 attention 내부 중간 텐서처럼
**물어볼 모듈이 없는 축**이다: `[B, n_h, T, T]` score, `[B, n_h, T, d_head/2]` RoPE 절반,
`[B, n_kv, n_h/n_kv, T, d_head]` repeat_kv 등.

### 5-1. 태그를 dispatch까지 가져오는 길은 막혔다 (검증됨)

`q.view(bsz, q_len, -1, self.head_dim)`에서 `self.head_dim`은 태그 붙은 값이지만,
`TorchDispatchMode`에는 **평범한 `int`로 도착한다.** torch가 C 경계에서 정규화한다.

```
aten.view.default  arg: ['int(1)', 'int(16)', 'int(-1)', 'int(128)']   ← 태그 없음
```

### 5-2. 대안 — 연산 의미로 물려주기

트레이서는 이미 **어느 텐서가 어느 텐서인지(신원)** 와 **무슨 연산인지**를 안다.
그러면 뿌리(모듈 선언 차원 = 태그, + 우리가 정한 실행 파라미터 B·T)에서 시작해 축 이름을
연산 규칙으로 물려줄 수 있다. 대부분 자명하다:

| 연산 | 규칙 | 추측? |
|---|---|---|
| transpose / permute | 이름을 자리바꿈 | 없음 |
| elementwise / norm / 활성 | 입력 그대로 | 없음 |
| matmul / bmm | `배치축 + [왼쪽[-2], 오른쪽[-1]]` | 없음 |
| embedding | `인덱스 축 + weight 마지막 축` | 없음 |
| cat | 이어붙인 축은 이름의 합 | 없음 |
| view / reshape | 부모의 이름을 인수분해 | 아래 참조 |

**op 분포 실측** (26개 모델, 260,812 op):
```
A. 규칙 자명 (전치·elementwise·matmul·norm·활성)   144,638   55.5%
B. reshape 계열 (축 분해/병합)                       56,615   21.7%
C. slice/index 계열                                  26,810   10.3%
D. 나머지 (div, ones, tril, arange, topk …)          32,749   12.6%
```

### 5-3. reshape가 이번엔 왜 되는가 — 예전 실패와의 차이

앵커 작업 때 split 규칙("인접 두 축의 곱이 어떤 모듈 출력 폭과 같으면 분해")을 만들었다가
**껐다.** 가드를 세 번 조여 30k → 7.9k → 4.5k축까지 줄였지만 여전히 우연한 곱에 걸렸다
(Llama-3.1-70B의 `[B, n_h, T, d_head]`에서 `n_h*T == 64*16 == 1024 == n_kv*d_head`).

**실패 원인은 "어떤 모듈이 이 텐서를 만들었는지 몰라 같은 블록의 아무 앵커나 갖다 썼다"**는 것.
신원 기반 전파는 **무엇을 reshape하는 중인지 정확히 안다.** 우연이 개입할 자리가 없다.
`anchors._ENABLE_SPLIT` 주석에 "켜려면 dataflow provenance가 선행되어야 한다"고 적어둔
바로 그 조건이 충족된다.

또 하나: 태그는 코드가 쓴 **순서**를 보존한다(`num_heads * head_dim`). PyTorch reshape도
왼쪽부터 쪼갠다. 둘이 같은 관례라 `n_h*d_head` → `[n_h, d_head]`가 순서까지 맞는다.

### 5-4. 손으로 쓴 전파 규칙은 기각 — 65.9%에 그쳤다

거친 시뮬레이터로 §5-2의 규칙을 재보니 **65.9%**(exact 51.2% + derived 14.8%)에 그쳤다.
장벽은 1,949축 / 5개 op(`t` 1420, `ones` 262, `transpose` 156, `_to_copy` 52, `arange` 38)로
작지만, 근본 문제는 다른 데 있었다: **aten op이 수백 개인데 규칙을 하나씩 손으로 쓰는 것 자체가
또 하나의 "틀릴 수 있는 자리"** 다. 이 프로젝트가 계속 당한 게 그것이다.

### 5-5. ★ 해결 — PyTorch의 심볼 shape 엔진을 쓴다 (실증 완료)

**우리가 규칙을 쓸 필요가 없다. PyTorch가 이미 정확히 그 계산을 한다.**

파라미터를 **SymInt 크기의 FakeTensor**로 만들고 `ShapeEnv` 아래서 실제 forward를 돌리면,
모든 중간 텐서의 shape이 torch 자신의 meta 커널이 계산한 **sympy 식**으로 나온다. 규칙 기반이
아니라 **구성상 정확**하다.

```
embedding   (1, s47, s90)         = [B, T, d_model]
view        (1, s47, s7, s42)     = [B, T, n_h, d_head]   ← reshape 축 분해가 공짜로
transpose   (1, s7, s47, s42)     = [B, n_h, T, d_head]
scores      (1, s7, s47, s47)     = [B, n_h, T, T]
cat         (1, s47 + 1)          = T+1  (KV 캐시 길이)
mask        (1, 1, s47, s47)      = [B, 1, T, T]
```

§5-3에서 "reshape 분해가 이번엔 왜 되는가"를 길게 논했는데, **그 문제 자체가 사라진다.**
`view`의 축 분해를 우리가 유도할 필요가 없다 — torch가 계산해준다.

**실제 모델에서 동작하게 만든 두 가지**

1. **backed 심볼 + hint** (`create_symintnode(..., hint=val)`). 모델 코드는 shape으로 분기한다
   (`if q_len > 1`). unbacked 심볼이면 `GuardOnDataDependentSymNode`로 죽지만, hint가 있으면
   분기가 실제 값으로 판정되고 guard만 기록된다. **제어 흐름이 그대로 돈다.**
2. **파라미터를 빌드 *후* 교체.** transformers 5.x config는 strict dataclass라 SymInt 필드를
   거부한다(`StrictDataclassFieldValidationError`). 반면 `nn.Module` 속성에는 검증이 없다.

**실측** (프로토타입 `develop/probe_symbolic_forward.py`, Linear leaf 7종 + embedding + 1-D norm만
심볼화한 1차 시도):

| 모델 | op 수 | 심볼 축 비율 |
|---|---|---|
| Qwen2.5-0.5B | 2,494 | 59% |
| Llama-3.1-8B | 3,302 | 59% |
| gemma-2-2b | 3,158 | 58% |
| DeepSeek-V2-Lite | — | **실패** |

**DeepSeek 실패가 오히려 설계를 확인해준다.** MLA의 `q_proj.out`은 `n_h*d_head`가 아니라
`n_h*(d_nope+d_rope)`인데 프로토타입이 Llama 배치를 하드코딩했다. 즉 **모듈별 심볼 식을
§3-C 태깅에서 받아와야 한다.** 두 방식이 합쳐진다:

```
태깅(§3-C)     각 파라미터 축이 "무엇"인지    →  n_h*(d_nope+d_rope)
    ↓
SymInt 파라미터로 구성
    ↓
PyTorch 심볼 forward   그 이름이 "어디로 가는지"  →  전 중간 텐서
```

59%는 Linear 7종만 심볼화한 1차 시도값이고, rotary `inv_freq` 등은 구체값으로 남겨둔 상태다.
파라미터 전체 + 모듈이 캐시한 정수까지 심볼화하면 더 올라간다. **다만 최종 도달률은 아직
측정되지 않았다.**

### 5-6. 외부 검토(ChatGPT) 반영 + 실측 — 2026-08-05

외부 LLM에 문제를 정리해 물었고, **주장은 전부 재현한 뒤에만 반영**했다(③ 자유 평가 규약).

**검증 결과**

| 주장 | 판정 |
|---|---|
| `ShapeEnv(duck_shape=False)` | **부분 정정.** 생성자 직접 인자가 아니라 `ShapeEnvSettings` 필드이며 `**kwargs`로 전달된다. 다만 **우리 방식에선 애초에 문제가 아니다** — 심볼을 `ConstantSource`로 명시 생성하므로 값이 같아도 뭉치지 않는다(실측 확인) |
| `specialize_zero_one=False` | **사실이고 필요하다.** 기본값이면 batch=1이 literal `1`로 고정된다. 끄면 `s44` 심볼로 남는다 |
| guard/specialization 감지 API | **사실.** `get_nontrivial_guards()`, `get_pruned_guards()`, `format_guards()`, `replace()`, `replacements`, `var_to_range` 전부 존재 |
| 모든 parameter/buffer 축 심볼화 | **사실이고 효과 큼.** Linear 7종만 → 전 파라미터로 넓히자 59% → 72% |
| 단일 실행(FakeTensor+ShapeEnv+Dispatch) | **사실이고 이미 되고 있다.** §5-5 프로토타입이 이미 한 번의 forward로 둘 다 잡는다 → §5-6 구버전에 적었던 "두 실행 정합" 문제는 **없다** |
| `ModuleTracker` | 존재 (`torch.utils.module_tracker`). 다만 우리는 이미 `ScopeLabeler` hook이 같은 일을 한다 |

**specialization이 진짜 천장이었다** — 그리고 옵션 하나로 풀렸다

전 파라미터를 심볼화해도 모델마다 편차가 컸다. 원인은 guard가 심볼을 상수로 고정하는 것이었고,
`prefer_deferred_runtime_asserts_over_guards=True`로 해결됐다:

| 모델 | 이전 | 이후 | 고정된 심볼 |
|---|---|---|---|
| DeepSeek-V2-Lite (MLA) | 17% | **63%** | 7 → **0** |
| Qwen3-30B-A3B (MoE) | 57% | **73%** | 4 → 1 (`s42->2*s6`, 상수가 아니라 심볼 관계라 무해) |
| OLMoE | 56% | **73%** | 4 → **0** |
| Llama-3.1-8B | 72% | **73%** | 1 → **0** |

고정되던 것들이 `T->16`, `d_model->2048` 같은 **치명적 고정**이었다는 점이 중요하다.

**남은 27%의 정체 — 사실상 없다**

Llama-3.1-8B에서 심볼이 아닌 축의 값을 전부 세어보니 **단 두 종류**였다:

```
1  x3,174   unsqueeze / mask / broadcast 가 만드는 진짜 singleton (원래 값이 1)
4  x  128   GQA repeat factor (n_h/n_kv = 32/8)
```

즉 비심볼 축의 **96%가 값이 진짜 1인 축**이다. 심볼이 "빠진" 게 아니라 **애초에 1**이다.
나머지 `4`는 `n_h/n_kv`로 이름 붙일 수 있는 파생 상수다.

**결론: §5-4의 불확실성은 해소됐다.** 구조적 축은 사실상 전부 추적된다. 단, 아래 조건에서다.

### 5-7. 필수 설정 (이 조합이 아니면 무너진다)

```python
ShapeEnv(specialize_zero_one=False,                      # batch=1 이 심볼로 남는다
         duck_shape=False,                               # 안전장치
         prefer_deferred_runtime_asserts_over_guards=True)  # ★ 이게 없으면 MLA 가 17% 로 붕괴
```
- 심볼은 **backed + hint** (`create_symintnode(sym, hint=val)`). unbacked면 모델 코드의
  shape 분기에서 `GuardOnDataDependentSymNode`로 죽는다
- 파라미터는 **빌드 후** 교체. transformers 5.x config는 strict dataclass라 SymInt를 거부한다
- 모든 parameter·buffer의 **모든 축**을 심볼화. 아는 폭은 이름 심볼, 모르는 폭은 서로 다른
  OPAQUE 심볼(값이 같다고 공유 금지)
- dtype: 일부 모델(DeepSeek, OLMoE)은 BF16을 요구한다 — `rules/error_remedies.yaml`의
  기존 `use_bf16` remedy와 같은 조치이므로 그대로 재사용하면 된다

### 5-8. 함대 전체 실측 (2026-08-05)

`develop/probe_symbolic_forward_v2.py`, 설정은 §5-7. 18개 모델.

**attention 계열 — 전부 동작**

| 모델 | 심볼 | 고정 | | 모델 | 심볼 | 고정 |
|---|---|---|---|---|---|---|
| gpt2-xl | 77% | 0 | | Qwen3-30B-A3B | 73% | 1 |
| gpt-oss-120b | 75% | 0 | | OLMoE | 73% | 0 |
| Llama-4 Maverick | 74% | 0 | | OLMo-2-7B | 72% | 0 |
| GLM-4.5-Air | 74% | 0 | | falcon-7b | 72% | 0 |
| SmolLM3-3B | 73% | 0 | | Qwen2.5-0.5B | 72% | 1 |
| Llama-3.1-8B | 73% | 0 | | DeepSeek-V2-Lite | 63% | 0 |
| | | | | gemma-3-270m | 62% | 1 |
| | | | | **DeepSeek-V4-Flash** | **47%** | 3 |

DeepSeek-V4가 낮은 건 **프로토타입 한계**다. 남은 구체 축이 `1`(65%) 다음으로 `4`(x17,572),
`2`, `8`, `6`, `64`인데 이는 이 모델의 `m_csa`/`m_hca`(압축률), `g_o`, `n_hc` 같은
아키텍처 상수다. 프로토타입 NAMED 맵에 d_model·n_h·d_head·d_ff·d_moe·E만 등록했기 때문이며,
`rules/symbols.yaml`에는 이미 다 있다. **등록만 하면 올라간다.**

"고정"으로 잡힌 3건도 `s68->4*s90` 같은 **심볼 간 관계**이지 상수 고정이 아니라 무해하다.

**SSM / scan 계열 — 실패**

| 모델 | T 심볼 | T 구체 |
|---|---|---|
| Zamba2-1.2B | **크래시** | 12% |
| Nemotron-3-Nano | **크래시** | 12% |
| Qwen3-Next-80B | **크래시** | 4% (op 42,267개로 스캔 언롤) |
| xLSTM-7b | config 검증 실패 (프로파일의 `config_overrides` 필요) | — |

크래시 원인은 정확히 특정된다:
```
RuntimeError: SymIntArrayRef expected to contain only concrete integers
  modeling_zamba2.py:363  pad_tensor_by_size
      F.pad(input_tensor, pad_shape, mode="constant", value=0)
```
Mamba 청크 스캔이 `pad_size = (chunk - T % chunk) % chunk`를 계산하는데 **`F.pad`가 심볼
크기를 거부**한다. T를 구체값으로 두면 크래시는 없어지지만, SSM은 축 대부분이 T 청킹에서
파생되므로 커버리지가 4~12%로 붕괴한다.

### 5-9. SSM/scan 해결 — `T = chunk_size × N_chunk` (검증 완료, 2026-08-05)

2차 외부 검토에서 나온 아이디어를 검증했고 **통한다.** shim이 필요 없다.

Mamba 청크 스캔은 `pad = (chunk - T % chunk) % chunk`를 `F.pad`에 넘기는데, T가 자유 심볼이면
`pad = Mod(256 - Mod(s47,256), 256)`이라 거부된다. 그런데 **T를 `chunk × N_chunk`로 정의하면
`T % chunk`가 심볼적으로 0으로 단순화**되고, pad가 사라져 `F.pad`가 통과한다. T는 여전히
완전히 심볼이다:

```
T free      T=s47        T%CH=Mod(s47,256)   pad=Mod(256-Mod(s47,256),256)   F.pad 거부
T = CH*N    T=256*s85    T%CH=0              pad=0                           F.pad OK
                                                          -> out [1, 256*s85, 64]
```

| 모델 | T 자유 | T 구체 | **T = chunk×N** |
|---|---|---|---|
| Zamba2-1.2B | 크래시 | 12% | **47%** |
| Nemotron-3-Nano | 크래시 | 12% | **48%** |
| Qwen3-Next-80B | 크래시 | 4% | **25%** (chunk=64) |

`chunk_size`는 Zamba2·Nemotron의 config에 그대로 있다(=256). Qwen3-Next는 config에 없어
modeling 파일의 값을 써야 하며, DeltaNet 스캔이 크게 언롤되어(op 48,858개) 여전히 낮다.

**이 방식은 트레이스 seq_len 선택 규칙과 자연스럽게 맞물린다** — `symbolic_shape.resolve_seq_len`이
이미 T를 자동 선택하고 있으므로, SSM 계열에서는 "config 값과 충돌하지 않는" 조건에 "chunk_size의
배수" 조건을 더하면 된다.

### 5-10. 남은 커버리지의 정체 — 대부분 "미등록 상수" 하나의 원인

SSM 잔여 축과 DeepSeek-V4의 낮은 수치가 **같은 원인**이었다.

```
Zamba2 잔여:      1(x7209, 55%)  256(chunk_size)  128(d_state)  5  64  4(conv_kernel)
DeepSeek-V4 잔여: 1(x36091, 65%) 4(x17572, m_csa/m_hca)  2  8  6  64
```

값이 `1`인 축을 빼면, 남는 것은 전부 **그 아키텍처의 구조 상수**다 — chunk_size, d_state,
conv_kernel, 압축률(m_csa/m_hca), 그룹 수(g_o, n_hc). 프로토타입이 d_model·n_h·d_head·d_ff·
d_moe·E만 등록했기 때문이고, **`rules/symbols.yaml`에는 이미 대부분 있다.**
심볼 레지스트리를 그 파일로 교체하면 전 모델에서 함께 오른다. 작업량은 작다.

### 5-11. 채택할 설계 결정 (외부 검토 2차, 반박 없음)

측정으로 검증할 성질의 것이 아니라 설계 취향에 가깝지만, 근거가 타당해 채택한다.

1. **식 3종 분리** — 원본식(코드가 계산한 순서) / 정규화식(SymPy 단순화) / 표시식(문맥에 맞는
   이름). `d_model = n_h*d_head` 같은 관계는 **전역 치환 금지, 동치 후보로만 저장**한다.
   잔차 축에서는 `d_model`, Q 출력에서는 `n_h*d_head`가 맞다.
2. **데이터 의존 심볼은 별도 타입** — `U_nnz`(mask 통과 수), `R_e`(expert e로 라우팅된 토큰 수)를
   아키텍처 심볼과 구분하고, 제약을 함께 기록한다: `0 <= R_e <= B*T*k`, `sum(R_e) = B*T*k`.
   실행값이 나와도 `R_e=37`로 고정하지 말고 `observed=37` 주석으로 둔다.
3. **scan region 추상화** — python 루프는 자동으로 접히지 않는다. 반복 body를 한 번만 기록하고
   `Scan(N_chunk, body)`로 표현하는 모드를 별도로 둔다(Qwen3-Next가 op 48,858개인 이유).
4. **YAML 매핑 검증에 provenance + perturbation** — 필드명만 보지 말고
   `config field → module attribute → parameter axis → tensor axis` 사슬로 검증한다.
   여기서 `develop/probe_config_provenance.py`(섭동법)의 쓸모가 확정된다 — 주 메커니즘이 아니라
   **매핑이 맞는지 독립적으로 확인하는 검증기**다.

### 5-12. ScanRegion — 3차 외부 검토의 엄격 조건은 **반증됐다**

제안: "반복 블록은 op·shape·의존관계·mutation까지 같고, 이전 반복 출력이 다음 반복 입력으로
이어질 때만 Scan으로 인정한다."

**측정 결과 절반만 맞다.**

| 모델 | op_type만 비교 | **+shape·의존관계까지** |
|---|---|---|
| xLSTM `mlstm_backend` | 57 x 16, 99% | **57 x 16, 99%** (유지) |
| Qwen3-Next `linear_attn` | 13 x 63, 79% | **5 x 2, 1%** (붕괴) |

정작 op 폭발이 가장 심한 Qwen3-Next에서 탐지가 죽는다. 이유는 DeltaNet 스캔이
**삼각(triangular) 루프**이기 때문이다 — 반복 k가 폭 k+1로 슬라이스한다:

```
반복 0:  slice -> [..., 1]
반복 1:  slice -> [..., 2]
반복 2:  slice -> [..., 3]
```

즉 **shape이 반복 인덱스의 함수로 변하는 것이 정상**이다. 엄격 조건은 이 계열을 놓친다.

**필요한 조건은 이렇게 완화해야 한다**: op 시퀀스 + 의존관계 *형태*는 동일하되,
**shape은 반복 인덱스 c의 함수로 변하는 것을 허용**한다. 즉 각 축이 상수인지 c의 함수인지
판정하고, 후자면 `f(c)`로 기록한다. xLSTM(상수 shape)과 DeltaNet(삼각) 둘 다 커버한다.

### 5-13. 이 조사가 현재 산출물의 실제 라벨 버그를 찾아냈다

스캔 루프를 들여다보다 **지금 발행된 Qwen3-Next 산출물의 오라벨**을 발견했다. 게이트는
못 잡았다(값이 전부 산술적으로 참이라).

`linear_attn`의 `slice`/`select` 출력 축 중 값이 2~63인 것이 37,728개이고, 그중 63%에
아키텍처 이름이 붙어 있다. 그 안에 **명백한 오라벨 패턴**이 있다:

```
라벨 n_kv          <- 2     x288      라벨 n_h/n_kv      <- 8     x288
라벨 d_conv_lin    <- 4     x288      라벨 k             <- 10    x288
라벨 3*n_kv        <- 6     x288      라벨 3*d_conv_lin  <- 12    x288
```

2, 4, 6, 8, 10, 12 — **하나의 축이 반복마다 커지는 것**인데 각 값마다 다른 아키텍처 이름이
붙었다. 한 축이 반복 1에서 `n_kv`이고 반복 2에서 `d_conv_lin`일 수는 없다. 이것이
[[heuristic-fabricated-labels]] 계열의 전형이며, **값 매칭을 은퇴시켜야 하는 이유의 실물 증거**다.

(주의: 같은 집계의 `n_h_lin_v <- 32` x18,828은 DeltaNet의 value head 수가 실제로 32라
아마 정당하다. 전부가 오라벨은 아니다.)

심볼 방식에서는 이 축이 slice에서 파생된 식(또는 반복 인덱스 `c`)으로 나오지, 무관한 config
이름을 받을 수 없다.

### 5-14. 채택할 나머지 설계 (3차 검토, 반박 없음)

- **ScanRegion 기록 포맷**: `logical_trip_count = N_chunk`(심볼)와 `observed_iterations = 63`
  (실제 실행값)을 분리 기록. PyTorch HOP를 억지로 삽입하지 않고 자체 포맷으로.
  반복 인덱스 `c`, 청크 내 위치 `τ`, 전체 위치 `t = c*chunk + τ`를 shape 축과 별도 개념으로 둔다.
- **데이터 의존 축**: `R[e]`, `U_nnz`를 별도 타입으로. 범위 제약은 ShapeEnv에 넣되
  `sum_e R[e] = B*T*k` 같은 **의미 제약은 별도 ConstraintStore**에. 식 폭발 방지를 위해
  `R_0+...+R_E`로 펼치지 말고 `Sum(e, R[e])` 형태 유지.
- **값이 1인 축은 값이 아니라 생성 근거로 구분**:
  `LiteralSingleton(1)`(unsqueeze/broadcast) / `SemanticDim(n_kv, hint=1)`(MQA) /
  `OpaqueDim(hint=1)`(근거 없음). provenance가 이미 사라진 concrete 1은 사후 구분하지 않고 1로 둔다.
- **증명 경로 기반 검증**: 모든 이름은 `config/module root → parameter/input axis →
  ATen shape rule → 현재 tensor axis`의 증명 경로를 가져야 한다. 여러 이름이 모든 제약을
  똑같이 만족하면 확정하지 말고 `ambiguous` 또는 숫자로 남긴다.
- **★ 다른 hint로 재실행해 숨은 값 매칭 검출**: 합법적인 다른 config 값으로 돌려도 라벨이
  유지되는지 확인한다. §5-13의 버그가 정확히 이 테스트로 잡히는 종류다.
  `d_model`과 `n_h*d_head`처럼 수학적으로 항상 같은 표현은 검증 문제가 아니라
  **문맥별 표시 선택 문제**로 처리한다.

### 5-15. 스캔 축 패턴 실측 — 형태가 둘뿐이다

Qwen3-Next `linear_attn`의 13-op 블록 x 63회에서, 블록 내 **57개 축 전부**의 반복별 값
패턴을 조사했다:

```
constant   43개  (1 x27, 32 x13, 64 x3)
c + 1      14개  (1,2,3,4,5,6,... 등차 1)
```

**비상수 형태가 `c`의 1차식 하나뿐**이다. 4차 외부 검토가 제안한 "상수 또는 `a*c+b`까지
허용" 조건이 실측과 정확히 맞는다. 탐지·표현 모두 간단하다.

또 이 루프는 결과를 stack 하지 않고 **in-place 누적**한다(블록 마지막 op이 `copy_`).
텐서 shape이 루프 전후 `[B, H, 1, 64, 64]`로 동일하고 반복 63 = 64-1이다. 즉 trip count가
shape 축으로 나타나지 않으므로, 이 모델에서는 "반복 횟수 심볼화"가 기록의 문제일 뿐이다.

### 5-16. 불변식을 검사로 만들어 함대에 돌렸다 — 그리고 감사기가 먼저 틀렸다

4차 검토의 제안 중 즉시 검사 가능한 것이 하나 있었다:

> 같은 반복 body의 같은 축은 모든 반복에서 동일한 의미 템플릿을 가져야 한다.
> 값이 `c+1`로 변하는 축에는 고정 아키텍처 이름을 붙일 수 없다.

구현해서 26개 모델에 돌렸더니 **1,342 위반**이 나왔다. 그런데 큰 건(DeepSeek-V4의 624건)을
직접 열어보니 **감사기가 틀렸다**:

```
탐지 주기 = 3,  그런데 실제 값은  1, 4, 1, 4, 1, 4, ...   (주기 2)
라벨:                            B, n_hc, B, n_hc, ...
```

주기를 잘못 잡아 **서로 다른 두 패턴을 비교**하고 있었고, `B<-1`·`n_hc<-4`는 **둘 다 옳은
라벨**이었다. 진짜 루프 인덱스는 단조증가하므로, "변하는 축이 단조가 아니면 주기 오탐"이라는
가드를 넣었다.

가드 후 **1,342 -> 82건**. 신뢰도별로 나누면:

| 모델 | 블록 | 판정 |
|---|---|---|
| Qwen3-Next `linear_attn` | 13 x 63 | **확정.** 값 1..63 단조, 이름 6개(`n_kv`,`d_conv_lin`,`3*n_kv`,`n_h/n_kv`,`k`,`3*d_conv_lin`) |
| Zamba2 `mamba` | 2 x 4 | 약한 신호. 반복 4회뿐이라 확신 부족 |
| Nemotron `mixer` | 2 x 4 | 약한 신호. 상동 |
| DeepSeek-V4 | — | **거짓 양성, 제거** |

**이 경험 자체가 결론의 일부다.** [[verify-the-verifier]]가 기록한 사고("감사 스크립트가
틀려서 잘못된 결론을 보고했다")가 그대로 재현됐다. 새 검사를 게이트에 넣기 전에
**검사기부터 반증**해야 한다.

### 5-17. 새로 발견한 문제 — modeling 코드에 하드코딩된 알고리즘 상수

Qwen3-Next의 DeltaNet 청크 크기 **64는 config에 없다.** config 필드 중 값이 64인 것이
하나도 없고, modeling 파일에 하드코딩되어 있다. 그런데 현재 출력은 이 축을 `d_rope`로
라벨링한다 — 우연히 값이 같아서다.

**config 기반 매핑의 원리적 한계**다. 태깅이든 값 매칭이든 config에서 나올 수 없는 값은
설명할 수 없다. 4차 검토의 제안(채택): config 유래 심볼과 코드 상수를 **별도 타입**으로 두고,
근거가 없으면 `64{algorithm_constant}`처럼 표시하며, AST나 YAML로 역할이 검증된 경우에만
`chunk_size=64`처럼 이름을 붙인다.

### 5-18. 그래서 진짜 남은 문제

1. **ScanRegion 구현** — 탐지 규칙은 실측으로 확정됐다(상수 또는 `a*c+b`, 단조성 가드 필수).
   `major_ops.collapse_repeats`와 같은 기법을 모듈 내부에 적용. 남은 설계는 in-place
   `copy_` 순서 보존과 중첩 루프(안쪽부터 접기)
2. **알고리즘 상수 타입** — §5-17. config에서 나올 수 없는 값의 표기 규약
3. **데이터 의존 축** — 규약 확정(`R[e]`/`U_nnz` + 별도 ConstraintStore + `Sum(e,R[e])` 미전개),
   구현 남음
4. **반복 횟수 기록** — `logical_trip_count` / `executed_iterations` 분리. stack 축은
   "반복당 출력 1개"가 증명될 때만 `N_chunk`로 복원
5. **xLSTM 미검증** — 프로파일 `config_overrides` 미반영 (사소)


## 6. 제안하는 3층 구조

```
1층  이름표 전파        코드가 실제로 한 계산     결정적·검증 가능, 앵커 지점 실질 100%
       ↓ 표가 안 붙으면
2층  LLM이 소스 확인    짧은 미등록 목록만        rules/symbols.yaml에 영구 등록
       ↓ 그래도 안 되면
3층  값 매칭            지금 방식                 최후 폴백 (지금보다 나빠지지 않음)
```

지금은 3층 하나로 다 한다. 1층을 깔면 대부분 거기서 끝나고, 2층이 구멍을 메우고, 3층은 거의
안 쓰인다. **2층은 이미 있는 절차다** — `02-new-module-handling.md`의 Tier 2 리서치가 정확히
"공식 소스 확인해서 규칙에 등록"이다. 지금은 뭘 조사할지 사람이 직접 찾아야 하는데, 1층이
**조사 목록을 자동으로 뽑아준다.**

LLM을 축 라벨링 주체로 쓰는 것은 권하지 않는다 — 306만 축을 판정할 수 없고, 비결정적이며,
게이트로 검증이 안 된다. 이 프로젝트가 계속 당한 게 "검증 없는 주장"이다.

---

## 7. 실행 순서 (하기로 결정할 경우)

| 단계 | 내용 | 크기 | 위험 |
|---|---|---|---|
| **0** | `d_moe` alias 승격 → 26개 재생성 → 게이트 | 작음 | 낮음. 태그와 독립이라 먼저 떼어내 검증 |
| **1** | `Dim` 태깅을 `src/`로 편입, 모듈별 `in`/`out` 식을 `provenance.json`에 저장 | 중간 | 낮음. **재트레이스 불필요** — regen이 이미 meta 빌드를 한다 |
| **2** | 앵커의 이름 출처를 태그로 교체 (값 매칭은 폴백) | 중간 | 중간. 라벨이 바뀌므로 전수 대조 감사 필수 |
| **3** | 미등록 필드 리포트 → C17 게이트 연결 | 작음 | 낮음 |
| **4** | ★ 축 이름 전파 엔진 (§5) | **큼** | **높음**. 도달률 미증명 |
| **5** | 전체 재생성 + 구/신 전수 대조 + 게이트 + 베이스라인 | 중간 | — |

**0→1→2→3을 먼저 끝내고 검증한 뒤 4로 가기를 권한다.** 원인이 섞이면 뭐가 뭘 바꿨는지 못
가린다. 4는 별개 프로젝트로 봐도 된다.

---

## 8. 각 단계에서 기대할 수 있는 것 / 없는 것

**확실한 것**: 모듈 선언 차원(`q_proj.in`/`out`, `embed_tokens`, norm 폭)의 이름이 추론에서
확인으로 바뀐다. 값이 같아도 구분된다. 태그 유지율 실질 100%로 뒷받침된다.

**확실하지 않은 것**: 전체 306만 축 중 모듈 선언 차원은 1.8%다. 나머지는 전파에 의존하고,
전파 도달률은 §5-4대로 **미증명**이다.

**리스크**: 라벨을 바꾸는 변경이라 **지금 맞는 것을 틀리게 만들 수 있다.** 앵커 작업 때 실제로
그랬다 — OLMo-2 회귀 816축, Zamba2 0→36, split 규칙 30k축. 전부 감사에서 잡아 되돌렸다.
**감사를 통과한 것만 반영하고, 통과 못 하면 그 규칙은 끈다**(split 규칙 선례).

---

## 9. 참고 — 만들어둔 프로토타입 (파이프라인 미연결)

| 파일 | 내용 |
|---|---|
| `develop/probe_symbolic_dims.py` | 이름표 전파 (3-C). **권장 방식** |
| `develop/probe_config_provenance.py` | config 섭동 (3-B). 열등하나 독립적이라 교차검증용 |

둘 다 `models/`를 건드리지 않고 읽기만 한다.

---

## 10. 별건 — Kimi-K3 온보딩 (결정 대기)

`moonshotai/Kimi-K3`. 93레이어 하이브리드(MLA full-attn 24 + **KDA linear 69**), MoE E=896
top-16 + shared 2, `situ` 활성화, mxfp4, 1M 컨텍스트.

**막힌 지점**: KDA 69개 레이어가 `fla-core`의 **Triton 커널**로 돈다. meta 텐서에서 실행되지
않고, aten op을 하나도 내지 않아 `TorchDispatchMode`에 불투명하다. repo 모델링 파일에
pure-torch 경로가 없다(`raise ImportError`). Qwen3-Next·xLSTM이 되는 건 transformers가
네이티브 폴백을 제공하기 때문인데, `kimi_linear`는 transformers 5.14.1에 없다.

`fla`에 `naive_chunk_kda`가 있지만 **drop-in이 아니다** — 실제 커널이 L2 정규화·beta sigmoid·
`A_log`/`dt_bias` 게이트를 안으로 흡수하고 있어(`use_qk_l2norm_in_kernel` 등) 그 전처리를
직접 써 넣어야 한다. `ShortConvolution`·`FusedRMSNormGated`도 Triton 모듈이다.

**선택지**: (a) fla torch 레퍼런스 주입 + 어느 레이어가 대체 구현인지 명시, (b) MLA·MoE만
트레이스하고 KDA는 공백, (c) transformers 네이티브 지원까지 보류.

**이미 반영된 것** (이 결정과 무관하게 유효한 파이프라인 버그 2건):
- `provenance.snapshot`이 중첩 `text_config`에 부모 repo id를 물려준다 (빈 `_name_or_path`로
  remote code 해석이 실패했다)
- `loader.py`가 `OutputRecorder`를 `transformers.utils.generic`에 재수출한다 (5.x에서
  `modeling_utils`로 이동했고, 4.5x 기준으로 쓰인 repo 파일이 import 단계에서 죽는다)
- `.venv`에 `einops`·`fla-core` 설치됨 (requirements.txt에는 미반영)
