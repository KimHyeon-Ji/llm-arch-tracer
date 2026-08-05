# llm-arch-tracer — 메인 스펙

이 문서는 LLM에서 레이어 단위 op·shape·의존관계를 뽑는 **전체 방법을 정의한 메인 스펙**이다. `src/`의 참조 구현과 짝을 이루며, **입력(모델 프로파일)부터 결과물 생성, 검증까지** 파이프라인의 각 단계를 규정한다. 무엇이 들어 있나:

- **결과물 뽑는 법** — 산출물 목록·레이아웃(§2), 설계 원칙(§3), 캡처 방식(meta/fake device·ATen dispatch, §4), 실행 단계(Step 1~8, §5), 표 스키마와 주요 op 파생 기준(§6).
- **검증하는 법** — 출력 검증 체크리스트 C1~C16과 리포트·재현성 요건(§9).
- **표기·요약 규약** — 공통 심볼 표기(§10), 구조 요약(`structure.yaml`)·모델 요약(`model_summary.md`) 형식(§11), 구조 라이브러리(§12).

신규/미지 구조를 만났을 때의 대응 절차(Tier 0~3)는 별도 문서 `02-new-module-handling.md`에 있다.

## 1. 목적
Hugging Face에 공개된 LLM의 layer별 연산(op)·shape·의존관계(dependency)를 GPU 없이, 공식 코드(config.json + modeling.py) 실행 결과만으로 재현 가능하게 추출한다.

이 문서는 `src/`의 참조 구현과 짝을 이룬다. 관련 문서: 신규/미지 구조 대응 절차 → `02-new-module-handling.md`.

## 2. 산출물

모델 폴더는 **최상위 6개 파일**(핵심 요약)과 **`full/` 하위 폴더**(전체 근거)로 나뉜다. 최상위에는 latency 분석에 바로 쓰는 주요 op 표와 구조/모델 요약만 두고, 전체 트레이스·프로버넌스·검증 리포트는 `full/`에 넣어 깔끔하게 유지한다.

| 파일 | 내용 |
|------|------|
| `<model>/<phase>.csv` | **주요 operator 표** — op_id, h1·h2·…(계층), op_type, input_shape, weight_shape, weight_pos, output_shape, depends_on, layer_idx, block, sub_block, depth, module_path, raw_op, params, phase, unmapped |
| `<model>/<phase>.jsonl` | 주요 operator 표의 JSONL 형태(csv와 동일 컬럼) |
| `<model>/structure.yaml`(또는 `.json`) | 모델 구조 요약 — 공통 심볼(§10) 기준, layer/block 단위로 롤업 |
| `<model>/model_summary.md` | 모델 정보 요약 + 추출 방법 + 검증 로그(§9 체크리스트) + 교차검증 소스(§11) |
| `<model>/full/<phase>.csv` | **전체 operator 표**(모든 aten 프리미티브, 위와 동일 컬럼) |
| `<model>/full/<phase>.trace.raw.jsonl` | 전체 표 각 행의 원시 aten 호출 근거(csv와 동일 컬럼 + JSON) |
| `<model>/full/provenance.json` | revision hash, 라이브러리 버전, 입력 설정, 적용된 조치 이력 |
| `<model>/full/report.md` | §9 검증 결과 (prefill·decode 모두 반영: C11은 decode, 나머지는 prefill 기준) |

주요 operator 표(최상위 `<phase>.csv`/`.jsonl`)는 전체 표(`full/`)에서 §6.1 기준으로 파생한다 — inference latency에 영향 없는 view/plumbing op은 빼고, norm은 한 행으로 롤업하며, 의존관계는 잘려나간 노드를 관통해 재연결(그래프 축약)해 축소된 표 안에서도 `depends_on`이 유효한 DAG를 이룬다.

별도 `.graph.json`은 내지 않는다 — 의존관계 그래프는 각 행의 `depends_on` 컬럼(csv·jsonl에 이미 존재)으로 완전히 복원된다. `report.json`도 내지 않는다 — 같은 체크 결과가 `model_summary.md`의 검증 로그 표에도 실리고, `report.md`는 줄 단위로 파싱 가능하다.

`<phase>.csv`/`.jsonl`(및 `full/`의 짝)은 prefill·decode 각각 별도 파일로 나온다(P6). `structure.yaml`·`model_summary.md`·`full/report.md`는 모델 1개당 1세트다 — 아키텍처 자체는 prefill/decode가 공유하는 정적 구조라 중복해서 낼 필요가 없고(prefill 기준으로 생성), 다만 report의 C11 항목만 decode 표를 따로 검사한다.

## 3. 원칙

| # | 원칙 |
|---|------|
| P1 | 모든 값은 config 숫자 또는 실행 결과에서만 나온다. 추측·임의 채움 금지, 항상 출처 추적 가능. |
| P2 | 실제 가중치 연산 금지. meta device(1순위) → FakeTensorMode → torch.export 순으로만 실행. |
| P3 | op 캡처는 dispatch(ATen) 층위에서 한다. module hook은 라벨링 전용이며 op 근거로 쓰지 않는다. |
| P4 | 재현성: revision hash·라이브러리 버전·입력 설정·적용 조치를 provenance에 기록. 동일 입력 → 동일 결과. |
| P5 | 확장성: 신규 모델 추가는 프로파일 1개로 끝난다(트레이서는 모델 무관). |
| P6 | prefill과 decode는 각각 트레이싱해 별도 표로 낸다. |
| P7 | 표에 걸린 param을 가진 모든 module은 표에 최소 1개 op로 기여해야 한다(커버리지 강제). |
| P8 | 매핑에 실패한 op은 숨기지 않고 `unmapped`로 표시한다. |

## 4. 캡처 방식

- **module hook만으로는 불충분하다.** hook은 module의 forward 경계에서만 발화하므로, attention 내부의 softmax·RoPE·정규화 연산 같은 functional 연산은 잡히지 않는다. 따라서 hook은 **"지금 어느 layer/block인가"를 표시하는 라벨링 전용**으로만 쓴다.
- **op 캡처는 `TorchDispatchMode`로 한다.** forward 중 dispatch되는 모든 ATen 연산을 가로챈다. meta 또는 fake tensor 위에서 실행되므로 실제 수치 계산 없이 shape과 연산 그래프만 얻는다.
- **의존관계(depends_on)는 텐서 신원 추적으로 얻는다.** 각 op이 실행될 때 입력 텐서를 만든 이전 op의 id를 조회해서 채운다. 코드 정적 분석이나 규칙 추정이 아니다.
- 가능한 경우 `torch.export`로 op 개수·의존 구조를 교차검증한다(동적 제어흐름이 있는 모델은 실패할 수 있으며, 그 경우 주 캡처만으로 진행한다).

## 5. 실행 단계

**Step 1 — Provenance 확보** (`src/provenance.py`)
모델 revision을 commit hash로 고정하고 config를 스냅샷한다. 설치된 라이브러리 버전에서 해당 아키텍처를 지원하는지 먼저 확인한다.

**Step 2 — Meta 로드** (`src/loader.py`)
```python
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
model.eval()
```
가중치 값 없이 config 숫자만으로 구조를 구성한다.

**Step 3 — 아키텍처 introspection** (`src/introspect.py`)
프로파일에 값을 하드코딩하지 않고 config를 훑어 자동 도출한다:
- 안전한 최소 seq_len(압축률·top-k·window 등 seq_len을 제약하는 모든 config 숫자의 최댓값 기반)
- 레이어별 스케줄(레이어 타입이 균일하지 않으면 이후 검증에서 클러스터별로 다뤄야 함)
- 메인 forward 밖에 있는 추가 진입점(보조 예측/추론 모듈 등)

**Step 4 — 입력 구성** (`src/inputs.py`)
- 토큰 ID는 `torch.arange(seq_len) % vocab_size`로 구성한다(0으로 채우지 않는다). 토큰 ID에 의존하는 라우팅·분기 구조가 있으면 0 입력은 이를 왜곡시킨다.
- forward 시그니처에 실제로 존재하는 인자만 전달한다(모델마다 다르므로 시그니처를 확인 후 필터링).
- **명시적 `attention_mask`는 넘기지 않는다.** 단일 무패딩 시퀀스에서 all-ones 마스크는 중복이고, 최신 transformers는 padding mask 값을 검사(`fast_all`→`.item()`)해 causal mask 생략 여부를 판단하는데 이 값 검사가 meta 텐서에서는 불가능하다(`Tensor.item() cannot be called on meta tensors`). sdpa·eager 모두 동일하게 실패하므로 `attn_implementation` 문제가 아니다. 마스크를 생략하면 모델이 `cache_position`/`position_ids`로 순수 causal mask를 직접 구성한다(P2 meta-first의 실무 귀결).
- prefill은 `use_cache=True`로 실행해 decode에 필요한 cache를 모델이 직접 생성하게 한다.

**Step 5 — 트레이싱** (`src/scope.py`, `src/tracer.py`, `src/adapt.py`)
ScopeLabeler(hook, 라벨링 전용)와 OpGraphTracer(dispatch, 실제 캡처)를 동시에 켜고 forward를 1회 실행한다. 실패하면 오류를 분류해 조치를 적용하고 재시도한다(상세 절차는 `02-new-module-handling.md`).

**Step 6 — Decode**
Cache shape을 손으로 추정하지 않는다. prefill 실행 결과로 모델이 만든 cache를 그대로 받아, 새 토큰 1개만 넣어 재실행한다.

**Step 7 — 정규화** (`src/normalize.py`)
raw aten 이름을 사람이 읽는 `op_type`으로 매핑한다(`rules/optype_map.yaml`, 확장 가능). 매핑되지 않는 op은 raw 이름을 유지하고 `unmapped=True`로 표시한다.

**Step 8 — 표/그래프 생성** (`src/build_table.py`)
아래 스키마로 CSV와 그래프 JSON을 만든다.

**Step 8.5 — 구조 요약·모델 요약 생성** (`src/summarize.py`)
표를 layer/block 단위로 롤업해 공통 심볼(§10) 기준 `structure.yaml`을 만들고, 이를 바탕으로 `model_summary.md`를 만든다. 구조 요약 자체는 표에서 결정적으로 도출되지만(코드만으로 충분), `model_summary.md`의 "참고 소스" 절은 §11의 Tier 2 리서치 결과가 있어야 채워진다 — 없으면 빈 채로 표시하고 리서치가 필요하다는 걸 명시한다.

## 6. 표 스키마

| 컬럼 | 의미 | 출처 |
|------|------|------|
| op_id | 실행 순서(PK) | 트레이서 |
| op_type | 정규화된 op 이름 | 정규화 규칙표 |
| input_shape / output_shape | 텐서 shape 리스트 — **심볼 표기**(§10) | 실행 결과 shape을 심볼로 렌더 |
| weight_shape | 소비한 대표 weight shape — **심볼 표기** | param 귀속 |
| weight_pos | 그 weight가 `input_shape`의 **몇 번째 피연산자인가**(§6.2) | 텐서 신원 + shape 대조 |
| depends_on | 입력을 만든 선행 op_id | 텐서 신원 추적 |
| layer_idx / block / sub_block | decoder layer 번호 / 상위·하위 블록 | scope |
| depth / h1 · h2 · … | 모듈 중첩 깊이 / 레벨별 계층 컬럼 | scope (module_path 분해) |
| module_path / raw_op / params | 추적성용 원본 정보 | 트레이서 |
| phase | prefill / decode | 실행 설정 |
| unmapped | 정규화 매핑 실패 여부 | 정규화 |

- **shape은 숫자가 아니라 심볼로 낸다**(csv·trace.raw.jsonl 공통). 구체적 batch(1)·seq_len은 우리가 임의로 고른 실행 파라미터라, 그대로 숫자로 박으면 산출물이 특정 실행에 묶인다. 대신 각 축을 아키텍처 심볼(§10: `B, T, d_model, n_h, d_head, E, k, …`)이나 단순 식(`B*T, n_h*d_head, 2*d_head, T*k, T+1, d_head/2`)으로 렌더한다. 어떤 심볼/식에도 안 맞는 순수 구조 상수만 정수로 남긴다(지어내지 않음, P1). 구체 숫자는 `provenance.json`의 `seq_len_used`+`symbol_table`로 완전 복원 가능하다(추적성 유지). 심볼 해소가 유일하도록 트레이스 seq_len(T)은 config의 어떤 차원값과도 겹치지 않게 자동 선택한다(§10 참고).
- **계층(hierarchy)을 컬럼으로 편다.** op을 순서대로만 나열하지 않고, `module_path`를 레이어 아래 모듈 중첩(예: `self_attn → q_proj`, `mlp → experts → act_fn`)으로 분해해 `depth`와 `h1, h2, …` 레벨 컬럼으로 낸다. 이러면 어느 수준으로든 그룹핑/롤업이 되고, HF 모듈이 겹겹이 중첩된 구조가 표에서 보인다.
- **컬럼 물리 순서**(모든 파일 공통): `op_id` → `h1, h2, …`(모델별 최대 깊이만큼) → `op_type` → `input_shape` → `weight_shape` → `weight_pos` → `output_shape` → `depends_on` → `layer_idx` → `block` → `sub_block` → `depth` → `module_path` → `raw_op` → `params` → `phase` → `unmapped`. 계층 컬럼을 op_id 바로 뒤 앞쪽에 두어 각 행을 "구조 먼저(op_id + 모듈 트리 위치)"로 읽게 한다.
- **shape·list 필드 직렬화**: 중첩 리스트인 shape/`depends_on`/`params`는, **CSV에선 읽기용으로 심볼을 따옴표 없이 bare**로 낸다(예: `[[V, d_model], [B, T]]`) — dim 심볼에 쉼표가 없어 모호하지 않다. **`.trace.raw.jsonl`·`.jsonl`은 순수 JSON**(`[["V","d_model"], …]`)이라 프로그램 파싱은 이쪽을 쓴다. 같은 데이터의 표기 차이일 뿐이다.

### 6.1 주요 operator 표 (`<phase>.csv`/`.jsonl`) — latency 관점 파생

전체 표(`full/`)는 모든 aten 프리미티브를 담아 view/transpose 같은 레이아웃 op까지 보이지만, inference latency는 `max(FLOPs/연산처리율, 이동바이트/대역폭)`로 결정되므로 **FLOPs가 크거나 큰 텐서를 읽고/쓰는 op만** latency에 의미가 있다. 최상위 주요 operator 표는 전체 표에서 아래 기준으로 파생한다(`src/major_ops.py`). 컬럼은 전체 표와 동일하고, `op_id`는 0부터 다시 매기며, `depends_on`은 잘려나간 op을 관통해 가장 가까운 살아남은 선행 op으로 재연결(그래프 축약)한다 — 축소된 표 안에서도 유효한 DAG가 된다.

- **위치 인코딩 precompute 모듈 통째로 제거**: `module_path`에 `rotary`/`rope`가 들어간 모듈(cos/sin 사전계산)은 forward당 1회 계산 후 전 레이어에 브로드캐스트되는 상수라 per-layer 비용에서 무시 가능 → 전부 제거. (RoPE *적용*은 attention 안의 elementwise로 남고, 아래 크기 게이트로 걸러진다.)
- **정규화 모듈은 1행으로 롤업**: leaf 이름에 `norm`이 있거나 GPT-2 `ln_1/ln_2/ln_f`인 모듈의 op들(RMSNorm은 `pow→mean→add→rsqrt→mul`로 분해됨)을 하나의 `rmsnorm`/`layernorm` 행으로 합친다(`rsqrt` 있으면 rmsnorm). input은 그룹 진입 op의 입력, output은 마지막 op의 출력, weight_shape는 `*.weight`를 소비한 op의 1-D 피연산자. 이때 input과 weight가 **서로 다른 멤버 op에서 오므로** `weight_pos`는 합쳐진 행 기준으로 다시 구한다 — 보통 `-1`(scale weight가 뒤쪽 멤버의 mul로 흡수돼 피연산자에 안 남음)이지만, `aten.native_layer_norm`처럼 weight를 피연산자로 직접 받는 경우(GPT-2)는 `1`이 된다.
- **항상 유지**(연산/attention/활성 코어): `linear, matmul, batched_matmul, grouped_matmul, sdpa, conv1d, embedding, softmax, silu, gelu, relu, sigmoid, tanh, exp, layernorm, rmsnorm`.
- **크기 게이트 후 유지**: `elementwise_add, elementwise_mul, concat, sum` — 피연산자/출력의 마지막 축이 **wide 심볼**(`d_model`/`d_ff`/`d_moe`)일 때만. residual add·GLU gating·MoE combine은 남고, 작은 RoPE 적용 mul·rotate_half/KV-append concat·attention mask add(전부 `d_head`·`T` 스케일)는 빠진다.
- **그 외 전부 제거**: view/transpose/expand/slice/select/clone/copy/cast, RoPE 삼각함수, MoE 라우팅 plumbing(topk/sort/gather/scatter/cumsum/where/…).

**반복 레이어 접기(repeat folding).** 위 필터 뒤, 구조가 같은 decoder 레이어를 대표 블록 하나로 접는다(`major_ops.collapse_repeats`). full 표는 전 레이어를 그대로 펼쳐 두지만, 주요 표는 기본 블록 단위만 깔끔하게 보이게 한다.
- **서명(signature)** = 레이어의 **full 트레이스 op 시퀀스**에 대한 `(op_type, 레이어번호 뗀 module_path, 심볼 input/weight/output shape)` 튜플(`major_ops.full_layer_signatures`). 그룹핑은 레이어 **전체 구조** 기준이라 — (a) 블록은 레이어 단위 통째로 묶고(공통 부분을 빼내지 않음: DeepSeek의 dense·MoE 레이어는 MLA attention이 완전히 같아도 MLP가 달라 별도 블록으로 남는다), (b) major에서 버리는 op만 다른 레이어도 별개로 센다(NoPE 레이어 vs RoPE 레이어는 rotary 적용 op만 달라도 별도 블록). 결과적으로 블록 수 = full 트레이스의 실제 per-layer 아키텍처 종류 수와 일치한다. 서명이 같은 레이어는 첫 등장 레이어만 남긴다. 이질적 레이어(DeepSeek dense→MoE, Nemotron Mamba/MLP/attention 교대)는 자동으로 별도 블록이 된다.
- 각 행에 **`block_type`**(블록 구성 라벨: `attn+FFN`, `MLA+MoE`, `SSM`, `xLSTM` 등 — mixer부 + FFN/MoE부를 op 구성에서 도출), **`repeat`**(그 블록이 대표하는 레이어 수), **`layers`**(해당 레이어 인덱스, `0-2`·`1,3,5` 식 압축)를 붙인다 — 컬럼 순서는 `op_id` 바로 뒤(`op_id → block_type → repeat → layers → h1 …`)라 어떤 종류의 블록이 몇 번 반복되는지 한눈에 보인다. 레이어에 안 속한 1회성 op은 `block_type`이 `embed`/`norm`/`head`(또는 미분류 `-`), `repeat=1`, `layers`=빈칸.
- 잘려나간 반복 레이어를 가리키던 `depends_on`은 위치 매핑으로 대표 블록의 대응 op에 재연결한다 → embed → block(×N) → norm → lm_head 형태의 유효한 DAG가 유지된다. latency는 블록 비용 × `repeat` + 1회성 op으로 계산하면 된다.

### 6.2 `weight_pos` — `input_shape` 안의 어느 피연산자가 weight인가

`input_shape`는 **그 aten op이 실제로 받은 모든 텐서**의 리스트다(`tree_flatten((args, kwargs))` 결과 그대로). "activation만"이 아니다. 그래서 weight를 쓰는 op에선 같은 텐서가 `input_shape` 안에도, `weight_shape`에도 나온다 — `weight_pos`는 그 둘이 어디서 겹치는지를 가리키는 컬럼이다.

| 값 | 뜻 |
|---|---|
| 빈칸(`null`) | 이 op엔 weight가 없다. `input_shape` 전부가 activation |
| `0` 이상 | `input_shape[weight_pos]`가 그 weight다 |
| `-1` | weight는 있으나 피연산자 리스트에 그 shape가 **그대로는** 없다 — 융합됐거나(RMSNorm), reshape/slice된 view로 들어갔다(DeepSeek-V4 `o_a_proj`) |

**실제 연산이 어떻게 일어나는가.** `nn.Linear`는 weight를 `[out, in]`으로 저장하고 `y = x @ W.T`를 계산한다. 그래서 `aten.t`가 중간에 끼고, 표에서는 두 컬럼이 전치 관계로 보인다:

```
model.layers.0.feed_forward.gate_proj   (Llama-4-Maverick, decode)

  저장된 W   [d_ff, d_model]   = [16384, 5120]     <- weight_shape
    | aten.t
  W.T        [d_model, d_ff]   = [5120, 16384]     <- input_shape[1], weight_pos=1
  x          [B, d_model]      = [1, 5120]         <- input_shape[0]
    | aten.mm
  y = x @ W.T  [B, d_ff]       = [1, 16384]        <- output_shape
```

내적 차원 `d_model`이 `x`의 뒤와 `W.T`의 앞에서 만나 소거되고 `d_ff`가 남는다. 별개의 텐서 두 개가 아니라, **같은 weight를 저장 형태와 연산 투입 형태로 각각 적은 것**이다.

**op마다 자리가 다르다.** 피연산자 개수와 weight 위치는 op의 호출 규약을 따른다.

| op | input_shape | weight_pos |
|---|---|---|
| `matmul` (Linear) | `[activation, W.T]` | `1` |
| `linear` (`addmm`, bias 있는 GPT-2·Qwen2.5) | `[bias, activation, W.T]` | `2` |
| `batched_matmul` (attention `Q@K^T`·`attn@V`) | `[activation, activation]` | 빈칸 |
| `batched_matmul` (MoE expert, 3-D `nn.Parameter`) | `[activation, W]` — 전치 없음 | `1` |
| `embedding` | `[W, index]` | `0` |
| `rmsnorm` (융합) | `[activation]` | `-1` |

즉 **"matmul이면 weight가 있다"가 아니다.** attention의 두 bmm은 피연산자 둘 다 activation이고, MoE expert weight는 전치 없이 그대로 들어가 `input_shape[1] == weight_shape`가 된다.

**FLOPs·바이트를 구할 때.** FLOPs는 `input_shape`만으로 나온다(matmul 계열은 `2 × input[0] ⊗ input[1]`) — `weight_shape`는 더하지 않는다. 바이트를 셀 때만 `weight_pos`가 필요한데, `weight_pos >= 0`인 피연산자는 weight 트래픽이고 나머지가 activation 트래픽이다. 이 구분 없이 `input_shape` 전부를 activation으로 세고 `weight_shape`를 또 더하면 weight를 두 번 세게 된다. `weight_pos = -1`인 행은 `weight_shape`가 파라미터 크기의 유일한 출처다.

> 산출 방식: 값의 정의는 항상 "`weight_shape`와 shape가 같은(또는 마지막 두 축만 뒤집힌) 피연산자"다. 트레이서의 **텐서 신원**(`param_origin`)은 그 조건을 만족하는 피연산자가 여럿일 때 **동점 처리에만** 쓴다 — 그래야 새로 트레이스한 모델과 재생성된 모델이 같은 뜻을 갖는다. 재생성 경로엔 신원 정보가 없어 shape 대조로 되돌리고(`build_table.derive_weight_pos`), 트레이스 시점 값은 concrete 사이드카에 저장돼 재생성에서 복원된다.

### 6.3 라벨의 1순위 근거는 **모듈이 선언한 차원**이다 (`src/anchors.py`)

심볼 렌더링은 원래 **정수 하나 + 경로 정규식**으로만 결정했다 — `dim(5120, module_path)`가 config 값들을 뒤져 맞는 심볼을 찾고, 값이 겹치면 `scope:`로 우선순위를 조정하는 방식이다. 지금까지 발견된 라벨 오류는 **전부 이 설계 하나**에서 나왔다: 값이 우연히 겹치면 구조가 아니라 priority가 이름을 정한다.

그런데 트레이스는 각 op이 **어떤 파라미터를 소비했는지 텐서 신원으로 이미 확정**하고 있고(`params`), `nn.Linear`의 weight shape는 `[out_features, in_features]`다. 즉 **모듈이 자기 폭을 직접 선언**한다. 이게 값 매칭보다 강한 근거다.

`anchors.py`가 그 선언을 앵커로 쓴다.

- **추출** — 파라미터를 가진 모든 모듈의 `in`/`out` 폭을 트레이스에서 복원한다. 어느 축이 `in`인지는 **곱셈 자체에서** 읽는다(활성 입력의 마지막 축과 같은 축이 contract되는 축). Linear는 전치된 weight를, 배치 expert weight는 저장 그대로를 받으므로 `[0]=out` 같은 고정 관례는 한쪽에서 반드시 틀린다. 구체 shape 사이드카(§6.2)만 있으면 되므로 **이미 발행된 모델도 재트레이스 없이** 적용된다.
- **모듈당 1회 확정** — 한 모듈의 라벨을 한 번만 정하고 모든 레이어·모든 op에서 재사용한다. 같은 파라미터가 transpose에서와 matmul에서 다른 이름을 갖던 문제가 구조적으로 사라진다.
- **데이터플로우 교정** — 모듈의 `in`은 **생산자의 `out`과 같은 텐서**이므로 같은 이름이어야 한다. Llama-3.1-405B는 `d_model == n_h*d_head == 16384`라 값만으로는 `q_proj`의 입력이 잔차 스트림인지 패킹된 head 배치인지 못 가린다 — 생산자(`input_layernorm`, 폭이 명백히 d_model)가 결정한다.
- **전파** — 앵커가 정한 이름은 그 **텐서**의 이름이므로, 같은 텐서를 보는 모든 op의 입력·출력 라벨에 전파한다. 소비자 쪽만 고치면 생산자 쪽과 어긋나 "한 텐서 두 이름"이 오히려 늘어난다.

**적용 범위는 그 모듈의 파라미터를 실제로 소비하는 행으로 한정한다.** "이 모듈에 들어오는 폭 = in_features"는 파라미터를 적용하는 op에서만 참이고, 모듈 안에서 도는 임의의 op에는 참이 아니다. DeepSeek-V4-Pro의 compressor는 `new_zeros([1,512,8,512])`를 부르는데 그 끝 축 512가 모듈 입력 폭과 우연히 같아서, 범위를 넓혔더니 압축 시퀀스 축(`T/m_csa`, 옳은 라벨) 1,440개가 `d_head`로 덮였다.

**측정 결과(26개 모델, 306만 축).** 3,422축이 바뀌었고 전부 교정 방향이다 — 잔차 스트림이 `n_h*d_head`/`n_h*d_head/g_o`/`n_h*d_v`로 잘못 찍혀 있던 것이 `d_model`로, norm 폭이 `T/m_csa`(정적 파라미터가 시퀀스 길이일 수 없다)로 찍혀 있던 것이 `d_head`로. 게이트의 데이터플로우 불일치(`flow_ambig`)는 **함대 전체 2,929 → 1,547 (-47%)**, 퇴행 0건.

> **켜지 않은 규칙 하나.** "인접 두 축의 곱이 어떤 모듈의 출력 폭과 같으면 그 폭의 `[count, size]` 분해"라는 규칙을 만들었다가 **비활성화했다**(`anchors._ENABLE_SPLIT`). 안전한 모델에서는 이미 맞는 라벨을 다시 맞게 만들 뿐이고(head-count/size 혼동은 `symbolic_shape._HEAD_COUNT_EXCLUSIVE`가 이미 원천에서 막는다), 구조가 특이한 모델에서는 **우연한 곱**으로 맞는 라벨을 덮었다 — 가드를 세 번 조이며 30k → 7.9k → 4.5k축까지 줄였지만 DeepSeek-V4의 compressor/indexer에서 여전히 ~1.7k가 남았다. "곱이 맞는다"는 근거가 아니라 우연 탐지기다. 켜려면 `depends_on`을 따라 **그 텐서를 실제로 생산한 모듈**을 찾아 그 모듈의 분해만 쓰는 작업이 선행되어야 한다.

### 6.4 미등록 config 필드 리포트 (`src/symbolic_dims.py`)

§6.3의 앵커는 모듈이 **선언한** 폭을 읽지만, 그 폭을 뭐라 **부를지**는 아직 값 매칭이다.
`5120`을 `d_model`이라 하는 근거가 "config에 5120인 필드가 있다"인 것이다.

`symbolic_dims.py`는 그 근거를 한 단계 끌어올린다. dim 심볼에 해당하는 config 필드를
**이름표가 붙은 `int` 하위 클래스(`Dim`)로 교체**한 뒤 모델을 짓는다. 연산자가 이름을
전파하므로, 다 짓고 나면 모듈에서 **읽기만 하면** 된다:

```
model.layers.0.self_attn.q_proj.out_features.expr == "n_h*d_head"
model.layers.0.self_attn.q_proj.in_features.expr  == "d_model"     # 둘 다 4096
```

`int` 하위 클래스라 torch·transformers는 평범한 정수로 취급한다 — 모델 코드는 그대로다.
심볼→값 매핑은 `aliases`를 직접 훑지 않고 **`summarize.resolve_symbols()`를 쓴다**. YAML이
아니라 파이썬에 들어 있는 보정(Llama-4는 expert 폭을 `moe_intermediate_size`가 아니라 그냥
`intermediate_size`에 넣는다)까지 따라가고, 그 모델에 없는 심볼은 `None`으로 돌려주므로
dense 모델이 엉뚱한 `d_moe`를 얻지 않는다.

**지금 반영된 것은 리포트뿐이다.** 라벨 출처를 태그로 바꾸는 것은 아직 하지 않았다 —
아래 이유로 식 정리 계층이 선행되어야 한다.

**태그가 안 붙은 정수 모듈 속성 = 등록 안 된 config 필드**다. 이 목록이
`structure.yaml`의 `unregistered_fields`와 `model_summary.md`의 「미등록 config 필드」 절로
나간다. 지금까지는 이 구멍이 **보이지 않았다** — 값 매칭이 조용히 뭔가를 골라주기 때문이다.
26개 중 **17개는 목록이 비어 있고**, 나머지 9개는 짧고 구체적이다:

```
Zamba2       chunk_size=256, group_size=4096      DeepSeek-V3  qk_head_dim=192, num_group, topk_group
Nemotron     chunk_size=256                       DeepSeek-V4  compress_rate, hc_sinkhorn_iters
Llama-4      intermediate_size=8192, expert_dim   xLSTM        v_dim=4096, up_proj_dim=8192
```

`02-new-module-handling.md`의 Tier 2 리서치가 "무엇을 조사해야 하는지"를 사람이 직접 찾아야
했는데, 이제 **조사 목록이 자동으로 나온다.**

**식 정리 계층** (`normalize()`). 태그가 기록하는 것은 코드가 실제로 한 산술이라 정리가 안 돼
있다. 원본식은 `*_raw`로 보존하고(그게 근거다), 표시용은 sympy로 정리한 뒤 이 프로젝트의
shape 셀 관례(개수 먼저, 공백 없음, 나눗셈은 `a/b`)로 렌더한다.

```
n_h*(((d_nope+d_rope)-d_rope)+d_v)  ->  n_h*(d_nope+d_v)
(c_kv+d_rope)                       ->  c_kv+d_rope
1*d_head                            ->  d_head
d_g*g_o                             ->  g_o*d_g
```

**순서가 중요하다 — 원자화가 정리보다 먼저다.** `head_dim`이 config에 없어
`hidden_size // num_heads`로 계산되는 모델(Qwen2.5 등)에서는 q_proj 폭이 `n_h*(d_model/n_h)`로
기록되는데, 이걸 sympy에 넘기면 **`d_model`로 접힌다.** 대수적으로는 옳지만 의미는 뒤집힌다 —
Q 투영 출력은 패킹된 head 배치이지 잔차 스트림이 아니다. 그래서 `Dim` 산술 단계에서
**파생 심볼을 자기 이름으로 되돌린다**(`_atomize`): `resolve_symbols`가 `d_head`를
`d_model // n_h`로 정의했다는 사실을 알고 있으므로, 그 정의를 선언된 이름으로 되돌리는 것이지
값으로 추측하는 것이 아니다. sympy는 값을 모르므로 그 뒤로는 유효한 대수만 한다.

**태그를 1순위 이름 출처로 채택했다** (2026-08-05). 앵커가 축을 정하고, 그 축의 **이름**을
태그에서 가져온다. 채택 조건은 둘이다.

1. **정수 리터럴이 남아 있으면 거부한다**(`anchors.tag_is_usable`). 태그는 코드가 계산한 것을
   그대로 적으므로, 이름 없는 숫자가 남았다는 건 그 자리에 **규칙 구멍**이 있다는 뜻이다.
   DeepSeek-V3의 `q_b_proj`가 `192*n_h`(192 = 미등록 `qk_head_dim`), Zamba2의 mamba
   `in_proj`가 `...+8192`로 나온다. 계수(`2*d_moe`)는 정상이므로 거부 대상이 아니다.
2. **그 축의 실제 값과 맞아야 한다**(`_evaluates_to`). 태그는 모듈의 in/out 관점이고 앵커는
   곱셈에서 얻은 축 역할이라, 드물게 어긋난다 — Zamba2 `linear_q_adapter`는 둘이 반대였다.
   심볼표로 계산해 축의 구체값과 일치할 때만 채택한다.

**함대 전체 5,227축이 바뀌었고 방향이 일관된다**: `n_h*d_head -> d_model` 3,368 (잔차 스트림),
`n_h*d_head/g_o -> d_model` 548, `n_h*d_v -> d_model` 108, `2*d_moe -> E_shared*d_moe` 156,
`n_h*d_head -> n_h_lin_v*d_head_lin_v` 72 (Qwen3-Next는 linear attention 헤드다).

> **베이스라인을 올린 근거.** 이 변경으로 `flow_ambig`가 xLSTM +2, OLMo-2 +3 늘었다. 규칙상
> 퇴행이지만 내용을 확인하고 수용했다: OLMo-2 한 모델에서 **384축이 옳아지고 3축이 나빠졌다**
> (128:1). 나빠진 3축은 `q_proj` 안의 `view` 입력으로, 이웃이 아직 옛 이름을 들고 있어 생긴
> 불일치다. `flow_ambig`는 **둘 다 산술적으로 참인** 모호 불일치를 세는 지표이고,
> 증명 가능한 오류를 세는 `label_false`는 전 모델 0으로 유지된다.

> **함께 고친 것 — `weight_pos` 사이드카 고착.** "사이드카 값 우선, 없으면 제거 후 재유도"
> 규칙이 **None에 고착**되는 상태를 만들었다: 이 필드가 추가된 첫 재생성에서 사이드카에 None이
> 쓰였고, 이후 모든 재생성이 그걸 다시 읽었다. `declared_dims`는 `weight_pos`로 활성 피연산자를
> 찾으므로, 그동안 **weight를 활성으로 오인**하고 있었다. 34,937행 전부 복구했다.

> **켜지 않은 것 두 가지.** (1) `relabel` 규칙 1(weight 축 라벨링)은 원래부터 no-op이었고,
> 살렸더니 396건의 산술적 거짓 라벨이 생겨 되돌렸다 — 곱셈에서 얻은 축 역할이 모듈의 in/out
> 관점과 항상 일치하지는 않는다. (2) 활성 라벨을 weight의 contract 축에 복사하는 재핀
> (`_ENABLE_REPIN`)은 op 내부 불일치를 없애주지만 xLSTM +2 / OLMo-2 +3을 유발해 껐다.
> 둘 다 근거를 코드 주석에 남겼다.

**미등록 필드를 규칙으로 승격했다** (2026-08-05). 리포트가 뽑아준 목록을 1차 소스(지금 실행 중인
modeling 코드)로 하나씩 확인하고 출처와 함께 등록했다.

| 등록 | 근거 |
|---|---|
| `n_grp`, `k_grp` (그룹 제한 라우팅) | 라우터가 `scores.view(-1, n_group, E//n_group)` 후 `topk(k=topk_group)` — 둘 다 실제 텐서 축. `modeling_deepseek_v3.py:147-162` |
| `d_chunk` (SSM 청크 길이) | `reshape_into_chunks`가 `[bsz, seq_len, ...] → [bsz, -1, chunk_size, ...]`. `modeling_nemotron_h.py:72-84` |
| `d_nope+d_rope` (derived) | V3는 config `__init__`에서 `qk_head_dim`을 미리 계산해 필드로 들고 있어 모델 코드가 합을 다시 계산하지 않는다. `configuration_deepseek_v3.py:122` |
| `d_moe` ← `shared_expert_intermediate_size` | Qwen3-Next의 shared expert MLP가 이 필드로 만들어진다. `modeling_qwen3_next.py:726` |
| `w_local` ← `attention_chunk_size` | Llama-4의 chunked local attention 창 크기. 48층 중 36층이 이 방식 |

**효과**: DeepSeek-V3의 이름 없는 정수가 **174 → 0**, tiny-deepseek-v3는 **72 → 0**이 됐다.
미등록 필드는 16종 → 4종, **24/26 모델이 0개**다.

**비차원은 리포트에서 제외한다.** 반복 횟수·스케일 상수·블록 개수는 텐서 축이 아니므로 이름이
없는 게 정상이다 — DeepSeek-V4 `hc_sinkhorn_iters`, Llama-4 `floor_scale`, Zamba2
`num_fwd_mem_blocks`(`for i in range(...)`로 LoRA 어댑터를 만드는 개수). 이걸 안 걸렀을 때
`floor_scale`(=8192)이 값이 같다는 이유로 `d_moe`로 태깅되는 사고가 실제로 났다.

> **새 심볼은 자기 group을 줘야 한다.** `n_grp`/`k_grp`를 `group: moe`에 넣었더니 그룹 라우팅을
> 쓰지 않는 평범한 MoE 모델(gpt-oss, OLMoE, Qwen3-30B, Llama-4)이 전부 "미확인 심볼 2개"로
> 잡혔다. group이 통째로 없으면 "해당 없음", 일부만 있으면 "미확인"이라는 §11 규약 때문이다.
> 고유 기능은 `moe_grouped` / `ssm_chunk` 처럼 별도 group + `group_key`로 등록한다.

**구조 차이 6종을 정리하고 각각 해결했다** (2026-08-05). "모델마다 구현이 달라 안 된다"를
26개 모델에서 실측해 유한한 목록으로 만들었다.

| # | 구조 차이 | 규모 | 해결 |
|---|---|---|---|
| ① | MoE expert가 raw `nn.Parameter` (3-D) | ~400 모듈 | 모듈 자신의 태그된 속성과 축을 대조 (`param_axis_expressions`) |
| ② | GPT-2 `Conv1D` — `in_features` 없음 | 192 모듈 | ①과 같은 코드. `nf`/`nx` 속성이 태그돼 있어 **gpt2-xl 축 100% 해결** |
| ③ | 한 모듈에 파라미터 여럿 | gpt-oss experts | 앵커를 **파라미터 단위**로 키잉 |
| ④ | config 필드 둘이 같은 객체 | DeepSeek-V4 | 첫 alias로 매칭되는 심볼 우선(`_claim_rank`) |
| ⑤ | 축 역할이 모듈 관점과 반대 | Zamba2 adapter | 심볼표로 검산 후 채택(`_evaluates_to`) |
| ⑥ | `int()`가 태그를 벗김 | Zamba2 | 검증된 derived 식으로 되찾기 (`d_inner = n_h_ssm·d_head_ssm`) |

①②는 **그 모듈이 생성될 때 쓴 폭을 스스로 속성으로 캐싱**하고 있다는 관찰에서 나온다.
`nn.Linear`가 아니어도 `num_experts`/`hidden_size`/`nf` 같은 속성은 태그를 유지하므로, 파라미터의
각 축을 **그 모듈의 속성하고만** 대조하면 된다. 후보가 서너 개뿐이라 전역 값 매칭보다 훨씬
좁고, **둘 이상이 설명하면 거부**한다 — gpt-oss는 `d_model == d_ff == 2880`이라 정확히 그 이유로
expert 축 두 개를 비워 둔다.

실측: 파라미터 축의 **74%가 이름을 얻는다** (gpt2-xl 100%, Llama-3.1-8B 100%, DeepSeek-V2-Lite
94%, gpt-oss 80%, Qwen3-30B 75%, SSM 계열 44~50%).

> **후보가 여럿일 때의 우선순위** — 스코프 밖 규칙을 먼저 제외하고(4096은 mamba 안에서
> `n_h*d_head`일 수 없다), 남은 것 중 **이름 없는 정수가 들어간 식은 항상 진다**. 이 두 번째
> 규칙이 없으면 반쪽만 태그된 `(4096+2*d_state)`가 더 짧다는 이유로 완전한
> `d_inner+2*n_g*d_state`를 이긴다. 계수(`2*d_state`의 2)는 리터럴이 아니다 — 같은 구분을
> `anchors.tag_is_usable`에서도 쓴다.

**결과: 미등록 config 필드가 26/26 모델에서 0개다.**

> **`param_axes`는 라벨링에 쓰지 않는다 — 연결을 시도했고 측정 후 껐다**(`_ENABLE_AXIS_LABELS`).
> 축을 모듈이 캐싱한 폭과 대조하는 것은 **숫자가 같다**는 말이지 **의미가 같다**는 말이 아니고,
> 유일성 검사는 "설명이 둘"만 막을 뿐 "설명이 하나인데 틀림"은 못 막는다. 전수 감사 결과
> derived 식까지 후보에 넣으면 3,365축이 바뀌는데 `d_moe → E*k`(x696), `d_moe → E`(x384)처럼
> expert의 **폭**이 expert **개수**가 되는 오류가 대부분이었고, 모듈 속성만 써도 830축이 남아
> 완전한 `d_inner+2*n_g*d_state`가 반쪽짜리 `(4096+2*d_state)`로 퇴행하는 사례까지 있었다.
> 켜려면 축의 **크기**가 아니라 **데이터플로우상의 역할**을 확인하는 장치가 선행되어야 한다.
>
> 그래도 산출물에는 계속 싣는다. 사람이 읽기에도, Tier 2 조사에도 좋은 근거이고,
> 이것 말고는 GPT-2의 `Conv1D`나 MoE의 3-D 파라미터 축을 설명하는 수단이 없다.

> **여전히 남은 한계** — 25개 모델 전수 대조에서 403 일치 / 70 불일치이고,
> 불일치가 여전히 **양방향**이다. 태그가 옳은 쪽: `n_h*d_head -> d_model`(잔차 스트림, x18),
> `n_h*d_head -> n_kv*d_head`(k/v 투영, x6). 태그가 못한 쪽은 대부분 **미등록 필드 때문**이다:
> `(n_h+2*n_kv)*d_head -> 192*n_h`(192 = 미등록 `qk_head_dim`),
> `2*d_inner+... -> 2*d_state+n_h_ssm+8192`. 즉 태그에 **정수 리터럴이 남아 있으면 그 자리는
> 규칙 구멍**이므로 앵커를 유지해야 한다 — 이것이 다음 단계의 채택 규칙 후보다.
>
> 한계 하나 더: GPT-2는 `nn.Linear`가 아니라 `Conv1D`를 써서 `in_features`가 없다.
> gpt2-xl은 모듈 식이 3개뿐이다. 태그 방식이 모든 아키텍처에 균등하게 통하지는 않는다.

## 7. 참조 구현 위치

핵심 로직 전체(`ScopeLabeler`, `OpGraphTracer` 등)는 `src/` 폴더의 실제 파일로 존재한다. 이 문서는 각 단계가 무엇을 해야 하는지의 스펙이고, `src/*.py`가 그 구현이다. 구현이 스펙과 어긋나면 스펙(이 문서)이 기준이다.

## 8. 신규 모델 추가

```yaml
# develop/models/<id>.yaml   (프로파일은 전부 develop/models/에 둔다)
model_id: <hf model id>
revision: null            # null이면 최신을 resolve해 hash로 고정
phases: [prefill, decode]
seq_len: auto             # introspection이 자동 산출
overrides:                # 필요한 경우에만
  attn_implementation: auto
extra_entrypoints: auto   # 메인 forward 밖 모듈 자동 탐색, 필요 시 수동 지정
```

프로파일은 `develop/models/`에 두고 출력은 `develop/out/`에서 검증한다. §9 검증(FAIL 0)을 통과한 **출력 폴더**만 최상위 `models/`로 승격한다(`develop/promote.py`) — `models/`는 믿을 수 있는 완성 산출물의 집이다. 관리 흐름은 `README.md` 참고.

## 9. 출력 검증

추출된 표·의존관계가 올바른지 자동으로 판정하고 리포트를 생성하는 절차(`src/validate.py`). `report.md`(사람용, 줄 단위 파싱 가능)를 낸다. 체크 결과는 `model_summary.md`의 검증 로그 표에도 함께 실린다(별도 `report.json`은 내지 않음).

### 9.1 사전 게이트 — Canary 검증
신규 모델(특히 크거나 복잡한 모델)에 투입하기 전에, 구조가 이미 알려진 소규모 참조 모델로 파이프라인 자체를 먼저 검증한다. 이 검증 과정 자체는 `develop/`에서 진행하고, 통과한 **출력 폴더**만 `models/`로 승격한다(프로파일은 `develop/models/`에 남는다).

- 최소 구성: dense 소형 모델 1개, GQA 소형 모델 1개, MoE 소형 모델 1개.
- 셋 다 아래 체크리스트를 FAIL 없이 통과해야 다음 단계로 진행한다. 여기서 FAIL이 나오면 대상 모델이 아니라 파이프라인 자체의 문제이므로 먼저 여기서 고친다.

### 9.2 체크리스트

| # | 항목 | 확인 방법 |
|---|------|-----------|
| C1 | layer 수 | distinct layer_idx 개수 == config의 layer 수 |
| C2 | layer 클러스터링 | 레이어별 op-시퀀스 서명을 클러스터링해, 레이어 스케줄이 균일하지 않다면 그 클러스터 수·배치가 config와 일치하는지 확인 |
| C3 | DAG 무결성 | depends_on 그래프에 순환 없음, 모든 비-source op은 depends_on ≥ 1, orphan 목록화 |
| C4 | 도달성 | 입력(embedding) → 출력(lm_head)까지 경로 존재 |
| C5 | 연결 불변식 | (a) matmul 입력·weight 차원 정합 (b) 각 블록의 입력/출력 hidden shape 동일(표준 residual과 비표준 믹싱 구조 모두 포괄) |
| C6 | hidden/head 정합 | attention shape의 hidden, head_dim, heads가 config와 일치 |
| C7 | GQA | q head 수 vs k/v head 수, repeat 배수 = heads/kv_heads |
| C8 | MoE | router 출력 dim = num_experts, top-k = experts_per_tok, expert FFN weight shape 일치. 라우팅된 토큰 수가 값 의존적이면 심볼릭 처리(WARN, FAIL 아님) |
| C9 | embedding/lm_head | weight shape = [vocab_size, hidden], tie 여부 명시 |
| C10 | 커버리지 강제 | param을 가진 모든 module이 표에 최소 1개 op으로 기여. 미달 시 FAIL, 목록화 |
| C11 | prefill/decode 정합 | decode에 cache 갱신 op 존재, 새 토큰 seq 차원 확인 |
| C12 | 교차검증 | torch.export 성공 시 op 개수·의존 구조가 주 캡처와 일치(허용 오차 내) |
| C13 | 재현성 | provenance에 revision hash, 버전, 입력 설정, 적용 조치 이력 존재. **동일 프로파일로 두 번 실행해 결과 해시가 같은지 실제로 확인**(원칙만이 아니라 실행으로 검증) |
| C14 | seq_len 적정성 | 사용 seq_len ≥ config의 모든 top-k/window/압축 관련 파라미터. 미달이면 FAIL |
| C15 | 진입점 완전성 | 발견된 추가 진입점(보조 모듈 등)이 모두 트레이싱됨. 누락 시 WARN + 목록 |
| C16 | 신규 op 표면화 | 매핑 안 된 op의 종류·개수·위치를 보고(INFO, FAIL 아님) |

### 9.3 리포트 형식 예시

```
# Extraction Report — <model_id> @ <revision hash>
phase: prefill | seq_len=<n> | backend=<meta|fake> | attn=<sdpa|eager>

C1  layer count ....... PASS (n == n)
C2  clustering ........ PASS
C5  connection ........ PASS
C8  MoE ............... WARN (routed token dim symbolic — see op_ids [...])
C10 coverage .......... PASS
C14 seq_len ........... PASS
C16 unmapped .......... INFO (n new op kinds — see list)
Overall: PASS (n WARN, n INFO)
```

WARN은 방법론상 정상인 경우(예: 값 의존적 라우팅의 심볼릭 처리)에 쓰고, config와 실제로 어긋나는 경우만 FAIL로 처리한다.

### 9.4 재현성 요건
`provenance.json`에 다음을 반드시 포함한다: model_id, revision(resolved hash), 라이브러리 버전, capture backend(meta/fake), 사용된 seq_len, attn_implementation, 적용된 조치 이력(Tier 0/1/2/3 여부와 근거), config 스냅샷.

## 10. 공통 심볼 (Symbol) 표기

모든 모델의 `structure.yaml`·`model_summary.md`·`src/validate.py`가 공통으로 쓰는 표기다. 정의는 `rules/symbols.yaml`에 있고, 모델 계열마다 다른 config 필드명을 심볼별 별칭 목록으로 흡수한다.

**런타임 축 심볼**: shape 표기에는 config 심볼(아래 표)에 더해 실행 파라미터를 나타내는 두 심볼을 쓴다 — `B`(batch, 여기선 1), `T`(트레이스에 쓴 seq_len).

> **`B`가 될 수 없는 자리** (게이트 `batch_excl`, 2026-08-05). 배치가 1이라 크기-1 축이 전부 `B`로 렌더되고 있었다. 맨 앞 축에는 맞지만 나머지에는 틀리며, 세 가지 형태로 나타났다:
>
> | 형태 | 근거 | 예 | 건수 |
> |---|---|---|---|
> | 한 shape에 `B` 2회 이상 | 텐서의 배치 축은 하나뿐 | RMSNorm 분산 축소 `[B,T,B]`, `unsqueeze` `[B,n_h,B]` | 200,055 |
> | `T` 뒤에 오는 `B` | HF 레이아웃은 배치가 시퀀스보다 앞(`[B,T,d]`) | 라우터 `sum(keepdim=True)` → `[T,B]` | 2,301 |
> | 가중치 축의 `B` | 파라미터는 load 시점에 config로 할당 — 배치 차원이 없음 | Qwen3-Next `nn.Linear(d_model,1)` 게이트 → `[B,d_model]` | 96 |
>
> 셋 다 **리터럴 `1`로 렌더한다.** 마지막 건은 `_propagate_labels`가 그 `B`를 matmul 출력까지 옮겨 나르고 있었다 — 가중치 라벨 하나가 활성 텐서를 오염시킨 사례다. `bare` 지표는 설계상 `1`을 세지 않아 이 오류를 전혀 볼 수 없었고, 그래서 규칙 게이트가 아니라 §13 ③ 자유 평가에서 발견됐다. 지금은 `batch_excl`로 게이트에 승격됐고 폴트 인젝션 2건으로 살아 있음이 확인된다. `T`는 심볼 해소가 유일해지도록 config의 어떤 차원값과도 겹치지 않는 값으로 자동 선택한다(겹치면 예: seq_len 16 == n_h 16처럼 한 숫자가 두 심볼로 해석될 수 있으므로 하나 키워서 회피, C14는 여전히 만족). shape의 각 축은 이 심볼들 또는 단순 식(`B*T, n_h*d_head, 2*d_head, T*k, T+1, d_head/2`)으로 렌더된다(§6).

| symbol | 의미 | 흔한 HF config 별칭 |
|---|---|---|
| L | 총 decoder layer 수 | `num_hidden_layers`, `n_layer`, `num_layers` |
| d_model | hidden size | `hidden_size`, `d_model`, `n_embd` |
| n_h | attention head 수 (Q) | `num_attention_heads`, `n_head` |
| n_kv | key/value head 수 (GQA) | `num_key_value_heads` |
| d_head | head당 차원 | `head_dim` |
| d_ff | dense FFN intermediate size | `intermediate_size` |
| V | vocab size | `vocab_size` |
| ctx | 최대 context length | `max_position_embeddings` |
| E | routed expert 총 수 | `num_experts`, `n_routed_experts`, `num_local_experts` |
| E_shared | shared expert 수 | `n_shared_experts`, `num_shared_experts` |
| k | 토큰당 활성 expert 수 | `num_experts_per_tok`, `moe_topk`, `top_k` |
| d_moe | expert FFN intermediate size | `moe_intermediate_size` |
| w_local | sliding window 크기(있는 경우) | `sliding_window` |
| layer_sched | 레이어별 유형 스케줄(있는 경우) | `layer_types` |

별칭으로 못 찾은 심볼은 `null`로 남긴다(임의 채움 금지, P1). 값이 `null`인 심볼이 있으면 그 자체가 `02-new-module-handling.md` Tier 2 리서치 대상이다 — 확인되면 `rules/symbols.yaml`에 별칭을 추가한다. MLA류 KV 압축 latent 차원처럼 모델마다 이름이 크게 갈리는 것들은 아직 공통 심볼에 없다 — Tier 2로 확인되는 대로 추가해나간다.

### 10.1 라벨 출처 기록 (label_provenance)

축에 이름이 붙었다는 것과 그 이름이 **근거 있게** 붙었다는 것은 다르다. 2026-08-05 이전에는 이 둘이 산출물에서 구분되지 않았다 — 스코프 규칙이 답을 준 축과, 등록된 규칙이 없어 산술적으로 맞는 이름을 지어낸 축이 표에서 똑같이 보였다. `symbolic_shape.dim()`의 모든 `return`이 이제 **어느 분기가 답했는지**를 집계하고(`resolver.stats`), `structure.yaml.label_provenance`와 `model_summary.md`에 공개된다.

| 근거 | 신뢰도 | 뜻 |
|---|---|---|
| `scoped_symbol` / `scoped_formula` | 강함 | 이 모듈을 스코프로 갖는 심볼·유도식이 값을 정확히 설명 |
| `plain_symbol` / `derived_formula` | 강함 | 스코프 없는 심볼, `derived_dims.yaml`의 검증된 식 |
| `runtime` | 강함 | `B`/`T`/`1` |
| `reused_symbol` / `out_of_scope_symbol` | 약함 | 같은 shape에서 이미 쓴 이름의 재사용, 스코프가 **배제한** 심볼 |
| `heur_*` | **지어낸 이름** | 등록된 규칙 없이 곱·배수·절반·+1로 맞춘 것 |
| `bare` | 이름 없음 | 정수로 남김(정직한 실패) |

`heur_*`는 이번 트레이스의 seq_len에서만 참일 수 있다. 함대 평균 **3.17%**(gpt2-xl 9.03%가 최대, GPT-2가 `Conv1D`라 모듈 선언 폭이 없는 것과 같은 원인). 게이트가 `heur`를 베이스라인 지표로 추적하므로 "유도"를 "추측"으로 바꾸는 변경은 조용히 통과하지 못한다. `model_summary.md`는 지어낸 이름이 몰린 모듈을 함께 출력해 Tier 2 조사 대상을 바로 지목한다.

> **알려진 결함(WARN `ident_incons`)**: DeepSeek-V4-Pro는 T=2048에서 `T/m_csa`(=512)가 `d_head`(=512)와 겹쳐 `compressor.kv_norm`의 축 순서가 뒤집혔다(30건). 같은 모듈을 T=1032로 트레이스한 **DeepSeek-V4-Flash에서는 258 ≠ 512**라 모호성이 없고 참인 배치가 `[B, T/m_csa, d_head]`임을 보여준다. 근본 해결은 `resolve_seq_len()`이 심볼뿐 아니라 **스코프 유도식까지** 비충돌 조건에 넣고 재트레이스하는 것이다(§6, `t-fixed-at-trace-time`과 같은 경로). 복사 op가 축 라벨을 바꾸는 것은 물리적으로 불가능하므로 게이트가 이 부류를 상시 감시한다.

## 11. 구조 요약 (structure.yaml) · 모델 요약 (model_summary.md)

### 11.1 structure.yaml
표(§6)를 layer 단위로 롤업한 것. op-타입 시퀀스가 동일한 연속 레이어는 하나의 range로 묶는다(예: "layer 1-26: MLA, MoE"). 형식:

```yaml
model_id: <hf model id>
revision: <resolved hash>
symbols:
  L: 27
  d_model: 2048
  n_h: 16
  n_kv: 16
  # ... (§10 표 순서대로, 못 찾은 건 null)
layers:
  - range: [0, 0]
    blocks: [self_attn, mlp]        # 이 레이어에서 관측된 상위 block 종류
  - range: [1, 26]
    blocks: [self_attn, moe]
literal_dims:                       # 심볼로 안 바뀌고 정수로 남은 고정 차원(있을 때만)
  - value: 32768
    expr: n_h·(d_nope+d_v)          # config에서 계산한 알려진 합성식과 값이 정확히 일치할 때만 채움
    where: [kv_b_proj, self_attn]   # 나타나는 모듈(미상이면 expr=null, 유래 추적용)
note: "symbols의 null은 임의 채움이 아니라 미확인 표시(P1) — Tier 2로 확인 후 별칭 추가"
```

`literal_dims`는 심볼라이저가 이름을 못(안) 붙여 **정수로 남긴 고정 차원**을 모아둔 것이다(MLA `kv_b_proj`의 32768 등 — §6 참고). 이 값들은 config에서 유도된 **입력 불변 상수**라 배치·시퀀스와 무관하게 항상 같다. "이 숫자 뭐지?"를 방지하려고, 값이 config에서 계산한 알려진 합성식(`n_h·(d_nope+d_v)` 등)과 정확히 일치하면 `expr`에 유래를 적고(우연 분해가 아니라 값 대조), 아니면 `where`의 모듈로 출처를 남긴다. 작은 잡음 값은 유래가 확인된 것만 싣는다.

### 11.2 model_summary.md
모델을 처음 보는 사람이 읽을 요약 문서. 아키텍처 사실과 검증 로그는 config+트레이스에서 **결정적으로** 채워지고(코드만으로 충분), **추가 교차검증 "참고 소스"만 Tier 2 리서치 결과**로 코딩 에이전트가 `02-new-module-handling.md` Tier 2 절차로 조사해 보탠다(없어도 shape·dependency는 1차 소스로 확정됨).

포함 항목:
- 기본 정보(revision, capture backend, 트레이스 seq_len(T), attn_implementation, torch/transformers 버전)
- **요약 정보**(9개 필드 카드) — (1) SCALE(총·활성 파라미터+활성%), (2) Context(tokens), (3) DATE(HF repo 생성일 — 대략적 출시 시점, 정확한 발표일과 다를 수 있음), (4) DECODER TYPE(Sparse MoE / Dense — config 필드가 아니라 실제 트레이스의 expert op/param 유무로 판정, vestigial 필드 배제), (5) Attention(MHA/GQA/MQA/MLA/attention-free), (6) LAYER MIX(레이어별 유형 카운트 + dense/MoE 분할), (7) KV CACHE / TOKEN(BF16 2바이트, **attention 층만** 카운트 — 하이브리드의 Mamba/DeltaNet/mlp 층은 제외, MLA는 압축 latent), (8) KEY DETAIL(도출 사실 기반 자동 요약), (9) Related concepts. 전부 config/트레이스에서 결정적 도출(P1), (3)만 HF 메타데이터, (8)의 편집 세부는 Tier 2. 카드 필드 구성은 [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/) 카드 형식을 `ref)`로만 참고(브랜딩성 소제목은 쓰지 않는다).
- **아키텍처 특성**(정성 요약) — 해석성 사실만 담고 **수치는 담지 않는다**(아래 차원·심볼 표와 중복·상충 방지): 모델 타입, attention 계열(MHA/GQA/MQA/MLA + head 수·repeat, **MLA는 단일 head_dim이 오해를 부르므로** q/k=nope+rope, v를 풀어 쓴다; 비-MLA는 d_head 병기), attention 커널(sdpa/eager), 위치 인코딩(RoPE θ/scaling·learned·NoPE), FFN(dense d_ff / MoE E·top-k·shared·활성화), 정규화(RMSNorm/LayerNorm — op으로 판별, 모듈 이름 아님), tie embeddings, **decode 방식·KV cache 크기**(2·n_kv·d_head/token/layer, MLA면 압축 latent). 값은 전부 config/실행 결과에서만 나온다(P1).
- **유도 상수**(`literal_dims`가 있을 때만) — 심볼로 안 바뀌고 정수로 남은 고정 차원을 값·유래(`expr`)·출처 모듈로 정리한 표. 독자가 shape의 "낯선 숫자"(예: MLA `kv_b_proj`의 32768=`n_h·(d_nope+d_v)`)를 버그가 아니라 입력 불변 고정 차원으로 이해하게 한다.
- **차원·심볼**(§10 심볼 표) — **모든 수치(L·d_model·head dim·V·ctx·E·d_ff·MLA 압축차원 …)의 단일 출처**. 파라미터 수는 (1) SCALE 카드가, 여타 치수는 이 표가 담아 특성 표와 겹치지 않게 한다. 균일 layer_sched 등 긴 리스트는 축약 표기. Context(ctx)는 config `max_position_embeddings`이며 RoPE 스케일(YaRN 등)이 있으면 원본·factor를 주석으로 달아 벤더 광고 컨텍스트(예: DeepSeek-V3 163,840 vs 광고 128K)와의 차이를 설명한다(값을 지어내지 않음, P1).
- 레이어 구조(§11.1 롤업)
- **검증 로그** — §9 체크리스트(C1~C16) 결과 표 + 종합 판정 + 재현성(C13) 상태. "잘 검증됐다"를 요약 문서 자체에서 보이게 한다.
- 추출 방법 요약(이 문서 Step 1~8 참조, "값은 실행 결과에서만 나온다"는 원칙 재확인)
- **구성 근거/소스** — shape·dependency의 **1차 출처를 명시**한다: (1) HF config.json @ revision hash(+config_sha256), (2) 실행한 transformers 공식 modeling forward(버전 명시), (3) dispatch 트레이스(seq_len). 그다음 **교차검증(Tier 2)** 표 — 아래 카테고리를 전부 검토 대상으로 삼는다(라벨·해석용, shape 값 출처 아님):

| 카테고리 | 무엇을 확인하는가 |
|---|---|
| Hugging Face (config docstring, model card) | 파라미터 의미, 아키텍처 설명 |
| 독립 서빙 구현 — vLLM / SGLang / TensorRT-LLM 소스 | 같은 모델을 별도로 재구현한 코드라서, HF 구현이 모호하거나 새 모델이라 설명이 부족할 때 op 구조·이름 관례를 교차검증하는 데 특히 유용 |
| 라이브러리 공식 문서(transformers model_doc 등) | 아키텍처 설명 |
| 논문/기술 리포트 | 설계 의도, 수식 |
| [Sebastian Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/) | 여러 모델의 아키텍처를 한곳에 정리한 2차 자료. HF와 나란히 확인하면 attention/잔차 변형의 명칭·계보를 빠르게 파악하는 데 유용하다(원 소스로 반드시 재확인) |
| 공개 벤치마크 자료 | 활성 파라미터 수·처리량 등 참고 수치와의 정합성(참고용, shape 값의 출처 아님) |

**중요**: 이 표는 라벨링·해석 교차검증용이다. `structure.yaml`의 shape·dependency에 해당하는 어떤 값도 이 소스들로 덮어쓰지 않는다(P1, `02-new-module-handling.md` Tier 2 규칙과 동일).

## 12. 구조 라이브러리 (rules/structures/)

attention 방식·잔차 연결 방식뿐 아니라 MoE/FFN, 정규화, 위치 인코딩, 보조 모듈까지 — 구조적 패턴(MHA, GQA, MLA, CSA, HCA, mHC, MTP, MoE 변형 등)을 모델 하나에 묶어두지 않고, **패턴 단위로 별도 문서화**해 계속 쌓아가는 곳. Hugging Face의 attention 구현 라이브러리처럼, "이 패턴이 뭐고 어떤 모델들이 썼는지"를 한곳에서 찾을 수 있게 한다.

- 위치: `rules/structures/<category>/<name>.md` (예: `rules/structures/attention/mla.md`). 카테고리는 `attention`, `moe`, `normalization`, `position_encoding`, `residual`, `auxiliary`로 시작하되, 새로운 축(캐시 압축 방식 등)이 필요해지면 늘어난다.
- **지금은 각 카테고리에서 가장 기본적인/이미 잘 알려진 패턴만 미리 채워뒀다**(attention의 MHA·GQA, MoE의 dense·기본형 top-k, 정규화의 RMSNorm, 위치 인코딩의 표준 RoPE, 잔차 연결의 표준 add). 그보다 복잡하거나 모델 고유의 패턴(MLA, CSA, HCA, mHC, MTP, shared+routed MoE 등)은 의도적으로 비워뒀다 — `develop/`에서 새 모델을 처리하다가 실제로 만날 때(`02-new-module-handling.md` Tier 2/3으로 확인한 뒤) 그 시점에 채운다. 미리 답을 적어두면 시행착오 자체의 훈련 효과가 없어지고, 이 중 일부는 예약된 최종 테스트 모델과 같은 계열이라 미리 채우면 최종 테스트 의미도 옅어진다.
- 각 항목의 형식: 정의, 관련 심볼(§10), 트레이스에서 식별하는 방법(탐지 패턴), 확인된 모델 목록(계속 추가), 참고 소스.
- 소스는 `02-new-module-handling.md` Tier 2 순서를 따르되, Hugging Face와 [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/)를 나란히 확인하는 걸 기본으로 한다 — 후자는 여러 모델의 아키텍처가 이미 비교 정리되어 있어 계보 파악이 빠르다.
