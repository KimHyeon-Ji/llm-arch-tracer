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
| `<model>/<phase>.csv` | **주요 operator 표** — op_id, h1·h2·…(계층), op_type, input_shape, weight_shape, output_shape, depends_on, layer_idx, block, sub_block, depth, module_path, raw_op, params, phase, unmapped |
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
| depends_on | 입력을 만든 선행 op_id | 텐서 신원 추적 |
| layer_idx / block / sub_block | decoder layer 번호 / 상위·하위 블록 | scope |
| depth / h1 · h2 · … | 모듈 중첩 깊이 / 레벨별 계층 컬럼 | scope (module_path 분해) |
| module_path / raw_op / params | 추적성용 원본 정보 | 트레이서 |
| phase | prefill / decode | 실행 설정 |
| unmapped | 정규화 매핑 실패 여부 | 정규화 |

- **shape은 숫자가 아니라 심볼로 낸다**(csv·trace.raw.jsonl 공통). 구체적 batch(1)·seq_len은 우리가 임의로 고른 실행 파라미터라, 그대로 숫자로 박으면 산출물이 특정 실행에 묶인다. 대신 각 축을 아키텍처 심볼(§10: `B, T, d_model, n_h, d_head, E, k, …`)이나 단순 식(`B*T, n_h*d_head, 2*d_head, T*k, T+1, d_head/2`)으로 렌더한다. 어떤 심볼/식에도 안 맞는 순수 구조 상수만 정수로 남긴다(지어내지 않음, P1). 구체 숫자는 `provenance.json`의 `seq_len_used`+`symbol_table`로 완전 복원 가능하다(추적성 유지). 심볼 해소가 유일하도록 트레이스 seq_len(T)은 config의 어떤 차원값과도 겹치지 않게 자동 선택한다(§10 참고).
- **계층(hierarchy)을 컬럼으로 편다.** op을 순서대로만 나열하지 않고, `module_path`를 레이어 아래 모듈 중첩(예: `self_attn → q_proj`, `mlp → experts → act_fn`)으로 분해해 `depth`와 `h1, h2, …` 레벨 컬럼으로 낸다. 이러면 어느 수준으로든 그룹핑/롤업이 되고, HF 모듈이 겹겹이 중첩된 구조가 표에서 보인다.
- **컬럼 물리 순서**(모든 파일 공통): `op_id` → `h1, h2, …`(모델별 최대 깊이만큼) → `op_type` → `input_shape` → `weight_shape` → `output_shape` → `depends_on` → `layer_idx` → `block` → `sub_block` → `depth` → `module_path` → `raw_op` → `params` → `phase` → `unmapped`. 계층 컬럼을 op_id 바로 뒤 앞쪽에 두어 각 행을 "구조 먼저(op_id + 모듈 트리 위치)"로 읽게 한다.
- **shape·list 필드 직렬화**: 중첩 리스트인 shape/`depends_on`/`params`는, **CSV에선 읽기용으로 심볼을 따옴표 없이 bare**로 낸다(예: `[[V, d_model], [B, T]]`) — dim 심볼에 쉼표가 없어 모호하지 않다. **`.trace.raw.jsonl`·`.jsonl`은 순수 JSON**(`[["V","d_model"], …]`)이라 프로그램 파싱은 이쪽을 쓴다. 같은 데이터의 표기 차이일 뿐이다.

### 6.1 주요 operator 표 (`<phase>.csv`/`.jsonl`) — latency 관점 파생

전체 표(`full/`)는 모든 aten 프리미티브를 담아 view/transpose 같은 레이아웃 op까지 보이지만, inference latency는 `max(FLOPs/연산처리율, 이동바이트/대역폭)`로 결정되므로 **FLOPs가 크거나 큰 텐서를 읽고/쓰는 op만** latency에 의미가 있다. 최상위 주요 operator 표는 전체 표에서 아래 기준으로 파생한다(`src/major_ops.py`). 컬럼은 전체 표와 동일하고, `op_id`는 0부터 다시 매기며, `depends_on`은 잘려나간 op을 관통해 가장 가까운 살아남은 선행 op으로 재연결(그래프 축약)한다 — 축소된 표 안에서도 유효한 DAG가 된다.

- **위치 인코딩 precompute 모듈 통째로 제거**: `module_path`에 `rotary`/`rope`가 들어간 모듈(cos/sin 사전계산)은 forward당 1회 계산 후 전 레이어에 브로드캐스트되는 상수라 per-layer 비용에서 무시 가능 → 전부 제거. (RoPE *적용*은 attention 안의 elementwise로 남고, 아래 크기 게이트로 걸러진다.)
- **정규화 모듈은 1행으로 롤업**: leaf 이름에 `norm`이 있거나 GPT-2 `ln_1/ln_2/ln_f`인 모듈의 op들(RMSNorm은 `pow→mean→add→rsqrt→mul`로 분해됨)을 하나의 `rmsnorm`/`layernorm` 행으로 합친다(`rsqrt` 있으면 rmsnorm). input은 그룹 진입 op의 입력, output은 마지막 op의 출력, weight_shape는 `*.weight`를 소비한 op의 1-D 피연산자.
- **항상 유지**(연산/attention/활성 코어): `linear, matmul, batched_matmul, grouped_matmul, sdpa, conv1d, embedding, softmax, silu, gelu, relu, sigmoid, tanh, exp, layernorm, rmsnorm`.
- **크기 게이트 후 유지**: `elementwise_add, elementwise_mul, concat, sum` — 피연산자/출력의 마지막 축이 **wide 심볼**(`d_model`/`d_ff`/`d_moe`)일 때만. residual add·GLU gating·MoE combine은 남고, 작은 RoPE 적용 mul·rotate_half/KV-append concat·attention mask add(전부 `d_head`·`T` 스케일)는 빠진다.
- **그 외 전부 제거**: view/transpose/expand/slice/select/clone/copy/cast, RoPE 삼각함수, MoE 라우팅 plumbing(topk/sort/gather/scatter/cumsum/where/…).

**반복 레이어 접기(repeat folding).** 위 필터 뒤, 구조가 같은 decoder 레이어를 대표 블록 하나로 접는다(`major_ops.collapse_repeats`). full 표는 전 레이어를 그대로 펼쳐 두지만, 주요 표는 기본 블록 단위만 깔끔하게 보이게 한다.
- **서명(signature)** = 레이어의 **full 트레이스 op 시퀀스**에 대한 `(op_type, 레이어번호 뗀 module_path, 심볼 input/weight/output shape)` 튜플(`major_ops.full_layer_signatures`). 그룹핑은 레이어 **전체 구조** 기준이라 — (a) 블록은 레이어 단위 통째로 묶고(공통 부분을 빼내지 않음: DeepSeek의 dense·MoE 레이어는 MLA attention이 완전히 같아도 MLP가 달라 별도 블록으로 남는다), (b) major에서 버리는 op만 다른 레이어도 별개로 센다(NoPE 레이어 vs RoPE 레이어는 rotary 적용 op만 달라도 별도 블록). 결과적으로 블록 수 = full 트레이스의 실제 per-layer 아키텍처 종류 수와 일치한다. 서명이 같은 레이어는 첫 등장 레이어만 남긴다. 이질적 레이어(DeepSeek dense→MoE, Nemotron Mamba/MLP/attention 교대)는 자동으로 별도 블록이 된다.
- 각 행에 **`block_type`**(블록 구성 라벨: `attn+FFN`, `MLA+MoE`, `SSM`, `xLSTM` 등 — mixer부 + FFN/MoE부를 op 구성에서 도출), **`repeat`**(그 블록이 대표하는 레이어 수), **`layers`**(해당 레이어 인덱스, `0-2`·`1,3,5` 식 압축)를 붙인다 — 컬럼 순서는 `op_id` 바로 뒤(`op_id → block_type → repeat → layers → h1 …`)라 어떤 종류의 블록이 몇 번 반복되는지 한눈에 보인다. 레이어에 안 속한 1회성 op은 `block_type`이 `embed`/`norm`/`head`(또는 미분류 `-`), `repeat=1`, `layers`=빈칸.
- 잘려나간 반복 레이어를 가리키던 `depends_on`은 위치 매핑으로 대표 블록의 대응 op에 재연결한다 → embed → block(×N) → norm → lm_head 형태의 유효한 DAG가 유지된다. latency는 블록 비용 × `repeat` + 1회성 op으로 계산하면 된다.

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

**런타임 축 심볼**: shape 표기에는 config 심볼(아래 표)에 더해 실행 파라미터를 나타내는 두 심볼을 쓴다 — `B`(batch, 여기선 1), `T`(트레이스에 쓴 seq_len). `T`는 심볼 해소가 유일해지도록 config의 어떤 차원값과도 겹치지 않는 값으로 자동 선택한다(겹치면 예: seq_len 16 == n_h 16처럼 한 숫자가 두 심볼로 해석될 수 있으므로 하나 키워서 회피, C14는 여전히 만족). shape의 각 축은 이 심볼들 또는 단순 식(`B*T, n_h*d_head, 2*d_head, T*k, T+1, d_head/2`)으로 렌더된다(§6).

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
