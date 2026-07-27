# 기본형 MoE (token-choice top-k 라우팅, shared expert 없음)

## 정의
토큰마다 라우터(linear)가 전체 expert에 대한 점수를 내고, 그중 top-k개 expert로만
토큰을 보내는 가장 단순한 MoE 형태. shared expert(모든 토큰이 공통으로 거치는
expert)가 없는 순수 routed-only 구조.

## 관련 심볼 (rules/symbols.yaml)
`E`(routed expert 총 수), `k`(토큰당 활성 expert 수), `d_moe`(expert FFN intermediate
size). `E_shared`는 이 기본형에서는 0/해당 없음.

## 트레이스에서 식별하는 방법
- block 안에 router `linear`(출력 last-dim = `E`) → `softmax` → `topk`(출력에 `k` 차원) 로
  expert 선택. (`src/validate.py` C8이 이 셋 — router dim=E, top-k, expert weight — 을
  트레이스에서 직접 확인한다.)
- 선택 후 토큰을 expert별로 묶는 방식은 구현마다 다르다:
  - **grouped 방식(OLMoE 확인)**: `sort`/`floor_divide`/`histc`/`cumsum`/`index`로 토큰을
    expert 순으로 정렬·그룹핑한 뒤, `_grouped_mm`(op_type `grouped_matmul`)으로 `E`개
    expert FFN을 한 번에 계산. weight_shape가 `[E, ...]` 형태(예: gate_up `[E, d_model, *]`,
    down `[E, *, d_ff]`). SwiGLU면 gate_up `grouped_matmul` → `split` → `silu` → `mul` →
    down `grouped_matmul` → `index_put`/`sum`으로 되돌림.
  - **per-expert 방식**: expert마다 별도 `nn.Linear` 모듈(`...experts.<i>.gate_proj` 등)로
    나타나 `gather`/`scatter_add` 계열로 합침. (아직 이 라이브러리에서 직접 확인 전.)
- shared expert에 해당하는 별도 FFN 경로가 없음(있으면 "shared+routed" 변형 —
  develop/에서 만나면 추가: Phase 6/15 예정)
- 라우팅된 토큰 수 자체는 값 의존적(`k*T` 심볼로 렌더됨)이라 `01-main.md` C8이 이를
  심볼릭 처리로 다룬다(WARN이 정상, FAIL 아님)

## 확인된 모델 (계속 추가)
- **`allenai/OLMoE-1B-7B-0924`** (Phase 4): 16 layers, `E`=64, `k`=8, shared expert 없음,
  grouped 방식(`_grouped_mm`) SwiGLU expert. C8 트레이스 검증 통과(router_dim/top_k/
  expert_weight=grouped), C10 전 param 커버. revision `6d84c48581ece794365f2b8e9cfb043c68ade9c5`.
- **`Qwen/Qwen3-30B-A3B`** (Phase 14): 48 layers, GQA 32:4, `E`=128, `k`=8, shared expert 없음,
  grouped SwiGLU. OLMoE와 다른 회사·다른 E(128 vs 64)로 "shared 없는 MoE" 일반화 재확인.
  C8 트레이스 검증(E=128 top-8 grouped), C10 531 params 전부, C13 repro.
- **`openai/gpt-oss-20b` / `-120b`** (예약 최종테스트, 2026-07-23): 24층 `E`=32 / 36층 `E`=128,
  둘 다 `k`=4, shared expert 없음, expert 폭 = `intermediate_size`(2880) — **`moe_intermediate_size`
  필드가 따로 없다**(순수 MoE 스택이라 dense FFN이 없음). SwiGLU에 `swiglu_limit`=7.0로
  게이트/업 pre-activation을 클리핑하는 점이 특이. 매 attention 층이 sliding/full 교대이고
  [../attention/attention-sink.md](../attention/attention-sink.md)를 함께 쓴다.
  MoE grouped-matmul 커널이 BF16을 요구해 Tier-1 `use_bf16` 조치가 필요했다.
- **`meta-llama/Llama-4-Maverick-17B-128E`** (예약 최종테스트, 2026-07-27): 48층,
  `E`=128, **`k`=1 (top-1 라우팅)** + shared expert 1개, `interleave_moe_layer_step`=2라
  **짝수층 dense FFN / 홀수층 MoE** 교대. expert 폭 `intermediate_size`=8192, dense/shared 쪽은
  `intermediate_size_mlp`=16384로 **따로**다. top-1은 지금까지 본 MoE 중 가장 희소한 라우팅
  (활성 17B / 총 400B). NoPE 인터리브와 조합되는 부분은
  [../position_encoding/nope.md](../position_encoding/nope.md) 참고.

## 참고 소스
- 모델 config docstring (`num_experts`/`n_routed_experts`, `num_experts_per_tok` 등)
- HF `transformers` OLMoE modeling 소스(`_grouped_mm` 기반 expert 구현) — 트레이스로 직접 관측
