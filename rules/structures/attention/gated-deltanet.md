# Gated DeltaNet (선형 attention) + Gated full-attention 하이브리드

## 정의
softmax attention 대신 **선형 attention 계열(Gated DeltaNet)**을 쓰는 레이어와, 표준
**gated full-attention** 레이어를 섞은 하이브리드. DeltaNet은 KV를 O(N) 상태로 압축하는
순환/청크 스캔(delta rule)으로, KV cache 대신 고정 크기 상태 행렬을 굴린다. Qwen3-Next 계열.

## 관련 심볼 / config
`layer_types`에 `linear_attention`(DeltaNet)과 `full_attention`(gated attention)이 섞여 있고
(Qwen3-Next는 3:1 = DeltaNet 3 : full 1), 그 자체가 이종 스케줄이라 **C2가 2 클러스터로
분리**한다(C2 PASS: 2==2 from layer_types). GQA는 full_attention 층에 적용(16:2 등).

## 트레이스에서 식별하는 방법 (Phase 16 Qwen3-Next-80B 실측)
- **DeltaNet 층**(`self_attn` 아님, 모듈명 `linear_attn`): softmax/`sdpa` op이 **없다**.
  대신:
  - `constant_pad_nd` + `convolution`(op_type `conv1d`) — causal short conv(q/k/v 전처리).
  - `silu`, `exp`, `sigmoid` — 게이팅/감쇠 계수(gated delta rule).
  - `slice`/`select`/`copy_`/`clone` 다수 + `sum` — 청크/순차 스캔의 상태 재귀가 언롤되어
    잡힌다(한 층에 op이 수백 개; 이 스캔 내부 op은 라벨이 모호해 `unmapped`로 둔다, P8).
  - `matmul`/`batched_matmul` — 상태·쿼리 상호작용.
  - **주의: 이 계열은 표준 `2·n_kv·d_head` KV cache가 아니다**(고정 상태 행렬). model_summary의
    KV cache 표기는 full_attention 층 기준이며, DeltaNet 층은 상태 기반임을 별도로 이해할 것.
- **full_attention 층**: 표준 gated attention(softmax/`sdpa`, GQA). "gated"는 attention
  출력에 게이트(silu·mul)를 거는 변형.
- **커널 주의**: fast-path(flash-linear-attention / causal-conv1d)가 설치 안 되면 transformers가
  **순수 torch 폴백**으로 실행 → meta 트레이싱이 가능해진다(로그: "fast path is not available
  ... Falling back to torch implementation"). fast-path 커널만 있고 eager 대체가 없으면
  `02-new-module-handling.md`의 "구조적 한계"에 해당하겠지만, 여기선 폴백이 있어 문제없음.

## 확인된 모델
- **`Qwen/Qwen3-Next-80B-A3B-Instruct`** (Phase 16): 48 layers = 36 DeltaNet + 12 gated full-attn
  (3:1), GQA 16:2, MoE `E`=512 top-10 + shared expert. C1 48==48, C2 PASS 2==2, C5 48/48층
  (DeltaNet 층도 잔차 d_model=2048 유지), C8 E=512, C10 759 params 전부 커버, C13 repro.
  revision `9c7f2fbe84465e40164a94cc16cd30b6999b0cc7`.

## 참고 소스
- transformers `models/qwen3_next` 구현(torch 폴백 경로) — 트레이스로 직접 관측
- Gated DeltaNet / DeltaNet 논문, Raschka's LLM Architecture Gallery(선형 attention 계보; 교차검증용)
- flash-linear-attention 저장소(fast-path 구현 참고)
