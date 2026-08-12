# Mamba / SSM (선택적 상태공간 시퀀스 믹서) — 새 축(sequence mixing)

attention도 MoE도 아닌 **상태공간 모델(SSM)** 계열 시퀀스 믹서. `01-main.md` §12가 허용한
새 카테고리(`ssm/`)로 추가한다. hybrid 모델에서 attention·FFN 층과 섞여 나온다.

## 정의
토큰을 순차/청크 스캔하며 고정 크기 **상태(state)** 를 갱신하는 selective-scan 믹서.
Mamba-2는 SSD(state-space duality)로 청크 단위 병렬 스캔을 한다. KV cache 대신 **conv state +
ssm state**를 굴리므로 표준 attention의 `2·n_kv·d_head` KV cache 개념이 없다.

## 관련 심볼 / config / 파라미터
config: `mamba_num_heads`, `mamba_d_state`, `hybrid_override_pattern`(M=mamba/`*`=attention/
`-`=mlp 스케줄), `layers_block_type`(레이어별 유형 리스트). 모듈은 `mixer`.
파라미터(트레이스에서 관측): `A_log`(상태 감쇠 로그), `D`(skip 연결), `dt_bias`(시간 스텝),
`conv1d.weight/bias`(short causal conv), `in_proj`/`down_proj`(입출력 투영). 이들 전용 심볼은
아직 공통 심볼표에 없음 — 필요 시 Tier 2로 추가.

## 트레이스에서 식별하는 방법 (Phase 17 Nemotron-3-Nano 실측)
- 블록명 `mixer`(attention의 `self_attn` 아님). softmax/`sdpa` **없음**.
- causal short conv: `constant_pad_nd` → `convolution`(op_type `conv1d`).
- selective scan: `exp`(상태 감쇠 exp(A·dt)), `cumsum`(누적), `tril`+`masked_fill`(청크 스캔
  마스킹), `elementwise_mul/add`+`sum`(상태 갱신·집계), 다수의 `slice`/`view`/`permute`.
- `dt_bias`/`A_log`/`D` param을 쓰는 op이 있으면 Mamba 층으로 확정.
- **fast-path 주의**: `selective_state_update`/`causal_conv1d_fn`이 미설치면 transformers가 naive
  torch 폴백으로 실행 → meta 트레이싱 가능(로그: "fast path is not available ... naive
  implementation"). 폴백이 없고 CUDA 커널만 있으면 `02-new-module-handling.md`의 구조적 한계.
- **C10(커버리지)이 핵심 검증**: Mamba 전용 param(A_log/D/dt_bias/conv1d/…)이 전부 표에
  기여하는지 확인(Phase 17: 263 params 전부 커버 PASS). **C11/KV-cache 가정은 SSM엔 부적합** —
  cache는 conv/ssm state이며 표준 KV가 아니다.

## 확인된 모델
- **`nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`** (Phase 17, "nemotron_h" arch): 42 layers =
  Mamba(linear_attention) 21 + full_attention 4(GQA 40:8) + mlp 17. C2 PASS(3 clusters ==
  `layers_block_type` 3종), C5 42/42층, C10 263 params 전부. dense(config `n_routed_experts=8`은
  vestigial — 실제 expert 없음 → C8 WARN). C13 repro.
  ※ transformers cache dispatch가 'mlp' 층 유형에서 KeyError → loader.py에서 'mlp'를 no-op
  cache 층으로 등록하는 compat shim으로 해결.

## 참고 소스
- transformers `models/nemotron_h` 구현(naive 폴백) — 트레이스로 직접 관측
- Mamba / Mamba-2(SSD) 논문, Raschka's LLM Architecture Gallery(SSM 하이브리드 계보; 교차검증용)

- **`ibm-granite/granite-4.0-h-small`** (Phase 33): Mamba2 + MoE 하이브리드. `n_h_ssm`=128,
  `d_head_ssm`=64, `d_state`=128, `n_g_ssm`=1, `d_conv`=4, `d_chunk`=256.
  config 필드명이 `mamba_n_heads` / `mamba_d_head` 라 별칭 표에 없었고, 그 탓에
  `d_inner`(=n_h_ssm·d_head_ssm=8192)가 안 풀려 그 위에 얹힌 conv_dim(8448)·
  projection_size(16768)까지 통째로 미해결 상수로 남아 있었다 — **규칙은 다 있었는데 입구가
  막혀 있던 경우다.** 별칭 두 개를 추가하니 셋 다 닫혔다.
  출처: `modeling_granitemoehybrid.py:513,525,534`.
