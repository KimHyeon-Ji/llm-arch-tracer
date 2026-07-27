# MLA — Multi-head Latent Attention (KV 압축 + decoupled RoPE)

## 정의
KV(및 큰 모델은 Q)를 저차원 **latent**로 압축(down-proj)했다가 head별로 복원(up-proj)해
KV cache를 크게 줄이는 attention. RoPE는 전체 head 차원이 아니라 별도의 소수 차원
(**decoupled RoPE**)에만 걸고, 나머지 non-RoPE 차원과 concat해서 attention을 수행한다.
DeepSeek 계열(V2/V3)의 핵심 attention.

## 관련 심볼 (rules/symbols.yaml)
`c_kv`(KV 압축 latent = `kv_lora_rank`), `c_q`(Q 압축 latent = `q_lora_rank`, 작은 모델은
없음), `d_rope`(decoupled RoPE head 차원 = `qk_rope_head_dim`), `d_nope`(non-RoPE QK head
차원 = `qk_nope_head_dim`), `d_v`(value head 차원 = `v_head_dim`), 그리고 `n_h`, `d_model`.

## 트레이스에서 식별하는 방법 (Phase 6 DeepSeek-V2-Lite 실측)
self_attn 블록의 projection 모듈(weight_shape는 심볼 렌더):
- `q_proj` `[n_h*(d_nope+d_rope), d_model]` — 작은 모델은 Q 비압축(예: V2-Lite `[3072, d_model]`).
  큰 모델은 `q_a_proj`(→`c_q`) + `q_a_layernorm` + `q_b_proj`로 Q도 압축(V3 계열).
- `kv_a_proj_with_mqa` `[c_kv + d_rope, d_model]` — KV를 latent(`c_kv`)로 압축 + 공유 RoPE 키
  차원(`d_rope`)을 함께 뽑음(예: V2-Lite `[576, d_model]` = 512+64).
- `kv_a_layernorm` — 압축 latent에 대한 RMSNorm(pow/mean/rsqrt/mul로 분해되어 잡힘).
- `kv_b_proj` `[n_h*(d_nope+d_v), c_kv]` — latent에서 head별 k(non-RoPE)와 v로 복원
  (예: V2-Lite `[2*d_model, 512]` = 16*(128+128) ← 512).
- `o_proj` `[d_model, d_model]`.

특징 op: `split_with_sizes`(q를 nope/rope로, kv latent를 압축분/rope분으로 분리),
`view_as_complex`/`view_as_real`(네이티브 transformers DeepSeek-V2는 **복소수 곱 기반 RoPE**),
`concat`(k = k_nope ⊕ k_rope 조립), 그리고 압축→복원 경로가 `depends_on`으로 이어짐
(Phase 1에서 격리 검증한 MLA 패턴과 동일).

KV cache는 head별 K/V 전체가 아니라 압축 latent(`c_kv`, +`d_rope`)만 저장 → `model_summary.md`
의 "KV cache 크기"가 표준 `2·n_kv·d_head`가 아닌 압축분으로 표기된다.

## 확인된 모델 (계속 추가)
- **`deepseek-ai/DeepSeek-V2-Lite`** (Phase 6): 27 layers, `n_h`=16, `c_kv`=512, `d_rope`=64,
  `d_nope`=128, `d_v`=128, Q 비압축. revision `604d5664dddd88a0433dbae533b7fe9472482de0`.
  네이티브 transformers `deepseek_v2` 구현으로 로드(원 repo의 remote 코드는 구버전
  transformers 의존이라 5.x에서 import 실패 — native 우선 정책, provenance.needs_remote_code).

## 참고 소스
- transformers `models/deepseek_v2/modeling_deepseek_v2.py`(네이티브 구현) — 트레이스로 직접 관측
- DeepSeek-V2 Technical Report (arXiv:2405.04434) — MLA 설계·decoupled RoPE 근거(교차검증용)
- config docstring: `kv_lora_rank`, `q_lora_rank`, `qk_rope_head_dim`, `qk_nope_head_dim`, `v_head_dim`
