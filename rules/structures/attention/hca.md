# HCA — Heavily Compressed Attention (고압축률 블록 압축, indexer 없음)

## 정의
[csa.md](csa.md)와 같은 블록 압축이되 **압축률이 훨씬 크고(m′=128 vs m=4) 희소 선택이 없는** 변형.
닫힌 윈도우 m′개 토큰마다 압축 엔트리 하나를 만들고
(`C^Comp_i = Σ_{j∈window} softmax(Z_j + B)_j ⊙ C_j`), 쿼리는 지금까지 쌓인 **모든** 압축
엔트리를 본다. 엔트리 수가 T/m′로 이미 매우 작아서 Lightning Indexer의 top-k 선별이 불필요하다.

CSA와의 구조 차이:

| | CSA | HCA |
|---|---|---|
| 압축률 | m=4 | m′=128 |
| 계열 | Ca/Cb 2계열, 겹치는 윈도우(`2*d_head` 투영) | **1계열, 겹침 없음**(`d_head` 투영) |
| 선택 | Lightning Indexer top-`k^I` | 없음 — 전체 압축 엔트리 |
| block bias | indexer 유효성 + causal (`scatter_`) | causal만 (`masked_fill`) |
| 캐시 | `DeepseekV4CSACache` (overlap 상태 보유) | `DeepseekV4HCACache` |

압축 엔트리의 RoPE는 결정적 절대 위치 `i * m′ + first_window_position`에 걸려서, forward 호출을
넘나드는 concat에서도 causality가 유지된다. CSA와 마찬가지로 압축 KV는 sliding-window KV 뒤에
concat되어 코어 attention에 들어간다(로컬 + 장거리 2경로).

causal 규칙: 쿼리 t는 `entry_idx < (position_ids + 1) // m′`인 엔트리만 볼 수 있다. 윈도우 w의
엔트리는 위치 `w*m′ … (w+1)*m′`의 정보를 담으므로, 아직 닫히지 않은 윈도우를 미리 보면 누설이다.

## 관련 심볼 (rules/symbols.yaml)
`n_h`, `n_kv`(=1), `d_head`, `d_model`, `w_local`, `layer_sched`, `d_rope`.
V4 고유 필드 `compress_rates["heavily_compressed_attention"]`(m′)는 아직 별칭 없음(SKIP).

## 트레이스에서 식별하는 방법 (DeepSeek-V4-Pro 실측, T=2048, m′=128)
`self_attn.compressor`는 있는데 **`indexer` 서브모듈이 없으면 HCA**:
- `self_attn.compressor.kv_proj` `[d_head, d_model]` — CSA와 달리 `2*`가 **아님**(단일 계열)
- `self_attn.compressor.gate_proj` `[d_head, d_model]`
- `self_attn.compressor.position_bias` `[m′, d_head]`
- `self_attn.compressor.kv_norm`
- `topk`/`scatter_`/`relu`가 **없음** — 있으면 CSA다

실측 shape (T=2048, m′=128 ⇒ 엔트리 수 = T/m′ = 16):
| 값 | 유래 | 확인 위치 |
|---|---|---|
| 16 | T/m′ | mask cat `[[B,B,T,T],[B,B,T,16]]`, op 464 |
| 2064 | T + T/m′ = 2048+16 | `self_attn` cat (sliding KV ⊕ 압축 KV) |
| 2065 | 2064 + 1 (attention sink 1열) | `self_attn` cat, params=`self_attn.sinks` |

KV cache 기여: 토큰당 `d_head/m′` = 512/128 = 4 elem, K==V 단일 텐서 ⇒ BF16 **8 B/layer/token**
(CSA의 1/32). 상세·재현 검증은 [csa.md](csa.md)의 "KV cache 기여" 참고.

부분 RoPE 흔적(코어 attention·압축 분기 공통): `d_head`를 `d_head - d_rope` / `d_rope`로 쪼개
회전분만 처리 후 다시 concat →
`slice [B,n_h,T,d_head] → [B,n_h,T,448]`, `cat [448, d_rope] → d_head` (op 321·334).
**448 = `d_head` - `d_rope` = 512 - 64**(`partial_rotary_factor`=0.125). CSA/HCA/코어 attention
어디서나 나오므로 V4 계열 식별 보조 신호로 쓸 수 있다.

## 확인된 모델 (계속 추가)
- **`deepseek-ai/DeepSeek-V4-Pro`** (예약 최종테스트, 2026-07-23): 61층 중 **31층이 HCA**.
  `layer_types` = HCA 2층 부트스트랩(layer 0,1) 후 CSA/HCA 교대로 layer 3,5,7,… 이 HCA.
  m′=128, `d_head`=512, `w_local`=128.
  revision `b5968e9190ef611bbf34a7229255be88a0e937c1`.
- **`deepseek-ai/DeepSeek-V4-Flash`** (예약 최종테스트, 2026-07-23): 43층 동일 구조.

이 HCA/CSA 교대 스케줄이 C2(레이어 이종성)에서 4개 클러스터로 잡힌 이유다 —
attention 2종 × MoE 2종(`mlp_layer_types`의 `hash_moe`/`moe`) 조합.

## 참고 소스
- transformers 5.14.1 `models/deepseek_v4/modular_deepseek_v4.py`
  — `DeepseekV4HCACompressor`(§2.3.2 eqs. 20–23), `DeepseekV4HCACache`. **트레이스로 직접 관측**
- `configuration_deepseek_v4.py` — `DEEPSEEK_V4_LAYER_TYPES`,
  `_COMPRESS_RATIO_TO_LAYER_TYPE = {0: sliding, 4: CSA, 128: HCA}`
- DeepSeek-V4 technical report §2.3.2 — 교차검증용
