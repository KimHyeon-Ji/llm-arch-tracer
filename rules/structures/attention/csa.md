# CSA — Compressed Sparse Attention (블록 압축 + Lightning Indexer top-k)

## 정의
KV를 **m개 토큰마다 1개의 압축 엔트리**로 줄인 뒤(`compress_rates["compressed_sparse_attention"]`,
DeepSeek-V4 기본 m=4), 그 압축 시퀀스 전체가 아니라 **Lightning Indexer가 쿼리마다 고른
top-`index_topk`개 엔트리만** 보게 하는 attention. 압축(dense→short)과 희소 선택(short→top-k)이
2단으로 걸린다.

핵심은 압축 윈도우가 **겹친다**는 점이다. `kv_proj`/`gate_proj`/`position_bias`가 `2*d_head`로
투영해 토큰마다 두 계열 Ca/Cb를 만든다 — Ca(`[..., :d_head]`)는 **다음** 윈도우의 압축
엔트리에, Cb(`[..., d_head:]`)는 **현재** 윈도우에 기여한다. 압축 엔트리 w는 윈도우 w-1의 Ca와
윈도우 w의 Cb를 합친 `2*m`개 슬롯에 대한 softmax 가중합이다(너비 `2*m`, stride `m`).
w=0의 앞 절반은 이전 forward 호출의 Ca가 필요해 캐시(`overlap_kv`)가 넘겨주며, 첫 호출에는
zero-kv / `-inf`-gate로 남아 softmax 가중치 0이 된다.

압축 엔트리는 sliding-window KV 뒤에 **concat**되어 코어 attention에 들어간다. 즉 한 레이어가
로컬(sliding `w_local`) + 장거리(압축 top-k) 두 경로를 동시에 본다.

MLA와 혼동 주의: MLA는 KV를 latent로 압축했다가 **head별로 복원**한다(시퀀스 길이 불변).
CSA는 **시퀀스 축을 m배 줄인다**(head 복원 없음). V4는 MLA가 아니라 shared-KV MQA
(`n_kv`=1) + 압축 분기 조합이다 — [mla.md](mla.md)의 `c_kv`/`d_nope`/`d_v` 별칭이 V4 config에
없는 건 오류가 아니라 정상이다.

## 관련 심볼 (rules/symbols.yaml)
`n_h`, `n_kv`(=1), `d_head`, `d_model`, `w_local`(sliding_window), `layer_sched`, `c_q`(=`q_lora_rank`,
indexer의 `q_b_proj` 입력), `d_rope`(= `d_head * partial_rotary_factor`).

아직 별칭이 없는 V4 고유 필드(SKIP으로 뜸 — symbols.yaml 추가 후보):
`compress_rates["compressed_sparse_attention"]`(m), `index_n_heads`(n_h^I), `index_head_dim`(c^I),
`index_topk`(k^I).

## 트레이스에서 식별하는 방법 (DeepSeek-V4-Pro 실측, T=2048, m=4)
모듈 경로 — `self_attn.compressor` 아래에 **`indexer` 서브모듈이 있으면 CSA**(없으면 [hca.md](hca.md)):
- `self_attn.compressor.kv_proj` `[2*d_head, d_model]`, `gate_proj` 동일 — Ca/Cb 두 계열이라 `2*`
- `self_attn.compressor.position_bias` `[m, 2*d_head]` (파라미터)
- `self_attn.compressor.kv_norm` — 압축 엔트리 RMSNorm
- `self_attn.compressor.indexer.kv_proj` / `gate_proj` `[2*c^I, d_model]`, `position_bias` `[m, 2*c^I]`
- `self_attn.compressor.indexer.q_b_proj` `[n_h^I * c^I, c_q]` — 쿼리는 `q_residual`(= `q_a_norm(q_a_proj(h))`)
  재사용, 즉 코어 attention과 Q 저랭크 경로를 공유
- `self_attn.compressor.indexer.scorer.weights_proj` `[n_h^I, d_model]`

특징 op:
- `aten.topk` (indexer의 top-k 선택) + `aten.masked_fill` (future_mask, `-inf`) + `aten.where` (`-1` 센티널)
- `aten.scatter_` — per-query block bias를 `compressed_len+1` 버퍼에 흩뿌린 뒤 마지막 열을 잘라냄
- `aten.relu` — scorer의 `∑_h w · ReLU(q·K)` (softmax가 아니라 ReLU인 게 식별 포인트)
- `aten.cat` — sliding KV + 압축 KV 결합
- 겹침 레이아웃의 `chunk_kv[:, :-1]` 슬라이스 → **`n_win - 1`** 크기 축이 나타남

실측 shape (T=2048, m=4 ⇒ `n_win` = T/m = 512):
| 값 | 유래 | 확인 위치 |
|---|---|---|
| 512 | `n_win` = T/m | 압축 엔트리 수 |
| 511 | `n_win - 1` (Ca 겹침 shift) | `compressor` slice, op 1709~1712 |
| 2560 | T + `n_win` = 2048+512 | `self_attn` cat (sliding KV ⊕ 압축 KV) |
| 2561 | 2560 + 1 (**attention sink** 1열) | `self_attn` cat, params=`self_attn.sinks` |

> `n_win`=512가 `d_head`=512와 값이 같아 심볼라이저가 `[B, d_head, 4, 2*d_head]`처럼 렌더한다
> (심볼 우선순위 충돌). OLMoE `2*d_ff==d_model` 케이스와 동일한 알려진 표기 한계 —
> 숫자 자체는 정확하다.

## KV cache 기여
압축 레이어가 컨텍스트에 따라 **증가시키는** 캐시는 토큰당 `d_head / m` elem이다(m 토큰마다
엔트리 1개). K와 V가 같은 텐서(`update(kv, kv, …)`, `attention(q, kv, kv)`)라 **단일 텐서**로
센다 ⇒ BF16에서 `d_head/m × 2` B/layer/token. CSA는 512/4×2 = **256 B**, HCA는 512/128×2 = **8 B**.

sliding 분기는 `w_local`로 상한이 있어 증가하지 않고(V4-Pro: 61층×128×512×2B ≈ 7.63 MiB 고정),
Lightning Indexer는 자체 압축 캐시를 따로 든다(`c^I/m` = 128/4 elem/token/CSA층).

Raschka 갤러리의 "KV cache / token" 수치는 **압축 엔트리만** 센 값이며, 위 두 항목은 제외한다
([계산 규약](https://sebastianraschka.com/llm-architecture-gallery/kv-cache-calculations/)).
재현 확인:

| 모델 | 구성 | 계산 | 갤러리 |
|---|---|---|---|
| V4-Pro | 30 CSA + 31 HCA | 30×256 + 31×8 = 7928 B = **7.74 KiB** | 7.7 KiB · Very low |
| V4-Flash | 21 CSA + 20 HCA + 2 sliding | 21×256 + 20×8 = 5536 B = **5.41 KiB** | 5.4 KiB · Very low |

`sliding_attention` 레이어(V4-Flash의 2층)는 압축기가 없어 **0 기여**다 — 이게 갤러리 값과
맞아떨어지는 것으로 교차검증됐다. `src/summarize.py`가 이 규약대로 계산한다.

**attention sink**: `self_attn.sinks` `[n_h]` 파라미터가 score 행렬에 1열로 concat된다
(gpt-oss와 동일 계열). `kv_len + 1`이 나오면 이것 — [hca.md](hca.md)에도 동일하게 적용된다.

## 확인된 모델 (계속 추가)
- **`zai-org/GLM-5.2`** (Phase 26): 78층, `model_type: glm_moe_dsa` — Zhipu 가 DeepSeek Sparse
  Attention 을 채택한 것으로, config 의 `index_head_dim`/`index_n_heads`/`index_topk` 가
  DeepSeek-V4 의 indexer 와 같은 자리를 차지한다. 실측 `n_h_I`=32, `c_I`=128, `k_I`=2048,
  MLA 쪽은 `c_q`=2048, `c_kv`=512, `d_nope`=192, `d_v`=256, `d_rope`=64, MoE `E`=256 top-8.
  **새 규칙이 하나도 필요 없었다** — V4 용으로 등록해 둔 indexer/MLA 심볼이 그대로 맞았고
  휴리스틱 0.00%, 미등록 config 필드 0, 검토 의뢰서도 비었다. 다른 벤더의 새 아키텍처가
  기존 규칙으로 그대로 설명된 첫 사례다.
- **`deepseek-ai/DeepSeek-V4-Pro`** (예약 최종테스트, 2026-07-23): 61층 중 **30층이 CSA**
  (`layer_types`에서 2층 HCA 부트스트랩 후 HCA/CSA 교대 — layer 2,4,6,… 가 CSA).
  `n_h`=128, `n_kv`=1, `d_head`=512, `c_q`=1536, m=4, `n_h^I`=64, `c^I`=128, `k^I`=1024,
  `w_local`=128. revision `b5968e9190ef611bbf34a7229255be88a0e937c1`.
  네이티브 transformers `deepseek_v4` 구현으로 로드(native 우선 정책,
  `provenance.needs_remote_code`).
- **`deepseek-ai/DeepSeek-V4-Flash`** (예약 최종테스트, 2026-07-23): 43층, 동일 구조의
  소형 구성(`n_h`=64, `o_groups`=8).

## 참고 소스
- transformers 5.14.1 `models/deepseek_v4/modular_deepseek_v4.py`
  — `DeepseekV4CSACompressor`(§2.3.1 eqs. 9–17), `DeepseekV4Indexer`(eqs. 13–17),
  `DeepseekV4IndexerScorer`. **트레이스로 직접 관측**(1차 소스)
- `configuration_deepseek_v4.py` docstring — `compress_rates`, `index_n_heads`,
  `index_head_dim`, `index_topk` 정의 및 paper 절 번호
- DeepSeek-V4 technical report §2.3.1 (구현 주석이 인용) — 설계 의도 교차검증용
