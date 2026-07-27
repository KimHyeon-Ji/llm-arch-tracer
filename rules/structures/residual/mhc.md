# mHC — Manifold-Constrained Hyper-Connections (다중 스트림 잔차 + Sinkhorn 이중확률 제약)

## 정의
잔차를 **단일 스트림 `[B, T, d_model]`이 아니라 `n_hc`개 병렬 스트림
`[B, T, n_hc, d_model]`로 유지**하는 잔차 연결(Xie et al., 2026; DeepSeek-V4 paper §2.2).
[standard.md](standard.md)의 `h = h + sublayer(norm(h))` 단일 add를 다음으로 대체한다:

```
post, comb, collapsed = HyperConnection(hidden_streams)      # hidden_streams: [B, T, n_hc, d_model]
out = sublayer(norm(collapsed))                              # collapsed: [B, T, d_model]
hidden_streams = post[..., None] * out[..., None, :] + (comb^T @ hidden_streams)
```

- **pre** `[B,T,n_hc]` — `n_hc`개 스트림을 sublayer 입력용 하나로 collapse하는 가중치.
  `σ(pre_w·scale₀ + base₀) + eps`
- **post** `[B,T,n_hc]` — sublayer 출력을 각 스트림에 어떻게 배분할지. `2·σ(...)`라 범위 [0,2]
- **comb** `[B,T,n_hc,n_hc]` — 스트림끼리 섞는 행렬. softmax 후 **Sinkhorn-Knopp `t_max`회
  반복으로 이중확률(doubly-stochastic) 행렬 다양체에 사영**된다. 이게 "Manifold-Constrained"의
  뜻이며, 행·열 합이 1로 묶여 깊은 스택에서 신호 전파가 non-expansive해진다.

`comb`는 **전치해서** 소비된다(`comb.transpose(-1,-2) @ hidden_streams`). Sinkhorn 결과는
이중확률이지만 대칭은 아니라서 방향이 의미를 가진다.

레이어마다 HyperConnection이 **2개**(attention 자리 `attn_hc`, MLP 자리 `ffn_hc`)이고,
스택 마지막에 `HyperHead`가 `n_hc` 스트림을 하나로 접어 최종 RMSNorm에 넘긴다.

`pre`/`post`/`comb` 계산과 Sinkhorn은 전부 **fp32**로 돌고(`.float()`), 섞기 직전에 입력 dtype으로
되돌린다.

## 관련 심볼 (rules/symbols.yaml)
`d_model`, `L`. V4 고유 필드는 아직 별칭 없음(SKIP):
`hc_mult`(n_hc), `hc_sinkhorn_iters`(t_max), `hc_eps`.

## 트레이스에서 식별하는 방법 (DeepSeek-V4-Pro 실측, n_hc=4, t_max=20)
모듈: `model.layers.N.attn_hc`, `model.layers.N.ffn_hc`, 그리고 모델 레벨 `HyperHead`.

파라미터 shape:
- `attn_hc.fn` / `ffn_hc.fn` `[(2+n_hc)*n_hc, n_hc*d_model]` — V4-Pro는 `[24, 28672]`
- `attn_hc.base` `[(2+n_hc)*n_hc]` = `[24]`, `attn_hc.scale` `[3]` (pre/post/comb 각각의 스케일)
- `HyperHead.hc_fn` `[n_hc, n_hc*d_model]`, `hc_base` `[n_hc]`, `hc_scale` `[1]`

**가장 확실한 식별 신호 — Sinkhorn 반복이 트레이스에 그대로 펼쳐진다.**
`hc_*` 모듈 하나(layer 2 `attn_hc`, 총 145 op) 실측:

| op_type | 개수 | 유래 |
|---|---|---|
| `div` | **39** | Sinkhorn 정규화 = `2 * t_max - 1` (초기 열정규화 1 + 19회 × 행·열 2) |
| `sum` | 40 | Sinkhorn 39 + 스트림 collapse 1 |
| `sigmoid` | 2 | `pre`, `post` |
| `split_with_sizes` | 2 | `fn` 출력과 `base`를 `[n_hc, n_hc, n_hc²]`로 분해 |

⇒ **`div` 개수 = `2 * hc_sinkhorn_iters - 1`**. `hc_*` 모듈에서 `[B,T,n_hc,n_hc]` 텐서에
`div`/`sum`이 수십 번 교대로 걸려 있으면 mHC다. 대표 op:
`div [[B,T,4,4],[B,T,B,4]] → [B,T,4,4]` (op 1511), `sum [B,T,4,4] → [B,T,4,B]` (op 1512).

정규화는 **weightless RMSNorm**(`DeepseekV4UnweightedRMSNorm`, 학습 파라미터 없음)이라
`pow/mean/rsqrt/mul`은 나오지만 대응하는 weight 파라미터가 없다 — C10 커버리지에서
"파라미터 없는 norm"으로 보이는 게 정상이다.

## 검증(C1~C16)에 주는 영향
- **C5**: 잔차 스트림이 `[B,T,n_hc,d_model]`이라 "두 피연산자 shape 동일한 단일 elementwise add"
  가정이 깨진다. `02-new-module-handling.md`의 "비표준 잔차 연결 — 다중 스트림 믹싱" 범주에
  정확히 해당. V4-Pro에서 C5는 **PASS**했고 `d_model=7168 잔차 스트림 61/61층` 으로 보고됐다
  (마지막 축 기준으로 보므로 완화된 불변식이 이미 성립).
- **C2**: `hc_*` 모듈은 모든 레이어에 동일하게 있어 클러스터 구분에 기여하지 않는다.
  V4-Pro의 4개 클러스터는 attention 2종([../attention/csa.md](../attention/csa.md) /
  [../attention/hca.md](../attention/hca.md)) × MoE 2종에서 온다.
- 레이어당 op 수가 크게 는다(`hc_*` 2개 × 145 op ≈ 290 op/layer). 61층이면 이것만 약 1.8만 op —
  C16 unmapped 카운트(31,440행)가 큰 주된 이유 중 하나다.

## 확인된 모델 (계속 추가)
- **`deepseek-ai/DeepSeek-V4-Pro`** (예약 최종테스트, 2026-07-23): 61층, `n_hc`=4, `t_max`=20,
  `hc_eps`=1e-6, `d_model`=7168. 레이어당 `attn_hc`+`ffn_hc` 2개 + 모델 레벨 `HyperHead` 1개.
  revision `b5968e9190ef611bbf34a7229255be88a0e937c1`.
- **`deepseek-ai/DeepSeek-V4-Flash`** (예약 최종테스트, 2026-07-23): 43층, 동일 `n_hc`=4.

config 주석상 mHC는 **항상 활성**(`hc_mult`, "always active; Section 2.2")이라 V4 계열은
표준 잔차 연결을 쓰지 않는다.

## 참고 소스
- transformers 5.14.1 `models/deepseek_v4/modular_deepseek_v4.py`
  — `DeepseekV4HyperConnection`(ASCII shape 가이드 포함), `DeepseekV4HyperHead`,
  `DeepseekV4DecoderLayer.forward`. **트레이스로 직접 관측**(1차 소스)
- `configuration_deepseek_v4.py` docstring — `hc_mult`, `hc_sinkhorn_iters`, `hc_eps`
- Xie et al. 2026, Manifold-Constrained Hyper-Connections (구현 주석이 인용) — 교차검증용
- DeepSeek-V4 technical report §2.2 eq. 8
