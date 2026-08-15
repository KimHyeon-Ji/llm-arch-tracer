# 병렬 하이브리드 (attention ‖ SSM, 한 층 안에서 동시에) — sequence mixing 배치

`ssm/mamba.md` 가 문서화한 하이브리드는 **층을 번갈아 쓰는** 방식이다(Nemotron-H, granite-4.0-h:
어떤 층은 Mamba, 어떤 층은 attention). 이 문서는 다른 배치를 다룬다 — **모든 층이 두 믹서를
동시에** 돌리고 그 출력을 더한다.

## 정의

한 디코더 층이 같은 입력을 attention 과 SSM 에 **각각** 통과시킨 뒤 합산하고, 잔차는 그
합에 **한 번만** 더한다:

```
residual = x
h = mamba(norm(x)) * ssm_out_mult  +  attn(norm(x) * attn_in_mult) * attn_out_mult
x = residual + h            # 잔차 하나, 믹서 둘
x = x + mlp(norm(x))
```

층 교대형과 구별되는 지점은 셋이다:

1. **`layer_sched` 가 전부 같은 값**이다. 교대형은 층마다 유형이 달라 클러스터가 2~3개로
   갈리지만, 병렬형은 모든 층이 동일하다 — C2 가 클러스터 1개를 보고해도 정상이다.
2. **KV cache 와 SSM state 가 같은 층에 공존**한다. 교대형은 층마다 둘 중 하나만 굴린다.
3. **파라미터 수가 층당 두 믹서 몫**이다. attention 층 수로 KV cache 를 세면 안 되고,
   `L` 전체가 attention 층이자 SSM 층이다.

## 관련 심볼 / config / 파라미터

config: `mamba_d_state`, `mamba_chunk_size`, `mamba_n_heads`, `mamba_d_head`, `mamba_d_ssm`,
`mamba_n_groups`, `mamba_d_conv`, `mamba_expand` 와 표준 attention 필드
(`num_attention_heads`, `num_key_value_heads`, `head_dim`) 가 **동시에** 존재한다.
둘 다 있으면서 `layers_block_type` / `hybrid_override_pattern` 같은 **스케줄 필드가 없으면**
병렬형을 의심하는 것이 맞다 — 교대형은 스케줄을 반드시 명시해야 하기 때문이다.

**MuP 배수**: `attention_in_multiplier`, `attention_out_multiplier`, `ssm_in_multiplier`,
`ssm_out_multiplier`, `ssm_multipliers`(구간별 벡터), `embedding_multiplier`,
`lm_head_multiplier`, `key_multiplier`. 트레이스에는 평범한 `elementwise_mul` 로 나타나므로
**op 만 보면 안 보인다** — config 에서 확인해야 한다.

## 트레이스에서 식별하는 방법 (블라인드 온보딩 2026-08-15, Falcon-H1-7B 실측)

- 한 `*DecoderLayer` 안에 `self_attn` 과 `mamba` 두 서브모듈이 **모두** 있고, 둘의 출력이
  `elementwise_add` 로 합쳐진 뒤 잔차가 붙는다.
- attention 쪽은 표준 GQA(`sdpa`/softmax, RoPE). SSM 쪽은 `ssm/mamba.md` 의 Mamba2 지문 그대로
  (`constant_pad_nd` → `convolution`, `exp`, `cumsum`, `tril`+`masked_fill`, `sum`).
- `RMSNormGated` 가 스캔 출력과 gate 를 함께 받는다(`norm_before_gate` 로 순서가 바뀜).
- **fast-path 폴백 확인 필요**: `causal_conv1d_fn` / `selective_state_update` 미설치 시
  transformers 가 naive torch 로 떨어져야 meta 트레이싱이 된다(로그: "fast path is not
  available ... naive implementation"). Falcon-H1 은 폴백이 있어 문제 없었다.

### `d_chunk` 와 `d_state` 가 같은 값일 때 (이 계열의 고유 함정)

`mamba_chunk_size == mamba_d_state` 인 모델이 있다(Falcon-H1-7B: 둘 다 256). 값으로는
**원리적으로** 못 가리므로 소스의 축 위치로만 갈린다:

- `reshape_into_chunks` 출력 `[B, n_chunks, chunk_size, n_heads, state_size]` — 축 2 가 청크,
  마지막 축이 state.
- 청크 내 6-D 텐서는 `(b, c, l, s, h, n)` 이고 `l`·`s` 가 chunk_size, `h` 가 n_heads,
  `n` 이 state_size다. 근거가 **둘 다** 있다:
  1. `C[:, :, :, None, :, :] * B[:, :, None, :, :, :]` 브로드캐스트와 뒤따르는 `sum(-1)` 이
     그 형태를 **강제한다** — 주석이 없어도 도출된다.
  2. 설치된 transformers 5.14.1 의 `modeling_falcon_h1.py:818` 에는 그 형태가 **주석으로도**
     적혀 있다(`# shape: (b, c, l, s, h, n)`).

  > **버전 주의.** 2 는 실행된 코드에는 있지만 GitHub `main` 사본에는 없다. 2026-08-15 에
  > 두 검토자가 "소스를 읽었다"면서 이 점에서 엇갈렸고, 각자 자기 파일 기준으로는 맞았다 —
  > `develop/sources/` 가 `main` 을 받아 캐시하고 있었기 때문이다(32개 model_type 중 29개가
  > 실행된 코드와 달랐다). `source_check.fetch` 가 이제 **설치본을 먼저** 읽는다.
  > 근거를 인용할 때는 어느 버전인지 함께 적는 것이 안전하다.
- `segment_sum` 의 `torch.ones(chunk_size, chunk_size)` 는 **양 축 모두** chunk_size 다.

같은 함정이 `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B` 에도 있다(두 값이 다른 Zamba2 ·
granite-4.0-h 는 값으로 저절로 갈려서 드러나지 않았을 뿐이다).

이 축 순서는 Zamba2, Nemotron-3-Super/Ultra, granite-4.0-h, Falcon-H1의 naive torch
SSD 구현에서 각각 대조했다. 구현마다 변수명은 조금 다르지만 chunk reshape, 정사각
segment mask, 6-D broadcast product의 축 의미는 같다.

## 확인된 모델

- **`tiiuae/Falcon-H1-7B-Instruct`** (블라인드 온보딩 2026-08-15, `falcon_h1` arch):
  44 층 전부 hybrid. attention GQA `n_h`=12 / `n_kv`=2 / `d_head`=128, SSM `d_state`=256 /
  `d_chunk`=256 / `mamba_n_heads`=24 / `mamba_d_head`=128 / `d_ssm`=3072 / `n_groups`=1 /
  `d_conv`=4, `d_model`=3072, `d_ff`=12288, `V`=130049.
  코드 변경 0으로 트레이스 통과(C1–C16 중 12 PASS / 0 FAIL). 남은 인계는 위의
  `d_chunk`/`d_state` 한 가지뿐이었다.
  출처: `modeling_falcon_h1.py` `FalconH1DecoderLayer.forward` (두 믹서 합산 후 잔차 1회),
  `reshape_into_chunks` / `segment_sum` (청크 축 위치).

## 참고 소스

- transformers `models/falcon_h1` 구현 (naive 폴백) — 트레이스로 직접 관측
- Falcon-H1 기술 보고서 / TII 모델 카드 (병렬 배치와 MuP 배수의 근거)
- Mamba-2(SSD) 논문 — 청크 스캔의 축 의미
