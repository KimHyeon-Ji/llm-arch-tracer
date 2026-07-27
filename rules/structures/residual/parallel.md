# 병렬 잔차 (Parallel Attention + MLP, 단일 LayerNorm)

## 정의
attention과 MLP를 **직렬이 아니라 병렬**로 놓고, 블록당 **LayerNorm 하나**만 써서
같은 정규화 출력을 두 분기가 함께 받는 구조. GPT-J(Wang & Komatsuzaki, 2021)에서 도입됐고
Falcon 계열이 채택했다.

```
표준([standard.md](standard.md)):     h = h + attn(ln1(h))
                                      h = h + mlp(ln2(h))        ← 순차 2단, LN 2개

병렬(이 문서):                         n = ln(h)                  ← LN 1개
                                      h = h + attn(n) + mlp(n)   ← 같은 n에서 분기, 한 번에 합류
```

**동기는 정확도가 아니라 통신량**이다. 텐서 병렬 실행 시 레이어당 `all_reduce`가
2회 → **1회**로 줄어든다(attention과 MLP를 서로 다른 디바이스에서 동시에 돌리고 마지막에
한 번만 모음). Falcon 논문이 밝힌 채택 이유가 이것이다.

## 관련 심볼 (rules/symbols.yaml)
별도 심볼 없음. config 플래그로 식별한다: `parallel_attn`(true면 병렬),
`num_ln_in_parallel_attn`(병렬 블록의 LN 개수 — Falcon-40B 계열 `new_decoder_architecture`는
2개를 쓰기도 한다), `apply_residual_connection_post_layernorm`.

## 트레이스에서 식별하는 방법 (falcon-7b 실측, 32층)
- **레이어 구조에 LayerNorm이 하나뿐이다**: `layer 0-31: input_layernorm, mlp, self_attention`
  — `post_attention_layernorm`이 **없다**. 표준 블록이면 반드시 둘이다.
- 의존성 그래프에서 `self_attention`과 `mlp`의 입력이 **같은 노드**를 가리킨다(fork).
  직렬 블록은 mlp 입력이 attention 출력 이후 잔차 add에 의존한다.
- C5(잔차 불변식)는 **PASS**한다 — 마지막 축(d_model) 기준으로 보기 때문에 분기/합류
  형태에 영향받지 않는다(falcon-7b: `residual stream at d_model=4544 in 32/32 layers`).

## ⚠ 이 계열에서 반드시 조심할 것 — config 필드가 실제 값이 아니다
Falcon은 `num_kv_heads`를 갖고 있지만 **그 값이 실제 KV head 수가 아닐 수 있다.**
`multi_query=True`(이고 `new_decoder_architecture=False`)이면 실제 KV head는 **1개**다.

```python
# transformers models/falcon/modeling_falcon.py
num_kv_heads = config.num_kv_heads if (new_decoder_architecture or not multi_query) else 1
qkv_out_dim  = (num_kv_heads * 2 + num_attention_heads) * head_dim
```

falcon-7b는 `num_kv_heads=71 == num_attention_heads=71`이라 **필드만 보면 MHA로 보이지만
실제로는 MQA**다. 블라인드 검증(2026-07-27)에서 이걸 놓쳐 C7이 "MHA"로 PASS하고
**KV cache 카드가 71배 과대평가**(568 KiB vs 실제 8 KiB)됐다. 지금은 `resolve_symbols`가
위 규칙을 그대로 반영한다.

**트레이스가 진실을 말해준다**: fused QKV 가중치 폭이 `(n_h + 2·n_kv)·d_head`이므로
falcon-7b는 `(71+2)·64 = 4672`. 4544(=71·64)가 아니라 4672라는 사실 자체가 KV head가
1개임을 증명한다. config 플래그 해석이 의심스러우면 이 폭을 역산해 확인할 것.

## 확인된 모델 (계속 추가)
- **`tiiuae/falcon-7b`** (블라인드 검증 #2, 2026-07-27): 32층, `d_model`=4544,
  `n_h`=71(소수), `d_head`=64, **`n_kv`=1 (MQA, `multi_query=True`)**, `V`=65024, tie=True,
  bias 없음, RoPE. `parallel_attn=True`, LN 1개. fused QKV 폭 4672.
- 참고: **`tiiuae/falcon-rw-1b`은 이 구조가 아니다** — `parallel_attn=False`, ALiBi 사용,
  MHA(32:32), bias 있음. RefinedWeb 어블레이션 모델이라 아키텍처가 다르다.
  (ALiBi는 아직 `position_encoding/`에 문서가 없다 — 후속 과제.)

## 참고 소스
- **1차** transformers 5.14.1 `models/falcon/modeling_falcon.py`, `configuration_falcon.py`
  — 트레이스로 직접 관측
- [The Falcon Series of Open Language Models (arXiv:2311.16867)](https://arxiv.org/pdf/2311.16867)
  — 병렬 attention+MLP 채택 이유(all_reduce 2회→1회), bias 제거, multi-query
- Wang & Komatsuzaki, GPT-J (2021) — 병렬 블록의 원 출처
