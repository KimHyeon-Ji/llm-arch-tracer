# 부록 — 표가 주장하는 구조 (모델별)

각 모델의 발행 CSV 에서 **템플릿마다 한 벌씩** 뽑은 것입니다. 표는 접혀 있으므로
(같은 구조의 레이어는 한 번만 적고 `repeat`/`layers` 가 몇 개를 대표하는지 말합니다)
이것이 표가 그 모델의 아키텍처라고 주장하는 내용 전부입니다.

묶는 키는 `(block_type, repeat, layers)` 입니다. `block_type` 만으로 묶으면 서로 다른
블록이 합쳐져 구조가 사라집니다 -- 2026-09-16 검토에서 실제로 그랬습니다.

## meta-llama/Llama-4-Maverick-17B-128E

발행 좌표 **B=3, T=17**. 심볼 표:

```
B=3, E=128, T=17, V=202048, chunk_size=8192, ctx=262144, d_ff=16384, d_head=128, d_model=5120, d_moe=8192, n_h=40, n_kv=8
```

### prefill — 템플릿 6개 / 전체 67행

**`block_type=embed`  repeat=1  layers=`-`**  (1행)

```
   0 embedding        model.embed_tokens                 [[V, d_model], [B, T]] -> [[B, T, d_model]]  w=[V, d_model]
```

**`block_type=attn+FFN`  repeat=24  layers=`0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46`**  (16행)

```
   1 rmsnorm          input_layernorm                    [[B, T, d_model]] -> [[B, T, d_model]]  w=[d_model]
   2 matmul           self_attn.q_proj                   [[B*T, d_model], [d_model, n_h*d_head]] -> [[B*T, n_h*d_head]]  w=[n_h*d_head, d_model]
   3 matmul           self_attn.k_proj                   [[B*T, d_model], [d_model, n_kv*d_head]] -> [[B*T, n_kv*d_head]]  w=[n_kv*d_head, d_model]
   4 matmul           self_attn.v_proj                   [[B*T, d_model], [d_model, n_kv*d_head]] -> [[B*T, n_kv*d_head]]  w=[n_kv*d_head, d_model]
   5 batched_matmul   self_attn                          [[B*n_h, T, d_head], [B*n_h, d_head, T]] -> [[B*n_h, T, T]]
   6 softmax          self_attn                          [[B, n_h, T, T]] -> [[B, n_h, T, T]]
   7 batched_matmul   self_attn                          [[B*n_h, T, T], [B*n_h, T, d_head]] -> [[B*n_h, T, d_head]]
   8 matmul           self_attn.o_proj                   [[B*T, n_h*d_head], [n_h*d_head, d_model]] -> [[B*T, d_model]]  w=[d_model, n_h*d_head]
   9 elementwise_add  model.layers.0                     [[B, T, d_model], [B, T, d_model]] -> [[B, T, d_model]]
  10 rmsnorm          post_attention_layernorm           [[B, T, d_model]] -> [[B, T, d_model]]  w=[d_model]
  11 matmul           feed_forward.gate_proj             [[B*T, d_model], [d_model, d_ff]] -> [[B*T, d_ff]]  w=[d_ff, d_model]
  12 silu             feed_forward.activation_fn         [[B, T, d_ff]] -> [[B, T, d_ff]]
  13 matmul           feed_forward.up_proj               [[B*T, d_model], [d_model, d_ff]] -> [[B*T, d_ff]]  w=[d_ff, d_model]
  14 elementwise_mul  feed_forward                       [[B, T, d_ff], [B, T, d_ff]] -> [[B, T, d_ff]]
  15 matmul           feed_forward.down_proj             [[B*T, d_ff], [d_ff, d_model]] -> [[B*T, d_model]]  w=[d_model, d_ff]
  16 elementwise_add  model.layers.0                     [[B, T, d_model], [B, T, d_model]] -> [[B, T, d_model]]
```

**`block_type=attn+MoE`  repeat=12  layers=`1,5,9,13,17,21,25,29,33,37,41,45`**  (24행)

```
  17 rmsnorm          input_layernorm                    [[B, T, d_model]] -> [[B, T, d_model]]  w=[d_model]
  18 matmul           self_attn.q_proj                   [[B*T, d_model], [d_model, n_h*d_head]] -> [[B*T, n_h*d_head]]  w=[n_h*d_head, d_model]
  19 matmul           self_attn.k_proj                   [[B*T, d_model], [d_model, n_kv*d_head]] -> [[B*T, n_kv*d_head]]  w=[n_kv*d_head, d_model]
  20 matmul           self_attn.v_proj                   [[B*T, d_model], [d_model, n_kv*d_head]] -> [[B*T, n_kv*d_head]]  w=[n_kv*d_head, d_model]
  21 batched_matmul   self_attn                          [[B*n_h, T, d_head], [B*n_h, d_head, T]] -> [[B*n_h, T, T]]
  22 softmax          self_attn                          [[B, n_h, T, T]] -> [[B, n_h, T, T]]
  23 batched_matmul   self_attn                          [[B*n_h, T, T], [B*n_h, T, d_head]] -> [[B*n_h, T, d_head]]
  24 matmul           self_attn.o_proj                   [[B*T, n_h*d_head], [n_h*d_head, d_model]] -> [[B*T, d_model]]  w=[d_model, n_h*d_head]
  25 elementwise_add  model.layers.1                     [[B, T, d_model], [B, T, d_model]] -> [[B, T, d_model]]
  26 rmsnorm          post_attention_layernorm           [[B, T, d_model]] -> [[B, T, d_model]]  w=[d_model]
  27 matmul           feed_forward.router                [[B*T, d_model], [d_model, E]] -> [[B*T, E]]  w=[E, d_model]
  28 sigmoid          feed_forward.router                [[B*T, E]] -> [[B*T, E]]
  29 elementwise_mul  feed_forward                       [[B*E*T, d_model], [B*E*T, 1]] -> [[B*E*T, d_model]]
  30 batched_matmul   feed_forward.experts               [[E, B*T, d_model], [E, d_model, 2*d_moe]] -> [[E, B*T, 2*d_moe]]  w=[E, d_model, 2*d_moe]
  31 silu             feed_forward.experts.act_fn        [[E, B*T, d_moe]] -> [[E, B*T, d_moe]]
  32 elementwise_mul  feed_forward.experts               [[E, B*T, d_moe], [E, B*T, d_moe]] -> [[E, B*T, d_moe]]
  33 batched_matmul   feed_forward.experts               [[E, B*T, d_moe], [E, d_moe, d_model]] -> [[E, B*T, d_model]]  w=[E, d_moe, d_model]
  34 matmul           feed_forward.shared_expert.gate_pr [[B*T, d_model], [d_model, d_moe]] -> [[B*T, d_moe]]  w=[d_moe, d_model]
  35 silu             feed_forward.shared_expert.activat [[B*T, d_moe]] -> [[B*T, d_moe]]
  36 matmul           feed_forward.shared_expert.up_proj [[B*T, d_model], [d_model, d_moe]] -> [[B*T, d_moe]]  w=[d_moe, d_model]
  37 elementwise_mul  feed_forward.shared_expert         [[B*T, d_moe], [B*T, d_moe]] -> [[B*T, d_moe]]
  38 matmul           feed_forward.shared_expert.down_pr [[B*T, d_moe], [d_moe, d_model]] -> [[B*T, d_model]]  w=[d_model, d_moe]
  39 sum              feed_forward                       [[E, B*T, d_model]] -> [[B*T, d_model]]
  40 elementwise_add  model.layers.1                     [[B, T, d_model], [B, T, d_model]] -> [[B, T, d_model]]
```

**`block_type=attn+MoE`  repeat=12  layers=`3,7,11,15,19,23,27,31,35,39,43,47`**  (24행)

```
  41 rmsnorm          input_layernorm                    [[B, T, d_model]] -> [[B, T, d_model]]  w=[d_model]
  42 matmul           self_attn.q_proj                   [[B*T, d_model], [d_model, n_h*d_head]] -> [[B*T, n_h*d_head]]  w=[n_h*d_head, d_model]
  43 matmul           self_attn.k_proj                   [[B*T, d_model], [d_model, n_kv*d_head]] -> [[B*T, n_kv*d_head]]  w=[n_kv*d_head, d_model]
  44 matmul           self_attn.v_proj                   [[B*T, d_model], [d_model, n_kv*d_head]] -> [[B*T, n_kv*d_head]]  w=[n_kv*d_head, d_model]
  45 batched_matmul   self_attn                          [[B*n_h, T, d_head], [B*n_h, d_head, T]] -> [[B*n_h, T, T]]
  46 softmax          self_attn                          [[B, n_h, T, T]] -> [[B, n_h, T, T]]
  47 batched_matmul   self_attn                          [[B*n_h, T, T], [B*n_h, T, d_head]] -> [[B*n_h, T, d_head]]
  48 matmul           self_attn.o_proj                   [[B*T, n_h*d_head], [n_h*d_head, d_model]] -> [[B*T, d_model]]  w=[d_model, n_h*d_head]
  49 elementwise_add  model.layers.3                     [[B, T, d_model], [B, T, d_model]] -> [[B, T, d_model]]
  50 rmsnorm          post_attention_layernorm           [[B, T, d_model]] -> [[B, T, d_model]]  w=[d_model]
  51 matmul           feed_forward.router                [[B*T, d_model], [d_model, E]] -> [[B*T, E]]  w=[E, d_model]
  52 sigmoid          feed_forward.router                [[B*T, E]] -> [[B*T, E]]
  53 elementwise_mul  feed_forward                       [[B*E*T, d_model], [B*E*T, 1]] -> [[B*E*T, d_model]]
  54 batched_matmul   feed_forward.experts               [[E, B*T, d_model], [E, d_model, 2*d_moe]] -> [[E, B*T, 2*d_moe]]  w=[E, d_model, 2*d_moe]
  55 silu             feed_forward.experts.act_fn        [[E, B*T, d_moe]] -> [[E, B*T, d_moe]]
  56 elementwise_mul  feed_forward.experts               [[E, B*T, d_moe], [E, B*T, d_moe]] -> [[E, B*T, d_moe]]
  57 batched_matmul   feed_forward.experts               [[E, B*T, d_moe], [E, d_moe, d_model]] -> [[E, B*T, d_model]]  w=[E, d_moe, d_model]
  58 matmul           feed_forward.shared_expert.gate_pr [[B*T, d_model], [d_model, d_moe]] -> [[B*T, d_moe]]  w=[d_moe, d_model]
  59 silu             feed_forward.shared_expert.activat [[B*T, d_moe]] -> [[B*T, d_moe]]
  60 matmul           feed_forward.shared_expert.up_proj [[B*T, d_model], [d_model, d_moe]] -> [[B*T, d_moe]]  w=[d_moe, d_model]
  61 elementwise_mul  feed_forward.shared_expert         [[B*T, d_moe], [B*T, d_moe]] -> [[B*T, d_moe]]
  62 matmul           feed_forward.shared_expert.down_pr [[B*T, d_moe], [d_moe, d_model]] -> [[B*T, d_model]]  w=[d_model, d_moe]
  63 sum              feed_forward                       [[E, B*T, d_model]] -> [[B*T, d_model]]
  64 elementwise_add  model.layers.3                     [[B, T, d_model], [B, T, d_model]] -> [[B, T, d_model]]
```

**`block_type=norm`  repeat=1  layers=`-`**  (1행)

```
  65 rmsnorm          model.norm                         [[B, T, d_model]] -> [[B, T, d_model]]  w=[d_model]
```

**`block_type=head`  repeat=1  layers=`-`**  (1행)

```
  66 matmul           lm_head                            [[B*T, d_model], [d_model, V]] -> [[B*T, V]]  w=[V, d_model]
```

### decode — 템플릿 6개 / 전체 67행

**`block_type=embed`  repeat=1  layers=`-`**  (1행)

```
   0 embedding        model.embed_tokens                 [[V, d_model], [B, 1]] -> [[B, 1, d_model]]  w=[V, d_model]
```

**`block_type=attn+FFN`  repeat=24  layers=`0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46`**  (16행)

```
   1 rmsnorm          input_layernorm                    [[B, 1, d_model]] -> [[B, 1, d_model]]  w=[d_model]
   2 matmul           self_attn.q_proj                   [[B, d_model], [d_model, n_h*d_head]] -> [[B, n_h*d_head]]  w=[n_h*d_head, d_model]
   3 matmul           self_attn.k_proj                   [[B, d_model], [d_model, n_kv*d_head]] -> [[B, n_kv*d_head]]  w=[n_kv*d_head, d_model]
   4 matmul           self_attn.v_proj                   [[B, d_model], [d_model, n_kv*d_head]] -> [[B, n_kv*d_head]]  w=[n_kv*d_head, d_model]
   5 batched_matmul   self_attn                          [[B*n_h, 1, d_head], [B*n_h, d_head, T+1]] -> [[B*n_h, 1, T+1]]
   6 softmax          self_attn                          [[B, n_h, 1, T+1]] -> [[B, n_h, 1, T+1]]
   7 batched_matmul   self_attn                          [[B*n_h, 1, T+1], [B*n_h, T+1, d_head]] -> [[B*n_h, 1, d_head]]
   8 matmul           self_attn.o_proj                   [[B, n_h*d_head], [n_h*d_head, d_model]] -> [[B, d_model]]  w=[d_model, n_h*d_head]
   9 elementwise_add  model.layers.0                     [[B, 1, d_model], [B, 1, d_model]] -> [[B, 1, d_model]]
  10 rmsnorm          post_attention_layernorm           [[B, 1, d_model]] -> [[B, 1, d_model]]  w=[d_model]
  11 matmul           feed_forward.gate_proj             [[B, d_model], [d_model, d_ff]] -> [[B, d_ff]]  w=[d_ff, d_model]
  12 silu             feed_forward.activation_fn         [[B, 1, d_ff]] -> [[B, 1, d_ff]]
  13 matmul           feed_forward.up_proj               [[B, d_model], [d_model, d_ff]] -> [[B, d_ff]]  w=[d_ff, d_model]
  14 elementwise_mul  feed_forward                       [[B, 1, d_ff], [B, 1, d_ff]] -> [[B, 1, d_ff]]
  15 matmul           feed_forward.down_proj             [[B, d_ff], [d_ff, d_model]] -> [[B, d_model]]  w=[d_model, d_ff]
  16 elementwise_add  model.layers.0                     [[B, 1, d_model], [B, 1, d_model]] -> [[B, 1, d_model]]
```

**`block_type=attn+MoE`  repeat=12  layers=`1,5,9,13,17,21,25,29,33,37,41,45`**  (24행)

```
  17 rmsnorm          input_layernorm                    [[B, 1, d_model]] -> [[B, 1, d_model]]  w=[d_model]
  18 matmul           self_attn.q_proj                   [[B, d_model], [d_model, n_h*d_head]] -> [[B, n_h*d_head]]  w=[n_h*d_head, d_model]
  19 matmul           self_attn.k_proj                   [[B, d_model], [d_model, n_kv*d_head]] -> [[B, n_kv*d_head]]  w=[n_kv*d_head, d_model]
  20 matmul           self_attn.v_proj                   [[B, d_model], [d_model, n_kv*d_head]] -> [[B, n_kv*d_head]]  w=[n_kv*d_head, d_model]
  21 batched_matmul   self_attn                          [[B*n_h, 1, d_head], [B*n_h, d_head, T+1]] -> [[B*n_h, 1, T+1]]
  22 softmax          self_attn                          [[B, n_h, 1, T+1]] -> [[B, n_h, 1, T+1]]
  23 batched_matmul   self_attn                          [[B*n_h, 1, T+1], [B*n_h, T+1, d_head]] -> [[B*n_h, 1, d_head]]
  24 matmul           self_attn.o_proj                   [[B, n_h*d_head], [n_h*d_head, d_model]] -> [[B, d_model]]  w=[d_model, n_h*d_head]
  25 elementwise_add  model.layers.1                     [[B, 1, d_model], [B, 1, d_model]] -> [[B, 1, d_model]]
  26 rmsnorm          post_attention_layernorm           [[B, 1, d_model]] -> [[B, 1, d_model]]  w=[d_model]
  27 matmul           feed_forward.router                [[B, d_model], [d_model, E]] -> [[B, E]]  w=[E, d_model]
  28 sigmoid          feed_forward.router                [[B, E]] -> [[B, E]]
  29 elementwise_mul  feed_forward                       [[B*E, d_model], [B*E, 1]] -> [[B*E, d_model]]
  30 batched_matmul   feed_forward.experts               [[E, B, d_model], [E, d_model, 2*d_moe]] -> [[E, B, 2*d_moe]]  w=[E, d_model, 2*d_moe]
  31 silu             feed_forward.experts.act_fn        [[E, B, d_moe]] -> [[E, B, d_moe]]
  32 elementwise_mul  feed_forward.experts               [[E, B, d_moe], [E, B, d_moe]] -> [[E, B, d_moe]]
  33 batched_matmul   feed_forward.experts               [[E, B, d_moe], [E, d_moe, d_model]] -> [[E, B, d_model]]  w=[E, d_moe, d_model]
  34 matmul           feed_forward.shared_expert.gate_pr [[B, d_model], [d_model, d_moe]] -> [[B, d_moe]]  w=[d_moe, d_model]
  35 silu             feed_forward.shared_expert.activat [[B, d_moe]] -> [[B, d_moe]]
  36 matmul           feed_forward.shared_expert.up_proj [[B, d_model], [d_model, d_moe]] -> [[B, d_moe]]  w=[d_moe, d_model]
  37 elementwise_mul  feed_forward.shared_expert         [[B, d_moe], [B, d_moe]] -> [[B, d_moe]]
  38 matmul           feed_forward.shared_expert.down_pr [[B, d_moe], [d_moe, d_model]] -> [[B, d_model]]  w=[d_model, d_moe]
  39 sum              feed_forward                       [[E, B, d_model]] -> [[B, d_model]]
  40 elementwise_add  model.layers.1                     [[B, 1, d_model], [B, 1, d_model]] -> [[B, 1, d_model]]
```

**`block_type=attn+MoE`  repeat=12  layers=`3,7,11,15,19,23,27,31,35,39,43,47`**  (24행)

```
  41 rmsnorm          input_layernorm                    [[B, 1, d_model]] -> [[B, 1, d_model]]  w=[d_model]
  42 matmul           self_attn.q_proj                   [[B, d_model], [d_model, n_h*d_head]] -> [[B, n_h*d_head]]  w=[n_h*d_head, d_model]
  43 matmul           self_attn.k_proj                   [[B, d_model], [d_model, n_kv*d_head]] -> [[B, n_kv*d_head]]  w=[n_kv*d_head, d_model]
  44 matmul           self_attn.v_proj                   [[B, d_model], [d_model, n_kv*d_head]] -> [[B, n_kv*d_head]]  w=[n_kv*d_head, d_model]
  45 batched_matmul   self_attn                          [[B*n_h, 1, d_head], [B*n_h, d_head, T+1]] -> [[B*n_h, 1, T+1]]
  46 softmax          self_attn                          [[B, n_h, 1, T+1]] -> [[B, n_h, 1, T+1]]
  47 batched_matmul   self_attn                          [[B*n_h, 1, T+1], [B*n_h, T+1, d_head]] -> [[B*n_h, 1, d_head]]
  48 matmul           self_attn.o_proj                   [[B, n_h*d_head], [n_h*d_head, d_model]] -> [[B, d_model]]  w=[d_model, n_h*d_head]
  49 elementwise_add  model.layers.3                     [[B, 1, d_model], [B, 1, d_model]] -> [[B, 1, d_model]]
  50 rmsnorm          post_attention_layernorm           [[B, 1, d_model]] -> [[B, 1, d_model]]  w=[d_model]
  51 matmul           feed_forward.router                [[B, d_model], [d_model, E]] -> [[B, E]]  w=[E, d_model]
  52 sigmoid          feed_forward.router                [[B, E]] -> [[B, E]]
  53 elementwise_mul  feed_forward                       [[B*E, d_model], [B*E, 1]] -> [[B*E, d_model]]
  54 batched_matmul   feed_forward.experts               [[E, B, d_model], [E, d_model, 2*d_moe]] -> [[E, B, 2*d_moe]]  w=[E, d_model, 2*d_moe]
  55 silu             feed_forward.experts.act_fn        [[E, B, d_moe]] -> [[E, B, d_moe]]
  56 elementwise_mul  feed_forward.experts               [[E, B, d_moe], [E, B, d_moe]] -> [[E, B, d_moe]]
  57 batched_matmul   feed_forward.experts               [[E, B, d_moe], [E, d_moe, d_model]] -> [[E, B, d_model]]  w=[E, d_moe, d_model]
  58 matmul           feed_forward.shared_expert.gate_pr [[B, d_model], [d_model, d_moe]] -> [[B, d_moe]]  w=[d_moe, d_model]
  59 silu             feed_forward.shared_expert.activat [[B, d_moe]] -> [[B, d_moe]]
  60 matmul           feed_forward.shared_expert.up_proj [[B, d_model], [d_model, d_moe]] -> [[B, d_moe]]  w=[d_moe, d_model]
  61 elementwise_mul  feed_forward.shared_expert         [[B, d_moe], [B, d_moe]] -> [[B, d_moe]]
  62 matmul           feed_forward.shared_expert.down_pr [[B, d_moe], [d_moe, d_model]] -> [[B, d_model]]  w=[d_model, d_moe]
  63 sum              feed_forward                       [[E, B, d_model]] -> [[B, d_model]]
  64 elementwise_add  model.layers.3                     [[B, 1, d_model], [B, 1, d_model]] -> [[B, 1, d_model]]
```

**`block_type=norm`  repeat=1  layers=`-`**  (1행)

```
  65 rmsnorm          model.norm                         [[B, 1, d_model]] -> [[B, 1, d_model]]  w=[d_model]
```

**`block_type=head`  repeat=1  layers=`-`**  (1행)

```
  66 matmul           lm_head                            [[B, d_model], [d_model, V]] -> [[B, V]]  w=[V, d_model]
```

## openai/gpt-oss-20b

발행 좌표 **B=3, T=264**. 심볼 표:

```
B=3, E=32, T=264, V=201088, ctx=131072, d_head=64, d_model=2880, d_moe=2880, k=4, n_h=64, n_kv=8, w_local=128
```

### prefill — 템플릿 5개 / 전체 47행

**`block_type=embed`  repeat=1  layers=`-`**  (1행)

```
   0 embedding        model.embed_tokens                 [[V, d_model], [B, T]] -> [[B, T, d_model]]  w=[V, d_model]
```

**`block_type=attn+MoE`  repeat=12  layers=`0,2,4,6,8,10,12,14,16,18,20,22`**  (22행)

```
   1 rmsnorm          input_layernorm                    [[B, T, d_model]] -> [[B, T, d_model]]  w=[d_model]
   2 linear           self_attn.q_proj                   [[n_h*d_head], [B*T, d_model], [d_model, n_h*d_head]] -> [[B*T, n_h*d_head]]  w=[n_h*d_head, d_model]
   3 linear           self_attn.k_proj                   [[n_kv*d_head], [B*T, d_model], [d_model, n_kv*d_head]] -> [[B*T, n_kv*d_head]]  w=[n_kv*d_head, d_model]
   4 linear           self_attn.v_proj                   [[n_kv*d_head], [B*T, d_model], [d_model, n_kv*d_head]] -> [[B*T, n_kv*d_head]]  w=[n_kv*d_head, d_model]
   5 batched_matmul   self_attn                          [[B*n_h, T, d_head], [B*n_h, d_head, T]] -> [[B*n_h, T, T]]
   6 softmax          self_attn                          [[B, n_h, T, T+1]] -> [[B, n_h, T, T+1]]
   7 batched_matmul   self_attn                          [[B*n_h, T, T], [B*n_h, T, d_head]] -> [[B*n_h, T, d_head]]
   8 linear           self_attn.o_proj                   [[d_model], [B*T, n_h*d_head], [n_h*d_head, d_model]] -> [[B*T, d_model]]  w=[d_model, n_h*d_head]
   9 elementwise_add  model.layers.0                     [[B, T, d_model], [B, T, d_model]] -> [[B, T, d_model]]
  10 rmsnorm          post_attention_layernorm           [[B, T, d_model]] -> [[B, T, d_model]]  w=[d_model]
  11 linear           mlp.router                         [[E], [B*T, d_model], [d_model, E]] -> [[B*T, E]]  w=[E, d_model]
  12 softmax          mlp.router                         [[B*T, k]] -> [[B*T, k]]
  13 grouped_matmul   mlp.experts                        [[B*k*T, d_model], [E, d_model, 2*d_moe], [E]] -> [[B*k*T, 2*d_moe]]  w=[E, d_model, 2*d_moe]
  14 elementwise_mul  mlp.experts                        [[B*k*T, d_moe]] -> [[B*k*T, d_moe]]
  15 sigmoid          mlp.experts                        [[B*k*T, d_moe]] -> [[B*k*T, d_moe]]
  16 elementwise_mul  mlp.experts                        [[B*k*T, d_moe], [B*k*T, d_moe]] -> [[B*k*T, d_moe]]
  17 elementwise_add  mlp.experts                        [[B*k*T, d_moe]] -> [[B*k*T, d_moe]]
  18 elementwise_mul  mlp.experts                        [[B*k*T, d_moe], [B*k*T, d_moe]] -> [[B*k*T, d_moe]]
  19 grouped_matmul   mlp.experts                        [[B*k*T, d_moe], [E, d_moe, d_model], [E]] -> [[B*k*T, d_model]]  w=[E, d_moe, d_model]
  20 elementwise_mul  mlp.experts                        [[B*k*T, d_model], [B*k*T, 1]] -> [[B*k*T, d_model]]
  21 sum              mlp.experts                        [[B*T, k, d_model]] -> [[B*T, d_model]]
  22 elementwise_add  model.layers.0                     [[B, T, d_model], [B, T, d_model]] -> [[B, T, d_model]]
```

**`block_type=attn+MoE`  repeat=12  layers=`1,3,5,7,9,11,13,15,17,19,21,23`**  (22행)

```
  23 rmsnorm          input_layernorm                    [[B, T, d_model]] -> [[B, T, d_model]]  w=[d_model]
  24 linear           self_attn.q_proj                   [[n_h*d_head], [B*T, d_model], [d_model, n_h*d_head]] -> [[B*T, n_h*d_head]]  w=[n_h*d_head, d_model]
  25 linear           self_attn.k_proj                   [[n_kv*d_head], [B*T, d_model], [d_model, n_kv*d_head]] -> [[B*T, n_kv*d_head]]  w=[n_kv*d_head, d_model]
  26 linear           self_attn.v_proj                   [[n_kv*d_head], [B*T, d_model], [d_model, n_kv*d_head]] -> [[B*T, n_kv*d_head]]  w=[n_kv*d_head, d_model]
  27 batched_matmul   self_attn                          [[B*n_h, T, d_head], [B*n_h, d_head, T]] -> [[B*n_h, T, T]]
  28 softmax          self_attn                          [[B, n_h, T, T+1]] -> [[B, n_h, T, T+1]]
  29 batched_matmul   self_attn                          [[B*n_h, T, T], [B*n_h, T, d_head]] -> [[B*n_h, T, d_head]]
  30 linear           self_attn.o_proj                   [[d_model], [B*T, n_h*d_head], [n_h*d_head, d_model]] -> [[B*T, d_model]]  w=[d_model, n_h*d_head]
  31 elementwise_add  model.layers.1                     [[B, T, d_model], [B, T, d_model]] -> [[B, T, d_model]]
  32 rmsnorm          post_attention_layernorm           [[B, T, d_model]] -> [[B, T, d_model]]  w=[d_model]
  33 linear           mlp.router                         [[E], [B*T, d_model], [d_model, E]] -> [[B*T, E]]  w=[E, d_model]
  34 softmax          mlp.router                         [[B*T, k]] -> [[B*T, k]]
  35 grouped_matmul   mlp.experts                        [[B*k*T, d_model], [E, d_model, 2*d_moe], [E]] -> [[B*k*T, 2*d_moe]]  w=[E, d_model, 2*d_moe]
  36 elementwise_mul  mlp.experts                        [[B*k*T, d_moe]] -> [[B*k*T, d_moe]]
  37 sigmoid          mlp.experts                        [[B*k*T, d_moe]] -> [[B*k*T, d_moe]]
  38 elementwise_mul  mlp.experts                        [[B*k*T, d_moe], [B*k*T, d_moe]] -> [[B*k*T, d_moe]]
  39 elementwise_add  mlp.experts                        [[B*k*T, d_moe]] -> [[B*k*T, d_moe]]
  40 elementwise_mul  mlp.experts                        [[B*k*T, d_moe], [B*k*T, d_moe]] -> [[B*k*T, d_moe]]
  41 grouped_matmul   mlp.experts                        [[B*k*T, d_moe], [E, d_moe, d_model], [E]] -> [[B*k*T, d_model]]  w=[E, d_moe, d_model]
  42 elementwise_mul  mlp.experts                        [[B*k*T, d_model], [B*k*T, 1]] -> [[B*k*T, d_model]]
  43 sum              mlp.experts                        [[B*T, k, d_model]] -> [[B*T, d_model]]
  44 elementwise_add  model.layers.1                     [[B, T, d_model], [B, T, d_model]] -> [[B, T, d_model]]
```

**`block_type=norm`  repeat=1  layers=`-`**  (1행)

```
  45 rmsnorm          model.norm                         [[B, T, d_model]] -> [[B, T, d_model]]  w=[d_model]
```

**`block_type=head`  repeat=1  layers=`-`**  (1행)

```
  46 matmul           lm_head                            [[B*T, d_model], [d_model, V]] -> [[B*T, V]]  w=[V, d_model]
```

### decode — 템플릿 5개 / 전체 47행

**`block_type=embed`  repeat=1  layers=`-`**  (1행)

```
   0 embedding        model.embed_tokens                 [[V, d_model], [B, 1]] -> [[B, 1, d_model]]  w=[V, d_model]
```

**`block_type=attn+MoE`  repeat=12  layers=`0,2,4,6,8,10,12,14,16,18,20,22`**  (22행)

```
   1 rmsnorm          input_layernorm                    [[B, 1, d_model]] -> [[B, 1, d_model]]  w=[d_model]
   2 linear           self_attn.q_proj                   [[n_h*d_head], [B, d_model], [d_model, n_h*d_head]] -> [[B, n_h*d_head]]  w=[n_h*d_head, d_model]
   3 linear           self_attn.k_proj                   [[n_kv*d_head], [B, d_model], [d_model, n_kv*d_head]] -> [[B, n_kv*d_head]]  w=[n_kv*d_head, d_model]
   4 linear           self_attn.v_proj                   [[n_kv*d_head], [B, d_model], [d_model, n_kv*d_head]] -> [[B, n_kv*d_head]]  w=[n_kv*d_head, d_model]
   5 batched_matmul   self_attn                          [[B*n_h, 1, d_head], [B*n_h, d_head, w_local]] -> [[B*n_h, 1, w_local]]
   6 softmax          self_attn                          [[B, n_h, 1, w_local+n_sink]] -> [[B, n_h, 1, w_local+n_sink]]
   7 batched_matmul   self_attn                          [[B*n_h, 1, w_local], [B*n_h, w_local, d_head]] -> [[B*n_h, 1, d_head]]
   8 linear           self_attn.o_proj                   [[d_model], [B, n_h*d_head], [n_h*d_head, d_model]] -> [[B, d_model]]  w=[d_model, n_h*d_head]
   9 elementwise_add  model.layers.0                     [[B, 1, d_model], [B, 1, d_model]] -> [[B, 1, d_model]]
  10 rmsnorm          post_attention_layernorm           [[B, 1, d_model]] -> [[B, 1, d_model]]  w=[d_model]
  11 linear           mlp.router                         [[E], [B, d_model], [d_model, E]] -> [[B, E]]  w=[E, d_model]
  12 softmax          mlp.router                         [[B, k]] -> [[B, k]]
  13 grouped_matmul   mlp.experts                        [[B*k, d_model], [E, d_model, 2*d_moe], [E]] -> [[B*k, 2*d_moe]]  w=[E, d_model, 2*d_moe]
  14 elementwise_mul  mlp.experts                        [[B*k, d_moe]] -> [[B*k, d_moe]]
  15 sigmoid          mlp.experts                        [[B*k, d_moe]] -> [[B*k, d_moe]]
  16 elementwise_mul  mlp.experts                        [[B*k, d_moe], [B*k, d_moe]] -> [[B*k, d_moe]]
  17 elementwise_add  mlp.experts                        [[B*k, d_moe]] -> [[B*k, d_moe]]
  18 elementwise_mul  mlp.experts                        [[B*k, d_moe], [B*k, d_moe]] -> [[B*k, d_moe]]
  19 grouped_matmul   mlp.experts                        [[B*k, d_moe], [E, d_moe, d_model], [E]] -> [[B*k, d_model]]  w=[E, d_moe, d_model]
  20 elementwise_mul  mlp.experts                        [[B*k, d_model], [B*k, 1]] -> [[B*k, d_model]]
  21 sum              mlp.experts                        [[B, k, d_model]] -> [[B, d_model]]
  22 elementwise_add  model.layers.0                     [[B, 1, d_model], [B, 1, d_model]] -> [[B, 1, d_model]]
```

**`block_type=attn+MoE`  repeat=12  layers=`1,3,5,7,9,11,13,15,17,19,21,23`**  (22행)

```
  23 rmsnorm          input_layernorm                    [[B, 1, d_model]] -> [[B, 1, d_model]]  w=[d_model]
  24 linear           self_attn.q_proj                   [[n_h*d_head], [B, d_model], [d_model, n_h*d_head]] -> [[B, n_h*d_head]]  w=[n_h*d_head, d_model]
  25 linear           self_attn.k_proj                   [[n_kv*d_head], [B, d_model], [d_model, n_kv*d_head]] -> [[B, n_kv*d_head]]  w=[n_kv*d_head, d_model]
  26 linear           self_attn.v_proj                   [[n_kv*d_head], [B, d_model], [d_model, n_kv*d_head]] -> [[B, n_kv*d_head]]  w=[n_kv*d_head, d_model]
  27 batched_matmul   self_attn                          [[B*n_h, 1, d_head], [B*n_h, d_head, T+1]] -> [[B*n_h, 1, T+1]]
  28 softmax          self_attn                          [[B, n_h, 1, (T+1)+n_sink]] -> [[B, n_h, 1, (T+1)+n_sink]]
  29 batched_matmul   self_attn                          [[B*n_h, 1, T+1], [B*n_h, T+1, d_head]] -> [[B*n_h, 1, d_head]]
  30 linear           self_attn.o_proj                   [[d_model], [B, n_h*d_head], [n_h*d_head, d_model]] -> [[B, d_model]]  w=[d_model, n_h*d_head]
  31 elementwise_add  model.layers.1                     [[B, 1, d_model], [B, 1, d_model]] -> [[B, 1, d_model]]
  32 rmsnorm          post_attention_layernorm           [[B, 1, d_model]] -> [[B, 1, d_model]]  w=[d_model]
  33 linear           mlp.router                         [[E], [B, d_model], [d_model, E]] -> [[B, E]]  w=[E, d_model]
  34 softmax          mlp.router                         [[B, k]] -> [[B, k]]
  35 grouped_matmul   mlp.experts                        [[B*k, d_model], [E, d_model, 2*d_moe], [E]] -> [[B*k, 2*d_moe]]  w=[E, d_model, 2*d_moe]
  36 elementwise_mul  mlp.experts                        [[B*k, d_moe]] -> [[B*k, d_moe]]
  37 sigmoid          mlp.experts                        [[B*k, d_moe]] -> [[B*k, d_moe]]
  38 elementwise_mul  mlp.experts                        [[B*k, d_moe], [B*k, d_moe]] -> [[B*k, d_moe]]
  39 elementwise_add  mlp.experts                        [[B*k, d_moe]] -> [[B*k, d_moe]]
  40 elementwise_mul  mlp.experts                        [[B*k, d_moe], [B*k, d_moe]] -> [[B*k, d_moe]]
  41 grouped_matmul   mlp.experts                        [[B*k, d_moe], [E, d_moe, d_model], [E]] -> [[B*k, d_model]]  w=[E, d_moe, d_model]
  42 elementwise_mul  mlp.experts                        [[B*k, d_model], [B*k, 1]] -> [[B*k, d_model]]
  43 sum              mlp.experts                        [[B, k, d_model]] -> [[B, d_model]]
  44 elementwise_add  model.layers.1                     [[B, 1, d_model], [B, 1, d_model]] -> [[B, 1, d_model]]
```

**`block_type=norm`  repeat=1  layers=`-`**  (1행)

```
  45 rmsnorm          model.norm                         [[B, 1, d_model]] -> [[B, 1, d_model]]  w=[d_model]
```

**`block_type=head`  repeat=1  layers=`-`**  (1행)

```
  46 matmul           lm_head                            [[B, d_model], [d_model, V]] -> [[B, V]]  w=[V, d_model]
```

## openai/gpt-oss-120b

발행 좌표 **B=3, T=264**. 심볼 표:

```
B=3, E=128, T=264, V=201088, ctx=131072, d_head=64, d_model=2880, d_moe=2880, k=4, n_h=64, n_kv=8, w_local=128
```

### prefill — 템플릿 5개 / 전체 47행

**`block_type=embed`  repeat=1  layers=`-`**  (1행)

```
   0 embedding        model.embed_tokens                 [[V, d_model], [B, T]] -> [[B, T, d_model]]  w=[V, d_model]
```

**`block_type=attn+MoE`  repeat=18  layers=`0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34`**  (22행)

```
   1 rmsnorm          input_layernorm                    [[B, T, d_model]] -> [[B, T, d_model]]  w=[d_model]
   2 linear           self_attn.q_proj                   [[n_h*d_head], [B*T, d_model], [d_model, n_h*d_head]] -> [[B*T, n_h*d_head]]  w=[n_h*d_head, d_model]
   3 linear           self_attn.k_proj                   [[n_kv*d_head], [B*T, d_model], [d_model, n_kv*d_head]] -> [[B*T, n_kv*d_head]]  w=[n_kv*d_head, d_model]
   4 linear           self_attn.v_proj                   [[n_kv*d_head], [B*T, d_model], [d_model, n_kv*d_head]] -> [[B*T, n_kv*d_head]]  w=[n_kv*d_head, d_model]
   5 batched_matmul   self_attn                          [[B*n_h, T, d_head], [B*n_h, d_head, T]] -> [[B*n_h, T, T]]
   6 softmax          self_attn                          [[B, n_h, T, T+1]] -> [[B, n_h, T, T+1]]
   7 batched_matmul   self_attn                          [[B*n_h, T, T], [B*n_h, T, d_head]] -> [[B*n_h, T, d_head]]
   8 linear           self_attn.o_proj                   [[d_model], [B*T, n_h*d_head], [n_h*d_head, d_model]] -> [[B*T, d_model]]  w=[d_model, n_h*d_head]
   9 elementwise_add  model.layers.0                     [[B, T, d_model], [B, T, d_model]] -> [[B, T, d_model]]
  10 rmsnorm          post_attention_layernorm           [[B, T, d_model]] -> [[B, T, d_model]]  w=[d_model]
  11 linear           mlp.router                         [[E], [B*T, d_model], [d_model, E]] -> [[B*T, E]]  w=[E, d_model]
  12 softmax          mlp.router                         [[B*T, k]] -> [[B*T, k]]
  13 grouped_matmul   mlp.experts                        [[B*k*T, d_model], [E, d_model, 2*d_moe], [E]] -> [[B*k*T, 2*d_moe]]  w=[E, d_model, 2*d_moe]
  14 elementwise_mul  mlp.experts                        [[B*k*T, d_moe]] -> [[B*k*T, d_moe]]
  15 sigmoid          mlp.experts                        [[B*k*T, d_moe]] -> [[B*k*T, d_moe]]
  16 elementwise_mul  mlp.experts                        [[B*k*T, d_moe], [B*k*T, d_moe]] -> [[B*k*T, d_moe]]
  17 elementwise_add  mlp.experts                        [[B*k*T, d_moe]] -> [[B*k*T, d_moe]]
  18 elementwise_mul  mlp.experts                        [[B*k*T, d_moe], [B*k*T, d_moe]] -> [[B*k*T, d_moe]]
  19 grouped_matmul   mlp.experts                        [[B*k*T, d_moe], [E, d_moe, d_model], [E]] -> [[B*k*T, d_model]]  w=[E, d_moe, d_model]
  20 elementwise_mul  mlp.experts                        [[B*k*T, d_model], [B*k*T, 1]] -> [[B*k*T, d_model]]
  21 sum              mlp.experts                        [[B*T, k, d_model]] -> [[B*T, d_model]]
  22 elementwise_add  model.layers.0                     [[B, T, d_model], [B, T, d_model]] -> [[B, T, d_model]]
```

**`block_type=attn+MoE`  repeat=18  layers=`1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35`**  (22행)

```
  23 rmsnorm          input_layernorm                    [[B, T, d_model]] -> [[B, T, d_model]]  w=[d_model]
  24 linear           self_attn.q_proj                   [[n_h*d_head], [B*T, d_model], [d_model, n_h*d_head]] -> [[B*T, n_h*d_head]]  w=[n_h*d_head, d_model]
  25 linear           self_attn.k_proj                   [[n_kv*d_head], [B*T, d_model], [d_model, n_kv*d_head]] -> [[B*T, n_kv*d_head]]  w=[n_kv*d_head, d_model]
  26 linear           self_attn.v_proj                   [[n_kv*d_head], [B*T, d_model], [d_model, n_kv*d_head]] -> [[B*T, n_kv*d_head]]  w=[n_kv*d_head, d_model]
  27 batched_matmul   self_attn                          [[B*n_h, T, d_head], [B*n_h, d_head, T]] -> [[B*n_h, T, T]]
  28 softmax          self_attn                          [[B, n_h, T, T+1]] -> [[B, n_h, T, T+1]]
  29 batched_matmul   self_attn                          [[B*n_h, T, T], [B*n_h, T, d_head]] -> [[B*n_h, T, d_head]]
  30 linear           self_attn.o_proj                   [[d_model], [B*T, n_h*d_head], [n_h*d_head, d_model]] -> [[B*T, d_model]]  w=[d_model, n_h*d_head]
  31 elementwise_add  model.layers.1                     [[B, T, d_model], [B, T, d_model]] -> [[B, T, d_model]]
  32 rmsnorm          post_attention_layernorm           [[B, T, d_model]] -> [[B, T, d_model]]  w=[d_model]
  33 linear           mlp.router                         [[E], [B*T, d_model], [d_model, E]] -> [[B*T, E]]  w=[E, d_model]
  34 softmax          mlp.router                         [[B*T, k]] -> [[B*T, k]]
  35 grouped_matmul   mlp.experts                        [[B*k*T, d_model], [E, d_model, 2*d_moe], [E]] -> [[B*k*T, 2*d_moe]]  w=[E, d_model, 2*d_moe]
  36 elementwise_mul  mlp.experts                        [[B*k*T, d_moe]] -> [[B*k*T, d_moe]]
  37 sigmoid          mlp.experts                        [[B*k*T, d_moe]] -> [[B*k*T, d_moe]]
  38 elementwise_mul  mlp.experts                        [[B*k*T, d_moe], [B*k*T, d_moe]] -> [[B*k*T, d_moe]]
  39 elementwise_add  mlp.experts                        [[B*k*T, d_moe]] -> [[B*k*T, d_moe]]
  40 elementwise_mul  mlp.experts                        [[B*k*T, d_moe], [B*k*T, d_moe]] -> [[B*k*T, d_moe]]
  41 grouped_matmul   mlp.experts                        [[B*k*T, d_moe], [E, d_moe, d_model], [E]] -> [[B*k*T, d_model]]  w=[E, d_moe, d_model]
  42 elementwise_mul  mlp.experts                        [[B*k*T, d_model], [B*k*T, 1]] -> [[B*k*T, d_model]]
  43 sum              mlp.experts                        [[B*T, k, d_model]] -> [[B*T, d_model]]
  44 elementwise_add  model.layers.1                     [[B, T, d_model], [B, T, d_model]] -> [[B, T, d_model]]
```

**`block_type=norm`  repeat=1  layers=`-`**  (1행)

```
  45 rmsnorm          model.norm                         [[B, T, d_model]] -> [[B, T, d_model]]  w=[d_model]
```

**`block_type=head`  repeat=1  layers=`-`**  (1행)

```
  46 matmul           lm_head                            [[B*T, d_model], [d_model, V]] -> [[B*T, V]]  w=[V, d_model]
```

### decode — 템플릿 5개 / 전체 47행

**`block_type=embed`  repeat=1  layers=`-`**  (1행)

```
   0 embedding        model.embed_tokens                 [[V, d_model], [B, 1]] -> [[B, 1, d_model]]  w=[V, d_model]
```

**`block_type=attn+MoE`  repeat=18  layers=`0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34`**  (22행)

```
   1 rmsnorm          input_layernorm                    [[B, 1, d_model]] -> [[B, 1, d_model]]  w=[d_model]
   2 linear           self_attn.q_proj                   [[n_h*d_head], [B, d_model], [d_model, n_h*d_head]] -> [[B, n_h*d_head]]  w=[n_h*d_head, d_model]
   3 linear           self_attn.k_proj                   [[n_kv*d_head], [B, d_model], [d_model, n_kv*d_head]] -> [[B, n_kv*d_head]]  w=[n_kv*d_head, d_model]
   4 linear           self_attn.v_proj                   [[n_kv*d_head], [B, d_model], [d_model, n_kv*d_head]] -> [[B, n_kv*d_head]]  w=[n_kv*d_head, d_model]
   5 batched_matmul   self_attn                          [[B*n_h, 1, d_head], [B*n_h, d_head, w_local]] -> [[B*n_h, 1, w_local]]
   6 softmax          self_attn                          [[B, n_h, 1, w_local+n_sink]] -> [[B, n_h, 1, w_local+n_sink]]
   7 batched_matmul   self_attn                          [[B*n_h, 1, w_local], [B*n_h, w_local, d_head]] -> [[B*n_h, 1, d_head]]
   8 linear           self_attn.o_proj                   [[d_model], [B, n_h*d_head], [n_h*d_head, d_model]] -> [[B, d_model]]  w=[d_model, n_h*d_head]
   9 elementwise_add  model.layers.0                     [[B, 1, d_model], [B, 1, d_model]] -> [[B, 1, d_model]]
  10 rmsnorm          post_attention_layernorm           [[B, 1, d_model]] -> [[B, 1, d_model]]  w=[d_model]
  11 linear           mlp.router                         [[E], [B, d_model], [d_model, E]] -> [[B, E]]  w=[E, d_model]
  12 softmax          mlp.router                         [[B, k]] -> [[B, k]]
  13 grouped_matmul   mlp.experts                        [[B*k, d_model], [E, d_model, 2*d_moe], [E]] -> [[B*k, 2*d_moe]]  w=[E, d_model, 2*d_moe]
  14 elementwise_mul  mlp.experts                        [[B*k, d_moe]] -> [[B*k, d_moe]]
  15 sigmoid          mlp.experts                        [[B*k, d_moe]] -> [[B*k, d_moe]]
  16 elementwise_mul  mlp.experts                        [[B*k, d_moe], [B*k, d_moe]] -> [[B*k, d_moe]]
  17 elementwise_add  mlp.experts                        [[B*k, d_moe]] -> [[B*k, d_moe]]
  18 elementwise_mul  mlp.experts                        [[B*k, d_moe], [B*k, d_moe]] -> [[B*k, d_moe]]
  19 grouped_matmul   mlp.experts                        [[B*k, d_moe], [E, d_moe, d_model], [E]] -> [[B*k, d_model]]  w=[E, d_moe, d_model]
  20 elementwise_mul  mlp.experts                        [[B*k, d_model], [B*k, 1]] -> [[B*k, d_model]]
  21 sum              mlp.experts                        [[B, k, d_model]] -> [[B, d_model]]
  22 elementwise_add  model.layers.0                     [[B, 1, d_model], [B, 1, d_model]] -> [[B, 1, d_model]]
```

**`block_type=attn+MoE`  repeat=18  layers=`1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35`**  (22행)

```
  23 rmsnorm          input_layernorm                    [[B, 1, d_model]] -> [[B, 1, d_model]]  w=[d_model]
  24 linear           self_attn.q_proj                   [[n_h*d_head], [B, d_model], [d_model, n_h*d_head]] -> [[B, n_h*d_head]]  w=[n_h*d_head, d_model]
  25 linear           self_attn.k_proj                   [[n_kv*d_head], [B, d_model], [d_model, n_kv*d_head]] -> [[B, n_kv*d_head]]  w=[n_kv*d_head, d_model]
  26 linear           self_attn.v_proj                   [[n_kv*d_head], [B, d_model], [d_model, n_kv*d_head]] -> [[B, n_kv*d_head]]  w=[n_kv*d_head, d_model]
  27 batched_matmul   self_attn                          [[B*n_h, 1, d_head], [B*n_h, d_head, T+1]] -> [[B*n_h, 1, T+1]]
  28 softmax          self_attn                          [[B, n_h, 1, (T+1)+n_sink]] -> [[B, n_h, 1, (T+1)+n_sink]]
  29 batched_matmul   self_attn                          [[B*n_h, 1, T+1], [B*n_h, T+1, d_head]] -> [[B*n_h, 1, d_head]]
  30 linear           self_attn.o_proj                   [[d_model], [B, n_h*d_head], [n_h*d_head, d_model]] -> [[B, d_model]]  w=[d_model, n_h*d_head]
  31 elementwise_add  model.layers.1                     [[B, 1, d_model], [B, 1, d_model]] -> [[B, 1, d_model]]
  32 rmsnorm          post_attention_layernorm           [[B, 1, d_model]] -> [[B, 1, d_model]]  w=[d_model]
  33 linear           mlp.router                         [[E], [B, d_model], [d_model, E]] -> [[B, E]]  w=[E, d_model]
  34 softmax          mlp.router                         [[B, k]] -> [[B, k]]
  35 grouped_matmul   mlp.experts                        [[B*k, d_model], [E, d_model, 2*d_moe], [E]] -> [[B*k, 2*d_moe]]  w=[E, d_model, 2*d_moe]
  36 elementwise_mul  mlp.experts                        [[B*k, d_moe]] -> [[B*k, d_moe]]
  37 sigmoid          mlp.experts                        [[B*k, d_moe]] -> [[B*k, d_moe]]
  38 elementwise_mul  mlp.experts                        [[B*k, d_moe], [B*k, d_moe]] -> [[B*k, d_moe]]
  39 elementwise_add  mlp.experts                        [[B*k, d_moe]] -> [[B*k, d_moe]]
  40 elementwise_mul  mlp.experts                        [[B*k, d_moe], [B*k, d_moe]] -> [[B*k, d_moe]]
  41 grouped_matmul   mlp.experts                        [[B*k, d_moe], [E, d_moe, d_model], [E]] -> [[B*k, d_model]]  w=[E, d_moe, d_model]
  42 elementwise_mul  mlp.experts                        [[B*k, d_model], [B*k, 1]] -> [[B*k, d_model]]
  43 sum              mlp.experts                        [[B, k, d_model]] -> [[B, d_model]]
  44 elementwise_add  model.layers.1                     [[B, 1, d_model], [B, 1, d_model]] -> [[B, 1, d_model]]
```

**`block_type=norm`  repeat=1  layers=`-`**  (1행)

```
  45 rmsnorm          model.norm                         [[B, 1, d_model]] -> [[B, 1, d_model]]  w=[d_model]
```

**`block_type=head`  repeat=1  layers=`-`**  (1행)

```
  46 matmul           lm_head                            [[B, d_model], [d_model, V]] -> [[B, V]]  w=[V, d_model]
```

## deepseek-ai/DeepSeek-V4-Pro

발행 좌표 **B=3, T=2176**. 심볼 표:

```
B=3, E=384, T=2176, V=129280, c_I=128, c_q=1536, ctx=1048576, d_g=1024, d_head=512, d_model=7168, d_moe=3072, d_rope=64, g_o=16, k=6, k_I=1024, m_csa=4, m_hca=128, n_h=128, n_h_I=64, n_hc=4, w_local=128
```

### prefill — 템플릿 8개 / 전체 224행

**`block_type=embed`  repeat=1  layers=`-`**  (1행)

```
   0 embedding        model.embed_tokens                 [[V, d_model], [B, T]] -> [[B, T, d_model]]  w=[V, d_model]
```

**`block_type=MLA+MoE`  repeat=2  layers=`0-1`**  (50행)

```
   1 rmsnorm          attn_hc.input_norm                 [[B, T, n_hc*d_model]] -> [[B, T, n_hc*d_model]]
   2 matmul           attn_hc                            [[B*T, n_hc*d_model], [n_hc*d_model, (2+n_hc)*n_hc]] -> [[B*T, (2+n_hc)*n_hc]]  w=[(2+n_hc)*n_hc, n_hc*d_model]
   3 sigmoid          attn_hc                            [[B, T, n_hc]] -> [[B, T, n_hc]]
   4 sigmoid          attn_hc                            [[B, T, n_hc]] -> [[B, T, n_hc]]
   5 softmax          attn_hc                            [[B, T, n_hc, n_hc]] -> [[B, T, n_hc, n_hc]]
   6 elementwise_mul  attn_hc                            [[B, T, n_hc, 1], [B, T, n_hc, d_model]] -> [[B, T, n_hc, d_model]]
   7 sum              attn_hc                            [[B, T, n_hc, d_model]] -> [[B, T, d_model]]
   8 rmsnorm          input_layernorm                    [[B, T, d_model]] -> [[B, T, d_model]]  w=[d_model]
   9 matmul           self_attn.q_a_proj                 [[B*T, d_model], [d_model, c_q]] -> [[B*T, c_q]]  w=[c_q, d_model]
  10 rmsnorm          self_attn.q_a_norm                 [[B, T, c_q]] -> [[B, T, c_q]]  w=[c_q]
  11 matmul           self_attn.q_b_proj                 [[B*T, c_q], [c_q, n_h*d_head]] -> [[B*T, n_h*d_head]]  w=[n_h*d_head, c_q]
  12 rmsnorm          self_attn.q_b_norm                 [[B, n_h, T, d_head]] -> [[B, n_h, T, d_head]]
  13 matmul           self_attn.kv_proj                  [[B*T, d_model], [d_model, d_head]] -> [[B*T, d_head]]  w=[d_head, d_model]
  14 rmsnorm          self_attn.kv_norm                  [[B, T, d_head]] -> [[B, T, d_head]]  w=[d_head]
  15 matmul           self_attn.compressor.kv_proj       [[B*T, d_model], [d_model, d_head]] -> [[B*T, d_head]]  w=[d_head, d_model]
  16 matmul           self_attn.compressor.gate_proj     [[B*T, d_model], [d_model, d_head]] -> [[B*T, d_head]]  w=[d_head, d_model]
  17 softmax          self_attn.compressor               [[B, T/m_hca, m_hca, d_head]] -> [[B, T/m_hca, m_hca, d_head]]
  18 rmsnorm          self_attn.compressor.kv_norm       [[B, T/m_hca, d_head]] -> [[B, T/m_hca, d_head]]  w=[d_head]
  19 batched_matmul   self_attn                          [[B*n_h, T, d_head], [B*n_h, d_head, T+T/m_hca]] -> [[B*n_h, T, T+T/m_hca]]
  20 softmax          self_attn                          [[B, n_h, T, T+T/m_hca+1]] -> [[B, n_h, T, T+T/m_hca+1]]
  21 batched_matmul   self_attn                          [[B*n_h, T, T+T/m_hca], [B*n_h, T+T/m_hca, d_head]] -> [[B*n_h, T, d_head]]
  22 batched_matmul   self_attn.o_a_proj                 [[g_o, B*T, n_h*d_head/g_o], [g_o, n_h*d_head/g_o, d_g]] -> [[g_o, B*T, d_g]]  w=[g_o*d_g, n_h*d_head/g_o]
  23 matmul           self_attn.o_b_proj                 [[B*T, g_o*d_g], [g_o*d_g, d_model]] -> [[B*T, d_model]]  w=[d_model, g_o*d_g]
  24 elementwise_mul  model.layers.0                     [[B, T, n_hc, 1], [B, T, 1, d_model]] -> [[B, T, n_hc, d_model]]
  25 batched_matmul   model.layers.0                     [[B*T, n_hc, n_hc], [B*T, n_hc, d_model]] -> [[B*T, n_hc, d_model]]
  26 elementwise_add  model.layers.0                     [[B, T, n_hc, d_model], [B, T, n_hc, d_model]] -> [[B, T, n_hc, d_model]]
  27 rmsnorm          ffn_hc.input_norm                  [[B, T, n_hc*d_model]] -> [[B, T, n_hc*d_model]]
  28 matmul           ffn_hc                             [[B*T, n_hc*d_model], [n_hc*d_model, (2+n_hc)*n_hc]] -> [[B*T, (2+n_hc)*n_hc]]  w=[(2+n_hc)*n_hc, n_hc*d_model]
  29 sigmoid          ffn_hc                             [[B, T, n_hc]] -> [[B, T, n_hc]]
  30 sigmoid          ffn_hc                             [[B, T, n_hc]] -> [[B, T, n_hc]]
  31 softmax          ffn_hc                             [[B, T, n_hc, n_hc]] -> [[B, T, n_hc, n_hc]]
  32 elementwise_mul  ffn_hc                             [[B, T, n_hc, 1], [B, T, n_hc, d_model]] -> [[B, T, n_hc, d_model]]
  33 sum              ffn_hc                             [[B, T, n_hc, d_model]] -> [[B, T, d_model]]
  34 rmsnorm          post_attention_layernorm           [[B, T, d_model]] -> [[B, T, d_model]]  w=[d_model]
  35 matmul           mlp.gate                           [[B*T, d_model], [d_model, E]] -> [[B*T, E]]  w=[E, d_model]
  36 grouped_matmul   mlp.experts                        [[B*k*T, d_model], [E, d_model, 2*d_moe], [E]] -> [[B*k*T, 2*d_moe]]  w=[E, 2*d_moe, d_model]
  37 silu             mlp.experts.act_fn                 [[B*k*T, d_moe]] -> [[B*k*T, d_moe]]
  38 elementwise_mul  mlp.experts                        [[B*k*T, d_moe], [B*k*T, d_moe]] -> [[B*k*T, d_moe]]
  39 grouped_matmul   mlp.experts                        [[B*k*T, d_moe], [E, d_moe, d_model], [E]] -> [[B*k*T, d_model]]  w=[E, d_model, d_moe]
  40 elementwise_mul  mlp.experts                        [[B*k*T, d_model], [B*k*T, 1]] -> [[B*k*T, d_model]]
  41 sum              mlp.experts                        [[B*T, k, d_model]] -> [[B*T, d_model]]
  42 matmul           mlp.shared_experts.gate_proj       [[B*T, d_model], [d_model, d_moe]] -> [[B*T, d_moe]]  w=[d_moe, d_model]
  43 matmul           mlp.shared_experts.up_proj         [[B*T, d_model], [d_model, d_moe]] -> [[B*T, d_moe]]  w=[d_moe, d_model]
  44 silu             mlp.shared_experts.act_fn          [[B, T, d_moe]] -> [[B, T, d_moe]]
  45 elementwise_mul  mlp.shared_experts                 [[B, T, d_moe], [B, T, d_moe]] -> [[B, T, d_moe]]
  46 matmul           mlp.shared_experts.down_proj       [[B*T, d_moe], [d_moe, d_model]] -> [[B*T, d_model]]  w=[d_model, d_moe]
  47 elementwise_add  mlp                                [[B, T, d_model], [B, T, d_model]] -> [[B, T, d_model]]
  48 elementwise_mul  model.layers.0                     [[B, T, n_hc, 1], [B, T, 1, d_model]] -> [[B, T, n_hc, d_model]]
  49 batched_matmul   model.layers.0                     [[B*T, n_hc, n_hc], [B*T, n_hc, d_model]] -> [[B*T, n_hc, d_model]]
  50 elementwise_add  model.layers.0                     [[B, T, n_hc, d_model], [B, T, n_hc, d_model]] -> [[B, T, n_hc, d_model]]
```

**`block_type=MLA+MoE`  repeat=1  layers=`2`**  (58행)

```
  51 rmsnorm          attn_hc.input_norm                 [[B, T, n_hc*d_model]] -> [[B, T, n_hc*d_model]]
  52 matmul           attn_hc                            [[B*T, n_hc*d_model], [n_hc*d_model, (2+n_hc)*n_hc]] -> [[B*T, (2+n_hc)*n_hc]]  w=[(2+n_hc)*n_hc, n_hc*d_model]
  53 sigmoid          attn_hc                            [[B, T, n_hc]] -> [[B, T, n_hc]]
  54 sigmoid          attn_hc                            [[B, T, n_hc]] -> [[B, T, n_hc]]
  55 softmax          attn_hc                            [[B, T, n_hc, n_hc]] -> [[B, T, n_hc, n_hc]]
  56 elementwise_mul  attn_hc                            [[B, T, n_hc, 1], [B, T, n_hc, d_model]] -> [[B, T, n_hc, d_model]]
  57 sum              attn_hc                            [[B, T, n_hc, d_model]] -> [[B, T, d_model]]
  58 rmsnorm          input_layernorm                    [[B, T, d_model]] -> [[B, T, d_model]]  w=[d_model]
  59 matmul           self_attn.q_a_proj                 [[B*T, d_model], [d_model, c_q]] -> [[B*T, c_q]]  w=[c_q, d_model]
  60 rmsnorm          self_attn.q_a_norm                 [[B, T, c_q]] -> [[B, T, c_q]]  w=[c_q]
  61 matmul           self_attn.q_b_proj                 [[B*T, c_q], [c_q, n_h*d_head]] -> [[B*T, n_h*d_head]]  w=[n_h*d_head, c_q]
  62 rmsnorm          self_attn.q_b_norm                 [[B, n_h, T, d_head]] -> [[B, n_h, T, d_head]]
  63 matmul           self_attn.kv_proj                  [[B*T, d_model], [d_model, d_head]] -> [[B*T, d_head]]  w=[d_head, d_model]
  64 rmsnorm          self_attn.kv_norm                  [[B, T, d_head]] -> [[B, T, d_head]]  w=[d_head]
  65 matmul           self_attn.compressor.kv_proj       [[B*T, d_model], [d_model, 2*d_head]] -> [[B*T, 2*d_head]]  w=[2*d_head, d_model]
  66 matmul           self_attn.compressor.gate_proj     [[B*T, d_model], [d_model, 2*d_head]] -> [[B*T, 2*d_head]]  w=[2*d_head, d_model]
  67 softmax          self_attn.compressor               [[B, T/m_csa, 2*m_csa, d_head]] -> [[B, T/m_csa, 2*m_csa, d_head]]
  68 rmsnorm          self_attn.compressor.kv_norm       [[B, T/m_csa, d_head]] -> [[B, T/m_csa, d_head]]  w=[d_head]
  69 matmul           self_attn.compressor.indexer.kv_pr [[B*T, d_model], [d_model, 2*c_I]] -> [[B*T, 2*c_I]]  w=[2*c_I, d_model]
  70 matmul           self_attn.compressor.indexer.gate_ [[B*T, d_model], [d_model, 2*c_I]] -> [[B*T, 2*c_I]]  w=[2*c_I, d_model]
  71 softmax          self_attn.compressor.indexer       [[B, T/m_csa, 2*m_csa, c_I]] -> [[B, T/m_csa, 2*m_csa, c_I]]
  72 rmsnorm          self_attn.compressor.indexer.kv_no [[B, T/m_csa, c_I]] -> [[B, T/m_csa, c_I]]  w=[c_I]
  73 matmul           self_attn.compressor.indexer.q_b_p [[B*T, c_q], [c_q, n_h_I*c_I]] -> [[B*T, n_h_I*c_I]]  w=[n_h_I*c_I, c_q]
  74 batched_matmul   self_attn.compressor.indexer.score [[B*T, n_h_I, c_I], [B*T, c_I, T/m_csa]] -> [[B*T, n_h_I, T/m_csa]]
  75 relu             self_attn.compressor.indexer.score [[B, T, n_h_I, T/m_csa]] -> [[B, T, n_h_I, T/m_csa]]
  76 matmul           self_attn.compressor.indexer.score [[B*T, d_model], [d_model, n_h_I]] -> [[B*T, n_h_I]]  w=[n_h_I, d_model]
  77 batched_matmul   self_attn                          [[B*n_h, T, d_head], [B*n_h, d_head, T+T/m_csa]] -> [[B*n_h, T, T+T/m_csa]]
  78 softmax          self_attn                          [[B, n_h, T, T+T/m_csa+1]] -> [[B, n_h, T, T+T/m_csa+1]]
  79 batched_matmul   self_attn                          [[B*n_h, T, T+T/m_csa], [B*n_h, T+T/m_csa, d_head]] -> [[B*n_h, T, d_head]]
  80 batched_matmul   self_attn.o_a_proj                 [[g_o, B*T, n_h*d_head/g_o], [g_o, n_h*d_head/g_o, d_g]] -> [[g_o, B*T, d_g]]  w=[g_o*d_g, n_h*d_head/g_o]
  81 matmul           self_attn.o_b_proj                 [[B*T, g_o*d_g], [g_o*d_g, d_model]] -> [[B*T, d_model]]  w=[d_model, g_o*d_g]
  82 elementwise_mul  model.layers.2                     [[B, T, n_hc, 1], [B, T, 1, d_model]] -> [[B, T, n_hc, d_model]]
  83 batched_matmul   model.layers.2                     [[B*T, n_hc, n_hc], [B*T, n_hc, d_model]] -> [[B*T, n_hc, d_model]]
  84 elementwise_add  model.layers.2                     [[B, T, n_hc, d_model], [B, T, n_hc, d_model]] -> [[B, T, n_hc, d_model]]
  85 rmsnorm          ffn_hc.input_norm                  [[B, T, n_hc*d_model]] -> [[B, T, n_hc*d_model]]
  86 matmul           ffn_hc                             [[B*T, n_hc*d_model], [n_hc*d_model, (2+n_hc)*n_hc]] -> [[B*T, (2+n_hc)*n_hc]]  w=[(2+n_hc)*n_hc, n_hc*d_model]
  87 sigmoid          ffn_hc                             [[B, T, n_hc]] -> [[B, T, n_hc]]
  88 sigmoid          ffn_hc                             [[B, T, n_hc]] -> [[B, T, n_hc]]
  89 softmax          ffn_hc                             [[B, T, n_hc, n_hc]] -> [[B, T, n_hc, n_hc]]
  90 elementwise_mul  ffn_hc                             [[B, T, n_hc, 1], [B, T, n_hc, d_model]] -> [[B, T, n_hc, d_model]]
  91 sum              ffn_hc                             [[B, T, n_hc, d_model]] -> [[B, T, d_model]]
  92 rmsnorm          post_attention_layernorm           [[B, T, d_model]] -> [[B, T, d_model]]  w=[d_model]
  93 matmul           mlp.gate                           [[B*T, d_model], [d_model, E]] -> [[B*T, E]]  w=[E, d_model]
  94 grouped_matmul   mlp.experts                        [[B*k*T, d_model], [E, d_model, 2*d_moe], [E]] -> [[B*k*T, 2*d_moe]]  w=[E, 2*d_moe, d_model]
  95 silu             mlp.experts.act_fn                 [[B*k*T, d_moe]] -> [[B*k*T, d_moe]]
  96 elementwise_mul  mlp.experts                        [[B*k*T, d_moe], [B*k*T, d_moe]] -> [[B*k*T, d_moe]]
  97 grouped_matmul   mlp.experts                        [[B*k*T, d_moe], [E, d_moe, d_model], [E]] -> [[B*k*T, d_model]]  w=[E, d_model, d_moe]
  98 elementwise_mul  mlp.experts                        [[B*k*T, d_model], [B*k*T, 1]] -> [[B*k*T, d_model]]
  99 sum              mlp.experts                        [[B*T, k, d_model]] -> [[B*T, d_model]]
 100 matmul           mlp.shared_experts.gate_proj       [[B*T, d_model], [d_model, d_moe]] -> [[B*T, d_moe]]  w=[d_moe, d_model]
 101 matmul           mlp.shared_experts.up_proj         [[B*T, d_model], [d_model, d_moe]] -> [[B*T, d_moe]]  w=[d_moe, d_model]
 102 silu             mlp.shared_experts.act_fn          [[B, T, d_moe]] -> [[B, T, d_moe]]
 103 elementwise_mul  mlp.shared_experts                 [[B, T, d_moe], [B, T, d_moe]] -> [[B, T, d_moe]]
 104 matmul           mlp.shared_experts.down_proj       [[B*T, d_moe], [d_moe, d_model]] -> [[B*T, d_model]]  w=[d_model, d_moe]
 105 elementwise_add  mlp                                [[B, T, d_model], [B, T, d_model]] -> [[B, T, d_model]]
 106 elementwise_mul  model.layers.2                     [[B, T, n_hc, 1], [B, T, 1, d_model]] -> [[B, T, n_hc, d_model]]
 107 batched_matmul   model.layers.2                     [[B*T, n_hc, n_hc], [B*T, n_hc, d_model]] -> [[B*T, n_hc, d_model]]
 108 elementwise_add  model.layers.2                     [[B, T, n_hc, d_model], [B, T, n_hc, d_model]] -> [[B, T, n_hc, d_model]]
```

**`block_type=MLA+MoE`  repeat=29  layers=`3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59`**  (50행)

```
 109 rmsnorm          attn_hc.input_norm                 [[B, T, n_hc*d_model]] -> [[B, T, n_hc*d_model]]
 110 matmul           attn_hc                            [[B*T, n_hc*d_model], [n_hc*d_model, (2+n_hc)*n_hc]] -> [[B*T, (2+n_hc)*n_hc]]  w=[(2+n_hc)*n_hc, n_hc*d_model]
 111 sigmoid          attn_hc                            [[B, T, n_hc]] -> [[B, T, n_hc]]
 112 sigmoid          attn_hc                            [[B, T, n_hc]] -> [[B, T, n_hc]]
 113 softmax          attn_hc                            [[B, T, n_hc, n_hc]] -> [[B, T, n_hc, n_hc]]
 114 elementwise_mul  attn_hc                            [[B, T, n_hc, 1], [B, T, n_hc, d_model]] -> [[B, T, n_hc, d_model]]
 115 sum              attn_hc                            [[B, T, n_hc, d_model]] -> [[B, T, d_model]]
 116 rmsnorm          input_layernorm                    [[B, T, d_model]] -> [[B, T, d_model]]  w=[d_model]
 117 matmul           self_attn.q_a_proj                 [[B*T, d_model], [d_model, c_q]] -> [[B*T, c_q]]  w=[c_q, d_model]
 118 rmsnorm          self_attn.q_a_norm                 [[B, T, c_q]] -> [[B, T, c_q]]  w=[c_q]
 119 matmul           self_attn.q_b_proj                 [[B*T, c_q], [c_q, n_h*d_head]] -> [[B*T, n_h*d_head]]  w=[n_h*d_head, c_q]
 120 rmsnorm          self_attn.q_b_norm                 [[B, n_h, T, d_head]] -> [[B, n_h, T, d_head]]
 121 matmul           self_attn.kv_proj                  [[B*T, d_model], [d_model, d_head]] -> [[B*T, d_head]]  w=[d_head, d_model]
 122 rmsnorm          self_attn.kv_norm                  [[B, T, d_head]] -> [[B, T, d_head]]  w=[d_head]
 123 matmul           self_attn.compressor.kv_proj       [[B*T, d_model], [d_model, d_head]] -> [[B*T, d_head]]  w=[d_head, d_model]
 124 matmul           self_attn.compressor.gate_proj     [[B*T, d_model], [d_model, d_head]] -> [[B*T, d_head]]  w=[d_head, d_model]
 125 softmax          self_attn.compressor               [[B, T/m_hca, m_hca, d_head]] -> [[B, T/m_hca, m_hca, d_head]]
 126 rmsnorm          self_attn.compressor.kv_norm       [[B, T/m_hca, d_head]] -> [[B, T/m_hca, d_head]]  w=[d_head]
 127 batched_matmul   self_attn                          [[B*n_h, T, d_head], [B*n_h, d_head, T+T/m_hca]] -> [[B*n_h, T, T+T/m_hca]]
 128 softmax          self_attn                          [[B, n_h, T, T+T/m_hca+1]] -> [[B, n_h, T, T+T/m_hca+1]]
 129 batched_matmul   self_attn                          [[B*n_h, T, T+T/m_hca], [B*n_h, T+T/m_hca, d_head]] -> [[B*n_h, T, d_head]]
 130 batched_matmul   self_attn.o_a_proj                 [[g_o, B*T, n_h*d_head/g_o], [g_o, n_h*d_head/g_o, d_g]] -> [[g_o, B*T, d_g]]  w=[g_o*d_g, n_h*d_head/g_o]
 131 matmul           self_attn.o_b_proj                 [[B*T, g_o*d_g], [g_o*d_g, d_model]] -> [[B*T, d_model]]  w=[d_model, g_o*d_g]
 132 elementwise_mul  model.layers.3                     [[B, T, n_hc, 1], [B, T, 1, d_model]] -> [[B, T, n_hc, d_model]]
 133 batched_matmul   model.layers.3                     [[B*T, n_hc, n_hc], [B*T, n_hc, d_model]] -> [[B*T, n_hc, d_model]]
 134 elementwise_add  model.layers.3                     [[B, T, n_hc, d_model], [B, T, n_hc, d_model]] -> [[B, T, n_hc, d_model]]
 135 rmsnorm          ffn_hc.input_norm                  [[B, T, n_hc*d_model]] -> [[B, T, n_hc*d_model]]
 136 matmul           ffn_hc                             [[B*T, n_hc*d_model], [n_hc*d_model, (2+n_hc)*n_hc]] -> [[B*T, (2+n_hc)*n_hc]]  w=[(2+n_hc)*n_hc, n_hc*d_model]
 137 sigmoid          ffn_hc                             [[B, T, n_hc]] -> [[B, T, n_hc]]
 138 sigmoid          ffn_hc                             [[B, T, n_hc]] -> [[B, T, n_hc]]
 139 softmax          ffn_hc                             [[B, T, n_hc, n_hc]] -> [[B, T, n_hc, n_hc]]
 140 elementwise_mul  ffn_hc                             [[B, T, n_hc, 1], [B, T, n_hc, d_model]] -> [[B, T, n_hc, d_model]]
 141 sum              ffn_hc                             [[B, T, n_hc, d_model]] -> [[B, T, d_model]]
 142 rmsnorm          post_attention_layernorm           [[B, T, d_model]] -> [[B, T, d_model]]  w=[d_model]
 143 matmul           mlp.gate                           [[B*T, d_model], [d_model, E]] -> [[B*T, E]]  w=[E, d_model]
 144 grouped_matmul   mlp.experts                        [[B*k*T, d_model], [E, d_model, 2*d_moe], [E]] -> [[B*k*T, 2*d_moe]]  w=[E, 2*d_moe, d_model]
 145 silu             mlp.experts.act_fn                 [[B*k*T, d_moe]] -> [[B*k*T, d_moe]]
 146 elementwise_mul  mlp.experts                        [[B*k*T, d_moe], [B*k*T, d_moe]] -> [[B*k*T, d_moe]]
 147 grouped_matmul   mlp.experts                        [[B*k*T, d_moe], [E, d_moe, d_model], [E]] -> [[B*k*T, d_model]]  w=[E, d_model, d_moe]
 148 elementwise_mul  mlp.experts                        [[B*k*T, d_model], [B*k*T, 1]] -> [[B*k*T, d_model]]
 149 sum              mlp.experts                        [[B*T, k, d_model]] -> [[B*T, d_model]]
 150 matmul           mlp.shared_experts.gate_proj       [[B*T, d_model], [d_model, d_moe]] -> [[B*T, d_moe]]  w=[d_moe, d_model]
 151 matmul           mlp.shared_experts.up_proj         [[B*T, d_model], [d_model, d_moe]] -> [[B*T, d_moe]]  w=[d_moe, d_model]
 152 silu             mlp.shared_experts.act_fn          [[B, T, d_moe]] -> [[B, T, d_moe]]
 153 elementwise_mul  mlp.shared_experts                 [[B, T, d_moe], [B, T, d_moe]] -> [[B, T, d_moe]]
 154 matmul           mlp.shared_experts.down_proj       [[B*T, d_moe], [d_moe, d_model]] -> [[B*T, d_model]]  w=[d_model, d_moe]
 155 elementwise_add  mlp                                [[B, T, d_model], [B, T, d_model]] -> [[B, T, d_model]]
 156 elementwise_mul  model.layers.3                     [[B, T, n_hc, 1], [B, T, 1, d_model]] -> [[B, T, n_hc, d_model]]
 157 batched_matmul   model.layers.3                     [[B*T, n_hc, n_hc], [B*T, n_hc, d_model]] -> [[B*T, n_hc, d_model]]
 158 elementwise_add  model.layers.3                     [[B, T, n_hc, d_model], [B, T, n_hc, d_model]] -> [[B, T, n_hc, d_model]]
```

**`block_type=MLA+MoE`  repeat=29  layers=`4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60`**  (58행)

```
 159 rmsnorm          attn_hc.input_norm                 [[B, T, n_hc*d_model]] -> [[B, T, n_hc*d_model]]
 160 matmul           attn_hc                            [[B*T, n_hc*d_model], [n_hc*d_model, (2+n_hc)*n_hc]] -> [[B*T, (2+n_hc)*n_hc]]  w=[(2+n_hc)*n_hc, n_hc*d_model]
 161 sigmoid          attn_hc                            [[B, T, n_hc]] -> [[B, T, n_hc]]
 162 sigmoid          attn_hc                            [[B, T, n_hc]] -> [[B, T, n_hc]]
 163 softmax          attn_hc                            [[B, T, n_hc, n_hc]] -> [[B, T, n_hc, n_hc]]
 164 elementwise_mul  attn_hc                            [[B, T, n_hc, 1], [B, T, n_hc, d_model]] -> [[B, T, n_hc, d_model]]
 165 sum              attn_hc                            [[B, T, n_hc, d_model]] -> [[B, T, d_model]]
 166 rmsnorm          input_layernorm                    [[B, T, d_model]] -> [[B, T, d_model]]  w=[d_model]
 167 matmul           self_attn.q_a_proj                 [[B*T, d_model], [d_model, c_q]] -> [[B*T, c_q]]  w=[c_q, d_model]
 168 rmsnorm          self_attn.q_a_norm                 [[B, T, c_q]] -> [[B, T, c_q]]  w=[c_q]
 169 matmul           self_attn.q_b_proj                 [[B*T, c_q], [c_q, n_h*d_head]] -> [[B*T, n_h*d_head]]  w=[n_h*d_head, c_q]
 170 rmsnorm          self_attn.q_b_norm                 [[B, n_h, T, d_head]] -> [[B, n_h, T, d_head]]
 171 matmul           self_attn.kv_proj                  [[B*T, d_model], [d_model, d_head]] -> [[B*T, d_head]]  w=[d_head, d_model]
 172 rmsnorm          self_attn.kv_norm                  [[B, T, d_head]] -> [[B, T, d_head]]  w=[d_head]
 173 matmul           self_attn.compressor.kv_proj       [[B*T, d_model], [d_model, 2*d_head]] -> [[B*T, 2*d_head]]  w=[2*d_head, d_model]
 174 matmul           self_attn.compressor.gate_proj     [[B*T, d_model], [d_model, 2*d_head]] -> [[B*T, 2*d_head]]  w=[2*d_head, d_model]
 175 softmax          self_attn.compressor               [[B, T/m_csa, 2*m_csa, d_head]] -> [[B, T/m_csa, 2*m_csa, d_head]]
 176 rmsnorm          self_attn.compressor.kv_norm       [[B, T/m_csa, d_head]] -> [[B, T/m_csa, d_head]]  w=[d_head]
 177 matmul           self_attn.compressor.indexer.kv_pr [[B*T, d_model], [d_model, 2*c_I]] -> [[B*T, 2*c_I]]  w=[2*c_I, d_model]
 178 matmul           self_attn.compressor.indexer.gate_ [[B*T, d_model], [d_model, 2*c_I]] -> [[B*T, 2*c_I]]  w=[2*c_I, d_model]
 179 softmax          self_attn.compressor.indexer       [[B, T/m_csa, 2*m_csa, c_I]] -> [[B, T/m_csa, 2*m_csa, c_I]]
 180 rmsnorm          self_attn.compressor.indexer.kv_no [[B, T/m_csa, c_I]] -> [[B, T/m_csa, c_I]]  w=[c_I]
 181 matmul           self_attn.compressor.indexer.q_b_p [[B*T, c_q], [c_q, n_h_I*c_I]] -> [[B*T, n_h_I*c_I]]  w=[n_h_I*c_I, c_q]
 182 batched_matmul   self_attn.compressor.indexer.score [[B*T, n_h_I, c_I], [B*T, c_I, T/m_csa]] -> [[B*T, n_h_I, T/m_csa]]
 183 relu             self_attn.compressor.indexer.score [[B, T, n_h_I, T/m_csa]] -> [[B, T, n_h_I, T/m_csa]]
 184 matmul           self_attn.compressor.indexer.score [[B*T, d_model], [d_model, n_h_I]] -> [[B*T, n_h_I]]  w=[n_h_I, d_model]
 185 batched_matmul   self_attn                          [[B*n_h, T, d_head], [B*n_h, d_head, T+T/m_csa]] -> [[B*n_h, T, T+T/m_csa]]
 186 softmax          self_attn                          [[B, n_h, T, T+T/m_csa+1]] -> [[B, n_h, T, T+T/m_csa+1]]
 187 batched_matmul   self_attn                          [[B*n_h, T, T+T/m_csa], [B*n_h, T+T/m_csa, d_head]] -> [[B*n_h, T, d_head]]
 188 batched_matmul   self_attn.o_a_proj                 [[g_o, B*T, n_h*d_head/g_o], [g_o, n_h*d_head/g_o, d_g]] -> [[g_o, B*T, d_g]]  w=[g_o*d_g, n_h*d_head/g_o]
 189 matmul           self_attn.o_b_proj                 [[B*T, g_o*d_g], [g_o*d_g, d_model]] -> [[B*T, d_model]]  w=[d_model, g_o*d_g]
 190 elementwise_mul  model.layers.4                     [[B, T, n_hc, 1], [B, T, 1, d_model]] -> [[B, T, n_hc, d_model]]
 191 batched_matmul   model.layers.4                     [[B*T, n_hc, n_hc], [B*T, n_hc, d_model]] -> [[B*T, n_hc, d_model]]
 192 elementwise_add  model.layers.4                     [[B, T, n_hc, d_model], [B, T, n_hc, d_model]] -> [[B, T, n_hc, d_model]]
 193 rmsnorm          ffn_hc.input_norm                  [[B, T, n_hc*d_model]] -> [[B, T, n_hc*d_model]]
 194 matmul           ffn_hc                             [[B*T, n_hc*d_model], [n_hc*d_model, (2+n_hc)*n_hc]] -> [[B*T, (2+n_hc)*n_hc]]  w=[(2+n_hc)*n_hc, n_hc*d_model]
 195 sigmoid          ffn_hc                             [[B, T, n_hc]] -> [[B, T, n_hc]]
 196 sigmoid          ffn_hc                             [[B, T, n_hc]] -> [[B, T, n_hc]]
 197 softmax          ffn_hc                             [[B, T, n_hc, n_hc]] -> [[B, T, n_hc, n_hc]]
 198 elementwise_mul  ffn_hc                             [[B, T, n_hc, 1], [B, T, n_hc, d_model]] -> [[B, T, n_hc, d_model]]
 199 sum              ffn_hc                             [[B, T, n_hc, d_model]] -> [[B, T, d_model]]
 200 rmsnorm          post_attention_layernorm           [[B, T, d_model]] -> [[B, T, d_model]]  w=[d_model]
 201 matmul           mlp.gate                           [[B*T, d_model], [d_model, E]] -> [[B*T, E]]  w=[E, d_model]
 202 grouped_matmul   mlp.experts                        [[B*k*T, d_model], [E, d_model, 2*d_moe], [E]] -> [[B*k*T, 2*d_moe]]  w=[E, 2*d_moe, d_model]
 203 silu             mlp.experts.act_fn                 [[B*k*T, d_moe]] -> [[B*k*T, d_moe]]
 204 elementwise_mul  mlp.experts                        [[B*k*T, d_moe], [B*k*T, d_moe]] -> [[B*k*T, d_moe]]
 205 grouped_matmul   mlp.experts                        [[B*k*T, d_moe], [E, d_moe, d_model], [E]] -> [[B*k*T, d_model]]  w=[E, d_model, d_moe]
 206 elementwise_mul  mlp.experts                        [[B*k*T, d_model], [B*k*T, 1]] -> [[B*k*T, d_model]]
 207 sum              mlp.experts                        [[B*T, k, d_model]] -> [[B*T, d_model]]
 208 matmul           mlp.shared_experts.gate_proj       [[B*T, d_model], [d_model, d_moe]] -> [[B*T, d_moe]]  w=[d_moe, d_model]
 209 matmul           mlp.shared_experts.up_proj         [[B*T, d_model], [d_model, d_moe]] -> [[B*T, d_moe]]  w=[d_moe, d_model]
 210 silu             mlp.shared_experts.act_fn          [[B, T, d_moe]] -> [[B, T, d_moe]]
 211 elementwise_mul  mlp.shared_experts                 [[B, T, d_moe], [B, T, d_moe]] -> [[B, T, d_moe]]
 212 matmul           mlp.shared_experts.down_proj       [[B*T, d_moe], [d_moe, d_model]] -> [[B*T, d_model]]  w=[d_model, d_moe]
 213 elementwise_add  mlp                                [[B, T, d_model], [B, T, d_model]] -> [[B, T, d_model]]
 214 elementwise_mul  model.layers.4                     [[B, T, n_hc, 1], [B, T, 1, d_model]] -> [[B, T, n_hc, d_model]]
 215 batched_matmul   model.layers.4                     [[B*T, n_hc, n_hc], [B*T, n_hc, d_model]] -> [[B*T, n_hc, d_model]]
 216 elementwise_add  model.layers.4                     [[B, T, n_hc, d_model], [B, T, n_hc, d_model]] -> [[B, T, n_hc, d_model]]
```

**`block_type=norm`  repeat=1  layers=`-`**  (2행)

```
 217 rmsnorm          model.hc_head.input_norm           [[B, T, n_hc*d_model]] -> [[B, T, n_hc*d_model]]
 222 rmsnorm          model.norm                         [[B, T, d_model]] -> [[B, T, d_model]]  w=[d_model]
```

**`block_type=-`  repeat=1  layers=`-`**  (4행)

```
 218 matmul           model.hc_head                      [[B*T, n_hc*d_model], [n_hc*d_model, n_hc]] -> [[B*T, n_hc]]  w=[n_hc, n_hc*d_model]
 219 sigmoid          model.hc_head                      [[B, T, n_hc]] -> [[B, T, n_hc]]
 220 elementwise_mul  model.hc_head                      [[B, T, n_hc, 1], [B, T, n_hc, d_model]] -> [[B, T, n_hc, d_model]]
 221 sum              model.hc_head                      [[B, T, n_hc, d_model]] -> [[B, T, d_model]]
```

**`block_type=head`  repeat=1  layers=`-`**  (1행)

```
 223 matmul           lm_head                            [[B*T, d_model], [d_model, V]] -> [[B*T, V]]  w=[V, d_model]
```

### decode — 템플릿 8개 / 전체 212행

**`block_type=embed`  repeat=1  layers=`-`**  (1행)

```
   0 embedding        model.embed_tokens                 [[V, d_model], [B, 1]] -> [[B, 1, d_model]]  w=[V, d_model]
```

**`block_type=MLA+MoE`  repeat=2  layers=`0-1`**  (48행)

```
   1 rmsnorm          attn_hc.input_norm                 [[B, 1, n_hc*d_model]] -> [[B, 1, n_hc*d_model]]
   2 matmul           attn_hc                            [[B, n_hc*d_model], [n_hc*d_model, (2+n_hc)*n_hc]] -> [[B, (2+n_hc)*n_hc]]  w=[(2+n_hc)*n_hc, n_hc*d_model]
   3 sigmoid          attn_hc                            [[B, 1, n_hc]] -> [[B, 1, n_hc]]
   4 sigmoid          attn_hc                            [[B, 1, n_hc]] -> [[B, 1, n_hc]]
   5 softmax          attn_hc                            [[B, 1, n_hc, n_hc]] -> [[B, 1, n_hc, n_hc]]
   6 elementwise_mul  attn_hc                            [[B, 1, n_hc, 1], [B, 1, n_hc, d_model]] -> [[B, 1, n_hc, d_model]]
   7 sum              attn_hc                            [[B, 1, n_hc, d_model]] -> [[B, 1, d_model]]
   8 rmsnorm          input_layernorm                    [[B, 1, d_model]] -> [[B, 1, d_model]]  w=[d_model]
   9 matmul           self_attn.q_a_proj                 [[B, d_model], [d_model, c_q]] -> [[B, c_q]]  w=[c_q, d_model]
  10 rmsnorm          self_attn.q_a_norm                 [[B, 1, c_q]] -> [[B, 1, c_q]]  w=[c_q]
  11 matmul           self_attn.q_b_proj                 [[B, c_q], [c_q, n_h*d_head]] -> [[B, n_h*d_head]]  w=[n_h*d_head, c_q]
  12 rmsnorm          self_attn.q_b_norm                 [[B, n_h, 1, d_head]] -> [[B, n_h, 1, d_head]]
  13 matmul           self_attn.kv_proj                  [[B, d_model], [d_model, d_head]] -> [[B, d_head]]  w=[d_head, d_model]
  14 rmsnorm          self_attn.kv_norm                  [[B, 1, d_head]] -> [[B, 1, d_head]]  w=[d_head]
  15 matmul           self_attn.compressor.kv_proj       [[B, d_model], [d_model, d_head]] -> [[B, d_head]]  w=[d_head, d_model]
  16 matmul           self_attn.compressor.gate_proj     [[B, d_model], [d_model, d_head]] -> [[B, d_head]]  w=[d_head, d_model]
  17 batched_matmul   self_attn                          [[B*n_h, 1, d_head], [B*n_h, d_head, w_local+T/m_hca]] -> [[B*n_h, 1, w_local+T/m_hca]]
  18 softmax          self_attn                          [[B, n_h, 1, w_local+T/m_hca+1]] -> [[B, n_h, 1, w_local+T/m_hca+1]]
  19 batched_matmul   self_attn                          [[B*n_h, 1, w_local+T/m_hca], [B*n_h, w_local+T/m_hca, d_head]] -> [[B*n_h, 1, d_head]]
  20 batched_matmul   self_attn.o_a_proj                 [[g_o, B, n_h*d_head/g_o], [g_o, n_h*d_head/g_o, d_g]] -> [[g_o, B, d_g]]  w=[g_o*d_g, n_h*d_head/g_o]
  21 matmul           self_attn.o_b_proj                 [[B, g_o*d_g], [g_o*d_g, d_model]] -> [[B, d_model]]  w=[d_model, g_o*d_g]
  22 elementwise_mul  model.layers.0                     [[B, 1, n_hc, 1], [B, 1, 1, d_model]] -> [[B, 1, n_hc, d_model]]
  23 batched_matmul   model.layers.0                     [[B, n_hc, n_hc], [B, n_hc, d_model]] -> [[B, n_hc, d_model]]
  24 elementwise_add  model.layers.0                     [[B, 1, n_hc, d_model], [B, 1, n_hc, d_model]] -> [[B, 1, n_hc, d_model]]
  25 rmsnorm          ffn_hc.input_norm                  [[B, 1, n_hc*d_model]] -> [[B, 1, n_hc*d_model]]
  26 matmul           ffn_hc                             [[B, n_hc*d_model], [n_hc*d_model, (2+n_hc)*n_hc]] -> [[B, (2+n_hc)*n_hc]]  w=[(2+n_hc)*n_hc, n_hc*d_model]
  27 sigmoid          ffn_hc                             [[B, 1, n_hc]] -> [[B, 1, n_hc]]
  28 sigmoid          ffn_hc                             [[B, 1, n_hc]] -> [[B, 1, n_hc]]
  29 softmax          ffn_hc                             [[B, 1, n_hc, n_hc]] -> [[B, 1, n_hc, n_hc]]
  30 elementwise_mul  ffn_hc                             [[B, 1, n_hc, 1], [B, 1, n_hc, d_model]] -> [[B, 1, n_hc, d_model]]
  31 sum              ffn_hc                             [[B, 1, n_hc, d_model]] -> [[B, 1, d_model]]
  32 rmsnorm          post_attention_layernorm           [[B, 1, d_model]] -> [[B, 1, d_model]]  w=[d_model]
  33 matmul           mlp.gate                           [[B, d_model], [d_model, E]] -> [[B, E]]  w=[E, d_model]
  34 grouped_matmul   mlp.experts                        [[B*k, d_model], [E, d_model, 2*d_moe], [E]] -> [[B*k, 2*d_moe]]  w=[E, 2*d_moe, d_model]
  35 silu             mlp.experts.act_fn                 [[B*k, d_moe]] -> [[B*k, d_moe]]
  36 elementwise_mul  mlp.experts                        [[B*k, d_moe], [B*k, d_moe]] -> [[B*k, d_moe]]
  37 grouped_matmul   mlp.experts                        [[B*k, d_moe], [E, d_moe, d_model], [E]] -> [[B*k, d_model]]  w=[E, d_model, d_moe]
  38 elementwise_mul  mlp.experts                        [[B*k, d_model], [B*k, 1]] -> [[B*k, d_model]]
  39 sum              mlp.experts                        [[B, k, d_model]] -> [[B, d_model]]
  40 matmul           mlp.shared_experts.gate_proj       [[B, d_model], [d_model, d_moe]] -> [[B, d_moe]]  w=[d_moe, d_model]
  41 matmul           mlp.shared_experts.up_proj         [[B, d_model], [d_model, d_moe]] -> [[B, d_moe]]  w=[d_moe, d_model]
  42 silu             mlp.shared_experts.act_fn          [[B, 1, d_moe]] -> [[B, 1, d_moe]]
  43 elementwise_mul  mlp.shared_experts                 [[B, 1, d_moe], [B, 1, d_moe]] -> [[B, 1, d_moe]]
  44 matmul           mlp.shared_experts.down_proj       [[B, d_moe], [d_moe, d_model]] -> [[B, d_model]]  w=[d_model, d_moe]
  45 elementwise_add  mlp                                [[B, 1, d_model], [B, 1, d_model]] -> [[B, 1, d_model]]
  46 elementwise_mul  model.layers.0                     [[B, 1, n_hc, 1], [B, 1, 1, d_model]] -> [[B, 1, n_hc, d_model]]
  47 batched_matmul   model.layers.0                     [[B, n_hc, n_hc], [B, n_hc, d_model]] -> [[B, n_hc, d_model]]
  48 elementwise_add  model.layers.0                     [[B, 1, n_hc, d_model], [B, 1, n_hc, d_model]] -> [[B, 1, n_hc, d_model]]
```

**`block_type=MLA+MoE`  repeat=1  layers=`2`**  (54행)

```
  49 rmsnorm          attn_hc.input_norm                 [[B, 1, n_hc*d_model]] -> [[B, 1, n_hc*d_model]]
  50 matmul           attn_hc                            [[B, n_hc*d_model], [n_hc*d_model, (2+n_hc)*n_hc]] -> [[B, (2+n_hc)*n_hc]]  w=[(2+n_hc)*n_hc, n_hc*d_model]
  51 sigmoid          attn_hc                            [[B, 1, n_hc]] -> [[B, 1, n_hc]]
  52 sigmoid          attn_hc                            [[B, 1, n_hc]] -> [[B, 1, n_hc]]
  53 softmax          attn_hc                            [[B, 1, n_hc, n_hc]] -> [[B, 1, n_hc, n_hc]]
  54 elementwise_mul  attn_hc                            [[B, 1, n_hc, 1], [B, 1, n_hc, d_model]] -> [[B, 1, n_hc, d_model]]
  55 sum              attn_hc                            [[B, 1, n_hc, d_model]] -> [[B, 1, d_model]]
  56 rmsnorm          input_layernorm                    [[B, 1, d_model]] -> [[B, 1, d_model]]  w=[d_model]
  57 matmul           self_attn.q_a_proj                 [[B, d_model], [d_model, c_q]] -> [[B, c_q]]  w=[c_q, d_model]
  58 rmsnorm          self_attn.q_a_norm                 [[B, 1, c_q]] -> [[B, 1, c_q]]  w=[c_q]
  59 matmul           self_attn.q_b_proj                 [[B, c_q], [c_q, n_h*d_head]] -> [[B, n_h*d_head]]  w=[n_h*d_head, c_q]
  60 rmsnorm          self_attn.q_b_norm                 [[B, n_h, 1, d_head]] -> [[B, n_h, 1, d_head]]
  61 matmul           self_attn.kv_proj                  [[B, d_model], [d_model, d_head]] -> [[B, d_head]]  w=[d_head, d_model]
  62 rmsnorm          self_attn.kv_norm                  [[B, 1, d_head]] -> [[B, 1, d_head]]  w=[d_head]
  63 matmul           self_attn.compressor.kv_proj       [[B, d_model], [d_model, 2*d_head]] -> [[B, 2*d_head]]  w=[2*d_head, d_model]
  64 matmul           self_attn.compressor.gate_proj     [[B, d_model], [d_model, 2*d_head]] -> [[B, 2*d_head]]  w=[2*d_head, d_model]
  65 matmul           self_attn.compressor.indexer.kv_pr [[B, d_model], [d_model, 2*c_I]] -> [[B, 2*c_I]]  w=[2*c_I, d_model]
  66 matmul           self_attn.compressor.indexer.gate_ [[B, d_model], [d_model, 2*c_I]] -> [[B, 2*c_I]]  w=[2*c_I, d_model]
  67 matmul           self_attn.compressor.indexer.q_b_p [[B, c_q], [c_q, n_h_I*c_I]] -> [[B, n_h_I*c_I]]  w=[n_h_I*c_I, c_q]
  68 batched_matmul   self_attn.compressor.indexer.score [[B, n_h_I, c_I], [B, c_I, T/m_csa]] -> [[B, n_h_I, T/m_csa]]
  69 relu             self_attn.compressor.indexer.score [[B, 1, n_h_I, T/m_csa]] -> [[B, 1, n_h_I, T/m_csa]]
  70 matmul           self_attn.compressor.indexer.score [[B, d_model], [d_model, n_h_I]] -> [[B, n_h_I]]  w=[n_h_I, d_model]
  71 batched_matmul   self_attn                          [[B*n_h, 1, d_head], [B*n_h, d_head, w_local+T/m_csa]] -> [[B*n_h, 1, w_local+T/m_csa]]
  72 softmax          self_attn                          [[B, n_h, 1, w_local+T/m_csa+1]] -> [[B, n_h, 1, w_local+T/m_csa+1]]
  73 batched_matmul   self_attn                          [[B*n_h, 1, w_local+T/m_csa], [B*n_h, w_local+T/m_csa, d_head]] -> [[B*n_h, 1, d_head]]
  74 batched_matmul   self_attn.o_a_proj                 [[g_o, B, n_h*d_head/g_o], [g_o, n_h*d_head/g_o, d_g]] -> [[g_o, B, d_g]]  w=[g_o*d_g, n_h*d_head/g_o]
  75 matmul           self_attn.o_b_proj                 [[B, g_o*d_g], [g_o*d_g, d_model]] -> [[B, d_model]]  w=[d_model, g_o*d_g]
  76 elementwise_mul  model.layers.2                     [[B, 1, n_hc, 1], [B, 1, 1, d_model]] -> [[B, 1, n_hc, d_model]]
  77 batched_matmul   model.layers.2                     [[B, n_hc, n_hc], [B, n_hc, d_model]] -> [[B, n_hc, d_model]]
  78 elementwise_add  model.layers.2                     [[B, 1, n_hc, d_model], [B, 1, n_hc, d_model]] -> [[B, 1, n_hc, d_model]]
  79 rmsnorm          ffn_hc.input_norm                  [[B, 1, n_hc*d_model]] -> [[B, 1, n_hc*d_model]]
  80 matmul           ffn_hc                             [[B, n_hc*d_model], [n_hc*d_model, (2+n_hc)*n_hc]] -> [[B, (2+n_hc)*n_hc]]  w=[(2+n_hc)*n_hc, n_hc*d_model]
  81 sigmoid          ffn_hc                             [[B, 1, n_hc]] -> [[B, 1, n_hc]]
  82 sigmoid          ffn_hc                             [[B, 1, n_hc]] -> [[B, 1, n_hc]]
  83 softmax          ffn_hc                             [[B, 1, n_hc, n_hc]] -> [[B, 1, n_hc, n_hc]]
  84 elementwise_mul  ffn_hc                             [[B, 1, n_hc, 1], [B, 1, n_hc, d_model]] -> [[B, 1, n_hc, d_model]]
  85 sum              ffn_hc                             [[B, 1, n_hc, d_model]] -> [[B, 1, d_model]]
  86 rmsnorm          post_attention_layernorm           [[B, 1, d_model]] -> [[B, 1, d_model]]  w=[d_model]
  87 matmul           mlp.gate                           [[B, d_model], [d_model, E]] -> [[B, E]]  w=[E, d_model]
  88 grouped_matmul   mlp.experts                        [[B*k, d_model], [E, d_model, 2*d_moe], [E]] -> [[B*k, 2*d_moe]]  w=[E, 2*d_moe, d_model]
  89 silu             mlp.experts.act_fn                 [[B*k, d_moe]] -> [[B*k, d_moe]]
  90 elementwise_mul  mlp.experts                        [[B*k, d_moe], [B*k, d_moe]] -> [[B*k, d_moe]]
  91 grouped_matmul   mlp.experts                        [[B*k, d_moe], [E, d_moe, d_model], [E]] -> [[B*k, d_model]]  w=[E, d_model, d_moe]
  92 elementwise_mul  mlp.experts                        [[B*k, d_model], [B*k, 1]] -> [[B*k, d_model]]
  93 sum              mlp.experts                        [[B, k, d_model]] -> [[B, d_model]]
  94 matmul           mlp.shared_experts.gate_proj       [[B, d_model], [d_model, d_moe]] -> [[B, d_moe]]  w=[d_moe, d_model]
  95 matmul           mlp.shared_experts.up_proj         [[B, d_model], [d_model, d_moe]] -> [[B, d_moe]]  w=[d_moe, d_model]
  96 silu             mlp.shared_experts.act_fn          [[B, 1, d_moe]] -> [[B, 1, d_moe]]
  97 elementwise_mul  mlp.shared_experts                 [[B, 1, d_moe], [B, 1, d_moe]] -> [[B, 1, d_moe]]
  98 matmul           mlp.shared_experts.down_proj       [[B, d_moe], [d_moe, d_model]] -> [[B, d_model]]  w=[d_model, d_moe]
  99 elementwise_add  mlp                                [[B, 1, d_model], [B, 1, d_model]] -> [[B, 1, d_model]]
 100 elementwise_mul  model.layers.2                     [[B, 1, n_hc, 1], [B, 1, 1, d_model]] -> [[B, 1, n_hc, d_model]]
 101 batched_matmul   model.layers.2                     [[B, n_hc, n_hc], [B, n_hc, d_model]] -> [[B, n_hc, d_model]]
 102 elementwise_add  model.layers.2                     [[B, 1, n_hc, d_model], [B, 1, n_hc, d_model]] -> [[B, 1, n_hc, d_model]]
```

**`block_type=MLA+MoE`  repeat=29  layers=`3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59`**  (48행)

```
 103 rmsnorm          attn_hc.input_norm                 [[B, 1, n_hc*d_model]] -> [[B, 1, n_hc*d_model]]
 104 matmul           attn_hc                            [[B, n_hc*d_model], [n_hc*d_model, (2+n_hc)*n_hc]] -> [[B, (2+n_hc)*n_hc]]  w=[(2+n_hc)*n_hc, n_hc*d_model]
 105 sigmoid          attn_hc                            [[B, 1, n_hc]] -> [[B, 1, n_hc]]
 106 sigmoid          attn_hc                            [[B, 1, n_hc]] -> [[B, 1, n_hc]]
 107 softmax          attn_hc                            [[B, 1, n_hc, n_hc]] -> [[B, 1, n_hc, n_hc]]
 108 elementwise_mul  attn_hc                            [[B, 1, n_hc, 1], [B, 1, n_hc, d_model]] -> [[B, 1, n_hc, d_model]]
 109 sum              attn_hc                            [[B, 1, n_hc, d_model]] -> [[B, 1, d_model]]
 110 rmsnorm          input_layernorm                    [[B, 1, d_model]] -> [[B, 1, d_model]]  w=[d_model]
 111 matmul           self_attn.q_a_proj                 [[B, d_model], [d_model, c_q]] -> [[B, c_q]]  w=[c_q, d_model]
 112 rmsnorm          self_attn.q_a_norm                 [[B, 1, c_q]] -> [[B, 1, c_q]]  w=[c_q]
 113 matmul           self_attn.q_b_proj                 [[B, c_q], [c_q, n_h*d_head]] -> [[B, n_h*d_head]]  w=[n_h*d_head, c_q]
 114 rmsnorm          self_attn.q_b_norm                 [[B, n_h, 1, d_head]] -> [[B, n_h, 1, d_head]]
 115 matmul           self_attn.kv_proj                  [[B, d_model], [d_model, d_head]] -> [[B, d_head]]  w=[d_head, d_model]
 116 rmsnorm          self_attn.kv_norm                  [[B, 1, d_head]] -> [[B, 1, d_head]]  w=[d_head]
 117 matmul           self_attn.compressor.kv_proj       [[B, d_model], [d_model, d_head]] -> [[B, d_head]]  w=[d_head, d_model]
 118 matmul           self_attn.compressor.gate_proj     [[B, d_model], [d_model, d_head]] -> [[B, d_head]]  w=[d_head, d_model]
 119 batched_matmul   self_attn                          [[B*n_h, 1, d_head], [B*n_h, d_head, w_local+T/m_hca]] -> [[B*n_h, 1, w_local+T/m_hca]]
 120 softmax          self_attn                          [[B, n_h, 1, w_local+T/m_hca+1]] -> [[B, n_h, 1, w_local+T/m_hca+1]]
 121 batched_matmul   self_attn                          [[B*n_h, 1, w_local+T/m_hca], [B*n_h, w_local+T/m_hca, d_head]] -> [[B*n_h, 1, d_head]]
 122 batched_matmul   self_attn.o_a_proj                 [[g_o, B, n_h*d_head/g_o], [g_o, n_h*d_head/g_o, d_g]] -> [[g_o, B, d_g]]  w=[g_o*d_g, n_h*d_head/g_o]
 123 matmul           self_attn.o_b_proj                 [[B, g_o*d_g], [g_o*d_g, d_model]] -> [[B, d_model]]  w=[d_model, g_o*d_g]
 124 elementwise_mul  model.layers.3                     [[B, 1, n_hc, 1], [B, 1, 1, d_model]] -> [[B, 1, n_hc, d_model]]
 125 batched_matmul   model.layers.3                     [[B, n_hc, n_hc], [B, n_hc, d_model]] -> [[B, n_hc, d_model]]
 126 elementwise_add  model.layers.3                     [[B, 1, n_hc, d_model], [B, 1, n_hc, d_model]] -> [[B, 1, n_hc, d_model]]
 127 rmsnorm          ffn_hc.input_norm                  [[B, 1, n_hc*d_model]] -> [[B, 1, n_hc*d_model]]
 128 matmul           ffn_hc                             [[B, n_hc*d_model], [n_hc*d_model, (2+n_hc)*n_hc]] -> [[B, (2+n_hc)*n_hc]]  w=[(2+n_hc)*n_hc, n_hc*d_model]
 129 sigmoid          ffn_hc                             [[B, 1, n_hc]] -> [[B, 1, n_hc]]
 130 sigmoid          ffn_hc                             [[B, 1, n_hc]] -> [[B, 1, n_hc]]
 131 softmax          ffn_hc                             [[B, 1, n_hc, n_hc]] -> [[B, 1, n_hc, n_hc]]
 132 elementwise_mul  ffn_hc                             [[B, 1, n_hc, 1], [B, 1, n_hc, d_model]] -> [[B, 1, n_hc, d_model]]
 133 sum              ffn_hc                             [[B, 1, n_hc, d_model]] -> [[B, 1, d_model]]
 134 rmsnorm          post_attention_layernorm           [[B, 1, d_model]] -> [[B, 1, d_model]]  w=[d_model]
 135 matmul           mlp.gate                           [[B, d_model], [d_model, E]] -> [[B, E]]  w=[E, d_model]
 136 grouped_matmul   mlp.experts                        [[B*k, d_model], [E, d_model, 2*d_moe], [E]] -> [[B*k, 2*d_moe]]  w=[E, 2*d_moe, d_model]
 137 silu             mlp.experts.act_fn                 [[B*k, d_moe]] -> [[B*k, d_moe]]
 138 elementwise_mul  mlp.experts                        [[B*k, d_moe], [B*k, d_moe]] -> [[B*k, d_moe]]
 139 grouped_matmul   mlp.experts                        [[B*k, d_moe], [E, d_moe, d_model], [E]] -> [[B*k, d_model]]  w=[E, d_model, d_moe]
 140 elementwise_mul  mlp.experts                        [[B*k, d_model], [B*k, 1]] -> [[B*k, d_model]]
 141 sum              mlp.experts                        [[B, k, d_model]] -> [[B, d_model]]
 142 matmul           mlp.shared_experts.gate_proj       [[B, d_model], [d_model, d_moe]] -> [[B, d_moe]]  w=[d_moe, d_model]
 143 matmul           mlp.shared_experts.up_proj         [[B, d_model], [d_model, d_moe]] -> [[B, d_moe]]  w=[d_moe, d_model]
 144 silu             mlp.shared_experts.act_fn          [[B, 1, d_moe]] -> [[B, 1, d_moe]]
 145 elementwise_mul  mlp.shared_experts                 [[B, 1, d_moe], [B, 1, d_moe]] -> [[B, 1, d_moe]]
 146 matmul           mlp.shared_experts.down_proj       [[B, d_moe], [d_moe, d_model]] -> [[B, d_model]]  w=[d_model, d_moe]
 147 elementwise_add  mlp                                [[B, 1, d_model], [B, 1, d_model]] -> [[B, 1, d_model]]
 148 elementwise_mul  model.layers.3                     [[B, 1, n_hc, 1], [B, 1, 1, d_model]] -> [[B, 1, n_hc, d_model]]
 149 batched_matmul   model.layers.3                     [[B, n_hc, n_hc], [B, n_hc, d_model]] -> [[B, n_hc, d_model]]
 150 elementwise_add  model.layers.3                     [[B, 1, n_hc, d_model], [B, 1, n_hc, d_model]] -> [[B, 1, n_hc, d_model]]
```

**`block_type=MLA+MoE`  repeat=29  layers=`4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60`**  (54행)

```
 151 rmsnorm          attn_hc.input_norm                 [[B, 1, n_hc*d_model]] -> [[B, 1, n_hc*d_model]]
 152 matmul           attn_hc                            [[B, n_hc*d_model], [n_hc*d_model, (2+n_hc)*n_hc]] -> [[B, (2+n_hc)*n_hc]]  w=[(2+n_hc)*n_hc, n_hc*d_model]
 153 sigmoid          attn_hc                            [[B, 1, n_hc]] -> [[B, 1, n_hc]]
 154 sigmoid          attn_hc                            [[B, 1, n_hc]] -> [[B, 1, n_hc]]
 155 softmax          attn_hc                            [[B, 1, n_hc, n_hc]] -> [[B, 1, n_hc, n_hc]]
 156 elementwise_mul  attn_hc                            [[B, 1, n_hc, 1], [B, 1, n_hc, d_model]] -> [[B, 1, n_hc, d_model]]
 157 sum              attn_hc                            [[B, 1, n_hc, d_model]] -> [[B, 1, d_model]]
 158 rmsnorm          input_layernorm                    [[B, 1, d_model]] -> [[B, 1, d_model]]  w=[d_model]
 159 matmul           self_attn.q_a_proj                 [[B, d_model], [d_model, c_q]] -> [[B, c_q]]  w=[c_q, d_model]
 160 rmsnorm          self_attn.q_a_norm                 [[B, 1, c_q]] -> [[B, 1, c_q]]  w=[c_q]
 161 matmul           self_attn.q_b_proj                 [[B, c_q], [c_q, n_h*d_head]] -> [[B, n_h*d_head]]  w=[n_h*d_head, c_q]
 162 rmsnorm          self_attn.q_b_norm                 [[B, n_h, 1, d_head]] -> [[B, n_h, 1, d_head]]
 163 matmul           self_attn.kv_proj                  [[B, d_model], [d_model, d_head]] -> [[B, d_head]]  w=[d_head, d_model]
 164 rmsnorm          self_attn.kv_norm                  [[B, 1, d_head]] -> [[B, 1, d_head]]  w=[d_head]
 165 matmul           self_attn.compressor.kv_proj       [[B, d_model], [d_model, 2*d_head]] -> [[B, 2*d_head]]  w=[2*d_head, d_model]
 166 matmul           self_attn.compressor.gate_proj     [[B, d_model], [d_model, 2*d_head]] -> [[B, 2*d_head]]  w=[2*d_head, d_model]
 167 matmul           self_attn.compressor.indexer.kv_pr [[B, d_model], [d_model, 2*c_I]] -> [[B, 2*c_I]]  w=[2*c_I, d_model]
 168 matmul           self_attn.compressor.indexer.gate_ [[B, d_model], [d_model, 2*c_I]] -> [[B, 2*c_I]]  w=[2*c_I, d_model]
 169 matmul           self_attn.compressor.indexer.q_b_p [[B, c_q], [c_q, n_h_I*c_I]] -> [[B, n_h_I*c_I]]  w=[n_h_I*c_I, c_q]
 170 batched_matmul   self_attn.compressor.indexer.score [[B, n_h_I, c_I], [B, c_I, T/m_csa]] -> [[B, n_h_I, T/m_csa]]
 171 relu             self_attn.compressor.indexer.score [[B, 1, n_h_I, T/m_csa]] -> [[B, 1, n_h_I, T/m_csa]]
 172 matmul           self_attn.compressor.indexer.score [[B, d_model], [d_model, n_h_I]] -> [[B, n_h_I]]  w=[n_h_I, d_model]
 173 batched_matmul   self_attn                          [[B*n_h, 1, d_head], [B*n_h, d_head, w_local+T/m_csa]] -> [[B*n_h, 1, w_local+T/m_csa]]
 174 softmax          self_attn                          [[B, n_h, 1, w_local+T/m_csa+1]] -> [[B, n_h, 1, w_local+T/m_csa+1]]
 175 batched_matmul   self_attn                          [[B*n_h, 1, w_local+T/m_csa], [B*n_h, w_local+T/m_csa, d_head]] -> [[B*n_h, 1, d_head]]
 176 batched_matmul   self_attn.o_a_proj                 [[g_o, B, n_h*d_head/g_o], [g_o, n_h*d_head/g_o, d_g]] -> [[g_o, B, d_g]]  w=[g_o*d_g, n_h*d_head/g_o]
 177 matmul           self_attn.o_b_proj                 [[B, g_o*d_g], [g_o*d_g, d_model]] -> [[B, d_model]]  w=[d_model, g_o*d_g]
 178 elementwise_mul  model.layers.4                     [[B, 1, n_hc, 1], [B, 1, 1, d_model]] -> [[B, 1, n_hc, d_model]]
 179 batched_matmul   model.layers.4                     [[B, n_hc, n_hc], [B, n_hc, d_model]] -> [[B, n_hc, d_model]]
 180 elementwise_add  model.layers.4                     [[B, 1, n_hc, d_model], [B, 1, n_hc, d_model]] -> [[B, 1, n_hc, d_model]]
 181 rmsnorm          ffn_hc.input_norm                  [[B, 1, n_hc*d_model]] -> [[B, 1, n_hc*d_model]]
 182 matmul           ffn_hc                             [[B, n_hc*d_model], [n_hc*d_model, (2+n_hc)*n_hc]] -> [[B, (2+n_hc)*n_hc]]  w=[(2+n_hc)*n_hc, n_hc*d_model]
 183 sigmoid          ffn_hc                             [[B, 1, n_hc]] -> [[B, 1, n_hc]]
 184 sigmoid          ffn_hc                             [[B, 1, n_hc]] -> [[B, 1, n_hc]]
 185 softmax          ffn_hc                             [[B, 1, n_hc, n_hc]] -> [[B, 1, n_hc, n_hc]]
 186 elementwise_mul  ffn_hc                             [[B, 1, n_hc, 1], [B, 1, n_hc, d_model]] -> [[B, 1, n_hc, d_model]]
 187 sum              ffn_hc                             [[B, 1, n_hc, d_model]] -> [[B, 1, d_model]]
 188 rmsnorm          post_attention_layernorm           [[B, 1, d_model]] -> [[B, 1, d_model]]  w=[d_model]
 189 matmul           mlp.gate                           [[B, d_model], [d_model, E]] -> [[B, E]]  w=[E, d_model]
 190 grouped_matmul   mlp.experts                        [[B*k, d_model], [E, d_model, 2*d_moe], [E]] -> [[B*k, 2*d_moe]]  w=[E, 2*d_moe, d_model]
 191 silu             mlp.experts.act_fn                 [[B*k, d_moe]] -> [[B*k, d_moe]]
 192 elementwise_mul  mlp.experts                        [[B*k, d_moe], [B*k, d_moe]] -> [[B*k, d_moe]]
 193 grouped_matmul   mlp.experts                        [[B*k, d_moe], [E, d_moe, d_model], [E]] -> [[B*k, d_model]]  w=[E, d_model, d_moe]
 194 elementwise_mul  mlp.experts                        [[B*k, d_model], [B*k, 1]] -> [[B*k, d_model]]
 195 sum              mlp.experts                        [[B, k, d_model]] -> [[B, d_model]]
 196 matmul           mlp.shared_experts.gate_proj       [[B, d_model], [d_model, d_moe]] -> [[B, d_moe]]  w=[d_moe, d_model]
 197 matmul           mlp.shared_experts.up_proj         [[B, d_model], [d_model, d_moe]] -> [[B, d_moe]]  w=[d_moe, d_model]
 198 silu             mlp.shared_experts.act_fn          [[B, 1, d_moe]] -> [[B, 1, d_moe]]
 199 elementwise_mul  mlp.shared_experts                 [[B, 1, d_moe], [B, 1, d_moe]] -> [[B, 1, d_moe]]
 200 matmul           mlp.shared_experts.down_proj       [[B, d_moe], [d_moe, d_model]] -> [[B, d_model]]  w=[d_model, d_moe]
 201 elementwise_add  mlp                                [[B, 1, d_model], [B, 1, d_model]] -> [[B, 1, d_model]]
 202 elementwise_mul  model.layers.4                     [[B, 1, n_hc, 1], [B, 1, 1, d_model]] -> [[B, 1, n_hc, d_model]]
 203 batched_matmul   model.layers.4                     [[B, n_hc, n_hc], [B, n_hc, d_model]] -> [[B, n_hc, d_model]]
 204 elementwise_add  model.layers.4                     [[B, 1, n_hc, d_model], [B, 1, n_hc, d_model]] -> [[B, 1, n_hc, d_model]]
```

**`block_type=norm`  repeat=1  layers=`-`**  (2행)

```
 205 rmsnorm          model.hc_head.input_norm           [[B, 1, n_hc*d_model]] -> [[B, 1, n_hc*d_model]]
 210 rmsnorm          model.norm                         [[B, 1, d_model]] -> [[B, 1, d_model]]  w=[d_model]
```

**`block_type=-`  repeat=1  layers=`-`**  (4행)

```
 206 matmul           model.hc_head                      [[B, n_hc*d_model], [n_hc*d_model, n_hc]] -> [[B, n_hc]]  w=[n_hc, n_hc*d_model]
 207 sigmoid          model.hc_head                      [[B, 1, n_hc]] -> [[B, 1, n_hc]]
 208 elementwise_mul  model.hc_head                      [[B, 1, n_hc, 1], [B, 1, n_hc, d_model]] -> [[B, 1, n_hc, d_model]]
 209 sum              model.hc_head                      [[B, 1, n_hc, d_model]] -> [[B, 1, d_model]]
```

**`block_type=head`  repeat=1  layers=`-`**  (1행)

```
 211 matmul           lm_head                            [[B, d_model], [d_model, V]] -> [[B, V]]  w=[V, d_model]
```

