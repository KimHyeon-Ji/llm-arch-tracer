# 부록 — 표가 주장하는 구조 (모델별)

각 모델의 `prefill.csv` 에서 **block_type 별 대표 1 반복**만 뽑은 것입니다.
이것이 우리 표가 그 모델의 아키텍처라고 주장하는 내용 전부입니다.
한 줄씩 공식 소스와 대조해 주십시오.

## meta-llama/Llama-4-Maverick-17B-128E-Instruct

발행 좌표 **B=3, T=17**. 심볼 표:

```
B=3, E=128, T=17, V=202048, chunk_size=8192, ctx=262144, d_ff=16384, d_head=128, d_model=5120, d_moe=8192, n_h=40, n_kv=8
```

### `block_type = embed`  — repeat 1, layers `-`

```
embedding        model.embed_tokens                 [[V, d_model], [B, T]] -> [[B, T, d_model]]  w=[V, d_model]
```

### `block_type = attn+FFN`  — repeat 24, layers `0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46`

```
rmsnorm          input_layernorm                    [[B, T, d_model]] -> [[B, T, d_model]]  w=[d_model]
matmul           self_attn.q_proj                   [[B*T, d_model], [d_model, n_h*d_head]] -> [[B*T, n_h*d_head]]  w=[n_h*d_head, d_model]
matmul           self_attn.k_proj                   [[B*T, d_model], [d_model, n_kv*d_head]] -> [[B*T, n_kv*d_head]]  w=[n_kv*d_head, d_model]
matmul           self_attn.v_proj                   [[B*T, d_model], [d_model, n_kv*d_head]] -> [[B*T, n_kv*d_head]]  w=[n_kv*d_head, d_model]
batched_matmul   self_attn                          [[B*n_h, T, d_head], [B*n_h, d_head, T]] -> [[B*n_h, T, T]]
softmax          self_attn                          [[B, n_h, T, T]] -> [[B, n_h, T, T]]
batched_matmul   self_attn                          [[B*n_h, T, T], [B*n_h, T, d_head]] -> [[B*n_h, T, d_head]]
matmul           self_attn.o_proj                   [[B*T, n_h*d_head], [n_h*d_head, d_model]] -> [[B*T, d_model]]  w=[d_model, n_h*d_head]
elementwise_add  model.layers.0                     [[B, T, d_model], [B, T, d_model]] -> [[B, T, d_model]]
rmsnorm          post_attention_layernorm           [[B, T, d_model]] -> [[B, T, d_model]]  w=[d_model]
matmul           feed_forward.gate_proj             [[B*T, d_model], [d_model, d_ff]] -> [[B*T, d_ff]]  w=[d_ff, d_model]
silu             feed_forward.activation_fn         [[B, T, d_ff]] -> [[B, T, d_ff]]
matmul           feed_forward.up_proj               [[B*T, d_model], [d_model, d_ff]] -> [[B*T, d_ff]]  w=[d_ff, d_model]
elementwise_mul  feed_forward                       [[B, T, d_ff], [B, T, d_ff]] -> [[B, T, d_ff]]
matmul           feed_forward.down_proj             [[B*T, d_ff], [d_ff, d_model]] -> [[B*T, d_model]]  w=[d_model, d_ff]
elementwise_add  model.layers.0                     [[B, T, d_model], [B, T, d_model]] -> [[B, T, d_model]]
```

### `block_type = attn+MoE`  — repeat 12, layers `1,5,9,13,17,21,25,29,33,37,41,45`

```
rmsnorm          input_layernorm                    [[B, T, d_model]] -> [[B, T, d_model]]  w=[d_model]
matmul           self_attn.q_proj                   [[B*T, d_model], [d_model, n_h*d_head]] -> [[B*T, n_h*d_head]]  w=[n_h*d_head, d_model]
matmul           self_attn.k_proj                   [[B*T, d_model], [d_model, n_kv*d_head]] -> [[B*T, n_kv*d_head]]  w=[n_kv*d_head, d_model]
matmul           self_attn.v_proj                   [[B*T, d_model], [d_model, n_kv*d_head]] -> [[B*T, n_kv*d_head]]  w=[n_kv*d_head, d_model]
batched_matmul   self_attn                          [[B*n_h, T, d_head], [B*n_h, d_head, T]] -> [[B*n_h, T, T]]
softmax          self_attn                          [[B, n_h, T, T]] -> [[B, n_h, T, T]]
batched_matmul   self_attn                          [[B*n_h, T, T], [B*n_h, T, d_head]] -> [[B*n_h, T, d_head]]
matmul           self_attn.o_proj                   [[B*T, n_h*d_head], [n_h*d_head, d_model]] -> [[B*T, d_model]]  w=[d_model, n_h*d_head]
elementwise_add  model.layers.1                     [[B, T, d_model], [B, T, d_model]] -> [[B, T, d_model]]
rmsnorm          post_attention_layernorm           [[B, T, d_model]] -> [[B, T, d_model]]  w=[d_model]
matmul           feed_forward.router                [[B*T, d_model], [d_model, E]] -> [[B*T, E]]  w=[E, d_model]
sigmoid          feed_forward.router                [[B*T, E]] -> [[B*T, E]]
elementwise_mul  feed_forward                       [[B*E*T, d_model], [B*E*T, 1]] -> [[B*E*T, d_model]]
batched_matmul   feed_forward.experts               [[E, B*T, d_model], [E, d_model, 2*d_moe]] -> [[E, B*T, 2*d_moe]]  w=[E, d_model, 2*d_moe]
silu             feed_forward.experts.act_fn        [[E, B*T, d_moe]] -> [[E, B*T, d_moe]]
elementwise_mul  feed_forward.experts               [[E, B*T, d_moe], [E, B*T, d_moe]] -> [[E, B*T, d_moe]]
batched_matmul   feed_forward.experts               [[E, B*T, d_moe], [E, d_moe, d_model]] -> [[E, B*T, d_model]]  w=[E, d_moe, d_model]
matmul           feed_forward.shared_expert.gate_pr [[B*T, d_model], [d_model, d_moe]] -> [[B*T, d_moe]]  w=[d_moe, d_model]
silu             feed_forward.shared_expert.activat [[B*T, d_moe]] -> [[B*T, d_moe]]
matmul           feed_forward.shared_expert.up_proj [[B*T, d_model], [d_model, d_moe]] -> [[B*T, d_moe]]  w=[d_moe, d_model]
elementwise_mul  feed_forward.shared_expert         [[B*T, d_moe], [B*T, d_moe]] -> [[B*T, d_moe]]
matmul           feed_forward.shared_expert.down_pr [[B*T, d_moe], [d_moe, d_model]] -> [[B*T, d_model]]  w=[d_model, d_moe]
sum              feed_forward                       [[E, B*T, d_model]] -> [[B*T, d_model]]
elementwise_add  model.layers.1                     [[B, T, d_model], [B, T, d_model]] -> [[B, T, d_model]]
```

### `block_type = norm`  — repeat 1, layers `-`

```
rmsnorm          model.norm                         [[B, T, d_model]] -> [[B, T, d_model]]  w=[d_model]
```

### `block_type = head`  — repeat 1, layers `-`

```
matmul           lm_head                            [[B*T, d_model], [d_model, V]] -> [[B*T, V]]  w=[V, d_model]
```

## openai/gpt-oss-20b

발행 좌표 **B=3, T=264**. 심볼 표:

```
B=3, E=32, T=264, V=201088, ctx=131072, d_head=64, d_model=2880, d_moe=2880, k=4, n_h=64, n_kv=8, w_local=128
```

### `block_type = embed`  — repeat 1, layers `-`

```
embedding        model.embed_tokens                 [[V, d_model], [B, T]] -> [[B, T, d_model]]  w=[V, d_model]
```

### `block_type = attn+MoE`  — repeat 12, layers `0,2,4,6,8,10,12,14,16,18,20,22`

```
rmsnorm          input_layernorm                    [[B, T, d_model]] -> [[B, T, d_model]]  w=[d_model]
linear           self_attn.q_proj                   [[n_h*d_head], [B*T, d_model], [d_model, n_h*d_head]] -> [[B*T, n_h*d_head]]  w=[n_h*d_head, d_model]
linear           self_attn.k_proj                   [[n_kv*d_head], [B*T, d_model], [d_model, n_kv*d_head]] -> [[B*T, n_kv*d_head]]  w=[n_kv*d_head, d_model]
linear           self_attn.v_proj                   [[n_kv*d_head], [B*T, d_model], [d_model, n_kv*d_head]] -> [[B*T, n_kv*d_head]]  w=[n_kv*d_head, d_model]
batched_matmul   self_attn                          [[B*n_h, T, d_head], [B*n_h, d_head, T]] -> [[B*n_h, T, T]]
softmax          self_attn                          [[B, n_h, T, T+1]] -> [[B, n_h, T, T+1]]
batched_matmul   self_attn                          [[B*n_h, T, T], [B*n_h, T, d_head]] -> [[B*n_h, T, d_head]]
linear           self_attn.o_proj                   [[d_model], [B*T, n_h*d_head], [n_h*d_head, d_model]] -> [[B*T, d_model]]  w=[d_model, n_h*d_head]
elementwise_add  model.layers.0                     [[B, T, d_model], [B, T, d_model]] -> [[B, T, d_model]]
rmsnorm          post_attention_layernorm           [[B, T, d_model]] -> [[B, T, d_model]]  w=[d_model]
linear           mlp.router                         [[E], [B*T, d_model], [d_model, E]] -> [[B*T, E]]  w=[E, d_model]
softmax          mlp.router                         [[B*T, k]] -> [[B*T, k]]
grouped_matmul   mlp.experts                        [[B*k*T, d_model], [E, d_model, 2*d_moe], [E]] -> [[B*k*T, 2*d_moe]]  w=[E, d_model, 2*d_moe]
elementwise_mul  mlp.experts                        [[B*k*T, d_moe]] -> [[B*k*T, d_moe]]
sigmoid          mlp.experts                        [[B*k*T, d_moe]] -> [[B*k*T, d_moe]]
elementwise_mul  mlp.experts                        [[B*k*T, d_moe], [B*k*T, d_moe]] -> [[B*k*T, d_moe]]
elementwise_add  mlp.experts                        [[B*k*T, d_moe]] -> [[B*k*T, d_moe]]
elementwise_mul  mlp.experts                        [[B*k*T, d_moe], [B*k*T, d_moe]] -> [[B*k*T, d_moe]]
grouped_matmul   mlp.experts                        [[B*k*T, d_moe], [E, d_moe, d_model], [E]] -> [[B*k*T, d_model]]  w=[E, d_moe, d_model]
elementwise_mul  mlp.experts                        [[B*k*T, d_model], [B*k*T, 1]] -> [[B*k*T, d_model]]
sum              mlp.experts                        [[B*T, k, d_model]] -> [[B*T, d_model]]
elementwise_add  model.layers.0                     [[B, T, d_model], [B, T, d_model]] -> [[B, T, d_model]]
```

### `block_type = norm`  — repeat 1, layers `-`

```
rmsnorm          model.norm                         [[B, T, d_model]] -> [[B, T, d_model]]  w=[d_model]
```

### `block_type = head`  — repeat 1, layers `-`

```
matmul           lm_head                            [[B*T, d_model], [d_model, V]] -> [[B*T, V]]  w=[V, d_model]
```

## openai/gpt-oss-120b

발행 좌표 **B=3, T=264**. 심볼 표:

```
B=3, E=128, T=264, V=201088, ctx=131072, d_head=64, d_model=2880, d_moe=2880, k=4, n_h=64, n_kv=8, w_local=128
```

### `block_type = embed`  — repeat 1, layers `-`

```
embedding        model.embed_tokens                 [[V, d_model], [B, T]] -> [[B, T, d_model]]  w=[V, d_model]
```

### `block_type = attn+MoE`  — repeat 18, layers `0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34`

```
rmsnorm          input_layernorm                    [[B, T, d_model]] -> [[B, T, d_model]]  w=[d_model]
linear           self_attn.q_proj                   [[n_h*d_head], [B*T, d_model], [d_model, n_h*d_head]] -> [[B*T, n_h*d_head]]  w=[n_h*d_head, d_model]
linear           self_attn.k_proj                   [[n_kv*d_head], [B*T, d_model], [d_model, n_kv*d_head]] -> [[B*T, n_kv*d_head]]  w=[n_kv*d_head, d_model]
linear           self_attn.v_proj                   [[n_kv*d_head], [B*T, d_model], [d_model, n_kv*d_head]] -> [[B*T, n_kv*d_head]]  w=[n_kv*d_head, d_model]
batched_matmul   self_attn                          [[B*n_h, T, d_head], [B*n_h, d_head, T]] -> [[B*n_h, T, T]]
softmax          self_attn                          [[B, n_h, T, T+1]] -> [[B, n_h, T, T+1]]
batched_matmul   self_attn                          [[B*n_h, T, T], [B*n_h, T, d_head]] -> [[B*n_h, T, d_head]]
linear           self_attn.o_proj                   [[d_model], [B*T, n_h*d_head], [n_h*d_head, d_model]] -> [[B*T, d_model]]  w=[d_model, n_h*d_head]
elementwise_add  model.layers.0                     [[B, T, d_model], [B, T, d_model]] -> [[B, T, d_model]]
rmsnorm          post_attention_layernorm           [[B, T, d_model]] -> [[B, T, d_model]]  w=[d_model]
linear           mlp.router                         [[E], [B*T, d_model], [d_model, E]] -> [[B*T, E]]  w=[E, d_model]
softmax          mlp.router                         [[B*T, k]] -> [[B*T, k]]
grouped_matmul   mlp.experts                        [[B*k*T, d_model], [E, d_model, 2*d_moe], [E]] -> [[B*k*T, 2*d_moe]]  w=[E, d_model, 2*d_moe]
elementwise_mul  mlp.experts                        [[B*k*T, d_moe]] -> [[B*k*T, d_moe]]
sigmoid          mlp.experts                        [[B*k*T, d_moe]] -> [[B*k*T, d_moe]]
elementwise_mul  mlp.experts                        [[B*k*T, d_moe], [B*k*T, d_moe]] -> [[B*k*T, d_moe]]
elementwise_add  mlp.experts                        [[B*k*T, d_moe]] -> [[B*k*T, d_moe]]
elementwise_mul  mlp.experts                        [[B*k*T, d_moe], [B*k*T, d_moe]] -> [[B*k*T, d_moe]]
grouped_matmul   mlp.experts                        [[B*k*T, d_moe], [E, d_moe, d_model], [E]] -> [[B*k*T, d_model]]  w=[E, d_moe, d_model]
elementwise_mul  mlp.experts                        [[B*k*T, d_model], [B*k*T, 1]] -> [[B*k*T, d_model]]
sum              mlp.experts                        [[B*T, k, d_model]] -> [[B*T, d_model]]
elementwise_add  model.layers.0                     [[B, T, d_model], [B, T, d_model]] -> [[B, T, d_model]]
```

### `block_type = norm`  — repeat 1, layers `-`

```
rmsnorm          model.norm                         [[B, T, d_model]] -> [[B, T, d_model]]  w=[d_model]
```

### `block_type = head`  — repeat 1, layers `-`

```
matmul           lm_head                            [[B*T, d_model], [d_model, V]] -> [[B*T, V]]  w=[V, d_model]
```

## deepseek-ai/DeepSeek-V4-Pro

발행 좌표 **B=3, T=2176**. 심볼 표:

```
B=3, E=384, T=2176, V=129280, c_I=128, c_q=1536, ctx=1048576, d_g=1024, d_head=512, d_model=7168, d_moe=3072, d_rope=64, g_o=16, k=6, k_I=1024, m_csa=4, m_hca=128, n_h=128, n_h_I=64, n_hc=4, w_local=128
```

### `block_type = embed`  — repeat 1, layers `-`

```
embedding        model.embed_tokens                 [[V, d_model], [B, T]] -> [[B, T, d_model]]  w=[V, d_model]
```

### `block_type = MLA+MoE`  — repeat 2, layers `0-1`

```
rmsnorm          attn_hc.input_norm                 [[B, T, n_hc*d_model]] -> [[B, T, n_hc*d_model]]
matmul           attn_hc                            [[B*T, n_hc*d_model], [n_hc*d_model, (2+n_hc)*n_hc]] -> [[B*T, (2+n_hc)*n_hc]]  w=[(2+n_hc)*n_hc, n_hc*d_model]
sigmoid          attn_hc                            [[B, T, n_hc]] -> [[B, T, n_hc]]
sigmoid          attn_hc                            [[B, T, n_hc]] -> [[B, T, n_hc]]
softmax          attn_hc                            [[B, T, n_hc, n_hc]] -> [[B, T, n_hc, n_hc]]
elementwise_mul  attn_hc                            [[B, T, n_hc, 1], [B, T, n_hc, d_model]] -> [[B, T, n_hc, d_model]]
sum              attn_hc                            [[B, T, n_hc, d_model]] -> [[B, T, d_model]]
rmsnorm          input_layernorm                    [[B, T, d_model]] -> [[B, T, d_model]]  w=[d_model]
matmul           self_attn.q_a_proj                 [[B*T, d_model], [d_model, c_q]] -> [[B*T, c_q]]  w=[c_q, d_model]
rmsnorm          self_attn.q_a_norm                 [[B, T, c_q]] -> [[B, T, c_q]]  w=[c_q]
matmul           self_attn.q_b_proj                 [[B*T, c_q], [c_q, n_h*d_head]] -> [[B*T, n_h*d_head]]  w=[n_h*d_head, c_q]
rmsnorm          self_attn.q_b_norm                 [[B, n_h, T, d_head]] -> [[B, n_h, T, d_head]]
matmul           self_attn.kv_proj                  [[B*T, d_model], [d_model, d_head]] -> [[B*T, d_head]]  w=[d_head, d_model]
rmsnorm          self_attn.kv_norm                  [[B, T, d_head]] -> [[B, T, d_head]]  w=[d_head]
matmul           self_attn.compressor.kv_proj       [[B*T, d_model], [d_model, d_head]] -> [[B*T, d_head]]  w=[d_head, d_model]
matmul           self_attn.compressor.gate_proj     [[B*T, d_model], [d_model, d_head]] -> [[B*T, d_head]]  w=[d_head, d_model]
softmax          self_attn.compressor               [[B, T/m_hca, m_hca, d_head]] -> [[B, T/m_hca, m_hca, d_head]]
rmsnorm          self_attn.compressor.kv_norm       [[B, T/m_hca, d_head]] -> [[B, T/m_hca, d_head]]  w=[d_head]
batched_matmul   self_attn                          [[B*n_h, T, d_head], [B*n_h, d_head, T+T/m_hca]] -> [[B*n_h, T, T+T/m_hca]]
softmax          self_attn                          [[B, n_h, T, T+T/m_hca+1]] -> [[B, n_h, T, T+T/m_hca+1]]
batched_matmul   self_attn                          [[B*n_h, T, T+T/m_hca], [B*n_h, T+T/m_hca, d_head]] -> [[B*n_h, T, d_head]]
batched_matmul   self_attn.o_a_proj                 [[g_o, B*T, n_h*d_head/g_o], [g_o, n_h*d_head/g_o, d_g]] -> [[g_o, B*T, d_g]]  w=[g_o*d_g, n_h*d_head/g_o]
matmul           self_attn.o_b_proj                 [[B*T, g_o*d_g], [g_o*d_g, d_model]] -> [[B*T, d_model]]  w=[d_model, g_o*d_g]
elementwise_mul  model.layers.0                     [[B, T, n_hc, 1], [B, T, 1, d_model]] -> [[B, T, n_hc, d_model]]
batched_matmul   model.layers.0                     [[B*T, n_hc, n_hc], [B*T, n_hc, d_model]] -> [[B*T, n_hc, d_model]]
elementwise_add  model.layers.0                     [[B, T, n_hc, d_model], [B, T, n_hc, d_model]] -> [[B, T, n_hc, d_model]]
rmsnorm          ffn_hc.input_norm                  [[B, T, n_hc*d_model]] -> [[B, T, n_hc*d_model]]
matmul           ffn_hc                             [[B*T, n_hc*d_model], [n_hc*d_model, (2+n_hc)*n_hc]] -> [[B*T, (2+n_hc)*n_hc]]  w=[(2+n_hc)*n_hc, n_hc*d_model]
sigmoid          ffn_hc                             [[B, T, n_hc]] -> [[B, T, n_hc]]
sigmoid          ffn_hc                             [[B, T, n_hc]] -> [[B, T, n_hc]]
softmax          ffn_hc                             [[B, T, n_hc, n_hc]] -> [[B, T, n_hc, n_hc]]
elementwise_mul  ffn_hc                             [[B, T, n_hc, 1], [B, T, n_hc, d_model]] -> [[B, T, n_hc, d_model]]
sum              ffn_hc                             [[B, T, n_hc, d_model]] -> [[B, T, d_model]]
rmsnorm          post_attention_layernorm           [[B, T, d_model]] -> [[B, T, d_model]]  w=[d_model]
matmul           mlp.gate                           [[B*T, d_model], [d_model, E]] -> [[B*T, E]]  w=[E, d_model]
grouped_matmul   mlp.experts                        [[B*k*T, d_model], [E, d_model, 2*d_moe], [E]] -> [[B*k*T, 2*d_moe]]  w=[E, 2*d_moe, d_model]
silu             mlp.experts.act_fn                 [[B*k*T, d_moe]] -> [[B*k*T, d_moe]]
elementwise_mul  mlp.experts                        [[B*k*T, d_moe], [B*k*T, d_moe]] -> [[B*k*T, d_moe]]
grouped_matmul   mlp.experts                        [[B*k*T, d_moe], [E, d_moe, d_model], [E]] -> [[B*k*T, d_model]]  w=[E, d_model, d_moe]
elementwise_mul  mlp.experts                        [[B*k*T, d_model], [B*k*T, 1]] -> [[B*k*T, d_model]]
sum              mlp.experts                        [[B*T, k, d_model]] -> [[B*T, d_model]]
matmul           mlp.shared_experts.gate_proj       [[B*T, d_model], [d_model, d_moe]] -> [[B*T, d_moe]]  w=[d_moe, d_model]
matmul           mlp.shared_experts.up_proj         [[B*T, d_model], [d_model, d_moe]] -> [[B*T, d_moe]]  w=[d_moe, d_model]
silu             mlp.shared_experts.act_fn          [[B, T, d_moe]] -> [[B, T, d_moe]]
elementwise_mul  mlp.shared_experts                 [[B, T, d_moe], [B, T, d_moe]] -> [[B, T, d_moe]]
matmul           mlp.shared_experts.down_proj       [[B*T, d_moe], [d_moe, d_model]] -> [[B*T, d_model]]  w=[d_model, d_moe]
elementwise_add  mlp                                [[B, T, d_model], [B, T, d_model]] -> [[B, T, d_model]]
elementwise_mul  model.layers.0                     [[B, T, n_hc, 1], [B, T, 1, d_model]] -> [[B, T, n_hc, d_model]]
batched_matmul   model.layers.0                     [[B*T, n_hc, n_hc], [B*T, n_hc, d_model]] -> [[B*T, n_hc, d_model]]
elementwise_add  model.layers.0                     [[B, T, n_hc, d_model], [B, T, n_hc, d_model]] -> [[B, T, n_hc, d_model]]
```

### `block_type = norm`  — repeat 1, layers `-`

```
rmsnorm          model.hc_head.input_norm           [[B, T, n_hc*d_model]] -> [[B, T, n_hc*d_model]]
rmsnorm          model.norm                         [[B, T, d_model]] -> [[B, T, d_model]]  w=[d_model]
```

### `block_type = -`  — repeat 1, layers `-`

```
matmul           model.hc_head                      [[B*T, n_hc*d_model], [n_hc*d_model, n_hc]] -> [[B*T, n_hc]]  w=[n_hc, n_hc*d_model]
sigmoid          model.hc_head                      [[B, T, n_hc]] -> [[B, T, n_hc]]
elementwise_mul  model.hc_head                      [[B, T, n_hc, 1], [B, T, n_hc, d_model]] -> [[B, T, n_hc, d_model]]
sum              model.hc_head                      [[B, T, n_hc, d_model]] -> [[B, T, d_model]]
```

### `block_type = head`  — repeat 1, layers `-`

```
matmul           lm_head                            [[B*T, d_model], [d_model, V]] -> [[B*T, V]]  w=[V, d_model]
```

