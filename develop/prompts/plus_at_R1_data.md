# R1 데이터 패킷 — Kimi-K3 발행본의 맨 정수 자리

생성 2026-09-27. **제안된 이름은 이 문서에 없다.**

## 심볼표 (트레이서가 낸 것)
```
{"B": 3, "V": 163840, "d_model": 7168, "d_ff": 33792, "d_moe": 3072, "E": 896, "E_shared": 2, "n_h": 96, "n_kv": 96, "d_head": 74, "d_nope": 128, "d_v": 128, "d_rope": 64, "c_kv": 512, "c_q": 1536, "k": 16, "d_conv": 4, "ctx": 1048576, "d_chunk": 64, "n_h_kda": 96, "d_head_kda": 128, "d_moe_lat": 3584, "T": 320}
```

## config 발췌
```
text_config.activation_situ_linear_beta = 25.0
text_config.architectures = ["KimiLinearForCausalLM"]
text_config.attn_res_block_size = 12
text_config.chunk_size_feed_forward = 0
text_config.cross_attention_hidden_size = null
text_config.head_dim = 74
text_config.hidden_act = "situ"
text_config.hidden_size = 7168
text_config.latent_moe_use_norm = true
text_config.linear_attn_config = {"full_attn_layers": [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 93], "gate_lower_bound": -5.0, "head_dim": 128, 
text_config.moe_intermediate_size = 3072
text_config.moe_layer_freq = 1
text_config.moe_renormalize = true
text_config.moe_router_activation_func = "sigmoid"
text_config.num_attention_heads = 96
text_config.num_expert_group = 1
text_config.num_experts = 896
text_config.num_experts_per_token = 16
text_config.num_hidden_layers = 93
text_config.num_key_value_heads = 96
text_config.num_nextn_predict_layers = 0
text_config.num_shared_experts = 2
text_config.output_hidden_states = false
text_config.pruned_heads = {}
text_config.qk_nope_head_dim = 128
text_config.qk_rope_head_dim = 64
text_config.routed_expert_hidden_size = 3584
text_config.v_head_dim = 128
```

## 트레이스 대체 기록 (adaptation_log)
```
{"attempt": 0, "error": "RuntimeError: Tensor.item() cannot be called on meta tensors", "remedy": "meta_to_fake"}
{"tier": 1, "remedy": "kda_torch_reference", "detail": "KDA traced through fla's OWN torch reference (naive_chunk_kda / naive_recurrent_kda / naive_kda_gate / naive_kda_lowerbound_gate) plus torch equivalents of ShortConvolution and FusedRMSNormGated. The Triton kernel the model runs on GPU is opaque to TorchDispatchMode, so no trace of it is possible; shapes here describe the reference implementation. NOTE (2026-08-25, external/Codex review of commits since 3c955a3a): `_wrap_kda` branches on `safe_gate`/`lower_bound` (Kimi-K3's own config sets `gate_lower_bound=-5.0`) and `ShortConvolution.fo
{"tier": 1, "remedy": "moe_infer_even_split", "expert_cap": 4, "experts_per_layer": 896, "affects": "(?:^|\\.)(?:experts\\.\\d+|block_sparse_moe)(?:\\.|$)", "detail": "KimiSparseMoeBlock.moe_infer drives its expert loop off `tokens_per_expert.cpu().numpy()`, i.e. off routing VALUES, which a shape-only trace does not have. Replaced with an even split of the sorted tokens across experts: the op structure (per-expert gate/up/down on [n, d_model]) is faithful, the per-expert token COUNT is not -- that is runtime data no single trace could report. Experts traced: 4 of 896 per layer; the rest are st
caveat: 이 MoE 블록의 전문가 디스패치는 대체됐다: 레이어당 전문가 896개 중 4개만 트레이스했고, 전문가에 들어가는 토큰 수는 정렬된 토큰을 균등 분할한 **대체값**이다 -- 실제 값은 라우팅이 정하는 런타임 데이터라 한 번의 트레이스로는 알 수 없다. 라우터(gate)·scatter·argsort·가중합은 모델 자기 코드 그대로이고, 전문가 projection 의 op 구성과 폭도 충실하다. 토큰 축 크기만 신뢰하면 안 된다
```

## prefill: 맨 정수가 든 shape 패턴 (37 종, 5104 자리)

| 자리수 | shape | 대표 module_path | op_type |
|---:|---|---|---|
| 2080 | `[B, n_h_kda, 5, d_chunk, d_head_kda]` | `model.layers.N.self_attn` | exp |
| 1932 | `[3840, d_moe]` | `model.layers.N.block_sparse_moe.experts.0.w1` | matmul |
| 368 | `[3840, d_moe_lat]` | `model.layers.N.block_sparse_moe.experts.0.w1` | matmul |
| 92 | `[3840, 2*d_moe]` | `model.layers.N.block_sparse_moe.experts.0` | concat |
| 50 | `[B*T, 2, d_model]` | `model.layers.N` | concat |
| 50 | `[B*T, 3, d_model]` | `model.layers.N` | concat |
| 50 | `[B*T, 4, d_model]` | `model.layers.N` | concat |
| 50 | `[B*T, 5, d_model]` | `model.layers.N` | concat |
| 50 | `[B*T, 6, d_model]` | `model.layers.N` | concat |
| 50 | `[B*T, 7, d_model]` | `model.layers.N` | concat |
| 49 | `[B*T, 8, d_model]` | `model.layers.N` | concat |
| 42 | `[B*T, 9, d_model]` | `model.layers.N` | concat |
| 18 | `[B*T, 2]` | `model.layers.N` | sum |
| 18 | `[B*T, 3]` | `model.layers.N` | sum |
| 18 | `[B*T, 4]` | `model.layers.N` | sum |
| 18 | `[B*T, 5]` | `model.layers.N` | sum |
| 18 | `[B*T, 6]` | `model.layers.N` | sum |
| 18 | `[B*T, 7]` | `model.layers.N` | sum |
| 18 | `[B*T, 8]` | `model.layers.N` | sum |
| 18 | `[B*T, 9]` | `model.layers.N` | sum |
| 6 | `[B*T, 2, 1]` | `model.layers.N` | elementwise_mul |
| 6 | `[B*T, 1, 2]` | `model.layers.N` | batched_matmul |
| 6 | `[B*T, 3, 1]` | `model.layers.N` | elementwise_mul |
| 6 | `[B*T, 1, 3]` | `model.layers.N` | batched_matmul |
| 6 | `[B*T, 4, 1]` | `model.layers.N` | elementwise_mul |
| 6 | `[B*T, 1, 4]` | `model.layers.N` | batched_matmul |
| 6 | `[B*T, 5, 1]` | `model.layers.N` | elementwise_mul |
| 6 | `[B*T, 1, 5]` | `model.layers.N` | batched_matmul |
| 6 | `[B*T, 6, 1]` | `model.layers.N` | elementwise_mul |
| 6 | `[B*T, 1, 6]` | `model.layers.N` | batched_matmul |
| 6 | `[B*T, 7, 1]` | `model.layers.N` | elementwise_mul |
| 6 | `[B*T, 1, 7]` | `model.layers.N` | batched_matmul |
| 6 | `[B*T, 8, 1]` | `model.layers.N` | elementwise_mul |
| 6 | `[B*T, 1, 8]` | `model.layers.N` | batched_matmul |
| 6 | `[B*T, 9, 1]` | `model.layers.N` | elementwise_mul |
| 6 | `[B*T, 1, 9]` | `model.layers.N` | batched_matmul |
| 1 | `[B*T, 0, d_model]` | `model.layers.N` | concat |

### prefill: 대표 자리의 이웃 문맥 (depends_on 으로 앞뒤 2 단계)

**대상 shape** `[B, n_h_kda, 5, d_chunk, d_head_kda]`  (op18)
```
op18      exp               model.layers.0.self_attn                            
          i=[[B, n_h_kda, 5, d_chunk, d_head_kda]]
          o=[[B, n_h_kda, 5, d_chunk, d_head_kda]]
op16      sigmoid           model.layers.0.self_attn                            
          i=[[B, T, n_h_kda, d_head_kda]]
          o=[[B, T, n_h_kda, d_head_kda]]
op13      matmul            model.layers.0.self_attn.f_b_proj                   
          i=[[B*T, d_head_kda], [d_head_kda, n_h_kda*d_head_kda]]
          o=[[B*T, n_h_kda*d_head_kda]]
op15      exp               model.layers.0.self_attn                            
          i=[[n_h_kda, 1]]
          o=[[n_h_kda, 1]]
소비 op19   batched_matmul    i=[[B*n_h_kda*n_chunk, d_chunk, d_head_kda], [B*n_h_kda*n_chunk, d_head_kda, 1]] o=[[B*n_h_kda*n_chunk, d_chunk, 1]]
소비 op119  batched_matmul    i=[[B*n_h_kda*n_chunk, d_chunk, d_head_kda], [B*n_h_kda*n_chunk, d_head_kda, 1]] o=[[B*n_h_kda*n_chunk, d_chunk, 1]]
소비 op182  batched_matmul    i=[[B*n_h_kda, d_chunk, d_head_kda], [B*n_h_kda, d_head_kda, 1]] o=[[B*n_h_kda, d_chunk, 1]]
```

**대상 shape** `[3840, d_moe]`  (op1691)
```
op1691    matmul            model.layers.1.block_sparse_moe.experts.0.w1        
          i=[[3840, d_moe_lat], [d_moe_lat, d_moe]]
          o=[[3840, d_moe]]
op1689    sigmoid           model.layers.1.block_sparse_moe.gate                
          i=[[B*T, E]]
          o=[[B*T, E]]
op1690    matmul            model.layers.1.block_sparse_moe.routed_expert_down_proj
          i=[[B*T, d_model], [d_model, d_moe_lat]]
          o=[[B*T, d_moe_lat]]
op1688    matmul            model.layers.1.block_sparse_moe.gate                
          i=[[B*T, d_model], [d_model, E]]
          o=[[B*T, E]]
op1687    rmsnorm           model.layers.1.post_attention_layernorm             
          i=[[B, T, d_model]]
          o=[[B, T, d_model]]
소비 op1693 concat            i=[[3840, d_moe], [3840, d_moe]] o=[[3840, 2*d_moe]]
소비 op11692 batched_matmul    i=[[B*n_h_kda, d_chunk, d_head_kda], [B*n_h_kda, d_head_kda, 1]] o=[[B*n_h_kda, d_chunk, 1]]
```

**대상 shape** `[3840, 2*d_moe]`  (op1693)
```
op1693    concat            model.layers.1.block_sparse_moe.experts.0           
          i=[[3840, d_moe], [3840, d_moe]]
          o=[[3840, 2*d_moe]]
op1691    matmul            model.layers.1.block_sparse_moe.experts.0.w1        
          i=[[3840, d_moe_lat], [d_moe_lat, d_moe]]
          o=[[3840, d_moe]]
op1692    matmul            model.layers.1.block_sparse_moe.experts.0.w3        
          i=[[3840, d_moe_lat], [d_moe_lat, d_moe]]
          o=[[3840, d_moe]]
op1689    sigmoid           model.layers.1.block_sparse_moe.gate                
          i=[[B*T, E]]
          o=[[B*T, E]]
op1690    matmul            model.layers.1.block_sparse_moe.routed_expert_down_proj
          i=[[B*T, d_model], [d_model, d_moe_lat]]
          o=[[B*T, d_moe_lat]]
op1689    sigmoid           model.layers.1.block_sparse_moe.gate                
          i=[[B*T, E]]
          o=[[B*T, E]]
소비 op1694 tanh              i=[[3840, d_moe]] o=[[3840, d_moe]]
소비 op1696 sigmoid           i=[[3840, d_moe]] o=[[3840, d_moe]]
소비 op1698 tanh              i=[[3840, d_moe]] o=[[3840, d_moe]]
```

**대상 shape** `[B*T, 2, d_model]`  (op827)
```
op827     concat            model.layers.0                                      
          i=[[B*T, 1, d_model], [B*T, 1, d_model]]
          o=[[B*T, 2, d_model]]
op1       concat            model.layers.0                                      
          i=[[B*T, 0, d_model], [B*T, 1, d_model]]
          o=[[B*T, 1, d_model]]
op826     matmul            model.layers.0.self_attn.o_proj                     
          i=[[B*T, n_h_kda*d_head_kda], [n_h_kda*d_head_kda, d_model]]
          o=[[B*T, d_model]]
op0       embedding         model.embed_tokens                                  
          i=[[V, d_model], [B, T]]
          o=[[B, T, d_model]]
op825     rmsnorm           model.layers.0.self_attn.o_norm                     
          i=[[B, T, n_h_kda, d_head_kda]]
          o=[[B, T, n_h_kda, d_head_kda]]
소비 op828  elementwise_mul   i=[[B*T, 2, d_model], [B*T, 2, 1]] o=[[B*T, 2, d_model]]
소비 op833  batched_matmul    i=[[B*T, 1, 2], [B*T, 2, d_model]] o=[[B*T, 1, d_model]]
소비 op1828 matmul            i=[[3840, d_moe], [d_moe, d_moe_lat]] o=[[3840, d_moe_lat]]
```

**대상 shape** `[B*T, 3, d_model]`  (op2680)
```
op2680    concat            model.layers.12                                     
          i=[[B*T, 2, d_model], [B*T, 1, d_model]]
          o=[[B*T, 3, d_model]]
op1854    concat            model.layers.12                                     
          i=[[B*T, 1, d_model], [B*T, 1, d_model]]
          o=[[B*T, 2, d_model]]
op2679    matmul            model.layers.12.self_attn.o_proj                    
          i=[[B*T, n_h_kda*d_head_kda], [n_h_kda*d_head_kda, d_model]]
          o=[[B*T, d_model]]
op1       concat            model.layers.0                                      
          i=[[B*T, 0, d_model], [B*T, 1, d_model]]
          o=[[B*T, 1, d_model]]
op1846    elementwise_add   model.layers.3                                      
          i=[[B, T, d_model], [B, T, d_model]]
          o=[[B, T, d_model]]
op2678    rmsnorm           model.layers.12.self_attn.o_norm                    
          i=[[B, T, n_h_kda, d_head_kda]]
          o=[[B, T, n_h_kda, d_head_kda]]
소비 op2681 elementwise_mul   i=[[B*T, 3, d_model], [B*T, 3, 1]] o=[[B*T, 3, d_model]]
소비 op2686 batched_matmul    i=[[B*T, 1, 3], [B*T, 3, d_model]] o=[[B*T, 1, d_model]]
소비 op12681 batched_matmul    i=[[B*n_h_kda, d_chunk, d_head_kda], [B*n_h_kda, d_head_kda, 1]] o=[[B*n_h_kda, d_chunk, 1]]
```

## decode: 맨 정수가 든 shape 패턴 (36 종, 3024 자리)

| 자리수 | shape | 대표 module_path | op_type |
|---:|---|---|---|
| 1932 | `[12, d_moe]` | `model.layers.N.block_sparse_moe.experts.0.w1` | matmul |
| 368 | `[12, d_moe_lat]` | `model.layers.N.block_sparse_moe.experts.0.w1` | matmul |
| 92 | `[12, 2*d_moe]` | `model.layers.N.block_sparse_moe.experts.0` | concat |
| 50 | `[B, 2, d_model]` | `model.layers.N` | concat |
| 50 | `[B, 3, d_model]` | `model.layers.N` | concat |
| 50 | `[B, 4, d_model]` | `model.layers.N` | concat |
| 50 | `[B, 5, d_model]` | `model.layers.N` | concat |
| 50 | `[B, 6, d_model]` | `model.layers.N` | concat |
| 50 | `[B, 7, d_model]` | `model.layers.N` | concat |
| 49 | `[B, 8, d_model]` | `model.layers.N` | concat |
| 42 | `[B, 9, d_model]` | `model.layers.N` | concat |
| 18 | `[B, 2]` | `model.layers.N` | sum |
| 18 | `[B, 3]` | `model.layers.N` | sum |
| 18 | `[B, 4]` | `model.layers.N` | sum |
| 18 | `[B, 5]` | `model.layers.N` | sum |
| 18 | `[B, 6]` | `model.layers.N` | sum |
| 18 | `[B, 7]` | `model.layers.N` | sum |
| 18 | `[B, 8]` | `model.layers.N` | sum |
| 18 | `[B, 9]` | `model.layers.N` | sum |
| 6 | `[B, 2, 1]` | `model.layers.N` | elementwise_mul |
| 6 | `[B, 1, 2]` | `model.layers.N` | batched_matmul |
| 6 | `[B, 3, 1]` | `model.layers.N` | elementwise_mul |
| 6 | `[B, 1, 3]` | `model.layers.N` | batched_matmul |
| 6 | `[B, 4, 1]` | `model.layers.N` | elementwise_mul |
| 6 | `[B, 1, 4]` | `model.layers.N` | batched_matmul |
| 6 | `[B, 5, 1]` | `model.layers.N` | elementwise_mul |
| 6 | `[B, 1, 5]` | `model.layers.N` | batched_matmul |
| 6 | `[B, 6, 1]` | `model.layers.N` | elementwise_mul |
| 6 | `[B, 1, 6]` | `model.layers.N` | batched_matmul |
| 6 | `[B, 7, 1]` | `model.layers.N` | elementwise_mul |
| 6 | `[B, 1, 7]` | `model.layers.N` | batched_matmul |
| 6 | `[B, 8, 1]` | `model.layers.N` | elementwise_mul |
| 6 | `[B, 1, 8]` | `model.layers.N` | batched_matmul |
| 6 | `[B, 9, 1]` | `model.layers.N` | elementwise_mul |
| 6 | `[B, 1, 9]` | `model.layers.N` | batched_matmul |
| 1 | `[B, 0, d_model]` | `model.layers.N` | concat |

### decode: 대표 자리의 이웃 문맥 (depends_on 으로 앞뒤 2 단계)

**대상 shape** `[12, d_moe]`  (op83)
```
op83      matmul            model.layers.1.block_sparse_moe.experts.0.w1        
          i=[[12, d_moe_lat], [d_moe_lat, d_moe]]
          o=[[12, d_moe]]
op81      sigmoid           model.layers.1.block_sparse_moe.gate                
          i=[[B, E]]
          o=[[B, E]]
op82      matmul            model.layers.1.block_sparse_moe.routed_expert_down_proj
          i=[[B, d_model], [d_model, d_moe_lat]]
          o=[[B, d_moe_lat]]
op80      matmul            model.layers.1.block_sparse_moe.gate                
          i=[[B, d_model], [d_model, E]]
          o=[[B, E]]
op79      rmsnorm           model.layers.1.post_attention_layernorm             
          i=[[B, 1, d_model]]
          o=[[B, 1, d_model]]
소비 op85   concat            i=[[12, d_moe], [12, d_moe]] o=[[12, 2*d_moe]]
소비 op186  elementwise_mul   i=[[12, d_moe], [12, d_moe]] o=[[12, d_moe]]
소비 op285  elementwise_mul   i=[[12, d_moe], [12, d_moe]] o=[[12, d_moe]]
```

**대상 shape** `[12, 2*d_moe]`  (op85)
```
op85      concat            model.layers.1.block_sparse_moe.experts.0           
          i=[[12, d_moe], [12, d_moe]]
          o=[[12, 2*d_moe]]
op83      matmul            model.layers.1.block_sparse_moe.experts.0.w1        
          i=[[12, d_moe_lat], [d_moe_lat, d_moe]]
          o=[[12, d_moe]]
op84      matmul            model.layers.1.block_sparse_moe.experts.0.w3        
          i=[[12, d_moe_lat], [d_moe_lat, d_moe]]
          o=[[12, d_moe]]
op81      sigmoid           model.layers.1.block_sparse_moe.gate                
          i=[[B, E]]
          o=[[B, E]]
op82      matmul            model.layers.1.block_sparse_moe.routed_expert_down_proj
          i=[[B, d_model], [d_model, d_moe_lat]]
          o=[[B, d_moe_lat]]
op81      sigmoid           model.layers.1.block_sparse_moe.gate                
          i=[[B, E]]
          o=[[B, E]]
소비 op86   tanh              i=[[12, d_moe]] o=[[12, d_moe]]
소비 op88   sigmoid           i=[[12, d_moe]] o=[[12, d_moe]]
소비 op90   tanh              i=[[12, d_moe]] o=[[12, d_moe]]
```

**대상 shape** `[B, 2, d_model]`  (op23)
```
op23      concat            model.layers.0                                      
          i=[[B, 1, d_model], [B, 1, d_model]]
          o=[[B, 2, d_model]]
op1       concat            model.layers.0                                      
          i=[[B, 0, d_model], [B, 1, d_model]]
          o=[[B, 1, d_model]]
op22      matmul            model.layers.0.self_attn.o_proj                     
          i=[[B, n_h_kda*d_head_kda], [n_h_kda*d_head_kda, d_model]]
          o=[[B, d_model]]
op0       embedding         model.embed_tokens                                  
          i=[[V, d_model], [B, 1]]
          o=[[B, 1, d_model]]
op21      rmsnorm           model.layers.0.self_attn.o_norm                     
          i=[[B, 1, n_h_kda, d_head_kda]]
          o=[[B, 1, n_h_kda, d_head_kda]]
소비 op24   elementwise_mul   i=[[B, 2, d_model], [B, 2, 1]] o=[[B, 2, d_model]]
소비 op29   batched_matmul    i=[[B, 1, 2], [B, 2, d_model]] o=[[B, 1, d_model]]
소비 op124  elementwise_mul   i=[[12, d_moe]] o=[[12, d_moe]]
```

**대상 shape** `[B, 3, d_model]`  (op268)
```
op268     concat            model.layers.12                                     
          i=[[B, 2, d_model], [B, 1, d_model]]
          o=[[B, 3, d_model]]
op246     concat            model.layers.12                                     
          i=[[B, 1, d_model], [B, 1, d_model]]
          o=[[B, 2, d_model]]
op267     matmul            model.layers.12.self_attn.o_proj                    
          i=[[B, n_h_kda*d_head_kda], [n_h_kda*d_head_kda, d_model]]
          o=[[B, d_model]]
op1       concat            model.layers.0                                      
          i=[[B, 0, d_model], [B, 1, d_model]]
          o=[[B, 1, d_model]]
op238     elementwise_add   model.layers.3                                      
          i=[[B, 1, d_model], [B, 1, d_model]]
          o=[[B, 1, d_model]]
op266     rmsnorm           model.layers.12.self_attn.o_norm                    
          i=[[B, 1, n_h_kda, d_head_kda]]
          o=[[B, 1, n_h_kda, d_head_kda]]
소비 op269  elementwise_mul   i=[[B, 3, d_model], [B, 3, 1]] o=[[B, 3, d_model]]
소비 op274  batched_matmul    i=[[B, 1, 3], [B, 3, d_model]] o=[[B, 1, d_model]]
소비 op1269 softmax           i=[[B, 6]] o=[[B, 6]]
```

**대상 shape** `[B, 4, d_model]`  (op566)
```
op566     concat            model.layers.24                                     
          i=[[B, 3, d_model], [B, 1, d_model]]
          o=[[B, 4, d_model]]
op544     concat            model.layers.24                                     
          i=[[B, 2, d_model], [B, 1, d_model]]
          o=[[B, 3, d_model]]
op565     matmul            model.layers.24.self_attn.o_proj                    
          i=[[B, n_h_kda*d_head_kda], [n_h_kda*d_head_kda, d_model]]
          o=[[B, d_model]]
op246     concat            model.layers.12                                     
          i=[[B, 1, d_model], [B, 1, d_model]]
          o=[[B, 2, d_model]]
op536     elementwise_add   model.layers.15                                     
          i=[[B, 1, d_model], [B, 1, d_model]]
          o=[[B, 1, d_model]]
op564     rmsnorm           model.layers.24.self_attn.o_norm                    
          i=[[B, 1, n_h_kda, d_head_kda]]
          o=[[B, 1, n_h_kda, d_head_kda]]
소비 op567  elementwise_mul   i=[[B, 4, d_model], [B, 4, 1]] o=[[B, 4, d_model]]
소비 op572  batched_matmul    i=[[B, 1, 4], [B, 4, d_model]] o=[[B, 1, d_model]]
소비 op1567 softmax           i=[[B, 7]] o=[[B, 7]]
```

