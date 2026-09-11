# Llama-4-Maverick 산출물 정확성 검토 요청

아래 네 파일이 최종 산출물입니다. **공식 자료와 소스를 직접 보고** 축 이름이 정확한지
판정해 주십시오. 저희 설명을 믿지 마시고 반증해 주시면 좋겠습니다.

```
models/meta-llama__Llama-4-Maverick-17B-128E/prefill.csv     (67행)
models/meta-llama__Llama-4-Maverick-17B-128E/decode.csv      (67행)
models/meta-llama__Llama-4-Maverick-17B-128E/prefill.jsonl   (같은 내용, 행당 JSON)
models/meta-llama__Llama-4-Maverick-17B-128E/decode.jsonl
```

## 이 표가 무엇인가

`TorchDispatchMode` 로 실제 forward 를 잡아 ATen op 단위로 기록한 뒤, 같은 구조가
반복되는 레이어를 접어 요약한 것입니다. shape 의 각 축은 **정수가 아니라 심볼**로
렌더됩니다. 검토 대상은 바로 그 축 이름입니다.

`repeat` 는 이 행이 몇 개 레이어에서 반복되는지, `layers` 는 그 레이어 번호입니다.

## 심볼 표 (이 모델에서 확정된 값)

```
L            = 48
d_model      = 5120
n_h          = 40
n_kv         = 8
d_head       = 128
d_ff         = 16384
V            = 202048
ctx          = 262144
E            = 128
E_shared     = 1
k            = 1
d_moe        = 8192
w_local      = 8192
```

레이어 스케줄(48층) 앞 12: ['chunked_attention', 'chunked_attention', 'chunked_attention', 'full_attention', 'chunked_attention', 'chunked_attention', 'chunked_attention', 'full_attention', 'chunked_attention', 'chunked_attention', 'chunked_attention', 'full_attention']

## prefill.csv

| op_id | block_type | repeat | h1 | h2 | op_type | input_shape | weight_shape | output_shape |
|---|---|---|---|---|---|---|---|---|
| 0 | embed | 1 | model | embed_tokens | embedding | [[V, d_model], [B, T]] | [V, d_model] | [[B, T, d_model]] |
| 1 | attn+FFN | 24 | input_layernorm |  | rmsnorm | [[B, T, d_model]] | [d_model] | [[B, T, d_model]] |
| 2 | attn+FFN | 24 | self_attn | q_proj | matmul | [[T, d_model], [d_model, n_h*d_head]] | [n_h*d_head, d_model] | [[T, n_h*d_head]] |
| 3 | attn+FFN | 24 | self_attn | k_proj | matmul | [[T, d_model], [d_model, n_kv*d_head]] | [n_kv*d_head, d_model] | [[T, n_kv*d_head]] |
| 4 | attn+FFN | 24 | self_attn | v_proj | matmul | [[T, d_model], [d_model, n_kv*d_head]] | [n_kv*d_head, d_model] | [[T, n_kv*d_head]] |
| 5 | attn+FFN | 24 | self_attn |  | batched_matmul | [[n_h, T, d_head], [n_h, d_head, T]] |  | [[n_h, T, T]] |
| 6 | attn+FFN | 24 | self_attn |  | softmax | [[B, n_h, T, T]] |  | [[B, n_h, T, T]] |
| 7 | attn+FFN | 24 | self_attn |  | batched_matmul | [[n_h, T, T], [n_h, T, d_head]] |  | [[n_h, T, d_head]] |
| 8 | attn+FFN | 24 | self_attn | o_proj | matmul | [[T, n_h*d_head], [n_h*d_head, d_model]] | [d_model, n_h*d_head] | [[T, d_model]] |
| 9 | attn+FFN | 24 |  |  | elementwise_add | [[B, T, d_model], [B, T, d_model]] |  | [[B, T, d_model]] |
| 10 | attn+FFN | 24 | post_attention_layernorm |  | rmsnorm | [[B, T, d_model]] | [d_model] | [[B, T, d_model]] |
| 11 | attn+FFN | 24 | feed_forward | gate_proj | matmul | [[T, d_model], [d_model, d_ff]] | [d_ff, d_model] | [[T, d_ff]] |
| 12 | attn+FFN | 24 | feed_forward | activation_fn | silu | [[B, T, d_ff]] |  | [[B, T, d_ff]] |
| 13 | attn+FFN | 24 | feed_forward | up_proj | matmul | [[T, d_model], [d_model, d_ff]] | [d_ff, d_model] | [[T, d_ff]] |
| 14 | attn+FFN | 24 | feed_forward |  | elementwise_mul | [[B, T, d_ff], [B, T, d_ff]] |  | [[B, T, d_ff]] |
| 15 | attn+FFN | 24 | feed_forward | down_proj | matmul | [[T, d_ff], [d_ff, d_model]] | [d_model, d_ff] | [[T, d_model]] |
| 16 | attn+FFN | 24 |  |  | elementwise_add | [[B, T, d_model], [B, T, d_model]] |  | [[B, T, d_model]] |
| 17 | attn+MoE | 12 | input_layernorm |  | rmsnorm | [[B, T, d_model]] | [d_model] | [[B, T, d_model]] |
| 18 | attn+MoE | 12 | self_attn | q_proj | matmul | [[T, d_model], [d_model, n_h*d_head]] | [n_h*d_head, d_model] | [[T, n_h*d_head]] |
| 19 | attn+MoE | 12 | self_attn | k_proj | matmul | [[T, d_model], [d_model, n_kv*d_head]] | [n_kv*d_head, d_model] | [[T, n_kv*d_head]] |
| 20 | attn+MoE | 12 | self_attn | v_proj | matmul | [[T, d_model], [d_model, n_kv*d_head]] | [n_kv*d_head, d_model] | [[T, n_kv*d_head]] |
| 21 | attn+MoE | 12 | self_attn |  | batched_matmul | [[n_h, T, d_head], [n_h, d_head, T]] |  | [[n_h, T, T]] |
| 22 | attn+MoE | 12 | self_attn |  | softmax | [[B, n_h, T, T]] |  | [[B, n_h, T, T]] |
| 23 | attn+MoE | 12 | self_attn |  | batched_matmul | [[n_h, T, T], [n_h, T, d_head]] |  | [[n_h, T, d_head]] |
| 24 | attn+MoE | 12 | self_attn | o_proj | matmul | [[T, n_h*d_head], [n_h*d_head, d_model]] | [d_model, n_h*d_head] | [[T, d_model]] |
| 25 | attn+MoE | 12 |  |  | elementwise_add | [[B, T, d_model], [B, T, d_model]] |  | [[B, T, d_model]] |
| 26 | attn+MoE | 12 | post_attention_layernorm |  | rmsnorm | [[B, T, d_model]] | [d_model] | [[B, T, d_model]] |
| 27 | attn+MoE | 12 | feed_forward | router | matmul | [[T, d_model], [d_model, E]] | [E, d_model] | [[T, E]] |
| 28 | attn+MoE | 12 | feed_forward | router | sigmoid | [[T, E]] |  | [[T, E]] |
| 29 | attn+MoE | 12 | feed_forward |  | elementwise_mul | [[E*T, d_model], [E*T, 1]] |  | [[E*T, d_model]] |
| 30 | attn+MoE | 12 | feed_forward | experts | batched_matmul | [[E, T, d_model], [E, d_model, 2*d_moe]] | [E, d_model, 2*d_moe] | [[E, T, 2*d_moe]] |
| 31 | attn+MoE | 12 | feed_forward | experts | silu | [[E, T, d_moe]] |  | [[E, T, d_moe]] |
| 32 | attn+MoE | 12 | feed_forward | experts | elementwise_mul | [[E, T, d_moe], [E, T, d_moe]] |  | [[E, T, d_moe]] |
| 33 | attn+MoE | 12 | feed_forward | experts | batched_matmul | [[E, T, d_moe], [E, d_moe, d_model]] | [E, d_moe, d_model] | [[E, T, d_model]] |
| 34 | attn+MoE | 12 | feed_forward | shared_expert | matmul | [[T, d_model], [d_model, d_moe]] | [d_moe, d_model] | [[T, d_moe]] |
| 35 | attn+MoE | 12 | feed_forward | shared_expert | silu | [[T, d_moe]] |  | [[T, d_moe]] |
| 36 | attn+MoE | 12 | feed_forward | shared_expert | matmul | [[T, d_model], [d_model, d_moe]] | [d_moe, d_model] | [[T, d_moe]] |
| 37 | attn+MoE | 12 | feed_forward | shared_expert | elementwise_mul | [[T, d_moe], [T, d_moe]] |  | [[T, d_moe]] |
| 38 | attn+MoE | 12 | feed_forward | shared_expert | matmul | [[T, d_moe], [d_moe, d_model]] | [d_model, d_moe] | [[T, d_model]] |
| 39 | attn+MoE | 12 | feed_forward |  | sum | [[E, T, d_model]] |  | [[T, d_model]] |
| 40 | attn+MoE | 12 |  |  | elementwise_add | [[B, T, d_model], [B, T, d_model]] |  | [[B, T, d_model]] |
| 41 | attn+MoE | 12 | input_layernorm |  | rmsnorm | [[B, T, d_model]] | [d_model] | [[B, T, d_model]] |
| 42 | attn+MoE | 12 | self_attn | q_proj | matmul | [[T, d_model], [d_model, n_h*d_head]] | [n_h*d_head, d_model] | [[T, n_h*d_head]] |
| 43 | attn+MoE | 12 | self_attn | k_proj | matmul | [[T, d_model], [d_model, n_kv*d_head]] | [n_kv*d_head, d_model] | [[T, n_kv*d_head]] |
| 44 | attn+MoE | 12 | self_attn | v_proj | matmul | [[T, d_model], [d_model, n_kv*d_head]] | [n_kv*d_head, d_model] | [[T, n_kv*d_head]] |
| 45 | attn+MoE | 12 | self_attn |  | batched_matmul | [[n_h, T, d_head], [n_h, d_head, T]] |  | [[n_h, T, T]] |
| 46 | attn+MoE | 12 | self_attn |  | softmax | [[B, n_h, T, T]] |  | [[B, n_h, T, T]] |
| 47 | attn+MoE | 12 | self_attn |  | batched_matmul | [[n_h, T, T], [n_h, T, d_head]] |  | [[n_h, T, d_head]] |
| 48 | attn+MoE | 12 | self_attn | o_proj | matmul | [[T, n_h*d_head], [n_h*d_head, d_model]] | [d_model, n_h*d_head] | [[T, d_model]] |
| 49 | attn+MoE | 12 |  |  | elementwise_add | [[B, T, d_model], [B, T, d_model]] |  | [[B, T, d_model]] |
| 50 | attn+MoE | 12 | post_attention_layernorm |  | rmsnorm | [[B, T, d_model]] | [d_model] | [[B, T, d_model]] |
| 51 | attn+MoE | 12 | feed_forward | router | matmul | [[T, d_model], [d_model, E]] | [E, d_model] | [[T, E]] |
| 52 | attn+MoE | 12 | feed_forward | router | sigmoid | [[T, E]] |  | [[T, E]] |
| 53 | attn+MoE | 12 | feed_forward |  | elementwise_mul | [[E*T, d_model], [E*T, 1]] |  | [[E*T, d_model]] |
| 54 | attn+MoE | 12 | feed_forward | experts | batched_matmul | [[E, T, d_model], [E, d_model, 2*d_moe]] | [E, d_model, 2*d_moe] | [[E, T, 2*d_moe]] |
| 55 | attn+MoE | 12 | feed_forward | experts | silu | [[E, T, d_moe]] |  | [[E, T, d_moe]] |
| 56 | attn+MoE | 12 | feed_forward | experts | elementwise_mul | [[E, T, d_moe], [E, T, d_moe]] |  | [[E, T, d_moe]] |
| 57 | attn+MoE | 12 | feed_forward | experts | batched_matmul | [[E, T, d_moe], [E, d_moe, d_model]] | [E, d_moe, d_model] | [[E, T, d_model]] |
| 58 | attn+MoE | 12 | feed_forward | shared_expert | matmul | [[T, d_model], [d_model, d_moe]] | [d_moe, d_model] | [[T, d_moe]] |
| 59 | attn+MoE | 12 | feed_forward | shared_expert | silu | [[T, d_moe]] |  | [[T, d_moe]] |
| 60 | attn+MoE | 12 | feed_forward | shared_expert | matmul | [[T, d_model], [d_model, d_moe]] | [d_moe, d_model] | [[T, d_moe]] |
| 61 | attn+MoE | 12 | feed_forward | shared_expert | elementwise_mul | [[T, d_moe], [T, d_moe]] |  | [[T, d_moe]] |
| 62 | attn+MoE | 12 | feed_forward | shared_expert | matmul | [[T, d_moe], [d_moe, d_model]] | [d_model, d_moe] | [[T, d_model]] |
| 63 | attn+MoE | 12 | feed_forward |  | sum | [[E, T, d_model]] |  | [[T, d_model]] |
| 64 | attn+MoE | 12 |  |  | elementwise_add | [[B, T, d_model], [B, T, d_model]] |  | [[B, T, d_model]] |
| 65 | norm | 1 | model | norm | rmsnorm | [[B, T, d_model]] | [d_model] | [[B, T, d_model]] |
| 66 | head | 1 | lm_head |  | matmul | [[T, d_model], [d_model, V]] | [V, d_model] | [[T, V]] |

## decode.csv

| op_id | block_type | repeat | h1 | h2 | op_type | input_shape | weight_shape | output_shape |
|---|---|---|---|---|---|---|---|---|
| 0 | embed | 1 | model | embed_tokens | embedding | [[V, d_model], [B, 1]] | [V, d_model] | [[B, 1, d_model]] |
| 1 | attn+FFN | 24 | input_layernorm |  | rmsnorm | [[B, 1, d_model]] | [d_model] | [[B, 1, d_model]] |
| 2 | attn+FFN | 24 | self_attn | q_proj | matmul | [[B, d_model], [d_model, n_h*d_head]] | [n_h*d_head, d_model] | [[B, n_h*d_head]] |
| 3 | attn+FFN | 24 | self_attn | k_proj | matmul | [[B, d_model], [d_model, n_kv*d_head]] | [n_kv*d_head, d_model] | [[B, n_kv*d_head]] |
| 4 | attn+FFN | 24 | self_attn | v_proj | matmul | [[B, d_model], [d_model, n_kv*d_head]] | [n_kv*d_head, d_model] | [[B, n_kv*d_head]] |
| 5 | attn+FFN | 24 | self_attn |  | batched_matmul | [[n_h, B, d_head], [n_h, d_head, T+1]] |  | [[n_h, B, T+1]] |
| 6 | attn+FFN | 24 | self_attn |  | softmax | [[B, n_h, 1, T+1]] |  | [[B, n_h, 1, T+1]] |
| 7 | attn+FFN | 24 | self_attn |  | batched_matmul | [[n_h, B, T+1], [n_h, T+1, d_head]] |  | [[n_h, B, d_head]] |
| 8 | attn+FFN | 24 | self_attn | o_proj | matmul | [[B, n_h*d_head], [n_h*d_head, d_model]] | [d_model, n_h*d_head] | [[B, d_model]] |
| 9 | attn+FFN | 24 |  |  | elementwise_add | [[B, 1, d_model], [B, 1, d_model]] |  | [[B, 1, d_model]] |
| 10 | attn+FFN | 24 | post_attention_layernorm |  | rmsnorm | [[B, 1, d_model]] | [d_model] | [[B, 1, d_model]] |
| 11 | attn+FFN | 24 | feed_forward | gate_proj | matmul | [[B, d_model], [d_model, d_ff]] | [d_ff, d_model] | [[B, d_ff]] |
| 12 | attn+FFN | 24 | feed_forward | activation_fn | silu | [[B, 1, d_ff]] |  | [[B, 1, d_ff]] |
| 13 | attn+FFN | 24 | feed_forward | up_proj | matmul | [[B, d_model], [d_model, d_ff]] | [d_ff, d_model] | [[B, d_ff]] |
| 14 | attn+FFN | 24 | feed_forward |  | elementwise_mul | [[B, 1, d_ff], [B, 1, d_ff]] |  | [[B, 1, d_ff]] |
| 15 | attn+FFN | 24 | feed_forward | down_proj | matmul | [[B, d_ff], [d_ff, d_model]] | [d_model, d_ff] | [[B, d_model]] |
| 16 | attn+FFN | 24 |  |  | elementwise_add | [[B, 1, d_model], [B, 1, d_model]] |  | [[B, 1, d_model]] |
| 17 | attn+MoE | 12 | input_layernorm |  | rmsnorm | [[B, 1, d_model]] | [d_model] | [[B, 1, d_model]] |
| 18 | attn+MoE | 12 | self_attn | q_proj | matmul | [[B, d_model], [d_model, n_h*d_head]] | [n_h*d_head, d_model] | [[B, n_h*d_head]] |
| 19 | attn+MoE | 12 | self_attn | k_proj | matmul | [[B, d_model], [d_model, n_kv*d_head]] | [n_kv*d_head, d_model] | [[B, n_kv*d_head]] |
| 20 | attn+MoE | 12 | self_attn | v_proj | matmul | [[B, d_model], [d_model, n_kv*d_head]] | [n_kv*d_head, d_model] | [[B, n_kv*d_head]] |
| 21 | attn+MoE | 12 | self_attn |  | batched_matmul | [[n_h, B, d_head], [n_h, d_head, T+1]] |  | [[n_h, B, T+1]] |
| 22 | attn+MoE | 12 | self_attn |  | softmax | [[B, n_h, 1, T+1]] |  | [[B, n_h, 1, T+1]] |
| 23 | attn+MoE | 12 | self_attn |  | batched_matmul | [[n_h, B, T+1], [n_h, T+1, d_head]] |  | [[n_h, B, d_head]] |
| 24 | attn+MoE | 12 | self_attn | o_proj | matmul | [[B, n_h*d_head], [n_h*d_head, d_model]] | [d_model, n_h*d_head] | [[B, d_model]] |
| 25 | attn+MoE | 12 |  |  | elementwise_add | [[B, 1, d_model], [B, 1, d_model]] |  | [[B, 1, d_model]] |
| 26 | attn+MoE | 12 | post_attention_layernorm |  | rmsnorm | [[B, 1, d_model]] | [d_model] | [[B, 1, d_model]] |
| 27 | attn+MoE | 12 | feed_forward | router | matmul | [[B, d_model], [d_model, E]] | [E, d_model] | [[B, E]] |
| 28 | attn+MoE | 12 | feed_forward | router | sigmoid | [[B, E]] |  | [[B, E]] |
| 29 | attn+MoE | 12 | feed_forward |  | elementwise_mul | [[E, d_model], [E, 1]] |  | [[E, d_model]] |
| 30 | attn+MoE | 12 | feed_forward | experts | batched_matmul | [[E, B, d_model], [E, d_model, 2*d_moe]] | [E, d_model, 2*d_moe] | [[E, B, 2*d_moe]] |
| 31 | attn+MoE | 12 | feed_forward | experts | silu | [[E, B, d_moe]] |  | [[E, B, d_moe]] |
| 32 | attn+MoE | 12 | feed_forward | experts | elementwise_mul | [[E, B, d_moe], [E, B, d_moe]] |  | [[E, B, d_moe]] |
| 33 | attn+MoE | 12 | feed_forward | experts | batched_matmul | [[E, B, d_moe], [E, d_moe, d_model]] | [E, d_moe, d_model] | [[E, B, d_model]] |
| 34 | attn+MoE | 12 | feed_forward | shared_expert | matmul | [[B, d_model], [d_model, d_moe]] | [d_moe, d_model] | [[B, d_moe]] |
| 35 | attn+MoE | 12 | feed_forward | shared_expert | silu | [[B, d_moe]] |  | [[B, d_moe]] |
| 36 | attn+MoE | 12 | feed_forward | shared_expert | matmul | [[B, d_model], [d_model, d_moe]] | [d_moe, d_model] | [[B, d_moe]] |
| 37 | attn+MoE | 12 | feed_forward | shared_expert | elementwise_mul | [[B, d_moe], [B, d_moe]] |  | [[B, d_moe]] |
| 38 | attn+MoE | 12 | feed_forward | shared_expert | matmul | [[B, d_moe], [d_moe, d_model]] | [d_model, d_moe] | [[B, d_model]] |
| 39 | attn+MoE | 12 | feed_forward |  | sum | [[E, B, d_model]] |  | [[B, d_model]] |
| 40 | attn+MoE | 12 |  |  | elementwise_add | [[B, 1, d_model], [B, 1, d_model]] |  | [[B, 1, d_model]] |
| 41 | attn+MoE | 12 | input_layernorm |  | rmsnorm | [[B, 1, d_model]] | [d_model] | [[B, 1, d_model]] |
| 42 | attn+MoE | 12 | self_attn | q_proj | matmul | [[B, d_model], [d_model, n_h*d_head]] | [n_h*d_head, d_model] | [[B, n_h*d_head]] |
| 43 | attn+MoE | 12 | self_attn | k_proj | matmul | [[B, d_model], [d_model, n_kv*d_head]] | [n_kv*d_head, d_model] | [[B, n_kv*d_head]] |
| 44 | attn+MoE | 12 | self_attn | v_proj | matmul | [[B, d_model], [d_model, n_kv*d_head]] | [n_kv*d_head, d_model] | [[B, n_kv*d_head]] |
| 45 | attn+MoE | 12 | self_attn |  | batched_matmul | [[n_h, B, d_head], [n_h, d_head, T+1]] |  | [[n_h, B, T+1]] |
| 46 | attn+MoE | 12 | self_attn |  | softmax | [[B, n_h, 1, T+1]] |  | [[B, n_h, 1, T+1]] |
| 47 | attn+MoE | 12 | self_attn |  | batched_matmul | [[n_h, B, T+1], [n_h, T+1, d_head]] |  | [[n_h, B, d_head]] |
| 48 | attn+MoE | 12 | self_attn | o_proj | matmul | [[B, n_h*d_head], [n_h*d_head, d_model]] | [d_model, n_h*d_head] | [[B, d_model]] |
| 49 | attn+MoE | 12 |  |  | elementwise_add | [[B, 1, d_model], [B, 1, d_model]] |  | [[B, 1, d_model]] |
| 50 | attn+MoE | 12 | post_attention_layernorm |  | rmsnorm | [[B, 1, d_model]] | [d_model] | [[B, 1, d_model]] |
| 51 | attn+MoE | 12 | feed_forward | router | matmul | [[B, d_model], [d_model, E]] | [E, d_model] | [[B, E]] |
| 52 | attn+MoE | 12 | feed_forward | router | sigmoid | [[B, E]] |  | [[B, E]] |
| 53 | attn+MoE | 12 | feed_forward |  | elementwise_mul | [[E, d_model], [E, 1]] |  | [[E, d_model]] |
| 54 | attn+MoE | 12 | feed_forward | experts | batched_matmul | [[E, B, d_model], [E, d_model, 2*d_moe]] | [E, d_model, 2*d_moe] | [[E, B, 2*d_moe]] |
| 55 | attn+MoE | 12 | feed_forward | experts | silu | [[E, B, d_moe]] |  | [[E, B, d_moe]] |
| 56 | attn+MoE | 12 | feed_forward | experts | elementwise_mul | [[E, B, d_moe], [E, B, d_moe]] |  | [[E, B, d_moe]] |
| 57 | attn+MoE | 12 | feed_forward | experts | batched_matmul | [[E, B, d_moe], [E, d_moe, d_model]] | [E, d_moe, d_model] | [[E, B, d_model]] |
| 58 | attn+MoE | 12 | feed_forward | shared_expert | matmul | [[B, d_model], [d_model, d_moe]] | [d_moe, d_model] | [[B, d_moe]] |
| 59 | attn+MoE | 12 | feed_forward | shared_expert | silu | [[B, d_moe]] |  | [[B, d_moe]] |
| 60 | attn+MoE | 12 | feed_forward | shared_expert | matmul | [[B, d_model], [d_model, d_moe]] | [d_moe, d_model] | [[B, d_moe]] |
| 61 | attn+MoE | 12 | feed_forward | shared_expert | elementwise_mul | [[B, d_moe], [B, d_moe]] |  | [[B, d_moe]] |
| 62 | attn+MoE | 12 | feed_forward | shared_expert | matmul | [[B, d_moe], [d_moe, d_model]] | [d_model, d_moe] | [[B, d_model]] |
| 63 | attn+MoE | 12 | feed_forward |  | sum | [[E, B, d_model]] |  | [[B, d_model]] |
| 64 | attn+MoE | 12 |  |  | elementwise_add | [[B, 1, d_model], [B, 1, d_model]] |  | [[B, 1, d_model]] |
| 65 | norm | 1 | model | norm | rmsnorm | [[B, 1, d_model]] | [d_model] | [[B, 1, d_model]] |
| 66 | head | 1 | lm_head |  | matmul | [[B, d_model], [d_model, V]] | [V, d_model] | [[B, V]] |
---

## 확인해 주셨으면 하는 것

저희 판단을 먼저 적지 않겠습니다. **공식 자료(모델 카드 / HF config / 논문)와
`transformers` 5.14.1 설치본의 `modeling_llama4.py` 를 직접 보고** 판정해 주십시오.

### A. 심볼 값이 맞는가

위 심볼 표의 각 값이 Llama-4-Maverick 의 실제 설정과 맞습니까? 특히:

* `n_h = 40`, `n_kv = 8`, `d_head = 128` — GQA 비율이 5:1 이 맞습니까?
* `E = 128`, `E_shared = 1`, `k = 1` — routed expert 128개 + shared 1개, top-1 라우팅?
* `d_moe = 8192`(expert FFN 폭) vs `d_ff = 16384`(dense 층 FFN 폭) — 이 둘이 다른 것이 맞습니까?
* `w_local = 8192` — chunked attention 의 청크 크기입니까? 레이어 스케줄이
  `chunked_attention` 3개마다 `full_attention` 1개인데 공식 설명과 맞습니까?
* `ctx = 262144` — Maverick 의 공개 컨텍스트 길이와 맞습니까?

### B. 축 이름이 맞는가 (핵심)

각 행의 `input_shape` / `weight_shape` / `output_shape` 에서 **틀린 축 이름**을 짚어
주십시오. 특히 값이 겹쳐 구분이 어려운 자리들입니다.

```
d_head = E = 128          attention 의 head 폭과 expert 개수가 같은 값입니다
d_moe = w_local = 8192    expert FFN 폭과 chunked attention 청크 크기가 같은 값입니다
E_shared = k = 1          shared expert 수와 top-k 가 같은 값입니다
```

저희는 이 자리들을 소스로 확인했다고 기록해 두었습니다(`rules/axis_evidence.yaml`).
**그 판정이 틀렸으면 짚어 주십시오.** 근거로 적은 것은 이렇습니다:

* `self_attn` 의 128 축 -> `d_head` : `modeling_llama4.py:327` 이 `head_dim` 을 잡고
  `:362` 의 `hidden_shape = (*input_shape, -1, self.head_dim)` 가 마지막 축을 만든다.
  `num_local_experts` 는 이 클래스에서 안 읽힌다.
* `feed_forward` 의 128 축 -> `E` : `:58 self.num_experts = config.num_local_experts`,
  `:143-144` router 의 out_features 가 `num_local_experts`.
* `feed_forward` 의 8192 축 -> `d_moe` : `:59,61 self.expert_dim = self.intermediate_size`.
  반대 후보 `attention_chunk_size` 는 `modeling_llama4.py` 에 한 번도 나오지 않고
  (`configuration_llama4.py` 에만 2회) 청크 어텐션은 `transformers/masking_utils.py` 가
  마스크로 구현한다 -- 즉 마스크 파라미터이지 텐서 폭이 아니다.

### C. 빠진 것이 있는가

이 표는 ATen 디스패치에 올라온 op 만 담습니다. Llama-4 의 구조 중 **표에 없는 것**이
있습니까? 예를 들어 chunked attention 의 마스킹, QK-norm, temperature tuning,
shared expert 의 결합 방식 등이 제대로 나타나 있습니까?

`repeat`/`layers` 로 접힌 것이 옳습니까 -- 서로 다른 동작을 하는 레이어가 한 행으로
접혀 있지는 않습니까? (레이어 스케줄이 균일하지 않습니다.)

### D. 이름을 일부러 안 붙인 자리

축 672개는 **"이름이 없는 것이 정답"** 으로 판정했습니다.

* `[B, T, n_h, d_head/2, 2]` 의 마지막 축 `2` — RoPE 복소수 쌍(`view_as_complex`)
* `concat` 입력의 `[0]` — prefill 에 과거 KV 가 없어 붙는 빈 텐서

이 둘에 이름을 붙여야 합니까, 아니면 상수로 두는 것이 맞습니까?

### E. 표현 형식

`d_ff`, `n_h*d_head` 같은 식 표기, `repeat`/`layers` 접기, 열 구성이 **이 모델의 구조를
읽는 사람에게 정확히 전달되는가**를 봐 주십시오. 오해를 부를 표기가 있으면 짚어 주십시오.

---

각 지적마다 **소스의 어느 줄** 또는 **공식 자료의 어디**인지 함께 주시면 저희가 그대로
검증해서 반영하겠습니다. 값이 맞아떨어진다는 것만으로는 근거로 채택하지 않습니다.
