# Llama-4-Maverick 최종 산출물 검토 요청

네 파일이 최종본입니다. **공식 자료와 설치본 소스를 직접 보고** 축 이름과 표기가 정확한지
판정해 주십시오. 저희 설명을 믿지 마시고 반증해 주시면 좋겠습니다.

```
models/meta-llama__Llama-4-Maverick-17B-128E/prefill.csv     (67행)
models/meta-llama__Llama-4-Maverick-17B-128E/decode.csv      (67행)
models/meta-llama__Llama-4-Maverick-17B-128E/prefill.jsonl   (같은 내용, 행당 JSON)
models/meta-llama__Llama-4-Maverick-17B-128E/decode.jsonl
```

## 지난 검토 이후 바뀐 것

`prefill.csv` 무변화, `decode.csv` **12칸**입니다. 지적하신 가짜 `B` 축입니다.

```
op5 input  [[n_h, B, d_head], [n_h, d_head, T+1]] -> [[n_h, 1, d_head], [n_h, d_head, T+1]]
op5 output [[n_h, B, T+1]]                        -> [[n_h, 1, T+1]]
op7 input  [[n_h, B, T+1], [n_h, T+1, d_head]]    -> [[n_h, 1, T+1], [n_h, T+1, d_head]]
op7 output [[n_h, B, d_head]]                     -> [[n_h, 1, d_head]]
(op21/23, op45/47 도 같은 형태 -- 세 레이어 그룹)
```

전역 규칙이 아니라 **소스로 확인된 자리에만** 건 scoped override 이고, `spread: class` 로
등가류 전체에 폈습니다. 처음에 bmm 만 고쳤더니 그 피연산자를 만든 `view` 가 `B` 로 남아
`verify_all` 이 "한 축에 이름이 둘 이상인 등가류 192건" 으로 잡았습니다(지금 0).

## 검증 상태

```
verify_all            Llama-4 FAIL 0   (함대 164건은 다른 46개가 재생성 안 돼 나는 신선도 FAIL)
B=2 배치 축 프로브     미설명 밀림 0 / 축 0 밖 위반 0 / rank 불일치 0  -> PASS
검토 원장             PASS (2026-09-12, 발견 8건)
출고 게이트           PASS (나머지 13개는 BLOCK)
```

축 판정 사이드카(출고본에 함께 실립니다):

```json
{"kind": "summary", "occurrences": {"confirmed": 36247, "unresolved": 384}, "questions": 0, "sites": 36631, "coverage_ok": true, "evidence_entries": 6, "evidence_unused": []}
{"kind": "summary", "occurrences": {"confirmed": 36151, "unresolved": 288}, "questions": 0, "sites": 36439, "coverage_ok": true, "evidence_entries": 6, "evidence_unused": ["|0"]}
```

## 심볼 (이 모델에서 확정된 값)

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
chunk_size   = 8192
```

레이어 스케줄(48층) 앞 12: ['chunked_attention', 'chunked_attention', 'chunked_attention', 'full_attention', 'chunked_attention', 'chunked_attention', 'chunked_attention', 'full_attention', 'chunked_attention', 'chunked_attention', 'chunked_attention', 'full_attention']

`context` 블록:

```json
{"config_max_position_embeddings": 262144, "public_context_length": 1048576, "public_source": "https://github.com/meta-llama/llama-models/blob/main/models/llama4/MODEL_CARD.md", "note": "둘이 다르면 그 자체가 사실이다. config 값을 공개값에 맞추지 않는다."}
```

`scope.known_limits`:

- 연속된 elementwise 연산이 한 행으로 융합된다. 예를 들어 MoE 는 `shared_out += routed_sum` 과 `residual + combined` 가 **실제로는 두 번의 add** 인데 표에는 한 행으로 나온다.
- `block_type` 은 attention 종류와 위치 인코딩을 구분하지 않는다. 이 모델의 층은 2종(chunked_attention, full_attention)이고 실제 구성은 `symbols.layer_sched` 를 봐야 한다. 같은 `attn+MoE` 로 보이는 두 블록이 서로 다른 attention 일 수 있다.
- `E_shared = 1` 는 이 모델에서 **텐서 축이 아니다** -- shared expert 모듈 수라는 구조 사실이고, 표의 어떤 축 이름도 아니다 (`num_shared_experts` 같은 config 필드가 있는 것도 아니다).
- 표에 보이지 않는 연산: RoPE. major-op 선별에서 빠진 것이지 실행되지 않은 것이 아니다.

## 검토 기록 상태

```
fixed           decode bmm 축 1                 should_be_renamed
fixed           ctx                            should_be_renamed
fixed           w_local                        should_be_renamed
accepted_limit  E_shared                       undetermined
accepted_limit  block_type                     should_be_renamed
accepted_limit  MoE 결합                         should_be_renamed
accepted_limit  표에 없는 연산                       undetermined
fixed           산출물 범위                         undetermined
```

## prefill.csv

| op_id | block_type | repeat | layers | h1 | h2 | op_type | input_shape | weight_shape | weight_pos | output_shape | depends_on |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | embed | 1 |  | model | embed_tokens | embedding | [[V, d_model], [B, T]] | [V, d_model] | 0 | [[B, T, d_model]] | [] |
| 1 | attn+FFN | 24 | 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46 | input_layernorm |  | rmsnorm | [[B, T, d_model]] | [d_model] | -1 | [[B, T, d_model]] | [0] |
| 2 | attn+FFN | 24 | 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46 | self_attn | q_proj | matmul | [[T, d_model], [d_model, n_h*d_head]] | [n_h*d_head, d_model] | 1 | [[T, n_h*d_head]] | [1] |
| 3 | attn+FFN | 24 | 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46 | self_attn | k_proj | matmul | [[T, d_model], [d_model, n_kv*d_head]] | [n_kv*d_head, d_model] | 1 | [[T, n_kv*d_head]] | [1] |
| 4 | attn+FFN | 24 | 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46 | self_attn | v_proj | matmul | [[T, d_model], [d_model, n_kv*d_head]] | [n_kv*d_head, d_model] | 1 | [[T, n_kv*d_head]] | [1] |
| 5 | attn+FFN | 24 | 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46 | self_attn |  | batched_matmul | [[n_h, T, d_head], [n_h, d_head, T]] |  |  | [[n_h, T, T]] | [2, 3] |
| 6 | attn+FFN | 24 | 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46 | self_attn |  | softmax | [[B, n_h, T, T]] |  |  | [[B, n_h, T, T]] | [5] |
| 7 | attn+FFN | 24 | 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46 | self_attn |  | batched_matmul | [[n_h, T, T], [n_h, T, d_head]] |  |  | [[n_h, T, d_head]] | [4, 6] |
| 8 | attn+FFN | 24 | 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46 | self_attn | o_proj | matmul | [[T, n_h*d_head], [n_h*d_head, d_model]] | [d_model, n_h*d_head] | 1 | [[T, d_model]] | [7] |
| 9 | attn+FFN | 24 | 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46 |  |  | elementwise_add | [[B, T, d_model], [B, T, d_model]] |  |  | [[B, T, d_model]] | [0, 8] |
| 10 | attn+FFN | 24 | 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46 | post_attention_layernorm |  | rmsnorm | [[B, T, d_model]] | [d_model] | -1 | [[B, T, d_model]] | [9] |
| 11 | attn+FFN | 24 | 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46 | feed_forward | gate_proj | matmul | [[T, d_model], [d_model, d_ff]] | [d_ff, d_model] | 1 | [[T, d_ff]] | [10] |
| 12 | attn+FFN | 24 | 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46 | feed_forward | activation_fn | silu | [[B, T, d_ff]] |  |  | [[B, T, d_ff]] | [11] |
| 13 | attn+FFN | 24 | 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46 | feed_forward | up_proj | matmul | [[T, d_model], [d_model, d_ff]] | [d_ff, d_model] | 1 | [[T, d_ff]] | [10] |
| 14 | attn+FFN | 24 | 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46 | feed_forward |  | elementwise_mul | [[B, T, d_ff], [B, T, d_ff]] |  |  | [[B, T, d_ff]] | [12, 13] |
| 15 | attn+FFN | 24 | 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46 | feed_forward | down_proj | matmul | [[T, d_ff], [d_ff, d_model]] | [d_model, d_ff] | 1 | [[T, d_model]] | [14] |
| 16 | attn+FFN | 24 | 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46 |  |  | elementwise_add | [[B, T, d_model], [B, T, d_model]] |  |  | [[B, T, d_model]] | [9, 15] |
| 17 | attn+MoE | 12 | 1,5,9,13,17,21,25,29,33,37,41,45 | input_layernorm |  | rmsnorm | [[B, T, d_model]] | [d_model] | -1 | [[B, T, d_model]] | [16] |
| 18 | attn+MoE | 12 | 1,5,9,13,17,21,25,29,33,37,41,45 | self_attn | q_proj | matmul | [[T, d_model], [d_model, n_h*d_head]] | [n_h*d_head, d_model] | 1 | [[T, n_h*d_head]] | [17] |
| 19 | attn+MoE | 12 | 1,5,9,13,17,21,25,29,33,37,41,45 | self_attn | k_proj | matmul | [[T, d_model], [d_model, n_kv*d_head]] | [n_kv*d_head, d_model] | 1 | [[T, n_kv*d_head]] | [17] |
| 20 | attn+MoE | 12 | 1,5,9,13,17,21,25,29,33,37,41,45 | self_attn | v_proj | matmul | [[T, d_model], [d_model, n_kv*d_head]] | [n_kv*d_head, d_model] | 1 | [[T, n_kv*d_head]] | [17] |
| 21 | attn+MoE | 12 | 1,5,9,13,17,21,25,29,33,37,41,45 | self_attn |  | batched_matmul | [[n_h, T, d_head], [n_h, d_head, T]] |  |  | [[n_h, T, T]] | [18, 19] |
| 22 | attn+MoE | 12 | 1,5,9,13,17,21,25,29,33,37,41,45 | self_attn |  | softmax | [[B, n_h, T, T]] |  |  | [[B, n_h, T, T]] | [21] |
| 23 | attn+MoE | 12 | 1,5,9,13,17,21,25,29,33,37,41,45 | self_attn |  | batched_matmul | [[n_h, T, T], [n_h, T, d_head]] |  |  | [[n_h, T, d_head]] | [20, 22] |
| 24 | attn+MoE | 12 | 1,5,9,13,17,21,25,29,33,37,41,45 | self_attn | o_proj | matmul | [[T, n_h*d_head], [n_h*d_head, d_model]] | [d_model, n_h*d_head] | 1 | [[T, d_model]] | [23] |
| 25 | attn+MoE | 12 | 1,5,9,13,17,21,25,29,33,37,41,45 |  |  | elementwise_add | [[B, T, d_model], [B, T, d_model]] |  |  | [[B, T, d_model]] | [16, 24] |
| 26 | attn+MoE | 12 | 1,5,9,13,17,21,25,29,33,37,41,45 | post_attention_layernorm |  | rmsnorm | [[B, T, d_model]] | [d_model] | -1 | [[B, T, d_model]] | [25] |
| 27 | attn+MoE | 12 | 1,5,9,13,17,21,25,29,33,37,41,45 | feed_forward | router | matmul | [[T, d_model], [d_model, E]] | [E, d_model] | 1 | [[T, E]] | [26] |
| 28 | attn+MoE | 12 | 1,5,9,13,17,21,25,29,33,37,41,45 | feed_forward | router | sigmoid | [[T, E]] |  |  | [[T, E]] | [27] |
| 29 | attn+MoE | 12 | 1,5,9,13,17,21,25,29,33,37,41,45 | feed_forward |  | elementwise_mul | [[E*T, d_model], [E*T, 1]] |  |  | [[E*T, d_model]] | [26, 28] |
| 30 | attn+MoE | 12 | 1,5,9,13,17,21,25,29,33,37,41,45 | feed_forward | experts | batched_matmul | [[E, T, d_model], [E, d_model, 2*d_moe]] | [E, d_model, 2*d_moe] | 1 | [[E, T, 2*d_moe]] | [29] |
| 31 | attn+MoE | 12 | 1,5,9,13,17,21,25,29,33,37,41,45 | feed_forward | experts | silu | [[E, T, d_moe]] |  |  | [[E, T, d_moe]] | [30] |
| 32 | attn+MoE | 12 | 1,5,9,13,17,21,25,29,33,37,41,45 | feed_forward | experts | elementwise_mul | [[E, T, d_moe], [E, T, d_moe]] |  |  | [[E, T, d_moe]] | [30, 31] |
| 33 | attn+MoE | 12 | 1,5,9,13,17,21,25,29,33,37,41,45 | feed_forward | experts | batched_matmul | [[E, T, d_moe], [E, d_moe, d_model]] | [E, d_moe, d_model] | 1 | [[E, T, d_model]] | [32] |
| 34 | attn+MoE | 12 | 1,5,9,13,17,21,25,29,33,37,41,45 | feed_forward | shared_expert | matmul | [[T, d_model], [d_model, d_moe]] | [d_moe, d_model] | 1 | [[T, d_moe]] | [26] |
| 35 | attn+MoE | 12 | 1,5,9,13,17,21,25,29,33,37,41,45 | feed_forward | shared_expert | silu | [[T, d_moe]] |  |  | [[T, d_moe]] | [34] |
| 36 | attn+MoE | 12 | 1,5,9,13,17,21,25,29,33,37,41,45 | feed_forward | shared_expert | matmul | [[T, d_model], [d_model, d_moe]] | [d_moe, d_model] | 1 | [[T, d_moe]] | [26] |
| 37 | attn+MoE | 12 | 1,5,9,13,17,21,25,29,33,37,41,45 | feed_forward | shared_expert | elementwise_mul | [[T, d_moe], [T, d_moe]] |  |  | [[T, d_moe]] | [35, 36] |
| 38 | attn+MoE | 12 | 1,5,9,13,17,21,25,29,33,37,41,45 | feed_forward | shared_expert | matmul | [[T, d_moe], [d_moe, d_model]] | [d_model, d_moe] | 1 | [[T, d_model]] | [37] |
| 39 | attn+MoE | 12 | 1,5,9,13,17,21,25,29,33,37,41,45 | feed_forward |  | sum | [[E, T, d_model]] |  |  | [[T, d_model]] | [33] |
| 40 | attn+MoE | 12 | 1,5,9,13,17,21,25,29,33,37,41,45 |  |  | elementwise_add | [[B, T, d_model], [B, T, d_model]] |  |  | [[B, T, d_model]] | [25, 38, 39] |
| 41 | attn+MoE | 12 | 3,7,11,15,19,23,27,31,35,39,43,47 | input_layernorm |  | rmsnorm | [[B, T, d_model]] | [d_model] | -1 | [[B, T, d_model]] | [16] |
| 42 | attn+MoE | 12 | 3,7,11,15,19,23,27,31,35,39,43,47 | self_attn | q_proj | matmul | [[T, d_model], [d_model, n_h*d_head]] | [n_h*d_head, d_model] | 1 | [[T, n_h*d_head]] | [41] |
| 43 | attn+MoE | 12 | 3,7,11,15,19,23,27,31,35,39,43,47 | self_attn | k_proj | matmul | [[T, d_model], [d_model, n_kv*d_head]] | [n_kv*d_head, d_model] | 1 | [[T, n_kv*d_head]] | [41] |
| 44 | attn+MoE | 12 | 3,7,11,15,19,23,27,31,35,39,43,47 | self_attn | v_proj | matmul | [[T, d_model], [d_model, n_kv*d_head]] | [n_kv*d_head, d_model] | 1 | [[T, n_kv*d_head]] | [41] |
| 45 | attn+MoE | 12 | 3,7,11,15,19,23,27,31,35,39,43,47 | self_attn |  | batched_matmul | [[n_h, T, d_head], [n_h, d_head, T]] |  |  | [[n_h, T, T]] | [42, 43] |
| 46 | attn+MoE | 12 | 3,7,11,15,19,23,27,31,35,39,43,47 | self_attn |  | softmax | [[B, n_h, T, T]] |  |  | [[B, n_h, T, T]] | [45] |
| 47 | attn+MoE | 12 | 3,7,11,15,19,23,27,31,35,39,43,47 | self_attn |  | batched_matmul | [[n_h, T, T], [n_h, T, d_head]] |  |  | [[n_h, T, d_head]] | [44, 46] |
| 48 | attn+MoE | 12 | 3,7,11,15,19,23,27,31,35,39,43,47 | self_attn | o_proj | matmul | [[T, n_h*d_head], [n_h*d_head, d_model]] | [d_model, n_h*d_head] | 1 | [[T, d_model]] | [47] |
| 49 | attn+MoE | 12 | 3,7,11,15,19,23,27,31,35,39,43,47 |  |  | elementwise_add | [[B, T, d_model], [B, T, d_model]] |  |  | [[B, T, d_model]] | [16, 48] |
| 50 | attn+MoE | 12 | 3,7,11,15,19,23,27,31,35,39,43,47 | post_attention_layernorm |  | rmsnorm | [[B, T, d_model]] | [d_model] | -1 | [[B, T, d_model]] | [49] |
| 51 | attn+MoE | 12 | 3,7,11,15,19,23,27,31,35,39,43,47 | feed_forward | router | matmul | [[T, d_model], [d_model, E]] | [E, d_model] | 1 | [[T, E]] | [50] |
| 52 | attn+MoE | 12 | 3,7,11,15,19,23,27,31,35,39,43,47 | feed_forward | router | sigmoid | [[T, E]] |  |  | [[T, E]] | [51] |
| 53 | attn+MoE | 12 | 3,7,11,15,19,23,27,31,35,39,43,47 | feed_forward |  | elementwise_mul | [[E*T, d_model], [E*T, 1]] |  |  | [[E*T, d_model]] | [50, 52] |
| 54 | attn+MoE | 12 | 3,7,11,15,19,23,27,31,35,39,43,47 | feed_forward | experts | batched_matmul | [[E, T, d_model], [E, d_model, 2*d_moe]] | [E, d_model, 2*d_moe] | 1 | [[E, T, 2*d_moe]] | [53] |
| 55 | attn+MoE | 12 | 3,7,11,15,19,23,27,31,35,39,43,47 | feed_forward | experts | silu | [[E, T, d_moe]] |  |  | [[E, T, d_moe]] | [54] |
| 56 | attn+MoE | 12 | 3,7,11,15,19,23,27,31,35,39,43,47 | feed_forward | experts | elementwise_mul | [[E, T, d_moe], [E, T, d_moe]] |  |  | [[E, T, d_moe]] | [54, 55] |
| 57 | attn+MoE | 12 | 3,7,11,15,19,23,27,31,35,39,43,47 | feed_forward | experts | batched_matmul | [[E, T, d_moe], [E, d_moe, d_model]] | [E, d_moe, d_model] | 1 | [[E, T, d_model]] | [56] |
| 58 | attn+MoE | 12 | 3,7,11,15,19,23,27,31,35,39,43,47 | feed_forward | shared_expert | matmul | [[T, d_model], [d_model, d_moe]] | [d_moe, d_model] | 1 | [[T, d_moe]] | [50] |
| 59 | attn+MoE | 12 | 3,7,11,15,19,23,27,31,35,39,43,47 | feed_forward | shared_expert | silu | [[T, d_moe]] |  |  | [[T, d_moe]] | [58] |
| 60 | attn+MoE | 12 | 3,7,11,15,19,23,27,31,35,39,43,47 | feed_forward | shared_expert | matmul | [[T, d_model], [d_model, d_moe]] | [d_moe, d_model] | 1 | [[T, d_moe]] | [50] |
| 61 | attn+MoE | 12 | 3,7,11,15,19,23,27,31,35,39,43,47 | feed_forward | shared_expert | elementwise_mul | [[T, d_moe], [T, d_moe]] |  |  | [[T, d_moe]] | [59, 60] |
| 62 | attn+MoE | 12 | 3,7,11,15,19,23,27,31,35,39,43,47 | feed_forward | shared_expert | matmul | [[T, d_moe], [d_moe, d_model]] | [d_model, d_moe] | 1 | [[T, d_model]] | [61] |
| 63 | attn+MoE | 12 | 3,7,11,15,19,23,27,31,35,39,43,47 | feed_forward |  | sum | [[E, T, d_model]] |  |  | [[T, d_model]] | [57] |
| 64 | attn+MoE | 12 | 3,7,11,15,19,23,27,31,35,39,43,47 |  |  | elementwise_add | [[B, T, d_model], [B, T, d_model]] |  |  | [[B, T, d_model]] | [49, 62, 63] |
| 65 | norm | 1 |  | model | norm | rmsnorm | [[B, T, d_model]] | [d_model] | -1 | [[B, T, d_model]] | [64] |
| 66 | head | 1 |  | lm_head |  | matmul | [[T, d_model], [d_model, V]] | [V, d_model] | 1 | [[T, V]] | [65] |

## decode.csv

| op_id | block_type | repeat | layers | h1 | h2 | op_type | input_shape | weight_shape | weight_pos | output_shape | depends_on |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | embed | 1 |  | model | embed_tokens | embedding | [[V, d_model], [B, 1]] | [V, d_model] | 0 | [[B, 1, d_model]] | [] |
| 1 | attn+FFN | 24 | 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46 | input_layernorm |  | rmsnorm | [[B, 1, d_model]] | [d_model] | -1 | [[B, 1, d_model]] | [0] |
| 2 | attn+FFN | 24 | 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46 | self_attn | q_proj | matmul | [[B, d_model], [d_model, n_h*d_head]] | [n_h*d_head, d_model] | 1 | [[B, n_h*d_head]] | [1] |
| 3 | attn+FFN | 24 | 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46 | self_attn | k_proj | matmul | [[B, d_model], [d_model, n_kv*d_head]] | [n_kv*d_head, d_model] | 1 | [[B, n_kv*d_head]] | [1] |
| 4 | attn+FFN | 24 | 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46 | self_attn | v_proj | matmul | [[B, d_model], [d_model, n_kv*d_head]] | [n_kv*d_head, d_model] | 1 | [[B, n_kv*d_head]] | [1] |
| 5 | attn+FFN | 24 | 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46 | self_attn |  | batched_matmul | [[n_h, 1, d_head], [n_h, d_head, T+1]] |  |  | [[n_h, 1, T+1]] | [2, 3] |
| 6 | attn+FFN | 24 | 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46 | self_attn |  | softmax | [[B, n_h, 1, T+1]] |  |  | [[B, n_h, 1, T+1]] | [5] |
| 7 | attn+FFN | 24 | 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46 | self_attn |  | batched_matmul | [[n_h, 1, T+1], [n_h, T+1, d_head]] |  |  | [[n_h, 1, d_head]] | [4, 6] |
| 8 | attn+FFN | 24 | 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46 | self_attn | o_proj | matmul | [[B, n_h*d_head], [n_h*d_head, d_model]] | [d_model, n_h*d_head] | 1 | [[B, d_model]] | [7] |
| 9 | attn+FFN | 24 | 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46 |  |  | elementwise_add | [[B, 1, d_model], [B, 1, d_model]] |  |  | [[B, 1, d_model]] | [0, 8] |
| 10 | attn+FFN | 24 | 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46 | post_attention_layernorm |  | rmsnorm | [[B, 1, d_model]] | [d_model] | -1 | [[B, 1, d_model]] | [9] |
| 11 | attn+FFN | 24 | 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46 | feed_forward | gate_proj | matmul | [[B, d_model], [d_model, d_ff]] | [d_ff, d_model] | 1 | [[B, d_ff]] | [10] |
| 12 | attn+FFN | 24 | 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46 | feed_forward | activation_fn | silu | [[B, 1, d_ff]] |  |  | [[B, 1, d_ff]] | [11] |
| 13 | attn+FFN | 24 | 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46 | feed_forward | up_proj | matmul | [[B, d_model], [d_model, d_ff]] | [d_ff, d_model] | 1 | [[B, d_ff]] | [10] |
| 14 | attn+FFN | 24 | 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46 | feed_forward |  | elementwise_mul | [[B, 1, d_ff], [B, 1, d_ff]] |  |  | [[B, 1, d_ff]] | [12, 13] |
| 15 | attn+FFN | 24 | 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46 | feed_forward | down_proj | matmul | [[B, d_ff], [d_ff, d_model]] | [d_model, d_ff] | 1 | [[B, d_model]] | [14] |
| 16 | attn+FFN | 24 | 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46 |  |  | elementwise_add | [[B, 1, d_model], [B, 1, d_model]] |  |  | [[B, 1, d_model]] | [9, 15] |
| 17 | attn+MoE | 12 | 1,5,9,13,17,21,25,29,33,37,41,45 | input_layernorm |  | rmsnorm | [[B, 1, d_model]] | [d_model] | -1 | [[B, 1, d_model]] | [16] |
| 18 | attn+MoE | 12 | 1,5,9,13,17,21,25,29,33,37,41,45 | self_attn | q_proj | matmul | [[B, d_model], [d_model, n_h*d_head]] | [n_h*d_head, d_model] | 1 | [[B, n_h*d_head]] | [17] |
| 19 | attn+MoE | 12 | 1,5,9,13,17,21,25,29,33,37,41,45 | self_attn | k_proj | matmul | [[B, d_model], [d_model, n_kv*d_head]] | [n_kv*d_head, d_model] | 1 | [[B, n_kv*d_head]] | [17] |
| 20 | attn+MoE | 12 | 1,5,9,13,17,21,25,29,33,37,41,45 | self_attn | v_proj | matmul | [[B, d_model], [d_model, n_kv*d_head]] | [n_kv*d_head, d_model] | 1 | [[B, n_kv*d_head]] | [17] |
| 21 | attn+MoE | 12 | 1,5,9,13,17,21,25,29,33,37,41,45 | self_attn |  | batched_matmul | [[n_h, 1, d_head], [n_h, d_head, T+1]] |  |  | [[n_h, 1, T+1]] | [18, 19] |
| 22 | attn+MoE | 12 | 1,5,9,13,17,21,25,29,33,37,41,45 | self_attn |  | softmax | [[B, n_h, 1, T+1]] |  |  | [[B, n_h, 1, T+1]] | [21] |
| 23 | attn+MoE | 12 | 1,5,9,13,17,21,25,29,33,37,41,45 | self_attn |  | batched_matmul | [[n_h, 1, T+1], [n_h, T+1, d_head]] |  |  | [[n_h, 1, d_head]] | [20, 22] |
| 24 | attn+MoE | 12 | 1,5,9,13,17,21,25,29,33,37,41,45 | self_attn | o_proj | matmul | [[B, n_h*d_head], [n_h*d_head, d_model]] | [d_model, n_h*d_head] | 1 | [[B, d_model]] | [23] |
| 25 | attn+MoE | 12 | 1,5,9,13,17,21,25,29,33,37,41,45 |  |  | elementwise_add | [[B, 1, d_model], [B, 1, d_model]] |  |  | [[B, 1, d_model]] | [16, 24] |
| 26 | attn+MoE | 12 | 1,5,9,13,17,21,25,29,33,37,41,45 | post_attention_layernorm |  | rmsnorm | [[B, 1, d_model]] | [d_model] | -1 | [[B, 1, d_model]] | [25] |
| 27 | attn+MoE | 12 | 1,5,9,13,17,21,25,29,33,37,41,45 | feed_forward | router | matmul | [[B, d_model], [d_model, E]] | [E, d_model] | 1 | [[B, E]] | [26] |
| 28 | attn+MoE | 12 | 1,5,9,13,17,21,25,29,33,37,41,45 | feed_forward | router | sigmoid | [[B, E]] |  |  | [[B, E]] | [27] |
| 29 | attn+MoE | 12 | 1,5,9,13,17,21,25,29,33,37,41,45 | feed_forward |  | elementwise_mul | [[E, d_model], [E, 1]] |  |  | [[E, d_model]] | [26, 28] |
| 30 | attn+MoE | 12 | 1,5,9,13,17,21,25,29,33,37,41,45 | feed_forward | experts | batched_matmul | [[E, B, d_model], [E, d_model, 2*d_moe]] | [E, d_model, 2*d_moe] | 1 | [[E, B, 2*d_moe]] | [29] |
| 31 | attn+MoE | 12 | 1,5,9,13,17,21,25,29,33,37,41,45 | feed_forward | experts | silu | [[E, B, d_moe]] |  |  | [[E, B, d_moe]] | [30] |
| 32 | attn+MoE | 12 | 1,5,9,13,17,21,25,29,33,37,41,45 | feed_forward | experts | elementwise_mul | [[E, B, d_moe], [E, B, d_moe]] |  |  | [[E, B, d_moe]] | [30, 31] |
| 33 | attn+MoE | 12 | 1,5,9,13,17,21,25,29,33,37,41,45 | feed_forward | experts | batched_matmul | [[E, B, d_moe], [E, d_moe, d_model]] | [E, d_moe, d_model] | 1 | [[E, B, d_model]] | [32] |
| 34 | attn+MoE | 12 | 1,5,9,13,17,21,25,29,33,37,41,45 | feed_forward | shared_expert | matmul | [[B, d_model], [d_model, d_moe]] | [d_moe, d_model] | 1 | [[B, d_moe]] | [26] |
| 35 | attn+MoE | 12 | 1,5,9,13,17,21,25,29,33,37,41,45 | feed_forward | shared_expert | silu | [[B, d_moe]] |  |  | [[B, d_moe]] | [34] |
| 36 | attn+MoE | 12 | 1,5,9,13,17,21,25,29,33,37,41,45 | feed_forward | shared_expert | matmul | [[B, d_model], [d_model, d_moe]] | [d_moe, d_model] | 1 | [[B, d_moe]] | [26] |
| 37 | attn+MoE | 12 | 1,5,9,13,17,21,25,29,33,37,41,45 | feed_forward | shared_expert | elementwise_mul | [[B, d_moe], [B, d_moe]] |  |  | [[B, d_moe]] | [35, 36] |
| 38 | attn+MoE | 12 | 1,5,9,13,17,21,25,29,33,37,41,45 | feed_forward | shared_expert | matmul | [[B, d_moe], [d_moe, d_model]] | [d_model, d_moe] | 1 | [[B, d_model]] | [37] |
| 39 | attn+MoE | 12 | 1,5,9,13,17,21,25,29,33,37,41,45 | feed_forward |  | sum | [[E, B, d_model]] |  |  | [[B, d_model]] | [33] |
| 40 | attn+MoE | 12 | 1,5,9,13,17,21,25,29,33,37,41,45 |  |  | elementwise_add | [[B, 1, d_model], [B, 1, d_model]] |  |  | [[B, 1, d_model]] | [25, 38, 39] |
| 41 | attn+MoE | 12 | 3,7,11,15,19,23,27,31,35,39,43,47 | input_layernorm |  | rmsnorm | [[B, 1, d_model]] | [d_model] | -1 | [[B, 1, d_model]] | [16] |
| 42 | attn+MoE | 12 | 3,7,11,15,19,23,27,31,35,39,43,47 | self_attn | q_proj | matmul | [[B, d_model], [d_model, n_h*d_head]] | [n_h*d_head, d_model] | 1 | [[B, n_h*d_head]] | [41] |
| 43 | attn+MoE | 12 | 3,7,11,15,19,23,27,31,35,39,43,47 | self_attn | k_proj | matmul | [[B, d_model], [d_model, n_kv*d_head]] | [n_kv*d_head, d_model] | 1 | [[B, n_kv*d_head]] | [41] |
| 44 | attn+MoE | 12 | 3,7,11,15,19,23,27,31,35,39,43,47 | self_attn | v_proj | matmul | [[B, d_model], [d_model, n_kv*d_head]] | [n_kv*d_head, d_model] | 1 | [[B, n_kv*d_head]] | [41] |
| 45 | attn+MoE | 12 | 3,7,11,15,19,23,27,31,35,39,43,47 | self_attn |  | batched_matmul | [[n_h, 1, d_head], [n_h, d_head, T+1]] |  |  | [[n_h, 1, T+1]] | [42, 43] |
| 46 | attn+MoE | 12 | 3,7,11,15,19,23,27,31,35,39,43,47 | self_attn |  | softmax | [[B, n_h, 1, T+1]] |  |  | [[B, n_h, 1, T+1]] | [45] |
| 47 | attn+MoE | 12 | 3,7,11,15,19,23,27,31,35,39,43,47 | self_attn |  | batched_matmul | [[n_h, 1, T+1], [n_h, T+1, d_head]] |  |  | [[n_h, 1, d_head]] | [44, 46] |
| 48 | attn+MoE | 12 | 3,7,11,15,19,23,27,31,35,39,43,47 | self_attn | o_proj | matmul | [[B, n_h*d_head], [n_h*d_head, d_model]] | [d_model, n_h*d_head] | 1 | [[B, d_model]] | [47] |
| 49 | attn+MoE | 12 | 3,7,11,15,19,23,27,31,35,39,43,47 |  |  | elementwise_add | [[B, 1, d_model], [B, 1, d_model]] |  |  | [[B, 1, d_model]] | [16, 48] |
| 50 | attn+MoE | 12 | 3,7,11,15,19,23,27,31,35,39,43,47 | post_attention_layernorm |  | rmsnorm | [[B, 1, d_model]] | [d_model] | -1 | [[B, 1, d_model]] | [49] |
| 51 | attn+MoE | 12 | 3,7,11,15,19,23,27,31,35,39,43,47 | feed_forward | router | matmul | [[B, d_model], [d_model, E]] | [E, d_model] | 1 | [[B, E]] | [50] |
| 52 | attn+MoE | 12 | 3,7,11,15,19,23,27,31,35,39,43,47 | feed_forward | router | sigmoid | [[B, E]] |  |  | [[B, E]] | [51] |
| 53 | attn+MoE | 12 | 3,7,11,15,19,23,27,31,35,39,43,47 | feed_forward |  | elementwise_mul | [[E, d_model], [E, 1]] |  |  | [[E, d_model]] | [50, 52] |
| 54 | attn+MoE | 12 | 3,7,11,15,19,23,27,31,35,39,43,47 | feed_forward | experts | batched_matmul | [[E, B, d_model], [E, d_model, 2*d_moe]] | [E, d_model, 2*d_moe] | 1 | [[E, B, 2*d_moe]] | [53] |
| 55 | attn+MoE | 12 | 3,7,11,15,19,23,27,31,35,39,43,47 | feed_forward | experts | silu | [[E, B, d_moe]] |  |  | [[E, B, d_moe]] | [54] |
| 56 | attn+MoE | 12 | 3,7,11,15,19,23,27,31,35,39,43,47 | feed_forward | experts | elementwise_mul | [[E, B, d_moe], [E, B, d_moe]] |  |  | [[E, B, d_moe]] | [54, 55] |
| 57 | attn+MoE | 12 | 3,7,11,15,19,23,27,31,35,39,43,47 | feed_forward | experts | batched_matmul | [[E, B, d_moe], [E, d_moe, d_model]] | [E, d_moe, d_model] | 1 | [[E, B, d_model]] | [56] |
| 58 | attn+MoE | 12 | 3,7,11,15,19,23,27,31,35,39,43,47 | feed_forward | shared_expert | matmul | [[B, d_model], [d_model, d_moe]] | [d_moe, d_model] | 1 | [[B, d_moe]] | [50] |
| 59 | attn+MoE | 12 | 3,7,11,15,19,23,27,31,35,39,43,47 | feed_forward | shared_expert | silu | [[B, d_moe]] |  |  | [[B, d_moe]] | [58] |
| 60 | attn+MoE | 12 | 3,7,11,15,19,23,27,31,35,39,43,47 | feed_forward | shared_expert | matmul | [[B, d_model], [d_model, d_moe]] | [d_moe, d_model] | 1 | [[B, d_moe]] | [50] |
| 61 | attn+MoE | 12 | 3,7,11,15,19,23,27,31,35,39,43,47 | feed_forward | shared_expert | elementwise_mul | [[B, d_moe], [B, d_moe]] |  |  | [[B, d_moe]] | [59, 60] |
| 62 | attn+MoE | 12 | 3,7,11,15,19,23,27,31,35,39,43,47 | feed_forward | shared_expert | matmul | [[B, d_moe], [d_moe, d_model]] | [d_model, d_moe] | 1 | [[B, d_model]] | [61] |
| 63 | attn+MoE | 12 | 3,7,11,15,19,23,27,31,35,39,43,47 | feed_forward |  | sum | [[E, B, d_model]] |  |  | [[B, d_model]] | [57] |
| 64 | attn+MoE | 12 | 3,7,11,15,19,23,27,31,35,39,43,47 |  |  | elementwise_add | [[B, 1, d_model], [B, 1, d_model]] |  |  | [[B, 1, d_model]] | [49, 62, 63] |
| 65 | norm | 1 |  | model | norm | rmsnorm | [[B, 1, d_model]] | [d_model] | -1 | [[B, 1, d_model]] | [64] |
| 66 | head | 1 |  | lm_head |  | matmul | [[B, d_model], [d_model, V]] | [V, d_model] | 1 | [[B, V]] | [65] |

## prefill.jsonl 첫 줄 (형식 확인용)

```json
{"op_id": 0, "block_type": "embed", "repeat": 1, "layers": "", "h1": "model", "h2": "embed_tokens", "h3": "", "op_type": "embedding", "input_shape": [["V", "d_model"], ["B", "T"]], "weight_shape": ["V", "d_model"], "weight_pos": 0, "output_shape": [["B", "T", "d_model"]], "depends_on": [], "layer_idx": null, "block": "other", "sub_block": null, "depth": 2, "module_path": "model.embed_tokens", "raw_op": "aten.embedding.default", "params": ["model.embed_tokens.weight"], "phase": "prefill", "unmapped": false}
```

---

## 봐 주셨으면 하는 것

지난번처럼 저희 판정을 먼저 적지 않겠습니다. 값이 겹쳐 구분이 어려운 자리는 이렇습니다.

```
d_head = E = 128          attention head 폭과 expert 개수
d_moe = chunk_size = 8192 expert FFN 폭과 chunked attention 청크 크기
E_shared = k = 1          shared expert 수와 top-k
```

**Q1.** 표의 축 이름 중 **틀린 것**이 있습니까? 각 지적마다 소스 줄이나 공식 자료 위치를
함께 주시면 그대로 검증해 반영하겠습니다.

**Q2.** decode 의 `[n_h, 1, d_head]` 표기가 맞습니까? `1` 대신 `T_q` 같은 이름이 나은지,
아니면 `[B*n_h, 1, d_head]` 처럼 접힌 배치를 드러내야 하는지 봐 주십시오.

**Q3.** `known_limits` 문구로 처음 보는 사람이 표의 한계를 이해할 수 있겠습니까?

**Q4.** 이 네 파일을 지금 공개해도 됩니까? 남은 것이 있으면 짚어 주십시오.