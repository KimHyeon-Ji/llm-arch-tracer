codex 답변이야. 다음에 뭐해야 해?
읽었습니다. PREAMBLE 지침대로 각 패킷의 `5. 대표 트레이스 표본`만 판정 근거로 삼고, transformers 5.14.1 공식 modeling 소스와 대조했습니다.

결론부터 말하면, 숫자 shape 자체는 대체로 정확하지만 심볼의 의미까지 엄격히 보면 오류가 적지 않습니다.

- 검토 모델: 28개
- 확정적인 심볼 오류가 보이는 모델: 23개
- 대표 표본에서 확정 오류를 찾지 못한 모델: 5개
- 대부분은 연산 결과를 망가뜨리는 오류가 아니라, 같은 숫자를 가진 다른 심볼을 선택한 collision 오류입니다.
- 따라서 FLOPs·메모리 계산값은 맞을 가능성이 높지만, CSV/JSONL을 아키텍처 설명 데이터로 쓰면 의미가 잘못 전달될 수 있습니다.

같은 원인의 반복 행은 prefill/decode를 한 줄로 묶었습니다.

## Batch A

| 모델 | phase | module/op·field | 현재 라벨 (값) | 제안 라벨 | 근거 |
|---|---|---|---|---|---|
| MiniMax-M2 | P/D | `self_attn`, `slice/neg/concat`, RoPE 절반 축 | `n_h+2*n_kv` (64) | `d_head/2` | `modeling_minimax_m2.py:288-292`, `rotate_half()` |
| MiniMax-M2 | P/D | `mlp.experts`, `transpose/grouped_matmul`, gate-up weight 축 | `d_model` (3072) | `2*d_moe` | `modeling_minimax_m2.py:76`, weight가 `[E,2*intermediate,hidden]` |
| Qwen3-Next-80B-A3B | P/D | `linear_attn.in_proj_ba`, weight/output 축 | `d_chunk` (64) | `2*n_h_lin_v` | `modeling_qwen3_next.py:527-531`, `projection_size_ba=num_v_heads*2` |
| Qwen3-Next-80B-A3B | P/D | `self_attn`, fused-Q `view/split` 마지막 축 | `n_kv*d_head` (512) | `2*d_head` | `modeling_qwen3_next.py:268-299` |
| Qwen3-Next-80B-A3B | P/D | `mlp.experts`, expert-count 및 weight 첫 축 | `d_moe` (512) | `E` | `modeling_qwen3_next.py:743-747` |
| Qwen3-Next-80B-A3B | P | `linear_attn`, chunk mask `ones/triu/eye/masked_fill` | `d_rope` (64) | `d_chunk` | `torch_chunk_gated_delta_rule`, `torch.ones(chunk_size,chunk_size)` |
| Qwen3-Next-80B-A3B | D | `linear_attn`, recurrent output `transpose` | 입력 `d_head_lin_k` (128), 출력 `d_head_lin_v` | 입력도 `d_head_lin_v` | transpose는 축 이름을 바꿀 수 없고 결과는 value-head 계보 |
| OLMo-2-7B | P/D | `self_attn.k_proj/v_proj`, weight/output 폭 | `n_h*d_head` (4096) | `n_kv*d_head` | `modeling_olmo2.py:223-227` |

- NX-AI/xLSTM-7b: 이상 없음

## Batch B

| 모델 | phase | module/op·field | 현재 라벨 (값) | 제안 라벨 | 근거 |
|---|---|---|---|---|---|
| Qwen3.5-397B-A17B | P/D | `self_attn`, fused-Q `view/split` | `n_kv*d_head` (512) | `2*d_head` | `modeling_qwen3_5_moe.py:658-665` |
| Qwen3.5-397B-A17B | P | `linear_attn eye` | `[n_h_lin_v,d_chunk]` (64×64) | `[d_chunk,d_chunk]` | `modeling_qwen3_5_moe.py:294` |
| Qwen3.5-397B-A17B | D | `linear_attn transpose` value 축 | `d_head_lin_k` (128) | `d_head_lin_v` | 공식 recurrent delta-rule의 surviving 축은 value head |
| Qwen3.5-4B | P/D | `linear_attn`, conv 전 채널 축 | `2*n_h*d_head` (8192) | `2*d_k_lin+d_v_lin` | projection/conv는 linear-attention QKV 폭 |
| Qwen3.5-4B | P | chunk mask `ones/triu/eye` | `d_rope` (64) | `d_chunk` | `modeling_qwen3_5.py:287-306` |
| Qwen3.5-4B | P/D | `self_attn.k_proj/v_proj` | `n_h*d_rope` (1024) | `n_kv*d_head` | `modeling_qwen3_5.py:664-669` |
| Qwen3.5-4B | D | `linear_attn transpose` value 축 | `d_head_lin_k` (128) | `d_head_lin_v` | transpose 및 value 계보 |
| OLMoE-1B-7B | P/D | `mlp.experts`, fused gate-up weight | `d_model` (2048) | `2*d_moe` | `modeling_olmoe.py:310` |
| OLMoE-1B-7B | P/D | `self_attn.k_proj/v_proj` | `n_h*d_head` (2048) | `n_kv*d_head` | `modeling_olmoe.py:237-241` |

## Batch C

| 모델 | phase | module/op·field | 현재 라벨 (값) | 제안 라벨 | 근거 |
|---|---|---|---|---|---|
| Qwen3.6-27B | P/D | `self_attn` RoPE `slice/neg/concat` 절반 축 | `n_h+2*n_kv` (32) | `d_rope/2` | `rotate_half()` 계보 |
| Qwen3.6-27B | P | linear-attention chunk mask | `d_rope` (64) | `d_chunk` | `torch.ones(chunk_size,chunk_size)` |
| Qwen3.6-27B | D | recurrent result `transpose` | `d_head_lin_k` (128) | `d_head_lin_v` | value output 계보 |
| Qwen3.6-35B-A3B | P/D | linear-attention conv 채널 | `2*n_h*d_head` (8192) | `2*d_k_lin+d_v_lin` | Q/K/V linear projection 폭 |
| Qwen3.6-35B-A3B | P/D | fused-Q per-head 폭 | `n_kv*d_head` (512) | `2*d_head` | Q와 gate를 두 조각으로 분할 |
| Qwen3.6-35B-A3B | P | chunk mask | `d_rope` (64) | `d_chunk` | chunk kernel의 정사각 mask |
| Qwen3.6-35B-A3B | D | recurrent value transpose | `d_head_lin_k` (128) | `d_head_lin_v` | value 계보 |
| tiny-deepseek-v3 | P/D | MLA value 경로 | `d_nope` (128) | `d_v` | `modeling_deepseek_v3.py:421-470` |
| tiny-deepseek-v3 | P/D | q/k RoPE 조각 및 rotary table | `d_head` (64) | `d_rope` | `modeling_deepseek_v3.py:428-436` |
| tiny-deepseek-v3 | D | router 및 expert assignment 축 | `E` (8) | `k` | top-k 결과는 expert 전체 수가 아니라 token당 선택 수 |
| tiny-deepseek-v3 | D | `mlp.experts histc` 입력/출력 | `[E]→[E]` | `[k]→[E]` | histc 입력은 assignment, 출력은 expert별 count |

## Batch D

| 모델 | phase | module/op·field | 현재 라벨 (값) | 제안 라벨 | 근거 |
|---|---|---|---|---|---|
| Zamba2-1.2B | P/D | `mamba`/`mamba_decoder.mamba` head axes | `n_h_ssm`·`d_head_ssm` 혼용 (둘 다 64) | source 순서 `[n_h_ssm,d_head_ssm]` | `modeling_zamba2.py:805-840`; 일부 `permute`가 축 이름을 바꿈 |
| DeepSeek-V2-Lite | P/D | MLA value 경로 | `d_nope` (128) | `d_v` | `modeling_deepseek_v2.py:353-397` |
| DeepSeek-V2-Lite | P/D | q/k RoPE 조각 | `d_head` (64) | `d_rope` | `modeling_deepseek_v2.py:360-367` |
| tiny-random-Llama | P/D | `k_proj/v_proj` 출력 폭 | `n_h*d_head` (16) | `n_kv*d_head` | `modeling_llama.py:241-245` |

- GPT-2 XL: 이상 없음

## Batch E

| 모델 | phase | module/op·field | 현재 라벨 (값) | 제안 라벨 | 근거 |
|---|---|---|---|---|---|
| DeepSeek-V3 | P/D | MLA value 경로 | `d_nope` (128) | `d_v` | `modeling_deepseek_v3.py:421-470` |
| DeepSeek-V3 | P/D | q/k RoPE 및 rotary-table 폭 | `d_head` (64) | `d_rope` | `modeling_deepseek_v3.py:428-436` |
| Granite-4.0-H-Small | P | `mamba`, scan/chunk permute 축 | `d_state`·`n_h_ssm` 혼용 (둘 다 128) | source의 `[d_chunk,n_h_ssm,d_state]` 순서 | `modeling_granitemoehybrid.py:663-698` |

- Llama-3.1-405B: 이상 없음
- Hunyuan-A13B-Instruct: 이상 없음

## Batch F·G

두 DeepSeek-V4-Flash 체크포인트에 같은 종류의 문제가 있습니다.

| 모델 | phase | module/op·field | 현재 라벨 (값) | 제안 라벨 | 근거 |
|---|---|---|---|---|---|
| V4-Flash / V4-Flash-0731 | P/D | `mlp.experts`, fused gate-up weight | `d_model` (4096) | `2*d_moe` | `modeling_deepseek_v4.py`, gate-up `[E,2*intermediate,hidden]` |
| 동일 | P/D | main RoPE 복소수 `view` 결과 | `n_h` (64) | `d_rope` | `[d_rope/2,2]`를 합치는 연산 |
| 동일 | P/D | indexer RoPE 복소수 `view/concat` | `n_h_I` 또는 `n_h` (64) | `d_rope` | 동일한 RoPE feature 계보 |
| 동일 | P/D | `self_attn.o_a_proj transpose` 입력 group 축 | `T/m_hca` (8) | `g_o` | `modeling_deepseek_v4.py:326-332`, `n_groups` 축 |

참고로 packet에서 언급한 indexer의 `m_csa=4` 축은 대표 트레이스에는 이미 `m_csa`로 표시되어 있어 오류로 다시 세지 않았습니다.

## Batch H

Kimi 세 모델은 같은 MLA 충돌 유형입니다.

| 모델 | phase | module/op·field | 현재 라벨 (값) | 제안 라벨 | 근거 |
|---|---|---|---|---|---|
| Kimi-K2-Instruct | P/D | value 경로 | `d_nope` (128) | `d_v` | `modeling_deepseek_v3.py:421-470`과 동일 구조 |
| Kimi-K2-Instruct | P/D | q/k RoPE 조각과 결과 | `d_head` 또는 `n_h` (64) | `d_rope` | q/k의 rotary 조각 |
| Kimi-K2.6 | P/D | value 경로 | `d_nope` (128) | `d_v` | 동일 |
| Kimi-K2.6 | P/D | RoPE 경로 | `d_head` 또는 `n_h` (64) | `d_rope` | 동일 |
| Kimi-K2.7-Code | P/D | value 경로 | `d_nope` (128) | `d_v` | 동일 |
| Kimi-K2.7-Code | P/D | RoPE 경로 | `d_head` 또는 `n_h` (64) | `d_rope` | 동일 |

- NVIDIA Nemotron-3-Nano-4B: 이상 없음

## Batch I

| 모델 | phase | module/op·field | 현재 라벨 (값) | 제안 라벨 | 근거 |
|---|---|---|---|---|---|
| Nemotron-3-Super | P/D | `mixer`, SSM scan axes | `n_h_ssm`·`d_state`·`d_chunk` 혼용 (128) | source의 H/N/L 축 순서 | `modeling_nemotron_h.py:530-545`; 여러 `permute`가 축 이름을 변경하고 있음 |
| Nemotron-3-Ultra | P | `mixer`, chunk/state axes | `d_state`와 `d_chunk` 혼용 (128) | 연산별 `d_chunk`/`d_state` | 동일 source; `permute` 보존 조건으로 확정 |
| Falcon-H1-7B | P/D | `self_attn.k_proj/v_proj` | `d_chunk` (256) | `n_kv*d_head` | `modeling_falcon_h1.py:209-213` |
| Falcon-H1-7B | D | `mamba.out_proj` weight/output 및 이후 residual | `d_inner` (3072) | 출력 측 `d_model` | `modeling_falcon_h1.py:436,876-878`; out-proj는 `d_inner→d_model` |

## Batch J

GLM-5.2는 `64`와 `256` 충돌이 여러 번 겹칩니다.

| phase | module/op·field | 현재 라벨 (값) | 제안 라벨 | 근거 |
|---|---|---|---|---|
| P/D | `model.rotary_emb`, cos/sin·concat 폭 | `d_head` (64) | `d_rope` | rotary table 폭 |
| P/D | `self_attn`, q/k RoPE 결과 | `d_head` 또는 `n_h` (64) | `d_rope` | rotary feature 축 |
| P/D | `self_attn`, key-cache 폭 | `d_v` (256) | `d_nope+d_rope` | K와 V는 둘 다 256이지만 서로 다른 의미 |
| P/D | `self_attn.indexer`, q/k split 첫 조각 | `d_head`·`n_h` (64) | `d_rope` | `modeling_glm_moe_dsa.py:230-240` |
| P/D | `self_attn.indexer`, interleaved RoPE 절반 | `n_h_I` (32) | `d_rope/2` | 같은 소스의 interleaved RoPE |
| P/D | `self_attn.indexer`, 전체 per-head 폭 | `2*d_head` (128) | `c_I` | indexer의 `head_dim`은 `c_I` |
| P/D | `self_attn.indexer`, pass-through 절반 | `d_head`·`n_h` (64) | `c_I-d_rope` | q/k의 비회전 절반 |

## 최종 판단

이 결과는 “트레이스가 부정확하다”기보다는 다음과 같습니다.

- 실제 정수 shape와 연산 흐름: 상당히 정확함
- 일반적인 `d_model`, `n_h`, `n_kv`, `d_ff`: 대체로 잘 됨
- 같은 정수값을 가진 심볼의 역할 구분: 아직 취약함
- 특히 취약한 영역:
  - RoPE feature 폭 ↔ head 수
  - `E` ↔ `d_moe` ↔ `k`
  - `d_nope` ↔ `d_v`
  - `d_chunk` ↔ `d_state` ↔ `n_h_ssm`
  - fused projection 폭 ↔ 우연히 같은 attention 폭

따라서 현재 CSV/JSONL은 수치 분석용으로는 꽤 쓸 만하지만, “연산 의미가 보존된 symbolic architecture IR”이라고 부르기에는 아직 수정이 필요합니다. 특히 위 오류는 모두 숫자 검증만으로는 잡히지 않으므로, module-role·tensor-lineage·transpose/permute 보존 검사를 gate로 추가해야 합니다.