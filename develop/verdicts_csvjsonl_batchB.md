# 배치 B 판정 — Codex CSV/JSONL 라벨 주장 검증

대상: Qwen__Qwen3.5-397B-A17B, Qwen__Qwen3.5-4B, allenai__OLMoE-1B-7B-0924
근거 소스: 설치된 transformers 5.14.1. 판정일 2026-09-03.

## 소스 대조 (전건 확인됨)

- `modeling_olmoe.py:310` `gate_up_proj = nn.Parameter(torch.empty(num_experts, 2*intermediate_dim, hidden_dim))`
- `modeling_olmoe.py:237-241` k_proj/v_proj = `num_key_value_heads * head_dim`
- `modeling_qwen3_5.py:664-669` k_proj/v_proj = `num_key_value_heads * head_dim`, q_proj = `num_attention_heads * head_dim * 2`
- `modeling_qwen3_5.py:287,306` `torch.ones(chunk_size, chunk_size)` / `torch.eye(chunk_size)`

## 판정

| # | 모델 | 주장 | 판정 | 비고 |
|---|---|---|---|---|
| B-1 | Qwen3.5-397B | fused-Q split `n_kv*d_head`→`2*d_head` | **CONFIRMED** | A-4 와 동일 패턴. CSV `[B,T,n_h,n_kv*d_head]` 확인 |
| B-2 | Qwen3.5-397B | `eye` `[n_h_lin_v,d_chunk]`→`[d_chunk,d_chunk]` | **CONFIRMED** | 같은 모델의 `ones` 는 이미 `[d_chunk,d_chunk]` 로 정확 — `eye` 만 갈림 |
| B-3 | Qwen3.5-397B | transpose value 축 (Codex: decode) | **CONFIRMED, 단 위치 정정** | **decode 아니라 prefill 45행.** decode 위반 0건 |
| B-4 | Qwen3.5-4B | conv 채널축 `2*n_h*d_head`→`2*d_k_lin+d_v_lin` | **CONFIRMED** | `in_proj_qkv` 는 이미 `2*d_k_lin+d_v_lin`. 그 출력을 받는 conv 경로만 갈림(568건). 전치 한 행에서 입력 `2*d_k_lin+d_v_lin` → 출력 `2*n_h*d_head` 로 자체 모순 |
| B-5 | Qwen3.5-4B | chunk mask `d_rope`→`d_chunk` | **CONFIRMED** | A-6 과 동일 |
| B-6 | Qwen3.5-4B | k/v_proj `n_h*d_rope`→`n_kv*d_head` | **CONFIRMED** | `n_h*d_rope`(16·64=1024) 는 지어낸 이름. 실제 `n_kv*d_head`=4·256=1024 |
| B-7 | Qwen3.5-4B | transpose value 축 (Codex: decode) | **CONFIRMED, 범위 확대** | prefill 24 + decode 24 = **양쪽 phase** |
| B-8 | OLMoE | experts gate-up `d_model`→`2*d_moe` | **CONFIRMED** | A-2 와 동일. output 은 이미 `[k*T,2*d_moe]` 로 자체 모순 |
| B-9 | OLMoE | k/v_proj `n_h*d_head`→`n_kv*d_head` | **CONFIRMED** | A-8 과 동일. MHA(n_h=n_kv=16)라 숫자로는 불검출 |

**9/9 CONFIRMED, REJECTED 0.** 단 B-3 은 phase 가 틀렸고(decode→prefill), B-7 은 범위가
좁았다(decode only → 양쪽). 둘 다 Codex 의 표본 추출 한계이지 오탐은 아니다.

**OLMoE 는 4중 값 충돌**: `n_h*d_head` = `n_kv*d_head` = `2*d_moe` = `d_model` = 2048.

---

# 부수 발견 — 전치 불변식 전 모델 전수 검사

`aten.transpose` 는 축을 재배열만 하므로 입력과 출력의 축 이름 **집합**이 같아야 한다.
이 하드 불변식을 47개 모델 전체에 돌린 결과:

**13개 모델 / 1,072행 위반.** Codex 표본이 놓친 것이 대부분이다.

| 모델 | 행 | 대표 위반 |
|---|---|---|
| moonshotai__Kimi-K3 | 255 | `d_nope`→`d_v`, `n_h*d_v`→`n_h_kda*d_head_kda` |
| deepseek-ai__DeepSeek-V4-Pro | 122 | `T/m_hca`→`g_o` |
| Qwen__Qwen3.6-35B-A3B | 120 | `d_head_lin_k`→`d_head_lin_v`, `2*d_k_lin+d_v_lin`→`2*n_h*d_head` |
| Qwen__Qwen3.5-4B | 96 | 위와 동일 |
| Qwen__Qwen3.6-27B | 96 | `d_head_lin_k`→`d_head_lin_v` |
| deepseek-ai__DeepSeek-V4-Flash | 86 | `T/m_hca`→`g_o` |
| deepseek-ai__DeepSeek-V4-Flash-0731 | 86 | 위와 동일 |
| deepseek-ai__DeepSeek-V2-Lite | 54 | `d_nope`→`d_v` |
| Qwen__Qwen3.5-397B-A17B | 45 | `d_head_lin_k`→`d_head_lin_v` |
| Qwen__Qwen3-Next-80B-A3B-Instruct | 36 | `d_head_lin_k`→`d_head_lin_v` (**prefill — 내 A-7 이 decode 만 고쳤다**) |
| Zyphra__Zamba2-1.2B | 36 | `n_h`↔`n_kv` |
| ibm-granite__granite-4.0-h-small | 36 | `n_h_ssm`→`d_state` |
| hf-internal-testing__tiny-random-Llama | 4 | `n_kv`→`n_h` |

**주의 — 전부 결함은 아니다.** `n_h`↔`n_kv` 위반(Zamba2, tiny-random-Llama)은 메모리
`transpose-edge-rejected` 가 적어둔 기지의 경우일 수 있다: `repeat_kv(n_rep=1)` 경계가
트레이스에 안 잡혀 두 이름이 같은 텐서를 가리킨다. 이 두 모델은 별도 확인이 필요하다.
나머지(`d_head_lin_k`/`d_nope`/`T/m_hca`/`n_h_ssm`/`2*d_k_lin+d_v_lin`)는 이름 충돌이 명백하다.

**이 검사는 게이트로 승격할 후보다** — 외부 소스 없이 산출물 내부만으로 판정 가능하고,
값 충돌이라 어떤 수치 검사도 볼 수 없는 자리를 정확히 짚는다.

## 내 배치 A 작업의 구멍

A-7 override 는 `shape: [B, n_h_lin_v, 1, d_head_lin_k]` 로 **decode 만** 고정했다.
prefill 은 축이 `T` 라 매치되지 않아 36행이 그대로 남았다. 배치 B override 작성 시
prefill 짝을 반드시 추가할 것.
