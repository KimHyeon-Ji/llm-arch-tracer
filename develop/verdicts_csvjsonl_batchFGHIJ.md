# 배치 F·G·H·I·J 판정 — Codex CSV/JSONL 라벨 주장 검증

근거 소스: 설치된 transformers 5.14.1. 판정일 2026-09-03.

# 배치 F·G — DeepSeek-V4-Flash / V4-Flash-0731

| # | 주장 | 판정 | 근거 |
|---|---|---|---|
| F-1 | experts gate-up weight `d_model`→`2*d_moe` | **CONFIRMED** | `2*d_moe` = 2·2048 = 4096 = `d_model` 충돌. A-2/B-8 과 같은 계열 |
| F-2 | main RoPE 복소수 view 결과 `n_h`→`d_rope` | **CONFIRMED** | `view [[B,T,d_rope/2,2]] -> [[B,T,n_h]]` -- 절반×2 를 합치면 `d_rope`. **같은 모델의 compressor 는 이 자리를 이미 `d_rope` 로 정확히 렌더링**한다(`[B,T/m_csa,d_rope/2,2] -> [B,T/m_csa,d_rope]`). 387행 |
| F-3 | indexer RoPE `n_h_I`/`n_h`→`d_rope` | **CONFIRMED, 단 Codex 절반 오류** | 아래 참조 |
| F-4 | `o_a_proj` group 축 `T/m_hca`→`g_o` | **CONFIRMED** | 전치 불변식 위반 86행: `[T, T/m_hca, d_model] -> [g_o, T, d_model]`. 둘 다 8 |

## F-3 — Codex 는 두 조각을 모두 `d_rope` 라 했으나 절반이 틀렸다

`modeling_deepseek_v4.py:358-360`:

    nope, rope = x[..., :-rope_dim], x[..., -rope_dim:]
    rotated = ((rope.float() * cos) + (rotate_half(rope).float() * sin)).to(x.dtype)
    return torch.cat([nope, rotated], dim=-1)

즉 **앞 조각은 `c_I-d_rope`, 뒤 조각만 `d_rope`** 다. c_I=128, d_rope=64 이므로 둘 다 64 —
값으로는 구분되지 않는다. 트레이스에서도 두 조각이 확인된다:

    op 1731 slice [B,n_h_I,T,c_I] -> [B,n_h_I,T,n_h]     <- nope, c_I-d_rope 여야 함
    op 1732 slice [B,n_h_I,T,c_I] -> [B,n_h_I,T,d_rope]  <- rope, **이미 정확**
    op 1744 concat [[..,n_h],[..,n_h]] -> [[..,c_I]]     <- 피연산자 0=c_I-d_rope, 1=d_rope

Codex 제안을 그대로 적용했다면 nope 조각까지 `d_rope` 가 되어 **정확한 절반을 망가뜨렸을 것**이다.
shape_index 로 피연산자를 갈라 각각 다른 이름을 넣었다.

# 배치 H — Kimi-K2-Instruct / K2.6 / K2.7-Code

| # | 주장 | 판정 |
|---|---|---|
| H-1/3/5 | value 경로 `d_nope`→`d_v` | **REJECTED (3/3)** |
| H-2/4/6 | q/k RoPE `d_head`/`n_h`→`d_rope` | **CONFIRMED (3/3)** |

## value 경로는 세 모델 모두 이미 정확하다

세 모델 다 o_proj 직전이 `[B,T,n_h,d_v] -> [B,T,n_h*d_v]` 이고, 전 모델 전치 불변식
전수 검사에서도 위반 0건이었다.

## RoPE 는 실재한다 — 자체 모순이 결정적

    concat [[B,n_h,T,d_nope],[B,n_h,T,d_head]] -> [[B,n_h,T,d_nope+d_rope]]
    split_with_sizes [[B,n_h,1,d_nope+d_rope]] -> [[B,n_h,1,d_nope],[B,n_h,1,d_head]]

출력/입력이 **이미 `d_rope` 라고 말하는데** 조각만 `d_head` 다. d_head = d_rope = 64 충돌.
`expand [[B,1,T,n_h]] -> [[B,n_h,T,d_head]]` 의 입력 `n_h` 도 같은 축이라 함께 고쳤다.

# Codex 의 MLA value 주장은 5개 모델 중 1개만 맞았다

| 모델 | 판정 |
|---|---|
| bzantium__tiny-deepseek-v3 | REJECT |
| deepseek-ai__DeepSeek-V3 | REJECT |
| moonshotai__Kimi-K2-Instruct | REJECT |
| moonshotai__Kimi-K2.6 | REJECT |
| moonshotai__Kimi-K2.7-Code | REJECT |
| **deepseek-ai__DeepSeek-V2-Lite** | **CONFIRM** |

계열 단위 일괄 적용은 4개 모델의 정확한 라벨을 망가뜨렸을 것이다.

# 배치 I

| # | 주장 | 판정 | 근거 |
|---|---|---|---|
| I-1 | Nemotron-3-Super mixer SSM scan 축 혼용 | **CONFIRMED** | permute 불변식 위반 **160행**. `d_chunk` 가 `d_state` 로 바뀐다. n_h_ssm=d_state=d_chunk=128 **삼중 충돌** |
| I-2 | Nemotron-3-Ultra chunk/state 혼용 | **CONFIRMED** | 같은 결함 **192행**. 여기선 n_h_ssm=256 이라 구분되고 d_state/d_chunk(128)만 충돌 |
| I-3 | Falcon-H1 k/v_proj `d_chunk`→`n_kv*d_head` | **CONFIRMED** | `modeling_falcon_h1.py:209-213`. Mamba 청크 크기가 어텐션 투영에 붙었다. n_kv*d_head = 2·128 = 256 = d_chunk |
| I-4 | Falcon-H1 mamba.out_proj `d_inner`→`d_model` | **CONFIRMED** | `modeling_falcon_h1.py:436` `nn.Linear(self.intermediate_size, config.hidden_size)` -- 출력 폭은 d_model. d_inner = d_model = 3072 |

# 배치 J — GLM-5.2

값 충돌이 겹친다: `d_head`=`d_rope`=64 / `2*d_head`=`c_I`=128 / `n_h_I`=`d_rope/2`=32 /
`d_v`=`d_nope+d_rope`=256.

| # | 주장 | 판정 | 근거 |
|---|---|---|---|
| J-1 | rotary table `d_head`→`d_rope` | **CONFIRMED** | `concat [[B,T,d_rope/2],[B,T,d_rope/2]] -> [[B,T,d_head]]` 자체 모순 |
| J-2 | self_attn q/k RoPE `d_head`→`d_rope` | **CONFIRMED** | 같은 계열 |
| J-3 | key-cache `d_v`→`d_nope+d_rope` | **주장은 옳으나 적용 불가 — open** | 아래 |
| J-4 | indexer split 첫 조각 →`d_rope` | **CONFIRMED** | 소스가 순서를 명시 |
| J-5 | indexer interleaved RoPE 절반 `n_h_I`→`d_rope/2` | **CONFIRMED** | `slice [[B,T,d_head]] -> [[B,T,n_h_I]]` -- 폭을 자른 결과가 헤드 개수일 수 없다 |
| J-6 | indexer per-head 폭 `2*d_head`→`c_I` | **CONFIRMED** | `view [[B,T,n_h_I*c_I]] -> [[B,T,n_h_I,2*d_head]]` 자체 모순 |
| J-7 | indexer pass-through →`c_I-d_rope` | **CONFIRMED** | 소스가 순서를 명시 |

J-4/J-7 의 순서 근거 (`modeling_glm_moe_dsa.py:230-240`):

    q_rot, q_pass = torch.split(q, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)
    q = torch.cat([q_rot, q_pass], dim=-1)

첫 조각 `d_rope`, 둘째 `c_I-d_rope`. (DeepSeek-V4 는 `cat([nope, rotated])` 로 **순서가 반대**다 --
모델마다 확인해야 한다.)

## J-3 — 시도했다가 되돌렸다

key(`d_nope+d_rope`)와 value(`d_v`)는 둘 다 256 이고 **축 등가류가 둘을 하나로 묶는다.**
concat 출력에 `spread: class` 를 걸었더니 value 텐서까지 번져
`batched_matmul [n_h,T,T] x [n_h,T,d_v]` 의 value 피연산자가 오염됐고, **원래 0 이던 전치
불변식 위반이 156행 생겼다**(2026-09-03 실측). 되돌렸다.

Codex 지적 자체는 옳다 -- 키를 value 폭 이름으로 부르고 있다. 다만 현 메커니즘으로는
value 를 건드리지 않고 key 만 고칠 수 없다. `rules/label_overrides.yaml` 머리말이 예고한
"텐서가 어디서 왔는지로만 구별되는 충돌"의 실제 사례다. **open 으로 남긴다.**

---

# 배치 F~J 실적

| 모델 | override | 발화 | 미발화 | 위반 | C-FAIL |
|---|---|---|---|---|---|
| deepseek-ai__DeepSeek-V4-Flash | 19 | 5,179 | 0 | 0 | 0 |
| deepseek-ai__DeepSeek-V4-Flash-0731 | 17 | 4,486 | 0 | 0 | 0 |
| moonshotai__Kimi-K2-Instruct | 10 | 3,076 | 0 | 0 | 0 |
| moonshotai__Kimi-K2.6 | 10 | 3,076 | 0 | 0 | 0 |
| moonshotai__Kimi-K2.7-Code | 12 | 3,381 | 0 | 0 | 0 |
| nvidia__Nemotron-3-Super-120B | 30 | 9,136 | 0 | 0 | 0 |
| nvidia__Nemotron-3-Ultra-550B | 8 | 3,072 | 0 | 0 | 0 |
| tiiuae__Falcon-H1-7B-Instruct | 26 | 7,964 | 0 | 0 | 0 |
| zai-org__GLM-5.2 | 17 | 1,976 | 0 | 0 | 0 |
