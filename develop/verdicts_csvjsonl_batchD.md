# 배치 D 판정 — Codex CSV/JSONL 라벨 주장 검증

대상: Zyphra__Zamba2-1.2B, deepseek-ai__DeepSeek-V2-Lite,
hf-internal-testing__tiny-random-LlamaForCausalLM, openai-community__gpt2-xl
근거 소스: 설치된 transformers 5.14.1. 판정일 2026-09-03.

| # | 모델 | 주장 | 판정 | 근거 |
|---|---|---|---|---|
| D-1 | Zamba2-1.2B | mamba head 축 `n_h_ssm`/`d_head_ssm` 혼용 | **CONFIRMED** | permute/transpose 불변식 위반 **36행**. 입력의 `n_h_ssm` 이 출력에서 `d_head_ssm` 으로 바뀐다 -- 재배열 연산은 축 이름을 바꿀 수 없다. 소스 순서가 `[n_h_ssm, d_head_ssm]`(view `[B,T,d_inner]->[B,T,n_h_ssm,d_head_ssm]`)이므로 출력 쪽이 틀렸다. 둘 다 64 라 값으로는 불검출 |
| D-2 | DeepSeek-V2-Lite | MLA value 경로 `d_nope`→`d_v` | **CONFIRMED** | op 164 split 은 `[.., d_nope], [.., d_v]` 로 정확한데 value 가지가 이후 `d_nope` 로 되돌아간다. 결정적 증거: 전치 한 행에서 입력 `[B,n_h,T,d_nope]` → 출력 `[B,T,n_h,d_v]` (**54행**). d_nope = d_v = 128 |
| D-3 | DeepSeek-V2-Lite | q/k RoPE 조각 `d_head`→`d_rope` | **CONFIRMED** | 자체 모순: `concat [[B,n_h,T,d_nope],[B,n_h,T,d_head]] -> [[B,n_h,T,d_nope+d_rope]]` -- 출력이 이미 `d_rope` 라고 말하는데 피연산자만 `d_head` 다. self_attn 안의 `d_head` 108건이 전부 이 RoPE 텐서. d_head = d_rope = 64 |
| D-4 | tiny-random-Llama | k/v_proj `n_h*d_head`→`n_kv*d_head` | **CONFIRMED** | `modeling_llama.py:241-245`. n_h = n_kv = 4, d_head = 4 라 둘 다 16 -- 값으로는 영원히 불검출. A-8/B-9 와 같은 계열 |
| D-5 | GPT-2 XL | 이상 없음 | — | 주장 없음 |

**4/4 CONFIRMED, REJECTED 0.**

## D-2 는 C-8 과 같은 주장인데 판정이 반대다

Codex 는 tiny-deepseek-v3(C-8)와 DeepSeek-V2-Lite(D-2) 양쪽에 "MLA value 경로가
`d_nope` 로 잘못 붙었다"고 똑같이 주장했다. 실제로는:

- tiny-deepseek-v3: **이미 정확했다** → REJECT
- DeepSeek-V2-Lite: **진짜 틀렸다** → CONFIRM

같은 아키텍처 계열이라도 모델별로 확인해야 한다는 실증이다. 계열 단위로 일괄 적용했다면
tiny-deepseek-v3 의 정확한 라벨을 망가뜨렸을 것이다.

## 고치지 않고 남긴 것 — `repeat_kv(n_rep=1)` 경계

Zamba2(30행)와 tiny-random-Llama(4행)에 `transpose [B,n_kv,T,d_head] -> [B,n_h,d_head,T]`
형태의 전치 불변식 위반이 남는다. 두 모델 모두 **n_h = n_kv** 라 `repeat_kv` 가 no-op 이고,
그 경계가 트레이스에 잡히지 않아 한 전치 행이 (추적되지 않은) 경계를 가로지른다.
메모리 `transpose-edge-rejected` 가 기록한 기지의 한계이며, 억지로 한쪽 이름으로 밀면
반대쪽이 틀린다. **의도적으로 남긴다.**

---

## 배치 D 실적

| 모델 | override | 발화 | 미발화 | 남은 위반 | C-FAIL |
|---|---|---|---|---|---|
| Zyphra__Zamba2-1.2B | 129 | 15,826 | 0 | 30 (repeat_kv 기지 건) | 0 |
| deepseek-ai__DeepSeek-V2-Lite | 11 | 1,917 | 0 | 0 | 0 |
| hf-internal-testing__tiny-random-Llama | 11 | 192 | 0 | 4 (repeat_kv 기지 건) | 0 |
