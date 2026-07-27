# 깊이 공유 블록 + LoRA 어댑터 (Shared Block with per-invocation LoRA)

> 새 카테고리 `sharing/`. 기존 6개 축(attention/moe/normalization/position_encoding/residual/
> auxiliary)과 ssm은 "블록 **안**에서 뭘 하는가"를 다루는데, 이건 "같은 블록을 **몇 번 쓰는가**"
> 라는 다른 축이다(`rules/structures/README.md`가 예고한 축 추가 케이스).

## 정의
transformer 블록 하나를 **물리적으로 재사용**해 여러 깊이에서 호출하고, 호출마다 서로 다른
**LoRA 어댑터**를 태워 깊이별 특화를 주는 구조. 파라미터 비용은 블록 1개분 + 어댑터 몇 개분만
든다. Zyphra Zamba/Zamba2가 Mamba2 백본 사이사이에 이 공유 attention 블록을 끼워 넣는다.

Zyphra 표현: *"a LoRA projector to each shared MLP and attention block, which allows the network
to specialize at each invocation of the shared transformer layer across depth"* — 즉
**가중치는 공유, 어댑터는 호출별**이다.

또 하나의 특징: **원본 임베딩을 공유 블록 입력에 concat**한다
(*"concatenating the original model embeddings to the input to this attention block improves
performance"*). 그래서 공유 attention의 입력 폭이 `2·d_model`이 된다.

## 관련 심볼 (rules/symbols.yaml)
`n_mem`(공유 블록 수 = `num_mem_blocks`), `r_lora`(어댑터 rank = `adapter_rank`),
`d_attn`(공유 attention 입력 폭 = `attention_hidden_size` = `2·d_model`),
그리고 SSM 백본 쪽은 `n_h_ssm`/`d_head_ssm`/`n_g_ssm`/`d_state`/`d_conv`.

## 트레이스에서 식별하는 방법 (Zamba2-1.2B 실측, T=16)
- 레이어 구조에 **`shared_transformer`** 모듈이 주기적으로만 등장한다:
  `layer 5 / 11 / 17 / 23 / 29 / 35`에 `linear, mamba_decoder, shared_transformer`,
  나머지 32개 층은 `input_layernorm, mamba`뿐. `hybrid_layer_ids`가 그 주기를 준다.
- `layers_block_type`이 `linear_attention`(=Mamba) 32개 + `hybrid` 6개로 갈린다 → C2 = 2 클러스터.
- 공유 블록 앞에 `linear` 모듈이 붙는데, 이게 concat된 `2·d_model` 입력을 받는 자리다.

**layer_idx 귀속은 깨지지 않았다.** 공유 블록이지만 호출 지점이 각 레이어 아래에 있어
`scope.py`가 정상적으로 층 번호를 붙인다(C1 38==38, C5 38/38층 잔차 확인). 파라미터도
C10에서 406개 전부 커버된다 — `named_parameters()`가 공유 텐서를 한 번만 세기 때문.

**Mamba2 백본의 파생 차원**(Zamba2-1.2B: `n_h_ssm`=64, `d_head_ssm`=64, `n_g_ssm`=1,
`d_state`=128 ⇒ `d_inner`=4096):
| 값 | 유래 |
|---|---|
| 4352 | `d_inner + 2·n_g·d_state` = 4096+256 (causal conv1d 폭) |
| 8512 | `2·d_inner + 2·n_g·d_state + n_h_ssm` = 8192+256+64 (Mamba in_proj 출력) |

## ⚠ 알려진 모호성 (Tier 3 판단 필요)
Zamba2-1.2B에서 **4096이 두 가지 서로 다른 양과 값이 같다**:
- `n_h · d_head` = 32×128 = 4096 (q/k/v projection 폭)
- `d_attn` = `2·d_model` = 2×2048 = 4096 (원본 임베딩 concat된 공유 블록 입력 폭)
- (덤으로 Mamba `d_inner` = `expand·d_model` = 2×2048 = 4096도 같다)

값만으로는 구분이 불가능하다. 심볼 우선순위상 하나의 이름으로 렌더되므로,
`shared_transformer`/`input_layernorm` 자리의 4096은 **투영 폭이 아니라 concat 폭**임을
여기 문서로 보완한다. 이 모델에서는 셋을 자동 판별하지 말 것(P1 — 지어내지 않는다).

## 확인된 모델 (계속 추가)
- **`Zyphra/Zamba2-1.2B`** (블라인드 검증, 2026-07-27): 38층 = Mamba2 32 + hybrid 6,
  `n_mem`=**1**, `r_lora`=128, `d_attn`=4096, `n_h`=32(MHA, GQA 아님), `d_head`=128,
  `d_ff`=8192, `V`=32000, tie=True. Mamba2 커널 미설치로 순수 torch 폴백 → 트레이싱 성공.
  C1~C16 FAIL 0.

## 소스 간 불일치 (트레이스로 판정함)
- HF 모델카드: 1.2B는 공유 블록 **1개**
- Zyphra 블로그(Zamba2-mini): **2개**
→ config `num_mem_blocks` = **1**, 트레이스의 `shared_transformer` 등장 패턴도 단일 블록과
정합. **모델카드가 맞다.** 블로그는 Zamba2 계열 일반 설명으로 보인다.
1차 소스(실행되는 config/코드)가 2차 서술을 이긴다는 원칙(P1)의 실례.

## 참고 소스
- **1차** transformers 5.14.1 `models/zamba2/` — 트레이스로 직접 관측
- [HF 모델카드 Zyphra/Zamba2-1.2B](https://huggingface.co/Zyphra/Zamba2-1.2B)
- [Zyphra 블로그 — Zamba2-mini](https://zyphra.com/post/zamba2-mini),
  [Zamba2-7B](https://www.zyphra.com/post/zamba2-7b)
- Mamba2 백본 자체는 [../ssm/mamba.md](../ssm/mamba.md)
