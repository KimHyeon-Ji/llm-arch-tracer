# 블라인드 검증 — Zyphra/Zamba2-1.2B : Phase 0-A 가설

**작성 시점: modeling 코드를 열기 전, 트레이스 실행 전.**
`02-new-module-handling.md` Phase 0의 A단계(문서 우선)만 마친 상태에서 적는다.
목적은 이 도구의 신규 모듈 온보딩 절차가 **답을 본 뒤 끼워맞추는 게 아니라** 실제로
작동하는지 검증하는 것이다. 아래 예측은 이후 B(코드)·C(트레이스) 결과와 대조되며,
**틀린 예측도 그대로 남긴다**(사후 수정 금지).

## 출처 (A단계에서 읽은 것)
- [HF 모델카드 Zyphra/Zamba2-1.2B](https://huggingface.co/Zyphra/Zamba2-1.2B)
- [Zyphra 블로그 — Zamba2-mini](https://zyphra.com/post/zamba2-mini)
- [Zyphra — Zamba2-Small(2.7B)](https://www.zyphra.com/our-work/zamba2-small), [Zamba2-7B](https://www.zyphra.com/post/zamba2-7b)

## 오염 고지
후보 선정 단계에서 **config 필드명만** 조회했다(값·modeling 코드·트레이스는 미열람).
아래에서 config 필드명에 근거한 항목은 `[config엿봄]`으로 표기한다. 순수 문서 근거는 무표기.

## 문서에서 파악한 구조
1. **Mamba2 백본 + 공유 transformer 블록 interleave.** 공유 블록은 **가중치를 공유**해
   파라미터 비용을 줄인다.
2. **LoRA projector를 공유 attention/MLP 블록마다 적용** → 같은 블록이 깊이별로 조금씩
   특화된다("depth-specialization for minimal parameter increase").
3. **원본 임베딩을 공유 attention 블록 입력에 concat**한다("concatenating the original model
   embeddings to the input to this attention block improves performance").
4. 공유 attention에 **RoPE** 적용.

## 소스 간 모순 (B단계에서 판정할 것)
- 모델카드: 1.2B는 **공유 블록 1개**("a single shared transformer block", 2.7B는 2개)
- Zyphra 블로그: Zamba2-mini는 **2개**("two shared attention layers")
→ `num_mem_blocks` 값으로 판정. **트레이스가 최종 판정자다**(P1).

---

## 예측 (falsifiable)

### H1 — 공유 블록 때문에 layer_idx 귀속이 깨진다 ★핵심 예측
`src/scope.py`는 module_path의 `<stack>.<N>.` 패턴으로 layer_idx를 붙인다. 공유 블록은
**모듈 경로가 하나인데 여러 깊이에서 호출**되므로:
- 공유 attention/MLP의 op들은 layer_idx가 **None**이거나 **한 인덱스에 몰릴 것**이다.
- 결과적으로 레이어별 op 시그니처에서 attention이 빠지고, `## 레이어 구조` 표에
  self_attn이 안 보이거나 특정 층에만 보일 것이다.
- **C2(레이어 이종성) 클러스터 수가 config 스케줄과 어긋나 WARN/FAIL** 가능성이 높다.
- C1(레이어 수)은 Mamba 층 기준으로는 통과할 수도 있다.

이건 지금까지 24개 모델에서 **한 번도 의심해보지 않은 전제**(모듈 경로 ⇒ 레이어 인덱스)를
직접 친다. 이번 검증의 최대 관심사.

### H2 — 공유 attention 입력이 `2·d_model`
문서의 "원본 임베딩 concat"이 사실이라면 공유 attention의 입력 폭은 `hidden_size`의 2배다.
- `d_model`=2048이므로 **4096**이 q/k/v projection의 입력 축으로 나타날 것이다.
- 심볼라이저는 이를 `2*d_model`로 렌더할 것이고, 이번엔 그게 **실제로 맞는 유래**다.
- `[config엿봄]` `attention_hidden_size` 필드가 있으므로 = 4096일 것으로 예측.
  → `rules/symbols.yaml`에 새 심볼(`d_attn` 등) 등록이 필요해질 것.

### H3 — LoRA 파라미터가 호출 깊이마다 따로 존재
- 파라미터 이름에 `lora`(A/B 또는 유사)가 등장하고, **호출 횟수만큼 반복**될 것이다.
- LoRA rank가 새 상수로 등장 → 미해결 유도 상수가 되거나 새 심볼이 필요할 것.
- **C10(파라미터 커버리지)** 이 이 LoRA 파라미터들을 전부 op에 연결하는지가 관건.

### H4 — SSM 심볼 별칭이 안 맞아 3개가 미확인으로 뜬다 ★구체적 예측
`[config엿봄]` Zamba2의 Mamba 필드명은 Nemotron-H와 다르다. 현재 `rules/symbols.yaml` 별칭과 대조:

| 심볼 | 현재 별칭 | Zamba2 필드 | 예측 |
|---|---|---|---|
| `n_h_ssm` | `mamba_num_heads`, `n_mamba_heads` | `n_mamba_heads` | ✅ 잡힘 |
| `d_head_ssm` | `mamba_head_dim` | `mamba_headdim` (언더바 없음) | ❌ **미확인** |
| `n_g_ssm` | `n_groups`, `mamba_n_groups` | `mamba_ngroups` | ❌ **미확인** |
| `d_state` | `ssm_state_size`, `state_size`, `d_state` | `mamba_d_state` | ❌ **미확인** |

→ `ssm` group이 "부분 확인"으로 판정되어 3개가 `미확인, Tier 2 대상`으로 뜨고,
   `d_inner = n_h_ssm·d_head_ssm` 등 Mamba 파생 규칙이 **발동하지 않아** 미해결 유도 상수가
   남을 것이다. **조치는 별칭 3개 추가**로 끝날 것으로 예상.

### H5 — C17이 WARN을 낸다
`zamba2`가 `rules/structures/` 어디에도 없고 H4대로 미해결 상수가 남을 것이므로,
새로 넣은 온보딩 게이트 C17이 **WARN**을 내야 정상이다. C17이 PASS로 나오면 그건
게이트가 제 역할을 못 한다는 뜻이다.

### H6 — 트레이싱 자체는 성공할 것
Mamba2 커널(`mamba-ssm`, `causal-conv1d`)이 미설치라 순수 torch 폴백으로 떨어질 것이다
(Nemotron-H·Qwen3-Next와 동일). `use_mamba_kernels` 필드가 있으나 폴백 경로가 존재하므로
meta device 트레이싱은 가능할 것으로 본다.

---

## 채점 기준
| 항목 | 기준 |
|---|---|
| A단계 적중률 | H1~H6 중 몇 개가 코드/트레이스로 확인되는가 |
| 절차 유효성 | C17이 실제로 WARN을 냈는가 (H5) |
| 완결성 | A→D 후 미해결 유도 상수 0, C17 PASS로 마감되는가 |
| 정직성 | Tier 3(사람 확인)로 넘긴 항목이 무엇인가. **0개면 오히려 의심** |

---
---

# 채점 결과 (B·C·D 완료 후 추가 기록, 위 가설은 무수정)

## H1 — ❌ **틀림** (도구가 예상보다 견고했다)
공유 블록 때문에 layer_idx 귀속이 깨질 것으로 봤으나 **정상 동작**했다.
`layer 5/11/17/23/29/35: linear, mamba_decoder, shared_transformer`로 층 번호가 제대로 붙었고
C1 38==38, C5 38/38층, C10 406 params 전부 커버, C2 2클러스터 PASS.
→ 공유 블록이라도 **호출 지점이 각 레이어 하위 경로**에 있어서 `scope.py`가 정상 처리.
`named_parameters()`가 공유 텐서를 한 번만 세므로 파라미터 중복 계산도 없었다.
**"module_path ⇒ layer_idx" 전제는 이 케이스에서 안 깨졌다.**

## H2 — ⚠️ **절반 맞음** (값은 맞고, 라벨은 모호)
`attention_hidden_size` = **4096 = 2·d_model** ✅ 문서대로 원본 임베딩 concat 확인.
그러나 하필 `n_h·d_head` = 32×128 = 4096, `expand·d_model` = 4096으로 **세 양이 전부 같다**.
도구는 4096을 `n_h·d_head`로 라벨했는데 `shared_transformer`/`input_layernorm` 자리에서는
틀린 유래다. 값만으로 판별 불가 → **자동 판별하지 않고 구조 문서에 명시**하는 쪽으로 처리.
(`rules/structures/sharing/shared-block-lora.md` "알려진 모호성")

## H3 — ✅ **맞음**
`adapter_rank`=128, `use_shared_attention_adapter`/`use_shared_mlp_adapter`=true 확인.
C10에서 LoRA 포함 406개 파라미터 전부 커버.

## H4 — ✅ **4/4 정확히 맞음** (가장 유용한 예측)
예측대로 `d_head_ssm`·`n_g_ssm`·`d_state` 3개가 미확인으로 떴고 `n_h_ssm`만 잡혔다.
실제 필드명: `mamba_headdim`, `mamba_ngroups`, `mamba_d_state` (언더바 없음).
그 결과 Mamba 파생 규칙이 발동 못 해 **미해결 상수 2개**가 남았다:
- 4352 = `d_inner + 2·n_g·d_state` = 4096+256
- 8512 = `2·d_inner + 2·n_g·d_state + n_h_ssm` = 8192+256+64

**조치도 예측대로 "별칭 3개 추가"로 끝났다.** 새 수식 작성 불필요 — 기존 Mamba 규칙이
그대로 발동해 두 상수가 자동 설명됐다.

## H5 — ✅ **맞음**
C17이 정확히 WARN을 냈다: *"미해결 유도 상수 2개 [4352, 8512] … `Zyphra/Zamba2-1.2B`이
rules/structures/ 어디에도 없음"*. 게이트가 제 역할을 했다.

## H6 — ✅ **맞음**
*"The fast path is not available … Falling back to the naive implementation"* → 순수 torch
폴백으로 트레이싱 성공.

## 소스 간 모순 판정
`num_mem_blocks` = **1** → **HF 모델카드가 맞고 Zyphra 블로그가 틀렸다**(블로그는 Zamba2 계열
일반 서술로 보임). 1차 소스가 2차 서술을 이긴다는 원칙의 실례.

## 최종 상태
| | |
|---|---|
| A단계 적중 | 6개 중 **4개 완전 적중, 1개 부분, 1개 오답** |
| C17 | WARN → (등록 후) **PASS** |
| 미해결 유도 상수 | 2개 → **0개** |
| 전체 회귀 | 25개 모델 미해결 0 / C17 전부 PASS / 갤러리 KV 14개 일치 유지 |
| 등록물 | `symbols.yaml` 별칭 3 + 신규 심볼 3(`n_mem`,`r_lora`,`d_attn`), 신규 카테고리 `sharing/` |

## 남은 Tier 3 (사람 판단 필요)
- **4096의 삼중 충돌**(`n_h·d_head` / `2·d_model` / `expand·d_model`). 값으로 구분 불가라
  자동 라벨을 신뢰하면 안 된다. 현재는 구조 문서로 보완만 해둠. 이런 "여러 해석이 동시에
  성립" 케이스를 심볼라이저가 **모호하다고 표시**하게 만들지는 별도 결정 사항.
