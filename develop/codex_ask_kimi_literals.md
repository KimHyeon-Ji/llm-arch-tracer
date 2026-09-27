# Kimi-K3 의 맨 정수 8,128 자리 -- **식으로 바꿔야 합니까, 리터럴로 남겨야 합니까**

요청일: 2026-09-27. 선행: `codex_ask_direct_research.md`(1 단계 783 단위 직접 조사),
`codex_ask_provenance_spread.md`(spread: provenance_class 추가).

사용자의 목적은 **심볼로 매개화된 roofline + DAG 를 만들어 배치 크기와 config 를 스윕하는
것**입니다. 그래서 맨 정수로 남은 축은 "값이 맞다" 로 끝나지 않습니다 -- 스윕하면 틀립니다.

이 질문은 그 축들을 어떻게 처리할지 하나입니다. 판정에 필요한 수치는 전부 아래에 있습니다.

---

## 1. 현재 상태 -- 다섯 모델 중 넷은 이미 끝났습니다

발행본 csv 의 축 토큰을 전수 세었습니다(`1` 은 정당한 singleton 이라 제외).

```
모델                                     행      축     맨정수    비율   caveat행  unmapped
meta-llama__Llama-4-Maverick-17B-128E   138    1074       0    0.00%        0        8
openai__gpt-oss-20b                     102     752       0    0.00%        0       12
openai__gpt-oss-120b                    102     752       0    0.00%        0       12
deepseek-ai__DeepSeek-V4-Pro            436    3548       0    0.00%        0       30
moonshotai__Kimi-K3                   17532  142332    8128    5.71%     2944      188
```

**Kimi-K3 하나만 남았습니다.**

## 2. K3 의 8,128 자리는 세 갈래이고, **둘은 정확한 항등식입니다**

심볼표와 adaptation_log 에서 확인한 값:

```
B = 3,  T = 320,  k = 16,  E = 896,  d_chunk = 64
adaptation_log: remedy "moe_infer_even_split",  expert_cap = 4,  experts_per_layer = 896
```

### (a) `3840` / `12` -- 4,784 자리, **전부 caveat 행 안**

```
prefill   3840 = B*T*k / expert_cap = 3*320*16 / 4
decode      12 = B*k   / expert_cap = 3*16    / 4
```

자리: `block_sparse_moe/experts` 의 matmul / tanh / sigmoid / elementwise_mul / concat.
예: `matmul i=[[3840, d_moe_lat], [d_moe_lat, d_moe]] -> o=[[3840, d_moe]]`

caveat 문구(2,944 행 전부 동일):

> 이 MoE 블록의 전문가 디스패치는 대체됐다: 레이어당 전문가 896개 중 4개만 트레이스했고,
> 전문가에 들어가는 토큰 수는 정렬된 토큰을 균등 분할한 **대체값**이다 -- 실제 값은 라우팅이
> 정하는 런타임 데이터라 한 번의 트레이스로는 알 수 없다. 라우터(gate)·scatter·argsort·
> 가중합은 모델 자기 코드 그대로이고, 전문가 projection 의 op 구성과 폭도 충실하다.
> 토큰 축 크기만 신뢰하면 안 된다

**중요한 점**: 균등 분할이므로 `expert_cap` 개 행의 **합은 정확히 `B*T*k`** 입니다
(4 × 3840 = 15,360 = 3*320*16). 즉 **총량은 맞고 분포만 합성**입니다.

### (b) `5` -- 2,160 자리, caveat 없는 실측 축

```
5 = T / d_chunk = 320 / 64 = n_chunk
```

자리(발행본): `self_attn` 의 `exp`, shape `[B, n_h_kda, 5, d_chunk, d_head_kda]` 의 축 2.
KDA 청크 스캔입니다(`fla/ops/kda/naive.py:108-109` 이 T 를 `NT = T//BT` 개 청크로 쪼갭니다).

이 항등식은 이미 등록돼 있습니다 -- `src/build_table.py:_MERGE_IDENTITY_EQUIVALENTS =
{"T": {"d_chunk*n_chunk"}}` 이고, `5 -> n_chunk` 교정도 **한 자리에만** 걸려 있습니다.

`self_attn` 안에서 구체값 5 인 축을 전수 확인했습니다 -- **전부 축 2**(`[B, n_h_kda, 5, ...]`)
이고 의미가 균일합니다. 다른 뜻으로 5 가 쓰인 자리는 `self_attn` 밖입니다.

### (c) `2,3,4,6,7,8,9` -- 약 1,100 자리, **축이 아닙니다**

누적 concat 의 항목 수입니다: `concat i=[[B*T,4,d_model],[B*T,1,d_model]] -> [[B*T,5,d_model]]`.
KDA 청크 내부 causal loop 의 prefix 길이도 여기 속합니다(`naive.py:101-125`,
`for i in range(1, BT)`). 리터럴이 맞다고 판단했습니다 -- 루프 인덱스는 축이 아닙니다.

## 3. 무엇이 걸림돌인가

### (b) 의 도달 범위

발행본은 접혀 있어 2,160 자리지만 **원시 원장에는 245,433 자리**입니다. 실측했습니다:

```
앵커: self_attn$ / linear_attention / exp / i[0] / ax2 / [B, n_h_kda, 5, d_chunk, d_head_kda]

spread: class              발화  90,597   남은 (5 & d_chunk) 154,698
spread: provenance_class   발화 108,261   남은              137,034
```

둘 다 못 닫습니다. 남은 자리의 op 가 넓습니다 -- `select` 34,914, `copy_` 13,248,
`view`/`permute`/`clone` 각 8,832, `slice` 8,694, `elementwise_mul` 4,554, `unsqueeze` 4,485 …

**부분만 바꾸면 등가류 충돌이 납니다.** 바로 앞 라운드에서 같은 함정에 걸렸습니다
(`한 축에 이름이 둘 이상인 등가류 48건`, `flow_ambig 0 -> 96`). 그래서 전부 덮거나 전혀
안 건드려야 합니다.

**넓은 앵커 하나로 덮는 안**: `self_attn` 안 `linear_attention` 층에서 구체값 5 인 축은
전부 청크 개수이므로, `module: self_attn$` + `layer_types: [linear_attention]` +
`from: "5"` + `expect: 5` 만으로 shape/op_type 없이 거는 것. 위 전수 조사가 의미 균일성의
근거입니다. 다만 [[measure-before-promoting-a-rule]] 의 교훈(넓은 규칙 하나가 30만 행을
오판했다)이 정확히 이 형태였습니다.

### 비용

K3 재트레이스가 **64 분**입니다(방금 17:30 에 끝났습니다). 검증 라운드마다 한 번씩 듭니다.

## 4. 확인받고 싶은 것

### Q1. (a) `3840` / `12` 를 식으로 바꿔도 됩니까

사용자는 "다 심볼로 바꾸고 싶다" 고 합니다. 제 판단은 **바꾸는 쪽이 더 정직하다** 입니다:

```
찬성  - 3840 은 이 (B, T, k, expert_cap) 조합에서만 맞는 수다. 리터럴로 두면 배치를
        스윕할 때 조용히 틀린다. 식으로 두면 대입이 따라온다.
      - 합이 정확히 B*T*k 라 roofline 총량은 옳다.
      - caveat 는 모든 행에 그대로 남으므로 "분포는 합성" 이라는 공개가 사라지지 않는다.
반대  - `expert_cap` 은 config 필드가 아니라 **트레이스 파라미터**다. 아키텍처 심볼과
        같은 자리에 두면 읽는 쪽이 모델 속성으로 오해할 수 있다.
      - 실제 라우팅은 균등 분할이 아니다. 식이 "이 분포가 맞다" 는 주장으로 읽힐 수 있다.
```

**어느 쪽입니까?** 바꾸는 것이라면 이름을 어떻게 해야 합니까 --
`B*T*k/E_t` 처럼 새 심볼 `E_t`(traced experts)를 심볼표에 **트레이스 파라미터로 표시해서**
넣는 것이 맞습니까, 아니면 다른 표기가 낫습니까?

### Q2. (b) `5 -> n_chunk` 를 넓은 앵커로 덮어도 됩니까

의미 균일성은 전수 조사로 확인했습니다(자리 245,433 개 전부 `[B, n_h_kda, 5, ...]` 의 축 2).
그래도 shape/op_type 없는 넓은 앵커는 위험 계열입니다. 승인하시겠습니까, 아니면
op 별 앵커를 쌓아야 합니까(그러면 라운드가 여러 번이고 매번 64 분입니다)?

### Q3. (c) 를 리터럴로 두는 판단이 맞습니까

누적 concat 의 항목 수와 causal loop prefix 길이입니다. 심볼을 새로 만들지 않는 것이
맞다고 봤는데(축이 아니므로), 사용자 목적(roofline)에서 이 자리들이 문제가 됩니까?
`[B*T, 5, d_model]` 의 5 는 층 누적 개수라 `n_layer` 계열일 수도 있어 보입니다 -- 확인이
필요하면 지적해 주십시오.

### Q4. 순서

같은 라운드에 (a)+(b) 를 함께 넣고 한 번만 재트레이스하는 것이 맞습니까, 아니면 하나씩
넣어 도달 범위를 따로 확인해야 합니까? 앞 라운드 경험으로는 **한 번에 넣으면 어느 교정이
어디까지 갔는지 안 갈립니다.**

## 5. 읽을 곳

```
models/moonshotai__Kimi-K3/prefill.csv           맨 정수가 실제로 보이는 곳
models/moonshotai__Kimi-K3/full/provenance.json  symbol_table, adaptation_log(expert_cap 4)
models/moonshotai__Kimi-K3/UNKNOWNS.md           지금 공개 중인 모름 목록
rules/label_overrides.yaml                       기존 `5 -> n_chunk` 항목(한 자리)
src/build_table.py                               _MERGE_IDENTITY_EQUIVALENTS
```

산출물은 읽기만 하고 **수정하지 말아 달라.**

## 6. 원하는 판정

> Q1(`3840`/`12` 를 식으로 바꿀지, 바꾸면 어떤 표기로)이 가장 중요합니다. 사용자는 전부
> 심볼을 원하고 저는 그게 더 정직하다고 보지만, "트레이스 파라미터를 아키텍처 심볼처럼
> 보이게 한다" 는 반론이 실질적이라 판단을 받고 싶습니다.
>
> Q2(넓은 앵커 승인 여부), Q3(리터럴 유지 판단), Q4(한 라운드에 묶을지)도 정해 주십시오.
