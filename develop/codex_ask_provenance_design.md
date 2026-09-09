# provenance 보존 — 구현 전에 설계를 봐 주세요

라벨 정확도 라운드는 마무리됐습니다(마지막 판정: "339건 교정 PASS, 회귀 없음").
남은 결함이 전부 하나로 수렴했고, 다음 작업이 **`src/axis_classes.py` 의 provenance 보존**
하나라는 데 합의했습니다.

**코드를 쓰기 전에 설계를 검토받고 싶습니다.** 47개 모델 전부에 영향이 가는 코어 변경이고,
이번 라운드에만 값 기반 규칙으로 두 번 크게 되돌렸기 때문입니다(아래 "저희가 실제로 겪은 것").

## 지금 구조 — 무엇이 문제인가

`src/axis_classes.py` 가 "어느 칸들이 같은 축인가"를 계산해 등가류(union-find)를 만들고,
`build_table` 이 등가류마다 이름을 하나 고릅니다. 간선을 놓는 조건은 두 가지뿐입니다:

1. **생산자→소비자**: 피연산자 중 **구체 shape 이 정확히 하나만 일치**할 때
2. shape 을 보존하는 elementwise/copy 계열 op

즉 **간선의 근거가 정수값**입니다. 그래서 값이 겹치는 두 축이 한 등가류로 묶입니다.

| 모델 | 묶이는 두 축 | 값 | 드러나는 지표 |
|---|---|---|---|
| moonshotai__Kimi-K3 | MLA `n_h`/`d_v` ↔ KDA `n_h_kda`/`d_head_kda` | 96 / 128 | 전치 261, 이름생성 345 |
| ibm-granite__granite-4.0-h-small | dt·A·B/C 의 head ↔ state | 128 | expand 216 |
| nvidia__Nemotron-3-Super-120B | 〃 | 128 | expand 80 |
| Zyphra__Zamba2-1.2B | mamba head ↔ head feature | 64 | 전치 12, 이름생성 6, expand 18 |
| zai-org__GLM-5.2 | 캐시 key ↔ value | 256 | (J3~J7 미반영) |
| Qwen__Qwen3-Next-80B | conv_dim ↔ `2*n_h*d_head` | 8192 | (A2 미반영) |
| MiniMax-M2 / OLMoE / V4-Flash 2종 | experts gate-up 원본 ↔ 전치본 | `d_model = 2*d_moe` | param_incons |

## 저희가 실제로 겪은 것 — 왜 값 기반으로는 더 못 가는가

이번 라운드에 **두 번 크게 되돌렸고, 둘 다 실측이 근거**였습니다.

1. **교정 뒤 전치 불변식 재적용**: Kimi-K3 전치 261→168 이지만 **등가류 충돌 0→117**,
   Zamba2 는 전치 12→0 이지만 **정당한 `repeat_kv(n_rep=1)` 경계를 밀어버림**.
2. **head ↔ state 맞바꿈 교정**: 자리 단위로 넣으니 **등가류 충돌 0→887**.
   결함 653 을 887 로 바꾸는 거래라 하지 않았습니다.

여기서 얻은 규칙 하나: **맞바꿈은 `spread: class` 로 표현할 수 없습니다.** `spread` 는 한
클래스에 이름 하나를 쓰는 연산이고, 같은 등가류 안의 두 축을 서로 바꾸는 교정은 자리
단위로 넣을 수밖에 없어 반드시 클래스를 쪼갭니다. **한 방향 교정만 지금 도구로 풀립니다**
(그래서 `k_pe` feature = `d_rope` 339건은 풀렸습니다).

## 지난 라운드에 주신 origin 목록

- 생성 위치와 operand/axis 번호
- `split` 의 출력 슬롯 번호
- `concat` 구성 항의 순서
- `Cache.update` 의 key/value 인자 위치
- MLA / KDA / SSM / linear-attention 같은 **branch namespace**
- transpose/view/reshape/expand 를 통과한 축 계보

## 질문 1 — origin 키의 형태와 union 조건

이 여섯 가지를 **어떤 자료구조로** 들고 다니는 것이 좋을지 봐 주세요. 저희가 생각한 것은
축마다 불변 origin 하나를 붙이고, union 조건을 "구체 shape 일치" 대신 **"origin 이 같거나
계보로 이어진다"** 로 바꾸는 것입니다.

```
AxisOrigin = (
    producer_op_id,      # 그 축을 처음 만든 op
    producer_field,      # 'o' | 'i' | 'w'
    producer_slot,       # split 출력 슬롯 / concat 구성 항 번호 / Cache.update 인자 위치
    producer_axis,       # 그 op 안에서의 축 번호
    branch,              # 'mla' | 'kda' | 'ssm' | 'linear_attn' | 'moe' | None
    config_field,        # 이 폭을 낸 config 필드 (알 때만)
)
```

구체적으로 궁금한 것:

1. **`branch` 를 어떻게 정하는 것이 안전합니까?** 저희는 module_path 와 config 의 층
   스케줄(`layer_types`, `linear_attn_config.full_attn_layers` 등)로 유도하려 합니다.
   그런데 Kimi-K3 는 `layer_types` 가 비어 있고 `linear_attn_config` 도 없습니다.
   이런 모델은 module_path 의 형태(예: `q_conv1d` 가 있으면 KDA)로 추정해야 하는데,
   추정이 틀리면 오히려 지금보다 나빠집니다. 더 나은 근거가 있습니까?

2. **origin 이 다르면 절대 union 하지 않는 것이 맞습니까?** 지금 등가류가 잘못 묶는 것만큼,
   **끊어야 할 것을 안 끊는 것**도 문제지만 **이어야 할 것을 못 잇는 것**도 문제입니다
   (예: `_squeeze_view_keeps_names` 가 존재하는 이유). 어느 쪽으로 기울이는 것이 맞습니까?

3. **`Cache.update` 의 인자 위치를 어떻게 잡습니까?** 저희 트레이서는 `TorchDispatchMode` 로
   ATen 수준을 봅니다. `Cache.update` 는 파이썬 레벨이라 dispatch 에 안 보이고, 저희가 보는
   것은 그 안의 `concat` 뿐입니다. 모듈 후크를 따로 걸어야 합니까?

## 질문 2 — 전치 간선

저희 등가류에는 **전치 간선이 없습니다.** 예전에 넣었다가 물렸고, 이유는
`repeat_kv(n_rep=1)` 경계입니다 — `n_rep=1` 이면 `repeat_kv` 가 shape 을 안 바꾸므로
`n_h` 축과 `n_kv` 축이 한 클래스로 묶여 버립니다(`src/axis_classes.py` 헤더에 기록).

그 결과 Qwen3-Next A2 교정이 전치에서 멈춰 전치 위반 72 를 만들었습니다. 지적하신 대로
"별도 예외가 아니라 같은 provenance graph 단절"이 맞다고 봅니다.

**origin 이 있으면 전치 간선을 안전하게 넣을 수 있습니까?** 전치는 축을 재배열만 하므로
origin 을 그대로 옮기면 되고, `repeat_kv(n_rep=1)` 은 origin 이 다르므로(`n_h` 쪽과 `n_kv`
쪽의 producer 가 다름) 묶이지 않을 것으로 봅니다. 이 추론에 구멍이 있습니까?

## 질문 3 — 검증 방법

**되돌린 314건을 척도로 쓰려 합니다.** 이것들은 정답이 소스로 확정돼 있으므로
(Granite 216 / Nemotron-Super 80 / Zamba2 18), provenance 가 제대로 들어갔다면 **교정 없이
저절로 풀려야** 합니다. 풀리지 않으면 설계가 부족한 것입니다.

- 이 척도로 충분합니까? 통과했는데도 남을 수 있는 결함이 있습니까?
- 회귀를 잡으려면 무엇을 더 봐야 합니까? 지금 게이트는 등가류 무모순 / 미발화 교정 /
  C-check / reshape_incons / 전치·이름생성·expand 불변식 / `bare` 퇴행을 봅니다.
- **`expand` 검사는 자동 교정이 아니라 acceptance gate 로 유지**하라고 하셨는데, 다른
  불변식(전치·이름생성)도 같은 취급으로 바꾸는 것이 맞습니까? 지금 `_transpose_swaps_names`
  는 **교정**입니다(전치의 출력 = 입력의 순열이라는 정의라서). origin 이 들어오면
  이것도 게이트로 돌리고 이름은 origin 이 정하게 하는 편이 낫습니까?

## 질문 4 — 순서

한 번에 다 바꾸면 회귀 원인을 못 가립니다. 저희가 생각한 순서는:

1. origin 을 **기록만** 한다(라벨 결정에는 안 씀). 사이드카로 내보내고, 지금 등가류가
   origin 이 다른 축을 묶는 자리가 몇 개인지 **먼저 센다.**
2. 그 수를 보고 union 조건을 바꾼다.
3. 전치 간선을 넣는다.
4. 314건이 풀리는지 확인한다.

1번의 "먼저 센다"가 이번 라운드의 교훈입니다 — 규칙을 넣기 전에 함대 전체에서 잡을 행과
**깨뜨릴 행**을 세는 것. (그때 "B 는 축 0 이어야 한다"는 넓은 규칙이 307,958행을 오판하는
것을 발견해 버렸습니다.) 이 순서에 빠진 단계가 있습니까?

## 참고 — 지금 상태

47개 모델, 등가류 0 / 미발화 0 / C-FAIL 0 / unsqueeze singleton 0.
전치 277 · 이름생성 351 · reshape_incons 474 · expand 314.
`results` 브랜치에 기준선을 박아 뒀습니다(14개 모델 — "판단 필요 0건 + 검토 기록 있음"을
둘 다 만족하는 것만).
