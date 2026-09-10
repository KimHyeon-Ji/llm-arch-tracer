# 모르는 라벨 전수 목록이 나왔습니다 — 남은 결정 셋을 묻습니다

지시하신 3~6단계를 끝냈습니다. 목표 4개 모델을 **버리는 canary** 로 돌렸고(`models/` 에도
`develop/out` 에도 승격 안 했습니다), 축 판정 원장으로 모르는 라벨을 전수로 뽑았습니다.

## 1. 결과

```
| 모델                  |      자리 |      확정 | scope_inf | heuristic | open_tie | unresolved | 질문 |
|-----------------------|----------:|----------:|----------:|----------:|---------:|-----------:|-----:|
| Kimi-K3               | 5,572,323 | 1,910,978 | 3,149,889 |    62,663 |    5,400 |    443,393 |   14 |
| DeepSeek-V4-Pro       |   657,120 |   491,634 |    99,014 |    33,744 |   26,707 |      6,021 |   12 |
| Llama-4-Maverick      |    73,070 |    62,206 |     9,456 |       736 |        0 |        672 |    4 |
| gpt-oss-120b          |    69,254 |    51,396 |    11,610 |       556 |    5,620 |         72 |    8 |
| gpt-oss-20b           |    46,454 |    35,740 |     6,542 |       376 |    3,748 |         48 |    6 |

합계 자리 6,418,221 / 확정 2,551,954 (39.8%) / 모름 3,866,267 (60.2%)  ->  **질문 44개**
```

**합계 비율은 오해를 부릅니다.** Kimi-K3 는 KDA 참조 구현이 파이썬 루프를 돌아 트레이스가
63만 행이고, 자리 수가 나머지 넷을 합친 것의 6배입니다. Kimi 를 빼면 확정 75.8% /
모름 24.2% 입니다. **모델별로 봐야 합니다.**

큰 질문들:

```
1,316,451축  scope_inferred  n_h|n_h_kda|n_kv         -> n_h_kda      Kimi-K3
  952,062축  scope_inferred  d_chunk|d_rope           -> d_chunk      Kimi-K3
  880,992축  scope_inferred  d_head_kda|d_nope|d_v    -> d_head_kda   Kimi-K3
  338,963축  unresolved      bare                     -> 5            Kimi-K3
   82,772축  scope_inferred  m_csa|n_hc               -> n_hc         V4-Pro
   49,059축  heuristic       reused_symbol            -> d_chunk      Kimi-K3
   33,428축  heuristic       reused_symbol            -> n_hc         V4-Pro
   17,812축  open_tie        c_I|m_hca|n_h|w_local    -> n_h          V4-Pro
    9,360축  open_tie        d_head|n_h               -> n_h          gpt-oss(120b+20b)
    8,352축  scope_inferred  E|d_head                 -> d_head / E   Llama-4
```

Kimi-K3 의 세 질문이 314만축을 덮습니다. 전부 KDA 층이고, 겹치는 값은 `n_h = n_kv =
n_h_kda = 96`, `d_nope = d_v = d_head_kda = 128`, `d_rope = d_chunk = 64` 입니다.

## 2. 원장이 새로 드러낸 것 둘

### (가) `reused_symbol` — 값 충돌이 아닌데 이름이 지어진다

옛 지표(`ambiguous.json`)에는 **한 줄도 안 나오던** 것입니다.

```
V4-Pro    33,428축   reused_symbol -> n_hc
Llama-4      736축   reused_symbol -> T
gpt-oss      932축   reused_symbol -> T
```

`resolve_shape` 는 같은 shape 안에서 이미 쓴 심볼을 마지막 수단으로 재사용합니다. 그런데
**재사용은 근거가 아닙니다.** 지금 저는 이것을 `heuristic` 등급으로 뒀습니다.

### (나) 같은 후보 쌍이 자리에 따라 등급이 갈린다

gpt-oss-120b 의 `d_head|n_h`(둘 다 64):

```
2,844축  open_tie        scope 통과 = d_head|n_h   -> n_h
2,160축  scope_inferred  scope 통과 = d_head       -> d_head
```

같은 값·같은 후보인데 모듈에 따라 scope 가 하나로 좁히기도 하고 아니기도 합니다. 그래서
질문을 **`(후보, scope 통과 집합, 라벨, 등급)`** 으로 갈라야 하는 것으로 보입니다.

## 3. 그동안 만든 것

```
src/axis_ledger.py                       자리별 판정 원장. 등급 5개, intern, chain
src/symbolic_shape.py                    `_r()` 단일 통로와 `_pick()` 에서 계측
src/build_table.py                       `_ordered_row` 가 자리 id, `anchors.relabel` 이 덮어쓰기
full/<phase>.axis_resolution.jsonl       확정 아닌 자리 + 질문 + 요약(`coverage_ok`)
develop/unknown_inventory.py             위 표를 내는 도구
develop/test_axis_ledger.py              6/6 (다섯 등급 도달 가능성, 등식, 지어낸 이름 은닉 방지 …)
```

지시하신 대로 `_dim_core` 만이 아니라 앵커 덮어쓰기까지 같은 원장에 chain 으로 남깁니다.
발행 라벨은 한 글자도 안 바뀝니다(csv/jsonl 바이트 동일 확인).

`ties` 회계 버그 세 개(재시도·사전 렌더가 `ties` 를 안 되돌림, canary 의 counter key)도 전부
고쳤습니다. `axis_decision_trace.py` 는 밖에서 후보를 재구성하던 것을 버리고 원장을 읽습니다.

---

## 질문

**Q1 (CSV 표시 — 실제 비율이 나왔습니다).** 제안하신 등급 표기를 그대로 쓰면 **자리의
24.2%** 에 표시가 붙습니다. Llama-4 는 `scope_inferred` 만 9,456축이라 published shape 상당수에
`~` 가 붙습니다. 그래도 전부 표시하는 것이 맞습니까, 아니면 `scope_inferred` 는 평문으로
두고 `open_tie`/`unresolved` 만 표시합니까? 후자는 "사람이 계속 속는다" 는 지적과 충돌하는데,
전자는 표가 거의 다 물결표가 됩니다.

**Q2 (`reused_symbol` 의 등급).** 같은 shape 에서 이미 쓴 이름을 재사용한 5만여 축입니다.
근거가 **전혀** 없다는 점에서는 `bare` 와 같습니다. `heuristic` 이 아니라 `unresolved` 로
내려야 합니까? 지금은 이름이 붙어 있어 `unresolved` 로 내리면 라벨을 지워야 합니다.

**Q3 (질문 키).** 위 (나) 때문에 질문 키에 **scope 통과 집합**을 넣으려 합니다:
`(raw 후보, scope 통과 집합, 라벨, 등급)`. 이러면 gpt-oss 의 `d_head|n_h` 가 두 질문으로
갈립니다. 맞습니까? 아니면 모듈까지 넣어야 합니까(그러면 질문 수가 늘어납니다)?

**Q4 (게이트).** `scope_only_occurrences` / `scope_only_classes` / `scope_only_questions` 를
보고하라고 하셨습니다. **임계값**은 어떻게 잡습니까? 절대값은 모델 크기에 따라 달라지고,
비율은 좋아 보이지만 지금 기준선이 없습니다. 첫 재트레이스의 값을 기준선으로 박고
"악화만 FAIL" 로 가는 것이 맞습니까?

**Q5 (LLM 질문 묶음의 형식).** 질문 30개를 LLM 에 넘길 프롬프트를 만들려 합니다. 담을 것:
모듈 경로, op 종류, 심볼·구체 shape, 그 모듈이 읽는 config 필드와 값, 소스 조각, 후보,
자리 표본. 답에 요구할 것은 지난번에 주신 여섯 가지(소스 revision/SHA·줄 범위, config
필드에서 지역 변수까지의 binding, dataflow, 라벨과 축 크기의 식, 반증 조건, 가능하면
counterfactual probe)입니다. **더 넣거나 뺄 것이 있습니까?** 특히 `scope_inferred` 질문은
"이 scope 가 맞는가" 를 묻는 것이라 형식이 달라야 할 것 같습니다.

**Q6 (Kimi-K3 의 `bare -> 5` 338,963축).** 이름 없이 정수 `5` 로 남은 축이 34만입니다.
KDA 참조 구현의 내부 상수로 보이는데, 이런 것은 **이름이 없는 것이 정답**일 수 있습니다
(`review_notes` 에 `no_name_exists` 판정이 있습니다). `unresolved` 를 전부 질문으로 만들면
안 될 것 같은데, "이름이 없는 것이 정답" 을 어떻게 확정하고 기록합니까?

**Q7 (순서).** Q1~Q3 을 확정해 CSV 표시와 게이트를 먼저 넣고 그다음 LLM 질문 묶음으로
갑니까? 아니면 질문 44개를 먼저 LLM 에 돌려 답이 얼마나 쓸 만한지 보고 표시 설계를
정합니까? 저는 전자가 맞다고 봅니다 -- 후자는 "구현 후 결과에 맞춰 기준을 세우는" 순서라
전에 지적하신 것과 같습니다.
