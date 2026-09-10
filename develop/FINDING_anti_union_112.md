# anti-union 112 원인 분해 (2026-09-10)

외부 검토 지시 2단계: "anti-union 112 를 원인별로 쪼개고, #4 의 직접 대상이 Zamba2 24 뿐인지
확인하라." 세 모델 모두 **`batched_matmul` 의 앞쪽 배치 축 규칙**에서 합쳐진다
(`src/axis_classes.py:270-283`). 그런데 **원인은 셋 다 다르다.**

핵심: 세 경우 모두 **union 은 옳고 라벨이 틀렸다.** matmul 의 배치 축은 두 피연산자와 출력이
정의상 같은 축이다. 따라서 #4(축별 semantic barrier)로 112 가 0 이 되지 않는다 -- 24 만 된다.

---

## 1. Zamba2-1.2B — 24건. **#4 의 직접 대상.**

```
op856 batched_matmul   shared_transformer.self_attn
   inp0  [n_h,  T,      d_head]   [32, 16, 128]
   inp1  [n_kv, d_head, T     ]   [32, 128, 16]
   out0  [n_h,  T,      T     ]   [32, 16, 16]
```

`n_h = n_kv = 32`, `n_rep = 1`, noop barrier 12개. `repeat_kv` 가 no-op 이라 텐서는 그대로인데
**역할이 n_kv -> n_h 로 바뀌었다.** 축은 하나가 맞고 이름만 안 바뀌었다. `repeat_kv` 자리에
축별 barrier 를 넣어 그 축을 `n_h` 로 다시 이름 붙이면 사라진다.

## 2. Nemotron-3-Super — 40건. **view 의 축 병합 오라벨. #4 무관.**

```
op85 view    inp0 [B, n_g_ssm, n_h_ssm/n_g_ssm, d_state]  [1, 8, 16, 128]
             out0 [B, d_state, d_state]                   [1, 128, 128]
op89 batched_matmul
             inp0 [n_h_ssm, d_head_ssm, d_state]  [128, 64, 128]
             inp1 [d_state, d_state, B]           [128, 128, 1]
             out0 [n_h_ssm, d_head_ssm, B]        [128, 64, 1]
```

op85 가 축 1,2 를 하나로 접는데 `n_g_ssm * (n_h_ssm/n_g_ssm) = 8*16 = 128 = n_h_ssm` 이다.
그런데 `d_state` 도 128 이라(**값 충돌**) 출력 축 1 이 `d_state` 로 찍혔다. op89 의 배치 축은
실제로 `n_h_ssm` 이고 union 은 옳다. semantic 이벤트는 `n_rep=16`, noop barrier 0 -- barrier 로
풀 대상이 아니다.

## 3. Kimi-K3 — 48건. **같은 축에 두 이름. #4 무관.**

```
op58713 batched_matmul   self_attn
   inp0  [n_h,     T, T          ]   [96, 320, 320]
   inp1  [n_h,     T, d_head_kda ]   [96, 320, 128]
   out0  [n_h_kda, T, d_head_kda ]   [96, 320, 128]
```

`n_h = n_h_kda = 96`. **한 op 안에서** 입력은 `n_h`, 출력은 `n_h_kda` 다. 같은 배치 축이므로
둘 중 하나가 틀렸다(층 타입이 KDA 면 입력 쪽이 틀렸다). semantic 이벤트 **0개** -- barrier 가
아예 없는 모델이라 #4 로는 건드릴 수 없다.

---

## 계획에 미치는 영향

`anti_union` 지표는 **잘못 이은 것**과 **한 축에 두 이름이 붙은 것**을 구분하지 않는다.
`provenance` 모드가 두 축을 옳게 이으면서 기존 라벨 불일치가 드러난 것이 40+48 = 88 이다.
그래서:

* #4 완료 시 기대값은 `112 -> 0` 이 아니라 **`112 -> 88`** 이다. Zamba2 24 만 사라진다.
* 남는 88 은 라벨 결함이고 ④층(`label_overrides`) 또는 reshape 규칙이 다룰 문제다.
* 지표를 그대로 두면 #4 가 "실패" 로 보인다. **`anti_union` 을 두 지표로 나눠야 한다** --
  `wrongly_joined`(잘못 이음)와 `name_conflict`(한 축 두 이름).

---

## 후속 (2026-09-10, 외부 검토 반영)

외부 검토가 결론을 승인하면서 **지표 설계를 고쳤습니다.**

`anti_union` 은 이름 문자열만 보므로 `wrongly_joined` 를 판정할 수 없습니다. 셋으로 나눴습니다.

| 지표 | 뜻 | 게이트 |
|---|---|---|
| `symbol_pair_collision` | 옛 `anti_union`. 두 이름이 한 등가류에 | **진단만.** FAIL 아님 |
| `forbidden_union` | 구조로 고정한 site 쌍이 같은 UF | **하드 게이트** |
| `class_label_conflict` | 한 등가류에 서로 다른 렌더 라벨 | 라벨 층의 일 |

Kimi-K3 와 Nemotron 의 쌍은 **문맥에 따라 같은 축일 수 있어** 전역 금지 목록에서 뺐습니다.
Zamba2 도 `n_h vs n_kv` 문자열 쌍이 아니라 **`repeat_kv` 의 이벤트 전 자리와 이후
SemanticPort 자리**를 negative contract 로 고정해야 하며, 그건 #5 가 있어야 지목할 수
있습니다. 그래서 `FORBIDDEN_UNION` 은 지금 비어 있습니다.

실측: `class_label_conflict` 는 provenance 에서 **1,496 등가류 / 9개 모델**입니다
(legacy 는 0 -- 파이프라인이 등가류마다 이름을 하나로 강제하기 때문입니다). 112 는 우리가
목록에 적어 둔 쌍만 센 값이었고, 계보와 라벨이 어긋나는 실제 규모는 이쪽입니다.

### #4/#5 의 단계별 합격 기준 (외부 검토가 고쳐 준 것)

`112 -> 88` 은 **graph topology 만의 기준이 아닙니다.** SemanticPort 를 넣어도 하류 행에
찍힌 `n_kv` 문자열이 저절로 `n_h` 가 되지는 않습니다(`_class_name_pairs` 는 이미 렌더된
문자열을 읽습니다).

```
semantic graph 완료      Zamba upstream 축과 post-repeat 축이 서로 다른 root
                         하류 head 축은 SemanticPort 와 같은 root
                         Kimi/Nemotron 은 partition 변화 0
semantic anchor 를 이름 결정기에 적용한 뒤    Zamba 24 제거 -> 112 -> 88
나머지 라벨 교정 뒤                            88 -> 0
```

남는 88 은 값 기반 override 로 먼저 고치면 안 됩니다.
Nemotron 은 reshape merge 의 symbolic hard transport 가, Kimi-K3 는 KDA 층 layout hard
anchor 가 먼저 필요합니다.
