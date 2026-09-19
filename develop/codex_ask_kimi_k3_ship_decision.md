# Kimi-K3 출고 판정 요청 — round 3

요청일: 2026-09-19. 선행 검토: `develop/codex_answer_kimi_k3_batch_topology.md` (2026-09-18),
`develop/codex_four_models_round2_review_2026-09-18.md`.

**묻는 것은 하나다. 아래 잔여 차이를 `UNKNOWNS.md` 에 공개하는 조건으로 Kimi-K3 를
출고해도 되는가, 아니면 앞선 답변이 제시한 `contraction_lowering_verified`(ports DAG +
`scalar_args` 인덱스 대응) 를 먼저 구현해야 하는가.**

앞선 답변의 기준을 그대로 옮겨 적는다. 이 요청은 그 기준을 흔들려는 것이 아니라,
지금 상태가 그 기준의 어디에 있는지 판정받으려는 것이다.

> diff가 0이어야 한다는 기준은 버릴 수 있다. 미검증 diff를 독립 배치 PASS 하나로 승인하는
> 기준은 채택하면 안 된다.
> 출고 조건은 **(a) 기본 산술 검사와 누락 없는 독립 배치 검증, (b) 지원 lowering 패턴의
> 국소 동치 검증, (c) 남은 의미 변화와 미검증 구간의 해소** 로 권장한다.

---

## 0. 무엇이 바뀌었나 (round 2 지적 반영)

round 2 가 지적한 항목을 코드와 규칙에 반영했다. 확인해야 할 것은 "고쳤다"는 주장이 아니라
아래 파일의 실제 내용이다.

지적된 이름들은 **개별 교정(`rules/label_overrides.yaml`)이 아니라 resolver 의 가드**로
막았다. 값이 우연히 맞는 이름을 짓지 못하게 해서, 근거 없는 이름이 정수로 물러나게 했다.

| round 2 지적 | 반영 | 확인 위치 |
|---|---|---|
| `B*d_head-d_rope` (실제 30) | 정수로 물러남 | `src/symbolic_shape.py` 배치 접힘 가드 |
| `n_h_kda/2` (실제 48, 삼각 루프 i) | `heur_half` 가 **개수 심볼**(`n_*`, `E`, `k`, `B`)을 거부 | `src/symbolic_shape.py:705-714` |
| conv cache `1` (실제 3) | `heur_multiple` 하한 16 — `2*d_conv`(8)·`3*d_conv`(12) 불가 | `src/symbolic_shape.py:664-676` |
| `n_chunk` 값 매칭 이름 붙이기 | **단독 `n_chunk` 는 유도식으로 등록하지 않았다** — 정수 `5` 로 물러남. 복합 `n_h_kda*n_chunk`(=480, B>1 에서 `B*n_h_kda*n_chunk`)는 당신이 round 2 에서 옳다고 확인해 준 접힌 배치 축이라 유지했다 | `rules/derived_dims.yaml:735-747` |
| `cat([a,a])` 반례 (집합→다중집합) | `_shapes_of` 가 `Counter` 반환, `_tiling_ok` 로 범위 검증 | `develop/transition_diff.py:113-160, 202-246` |
| `weight_shape` 미검사 | 검사 추가 | `develop/check_batch_labels.py` |
| rank 불일치·평가 불가 식을 건너뜀 | `unver` 로 따로 세고 0 이 아니면 FAIL | 같음 |
| probe 가 반환한 phase 만 순회 | 발행 phase 기준 순회로 교체 | 같음 |
| model revision 미고정 | 후보의 `revision_resolved` 로 고정 | 같음 |

`heur_multiple` 에 하한(16)을 두어 작은 심볼의 배수(`2*d_conv`, `4*d_conv` 류)도
지어낸 이름으로 취급해 막았다 (`src/symbolic_shape.py`).

## 1. 통과한 증거

발행 B=3 / T=320. 독립 검증은 **발행 라벨을 다시 만들지 않고** B=4 실제 shape 과 대조한다.

```
독립 배치 검증 (develop/check_batch_labels.py, 발행 B=3 / 검증 B=4)
  decode : 검사한 축   203,626   짝 못 지은 op 0 / 미검증 0 / 어긋난 라벨 0
  prefill: 검사한 축 5,390,557   짝 못 지은 op 0 / 미검증 0 / 어긋난 라벨 0
  배치 차수 합 > 1 인 shape: 0
  PASS
게이트 (develop/verify_all.py): FAIL 0
산술적으로 거짓인 라벨: 0
원장 어긋남 (axis_ledger 발행 라벨 대조): 0
```

`대체값 15,088` 은 MoE even-split 대체 경로(896 expert 중 4 개만 추적)에서 나온
per-expert token count 다. 비례 관계(`lab * b_probe == cb * b_pri`)로만 통과시키고
`caveat` 컬럼에 표시했다. **동치 증명으로 세지 않았다.**

## 2. 이 검사가 증명하지 않는 것 (스스로 인정하는 한계)

- B=3 과 B=4 는 둘 다 B>1 lowering 경로다. 이 검사는 B=1 경로와의 계산 동치를 증명하지 않는다.
  (앞선 답변이 지적한 그대로다.)
- 고정 길이를 배치와 무관한 상수로 잘못 이름 붙여도, 그 길이가 고정이면 통과한다.
- 추적 대상은 FakeTensor/CPU reference KDA 와 MoE even-split 대체 경로다.
  실제 GPU Triton kernel 의 op 구성은 이 검증의 범위 밖이다.

## 3. 잔여 전환 diff (옛 B=1 판 → 새 B=3 판)

```
   같음                     4,047,411
   짝지은 op                   477,452
   semantic_topology_change   372,948      <- 쟁점
   batch_expected            31,745
   synthetic_dispatch_scaling    30,176
   fabrication_withdrawn      2,760
   singleton_fixed            1,339
   semantic_change              936        <- 아래에서 직접 검증
   layout_lowering_verified       207
   합성 디스패치 보존 검사: 통과 (조각 합 == 디스패치, 되모으기 일치, 최종 토큰 축 B*T)
```

### 3-1. `semantic_change` 936 — 직접 검증했다

전부 KDA 심볼이 MLA 심볼로 바뀐 것이다.

```
    120  n_h_kda    -> n_h       aten.view.default
     96  d_head_kda -> d_nope    aten.unsqueeze / permute / bmm
     96  d_head_kda -> d_v       aten.view.default
     72  n_h_kda    -> n_h       aten.transpose.int
     72  d_head_kda -> d_nope    aten.cat.default
     48  d_head_kda -> d_v       aten.split_with_sizes.default
```

원인은 **값 충돌**이다. 이 모델에서 세 폭이 전부 같은 값이다:

```
d_head_kda = 128      d_nope = 128      d_v = 128
n_h_kda    = 96       n_h    = 96
```

`layer_idx` 로 갈라 옛 판(`models/`, B=1)을 세어 봤다. K3 는 4 층마다 MLA+MoE 이고
나머지는 KDA 다.

```
옛 판 층별 심볼 등장 (full/prefill.csv, op 행 수)
layer  d_head_kda  n_h_kda  d_nope   d_v   n_h
    0        5950     8793       0     0     0     <- KDA 층: 맞다
    1        5950     8793       0     0     0
    2        5950     8793       0     0     0
    3          20       15      23     6    41     <- MLA 층인데 KDA 심볼이 섞여 있다
    4        5950     8793       0     0     0
    ...
    7          20       15      23     6    41
   11          20       15      23     6    41
```

MLA 층은 3,7,11,…,47 의 **24 개 전부**가 이 오염을 갖고 있었다. 그 층 안에서 KDA 심볼이
붙은 축은 prefill 1,872 + decode 1,848 = **3,720 축**이다.
(전환 diff 의 936 은 집계 단위가 달라 이 수와 직접 맞지 않는다. 같다고 주장하지 않는다.)

즉 **옛 판이 MLA 층의 128/96 을 KDA 이름으로 부르고 있었고, 새 판이 그것을 층 유형으로
갈라낸 것**이다. 회귀가 아니라 교정이라고 판단했다. 이 판단이 맞는지, 그리고 새 판이
붙인 `d_nope`/`d_v` 의 **경계**(어느 축이 nope 이고 어느 축이 v 인지)가 소스와 맞는지
확인해 달라 — 값이 둘 다 128 이라 우리 쪽 검사로는 갈 수 없다.

### 3-2. `semantic_topology_change` 372,948 — 미검증 잔여

이 숫자는 **축 수가 아니라 짝 못 지은 op 레코드 수**다 (`tot[kind] += len(la) + len(lb)`).
서로 다른 구간은 **두 종류뿐**이며, 전부 `self_attn` (KDA) 안이다.

```
   --- semantic_topology_change   (모듈 인스턴스 수)
          117  self_attn -> reshape 아닌 op: ['aten.permute.default']
           69  self_attn -> reshape 아닌 op: ['aten.bmm.default', 'aten.permute.default']
```

분류 코드는 `develop/transition_diff.py:202` 의 `classify_unmatched` 다.
`RESHAPE_OPS` 에 `permute`/`transpose` 를 **일부러 넣지 않았다**. shape 다중집합만으로는
`[2,2,3]` 의 `permute(1,0,2)` 와 항등을 구별할 수 없기 때문이다. 그래서 permute 가 하나라도
들어간 구간은 전부 이 범주로 떨어진다.

원인은 앞선 답변이 수치로 확인해 준 그 lowering 이라고 보고 있다:
`einsum('... c d, ... d -> ... c')` 가 B=1 일 때 batch 축을 `ro` 로 분류해
`swap_lo_ro` 로 피연산자를 교환하고, B>1 일 때는 `lro` 로 넣어 교환하지 않는다.
따라서 같은 contraction 이 전치된 bmm 으로 내려가고, 앞뒤 permute 가 달라진다.

**하지만 우리 도구는 이 구간들이 실제로 그 lowering 인지 확인하지 않았다.**
확인한 것은 "구간에 permute/bmm 이 들어 있다"뿐이다. 구간별 동치는 미검증이다.

## 4. 묻는 것

1. **§3-1 의 `semantic_change` 936 판정** — `layer_idx` 분포로 MLA/KDA 를 가른 근거가
   충분한가, 아니면 소스에서 확인해야 할 것이 더 있는가.

2. **§3-2 의 372,948 을 공개 사항으로 두고 출고할 수 있는가.**
   앞선 답변은 "증명된 lowering 차이는 공개 사항으로 기록할 수 있다"고 했다. 그런데
   여기서 증명된 것은 **lowering 메커니즘 일반**(당신이 float64 로 검증한 것)이지
   **이 372,948 개 구간 각각**이 아니다. 이 간극이 출고를 막는가.

3. 막는다면, 최소로 무엇을 구현해야 충분한가. 특히 **117 개 permute-only 구간**은
   `scalar_args` 의 permutation dims 를 구간을 따라 합성하면 bmm 없이도 검증 가능해
   보인다 (당신이 제시한 단계 2 의 축소판). 이것만으로 다수를 해소할 수 있다고 보는가,
   아니면 DAG 경계 추적(단계 1)이 없으면 permutation 합성도 의미가 없는가.

4. 반대로 이 잔여를 그대로 두고 출고할 때, `UNKNOWNS.md` 에 **정확히 무엇을 적어야**
   독자가 오해하지 않는가. 지금 적으려는 문안은 이것이다. 고쳐 달라.

   > KDA(`self_attn`) 의 chunked einsum 은 batch 크기에 따라 서로 다른 bmm 으로 내려간다.
   > B=1 에서는 피연산자가 교환된 전치 형태이고, B>1 에서는 교환되지 않는다.
   > 이 판(B=3)의 라벨은 B>1 경로를 기술한다. 두 경로가 같은 contraction 이라는 것은
   > 외부 검토에서 float64 수치로 확인됐으나, 본 산출물의 해당 구간 372,948 건 각각에
   > 대해서는 구간별 동치를 확인하지 않았다. 이 구간의 축 이름은 독립 배치 검증(B=4)을
   > 통과했으나, 그 검증은 B>1 경로 안에서의 일관성만 보인다.

5. 다른 4 모델(gpt-oss-120b/20b, Llama-4-Maverick, DeepSeek-V4-Pro)은 같은 파이프라인으로
   이미 두 번 출고했고 독립 배치 검증을 모두 통과했다. K3 만 이 잔여를 갖는다.
   **K3 를 미뤄 두고 4 모델만 유지하는 편이 나은가, 아니면 공개 조건부로 5 모델을 함께
   내는 편이 나은가.** 산출물의 신뢰도 관점에서 판단해 달라.

## 5. 재현

```powershell
.venv\Scripts\python.exe develop/transition_diff.py moonshotai__Kimi-K3 --show 8
.venv\Scripts\python.exe develop/check_batch_labels.py develop/models/phase25-kimi-k3.yaml `
    --model-dir develop/out/moonshotai__Kimi-K3
.venv\Scripts\python.exe develop/verify_all.py
```

검토 대상 산출물: `develop/out/moonshotai__Kimi-K3/` 의 `prefill.csv`, `decode.csv`,
`prefill.jsonl`, `decode.jsonl`, `full/`, `audit_manifest.json`.
기존 규칙·후보 파일은 수정하지 말아 달라.
