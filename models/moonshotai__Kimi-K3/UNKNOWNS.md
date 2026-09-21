# 확실하지 않은 것 -- moonshotai__Kimi-K3

이 파일은 **이 산출물에서 검증되지 않은 부분 전부**를 모은 것이다. 여기 없는 것은 규칙이 결정했고 게이트가 확인했다.

## 1. 축 이름 판정

축 자리 **5,579,877개** 중 확정 **2,056,658개 (36.9%)**, 미확정 **3,523,219개 (63.1%)**.

| 등급 | 자리 | 무엇을 믿어도 되나 |
|---|---:|---|
| `scope_inferred` | 3,013,269 | scope 정규식만이 후보를 갈랐다. 근거는 있으나 아무도 검증 안 했다. 구체 크기는 맞다 |
| `heuristic` | 57,419 | **산술로 지어낸 이름.** 값은 맞지만 이름이 틀릴 수 있다 -- 이 목록에서 가장 먼저 봐야 하는 등급이다 |
| `open_tie` | 4,584 | 후보 둘 이상이 같은 값이라 트레이스만으로 못 갈랐다. 구체 크기·FLOPs·바이트는 맞고 **이름만** 미정이다 |
| `unresolved` | 447,947 | 이름 붙일 근거가 없어 정수로 뒀다. 주장을 안 하므로 틀릴 것도 없다 |

### 아직 안 푼 질문

같은 `(등급, 후보, 현재 라벨)` 은 질문 하나다 -- 답 하나가 축 수천 개를 확정시킨다. 답은 `rules/axis_evidence.yaml` 에 인용과 함께 적는다.

질문 합계 **23개**.

**prefill** -- 질문 15개

| 축 수 | 등급 | 후보 | 현재 라벨 |
|---:|---|---|---|
| 1,164,168 | `scope_inferred` | n_h \| n_h_kda \| n_kv | `n_h_kda` |
| 952,062 | `scope_inferred` | d_chunk \| d_rope | `d_chunk` |
| 871,746 | `scope_inferred` | d_head_kda \| d_nope \| d_v | `d_head_kda` |
| 49,059 | `heuristic` | — | `d_chunk` |
| 6,279 | `heuristic` | — | `d_head_kda` |
| 1,776 | `open_tie` | n_h \| n_h_kda \| n_kv | `n_h` |
| 701 | `heuristic` | — | `T` |
| 456 | `open_tie` | d_head_kda \| d_nope \| d_v | `d_nope` |
| 414 | `scope_inferred` | — | `4` |
| 414 | `scope_inferred` | — | `10` |
| 414 | `scope_inferred` | — | `32` |
| 414 | `scope_inferred` | — | `37` |
| 276 | `scope_inferred` | — | `n_h_kda*d_head_kda` |
| 192 | `scope_inferred` | d_chunk \| d_rope | `d_rope` |
| 24 | `open_tie` | d_head_kda \| d_nope \| d_v | `d_v` |

**decode** -- 질문 8개

| 축 수 | 등급 | 후보 | 현재 라벨 |
|---:|---|---|---|
| 12,765 | `scope_inferred` | n_h \| n_h_kda \| n_kv | `n_h_kda` |
| 9,936 | `scope_inferred` | d_head_kda \| d_nope \| d_v | `d_head_kda` |
| 1,824 | `open_tie` | n_h \| n_h_kda \| n_kv | `n_h` |
| 1,380 | `heuristic` | — | `d_head_kda` |
| 480 | `open_tie` | d_head_kda \| d_nope \| d_v | `d_nope` |
| 276 | `scope_inferred` | — | `n_h_kda*d_head_kda` |
| 192 | `scope_inferred` | d_chunk \| d_rope | `d_rope` |
| 24 | `open_tie` | d_head_kda \| d_nope \| d_v | `d_v` |

## 2. 자유 평가(③층) 상태

**`STALE`** -- ③ 자유 평가 이후 산출물이 바뀜 (기록 c49f5ffaa5a0f017 != 현재 20dd51d7817efb32, 검토일 2026-09-02)

즉 규칙이 못 잡는 종류의 오류는 **이 판에서 다시 확인되지 않았다.** 규칙 게이트가 통과했다는 것과는 별개의 이야기다.

## 3. 손 안 댄 검토 지적

아직 안 본 것 **0건**.

아래 **4건**은 **판정이 끝난 것**이다 -- 소스를 보고 "지금 라벨이 맞다" 또는 "이름이 없는 것이 맞다" 고 결론 낸 자리다. 미수정 결함이 아니다.

| 축 | 판정 | 상태 |
|---|---|---|
| `KDA S state 정사각 [B, n_h_kda, d_head_kda,` | current_label_correct | current |
| `KDA 자신의 head 개수 (96) -- `n_h vs n_kv` 값 ` | current_label_correct | current |
| `KDA 자신의 head_dim (128) -- `d_nope vs d_v` | current_label_correct | current |
| `MoE 캡 셔플이 접은 토큰 배치 폭 -- prefill 1280 / d` | no_name_exists | current |

### 알고 받아들인 한계

**1건.** 고쳐야 할 결함이 아니라 **알고 받아들인 것**이다. 판정별로 뜻이 다르니 `판정` 열을 함께 보라 -- `table_omits_computation` 은 원시 trace(`full/`)에는 있고 요약 표에서 빠진 계산이고, `different_lowering_verified` 는 같은 계산이 다른 ATen 분해로 기록된 것이라 빠진 계산이 아니다.

| 모듈 | 무엇을 받아들였나 | 판정 | 근거 |
|---|---|---|---|
| `model.layers.*.self_attn` | 배치 크기에 따라 einsum 이 다른 bmm 으로 내려간다 (서명 불일치 372,948) | different_lowering_verified | 옛 발행본은 B=1 로 잡혔다. torch 의 einsum 은 `sumproduct_pair` 에서 크기에 따라 축을 lro/lo/ro 로 나누는데, B=1 이면 batch 축이 ro 로 들어가 `swap_lo_ro` 가 피연산자를 교환하고 B>1 이면 lro 라 교환하지 않는다. 그래서 같은 contraction 이 서로 전치된 bmm 으로 내려가고 앞뒤 permute/view 도 달라져 서명이 안 맞는다. |

## 4. 의뢰서의 판단 필요 항목

판단 필요 **6건**. 값 충돌이나 관례로 고른 자리라 규칙이 결정하지 못했다 -- 위 1절의 접힌 질문과 같은 종류다. 전문은 `review_request.md` 에 있다.

* 1. 이 config 필드가 정말 이 뜻인가
* 2. 이 정사각 축이 정말 같은 이름 두 번인가
* 6. 값이 겹쳐 **임의로** 고른 축
* 0. 규칙이 끝내지 못한 축 — **여기부터 답한다**
* A. 붙은 이름 전부 (40종)
* B. 이름 없이 남은 정수 전부 (265쌍)
* C. 모듈이 내는 출력 shape 전부 (144개 모듈 / 1889종)

## 5. 더 이상 안 맞는 소스 확인 기록

없다 -- 기록된 확인이 전부 지금 산출물의 축에 맞는다.

## 6. 표의 숫자가 관측값이 아닌 자리

해당 행에는 `caveat` 열이 차 있다. 그 열로 걸러 보면 된다.

* **`moe_infer_even_split`** -- 이 MoE 블록의 전문가 디스패치는 대체됐다: 레이어당 전문가 896개 중 4개만 트레이스했고, 전문가에 들어가는 토큰 수는 정렬된 토큰을 균등 분할한 **대체값**이다 -- 실제 값은 라우팅이 정하는 런타임 데이터라 한 번의 트레이스로는 알 수 없다. 라우터(gate)·scatter·argsort·가중합은 모델 자기 코드 그대로이고, 전문가 projection 의 op 구성과 폭도 충실하다. 토큰 축 크기만 신뢰하면 안 된다

## 7. 배치를 키우며 달라진 lowering

**직전 발행본과 이 판 사이에는 계산이 달라진 구간이 없다.** 라벨만 바뀐 전환이라 다시 실행해 대조할 대상이 없었다(레코드 0건). **이 전환에 대해 구간 동치는 주장하지 않는다** -- 대조를 해서 통과한 것이 아니라, 대조할 것이 없었다.


### 배치 전환(B=1 -> B>1) 대조 -- **이 판에는 없다**

접힌 배치 축(`B`, `B*X`)을 B=1 트레이스와 맞대어 확인한 기록이 이 판에는 **실려 있지 않다.** 그 대조는 B=1 로 한 번 더 잡아야 만들 수 있는데, 그 뒤로 추적된 계산 자체가 바뀌어(KDA forget gate, Q/K 정규화) 옛 기록은 지금 발행하는 계산을 설명하지 못한다. 그래서 **옮겨 싣지 않고 없다고 적는다.**

이 판에서 배치 라벨을 받치는 근거는 **독립 배치 검증** 하나다 -- 발행한 B 라벨을 다른 배치 크기로 실제로 잡은 shape 과 대조해 어긋나는 라벨이 없음을 본다. 두 트레이스를 재실행해 값까지 맞춰 보는 대조는 아니다.
