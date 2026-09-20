# 확실하지 않은 것 -- moonshotai__Kimi-K3

이 파일은 **이 산출물에서 검증되지 않은 부분 전부**를 모은 것이다. 여기 없는 것은 규칙이 결정했고 게이트가 확인했다.

## 1. 축 이름 판정

축 자리 **5,573,253개** 중 확정 **2,052,242개 (36.8%)**, 미확정 **3,521,011개 (63.2%)**.

| 등급 | 자리 | 무엇을 믿어도 되나 |
|---|---:|---|
| `scope_inferred` | 3,015,339 | scope 정규식만이 후보를 갈랐다. 근거는 있으나 아무도 검증 안 했다. 구체 크기는 맞다 |
| `heuristic` | 53,141 | **산술로 지어낸 이름.** 값은 맞지만 이름이 틀릴 수 있다 -- 이 목록에서 가장 먼저 봐야 하는 등급이다 |
| `open_tie` | 4,584 | 후보 둘 이상이 같은 값이라 트레이스만으로 못 갈랐다. 구체 크기·FLOPs·바이트는 맞고 **이름만** 미정이다 |
| `unresolved` | 447,947 | 이름 붙일 근거가 없어 정수로 뒀다. 주장을 안 하므로 틀릴 것도 없다 |

### 아직 안 푼 질문

같은 `(등급, 후보, 현재 라벨)` 은 질문 하나다 -- 답 하나가 축 수천 개를 확정시킨다. 답은 `rules/axis_evidence.yaml` 에 인용과 함께 적는다.

질문 합계 **24개**.

**prefill** -- 질문 16개

| 축 수 | 등급 | 후보 | 현재 라벨 |
|---:|---|---|---|
| 1,163,340 | `scope_inferred` | n_h \| n_h_kda \| n_kv | `n_h_kda` |
| 952,062 | `scope_inferred` | d_chunk \| d_rope | `d_chunk` |
| 871,470 | `scope_inferred` | d_head_kda \| d_nope \| d_v | `d_head_kda` |
| 44,781 | `heuristic` | — | `d_chunk` |
| 6,279 | `heuristic` | — | `d_head_kda` |
| 4,278 | `scope_inferred` | — | `64` |
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
| 11,937 | `scope_inferred` | n_h \| n_h_kda \| n_kv | `n_h_kda` |
| 9,660 | `scope_inferred` | d_head_kda \| d_nope \| d_v | `d_head_kda` |
| 1,824 | `open_tie` | n_h \| n_h_kda \| n_kv | `n_h` |
| 1,380 | `heuristic` | — | `d_head_kda` |
| 480 | `open_tie` | d_head_kda \| d_nope \| d_v | `d_nope` |
| 276 | `scope_inferred` | — | `n_h_kda*d_head_kda` |
| 192 | `scope_inferred` | d_chunk \| d_rope | `d_rope` |
| 24 | `open_tie` | d_head_kda \| d_nope \| d_v | `d_v` |

## 2. 자유 평가(③층) 상태

**`STALE`** -- ③ 자유 평가 이후 산출물이 바뀜 (기록 c49f5ffaa5a0f017 != 현재 fa285a226e2110c2, 검토일 2026-09-02)

즉 규칙이 못 잡는 종류의 오류는 **이 판에서 다시 확인되지 않았다.** 규칙 게이트가 통과했다는 것과는 별개의 이야기다.

## 3. 손 안 댄 검토 지적

아직 안 본 것 **3건**.

| 축 | 판정 | 내용 |
|---|---|---|
| `KDA 인트라-청크 순차 재귀 루프의 슬라이스 크기 -- 값 8/12/1` | no_name_exists |  |
| `KDA 인트라-청크 순차 재귀 루프의 슬라이스 크기 -- 값 48` | no_name_exists |  |
| `KDA naive_chunk_kda 청크 내부 루프 prefix 길이 (` | should_be_no_name |  |

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
* B. 이름 없이 남은 정수 전부 (266쌍)
* C. 모듈이 내는 출력 shape 전부 (144개 모듈 / 1889종)

## 5. 더 이상 안 맞는 소스 확인 기록

없다 -- 기록된 확인이 전부 지금 산출물의 축에 맞는다.

## 6. 표의 숫자가 관측값이 아닌 자리

해당 행에는 `caveat` 열이 차 있다. 그 열로 걸러 보면 된다.

* **`moe_infer_even_split`** -- 이 MoE 블록의 전문가 디스패치는 대체됐다: 레이어당 전문가 896개 중 4개만 트레이스했고, 전문가에 들어가는 토큰 수는 정렬된 토큰을 균등 분할한 **대체값**이다 -- 실제 값은 라우팅이 정하는 런타임 데이터라 한 번의 트레이스로는 알 수 없다. 라우터(gate)·scatter·argsort·가중합은 모델 자기 코드 그대로이고, 전문가 projection 의 op 구성과 폭도 충실하다. 토큰 축 크기만 신뢰하면 안 된다

## 7. 배치를 키우며 달라진 lowering

직전 발행본과 이 판을 대조하면 서명으로 짝이 안 지어지는 ATen op 레코드가 **276건** 있다. 같은 계산이 배치 크기에 따라 다른 연산으로 내려가기 때문이다 -- `einsum` 이 B=1 에서는 피연산자가 교환된 전치 `bmm` 으로, B>1 에서는 교환되지 않은 `bmm` 으로 내려간다.

짝이 안 지어진 구간은 `develop/lowering_proof.py` 가 **다시 실행해서** 대조한다. 모듈의 실제 바깥 경계를 ports 의 producer/consumer 로 찾고, 같은 경계 입력을 넣어 **모든 배치 조각에서** 같은 경계 출력이 나오는지 본다. 이 판의 불일치 **276건**.

**이것은 수치 시험이지 증명이 아니다.** 생성한 float64 입력 한 벌에 대해 결과가 일치했다는 뜻이고, 모든 입력에 대한 대수적 동치를 보인 것이 아니다. 산출물에서는 이 결과를 `lowering_replay_consistent` 라고 부른다.

| phase | 레코드 | 미증명 | template | 검증한 instance |
|---|---:|---:|---:|---|
| prefill | 0 | 0 | 0 |  |
| decode | 276 | 276 | 1 | 0/69 |

**이 증명이 말하지 않는 것:**

* 배치 축이 어디인지는 **발행 라벨을 가설로** 삼았다. 라벨이 틀렸으면 대조가 깨지므로 이 검사는 라벨의 검사이기도 하지만, 라벨을 독립적으로 세운 것은 아니다.
* 같은 op 이름 다중집합을 한 template 로 묶는다. DAG 간선과 `scalar_args` 까지 같은지는 지문에 들어 있지 않다 -- 다만 이번 실행은 모든 instance 를 재실행했으므로 표본 누락은 없다.
* 아무 op 도 소비하지 않는 중간 값은 경계에서 뺐다. 관측할 수 없는 값이라 대조 대상이 아니다.
* 트레이스 기록에 dtype 이 없어(`torch.bool` 이 직렬화되지 않는다), 마스크·색인 자리는 소비 지점에서 맞추고 팩토리 op 의 부동소수점 결과는 float64 로 통일했다. 양쪽에 같은 규칙을 쓰므로 대조는 성립하지만, 원본의 dtype 자체를 재현한 것은 아니다.
* 부동소수점 bitwise 일치를 요구하지 않는다(rtol=atol=1e-11).
* 추적 범위 밖(실제 GPU kernel 의 op 구성)은 들어 있지 않다.

### 배치 전환(B=1 -> B=3) 대조

같은 모델을 같은 규칙으로 B=1 과 B=3 두 번 잡아 대조했다. 서명으로 짝이 안 지어진 ATen op 레코드 **373,155건**, 불일치 **0건**. 근거 파일은 `full/batch_transition_proof.json`.

| phase | 레코드 | 불일치 | template | 검증한 instance |
|---|---:|---:|---:|---|
| prefill | 371,751 | 0 | 3 | 69/69, 24/24, 69/69 |
| decode | 1,404 | 0 | 2 | 69/69, 24/24 |
