# 최종 승인 요청 — 계획서와 새 실측

지난 조건부 승인의 다섯 가지 교정을 전부 반영해 계획서를 다시 썼습니다.

**읽어 주실 것: `develop/PLAN_undecided_first_class.md`**

이 문서는 (1) 무엇을 어떻게 고쳤는지와 (2) 그 사이에 나온 **새 실측** 두 가지만 적습니다.
**100% 동의하시는 부분만 실행하겠습니다.** 한 줄이라도 미심쩍으면 그 절을 짚어 주십시오.

---

## 1. 계획서에 반영한 다섯 교정

| # | 지적 | 반영 |
|---|---|---|
| 1 | hard evidence 를 "이름을 확정하는 anchor" 와 "확정 이름을 운반하는 관계" 로 분리 | **3종으로 갈랐습니다**: hard anchor / hard transport / hard role. `Cache.update` 의 key/value 는 **hard role 일 뿐 축 이름을 확정하지 못한다**고 적었습니다 -- "key tensor 다" 와 "그 tensor 의 축 1 이 `n_kv` 다" 사이에 layout 근거가 하나 더 필요하다는 것. `nn.Linear` 선언 폭은 **constructor 인자가 어느 config 필드인지 확인돼야** hard, `split_with_sizes` 는 **입력 항의 symbolic decomposition 이 확정돼야** hard transport 로 적었습니다. `scoped_symbol` 은 soft. |
| 2 | 크기 1 · conv 캐시 길이를 무조건 `literal` 로 보지 말 것 | **판정 기준 자체를 바꿨습니다**: `literal = 값이 작다` 가 아니라 `provenance 가 "익명 구조 상수로 생성됐다"고 증명한다`. `B=1`, MQA 의 `n_kv=1`, 길이 1 인 실제 축을 반례로 명시했고, conv 캐시 길이는 기본 literal 이 아니라고 적었습니다. hard literal 7종을 열거했습니다. |
| 3 | transform path 를 고정 N단계가 아니라 의미 경계까지 정규화 | **보존/접기 표**로 정리했습니다. 연속 transpose·permute 는 net permutation 하나로 합성. phase·layer 번호는 **무조건 빼는 것이 아니라 normalized path 가 같을 때만** 접는다고 적었습니다(decode cache 경로 때문). |
| 4 | raw 축 수 대신 member-set digest 로 답을 고정 | **신원 3종**(`semantic_fingerprint` / `class_fingerprint` / `member_set_digest`)과 변화별 처리(자동적용 / 경고 / `needs_rebase` / `stale`)를 표로 적었습니다. `members.axes` 는 hard gate 가 아니라 진단값. |
| 5 | 질문·답 적용기를 만든 뒤 canonical 산출물 전환 | **3단계를 3a/3b 로 쪼갰습니다.** 10단계 순서를 주신 그대로 옮겼고, "미결을 안전하게 해소할 경로가 완성되기 전에 canonical 표를 교체하지 않는다" 를 명시했습니다. |

그 밖에 반영한 것: UF 세 모드(`legacy`/`provenance`/`migration`)와 명시적 `mode=` 인자,
partition 기준 비교, `legacy_value_edges_emitted` 로 세기, virtual semantic port 를
**실제 graph 노드**(`ActualPort | SemanticPort | ExternalPort`)로, hard-anchor ledger,
게이트 9종 추가, 첫 발행의 미결 전파를 `직접 tie site + 같은 provenance class` 로 제한,
`split_required` 의 discriminator 순서, counterexamples 형식.

---

## 2. 새 실측 — 지적하신 hybrid UF 를 47개 모델에 돌린 결과

지적하신 대로 **지금 코드는 계보 간선을 놓은 뒤 같은 행에 값 기반 간선을 또 놓습니다.**
그 상태로 47개를 재트레이스했습니다. 최종 상태가 아니라 **0-A 작업의 기준선**으로 봅니다.

| 지표 | 계보 전 | 계보 후(hybrid) |
|---|---:|---:|
| **expand 위반** | 314 | **0** |
| **전치 위반** | 277 | **120** |
| **reshape_incons** | 474 | **352** |
| 이름생성 위반 | 351 | 387 |
| **등가류 충돌** | **0** | **1,184** |
| **미발화 override** | **0** | **217** |

**oracle 314건이 실제로 0 이 됐습니다** -- shadow 예측대로이고, Granite 216 / Nemotron-Super 80
/ Zamba2 18 이 전부 사라졌습니다. 전치도 Kimi-K3 261 -> 120 으로 내려갔습니다.

충돌 1,184 의 분포:

```
moonshotai__Kimi-Linear-48B-A3B-Instruct   680
moonshotai__Kimi-K3                        354
Qwen__Qwen3-Next-80B-A3B-Instruct          144
bzantium__tiny-deepseek-v3                   6
```

미발화 217 은 27개 모델에 흩어져 있습니다(Zamba2 49, V4-Pro 21, Nemotron-Super 14 …).

### 여쭙고 싶은 것

**질문 A.** 이 1,184 와 217 이 **hybrid 때문이라고 보는 저희 해석이 맞습니까?**
0-A(세 모드 분리)를 하면 내려갈 것으로 봅니다. 아니면 계보 규칙 자체에 결함이 있다는
신호일 수도 있어서, 어느 쪽으로 봐야 할지 판단이 안 섭니다.

특히 **Kimi-Linear 680** 이 걸립니다. 이 모델은 직전까지 여섯 지표가 전부 0 이었고
`n_h*d_v` 교정도 정상 반영된 상태였습니다. 계보를 켜자마자 최다 충돌 모델이 됐습니다.

**질문 B.** 미발화 217 은 **규칙이 판정을 대체한 것**(전에도 `split_with_sizes` 교정 39건이
그랬습니다)일 수도, **계보가 앵커 자리를 옮겨 놓친 것**일 수도 있습니다. 둘을 가르는
기계적인 방법이 있습니까? 지금은 자리마다 "이미 목표 이름인가"를 확인하는데, 217건이면
그 확인 자체가 작업이 됩니다.

**질문 C.** 이 상태를 **커밋해 두고** 0-A 로 가는 것이 맞습니까, 아니면 hybrid 는 되돌리고
0-A 를 먼저 넣은 뒤 한 번만 재트레이스하는 것이 낫습니까? 재트레이스가 40분 걸려서
횟수를 줄이고 싶지만, 되돌리면 이 기준선을 잃습니다.

---

## 3. 계획서에서 특히 봐 주셨으면 하는 곳

1. **0절의 hard evidence 3종 표** -- 각 항목의 "조건부" 서술이 지적하신 뜻과 맞습니까?
2. **`literal` 의 hard literal 7종** -- 빠지거나 잘못 들어간 것이 있습니까?
3. **7절 게이트 목록** -- 주신 것에 `question_group_inhomogeneous`,
   `answer_out_of_member_changes`, `stale_member_set` 을 더했는데 정의가 맞습니까?
4. **8절 순서** -- 주신 10단계를 그대로 옮겼는데, 지금 저희가 2번(0-A/0-B) 에 있다고 보는
   것이 맞습니까? 1번(기준선·oracle 고정)은 `develop/provenance_oracle.py` 로 이미
   끝냈다고 보고 있습니다.
