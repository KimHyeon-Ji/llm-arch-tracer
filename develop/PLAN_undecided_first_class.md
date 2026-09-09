# 실행 계획 — undecided-first-class (외부 검토 조건부 승인, 2026-09-09)

**근거 없이 고른 이름보다 명시적인 미결이 낫다.** 다만 tie 가 난 순간 지우는 것이 아니라,
불확실성을 축의 1급 상태로 보존한 뒤 provenance 와 hard evidence 를 **전부 적용한 마지막
단계에서** 확정 여부를 정한다.

이 문서는 외부 검토의 조건부 승인을 반영한 최종 계획이다. 승인 전 초안에서 **다섯 군데가
바뀌었다** -- 각 절에 `[교정]` 으로 표시했다.

---

## 0. 용어 — 축의 상태와 근거

### 상태 6종 (합치지 않는다)

| 상태 | 뜻 |
|---|---|
| `known` | hard proof 가 있는 확정 |
| `ambiguous` | 후보 둘 이상, hard proof 없음 -> **질문 대상** |
| `heuristic` | 후보가 하나뿐이지만 hard proof 없음 |
| `unknown` | 후보 자체가 없음 |
| `literal` | 익명 구조 상수 -- **이름이 없는 것이 정답** |
| `conflict` | 서로 다른 hard proof 충돌 -> 게이트 FAIL |

`literal` 과 `unknown` 을 갈라야 루프 카운터가 질문으로 다시 올라오지 않는다.

**[교정] hard 가 없고 후보가 하나뿐이면 `known` 이 아니라 `heuristic` 이다.**

### 근거 3종 [교정 — 초안은 이걸 한 덩어리로 봤다]

근거는 자료의 **종류**가 아니라 `(site, symbol)` 주장과 그 **proof chain** 이다.

| 종류 | 하는 일 | 예 |
|---|---|---|
| **hard anchor** | 축 이름을 **직접 확정** | 입력 contract 의 `B`/`T`; parameter 축이 어느 config 필드에서 왔는지 소스로 확인된 것; source verdict / confirmed / override; op 인자가 이미 확정된 symbol 의 사용인 것 |
| **hard transport** | 이름을 만들진 않지만 **확정 이름을 전달** | 정확한 producer/output-slot -> consumer/input-slot; transpose/permute 의 실제 dim 인자; identity/view 의 증명된 1:1 대응; split 의 비분할 축; concat 의 증명된 비-concat 축 |
| **hard role** | **tensor 역할만** 확정 | `Cache.update` 의 key/value 인자; q/k/v parameter identity; split output slot 의 역할 |

**hard role 은 축 이름을 확정하지 못한다.** "이것이 key tensor다" 와 "그 tensor 의 축 1 이
`n_kv` 다" 사이에는 layout 근거가 하나 더 필요하다.

조건부인 것들:

- `nn.Linear` 선언 폭 -> **constructor 인자가 어느 config 필드인지 확인돼야** hard.
  숫자 shape 만 알면 hard 가 아니다.
- `split_with_sizes` 순서 -> **입력 항의 symbolic decomposition 이 확정돼야** hard transport.
  `[128, 128]` 숫자만으로는 이름을 못 정한다.
- `scoped_symbol` (모듈 스코프에서 그 값을 가진 심볼) -> **soft**. 값 매칭이 근거다.
  지금 라벨의 43.6% 가 이것이다.

### `literal` 판정 [교정 — 초안이 크게 틀렸다]

```
literal = 값이 작다          (X)
literal = provenance 가 "익명 구조 상수로 생성됐다"고 증명한다   (O)
```

**크기 1 만으로 literal 이 아니다** -- decode 의 `B=1`, MQA 의 `n_kv=1`, 길이 1 인 실제
sequence/state 축이 있다. **conv 캐시 창 길이도 기본적으로 literal 이 아니다** --
`d_conv`/kernel size/state width 같은 실제 아키텍처 파라미터일 수 있다.

**[교정] 아래 7종은 "자동 literal" 이 아니라 proof kind 후보다.** 각각 조건이 붙는다.
`unsqueeze` 삽입 singleton 과 reduction `keepdim` singleton 만 직접 hard literal 로 본다.

hard literal proof kind 후보:

- `unsqueeze` 가 새로 삽입한 singleton
- reduction 의 `keepdim=True` 가 만든 singleton
- scalar lift / shape bookkeeping 축
- 소스와 계보로 확인된 loop/rung 축
- 빈 partition 이 만든 크기 0 축 -- **소스상 placeholder/empty sentinel 임이 증명된 경우만**
- 고정된 tuple/list arity -- **tensor axis 가 실제로 그 arity 를 표현하는 경우만**
- index helper tensor -- **`k`/`E`/`T` 같은 아키텍처 축이 아님이 확인된 경우만**
- scalar lift -- rank-0 에는 축이 없다. **후속 shape-bookkeeping 이 만든 축**을 뜻한다

---

## 1. 0단계 — 지금 코드의 결함 두 개

### 0-A. UF 를 세 모드로 [교정 — 초안은 두 갈래였다]

```python
build(rows, concrete, mode="provenance", semantic_events=...)   # 전역 플래그 금지
```

| mode | 내용 |
|---|---|
| `legacy` | 기존 값 기반만 |
| `provenance` | 정확한 port + typed op edge 만, **fallback 없음** |
| `migration` | provenance + **구 트레이스에만** legacy fallback |

지금 코드는 provenance 간선을 놓은 뒤 **같은 행에** 값 기반을 다시 놓는다. 이 hybrid UF 는
**비교용으로만** 두고 최종 provenance UF 로 간주하지 않는다.

비교는 class root ID 가 아니라 **partition** 으로 한다:

```
legacy 만 합친 축 쌍 / provenance 만 합친 축 쌍 / provenance 에서 갈라진 legacy class
/ 새 giant component / 각 차이가 어떤 edge rule 에서 왔는가
```

세는 것도 바꾼다: `legacy_fallback_ops` -> **`legacy_value_edges_emitted`**.
입력이 없는 factory, 외부 입력, parameter 입력은 fallback 이 아니다.

### 0-B. virtual semantic port 를 **실제 graph 노드로** [교정 — 초안은 간선을 생략만 했다]

```
ATen producer port
    └─ derived/barrier
       SemanticPort(repeat_kv_event, output=0, role=n_h)
            └─ same
               downstream ATen input port
```

```python
logical_producer[(tensor_uid, version)] = SemanticPort(event_id, 0)
```

version 이 다를 때 `same` 을 생략하기만 하면 부족하다 -- 하류 계보가 고아가 되고,
`n_kv -> n_h` 역할 전환 근거가 graph 에 안 남고, 어떤 barrier 가 끊었는지 감사할 수 없다.

`AxisSite` 형식은 유지하되 노드를 `ActualPort | SemanticPort | ExternalPort` 로 구분한다.
현재의 **op_id 시간 범위 barrier 는 폐기**한다(무관한 전치까지 막는다).

---

## 2. hard-anchor ledger [교정 — 초안은 override/confirmed 만 봤다]

`hard_known -> undecided 0` 을 재려면 "지금 known 인 축"의 기준이 필요한데, 렌더된 문자열은
기준이 못 된다(근거 없이 붙은 이름도 known 처럼 보인다).

별도 ledger 를 만든다:

```json
{"site": [812,"o",0,2], "symbol": "d_head",
 "proof_id": "parameter-config:model.layers.*.q_proj:out_features",
 "proof_kind": "source_config_binding"}
```

담는 것: override / confirmed / 339 source verdict / parameter·config constructor 연결 /
입력 contract 의 B·T / Cache·split 의 확정 역할에서 layout contract 까지 이어진 축 /
exact op mapping 으로 전달된 hard anchor.

**[교정] oracle 통과 자체는 hard anchor 가 아니다.** oracle 은 검증 장치다. ledger 에 넣는
것은 314건 **아래에 있는 source-confirmed anchor 와 그 transport proof** 이고,
"expand 314 -> 0" 은 **acceptance test 로만** 둔다.

비교는 **`(class fingerprint, symbol, proof)` 가 새 상태에서 살아남는가**로 한다.

---

## 3. 신원 3종 [교정 — 초안은 raw 축 수로 답을 고정하려 했다]

```
semantic_fingerprint   같은 질문인지. layer/phase 반복에 안정적
class_fingerprint      개별 provenance class 의 구조적 신원
member_set_digest      질문이 덮는 sorted class_fingerprint 목록의 hash
```

**답은 `member_set_digest` 에 묶는다.**

| 변화 | 처리 |
|---|---|
| digest 동일 | 자동 적용 |
| class 집합 동일, raw 축 수만 변경 | 경고 + selector cardinality 검증 |
| member 추가/삭제 | `needs_rebase` |
| semantic fingerprint 변경 | `stale` |

`members.axes` 정확 일치는 hard gate 가 아니라 **진단값**이다. 정확히 맞아야 하는 것은
member class 집합 / phase·layer coverage / selector 가 잡는 class 수 / 대상 밖 class 수 0.

정당한 변경이면 답을 폐기하지 말고 rebase 후보를 만들되, **새 member 에 기존 답을 조용히
확장하지 않는다.**

---

## 4. transform path 정규화 [교정 — 초안은 "몇 단계"를 물었다]

고정 N 단계가 아니라 **가장 가까운 semantic origin 또는 role-changing boundary 까지**
거슬러 올라가고 중간 plumbing 을 정규화한다.

| 보존 | 접기 |
|---|---|
| parameter / graph input / semantic virtual origin | clone, detach, contiguous |
| split output slot | 순수 identity elementwise |
| concat input slot 과 dim | 축 관계가 같은 연속 view |
| replication / no-op role transition | 연속 transpose·permute -> **net permutation 하나** |
| cache key/value | |
| reshape 의 merge/split 구조 | |
| transpose 의 합성된 최종 permutation | |
| select/slice 의 dim 과 의미 있는 index | |
| matmul 에서 그 축의 역할 | |
| branch/layer type 전환 | |

```
원본        split/out2 -> clone -> view -> transpose(1,2) -> contiguous -> view
fingerprint partition(out2) -> reshape_map(...) -> permute(net=...)
```

phase 와 layer 번호는 **무조건 빼는 것이 아니라 normalized path 가 같을 때만** 접는다 --
decode cache 경로는 prefill 과 다를 수 있다.

---

## 5. 발행 형식

### canonical CSV/JSONL — 문법 불변, 상태 코드만 최소로

```
output_shape        ["B","T","128","d_head_lin_v"]
output_axis_status  ["K","K","A","K"]
```

`K` known · `A` ambiguous · `H` heuristic · `U` unknown · `L` literal · `C` conflict

후보·evidence·class fingerprint·question ID 는 **사이드카에만**:
`full/<phase>.axis_status.jsonl`

`?a|b` 를 canonical shape 문자열에 넣지 않는다 -- 기존 label parser, 수식 parser,
reshape 검사, CSV 소비자가 전부 새 문법을 알아야 한다.

### 사람용

`review.md` 에서 `128?{d_head | n_h}` (구체 크기를 보존해 판정하기 쉽다).
`model_summary.md` 에 상태별 집계.

---

## 6. 질문과 답

### 접는 기준

`semantic_fingerprint` 동일성. **축 수는 기준이 아니다.**

### counterexamples (2~5개)

```yaml
counterexamples:
  - site: ...
    excluded_because: "split_output: out1 != out2"
    known_symbol: d_head_lin_k
    proof_id: ...
```

우선순위: 같은 module/value/candidates 인데 producer role 이 다른 것 / fingerprint 가 한
필드만 다른 가장 가까운 축 / 다른 split output slot / 다른 layer_type·phase 경로 /
과거 false-merge sentinel / 이미 hard-known 이며 답과 달라야 하는 축.

**적용 범위가 아니라 과도한 일반화를 막는 negative evidence 다.**

### 답

LLM 은 `answer.{status, symbol, rationale, citations}` 만 채운다.
정규식·shape selector·`nth`·fingerprint 는 생성기가 고정한다.

`status`: `resolved` / `resolved_unknown` / `split_required` / `invalid_question`

검증 사슬: symbol ∈ candidates · source file hash 동일 · 인용이 해당 함수 안 ·
member class 집합 불변 · phase/layer coverage 일치 · 대상 밖 변경 0 ·
class 안 반대 hard anchor 0 · dry-run 후 전체 게이트 + 339/314 oracle 통과 ·
다음 재생성에서 미발화하면 stale.

`resolved_unknown` 은 `rules/axis_decisions.yaml` 에 fingerprint + candidates +
source hash 로 저장하고 **그것들이 그대로인 동안만** 질문을 억제한다.
이유: `source_underdetermined` / `semantic_alias` / `trace_observability_gap` /
`not_architectural_axis`.

### `split_required` [교정 — 초안은 처리 방법을 안 정했다]

child question 을 **자동 제안하되 답을 자동 적용하지 않는다.** 분할 순서:

```
producer output slot -> semantic role -> layer type -> phase 별 normalized path
-> axis position/rank -> transform event 차이 -> module class/path -> source callsite
```

각 child 가 provenance-homogeneous 해지고 selector 가 유일하면 새 질문으로 자동 생성.
어떤 discriminator 로도 안 갈리면 `resolved_unknown / trace_observability_gap` 이거나
계측을 추가한다. **사람이 broad regex 로 억지로 나누는 경로는 만들지 않는다.**

---

## 7. 게이트

```
avoidable_undecided            0   class 안에 hard anchor 가 있는데 미결
unsupported_known              0   hard proof 없이 known 으로 발행
hard_conflict                  0   한 class 에 서로 다른 hard symbol
state_nonuniform               0   같은 class 안에서 최종 상태가 다름
undecided_without_path         0   seed 도 derived 경로도 없는 미결
unreviewed_giant_component_growth  0   [교정] "새 component 0" 은 너무 강하다 --
                                   정확한 provenance 가 기존 단절을 합법적으로 이을 수 있다.
                                   새로 생기면 중단하고 edge proof 와 hard symbol 일관성을
                                   검토한 뒤 승인 목록에 넣는다.
provenance_edge_contract_fail  0
virtual_port_orphans           0
semantic_event_unconsumed      0
question_group_inhomogeneous   0
answer_out_of_member_changes   0
stale_member_set               0
missing_port_records           0   최종 발행 시에는 hard gate. 모든 tensor 입력은
                                   ActualPort / SemanticPort / ExternalPort 중 하나로
                                   설명돼야 한다.
legacy_value_edges_emitted     0 이 되면 값 기반 간선 제거
source-confirmed regression    0   (339 / 314 oracle)
```

**`unexplained_undecided == 0` 은 필요조건일 뿐 충분조건이 아니다.** 너무 넓게 기록된
tie seed 가 class 전체를 미결로 만들면 "이 tie 에서 왔다"는 설명은 붙지만 분류가 틀린 것이다.

중단 신호: 2배 초과 -> rule 별 증폭 경로 조사 / 5배 초과 -> rollout 중단 /
**giant component 가 새로 생기면 배수와 무관하게 중단.**

**첫 발행에서는 미결 전파 범위를 `직접 tie site + 같은 provenance class` 로 제한한다.**
`derived` edge 를 통한 2차 확산은 별도 단계에서 연다.

---

## 7.5 실측 — 충돌 1,184 의 정체 [2026-09-09]

계보를 켠 뒤 등가류 충돌이 0 -> 1,184 가 됐다. **"hybrid 탓이니 모드를 분리하면 내려간다"는
저희 가정은 틀렸다.** `develop/mode_diff.py` 로 네 모드를 만들어 분해했다.

```
Kimi-Linear-48B      legacy 0 · provenance 680 · migration 680 · hybrid 680
이름쌍               전부 ['1', 'B']
경로                 view -> permute -> view
```

**`provenance` 단독에서도 전부 난다.** 계보가 새 결함을 만든 것이 아니라, **원래 있던 라벨
불일치를 처음 드러낸 것**이다 -- 정확한 permutation 으로 보면 같은 축인데 한쪽이 `1`,
다른 쪽이 `B` 였다. 값 기반 등가류는 그 둘을 **아예 안 이어서** 모순이 안 보였다.

decode 에서 `B == 1` 이므로 값으로는 영원히 못 가른다. **어느 쪽이 맞는지는 hard-anchor
ledger 가 정해야 한다.** 이것이 이 계획의 첫 실전 대상이다.

### 발견한 실행 순서 결함

`run.py` 가 `write_outputs()` 를 부른 **뒤에** semantic 사이드카를 쓰는데, `write_outputs()`
는 그 파일을 디스크에서 읽고 있었다. 즉 `repeat_kv` 경계는 **첫 실행에서는 비어 있었고 그
뒤로는 이전 실행의 것**이었다 -- 한 번도 제대로 적용된 적이 없다. 이벤트를 인자로 직접
넘기도록 고쳤다(외부 검토가 실행 순서로 짚었다).

**따라서 직전 hybrid 측정치(전치 277 -> 120 등)는 barrier 없이 나온 수치다.**

---

## 8. 순서 [교정 — 초안은 발행이 너무 빠르고 재트레이스가 너무 늦었다]

```
 1. 기준선 + oracle 고정
      1-A legacy 기준선 고정                                       [완료]
      1-B oracle 명세 고정                                         [대체로 완료]
      1-C **provenance mode oracle 실행**                          [미완료]
      -- `provenance_oracle.py` 가 raw trace 를 읽고 port sidecar 를 안 붙여서 지금 freeze 는
         사실상 legacy UF 기준이다. anti-union 도 "한 class 에 두 렌더 이름" 을 보는데,
         class 단위로 이름을 통일한 뒤에는 잘못 합쳐져도 이름 하나만 남아 통과한다.
         0-A 에서 oracle 이 `mode=` 를 받게 하고, anti-union 은 이름 쌍이 아니라 **구조적으로
         고정된 두 site/class fingerprint 의 root 가 다른지**를 검사해야 한다.
 2. 0-A 세 UF mode + 0-B virtual semantic port 구현
 3. sentinel 모델 재트레이스 -> shadow partition diff
 4. 47개 모델을 **candidate 디렉터리**로 재트레이스
 5. 축 상태 sidecar 생성 -- 표는 아직 기존 상태
 6. fleet 상태 집계 -> **판단 지점**
 7. 질문 생성기 + 답 적용기 + stale/rebase 검증 완성
 8. open 질문 생성, 필요한 판정 적용
 9. 전체 게이트 재실행
10. schema version 올리고 CSV/JSONL/review 를 canonical 발행
```

핵심은 **3단계를 둘로 나눈 것**이다:

```
3a 내부 candidate sidecar 발행  ->  질문/답 파이프라인 검증  ->  3b canonical 전환
```

답을 다 받아야 발행할 수 있는 것은 아니다(`ambiguous` 가 정식 상태이므로 open 질문이
남은 채로도 발행할 수 있다). 다만 **미결을 안전하게 해소할 질문·답 경로가 완성되기 전에
기존 canonical 표를 교체하지 않는다.**

모든 미결이 설명 가능하더라도 canonical 발행 여부는 별도 판단이다. 주요 operator 표의
유용성이 급락하면 source sampling 을 먼저 한다.
