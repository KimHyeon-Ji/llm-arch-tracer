# #4 설계 확인 — tensor-scoped virtual semantic port

지적하신 두 가지를 반영한 설계입니다. **승인 전에는 구현하지 않겠습니다.**

> 1. `SemanticPort` 를 기존 `op_id` 에 겹쳐 넣지 말고 명시적인 노드 타입으로
> 2. barrier 는 텐서 전체가 아니라 **역할이 바뀐 축 하나만** `derived` 로 끊을 것

제가 제안했던 `("sem", event_id, 0)` 이 왜 안 되는지도 확인했습니다. 지금 코드에
`pop, pslot = src` 로 **정확히 2개를 언팩**하는 곳과 `deps.append(src[0])` 이 있어,
3-튜플을 넣으면 `ValueError` 가 나거나 이벤트 id 가 op_id 로 섞입니다.
(`src/axis_classes.py:209`, `src/tracer.py:95`)

---

## 1. 왜 지금 필요한가 — 실측

`op_contract` sentinel 을 넣고 규칙을 채운 뒤 anti-union 이 0 -> 112 가 됐습니다.
**전부 알려진 값 충돌**입니다.

```
Kimi-K3          n_h vs n_h_kda      48
Nemotron-Super   d_state vs n_h_ssm  40
Zamba2           n_h vs n_kv         24
```

Zamba2 경로를 추적했습니다(`n_h = n_kv = 32`, `n_rep = 1`):

```
op827 view       [B, T, n_h, d_head]
op828 transpose  [B, n_h, T, d_head]
op836 slice      -> [B, n_h, T, d_head/2]     rotate_half
op838 neg
op839 concat     <- 여기서 n_kv 가 같은 클래스로 들어온다
```

지금 barrier 는 **전치 규칙 안에서만** `op_id` 시간 범위로 막습니다. 그래서
`concat`/`elementwise`/`slice` 경로로 그대로 샙니다. 지적하신 "축별로 끊어야 한다" 가
이 자리에서 확인됩니다.

---

## 2. NodeRef — discriminated union

축 슬롯의 첫 필드를 `op_id`(정수)에서 `NodeRef` 로 바꿉니다.

```python
NodeRef = tuple           # (kind, *ids) -- 직렬화 가능해야 하므로 tuple 로 둔다
("op",  op_id)                       # ActualPort  -- ATen op
("sem", phase, event_id)             # SemanticPort -- repeat_kv 등. **phase namespace 포함**
("ext", input_index)                 # ExternalPort -- graph 입력, parameter
```

축 슬롯:

```python
AxisSite = (NodeRef, field, slot, axis)      # field: "i" | "o" | "w"
```

`input_sources` 도 타입을 명시합니다. 지금은 `[op_id, slot] | None` 인데:

```json
{"kind":"op",  "op_id":242, "output":0}
{"kind":"sem", "phase":"prefill", "event_id":5, "output":0}
{"kind":"ext", "input":0}
```

**질문 2-1**: 축 슬롯의 첫 필드를 튜플로 바꾸면 기존 교정(`label_overrides`)의 selector 와
게이트가 전부 영향을 받습니다. 지금 그것들은 `(op_id, "o", 0, 2)` 형태를 씁니다.
어느 쪽이 낫습니까?

  (a) `AxisSite` 를 전부 `NodeRef` 로 바꾸고 호출부를 고친다 (파급 큼, 일관됨)
  (b) `ActualPort` 는 지금처럼 정수 `op_id` 를 유지하고, `SemanticPort` 만
      `("sem", phase, event_id)` 튜플로 둔다 (혼합이지만 파급 최소)

저희는 (b) 로 기울었습니다 -- 교정 selector 는 지적하신 대로 **`ActualPort` 만 대상**으로
하면 되고, semantic 노드는 계보와 게이트에서만 다루므로 기존 코드가 안 깨집니다.
다만 "정수 아니면 튜플" 이라는 혼합 타입이 나중에 사고를 부를지 걱정됩니다.

**질문 2-2**: `ext` 를 지금 도입해야 합니까? 지금은 입력이 없으면 `None` 을 넣는데,
지적하신 대로 factory / graph 입력 / parameter 입력을 구분하려면 `ext` 가 필요합니다.
다만 그것까지 한 번에 바꾸면 변경 폭이 커집니다. **`op`/`sem` 먼저 넣고 `ext` 는
다음 단계**로 미뤄도 됩니까? (`missing_port_records` 를 hard gate 로 올리는 시점에
어차피 필요해 보이는데, 그때가 맞습니까?)

---

## 3. 축별 barrier

`repeat_kv` 이벤트마다 **SemanticPort 노드 하나**를 만들고, 그 텐서의 축을 하나씩 잇습니다.
역할이 바뀌는 축만 `derived` 이고 나머지는 `same` 입니다.

```
upstream axis 0 ──same────────> sem axis 0
upstream axis 1 ──derived─────> sem axis 1     n_kv -> n_h     (UF union 안 함)
upstream axis 2 ──same────────> sem axis 2
upstream axis 3 ──same────────> sem axis 3

sem axes ──same──────────────> downstream input axes
```

구현:

```python
# semantic_events.py -- no-op 이어도 version 을 올린다
version[tensor_uid] += 1
logical_producer[(tensor_uid, version)] = ("sem", phase, event_id)

# tracer -- 입력 참조는 logical_producer 를 먼저 본다
src = logical_producer.get((uid, version[uid])) or ("op", producer_op, slot)

# axis_classes.lineage_edges -- SemanticPort 를 실제 노드로 둔다
for ax in range(rank):
    if ax == event["axis"]:
        directed.append((up_site, sem_site, "derived", event_id))   # UF 에 안 넣음
    else:
        edges.append((up_site, sem_site))                            # same
edges.extend((sem_site, down_site) for ...)                          # same
```

**질문 3-1**: `repeat_kv` 의 역할 전환 축이 **항상 축 1** 입니까? 지금 이벤트에
`"axis": 1` 을 하드코딩해 뒀습니다(`[B, n_kv, T, D]` 가정). 다른 레이아웃을 쓰는 구현이
있으면 틀립니다. 텐서 rank 와 `n_rep` 로부터 유도할 수 있습니까, 아니면 호출 시점의
shape 을 보고 "크기가 `n_rep` 배로 바뀐 축" 을 찾아야 합니까? **`n_rep == 1` 이면 그
방법도 못 씁니다** -- 그래서 하드코딩했는데, 더 나은 근거가 있습니까?

**질문 3-2**: `n_rep > 1` 인 경우에도 SemanticPort 를 만듭니까? 그때는 실제 ATen
`expand`/`reshape` 이 돌아 계보가 이미 이어집니다. 이벤트는 기록하되 **노드는 no-op 일
때만** 만드는 것이 맞습니까, 아니면 항상 만들어 일관성을 유지하는 것이 맞습니까?

**질문 3-3**: `Cache.update` 도 SemanticPort 로 만듭니까? 지금은 이벤트만 기록하고
계보에는 안 씁니다. 지적하신 "hard role 은 축 이름을 확정하지 못한다" 를 생각하면,
key/value 역할을 **노드 속성**으로 달아 두고 이름 결정 때 쓰는 것이 맞아 보이는데,
그러면 `repeat_kv` 와 달리 `derived` 로 끊을 축이 없습니다. 노드를 만들되 **모든 축을
`same`** 으로 잇고 역할만 기록하는 것이 맞습니까?

---

## 4. directed lineage 를 UF 와 분리해 보존

`derived`/`barrier` 는 union 하지 않고 **방향성 간선으로 따로** 남깁니다.

```python
# full/<phase>.lineage.jsonl
{"src": ["op", 827, "o", 0, 1], "dst": ["sem", "prefill", 5, "o", 0, 1],
 "relation": "derived", "kind": "repeat_kv", "axis": 1,
 "role_before": "n_kv", "role_after": "n_h", "event_id": 5}
```

이렇게 하면 상류 축을 `derived` 간선을 거꾸로 따라 조회할 수 있고, 어떤 barrier 가
끊었는지 감사됩니다.

**질문 4-1**: 이 파일을 **모든 `same` 간선까지** 담아야 합니까? 지금 계보는 1천만 간선
규모라(47개 모델 합계 10.7M) 전부 쓰면 사이드카가 매우 커집니다. `derived`/`barrier` 만
담고 `same` 은 규칙에서 재계산하는 것으로 충분합니까?

**질문 4-2**: 게이트 `virtual_port_orphans` 를 어떻게 정의합니까? 저희 안은
"SemanticPort 인데 하류 `same` 간선이 하나도 없는 것" 입니다. 상류는 `derived` 로만
이어지므로 상류가 없는 것은 정상입니다. 맞습니까?

---

## 5. 순서와 검증

```
1. semantic_events 에 logical version + SemanticPort id 부여 (기록만)
2. tracer 가 input_sources 에 타입을 명시해 기록
3. lineage_edges 가 SemanticPort 를 노드로 다루고 축별 barrier 적용
4. sentinel 에 negative barrier 검사 추가
     (repeat_kv 역할 전환 축이 barrier 전후로 같은 UF 이면 FAIL)
5. 47개 재트레이스 -> anti-union 112 가 0 이 되는지, permute 끊김 13,900 이 줄어드는지
```

**질문 5-1**: 4번의 negative barrier 검사를 `op_contract.py` 에 넣으려는데, 그러려면
`op_contract` 가 semantic 이벤트를 읽어야 합니다. 지금은 구체 shape 과 ATen 인자만 읽어
"계보 구현과 독립" 을 유지하고 있는데, semantic 이벤트를 읽는 것은 그 독립성을 깨지
않습니까? (이벤트는 계보 구현이 아니라 **트레이서가** 만든 것이므로 괜찮다고 봤습니다.)

**질문 5-2**: 재트레이스가 47개 40분입니다. 3번까지 하고 **sentinel 모델 3개**
(Zamba2 / Kimi-K3 / Nemotron-Super)만 먼저 돌려 확인한 뒤 전체를 도는 것이 맞습니까?

---

## 참고 — 지금까지의 상태

```
규칙       legacy                       provenance
expand     이어짐 66,602 / 끊김 24,233   이어짐 90,835 / 끊김 0
permute    이어짐 0 / 끊김 0             이어짐 409,775 / 끊김 13,900
split      이어짐 36 / 끊김 14,587       이어짐 14,623 / 끊김 0
잘못 이음  0                             0
anti-union 0                             112   <- #4 가 풀 대상
```

`legacy` 의 permute 이어짐 0 은 전치 간선이 아예 없기 때문입니다(예전에 넣었다 물렸습니다).
`positive_expand` 314 는 지금 `models/` 가 hybrid 산출물이라 양쪽 0 으로 나와
**아무것도 검증하지 못합니다** -- 지적하신 대로 불변 manifest + 구조 검사로 재정의하는 것을
#4 뒤에 하려 합니다. 순서가 맞습니까?
