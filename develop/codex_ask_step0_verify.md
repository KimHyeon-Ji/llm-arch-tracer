# 0단계 결과 확인 — 구현이 의도대로 됐는지 봐 주세요

승인 범위 6개 중 5개를 구현했습니다. **다음으로 넘어가기 전에 확인받고 싶습니다.**
이번 세션에 제가 같은 종류의 실수를 여러 번 했습니다(아래 "제가 틀렸던 것" 참조).

---

## 1. 실행 순서 결함 — 지적하신 것이 맞았습니다

`run.py` 가 `write_outputs()` 를 부른 **뒤에** semantic 사이드카를 쓰는데
`write_outputs()` 가 그 파일을 디스크에서 읽고 있었습니다. 즉 `repeat_kv` 경계는
**한 번도 적용된 적이 없습니다** -- 첫 실행에서는 파일이 없고, 그 뒤로는 이전 실행의 것을
읽었습니다.

이벤트를 인자로 직접 넘기도록 고쳤습니다:

```python
build_table.write_outputs(model_dir, phase, rows, resolver, tags, param_axes,
                          semantic_events=sem_events.get(phase) or [])
```

**따라서 제가 앞서 보고한 hybrid 수치(전치 277 -> 120 등)는 barrier 없이 나온 것입니다.**
`develop/verify/hybrid_baseline.json` 에 그 사실을 명시해 보존했습니다.

## 2. 모드 분리

```python
build(rows, concrete, singleton_edge=True, noop_barriers=None, mode=None)
DEFAULT_MODE = "legacy"          # hybrid 를 canonical 로 쓰지 않는다
_MODES = ("legacy", "provenance", "migration", "hybrid")
```

```python
if mode != "legacy":
    for a, b in lineage_edges(...): uf.union(a, b)
if mode == "provenance":
    return uf                                  # 폴백 없음
for r in rows:
    if mode == "migration" and "input_sources" in r:
        continue                               # 계보가 설명한 행 -- 값 기반을 얹지 않는다
    ... 기존 값 기반 간선 ...
```

`legacy_fallback_ops` 를 둘로 나눴습니다:

```python
missing_port_records(rows)              # "input_sources" 키 자체가 없는 행
legacy_value_edges_emitted(rows, mode)  # 그 모드에서 값 기반을 실제로 낸 행
```

**질문 2-1**: `mode="provenance"` 에서 `singleton_edge` 인자는 지금 무시됩니다(값 기반
경로에서만 쓰이던 것). 이대로 두어도 됩니까, 아니면 provenance 경로에도 대응하는 스위치가
있어야 합니까?

**질문 2-2**: `migration` 의 판별을 `"input_sources" in r` 로 했는데, 한 트레이스 안에
포트가 있는 행과 없는 행이 섞이면 **부분적으로만 계보**가 됩니다. 그 상태를 허용하는 것이
맞습니까, 아니면 트레이스 단위로 전부 있거나 없어야 합니까?

## 3. mode-aware oracle 과 structural sentinel

이름 쌍 anti-union 이 약하다는 지적을 반영해, **op 정의가 "같은 축"이라 보증하는 자리가
실제로 같은 클래스인가**를 봅니다. sentinel 은 `expand` 의 비방송 축(입력 크기 == 출력 크기,
크기 != 1)으로 잡았습니다.

| mode | 이어짐 | **끊김** |
|---|---:|---:|
| `legacy` | 66,602 | **24,233** |
| `provenance` | 90,835 | **0** |

**질문 3-1**: sentinel 을 `expand` 비방송 축 하나로만 잡았습니다. 더 넣어야 할 구조적
보증이 있습니까? 후보로 생각한 것은 `transpose` 의 순열 대응과 `split` 의 비분할 축인데,
그 둘은 제 계보 규칙이 직접 만드는 간선이라 **자기 자신을 검사하는 셈**이 될까 걱정됩니다.
`expand` 도 계보 규칙이 만들긴 하지만 legacy 와 대조가 되므로 의미가 있다고 봤습니다.
이 판단이 맞습니까?

**질문 3-2**: `positive_expand`(oracle 314) 가 지금 legacy/provenance 양쪽에서 0 으로
나옵니다. 지금 `models/` 가 hybrid 산출물이라 이미 고쳐진 라벨을 읽기 때문입니다. 즉
**이 oracle 은 지금 상태에서 아무것도 검증하지 못합니다.** legacy 기준 산출물로 되돌려
다시 freeze 해야 합니까, 아니면 oracle 을 "라벨"이 아니라 "클래스 연결"로 재정의하는 것이
맞습니까(그러면 sentinel 과 같은 것이 됩니다)?

## 4. 미발화 교정 217건 자동 분류

주신 분류를 구현했습니다. 다만 **`superseded` 를 아무것도 내지 않았습니다** --
상태 층이 없어 hard proof 유무를 모르기 때문입니다.

```
196  already_named_soft   자리는 있고 이미 목표 이름, hard proof 미확인
  9  member_rebased       자리는 있는데 제3의 이름
  6  selector_stale       구조 앵커는 잡히는데 기존 selector 만 실패
  6  site_missing         구조 앵커 자체가 사라짐
```

구조적 앵커는 `module` 정규식 + `op_type` + `nth` + `field` + `shape_index` + `axis` +
`expect` 로 잡고, **`from` 과 `shape` 은 뺐습니다**(이름에 기대지 않기 위해).

**질문 4-1**: `expect`(구체 크기)를 구조적 앵커에 넣은 것이 맞습니까? 값이므로 뺄까 했는데,
빼면 같은 모듈의 다른 축까지 잡혀 분류가 무의미해집니다.

**질문 4-2**: `site_missing` 6건과 `member_rebased` 9건은 계보가 자리를 옮기거나 없앤
것이라 손으로 봐야 한다고 생각합니다. 자동으로 더 좁힐 방법이 있습니까?

## 5. **가장 중요 — `['1','B']` 680건을 어느 쪽으로 확정합니까**

`provenance` 단독에서 640개 클래스, 680자리가 충돌합니다. 전부 같은 형태입니다.
Kimi-Linear 의 대표 클래스 전체(4자리):

```
op242 view     o0[1]  렌더 ['n_h_kda','B','1','d_head_kda']  구체 [32,1,1,128]  args=[[32,1,1,128]]
op244 permute  i0[1]  렌더 ['n_h_kda','B','1','d_head_kda']  구체 [32,1,1,128]  args=[[2,0,3,1]]
op244 permute  o0[3]  렌더 ['B','n_h_kda','d_head_kda','1']  구체 [1,32,128,1]  args=[[2,0,3,1]]
op246 view     i0[3]  렌더 ['B','n_h_kda','d_head_kda','1']  구체 [1,32,128,1]  args=[[1,32,128]]
```

순열 `[2,0,3,1]` 이므로 입력 축 1 -> 출력 축 3 이 맞습니다. 계보는 정확합니다.
**문제는 그 축의 이름이 한쪽에서 `B`, 다른 쪽에서 `1` 이라는 것입니다.**

배경: `op242` 의 view 는 `[32,1,1,128]` 을 만들고, 그 축 1 과 축 2 는 둘 다 크기 1 입니다.
지금 라벨러는 첫 크기-1 축을 `B` 로, 그 뒤를 `1` 로 씁니다(`batch_excl` 규칙).
그런데 `op244` 의 permute 뒤에는 그 두 축의 **위치가 바뀌어** 축 3 이 `1` 로 렌더됩니다.

즉 **정답이 `B` 인지 `1` 인지 자체가 불분명합니다.** 이 텐서는 KDA 의 `[n_h, B, T, d]`
레이아웃이고 prefill 에서도 `B=1`, `T=1` 로 둘 다 1 이라 값으로는 영원히 못 가릅니다.

**질문 5-1**: 이런 경우 hard-anchor ledger 는 무엇을 근거로 삼아야 합니까?
저희가 생각한 것은 **`view` 의 `scalar_args` 에 있는 target sizes 와 그 함수의 소스 줄**
입니다 -- `x.view(n_heads, batch, seq, head_dim)` 같은 호출이면 축 1 이 batch 라는 근거가
됩니다. 그런데 `args=[[32,1,1,128]]` 처럼 **숫자만 남아 있으면** 그 근거가 없습니다.
소스를 읽는 층(LLM 질문)으로 넘기는 것이 맞습니까?

**질문 5-2**: 그렇다면 이 680건은 **`ambiguous` 로 분류되어야 할 첫 사례**로 보입니다.
후보가 `{B, 1}` 인데, `1` 은 `literal` 이고 `B` 는 `known` 이라 **상태가 다른 두 후보**입니다.
이런 혼합 후보를 상태 6종 체계에서 어떻게 다뤄야 합니까?

---

## 제가 틀렸던 것 (이번 세션)

판단에 참고하시라고 적습니다.

1. tie 시점에 비우려 함 -> 지적받고 최종 단계 판정으로 수정
2. `origin` 을 union 키로 쓰려 함 -> 지적받고 `same` 관계로만 union
3. `expand` 를 view 계열에 뭉뚱그림 -> oracle 0/314 로 드러남
4. hybrid 충돌 1,184 를 "모드 분리하면 사라진다"고 가정 -> 지적받고 분해해 보니 틀림
5. 규칙을 **발행된 라벨**로 재고 파이프라인 중간 상태로 안 잼 -> 충돌 1,331 생성
6. 등가류 단위로 옮겨야 한다는 것을 V4-Pro 에서 배우고도 라운드 6 에서 또 밟음

같은 종류입니다 -- **검증하기 전에 성립한다고 가정**하는 것. 그래서 이번에는 다음
단계(#4 tensor-scoped virtual semantic port)를 **설계만 하고 확인받은 뒤** 구현하려 합니다.

## 6. 다음 단계 설계 확인 (#4)

지금 barrier 는 `repeat_kv(n_rep=1)` 이 일어난 `op_id` 목록이고, 전치 규칙에서 "그 앞뒤에
걸치면 잇지 않는다"로 씁니다. 무관한 전치까지 막는다는 지적을 이해했습니다.

바꾸려는 것:

```python
# semantic_events: no-op 이어도 반환 텐서에 새 logical version 을 준다
logical_version[tensor_uid] += 1
logical_producer[(tensor_uid, version)] = SemanticPort(event_id, output=0, role="n_h")

# tracer: 입력 참조를 (tensor_uid, version) 으로 기록
row["input_sources"] = [(producer_op, slot) | ("sem", event_id, 0)]

# lineage: SemanticPort 를 실제 노드로 두고
#   ATen producer port --derived/barrier--> SemanticPort --same--> downstream input
```

**질문 6-1**: `AxisSite` 를 `(op_id, field, slot, axis)` 로 유지하면서 `SemanticPort` 를
표현하려면 `op_id` 자리에 이벤트 id 를 넣고 field 를 `"sem"` 으로 두려 합니다.
이 방식이면 기존 코드(등가류, 교정 selector, 게이트)가 그대로 도는데, 노드 종류를
구분하려면 별도 판별이 필요합니다. 이대로 가도 됩니까?

**질문 6-2**: `derived/barrier` 간선은 union 하지 않는데, 그러면 `SemanticPort` 는
상류와 끊긴 채 하류만 붙습니다. 지적하신 "하류 계보가 고아가 된다"는 문제는 이걸로
해결되지만, **상류 쪽 축이 무엇이었는지**는 어떻게 추적합니까? 별도 lineage 기록으로만
남기고 UF 에는 안 넣는 것이 맞습니까?
