# 실행 계획을 승인해 주세요 — undecided-first-class

지난 답변을 반영해 계획을 다시 짰습니다. **승인 전에는 구현하지 않겠습니다.**
틀린 부분이 있으면 그 단계만 짚어 주시면 됩니다.

지난 답변에서 제가 이해한 핵심 교정 다섯 가지를 먼저 적습니다. 이 이해가 틀렸으면
그것부터 알려 주십시오.

1. **tie 시점에 비우지 않는다.** resolver 의 tie 는 파이프라인 초반이고, 그 뒤 provenance /
   split / transpose / anchor / source verdict 가 답을 줄 수 있다. 확정은 **마지막에**.
   따라서 "tie 였던 축"(465,213)과 "최종 미결 축"은 다른 집합이다.
2. **상태는 6종**: known / ambiguous / heuristic / unknown / literal / conflict. 합치지 않는다.
   특히 `literal` 과 `unknown` 을 갈라야 루프 카운터가 질문으로 다시 올라오지 않는다.
3. **canonical CSV 는 문법을 바꾸지 않는다.** 구체값을 유지하고 불확실성은 사이드카/열로.
   `?a|b` 는 사람용 `review.md` 에만(`128?{d_head | n_h}`).
4. **비율 임계치는 쓰지 않는다.** 척도는 `unexplained_undecided == 0`.
5. **질문을 접는 기준은 축 수가 아니라 provenance fingerprint 동일성**이고, 답의 적용은
   명시된 class fingerprint 목록에만 한다.

---

## 0단계 — 지금 코드의 결함 두 개 (지적해 주신 것)

### 0-A. 계보와 값 기반이 함께 돈다

지금 `build()` 는 `lineage_edges()` 로 union 한 뒤 **같은 행에** 기존 `depends_on + shape`
값 기반 간선을 무조건 다시 돌립니다. 그래서 값 기반의 잘못된 union 이 그대로 남습니다.

바꾸려는 것:

```python
for r in rows:
    if "input_sources" in r:          # 키 존재 여부 -- 값이 아니라
        edges = lineage_edges_for(r)  # 정확한 포트만
    else:
        edges = legacy_shape_edges_for(r)   # 구 트레이스 폴백
```

`row.get(...)` 가 아니라 키 존재를 보는 이유도 이해했습니다 -- **입력이 정당하게 빈
factory op** 와 **구 트레이스라 필드가 없는 행**은 다릅니다. `legacy_fallback_ops()` 도
같은 기준으로 고칩니다.

**질문 0-A**: 전환 기간에 두 근거를 나란히 돌려 비교하고 싶은데(shadow), 그러려면 같은
`rows` 로 UF 를 두 벌 만들어 diff 만 내는 것이 맞습니까? 아니면 바로 갈아타고 oracle
로만 검증하는 편이 낫습니까?

### 0-B. no-op barrier 가 시간 범위 기준이다

지금은 `repeat_kv(n_rep=1)` 이 일어난 `op_id` 를 모아, **그 앞뒤에 걸친 전치**를 잇지
않습니다. 무관한 텐서의 전치까지 막을 수 있다는 지적을 이해했습니다.

바꾸려는 것 -- **tensor-scoped**: `repeat_kv` 가 no-op 이면 그 반환 텐서에 **새 logical
version** 을 부여하고, 그 시점 이후 그 텐서를 읽는 소비자는 새 version 으로 본다.

```python
# semantic_events 에서
logical_version[tensor] = next(_ver)     # n_rep == 1 이어도 새 version
# tracer 는 행마다 입력 텐서의 (uid, version) 을 남긴다
```

**질문 0-B**: 축 계보의 노드 키를 `(op_id, field, slot, axis)` 에서 바꾸지 않고,
**입력 포트가 가리키는 생산자를 version 으로 갈라** 표현하려 합니다. 즉 같은 물리 텐서라도
version 이 다르면 `same` 간선을 놓지 않습니다. 이 방식으로 충분합니까, 아니면 축 노드 자체에
version 이 들어가야 합니까?

---

## 1단계 — 축 상태를 자료구조로 (관측만, 표기 변경 없음)

축 슬롯마다 상태를 계산해 **사이드카로만** 낸다. 산출물 표는 손대지 않는다.

```json
{"site": [812,"o",0,2], "value": 128, "status": "ambiguous",
 "reason": "value_tie", "candidates": ["d_head","n_h"],
 "evidence": {"hard": [], "soft": ["scoped_symbol"]},
 "class_fingerprint": "..." }
```

상태 판정 순서(지적하신 순서 그대로):

```
resolver 가 후보와 근거를 남긴다
  -> provenance / split / transpose / parameter anchor / source verdict 적용
  -> 등가류 전체의 hard evidence 를 모은다
  -> 정확히 하나면 known / 없고 후보 여럿이면 ambiguous / 충돌이면 conflict
```

**질문 1-A**: **무엇을 hard evidence 로 봅니까?** 저희가 생각한 것은 이 넷입니다.

| 근거 | 왜 hard 인가 |
|---|---|
| `nn.Linear` 등 모듈이 **선언한** in/out 폭 | 파라미터 shape 은 config 에서 온 것이지 값 매칭이 아니다 |
| `split_with_sizes` 의 항 순서 | op 인자가 폭을 직접 말한다 |
| `Cache.update` 의 key/value 인자 위치 | 파이썬 레벨에서 이름으로 확정했다 |
| `rules/label_overrides.yaml` 의 소스 인용 판정 | 사람/LLM 이 소스를 읽고 확정했다 |

`scoped_symbol`(모듈 스코프에서 그 값을 가진 심볼)은 **soft** 로 보려 합니다 -- 값 매칭이
근거이기 때문입니다. 지금 라벨의 43.6% 가 이것이라 hard 로 치면 아무것도 미결이 되지
않습니다. 이 구분이 맞습니까?

**질문 1-B**: `literal` 은 어떻게 판정합니까? 저희 후보는 "크기 1" + "루프 인덱스로 판명된
축"(`_unname_loop_indices` 가 이미 잡는 것) + "op 가 만든 상수 길이"(conv1d 의 캐시 창
길이 같은 것)입니다. 더 있어야 합니까?

---

## 2단계 — 실제 미결 수를 잰다 (**판단 지점**)

1단계 상태를 47개 모델에서 세고, **`unexplained_undecided` 가 0 인지** 본다.

정의: `ambiguous` 인데 그 이유를 대지 못하는 축. 즉 hard evidence 가 있는데도 미결이거나,
등가류에 확정 anchor 가 있는데 전파가 안 된 축.

내려면 하는 표:

```
모델별   known / ambiguous / heuristic / unknown / literal / conflict
전체     같은 6종 + unexplained_undecided
비교     지금 이름이 붙은 축 중 몇 개가 ambiguous 로 내려가는가
         (= 지금까지 근거 없이 붙어 있던 이름의 수)
```

**질문 2-A**: 이 단계에서 **중단해야 할 신호**는 무엇입니까? 지난 답변의
"직접 tie 대비 최종 미결이 2배 넘으면 원인 분석 / 5배면 즉시 중단"을 그대로 쓰면 됩니까?
`unexplained_undecided == 0` 이면 몇 배든 진행해도 됩니까?

**질문 2-B**: `hard_known -> undecided 0` 을 어떻게 잽니까? "지금 known 인 축"의 기준이
문제입니다 -- 지금은 근거 없이 붙은 이름도 known 처럼 보입니다. 저희 안은 **소스 인용
판정이 있는 자리**(overrides + confirmed)만 hard_known 으로 보고 그것들이 하나도
미결로 내려가지 않는지 보는 것입니다. 충분합니까?

---

## 3단계 — 발행 (산출물이 바뀐다)

- `<phase>.csv` / `.jsonl`: **구체값 유지, 문법 불변.**
  `output_axis_status` / `output_axis_candidates` 열을 추가.
- `full/<phase>.axis_status.jsonl`: 사이트별 전체 상태(위 JSON).
- `review.md`: 사람용 `128?{d_head | n_h}`.
- `model_summary.md`: 상태별 집계를 카드에 노출.

**질문 3-A**: CSV 에 열을 추가하는 것과 사이드카만 두는 것 중 어느 쪽을 권하십니까?
지금 `<phase>.csv` 는 사람이 읽는 주 산출물이고 검토 패킷도 여기서 나옵니다. 열을 늘리면
읽기가 나빠지고, 사이드카만 두면 표만 보는 사람은 불확실성을 못 봅니다.

---

## 4단계 — 질문 생성

`provenance fingerprint` 로 접는다. fingerprint 는 이 다섯으로 만든다:

```
producer op_type + output_slot + transform_path + module_class + layer_type
```

레이어 번호와 phase 는 fingerprint 에서 **뺀다**(반복이므로). 그 외가 다르면 다른 질문.

질문 파일은 지난 답변의 스키마를 그대로 쓰겠습니다(`schema`, `id: axisq:<hash>`,
`graph_fingerprint`, `source_revision`, `members.class_fingerprints`, `origin`,
`evidence.source_ranges` with `file_hash`, `answer`).

**질문 4-A**: `transform_path` 를 어디까지 담습니까? 예: `[split/out2, view, transpose]`.
너무 길면 fingerprint 가 과하게 갈려 질문이 폭증하고, 너무 짧으면 다른 역할이 한 질문에
묶입니다. **생산자로부터 몇 단계**까지가 적당합니까?

**질문 4-B**: `counterexamples` 필드를 주셨는데, 무엇을 담아야 합니까? 저희는 "같은 모듈
같은 값인데 fingerprint 가 달라 이 질문에서 **제외된** 자리"를 담으려 합니다 -- 답하는
쪽이 "이건 그 축이 아니다"를 확인할 수 있게. 맞습니까?

---

## 5단계 — 답 적용기

LLM 은 `answer.{status, symbol, rationale, citations}` 만 채운다. selector 는 생성기가 고정.

검증 사슬(지난 답변 그대로):

```
symbol ∈ candidates            source file hash 동일       인용이 해당 함수 안
class member 집합 불변          예상 phase/layer/class/axis 수 정확히 일치
대상 밖 변경 0                  class 안 반대 hard anchor 0
dry-run 후 전체 게이트 + 339/314 oracle 통과
다음 재생성에서 미발화하면 stale
```

`resolved_unknown` 은 `rules/axis_decisions.yaml` 에 fingerprint + candidates +
source hash 와 함께 저장하고, 그것들이 그대로인 동안만 질문을 억제한다.

**질문 5-A**: "예상 축 수 정확히 일치"를 어떻게 계산합니까? 질문 생성 시점의
`members.axes` 를 그대로 쓰면, 그 사이에 다른 판정이 반영돼 수가 달라졌을 때 정당한 변화도
거부합니다. `graph_fingerprint` 가 같으면 축 수도 같다고 봐도 됩니까?

**질문 5-B**: `split_required` 답이 오면 어떻게 처리하는 것이 좋습니까? 자동으로
fingerprint 를 더 잘게 쪼개 재생성해야 합니까, 아니면 사람이 개입해야 합니까?

---

## 6단계 — 전 모델 재트레이스 → 게이트 → 외부 검토

게이트에 추가할 것:

```
hard_known -> undecided        0
unexplained_undecided          0
conflict                       0
source-confirmed regression    0   (339 / 314 oracle)
legacy_fallback_ops            (0 이 되면 값 기반 간선 제거)
```

---

## 마지막 질문 — 순서가 맞습니까

0 → 1 → 2(판단) → 3 → 4 → 5 → 6 으로 보는데, 특히 **3단계(발행)를 4·5단계(질문·답) 앞에
두는 것**이 맞는지 확신이 없습니다. 답을 먼저 받아 미결을 줄인 뒤 발행하면 산출물이 덜
비어 보이지만, 그러면 "무엇을 물어야 하는가"를 정하는 근거가 발행되지 않은 상태가 됩니다.

어느 순서를 권하십니까?
