# 6단계 확인 — NodeRef / logical version / SemanticPort

1~5단계는 지시대로 끝냈습니다. **6단계는 처음으로 `src/` 를 건드리므로 승인 전에는
구현하지 않겠습니다.** 아래는 전부 실측이고, 마지막에 질문 6개가 있습니다.

## 1~5단계 결과 (요약)

| 도구 | legacy | provenance | 자기검사 |
|---|---|---|---|
| `develop/oracle_314.py` — 11 selector / **314 occurrence** | 끊김 314 | 끊김 0 | 4종 PASS |
| `develop/expand_contract.py` — 165 자리(개명) | 끊김 85 | 끊김 0 | 2종 PASS |
| `develop/test_semantic_fixtures.py` | 2 PASS / 3 XFAIL | | |

* **occurrence 하나 = expand 행 하나**입니다. 입력 축과 출력 축이 같은 행에서 나오므로
  `zip` 정렬이 어긋날 수 없습니다. 개수가 `expected` 와 다르면 FAIL 입니다(반복 레이어
  삭제/추가를 자기검사로 확인했습니다).
* Nemotron `mixer|prefill|nth=3|axis=1` 은 지문 없이 48이 잡히는데 **40이 mamba 층, 8이
  attention 층**이었습니다. `layer_sig` 로 갈라 원본 40을 정확히 재현했습니다.
* fixture 3 의 `scalar_args` 지적이 맞았습니다. `_perm_from_args` 는 `args["pos"]` 를 읽습니다.
  이제 **barrier 없는 대조군을 먼저 통과**시킨 뒤 barrier 판을 검사합니다.
* 지표를 `symbol_pair_collision`(진단) / `forbidden_union`(하드 게이트, 지금 비어 있음) /
  `class_label_conflict` 로 나눴습니다. Kimi·Nemotron 쌍은 전역 금지 목록에서 뺐습니다.
* 새 실측: `class_label_conflict` 는 provenance 에서 **1,496 등가류 / 9개 모델**입니다
  (legacy 는 0 — 파이프라인이 등가류마다 이름을 하나로 강제하기 때문입니다).

한 가지 사고를 보고합니다. `provenance_oracle.py --freeze` 가 `positive_expand` 를 현재값
(0)으로 덮어 **314 원본 기록을 한 번 날렸습니다.** `git checkout HEAD` 로 복구했고, 이제
원본은 `develop/verify/oracle_314.json` 에 따로 있어 freeze 가 못 건드립니다.

---

## 6단계 변경 범위 (실측)

```
input_sources 를 쓰는 곳   src/tracer.py:89-93, 170                     1곳
2-튜플을 푸는 곳           src/axis_classes.py:209  pop, pslot = src
                          src/tracer.py:96         deps.append(src[0])
                          develop/axis_lineage.py:176                   3곳
사이드카를 읽는 곳         src/axis_classes.py:868 attach_ports
디스크의 사이드카          models/*/full/*.ports.jsonl                  47/47 모델
```

현재 형식:

```json
{"input_sources": [[0, 0]],  ...}   // 정상 포트
{"input_sources": [null, null], ...} // 생산자를 모름
{"input_sources": [], ...}           // 텐서 입력이 아예 없음(factory)
```

`null` 이 무엇인지 실측했습니다(3개 모델, prefill):

```
정상 포트            19,356
null / params 있는 행 1,330   <- 대부분 가중치 입력
null / params 없는 행    73   <- graph 입력·buffer·미상
입력 없음(factory)      425
```

`param_origin` 은 **텐서 단위**라 슬롯별 parameter 판정이 트레이스 시점에 정확히 됩니다.
`graph_input` / `buffer` / `unknown_external` 을 가르는 것은 그것만으로는 안 됩니다.

## 위험도 — legacy 는 안 바뀝니다

`src/axis_classes.py:345` 에서 `if mode != "legacy":` 안에서만 `lineage_edges` 를 부릅니다.
즉 **legacy 는 `noop_barriers` 를 아예 안 읽습니다.** 발행 산출물은 legacy 로 만들어지므로
전역 시간 barrier 를 지워도 CSV/JSONL 의 라벨은 한 글자도 안 바뀝니다. 회귀 위험은
`develop/` 의 측정값에 한정됩니다.

---

## 구현안

### (1) NodeRef — 내부는 일관, 경계에서 변환 (지시하신 (a))

```python
class OpRef(NamedTuple):    kind="op";  op_id: int;  output: int
class SemRef(NamedTuple):   kind="sem"; phase: str;  event_id: int; output: int = 0
class ExtRef(NamedTuple):   kind="ext"; external: str   # graph_input|parameter|buffer|unknown_external
                                        name: str | None
```

JSON 코덱:

```json
{"kind":"op","op_id":242,"output":0}
{"kind":"sem","phase":"prefill","event_id":5,"output":0}
{"kind":"ext","external":"parameter","name":"model.layers.0.self_attn.q_proj.weight"}
```

factory op 은 `input_sources: []` 로 그대로 둡니다(입력이 없는 것이지 모르는 것이 아닙니다).
`label_overrides` selector 는 지시대로 **`ActualPort` 만 대상**이므로 `(op_id, "o", 0, 2)`
형태를 그대로 씁니다. 변환은 selector 경계에서만 합니다.

### (2) logical version — 지시하신 흐름

```python
pre_ref = tracer.logical_source(hidden_states)      # orig 호출 전 보존
out = orig(hidden_states, n_rep, ...)
event = make_event(pre_ref, out, at_op_id)
tracer.bump_version(out)
tracer.bind_logical_source(out, SemRef(phase, event.id))
```

각 ATen 소비자는 시간 비교 없이 **실행 당시** 그 텐서의 논리 출처를 `input_sources` 에
적습니다. `op_id > at_op_id` 는 결과이지 판별식이 아닙니다.

### (3) SemanticSpec 레지스트리

`rules/semantic_specs.yaml` 에 두려 합니다(버전 관리되고 검토 가능하므로).

```yaml
- qualified_name: transformers.models.llama.modeling_llama.repeat_kv
  source_sha256: <설치본에서 읽은 해시>
  axis: 1
  role_before: n_kv
  role_after: n_h
```

등록에 없으면 **축을 추측하지 않고** `resolved: false` 로 기록하고 barrier 를 안 겁니다.

### (4) 축별 barrier + 전역 시간 barrier 원자적 제거

`SemanticPort` 를 실제 노드로 두고, 역할이 바뀌는 축만 `derived`(UF union 안 함),
나머지는 `same` 입니다. `lineage_edges` 의 `bars` 인자와
`min(bars) <= oid <= max(bars)` 조건은 같은 커밋에서 지웁니다.

### (5) 검증

fixture 3·4·5 가 초록으로 뒤집힙니다. 그 외:

```
oracle 314           끊김 0 유지 (회귀 검사)
expand 계약 165      끊김 0 유지
permute sentinel     13,900 이 얼마나 내려가는지 **규칙별 잔여를 출력**합니다.
                     0 으로 미리 고정하지 않겠습니다 -- 실제 scalar-arg 누락과 분리해야 하므로.
Zamba2 canary        barrier contract 통과 / 무관한 전치 복구 / topology 회귀 없음
Kimi-K3, Nemotron    partition 불변 (semantic 이벤트가 0개 / n_rep=16, noop=0)
```

---

## 질문

**6-1 (사이드카 이행).** 47개 모델의 `*.ports.jsonl` 이 전부 옛 2-튜플 형식입니다. 읽는 쪽에
**두 형식을 다 받는 코덱**을 두고(`[op_id, slot]` -> `OpRef`, `null` -> `ExtRef("unknown_external")`),
`schema: 2` 필드를 새 트레이스에만 붙이려 합니다. 전체 재트레이스는 어차피 10단계에서
합니다. 이 방식이 맞습니까, 아니면 옛 형식을 아예 거부하고 재트레이스를 6단계 안으로
당겨야 합니까?

**6-2 (ExternalRef 분해 범위).** 위 실측대로 `null` 1,403건 중 1,330건이 parameter 쪽이고
73건이 나머지입니다. `parameter` 는 `param_origin` 으로 슬롯별로 정확히 가를 수 있지만
`graph_input` / `buffer` / `unknown_external` 은 아직 못 가릅니다. **지금은 `parameter` 만
정확히 붙이고 나머지를 `unknown_external` 로 두는 것**이 맞습니까? (`missing_port_records`
를 하드 게이트로 올릴 때 다시 보면 된다고 이해했습니다.)

**6-3 (제자리 연산).** Zamba2 prefill 에 `copy_` 76건, `_to_copy` 1건이 있습니다. 제자리
연산은 **같은 `tensor_uid` 인데 내용이 바뀝니다.** `bump_version` 을 semantic 이벤트에서만
합니까, 아니면 제자리 ATen 도 version 을 올려야 합니까? 안 올리면 `logical_source` 가
덮어써진 텐서에 옛 SemanticPort 를 계속 가리킬 수 있어 보입니다.

**6-4 (source hash 를 못 읽는 경우).** `inspect.getsource` 가 실패하는 구현(C 확장, 컴파일된
것, 동적 생성)이 있습니다. 그때는 `qualified_name` 만으로 매칭합니까, 아니면
`resolved: false` 로 둡니까? 후자가 안전해 보이지만 정상 구현까지 놓칠까 걱정됩니다.

**6-5 (SemanticPort 의 phase namespace).** `SemRef` 에 `phase` 를 넣었습니다. prefill 과
decode 는 **별도 트레이스/별도 사이드카**라 사실 충돌하지 않습니다. 넣는 것이 맞습니까,
아니면 불필요한 필드입니까?

**6-6 (커밋 단위).** (1)(2) 를 먼저 넣고 라벨 무변화를 확인한 뒤, (3)(4) 를 두 번째 커밋으로
넣으려 합니다. 그런데 (2) 만 넣고 (4) 를 안 넣으면 전역 시간 barrier 와 SemanticPort 가
**동시에 살아 있는 중간 상태**가 됩니다. 지시하신 "원자적 제거" 와 맞습니까, 아니면
(1)~(4) 를 한 커밋으로 가야 합니까?
