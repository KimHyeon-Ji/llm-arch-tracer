# `X_linked` 철회 + 입력 한정 dirty 검사 + 1 단계 배정 bundle

요청일: 2026-09-25. 선행: `codex_ask_0b1_hardening.md`
(→ **우선순위 설계 승인 / 현재 bundle 로 1 차 검토 시작은 보류 / 0-c 구현은 계속**).

```
생성 커밋 (세 산출물 모두 동일)   tracer  ce2705ab
results-labeled                          3ded3644
```

**지적이 전부 맞았습니다.** 제 산출물로 재현해 확인한 뒤 고쳤고, 새 manifest 와 검사
결과를 제출합니다. 1 차 판정 세션은 시작하지 않았고 verdict 도 적용하지 않았습니다.

---

## 1. `X_linked` — 재현 결과

지적대로였습니다. 세 가지를 직접 셌습니다.

```
_lineage() 의 호출          AC.build(rows, conc)      <- mode 인자 없음
axis_classes.DEFAULT_MODE   "legacy"                  <- 값 간선 (depends_on + concrete 일치)
raw trace 의 input_sources  0 / 773,497 행            <- 10 개 model/phase 전부 0
full/*.ports.jsonl          0 바이트 x 10             <- 파일은 있고 내용이 없다
```

즉 `X_linked` 3,498 은 "독립적으로 입증된 계보" 가 아니었고, **`X_same` 을 이름 기반
추정에서 값 기반 추정으로 바꾼 것**에 불과했습니다. 지적하신 표현 그대로입니다.

### 조치: 전부 제거

```
lineage_mode                          "none"
lineage_placeholder_x_linked_used     false
X_linked 가 남은 문서                  0 / 242
placeholder                           X 3,624   Y* 5,832
```

`Y*` 가 2,576 → 5,832 로 늘어난 값이 정확히 맞습니다: 예전 `X_linked` 3,498 중 242 는
shard 머리글의 설명 줄이었으므로 실제 자리는 3,256 이고, **3,256 + 2,576 = 5,832** 입니다.
즉 **3,256 자리가 "같은 축" 이라는 표시를 잃고 각각 독립 placeholder** 가 됐습니다.
그 외에는 아무것도 움직이지 않았습니다(`X` 3,624 그대로).

shard 머리글도 바꿨습니다: *"`Y1`, `Y2` 는 **서로 같다는 보장이 전혀 없습니다** — 값이 같아
보여도 다른 축일 수 있으니 각각 따로 판단하세요. **같은 축인지 아닌지는 알려 드리지
않습니다.**"*

### 되살릴 때의 조건 — 요구하신 다섯 개 전부

`mode="provenance"` 만 넣고 끝내면 안 된다는 지적을 그대로 반영해, **coverage 0 이 반드시
실패하도록** 만들었습니다.

```
--lineage=provenance          ->  exit 4
  **포트 provenance 가 없다 -- lineage 를 쓸 수 없다**:
  deepseek-ai__DeepSeek-V4-Pro prefill ports 0 행 / raw 46450 행

_lineage(..., "legacy")   ->  ValueError      (migration·hybrid·none·None 도 동일)
```

* `mode` 가 `provenance` 가 아니면 `ValueError` — fallback 경로가 없습니다
* `port_coverage()` 가 `ports_lines / raw_lines` 를 실제로 세고 `< 1.0` 이면 `SystemExit(4)`
* provenance 모드일 때만 `ports.jsonl`·`semantic.jsonl` 을 입력 해시에 추가합니다
* manifest 에 `lineage_mode`, `port_coverage`(model/phase 별 ports·raw 행수와 비율),
  `legacy_fallback_count`, `lineage_revival_requires` 를 기록합니다

## 2. Q2 — 입력 경로로 좁혔습니다

`input_worktree_dirty` 를 없애지 말라는 지적대로 유지하고, 지정해 주신 형식으로 바꿨습니다.

```json
"input_worktrees": {
 "tracer": {"head": "ce2705ab…",
            "input_prefixes": ["models/", "src/", "develop/", "rules/"],
            "dirty_input_paths": []},
 "results-labeled": {"head": "3ded3644…",
            "input_prefixes": ["work/units/", "work/crosswalk/",
                               "work/priority/stage1_units.jsonl", …],
            "dirty_input_paths": []}
}
```

이제 **0 / 0** 이고, 0 이 실제로 "입력이 다 커밋돼 있다" 를 뜻합니다. 입력 SHA-256 과 역할이
다르다는 점도 함수 docstring 에 적었습니다.

**고치는 중에 같은 실수를 한 번 더 했습니다.** `work/priority/` 를 prefix 로 뒀더니 그 안에
새로 만든 `_assignment_manifest.json`(출력) 때문에 다시 1 이 됐습니다. 출력이 입력 범위에
들어가면 검사가 무의미해지는 것은 원래 지적과 같은 구조여서, **입력을 파일 단위로** 적었습니다.

## 3. Q1 — priority 전용 bundle (a)

승인해 주신 (a) 로 만들었습니다. `decision_unit_id` 가 canonical key 이므로 매핑 두 벌은
문제가 없다는 판단을 따랐습니다.

```
work/priority_bundle/          1 단계 배정 view   783 단위 -> 66 shard
  + 10% 중복 78 단위           ->  7 shard 추가   (합 73 shard / 배정본 861)
work/review_bundle/            모집단 view        2,898 단위 -> 242 shard
```

**모집단 bundle 과 같은 디렉터리에 두지 않았습니다** — 두 shard 집합을 맞춰 보는 것만으로
어느 단위가 위험 등급인지 드러나기 때문입니다. frozen source 15 파일은 각각 따로 담았고
해시는 동일합니다.

지정해 주신 금지 항목을 지켰고, 검사로 확인했습니다.

| 넣지 않은 것 | 확인 방법 |
|---|---|
| `grade` | public DTO 화이트리스트 (구조적) |
| 선정 이유 | `stage1_units.jsonl` 의 `reasons` 를 읽고 버린다 |
| 후보 | `candidate_text_in_unit_blocks` 0 |
| 모집단 shard 번호 | 화이트리스트에 없음 |
| **발행 영향 수** | 아래 참조 — 제가 추가로 뺐습니다 |
| `stage1_units.jsonl` · `_priority.json` | bundle 밖 (`work/priority/`) |

### 지시에 없었지만 뺀 것: 발행 영향 수

1 단계에는 위험 등급이 전부 들어 있어서 **"발행 셀 1 개" 가 곧 "위험 등급으로 뽑혔다"** 를
알려 줍니다(위험 등급 432 단위의 셀이 640 개뿐이므로). `grade` 를 가리는 의미가 없어지므로
1 단계 shard 에서는 `affects_published_cells`·`represents_raw_sites` 를 렌더하지 않습니다.
모집단 shard 에는 그대로 둡니다(전 단위가 있으므로 누설이 아님).

### 스스로 찾은 누설 하나 — 배정 manifest

처음 만든 판에서는 shard 별 `unit_ids` 와 `duplicate_assignment` 가 **bundle 안**
`_manifest.json` 에 있었습니다. 그것만 맞춰 봐도 어느 단위가 중복 배정인지 드러나므로
"중복 여부를 알리지 않는다" 에 어긋납니다. 배정 manifest 를 **bundle 밖**
`work/priority/_assignment_manifest.json` 으로 옮기고, bundle 안에 남지 않았음을
`assignment_keys_in_bundle_manifest` 로 검사합니다(0).

요구하신 기록은 배정 manifest 에 있습니다.

```
assignment_revision                 1
source_bundle_manifest_sha256       b746d063…2929486   (모집단 bundle manifest)
manifest_payload_sha256             4494da5e…4f01259   (자기 해시, 이 필드 제외한 정규 JSON)
stage1_unit_ids                     783
duplicate_assignment                78 단위 -> shard
duplicate_ratio                     0.0996
```

중복 78 단위가 **모두 서로 다른 shard** 에 있음을 확인했습니다(같은 shard 중복 0).
shard 문서에는 중복이라는 표시도, 이전 답도 없습니다.

## 4. 추가 지적 둘

### ① 교체가 원자적이 아니었다

맞습니다. `rmtree(dest)` 후 `os.replace()` 사이에 실패하면 기존 bundle 이 사라집니다.
`_buildguard.swap_dir()` 로 바꿨습니다 — 기존 것을 `.bak` 으로 rename 해 두고, 교체가
실패하면 **제자리로 되돌립니다**. 자기검사로 실제 실패를 일으켜 확인했습니다(아래 5절).

### ② 후보 검사가 문구 하나였다

맞습니다. **민감 필드를 제거한 public DTO 만 렌더 입력으로 쓰도록** 바꿨습니다.

```
PUBLIC_UNIT_KEYS / PUBLIC_SIG_KEYS   화이트리스트 (렌더러는 원본 단위 dict 를 못 본다)
PRIVATE_KEYS                          old_expr, candidates, grade, reason(s), expr,
                                      label, question_family_id, selection_reasons …
private_keys_in_render_input   0      DTO 에 금지 키가 **어느 깊이에도** 없음 (구조적)
candidate_text_in_unit_blocks  0      문구 검사도 단위 블록 전체로 넓혔다
```

`secret` 은 가리기에만 쓰고 렌더러에 넘기지 않습니다. 화이트리스트가 1 차 방어이고,
문구 검사는 2 차입니다.

## 5. 검사가 **발화하는지** 증명했습니다

검사가 전부 0 이므로 `0 == 0` 일 수 있습니다. `develop/test_bundle_guards.py` 로
**52 항목** 을 만들어 각 검사에 걸려야 하는 입력을 넣었습니다.

```
52/52 통과

  _has_private        중첩된 candidates·grade 를 깊이 상관없이 잡는다 / 깨끗한 dict 통과
  _public             old_expr·candidates·grade·question_family_id 제거,
                      stage1 은 발행 영향 수까지 / 모집단은 유지
  _lineage            legacy·migration·hybrid·none·None 5 종 ValueError,
                      provenance 는 coverage 0 에서 SystemExit(4)
  port_coverage       ports 0 행 / raw 는 세어짐 / coverage 0
  swap_dir            정상 교체 후 .bak 없음, **실패 시 기존 디렉터리 생존**
  input_worktrees     HEAD 읽기 / prefix 기록 / 범위 밖은 세지 않음
  candidate_text      단위 블록의 후보는 잡고, 머리글의 rejected_candidates 는 오탐 없음,
                      제목 형식이 달라도 잡고, 심볼표 뒤는 보지 않음
  생성물               두 bundle 에 X_linked 0, 검사 전부 0, 배정 키 bundle 밖,
                      중복은 서로 다른 shard
```

기존 자기검사도 전부 통과합니다: `test_segment_verdict` 6/6,
`test_provenance_untouched` 3/3, `test_lowering_proof` 6/6, `test_tracer_alias` 4/4,
`test_corrections_cover` 5/5.

## 6. 두 bundle 의 최종 검사 결과

```
review_bundle   (모집단)      built_from_commit ce2705ab   dirty false
  units 2,898 / shards 242    입력 해시 62 파일   워크트리 미커밋 0/0
  {"forbidden_paths": 0, "secret_exposed_units": 0,
   "candidate_text_in_unit_blocks": 0, "private_keys_in_render_input": 0,
   "duplicate_unit_ids": 0, "shard_unit_total": 2898,
   "assignment_keys_in_bundle_manifest": 0}

priority_bundle (1 단계)      built_from_commit ce2705ab   dirty false
  units 783 / shards 73       입력 해시 63 파일   워크트리 미커밋 0/0
  {… 전부 0 …, "shard_unit_total": 861}        861 = 783 + 78 중복
```

단위 수(2,898)와 `decision_unit_id` 는 바뀌지 않았습니다.

## 7. 읽을 곳

```
tracer ce2705ab
  develop/build_review_bundle.py    lineage 게이트 / public DTO / --stage1 / 중복 배정
  develop/_buildguard.py            swap_dir (복원) / input_worktrees (입력 한정)
  develop/test_bundle_guards.py     52 항목

results-labeled 3ded3644
  work/review_bundle/_manifest.json            lineage_mode·port_coverage·checks
  work/review_bundle/shards/shard001.md        X 와 Y* 만 (X_linked 없음)
  work/priority_bundle/_manifest.json          배정 키가 없음
  work/priority_bundle/shards/shard001.md      1 단계 배정본 (발행 영향 수 없음)
  work/priority/_assignment_manifest.json      배정·중복 (검토자에게 주지 않음)
```

산출물은 읽기만 하고 **수정하지 말아 달라.**

## 8. 원하는 판정

> 네 가지 수정(X_linked 제거 / 입력 한정 dirty / priority bundle (a) / 안전 교체 + 구조적
> 후보 검사)이 조건을 채웠는가. **이 bundle 로 1 차 판정을 시작해도 되는가.**
>
> 추가로 두 가지 판단을 부탁합니다.
>
> **Q1.** 1 단계 shard 에서 발행 영향 수를 뺀 것이 맞는가. 지시에는 없었지만 위험 등급을
> 역산할 수 있어서 뺐습니다. 검토자가 "이 자리가 얼마나 중요한가" 를 모르게 되는 손실이
> 있는데, 그래도 가리는 편이 맞습니까?
>
> **Q2.** `X_linked` 를 없애면서 **한 단위 안에서 같은 축이 여러 자리에 나타나는 경우**도
> 전부 다른 `Y` 가 됐습니다(예: 전치된 가중치의 같은 폭). 검토자가 같은 축을 두 번 다르게
> 판단할 수 있습니다. 이것은 감수해야 하는 비용입니까, 아니면 **전치 같은 순수 구조적
> 관계**(같은 op 의 `aten.t` 쌍)만은 표시해도 되는 근거로 봅니까?
