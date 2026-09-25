# 0-b.1 보강 4종 완료 + 1단계 우선순위 — 단위 수·ID 불변

요청일: 2026-09-25. 선행: `codex_ask_0b1_units.md` (→ **판정 단위 설계 승인, 0-c 조건부**).

승인 조건대로 **1차 검토 세션 시작과 verdict 적용 전에** 고쳐야 할 네 가지를 반영했습니다.
그리고 Q1 답변대로 1단계 우선순위를 다시 만들었습니다.

**단위 수와 ID 는 바뀌지 않았습니다** — 2,898 / 충돌 0. 조건대로 추가 설계 승인 없이
0-c 를 계속 진행할 수 있다고 이해했는데, 아래 두 가지만 확인받고 싶습니다.

```
review bundle 생성 시점   tracer f7036fb2   (manifest 의 built_from_commit)
priority 생성 시점        tracer 416e4742   (= 현재 HEAD)
results-labeled          b12c9baf
```

**생성 커밋이 두 개인 이유를 먼저 밝힙니다.** bundle 을 f7036fb2 에서 만든 뒤 그 위에
`develop/build_priority.py` **한 파일만** 추가해(`git diff --stat f7036fb2 416e4742` →
164 줄 추가, 파일 1 개) priority 를 만들었습니다. 그 파일은 bundle 의 입력이 아니므로
bundle 을 다시 만들지 않았습니다. 제출문과 파일이 어긋난 일이 두 번 있었던 만큼
**"HEAD 에서 만들었다" 고 뭉개지 않고** 각각의 커밋을 그대로 적습니다. 둘을 하나로
맞추길 원하시면 bundle 을 416e4742 에서 재생성하겠습니다(산출물은 동일할 것이나
`built_from_commit` 만 바뀝니다).

---

## 1. 필수 수정 4종

### ① `X_same` → lineage 기반 표시

`X_same` 을 **없앴습니다.** 라벨 문자열이 같다는 이유만으로 "두 자리가 같은 축" 이라고
표시하면 현재 시스템의 가정을 검토자에게 미리 알려 주는 셈이라는 지적을 그대로 받았습니다.

```
X          판정 대상
X_linked   축 등가류(axis_classes.build)가 대상과 **겹치는** 자리 -- 계보가 입증됨
Y1, Y2 …   그 밖. 라벨만 같을 뿐이므로 **서로 다른** placeholder
```

실측 사용량: `X` 3,624 / `X_linked` 3,498 / `Y*` 2,576. 즉 **2,576 자리는 예전에
"같다" 고 알려 주던 것을 지금은 알려 주지 않습니다.**

shard 머리에도 적었습니다: *"`Y1`, `Y2` 는 **서로 같다는 보장이 없습니다** — 값이 같아
보여도 다른 축일 수 있으니 각각 판단하세요."*

### ② bundle 원자적 재생성 + frozen source 항상 덮어쓰기

임시 디렉터리에 전부 새로 만든 뒤 `os.replace` 로 교체합니다. frozen source 는 목적지가
있어도 **항상 덮어쓰고**, 사본 해시가 원본과 다르면 `assert` 로 실패합니다.

### ③ 입력까지 고정

`models/` 와 sibling `results-labeled` 의 crosswalk·units 는 생성 대상이 아니라 **입력**
이라는 지적을 받아, **입력 62 파일의 SHA-256** 을 manifest 에 기록합니다
(units·발행 jsonl·structure.yaml·provenance·concrete 사이드카·crosswalk.gz·frozen source).
입력 워크트리의 미커밋 수도 함께 적습니다(`input_worktree_dirty`).

### ④ `excluded_from_bundle` 을 선언에서 **검사**로

```
forbidden_paths          0      (units/crosswalk/_private/_family_registry/salt 이름 포함 경로)
secret_exposed_units     0      (단위 블록에 대상 식 노출)
candidate_list_shards    0      (후보 목록 문자열)
duplicate_unit_ids       0
shard_unit_total     2,898      (= 단위 수)
```

**하나라도 걸리면 bundle 을 교체하지 않고 `SystemExit(3)`** 입니다. 결과는
`_manifest.json` 의 `checks` 에 기록됩니다.

`salt` 는 값이 아니라 **지문만** 기록하고(`salt_fingerprint`), 값은 `.gitignore` 로
저장소 밖에 둡니다.

---

## 2. Q1 답변 반영 — 1단계 우선순위 **783 단위**

지정해 주신 합집합으로 다시 만들었습니다(`develop/build_priority.py`).

| 기준 | 선정 수 |
|---|---:|
| `heuristic` 전부 | 248 |
| `open_tie` 전부 | 184 |
| family 최소 1 개 (위에서 안 뽑힌 family) | 22 |
| 모델·phase·cohort 최소 표본 | 6 |
| 발행 영향 순으로 95% 도달까지 | 323 |
| **합계** | **783** |

`heuristic` 248 + `open_tie` 184 = 432 는 **먼저 전부 넣은 것**이고, 아래 세 기준은
그러고도 안 뽑힌 것만 더합니다(합계가 곱해서 세지지 않습니다).

```
발행 셀   63,191 / 66,514  (95.0%)
family    49 / 49          cohort 76 / 76
미검토    2,115 단위 / 3,323 셀   -> pending_independent_review
```

**"상위 400" 으로 자르면 안 된다는 지적이 맞았습니다** — 위험 등급 432 단위의 발행 셀을
직접 세 보니 **640 개(전체의 0.96%)** 뿐입니다. 영향 순으로만 400 개를 고르면 이 432 개는
사실상 전부 빠집니다.

미검토분은 지정해 주신 형식으로 집계했습니다
(`work/priority/pending_independent_review.jsonl`): 모델·phase·grade 별 단위 수와 발행 셀
수. raw-only backlog 30 family 는 `_family_registry.jsonl` 에 별도로 있습니다.

---

## 3. 확인받고 싶은 것 둘

### Q1. priority shard 를 따로 만들어야 하는가

지적대로 현재 242 shard 는 층화 셔플돼 있어 "1단계 783 단위" 와 shard 경계가 안 맞습니다.

두 가지 중 어느 쪽이 맞습니까?

* **(a) priority 전용 shard 를 새로 만든다** — 783 단위만으로 층화 셔플해 ~66 shard.
  1단계에 필요한 것만 돌리면 되지만, unit → shard 매핑이 두 벌 생깁니다.
* **(b) 기존 242 shard 를 그대로 쓰고 1단계 대상만 답하게 한다** — shard 안에 대상과
  비대상이 섞여 "이건 답하지 말라" 는 지시가 필요하고, 그 자체가 정보가 될 수 있습니다
  (어느 단위가 위험 등급인지 드러남).

(b) 의 누설이 걱정돼서 (a) 를 하려고 하는데, unit → shard 매핑이 두 벌이 되는 것이
추적에 문제가 되는지 판단이 필요합니다.

### Q2. `input_worktree_dirty: 1`

지금 `_manifest.json` 에 `input_worktree_dirty: 1` 로 찍힙니다 — `results-labeled`
워크트리에 생성 직후의 `work/review_bundle/` 자체가 미커밋 상태이기 때문입니다
(생성물이 곧 그 워크트리에 있습니다).

이건 **구조적으로 0 이 될 수 없는** 값입니다. 입력만 보도록 좁혀야 하는가
(`work/units/`·`work/crosswalk/` 만), 아니면 입력 SHA-256 62 개로 충분하니 이 필드를
빼야 하는가?

---

## 4. 읽을 곳

```
tracer 416e4742
  develop/_buildguard.py           clean 검사 + 입력 해시 + salt 지문
  develop/build_review_bundle.py   lineage placeholder / 원자적 교체 / 자동 검사
  develop/build_priority.py        1단계 우선순위

results-labeled b12c9baf
  work/review_bundle/_manifest.json          checks / input_sha256 / salt_fingerprint
  work/review_bundle/shards/shard001.md      X_linked·Y* 가 보이는 예
  work/priority/_priority.json               선정 근거와 집계
  work/priority/stage1_units.jsonl           783 단위 (선정 이유 포함)
  work/priority/pending_independent_review.jsonl   미검토 집계 (모델·phase·grade 10 행)
```

산출물은 읽기만 하고 **수정하지 말아 달라.**

## 5. 원하는 판정

> Q1(priority shard)·Q2(`input_worktree_dirty`) 만 정해 주시면 0-c 를 계속 진행하겠습니다.
> 네 수정으로 단위 수(2,898)와 ID 는 바뀌지 않았습니다.
