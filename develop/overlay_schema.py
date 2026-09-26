r"""0-c 순서 1: **raw overlay 후보**와 판정 상태의 스키마. 아직 적용하지 않는다.

이 모듈은 형식과 상태만 정의한다. 실제 적용(`published_overlay` 파생, csv/jsonl 수정)은
**사람 판정이 끝난 뒤**에만 하고, 그 단계는 별도 승인이 필요하다(외부 검토 2026-09-25).

## 왜 raw 사이트를 키로 쓰는가

발행 셀 하나가 raw 원장 사이트 여러 개에서 온다(`_collapse_norm` 이 합성한 norm 행은
입력은 `first`, 출력은 `last`, 가중치는 `*.weight` 를 가진 멤버에서 왔다). 발행 셀을
키로 쓰면 그 구조를 되돌릴 수 없다. 그래서 **원장 사이트 튜플**을 키로 둔다.

    site = (model, phase, raw_op_id, field, shape_index, axis)

발행 쪽으로 내려가는 것은 순서 4 다: `published_overlay` 는 그 셀의 **모든** raw 사이트가
`(verdict, resulting_expr, grade_after, answer_id)` 에서 일치할 때만 만든다.

## 상태 -- `no_name` 과 `cannot_determine` 는 **actionable 이 아니다**

```
pending_first_pass            답이 아직 없다
proposed_confirm              1차가 현재 라벨과 같은 이름을 냈다 (same / alias)
proposed_rename               1차가 다른 이름을 냈다 (different)
pending_no_name_adjudication  1차가 no_name 을 냈다 -- 라벨도 grade 도 **바꾸지 않는다**
pending_cannot_determine      1차가 판정을 보류했거나 비교가 불가했다
conflict_duplicate            중복 배정 둘이 어긋난다 -> 2차
pending_human                 사람 판정 대기 (위 상태들이 여기로 모인다)
approved_confirm              사람이 승인했다 -> actionable
approved_rename               사람이 승인했다 -> actionable
approved_no_name_exists       사람이 승인했다 -> actionable (정수 축 복원)
withdrawn                     철회됐다
```

**actionable 은 `approved_*` 뿐이다.** 그 밖의 상태에서는 `resulting_expr` 도
`grade_after` 도 쓰지 않는다(`None`).

## 입력 계약 -- 여기서 막는다

overlay 후보를 만들기 전에 수집 산출물이 온전한지 **다시** 본다. `accepted.jsonl` 만
읽는 것으로는 부족하다(외부 검토 2026-09-26).

```
_ingest.json     errors == 0 / rejected == 0 / strict == true / dirty_build == false
                 input_sha256 를 **다시 계산해** 대조
accepted.jsonl   모든 행이 eligible_for_adjudication == true
                 packet_quarantined == false
                 assignment_revision 이 현재 배정과 일치
```

하나라도 어긋나면 후보를 만들지 않는다.
"""
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import _buildguard                                               # noqa: E402

SCHEMA_VERSION = 1

# ---- 상태
STATES = (
    "pending_first_pass",
    "proposed_confirm",
    "proposed_rename",
    "pending_no_name_adjudication",
    "pending_cannot_determine",
    "conflict_duplicate",
    "pending_human",
    "approved_confirm",
    "approved_rename",
    "approved_no_name_exists",
    "withdrawn",
)
# **이것만 적용할 수 있다.** 사람 승인을 거친 상태다.
ACTIONABLE = ("approved_confirm", "approved_rename", "approved_no_name_exists")
# 사람 판정을 기다리는 상태
PENDING = ("pending_first_pass", "proposed_confirm", "proposed_rename",
           "pending_no_name_adjudication", "pending_cannot_determine",
           "conflict_duplicate", "pending_human")

# ---- verdict: 사람 승인 뒤에만 actionable 이 된다
VERDICTS = ("confirm", "rename", "no_name_exists", "undetermined")
GRADES_AFTER = ("confirmed", "unresolved", None)

CANDIDATE_FIELDS = (
    "schema_version", "site", "model", "phase", "decision_unit_id",
    "question_family_id", "answer_id", "submission_sha256", "packet_id",
    "session_id", "assignment_revision", "role",
    "proposal", "proposed_expr", "current_expr", "comparison",
    "comparison_reason", "verdict_candidate", "state", "actionable",
    "resulting_expr", "grade_after", "grade_before", "origin",
    "published_cell", "evidence", "assumptions", "confidence",
)


class SchemaError(ValueError):
    """후보 레코드가 계약을 어겼다."""


def state_of(proposal, comparison):
    """1 차 답 하나에서 나오는 상태. **actionable 을 만들지 않는다.**

    `no_name` 과 `cannot_determine` 은 verdict 가 아니라 대기 상태다. 1 차만으로
    라벨이나 grade 를 바꾸지 않는다(승인된 Q2).
    """
    if proposal == "no_name":
        return "pending_no_name_adjudication", "no_name_exists_candidate"
    if proposal == "cannot_determine":
        return "pending_cannot_determine", "undetermined"
    if proposal != "named":
        raise SchemaError(f"모르는 proposal {proposal!r}")
    if comparison in ("same", "alias"):
        return "proposed_confirm", "confirm"
    if comparison == "different":
        return "proposed_rename", "rename"
    # 비교가 불가하면(모르는 심볼·나눗셈·호출) 판정을 만들지 않는다
    return "pending_cannot_determine", "undetermined"


def validate(rec):
    """후보 레코드 하나를 검사한다. 문제 목록을 돌려준다(빈 목록이면 통과)."""
    errs = []
    if rec.get("schema_version") != SCHEMA_VERSION:
        errs.append(f"schema_version 이 {rec.get('schema_version')} 다")
    unknown = [k for k in rec if k not in CANDIDATE_FIELDS]
    if unknown:
        errs.append(f"모르는 필드 {sorted(unknown)}")
    site = rec.get("site")
    if (not isinstance(site, list) or len(site) != 6
            or not isinstance(site[2], int) or not isinstance(site[4], int)
            or not isinstance(site[5], int) or site[3] not in ("i", "o", "w")):
        errs.append(f"site 형식이 틀렸다 {site!r}")
    if rec.get("state") not in STATES:
        errs.append(f"모르는 state {rec.get('state')!r}")
    if rec.get("verdict_candidate") not in VERDICTS:
        errs.append(f"모르는 verdict {rec.get('verdict_candidate')!r}")
    if rec.get("grade_after") not in GRADES_AFTER:
        errs.append(f"모르는 grade_after {rec.get('grade_after')!r}")
    act = rec.get("actionable")
    if act is not (rec.get("state") in ACTIONABLE):
        errs.append(f"actionable 이 state 와 맞지 않는다 "
                    f"({act} / {rec.get('state')})")
    # **actionable 이 아니면 결과를 쓰지 않는다**
    if not act:
        for k in ("resulting_expr", "grade_after"):
            if rec.get(k) is not None:
                errs.append(f"actionable 이 아닌데 `{k}` 가 있다 ({rec.get(k)!r})")
    else:
        if rec["state"] == "approved_rename" and not rec.get("resulting_expr"):
            errs.append("approved_rename 인데 resulting_expr 가 없다")
        if rec["state"] == "approved_no_name_exists" and rec.get("resulting_expr"):
            errs.append("approved_no_name_exists 인데 resulting_expr 가 있다")
        if rec.get("grade_after") is None:
            errs.append("actionable 인데 grade_after 가 없다")
    for k in ("decision_unit_id", "answer_id", "packet_id", "session_id",
              "model", "phase"):
        if not rec.get(k):
            errs.append(f"`{k}` 가 없다")
    if rec.get("proposal") == "named" and not rec.get("proposed_expr"):
        errs.append("named 인데 proposed_expr 가 없다")
    if rec.get("proposal") != "named" and rec.get("proposed_expr"):
        errs.append("named 이 아닌데 proposed_expr 가 있다")
    return errs


def check_input_contract(ingest_dir, assignment_revision):
    """**수집 산출물이 온전한가.** 문제 목록을 돌려준다(빈 목록이면 통과).

    `accepted.jsonl` 만 읽는 것으로는 부족하다는 지적을 그대로 구현한다
    (외부 검토 2026-09-26). 입력 해시는 **다시 계산해** 대조한다.
    """
    errs = []
    mp = os.path.join(ingest_dir, "_ingest.json")
    ap = os.path.join(ingest_dir, "accepted.jsonl")
    if not os.path.exists(mp):
        return [f"_ingest.json 이 없다 ({mp})"], []
    if not os.path.exists(ap):
        return [f"accepted.jsonl 이 없다 ({ap})"], []
    meta = json.load(io.open(mp, encoding="utf-8"))
    for k, want in (("errors", 0), ("rejected", 0), ("strict", True),
                    ("dirty_build", False)):
        if meta.get(k) != want:
            errs.append(f"_ingest.json 의 {k} 가 {meta.get(k)!r} 다 "
                        f"({want!r} 이어야 한다)")
    # 입력 해시를 다시 계산한다
    recorded = meta.get("input_sha256") or {}
    if not recorded:
        errs.append("_ingest.json 에 input_sha256 가 없다")
    for rel, info in sorted(recorded.items()):
        p = os.path.join(PROJ, rel)
        real = _buildguard.sha256_file(p)
        if real is None:
            errs.append(f"입력이 없어졌다 {rel}")
        elif real != info.get("sha256"):
            errs.append(f"입력이 바뀌었다 {rel} "
                        f"(기록 {str(info.get('sha256'))[:12]}… 실제 {real[:12]}…)")
    rows = []
    for i, line in enumerate(io.open(ap, encoding="utf-8"), 1):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("eligible_for_adjudication") is not True:
            errs.append(f"accepted.jsonl {i}: eligible_for_adjudication 이 "
                        f"{r.get('eligible_for_adjudication')!r} 다")
        if r.get("packet_quarantined") is not False:
            errs.append(f"accepted.jsonl {i}: packet_quarantined 가 "
                        f"{r.get('packet_quarantined')!r} 다")
        if r.get("assignment_revision") != assignment_revision:
            errs.append(f"accepted.jsonl {i}: assignment_revision 이 "
                        f"{r.get('assignment_revision')!r} 다 "
                        f"({assignment_revision!r} 이어야 한다)")
        rows.append(r)
    return errs, rows


def state_table():
    """문서에 실을 상태 표."""
    out = [("상태", "actionable", "뜻")]
    mean = {
        "pending_first_pass": "답이 아직 없다",
        "proposed_confirm": "1차가 현재 라벨과 같은 이름 (same/alias)",
        "proposed_rename": "1차가 다른 이름 (different)",
        "pending_no_name_adjudication": "1차가 no_name -- 라벨·grade 를 바꾸지 않는다",
        "pending_cannot_determine": "판정 보류 또는 비교 불가",
        "conflict_duplicate": "중복 배정 둘이 어긋난다 -> 2차",
        "pending_human": "사람 판정 대기",
        "approved_confirm": "사람이 승인 -- 현재 라벨 확정",
        "approved_rename": "사람이 승인 -- 이름 교체",
        "approved_no_name_exists": "사람이 승인 -- 정수 축으로 복원",
        "withdrawn": "철회",
    }
    for s in STATES:
        out.append((s, "O" if s in ACTIONABLE else "-", mean[s]))
    return out


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")
    print(f"schema_version {SCHEMA_VERSION}")
    w = max(len(r[0]) for r in state_table())
    for name, act, mean in state_table():
        print(f"  {name:<{w}}  {act:^10}  {mean}")
    print()
    print(f"actionable 은 {len(ACTIONABLE)} 개뿐이고 전부 사람 승인을 거친다.")
