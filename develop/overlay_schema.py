r"""0-c 순서 1: overlay 레코드 **세 종류**와 판정 상태. 아직 적용하지 않는다.

이 모듈은 형식과 상태만 정의한다. 실제 적용(`published_overlay` 파생, csv/jsonl 수정)은
**사람 판정이 끝난 뒤**에만 하고, 그 단계는 별도 승인이 필요하다.

## 한 스키마가 세 역할을 맡으면 모순 조합이 통과한다

처음에는 후보·조정·최종을 한 스키마로 묶었다. 그래서 다음이 검사를 통과했다
(외부 검토 2026-09-26 이 재현했다).

    state=approved_confirm / verdict=rename / grade_after=unresolved   -> 통과 (모순)
    state=pending_first_pass 인데 answer_id·packet_id 를 요구          -> 답이 없는데 필수

그리고 `no_name` 후보가 **실제로는 스키마를 통과하지 못했다**: `state_of()` 가
`no_name_exists_candidate` 를 냈는데 `VERDICTS` 에 그 값이 없었다. 시험이 `state_of()` 의
반환값만 보고 `validate()` 에 넣지 않아 놓쳤다.

그래서 종류를 나눈다.

```
answer_candidate   답 하나에서 나온 **불변** 후보. 절대 actionable 이 아니다
                   키: (site, answer_id)      candidate_kind 만 있고 final_verdict 는 없다
unit_adjudication  판정 단위별 조정. 여러 답을 모은다(`source_answer_ids`)
                   승인 상태에서만 final_verdict·resulting_expr·grade_after 를 갖는다
raw_overlay        사이트별 **최종** 결과. 사람 승인 참조(`approval_id`)가 있어야 한다
```

## 상태

`answer_candidate` 는 1 차 답에서 나오는 넷뿐이다.

```
proposed_confirm              1차가 현재 라벨과 같은 이름 (same / alias)
proposed_rename               1차가 다른 이름 (different)
pending_no_name_adjudication  1차가 no_name -- 라벨도 grade 도 바꾸지 않는다
pending_cannot_determine      판정 보류 또는 비교 불가
```

`unit_adjudication` 은 여기에 조정·승인 상태를 더한다.

```
pending_first_pass        답이 아직 없다 (answer 관련 필드가 **없어야** 한다)
response_agreement        중복 배정 둘이 같은 답 -- **판정 합의가 아니다**
conflict_duplicate        중복 배정 둘이 어긋난다 -> 2차
pending_human             사람 판정 대기
approved_confirm          사람 승인. final_verdict=confirm,       grade_after=confirmed
                                     resulting_expr **null** -- `confirm` 의 뜻이
                                     "현재 라벨 유지" 이므로 적용기가 추측하는 것이
                                     아니다. 같은 결정을 두 방식으로 직렬화하면
                                     digest·충돌 비교에 불필요한 차이가 생긴다
approved_rename           사람 승인. final_verdict=rename,        grade_after=confirmed
                                     resulting_expr **필수**
approved_no_name_exists   사람 승인. final_verdict=no_name_exists, grade_after=confirmed_no_name
                                     resulting_expr **null** (정수 축으로 복원)
withdrawn                 철회
```

**actionable 은 `approved_*` 셋뿐이다.** 비승인 상태에서는 `final_verdict`·
`resulting_expr`·`grade_after` 가 **모두 null** 이어야 한다.

## 값이 enum 에 속하는지만 보면 모순이 통과한다

각 필드가 enum 에 있는지만 검사했더니 다음이 전부 통과했다(외부 검토 2026-09-26).

    state=proposed_rename / candidate_kind=no_name_exists /
      proposal=named / comparison=same          -> 통과 (서로 모순)
    site=[m, prefill, ...] / model=other / phase=decode  -> 통과
    state=pending_first_pass / answers=[...]             -> 통과
    state=response_agreement / response_agreement=False  -> 통과
    semantic_resolution="no"  (문자열)                   -> 통과
    비승인 상태 / approval_id=AP-1                       -> 통과

그래서

* `answer_candidate` 는 `state_of(proposal, comparison)` 을 **다시 계산해** `state` 와
  `candidate_kind` 가 정확히 같은지 본다
* `site[:2] == [model, phase]` 를 `answer_candidate` 와 `raw_overlay` 에 강제한다
* 상태별 **필수·금지 필드 표**(`STATE_FIELDS`)를 하나 두고 정확히 대조한다
* `semantic_resolution` 은 엄격한 bool 이다

## 입력 계약 -- 여기서 막는다

overlay 후보를 만들기 전에 수집 산출물이 온전한지 **다시** 본다.

```
_ingest.json     errors 0 / rejected 0 / strict true / dirty_build false
                 built_from_commit 이 40 자리 hex / generator_sha256 존재
                 input_sha256 를 **다시 계산해** 대조
                 accepted_sha256 · accepted_rows 를 **다시 계산해** 대조
accepted.jsonl   모든 행이 eligible_for_adjudication true / packet_quarantined false
                 assignment_revision 일치
```

하나라도 어긋나면 후보를 만들지 않는다.
"""
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import _buildguard                                               # noqa: E402

SCHEMA_VERSION = 2
KINDS = ("answer_candidate", "unit_adjudication", "raw_overlay")

# ---- 상태
ANSWER_STATES = ("proposed_confirm", "proposed_rename",
                 "pending_no_name_adjudication", "pending_cannot_determine")
APPROVED_STATES = ("approved_confirm", "approved_rename",
                   "approved_no_name_exists")
UNIT_STATES = (("pending_first_pass", "response_agreement",
                "conflict_duplicate", "pending_human")
               + ANSWER_STATES + APPROVED_STATES + ("withdrawn",))
ACTIONABLE = APPROVED_STATES

CANDIDATE_KINDS = ("confirm", "rename", "no_name_exists", "undetermined")
FINAL_VERDICTS = ("confirm", "rename", "no_name_exists")
GRADES_AFTER = ("confirmed", "confirmed_no_name")

# 승인 상태마다 **정확히** 무엇이어야 하는가. 조합을 여기 한 곳에 못 박는다.
APPROVED_RULES = {
    "approved_confirm": {"final_verdict": "confirm",
                         "grade_after": "confirmed",
                         "resulting_expr": "null"},
    "approved_rename": {"final_verdict": "rename",
                        "grade_after": "confirmed",
                        "resulting_expr": "required"},
    "approved_no_name_exists": {"final_verdict": "no_name_exists",
                                "grade_after": "confirmed_no_name",
                                "resulting_expr": "null"},
}

_HEX40 = re.compile(r"^[0-9a-f]{40}$")

# 승인 참조. **비승인 상태에서는 전부 금지다.**
APPROVAL_FIELDS = ("approval_id", "adjudicated_by", "adjudicated_at")
AGREEMENT_FIELDS = ("response_agreement", "semantic_resolution")
ANSWER_LINK_FIELDS = ("source_answer_ids", "answers")

# `unit_adjudication` 의 상태별 계약: (필수, 금지).
# 여기 적히지 않은 상태는 `_DEFAULT_UNIT_FIELDS` 를 쓴다.
_DEFAULT_UNIT_FIELDS = (("source_answer_ids",),
                        AGREEMENT_FIELDS + APPROVAL_FIELDS
                        + ("final_verdict", "resulting_expr", "grade_after"))
STATE_FIELDS = {
    # 답이 **없는** 상태. answer·agreement·승인 필드를 모두 금지한다.
    "pending_first_pass": ((), ANSWER_LINK_FIELDS + AGREEMENT_FIELDS
                           + APPROVAL_FIELDS
                           + ("final_verdict", "resulting_expr",
                              "grade_after")),
    # 중복 응답이 같다 -- 판정 합의는 아니므로 semantic_resolution 을 요구한다
    "response_agreement": (("source_answer_ids", "response_agreement",
                            "semantic_resolution"),
                           APPROVAL_FIELDS + ("final_verdict",
                                              "resulting_expr",
                                              "grade_after")),
    "conflict_duplicate": (("source_answer_ids", "response_agreement"),
                           APPROVAL_FIELDS + ("final_verdict",
                                              "resulting_expr",
                                              "grade_after")),
}
for _s in APPROVED_STATES:
    STATE_FIELDS[_s] = (("source_answer_ids",) + APPROVAL_FIELDS, ())

COMMON = ("schema_version", "kind", "model", "phase", "decision_unit_id",
          "question_family_id", "current_expr", "grade_before", "state")
ANSWER_FIELDS = COMMON + (
    "site", "answer_id", "submission_sha256", "packet_id", "session_id",
    "assignment_revision", "role", "proposal", "proposed_expr", "comparison",
    "comparison_reason", "candidate_kind", "actionable", "origin",
    "published_cell", "evidence", "assumptions", "confidence")
UNIT_FIELDS = COMMON + (
    "source_answer_ids", "answers", "response_agreement", "semantic_resolution",
    "final_verdict", "resulting_expr", "grade_after", "actionable",
    "approval_id", "adjudicated_by", "adjudicated_at", "note")
OVERLAY_FIELDS = COMMON + (
    "site", "final_verdict", "resulting_expr", "grade_after", "actionable",
    "approval_id", "unit_adjudication_state", "origin", "published_cell",
    "source_answer_ids")


class SchemaError(ValueError):
    """레코드가 계약을 어겼다."""


def state_of(proposal, comparison):
    """1 차 답 하나에서 나오는 `(state, candidate_kind)`.

    **`approved_*` 를 만들 경로가 없다.** `no_name` 과 `cannot_determine` 은 verdict 가
    아니라 대기 상태다(승인된 Q2).
    """
    if proposal == "no_name":
        return "pending_no_name_adjudication", "no_name_exists"
    if proposal == "cannot_determine":
        return "pending_cannot_determine", "undetermined"
    if proposal != "named":
        raise SchemaError(f"모르는 proposal {proposal!r}")
    if comparison in ("same", "alias"):
        return "proposed_confirm", "confirm"
    if comparison == "different":
        return "proposed_rename", "rename"
    return "pending_cannot_determine", "undetermined"


def _site_errs(site):
    if (not isinstance(site, list) or len(site) != 6
            or not isinstance(site[2], int) or not isinstance(site[4], int)
            or not isinstance(site[5], int) or site[3] not in ("i", "o", "w")
            or not isinstance(site[0], str) or not isinstance(site[1], str)):
        return [f"site 형식이 틀렸다 {site!r}"]
    return []


def _unknown(rec, allowed):
    bad = [k for k in rec if k not in allowed]
    return [f"모르는 필드 {sorted(bad)}"] if bad else []


def _require(rec, keys):
    return [f"`{k}` 가 없다" for k in keys
            if rec.get(k) in (None, "", [], {})]


def _forbid_results(rec, keys=("final_verdict", "resulting_expr",
                               "grade_after")):
    """비승인 상태에서는 결과를 쓸 수 없다."""
    return [f"actionable 이 아닌데 `{k}` 가 있다 ({rec.get(k)!r})"
            for k in keys if rec.get(k) is not None]


def _forbid(rec, keys, why):
    return [f"{why} `{k}` 를 쓸 수 없다 ({rec.get(k)!r})"
            for k in keys if rec.get(k) is not None]


def _site_binds(rec):
    """`site` 의 model·phase 가 레코드와 같은가. 다르면 어느 쪽을 믿을지 알 수 없다."""
    site = rec.get("site")
    if not isinstance(site, list) or len(site) < 2:
        return []
    if list(site[:2]) != [rec.get("model"), rec.get("phase")]:
        return [f"site 의 model·phase 가 레코드와 다르다 "
                f"({site[:2]} != {[rec.get('model'), rec.get('phase')]})"]
    return []


def _approved_errs(rec):
    """승인 상태의 조합을 규칙표와 **정확히** 대조한다."""
    st = rec["state"]
    rule = APPROVED_RULES[st]
    errs = []
    if rec.get("final_verdict") != rule["final_verdict"]:
        errs.append(f"{st} 는 final_verdict 가 {rule['final_verdict']!r} 여야 한다 "
                    f"(받은 값 {rec.get('final_verdict')!r})")
    if rec.get("grade_after") != rule["grade_after"]:
        errs.append(f"{st} 는 grade_after 가 {rule['grade_after']!r} 여야 한다 "
                    f"(받은 값 {rec.get('grade_after')!r})")
    want = rule["resulting_expr"]
    got = rec.get("resulting_expr")
    if want == "required" and not got:
        errs.append(f"{st} 는 resulting_expr 가 필요하다")
    elif want == "null" and got is not None:
        errs.append(f"{st} 는 resulting_expr 가 null 이어야 한다 ({got!r})")
    elif want == "current_or_null" and got is not None:
        if got != rec.get("current_expr"):
            errs.append(f"{st} 의 resulting_expr 는 현재 라벨과 같아야 한다 "
                        f"({got!r} != {rec.get('current_expr')!r})")
    if not rec.get("approval_id"):
        errs.append(f"{st} 인데 approval_id 가 없다 -- 사람 승인 참조가 필요하다")
    return errs


def validate(rec):
    """레코드 하나를 **종류에 맞게** 검사한다. 문제 목록(빈 목록이면 통과)."""
    errs = []
    if rec.get("schema_version") != SCHEMA_VERSION:
        errs.append(f"schema_version 이 {rec.get('schema_version')!r} 다")
    kind = rec.get("kind")
    if kind not in KINDS:
        return errs + [f"모르는 kind {kind!r}"]
    act = rec.get("actionable")

    if kind == "answer_candidate":
        errs += _unknown(rec, ANSWER_FIELDS)
        errs += _site_errs(rec.get("site"))
        errs += _site_binds(rec)
        if rec.get("state") not in ANSWER_STATES:
            errs.append(f"answer_candidate 의 state 가 {rec.get('state')!r} 다 "
                        f"(허용 {list(ANSWER_STATES)})")
        if rec.get("candidate_kind") not in CANDIDATE_KINDS:
            errs.append(f"모르는 candidate_kind {rec.get('candidate_kind')!r}")
        # **의미를 다시 계산해 대조한다.** enum 검사만으로는 서로 모순된 조합이 통과한다.
        try:
            want_st, want_ck = state_of(rec.get("proposal"),
                                        rec.get("comparison"))
        except SchemaError as e:
            errs.append(str(e))
        else:
            if rec.get("state") != want_st:
                errs.append(f"state 가 proposal·comparison 과 맞지 않는다 "
                            f"({rec.get('state')!r}, 기대 {want_st!r})")
            if rec.get("candidate_kind") != want_ck:
                errs.append(f"candidate_kind 가 맞지 않는다 "
                            f"({rec.get('candidate_kind')!r}, 기대 {want_ck!r})")
        errs += _forbid(rec, APPROVAL_FIELDS, "answer_candidate 는")
        if act is not False:
            errs.append("answer_candidate 는 **절대 actionable 이 아니다**")
        errs += _forbid_results(rec)
        errs += _require(rec, ("decision_unit_id", "answer_id", "packet_id",
                               "session_id", "model", "phase",
                               "submission_sha256"))
        if rec.get("proposal") == "named" and not rec.get("proposed_expr"):
            errs.append("named 인데 proposed_expr 가 없다")
        if rec.get("proposal") != "named" and rec.get("proposed_expr"):
            errs.append("named 이 아닌데 proposed_expr 가 있다")
        return errs

    if kind == "unit_adjudication":
        errs += _unknown(rec, UNIT_FIELDS)
        st = rec.get("state")
        if st not in UNIT_STATES:
            return errs + [f"모르는 state {st!r}"]
        if act is not (st in ACTIONABLE):
            errs.append(f"actionable 이 state 와 맞지 않는다 ({act!r} / {st})")
        errs += _require(rec, ("decision_unit_id", "model", "phase"))
        # ---- 상태별 **필수·금지 필드 표**로 대조한다
        need, forbid = STATE_FIELDS.get(st, _DEFAULT_UNIT_FIELDS)
        errs += _require(rec, need)
        errs += _forbid(rec, forbid, f"{st} 에서는")
        if st in ACTIONABLE:
            errs += _approved_errs(rec)
        # `semantic_resolution` 은 **엄격한 bool** 이다 ("no" 같은 문자열을 막는다)
        for k in AGREEMENT_FIELDS:
            if k in rec and rec[k] is not None and not isinstance(rec[k], bool):
                errs.append(f"`{k}` 는 bool 이어야 한다 ({rec[k]!r})")
        if st == "response_agreement" and rec.get("response_agreement") is not True:
            errs.append("response_agreement 상태인데 response_agreement 가 참이 아니다")
        if st == "conflict_duplicate" and rec.get("response_agreement") is not False:
            errs.append("conflict_duplicate 인데 response_agreement 가 거짓이 아니다")
        if "answers" in rec and rec["answers"] is not None:
            if not isinstance(rec["answers"], list):
                errs.append("`answers` 가 목록이 아니다")
        return errs

    # raw_overlay -- 최종. 사람 승인 참조가 있어야 한다.
    errs += _unknown(rec, OVERLAY_FIELDS)
    errs += _site_errs(rec.get("site"))
    errs += _site_binds(rec)
    if rec.get("state") not in ACTIONABLE:
        errs.append(f"raw_overlay 는 승인 상태여야 한다 ({rec.get('state')!r})")
        return errs
    if act is not True:
        errs.append("raw_overlay 는 actionable 이어야 한다")
    errs += _approved_errs(rec)
    errs += _require(rec, ("decision_unit_id", "model", "phase",
                           "source_answer_ids", "unit_adjudication_state"))
    if rec.get("unit_adjudication_state") != rec.get("state"):
        errs.append("unit_adjudication_state 가 state 와 다르다")
    return errs


def check_input_contract(ingest_dir, assignment_revision):
    """**수집 산출물이 온전한가.** `(문제 목록, 행)`.

    `accepted.jsonl` 만 읽는 것으로는 부족하다 -- 그 파일의 내용이 `_ingest.json` 에
    결박돼 있지 않으면 proposal·comparison 을 바꿔도 통과한다(외부 검토 2026-09-26).
    그래서 `accepted_sha256`·`accepted_rows` 를 **다시 계산해** 대조한다.
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
    commit = meta.get("built_from_commit") or ""
    if not _HEX40.match(commit):
        errs.append(f"built_from_commit 이 40 자리 hex 가 아니다 ({commit!r})")
    if not (meta.get("generator_sha256") or {}):
        errs.append("generator_sha256 가 없다")
    # ---- accepted.jsonl 을 metadata 에 결박
    real = _buildguard.sha256_file(ap)
    if meta.get("accepted_sha256") is None:
        errs.append("_ingest.json 에 accepted_sha256 가 없다")
    elif meta["accepted_sha256"] != real:
        errs.append(f"accepted.jsonl 이 바뀌었다 "
                    f"(기록 {str(meta['accepted_sha256'])[:12]}… 실제 {real[:12]}…)")
    # ---- 입력 해시를 다시 계산
    recorded = meta.get("input_sha256") or {}
    if not recorded:
        errs.append("_ingest.json 에 input_sha256 가 없다")
    for rel, info in sorted(recorded.items()):
        p = os.path.join(PROJ, rel)
        got = _buildguard.sha256_file(p)
        if got is None:
            errs.append(f"입력이 없어졌다 {rel}")
        elif got != info.get("sha256"):
            errs.append(f"입력이 바뀌었다 {rel} "
                        f"(기록 {str(info.get('sha256'))[:12]}… 실제 {got[:12]}…)")
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
    if meta.get("accepted_rows") is None:
        errs.append("_ingest.json 에 accepted_rows 가 없다")
    elif meta["accepted_rows"] != len(rows):
        errs.append(f"accepted_rows 가 {meta['accepted_rows']} 인데 실제 {len(rows)} 행")
    return errs, rows


def verify_against_manifest(paths, manifest_hashes, label=""):
    """쓰기 **전에** 신뢰 해시와 대조한다. 문제 목록.

    후보 생성기가 units·crosswalk 를 읽고 나서 그 파일의 해시를 **사후에** 기록하면,
    바뀐 자료로 만들고 바뀐 해시를 적을 뿐이다(외부 검토 2026-09-26). 배정 manifest 에
    고정된 해시와 사전 대조한다.
    """
    errs = []
    for p in paths:
        rel = os.path.relpath(p, PROJ).replace(os.sep, "/")
        want = (manifest_hashes.get(rel) or {}).get("sha256")
        if want is None:
            errs.append(f"{label}신뢰 해시에 없는 입력 {rel}")
            continue
        got = _buildguard.sha256_file(p)
        if got != want:
            errs.append(f"{label}입력이 배정 manifest 와 다르다 {rel} "
                        f"(기록 {str(want)[:12]}… 실제 {str(got)[:12]}…)")
    return errs


def state_table():
    mean = {
        "pending_first_pass": "답이 아직 없다 (answer 필드가 없어야 한다)",
        "proposed_confirm": "1차가 현재 라벨과 같은 이름 (same/alias)",
        "proposed_rename": "1차가 다른 이름 (different)",
        "pending_no_name_adjudication": "1차가 no_name -- 라벨·grade 를 바꾸지 않는다",
        "pending_cannot_determine": "판정 보류 또는 비교 불가",
        "response_agreement": "중복 둘이 같은 답 -- **판정 합의가 아니다**",
        "conflict_duplicate": "중복 둘이 어긋난다 -> 2차",
        "pending_human": "사람 판정 대기",
        "approved_confirm": "승인: confirm / confirmed / expr null",
        "approved_rename": "승인: rename / confirmed / resulting_expr 필수",
        "approved_no_name_exists": "승인: no_name_exists / confirmed_no_name / expr null",
        "withdrawn": "철회",
    }
    return [(s, "O" if s in ACTIONABLE else "-", mean[s]) for s in UNIT_STATES]


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")
    print(f"schema_version {SCHEMA_VERSION}   레코드 종류 {list(KINDS)}")
    print()
    w = max(len(r[0]) for r in state_table())
    for name, act, mean in state_table():
        mark = " (답 후보)" if name in ANSWER_STATES else ""
        print(f"  {name:<{w}}  {act:^10}  {mean}{mark}")
    print()
    print(f"actionable 은 {len(ACTIONABLE)} 개뿐이고 전부 사람 승인 + approval_id 를 요구한다.")
