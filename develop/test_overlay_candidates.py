r"""overlay 스키마 세 종류와 입력 계약이 **실제로 막는가.** 실패 주입으로 확인한다.

외부 검토가 요구한 완료 조건: 입력 계약을 실제 코드와 실패 주입 검사로 포함할 것
(2026-09-26). 그리고 지난번에 놓친 것 -- **`no_name` 후보를 `state_of()` 에서만 보지
말고 `validate()` 와 `build()` 에 실제로 통과시킬 것.** 그것을 안 해서
`no_name_exists_candidate` 가 `VERDICTS` 에 없는 것을 못 봤다.

실제 대장·패킷·산출물은 건드리지 않는다(임시 디렉터리에서 돈다).

실행:
    .venv\Scripts\python.exe develop\test_overlay_candidates.py
"""
import io
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import build_overlay_candidates as B                             # noqa: E402
import export_shard_packet as X                                  # noqa: E402
import ingest_answers as I                                       # noqa: E402
import overlay_schema as S                                       # noqa: E402

OK, FAIL = [], []
NL = chr(10)


def check(name, cond):
    (OK if cond else FAIL).append(name)
    print(("  OK   " if cond else "  FAIL ") + name)


def dirty_env(fn, *a, **kw):
    """`require_clean_tree` 가드만 열고 부른다. 임시 디렉터리에만 쓴다."""
    old = os.environ.get("ALLOW_DIRTY_BUILD")
    os.environ["ALLOW_DIRTY_BUILD"] = "1"
    try:
        return fn(*a, **kw)
    finally:
        if old is None:
            os.environ.pop("ALLOW_DIRTY_BUILD", None)
        else:
            os.environ["ALLOW_DIRTY_BUILD"] = old


def run(mod, *argv):
    old = sys.argv
    sys.argv = [mod.__name__, *argv]
    try:
        return dirty_env(mod.main) or 0
    except SystemExit as e:
        return e.code
    finally:
        sys.argv = old


print("1) 상태 -- actionable 은 사람 승인 뒤에만")
check("레코드 종류가 셋", S.KINDS == ("answer_candidate", "unit_adjudication",
                                    "raw_overlay"))
check("답 후보 상태는 넷", len(S.ANSWER_STATES) == 4)
check("actionable 이 셋", len(S.ACTIONABLE) == 3)
check("actionable 은 전부 approved_",
      all(s.startswith("approved_") for s in S.ACTIONABLE))
check("**답 후보 상태에 actionable 이 없다**",
      not (set(S.ANSWER_STATES) & set(S.ACTIONABLE)))
check("승인 규칙표가 actionable 과 정확히 같다",
      set(S.APPROVED_RULES) == set(S.ACTIONABLE))
check("grade_after 에 confirmed_no_name 이 있다",
      "confirmed_no_name" in S.GRADES_AFTER)
check("no_name 승인은 confirmed_no_name 이다",
      S.APPROVED_RULES["approved_no_name_exists"]["grade_after"]
      == "confirmed_no_name")

print()
print("2) 1 차 답 -> 상태·후보 종류. approved 를 만들 경로가 없다")
for prop, comp, want in (("named", "same", "proposed_confirm"),
                         ("named", "alias", "proposed_confirm"),
                         ("named", "different", "proposed_rename"),
                         ("named", "cannot_determine",
                          "pending_cannot_determine"),
                         ("named", None, "pending_cannot_determine"),
                         ("no_name", None, "pending_no_name_adjudication"),
                         ("cannot_determine", None,
                          "pending_cannot_determine")):
    st, ck = S.state_of(prop, comp)
    check(f"{prop}/{comp} -> {want}", st == want)
    check(f"   actionable 이 아니다", st not in S.ACTIONABLE)
    check(f"   candidate_kind 가 유효하다 ({ck})", ck in S.CANDIDATE_KINDS)
def rejects(p, c, why):
    try:
        S.state_of(p, c)
        check(f"**{why}**", False)
    except S.SchemaError:
        check(f"**{why}**", True)


rejects("maybe", None, "모르는 proposal 을 거부")
# 도메인을 강제하지 않으면 마지막 포괄 반환이 임의 문자열을 보류로 흡수한다
rejects("named", "garbage", "named + 모르는 comparison 을 거부")
rejects("no_name", "different", "no_name 에 comparison 이 붙으면 거부")
rejects("cannot_determine", "same", "cannot_determine 에 comparison 이 붙으면 거부")
rejects("named", 3, "comparison 이 문자열이 아니면 거부")
check("no_name + comparison=None 은 통과",
      S.state_of("no_name", None)[0] == "pending_no_name_adjudication")
check("named + comparison=None 은 보류",
      S.state_of("named", None)[0] == "pending_cannot_determine")

print()
print("3) answer_candidate -- **no_name 도 실제로 통과해야 한다**")


def ac(**kw):
    st, ck = S.state_of(kw.pop("_proposal", "named"), kw.pop("_comp", "same"))
    r = {"schema_version": S.SCHEMA_VERSION, "kind": "answer_candidate",
         "site": ["m", "prefill", 12, "i", 0, 1],
         "model": "m", "phase": "prefill", "decision_unit_id": "u1",
         "answer_id": "A-1", "submission_sha256": "0" * 64,
         "packet_id": "P-1", "session_id": "S-1",
         "proposal": "named", "proposed_expr": "d_head",
         "current_expr": "d_head", "comparison": "same",
         "candidate_kind": ck, "state": st, "actionable": False}
    r.update(kw)
    return r


check("named/same 후보가 통과", S.validate(ac()) == [])
no_name = ac(_proposal="no_name", _comp=None, proposal="no_name",
             proposed_expr=None, comparison=None)
check("**no_name 후보가 통과한다** (지난번 여기서 막혔다)",
      S.validate(no_name) == [])
cd = ac(_proposal="cannot_determine", _comp=None, proposal="cannot_determine",
        proposed_expr=None, comparison=None)
check("cannot_determine 후보가 통과", S.validate(cd) == [])
rn = ac(_comp="different", comparison="different", proposed_expr="d_qk")
check("rename 후보가 통과", S.validate(rn) == [])
check("**actionable=True 를 거부**",
      any("actionable" in e for e in S.validate(ac(actionable=True))))
for k in ("final_verdict", "resulting_expr", "grade_after"):
    check(f"**결과 필드 `{k}` 를 거부**",
          any("모르는 필드" in e or k in e for e in S.validate(ac(**{k: "x"}))))
check("승인 상태를 거부",
      any("state" in e for e in S.validate(ac(state="approved_confirm"))))
print("   -- 외부 검토가 재현한 의미 모순")
check("**state/candidate_kind 가 proposal·comparison 과 모순이면 거부**",
      any("맞지 않는다" in e for e in
          S.validate(ac(state="proposed_rename",
                        candidate_kind="no_name_exists"))))
check("candidate_kind 만 어긋나도 거부",
      any("candidate_kind" in e for e in
          S.validate(ac(candidate_kind="rename"))))
check("comparison 을 바꾸면 state 와 어긋나 거부",
      any("맞지 않는다" in e for e in S.validate(ac(comparison="different"))))
check("**site 의 model·phase 가 레코드와 다르면 거부**",
      any("site 의 model" in e for e in
          S.validate(ac(model="other", phase="decode"))))
check("site 의 phase 만 달라도 거부",
      any("site 의 model" in e for e in S.validate(ac(phase="decode"))))
check("모르는 candidate_kind 를 거부",
      any("candidate_kind" in e for e in S.validate(ac(candidate_kind="ok"))))
check("site 길이 검사", any("site" in e for e in
                          S.validate(ac(site=["m", "prefill", 12]))))
check("site field 검사", any("site" in e for e in
                           S.validate(ac(site=["m", "p", 12, "x", 0, 1]))))
check("site 정수 검사", any("site" in e for e in
                          S.validate(ac(site=["m", "p", "12", "i", 0, 1]))))
check("submission_sha256 필수",
      any("submission_sha256" in e for e in S.validate(ac(submission_sha256=None))))
check("schema_version 검사",
      any("schema_version" in e for e in S.validate(ac(schema_version=1))))
check("모르는 kind 를 거부", any("kind" in e for e in S.validate(ac(kind="뭐"))))

print()
print("4) unit_adjudication -- 상태별 의미 결박")


def ua(**kw):
    r = {"schema_version": S.SCHEMA_VERSION, "kind": "unit_adjudication",
         "model": "m", "phase": "prefill", "decision_unit_id": "u1",
         "current_expr": "d_head", "state": "pending_human",
         "source_answer_ids": ["A-1"], "actionable": False}
    r.update(kw)
    return r


check("pending_human 통과", S.validate(ua()) == [])
check("**pending_first_pass 는 source_answer_ids 를 요구하지 않는다**",
      S.validate(ua(state="pending_first_pass",
                    source_answer_ids=None)) == [])
check("pending_first_pass 인데 답이 있으면 거부",
      any("source_answer_ids" in e for e in
          S.validate(ua(state="pending_first_pass"))))
check("response_agreement 는 semantic_resolution 을 요구",
      any("semantic_resolution" in e for e in
          S.validate(ua(state="response_agreement"))))
check("response_agreement + 참/거짓 정상 통과",
      S.validate(ua(state="response_agreement", response_agreement=True,
                    semantic_resolution=False)) == [])
check("**response_agreement=False 이면 거부** (상태와 모순)",
      any("참이 아니다" in e for e in
          S.validate(ua(state="response_agreement", response_agreement=False,
                        semantic_resolution=False))))
check("**semantic_resolution 문자열을 거부**",
      any("bool 이어야" in e for e in
          S.validate(ua(state="response_agreement", response_agreement=True,
                        semantic_resolution="no"))))
check("conflict_duplicate 는 response_agreement=False 를 요구",
      any("거짓이 아니다" in e for e in
          S.validate(ua(state="conflict_duplicate", response_agreement=True))))
check("conflict_duplicate 정상 통과",
      S.validate(ua(state="conflict_duplicate",
                    response_agreement=False)) == [])
check("**pending_first_pass + answers 를 거부**",
      any("answers" in e for e in
          S.validate(ua(state="pending_first_pass", source_answer_ids=None,
                        answers=[{"x": 1}]))))
check("pending_first_pass + agreement 필드를 거부",
      any("response_agreement" in e for e in
          S.validate(ua(state="pending_first_pass", source_answer_ids=None,
                        response_agreement=True))))
for k in S.APPROVAL_FIELDS:
    check(f"**비승인 상태에서 `{k}` 를 거부**",
          any(k in e for e in S.validate(ua(**{k: "x"}))))
ok_confirm = ua(state="approved_confirm", actionable=True,
                final_verdict="confirm", grade_after="confirmed",
                approval_id="AP-1", adjudicated_by="사람",
                adjudicated_at="2026-09-26")
check("approved_confirm 정상 통과", S.validate(ok_confirm) == [])
print("   -- 외부 검토가 재현한 모순 조합")
check("**approved_confirm + verdict=rename 거부**",
      any("final_verdict" in e for e in
          S.validate({**ok_confirm, "final_verdict": "rename"})))
check("**approved_confirm + grade=unresolved 거부**",
      any("grade_after" in e for e in
          S.validate({**ok_confirm, "grade_after": "unresolved"})))
check("**approved_no_name_exists + grade=unresolved 거부**",
      any("grade_after" in e for e in
          S.validate({**ok_confirm, "state": "approved_no_name_exists",
                      "final_verdict": "no_name_exists",
                      "grade_after": "unresolved"})))
check("approved_no_name_exists 정상은 confirmed_no_name",
      S.validate({**ok_confirm, "state": "approved_no_name_exists",
                  "final_verdict": "no_name_exists",
                  "grade_after": "confirmed_no_name"}) == [])
check("approved_no_name_exists + resulting_expr 거부",
      any("resulting_expr" in e for e in
          S.validate({**ok_confirm, "state": "approved_no_name_exists",
                      "final_verdict": "no_name_exists",
                      "grade_after": "confirmed_no_name",
                      "resulting_expr": "d_head"})))
check("approved_rename 은 resulting_expr 필수",
      any("resulting_expr" in e for e in
          S.validate({**ok_confirm, "state": "approved_rename",
                      "final_verdict": "rename"})))
check("approved_rename 정상 통과",
      S.validate({**ok_confirm, "state": "approved_rename",
                  "final_verdict": "rename", "resulting_expr": "d_qk"}) == [])
check("**approval_id 가 없으면 거부**",
      any("approval_id" in e for e in
          S.validate({**ok_confirm, "approval_id": None})))
check("adjudicated_by 가 없으면 거부",
      any("adjudicated_by" in e for e in
          S.validate({**ok_confirm, "adjudicated_by": None})))
# 지시대로 **null 강제**로 좁혔다 -- 같은 결정을 두 방식으로 직렬화하지 않는다
check("**approved_confirm 은 resulting_expr 가 null 이어야 한다**",
      any("null 이어야" in e for e in
          S.validate({**ok_confirm, "resulting_expr": "d_head"})))
check("현재 라벨을 적어도 거부",
      any("null 이어야" in e for e in
          S.validate({**ok_confirm, "resulting_expr": "d_qk"})))
check("비승인 상태에 결과가 있으면 거부",
      any("final_verdict" in e for e in
          S.validate(ua(final_verdict="confirm"))))
check("actionable 과 state 불일치 거부",
      any("actionable" in e for e in S.validate(ua(actionable=True))))
print("   -- source_answer_ids 자료형")
check("**문자열 하나를 거부**",
      any("목록이 아니다" in e for e in
          S.validate(ua(source_answer_ids="A-1"))))
check("빈 목록을 거부",
      any("비어 있" in e for e in S.validate(ua(source_answer_ids=[]))))
check("빈 문자열 원소를 거부",
      any("비어 있지 않은 문자열" in e for e in
          S.validate(ua(source_answer_ids=[""]))))
check("문자열이 아닌 원소를 거부",
      any("비어 있지 않은 문자열" in e for e in
          S.validate(ua(source_answer_ids=[1]))))
check("**중복을 거부**",
      any("중복" in e for e in S.validate(ua(source_answer_ids=["A", "A"]))))
check("정상 목록은 통과",
      S.validate(ua(source_answer_ids=["A-1", "A-2"])) == [])
check("answers 가 목록이 아니면 거부",
      any("answers" in e for e in
          S.validate(ua(state="pending_human", answers="문자열"))))

print()
print("5) raw_overlay -- 승인 참조가 있어야 한다")


def ro(**kw):
    r = {"schema_version": S.SCHEMA_VERSION, "kind": "raw_overlay",
         "site": ["m", "prefill", 12, "i", 0, 1],
         "model": "m", "phase": "prefill", "decision_unit_id": "u1",
         "current_expr": "d_head", "state": "approved_rename",
         "unit_adjudication_state": "approved_rename",
         "final_verdict": "rename", "resulting_expr": "d_qk",
         "grade_after": "confirmed", "actionable": True,
         "approval_id": "AP-1", "source_answer_ids": ["A-1"]}
    r.update(kw)
    return r


check("정상 통과", S.validate(ro()) == [])
check("**비승인 상태를 거부**",
      any("승인 상태" in e for e in S.validate(ro(state="pending_human"))))
check("actionable=False 를 거부",
      any("actionable" in e for e in S.validate(ro(actionable=False))))
check("approval_id 없으면 거부",
      any("approval_id" in e for e in S.validate(ro(approval_id=None))))
check("unit_adjudication_state 불일치 거부",
      any("unit_adjudication_state" in e for e in
          S.validate(ro(unit_adjudication_state="approved_confirm"))))
check("grade 규칙 위반 거부",
      any("grade_after" in e for e in S.validate(ro(grade_after="unresolved"))))
check("**raw_overlay 도 site 의 model·phase 를 강제한다**",
      any("site 의 model" in e for e in S.validate(ro(model="other"))))
check("raw_overlay 의 source_answer_ids 자료형도 검사",
      any("목록이 아니다" in e for e in
          S.validate(ro(source_answer_ids="A-1"))))

print()
print("6) 입력 계약 -- 실패 주입")
tmp = tempfile.mkdtemp()
real = (X.LEDGER, I.OUT, I.PACKETS, B.OUT)
try:
    X.LEDGER = os.path.join(tmp, "led.jsonl")
    out = os.path.join(tmp, "packets")
    I.PACKETS = out
    I.OUT = os.path.join(tmp, "answers")
    B.OUT = os.path.join(tmp, "overlay")
    am, _ = X.load_assignment()
    rev = am["assignment_revision"]
    shard = am["manifest"][0]["shard"]
    assert run(X, "--shard", shard, "--session", "T-1", "--out", out) == 0
    pid = os.listdir(out)[0]
    pdir = os.path.join(out, pid)
    order = X.read_ledger()[0]["unit_order"]
    units = I.load_units()
    pj = json.load(io.open(os.path.join(pdir, "_packet.json"), encoding="utf-8"))
    src = pj["sources"][0]
    ev = [{"kind": "source", "file": src["path"], "lines": "1-2",
           "source_sha256": src["sha256"], "claim": "선언부"},
          {"kind": "trace_or_metamorphic", "artifact": "shape",
           "claim": "폭이 맞는다"}]

    def write_answers():
        """첫 둘은 현재 라벨, 셋째는 **no_name**, 나머지는 보류."""
        with io.open(os.path.join(pdir, "answers.jsonl"), "w",
                     encoding="utf-8", newline=NL) as f:
            for i, uid in enumerate(order):
                if i < 2:
                    r = {"proposal": "named",
                         "proposed_expr": units[uid]["current_expr"],
                         "evidence": ev}
                elif i == 2:
                    r = {"proposal": "no_name", "evidence": ev}
                else:
                    r = {"proposal": "cannot_determine", "evidence": []}
                r.update({"decision_unit_id": uid, "assumptions": [],
                          "rejected_candidates": [], "confidence": "high"})
                f.write(json.dumps(r, ensure_ascii=False) + NL)

    write_answers()
    check("수집 통과", run(I, pid) == 0)
    mp = os.path.join(I.OUT, "_ingest.json")
    ap = os.path.join(I.OUT, "accepted.jsonl")
    # 이 시험은 가드를 열고 돌므로 `dirty_build` 는 트리 상태에 따라 참일 수 있다.
    # 깨끗한 기준선은 그 값을 false 로 두고 만들고, dirty 거부는 따로 주입해 본다.
    _d = json.load(io.open(mp, encoding="utf-8"))
    _d["dirty_build"] = False
    json.dump(_d, io.open(mp, "w", encoding="utf-8", newline=NL),
              ensure_ascii=False)
    # accepted_sha256 는 파일 내용에 결박돼 있으므로 metadata 를 고쳐도 유효하다
    errs, rows = S.check_input_contract(I.OUT, rev)
    check("입력 계약 통과 (문제 0)", errs == [])
    check(f"답 {len(order)} 행", len(rows) == len(order))

    good_meta = io.open(mp, "rb").read()
    good_acc = io.open(ap, "rb").read()

    def poke_meta(**kw):
        d = json.loads(good_meta.decode("utf-8"))
        d.update(kw)
        json.dump(d, io.open(mp, "w", encoding="utf-8", newline=NL),
                  ensure_ascii=False)

    for kw, want in (({"errors": 1}, "errors"), ({"rejected": 1}, "rejected"),
                     ({"strict": False}, "strict"),
                     ({"dirty_build": True}, "dirty_build"),
                     ({"built_from_commit": "짧다"}, "built_from_commit"),
                     ({"generator_sha256": {}}, "generator_sha256"),
                     ({"input_sha256": {}}, "input_sha256"),
                     ({"accepted_sha256": None}, "accepted_sha256"),
                     ({"accepted_rows": None}, "accepted_rows"),
                     ({"accepted_rows": 999}, "accepted_rows")):
        poke_meta(**kw)
        e, _ = S.check_input_contract(I.OUT, rev)
        check(f"**_ingest.json 의 {want} 를 어기면 거부**",
              any(want in x for x in e))
        io.open(mp, "wb").write(good_meta)

    # **accepted.jsonl 의 내용을 바꾸면 걸린다** (플래그만 맞아도 통과하던 구멍)
    rows2 = [json.loads(l) for l in good_acc.decode("utf-8").splitlines()
             if l.strip()]
    rows2[0]["proposed_expr"] = "완전히다른이름"
    with io.open(ap, "w", encoding="utf-8", newline=NL) as f:
        for r in rows2:
            f.write(json.dumps(r, ensure_ascii=False) + NL)
    e, _ = S.check_input_contract(I.OUT, rev)
    check("**accepted.jsonl 의 내용을 바꾸면 거부** (해시 결박)",
          any("accepted.jsonl 이 바뀌었다" in x for x in e))
    io.open(ap, "wb").write(good_acc)

    for kw, want in (({"eligible_for_adjudication": False},
                      "eligible_for_adjudication"),
                     ({"packet_quarantined": True}, "packet_quarantined"),
                     ({"assignment_revision": rev + 99},
                      "assignment_revision")):
        rows3 = [json.loads(l) for l in good_acc.decode("utf-8").splitlines()
                 if l.strip()]
        rows3[0].update(kw)
        with io.open(ap, "w", encoding="utf-8", newline=NL) as f:
            for r in rows3:
                f.write(json.dumps(r, ensure_ascii=False) + NL)
        poke_meta(accepted_sha256=None)          # 내용이 바뀌었으므로 해시 검사를 끈다
        e, _ = S.check_input_contract(I.OUT, rev)
        check(f"**accepted.jsonl 의 {want} 를 어기면 거부**",
              any(want in x for x in e))
        io.open(ap, "wb").write(good_acc)
        io.open(mp, "wb").write(good_meta)

    e, _ = S.check_input_contract(I.OUT, rev)
    check("되돌리면 통과", e == [])

    print()
    print("7) 신뢰 해시 사전 대조")
    trusted = am.get("input_sha256") or {}
    up = os.path.join(X.LAB, "units",
                      sorted(f for f in os.listdir(os.path.join(X.LAB, "units"))
                             if f.endswith(".units.jsonl"))[0])
    check("배정 manifest 의 해시와 맞으면 통과",
          S.verify_against_manifest([up], trusted) == [])
    check("신뢰 목록에 없으면 거부",
          any("신뢰 해시에 없는" in e for e in
              S.verify_against_manifest([mp], trusted)))
    fake = {k: {"sha256": "0" * 64} for k in trusted}
    check("**해시가 다르면 거부**",
          any("배정 manifest 와 다르다" in e for e in
              S.verify_against_manifest([up], fake)))

    print()
    print("8) 후보를 실제로 만든다 -- no_name 을 포함해서")
    cands, cerrs, stats, inputs = B.build(I.OUT)
    check("후보가 만들어졌다", cands is not None and len(cands) > 0)
    check("만드는 중 문제 0", cerrs == [])
    check("전부 answer_candidate",
          all(c["kind"] == "answer_candidate" for c in cands))
    check("**actionable 이 하나도 없다**",
          not any(c["actionable"] for c in cands))
    check("**결과 필드를 아예 갖지 않는다**",
          not any(k in c for c in cands
                  for k in ("final_verdict", "resulting_expr", "grade_after")))
    check("스키마를 전부 통과", all(S.validate(c) == [] for c in cands))
    states = {c["state"] for c in cands}
    check(f"**no_name 상태가 실제로 들어 있다** {sorted(states)}",
          "pending_no_name_adjudication" in states)
    check("상태가 1 차 것뿐", states <= set(S.ANSWER_STATES))
    check("단위 수를 센다", stats.get("단위") == len(order))
    check("입력에 units 가 들어간다", any("units" in p for p in inputs))
    check("입력에 crosswalk 이 들어간다", any("crosswalk" in p for p in inputs))

    print()
    print("9) **어떤 실패에서도 공개 후보를 무효화한다**")
    pubp = os.path.join(B.OUT, "candidates.jsonl")

    def clean_meta():
        d = json.loads(good_meta.decode("utf-8"))
        d["dirty_build"] = False
        json.dump(d, io.open(mp, "w", encoding="utf-8", newline=NL),
                  ensure_ascii=False)

    clean_meta()
    check("정상 실행은 0", run(B) == 0)
    check("후보 파일이 생긴다", os.path.exists(pubp))
    # (가) 사전 대조 실패 -- units 하나를 신뢰 해시와 다르게 만든다
    orig_u = io.open(up, "rb").read()
    try:
        io.open(up, "ab").write(("{}" + NL).encode())
        check("**사전 대조 실패면 exit 2**", run(B) == 2)
        check("**공개 후보가 사라진다**", not os.path.exists(pubp))
    finally:
        io.open(up, "wb").write(orig_u)
    clean_meta()
    check("되돌리면 다시 만들어진다", run(B) == 0 and os.path.exists(pubp))
    # (나) units 파일이 **사라지면** 기대 집합 검사가 잡는다
    holder = up + ".hold"
    try:
        os.replace(up, holder)
        check("**기대 집합에 있는 units 가 없어지면 exit 2**", run(B) == 2)
        check("공개 후보가 사라진다", not os.path.exists(pubp))
    finally:
        os.replace(holder, up)
    clean_meta()
    check("되돌리면 다시 0", run(B) == 0 and os.path.exists(pubp))
    # (다) 입력 계약 위반
    poke_meta(errors=3)
    check("**계약 위반이면 exit 2**", run(B) == 2)
    check("공개 후보가 사라진다", not os.path.exists(pubp))
    io.open(mp, "wb").write(good_meta)
    clean_meta()
    check("되돌리면 다시 0", run(B) == 0 and os.path.exists(pubp))
    m2 = json.load(io.open(os.path.join(B.OUT, "_candidates.json"),
                           encoding="utf-8"))
    check("성공 시 candidates_written 이 참", m2.get("candidates_written") is True)
    check("units·crosswalk 검증 수를 적는다",
          m2["stats"].get("units 검증", 0) > 0
          and m2["stats"].get("crosswalk 검증", 0) > 0)
    print()
    print("10) **후보 파일이 metadata 에 결박됐는가**")
    clean_meta()
    check("정상 실행", run(B) == 0)
    cmp_ = os.path.join(B.OUT, "_candidates.json")

    def set_cand_dirty(v):
        """후보 metadata 의 `dirty_build` 를 정한다.

        **자연 상태를 단정하지 않는다.** `ALLOW_DIRTY_BUILD` 는 dirty 를 만드는 변수가
        아니라 dirty 트리에서 실행을 허용하는 변수이므로, 깨끗한 checkout 에서는 이 값이
        거짓이다. 그래서 거부는 **주입해서** 확인하고 기준선은 거짓으로 둔다.
        """
        d = json.load(io.open(cmp_, encoding="utf-8"))
        d["dirty_build"] = v
        json.dump(d, io.open(cmp_, "w", encoding="utf-8", newline=NL),
                  ensure_ascii=False, indent=1)

    def clean_cand_meta():
        set_cand_dirty(False)

    set_cand_dirty(True)
    check("**후보 metadata 의 dirty_build=true 를 거부한다**",
          any("dirty_build" in x for x in
              S.check_candidates_contract(B.OUT)[0]))
    clean_cand_meta()
    cerrs, crows = S.check_candidates_contract(B.OUT)
    check("후보 계약 통과 (문제 0)", cerrs == [])
    check(f"후보 {len(crows)} 행을 읽었다", len(crows) > 0)
    cm = json.load(io.open(os.path.join(B.OUT, "_candidates.json"),
                           encoding="utf-8"))
    check("candidates_sha256 를 기록한다",
          len(cm.get("candidates_sha256") or "") == 64)
    check("candidates_rows 를 기록한다", cm.get("candidates_rows") == len(crows))
    # **한 글자만 바꿔 본다**
    good_c = io.open(pubp, "rb").read()
    try:
        txt = good_c.decode("utf-8")
        assert '"proposal": "named"' in txt
        io.open(pubp, "wb").write(
            txt.replace('"proposal": "named"', '"proposal": "no_name"',
                        1).encode("utf-8"))
        e2, _ = S.check_candidates_contract(B.OUT)
        check("**후보를 한 글자 바꾸면 거부한다** (digest)",
              any("candidates.jsonl 이 바뀌었다" in x for x in e2))
    finally:
        io.open(pubp, "wb").write(good_c)
    e3, _ = S.check_candidates_contract(B.OUT)
    check("되돌리면 통과", e3 == [])
    # 행 수를 거짓으로 적으면
    cm2 = dict(cm, candidates_rows=99999, dirty_build=False)
    json.dump(cm2, io.open(os.path.join(B.OUT, "_candidates.json"), "w",
                           encoding="utf-8", newline=NL), ensure_ascii=False)
    e4, _ = S.check_candidates_contract(B.OUT)
    check("행 수가 다르면 거부", any("candidates_rows" in x for x in e4))
    # 스키마를 어긴 행을 끼워 넣으면
    json.dump({**cm, "dirty_build": False},
              io.open(os.path.join(B.OUT, "_candidates.json"), "w",
                      encoding="utf-8", newline=NL), ensure_ascii=False)
    bad_row = dict(json.loads(good_c.decode("utf-8").splitlines()[0]),
                   state="approved_confirm")
    with io.open(pubp, "ab") as f:
        f.write((json.dumps(bad_row, ensure_ascii=False) + NL).encode("utf-8"))
    e5, _ = S.check_candidates_contract(B.OUT)
    check("**스키마를 어긴 행이 있으면 거부**",
          any("candidates.jsonl" in x and "state" in x for x in e5))
    io.open(pubp, "wb").write(good_c)
    # 무효화된 뒤에는 계약이 막는다
    poke_meta(errors=3)
    check("무효화 실행", run(B) == 2)
    e6, _ = S.check_candidates_contract(B.OUT)
    check("**무효화 뒤에는 후보 계약이 거부한다**", e6 != [])
    check("   이유가 파일 없음 또는 무효화다",
          any("candidates.jsonl 이 없다" in x or "invalidated" in x
              or "candidates_written" in x for x in e6))
    io.open(mp, "wb").write(good_meta)
    clean_meta()
    ok_again = run(B) == 0
    clean_cand_meta()
    check("되돌리면 다시 통과",
          ok_again and S.check_candidates_contract(B.OUT)[0] == [])
finally:
    X.LEDGER, I.OUT, I.PACKETS, B.OUT = real
    shutil.rmtree(tmp, ignore_errors=True)

print(NL + f"{len(OK)}/{len(OK) + len(FAIL)} 통과 — "
      "actionable 은 사람 승인 + approval_id 를 요구한다")
if FAIL:
    print("실패: " + ", ".join(FAIL))
sys.exit(1 if FAIL else 0)
