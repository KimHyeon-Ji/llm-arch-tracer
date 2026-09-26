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
try:
    S.state_of("maybe", None)
    check("모르는 proposal 을 거부", False)
except S.SchemaError:
    check("모르는 proposal 을 거부", True)

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
check("response_agreement + semantic_resolution=False 통과",
      S.validate(ua(state="response_agreement",
                    semantic_resolution=False)) == [])
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
check("approved_confirm 의 resulting_expr 는 현재 라벨만 허용",
      any("현재 라벨" in e for e in
          S.validate({**ok_confirm, "resulting_expr": "d_qk"})))
check("approved_confirm + resulting_expr=현재 라벨 통과",
      S.validate({**ok_confirm, "resulting_expr": "d_head"}) == [])
check("비승인 상태에 결과가 있으면 거부",
      any("actionable 이 아닌데" in e for e in
          S.validate(ua(final_verdict="confirm"))))
check("actionable 과 state 불일치 거부",
      any("actionable" in e for e in S.validate(ua(actionable=True))))

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
    print("9) 문제가 있으면 **공개 후보 파일을 쓰지 않는다**")
    check("정상 실행은 0", run(B) == 0)
    pubp = os.path.join(B.OUT, "candidates.jsonl")
    check("후보 파일이 있다", os.path.exists(pubp))
    n_before = sum(1 for l in io.open(pubp, encoding="utf-8") if l.strip())
    # units 하나를 신뢰 해시와 다르게 만든다 -> 사전 대조에서 막힌다
    orig_u = io.open(up, "rb").read()
    try:
        io.open(up, "ab").write(("{}" + NL).encode())
        code = run(B)
        check("**사전 대조 실패면 exit 2**", code == 2)
        check("후보 파일이 그대로다 (덧쓰지 않았다)",
              sum(1 for l in io.open(pubp, encoding="utf-8") if l.strip())
              == n_before)
    finally:
        io.open(up, "wb").write(orig_u)
    # 입력 계약 위반
    poke_meta(errors=3)
    check("**계약 위반이면 exit 2**", run(B) == 2)
    io.open(mp, "wb").write(good_meta)
    _d = json.loads(good_meta.decode("utf-8"))
    _d["dirty_build"] = False
    json.dump(_d, io.open(mp, "w", encoding="utf-8", newline=NL),
              ensure_ascii=False)
    check("되돌리면 다시 0", run(B) == 0)
finally:
    X.LEDGER, I.OUT, I.PACKETS, B.OUT = real
    shutil.rmtree(tmp, ignore_errors=True)

print(NL + f"{len(OK)}/{len(OK) + len(FAIL)} 통과 — "
      "actionable 은 사람 승인 + approval_id 를 요구한다")
if FAIL:
    print("실패: " + ", ".join(FAIL))
sys.exit(1 if FAIL else 0)
