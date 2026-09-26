r"""overlay 후보의 **입력 계약과 스키마가 실제로 막는가.** 실패 주입으로 확인한다.

외부 검토가 요구한 완료 조건(2026-09-26): 6 절의 입력 계약을 **실제 코드와 실패 주입
검사로** 포함할 것.

    _ingest.json    errors == 0 / rejected == 0 / strict == true /
                    dirty_build == false / input_sha256 재계산 일치
    accepted.jsonl  eligible_for_adjudication == true /
                    packet_quarantined == false / assignment_revision 일치

그리고 `no_name` 과 `cannot_determine` 가 actionable verdict 가 **되지 않는지** 본다.

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


def run_export(shard, session, out):
    old, oldenv = sys.argv, os.environ.get("ALLOW_DIRTY_BUILD")
    sys.argv = ["x", "--shard", shard, "--session", session, "--out", out]
    os.environ["ALLOW_DIRTY_BUILD"] = "1"
    try:
        return X.main() or 0
    except SystemExit as e:
        return e.code
    finally:
        sys.argv = old
        if oldenv is None:
            os.environ.pop("ALLOW_DIRTY_BUILD", None)
        else:
            os.environ["ALLOW_DIRTY_BUILD"] = oldenv


print("1) 상태 표 -- actionable 은 사람 승인 뒤에만")
check("상태가 11 개", len(S.STATES) == 11)
check("actionable 이 3 개", len(S.ACTIONABLE) == 3)
check("actionable 은 전부 approved_",
      all(s.startswith("approved_") for s in S.ACTIONABLE))
check("**no_name 상태는 actionable 이 아니다**",
      "pending_no_name_adjudication" not in S.ACTIONABLE)
check("**cannot_determine 상태도 아니다**",
      "pending_cannot_determine" not in S.ACTIONABLE)
check("pending 과 actionable 이 겹치지 않는다",
      not (set(S.PENDING) & set(S.ACTIONABLE)))
check("둘을 합치면 전체", set(S.PENDING) | set(S.ACTIONABLE) | {"withdrawn"}
      == set(S.STATES))

print()
print("2) 1 차 답에서 나오는 상태 -- verdict 를 만들지 않는다")
for prop, comp, want_state, want_v in (
        ("named", "same", "proposed_confirm", "confirm"),
        ("named", "alias", "proposed_confirm", "confirm"),
        ("named", "different", "proposed_rename", "rename"),
        ("named", "cannot_determine", "pending_cannot_determine", "undetermined"),
        ("named", None, "pending_cannot_determine", "undetermined"),
        ("no_name", None, "pending_no_name_adjudication",
         "no_name_exists_candidate"),
        ("cannot_determine", None, "pending_cannot_determine", "undetermined")):
    st, v = S.state_of(prop, comp)
    ok = st == want_state
    check(f"{prop}/{comp} -> {want_state}", ok)
    check(f"   그 상태는 actionable 이 아니다", st not in S.ACTIONABLE)
try:
    S.state_of("maybe", None)
    check("모르는 proposal 을 거부한다", False)
except S.SchemaError:
    check("모르는 proposal 을 거부한다", True)

print()
print("3) 스키마 검사 -- actionable 이 아니면 결과를 쓸 수 없다")


def rec(**kw):
    r = {"schema_version": S.SCHEMA_VERSION,
         "site": ["m", "prefill", 12, "i", 0, 1],
         "model": "m", "phase": "prefill", "decision_unit_id": "u1",
         "answer_id": "A-1", "packet_id": "P-1", "session_id": "S-1",
         "proposal": "named", "proposed_expr": "d_head",
         "verdict_candidate": "rename", "state": "proposed_rename",
         "actionable": False, "resulting_expr": None, "grade_after": None}
    r.update(kw)
    return r


check("온전한 레코드는 통과", S.validate(rec()) == [])
check("**actionable 이 아닌데 resulting_expr 가 있으면 거부**",
      any("resulting_expr" in e for e in
          S.validate(rec(resulting_expr="d_head"))))
check("**actionable 이 아닌데 grade_after 가 있으면 거부**",
      any("grade_after" in e for e in S.validate(rec(grade_after="confirmed"))))
check("actionable 과 state 가 어긋나면 거부",
      any("actionable" in e for e in S.validate(rec(actionable=True))))
check("approved_rename 인데 resulting_expr 가 없으면 거부",
      any("resulting_expr 가 없다" in e for e in
          S.validate(rec(state="approved_rename", actionable=True,
                         grade_after="confirmed"))))
check("approved_no_name_exists 인데 resulting_expr 가 있으면 거부",
      any("resulting_expr 가 있다" in e for e in
          S.validate(rec(state="approved_no_name_exists", actionable=True,
                         grade_after="unresolved",
                         resulting_expr="d_head"))))
check("actionable 인데 grade_after 가 없으면 거부",
      any("grade_after 가 없다" in e for e in
          S.validate(rec(state="approved_confirm", actionable=True,
                         resulting_expr="d_head"))))
check("모르는 state 를 거부", any("state" in e for e in
                                 S.validate(rec(state="대충승인"))))
check("모르는 verdict 를 거부",
      any("verdict" in e for e in S.validate(rec(verdict_candidate="ok"))))
check("모르는 grade_after 를 거부",
      any("grade_after" in e for e in
          S.validate(rec(state="approved_confirm", actionable=True,
                         resulting_expr="d", grade_after="좋음"))))
check("site 형식을 검사한다 (길이)",
      any("site" in e for e in S.validate(rec(site=["m", "prefill", 12]))))
check("site 형식을 검사한다 (field)",
      any("site" in e for e in
          S.validate(rec(site=["m", "prefill", 12, "x", 0, 1]))))
check("site 형식을 검사한다 (정수)",
      any("site" in e for e in
          S.validate(rec(site=["m", "prefill", "12", "i", 0, 1]))))
check("모르는 필드를 거부", any("모르는 필드" in e for e in
                             S.validate(rec(엉뚱="x"))))
check("schema_version 이 다르면 거부",
      any("schema_version" in e for e in S.validate(rec(schema_version=99))))
check("named 인데 proposed_expr 가 없으면 거부",
      any("proposed_expr 가 없다" in e for e in
          S.validate(rec(proposed_expr=None))))
check("named 이 아닌데 proposed_expr 가 있으면 거부",
      any("proposed_expr 가 있다" in e for e in
          S.validate(rec(proposal="no_name",
                         state="pending_no_name_adjudication",
                         verdict_candidate="no_name_exists_candidate"))))

print()
print("4) 입력 계약 -- 실패 주입")
tmp = tempfile.mkdtemp()
real = (X.LEDGER, I.OUT, I.PACKETS)
try:
    X.LEDGER = os.path.join(tmp, "led.jsonl")
    out = os.path.join(tmp, "packets")
    I.PACKETS = out
    I.OUT = os.path.join(tmp, "answers")
    am, _ = X.load_assignment()
    rev = am["assignment_revision"]
    shard = am["manifest"][0]["shard"]
    assert run_export(shard, "T-1", out) == 0
    pid = os.listdir(out)[0]
    pdir = os.path.join(out, pid)
    order = X.read_ledger()[0]["unit_order"]
    units = I.load_units()
    pj = json.load(io.open(os.path.join(pdir, "_packet.json"), encoding="utf-8"))
    src = pj["sources"][0]

    def answer(uid, expr=None, proposal=None):
        ev = [{"kind": "source", "file": src["path"], "lines": "1-2",
               "source_sha256": src["sha256"], "claim": "선언부"},
              {"kind": "trace_or_metamorphic", "artifact": "shape",
               "claim": "폭이 맞는다"}]
        p = proposal or ("named" if expr else "cannot_determine")
        return {"decision_unit_id": uid, "proposal": p,
                **({"proposed_expr": expr} if expr else {}),
                "evidence": ev if p in ("named", "no_name") else [],
                "assumptions": [], "rejected_candidates": [],
                "confidence": "high"}

    # 모든 단위에 답을 채운다: 첫 둘은 현재 라벨, 나머지는 보류
    recs = []
    for i, uid in enumerate(order):
        cur = units[uid]["current_expr"]
        recs.append(answer(uid, cur if i < 2 else None))
    with io.open(os.path.join(pdir, "answers.jsonl"), "w", encoding="utf-8",
                 newline=NL) as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + NL)

    def ingest_strict():
        old, oldenv = sys.argv, os.environ.get("ALLOW_DIRTY_BUILD")
        sys.argv = ["ingest_answers.py", pid]
        os.environ["ALLOW_DIRTY_BUILD"] = "1"
        try:
            return I.main() or 0
        except SystemExit as e:
            return e.code
        finally:
            sys.argv = old
            if oldenv is None:
                os.environ.pop("ALLOW_DIRTY_BUILD", None)
            else:
                os.environ["ALLOW_DIRTY_BUILD"] = oldenv

    check("수집이 통과한다", ingest_strict() == 0)
    mp = os.path.join(I.OUT, "_ingest.json")
    ap = os.path.join(I.OUT, "accepted.jsonl")
    # 이 시험은 `ALLOW_DIRTY_BUILD` 로 돌므로 수집 산출물의 `dirty_build` 가 **참**이다.
    # 계약은 그것을 제대로 거부한다(아래 주입 검사가 확인한다). 기준선을 만들려면 그
    # 한 필드만 깨끗한 값으로 바꿔 둔다 -- 계약을 무르게 하는 것이 아니다.
    _d = json.load(io.open(mp, encoding="utf-8"))
    check("**시험 환경에서는 dirty_build 가 참이고, 계약이 그것을 거부한다**",
          _d.get("dirty_build") is True
          and any("dirty_build" in x for x in
                  S.check_input_contract(I.OUT, rev)[0]))
    _d["dirty_build"] = False
    json.dump(_d, io.open(mp, "w", encoding="utf-8", newline=NL),
              ensure_ascii=False)
    errs, rows = S.check_input_contract(I.OUT, rev)
    check("입력 계약 통과 (문제 0)", errs == [])
    check(f"답 {len(order)} 행을 읽었다", len(rows) == len(order))

    good_meta = io.open(mp, "rb").read()
    good_acc = io.open(ap, "rb").read()

    def poke_meta(**kw):
        d = json.loads(good_meta.decode("utf-8"))
        d.update(kw)
        json.dump(d, io.open(mp, "w", encoding="utf-8", newline=NL),
                  ensure_ascii=False)

    def poke_acc(**kw):
        rows2 = [json.loads(l) for l in good_acc.decode("utf-8").splitlines()
                 if l.strip()]
        rows2[0].update(kw)
        with io.open(ap, "w", encoding="utf-8", newline=NL) as f:
            for r in rows2:
                f.write(json.dumps(r, ensure_ascii=False) + NL)

    for kw, want in (({"errors": 1}, "errors"), ({"rejected": 1}, "rejected"),
                     ({"strict": False}, "strict"),
                     ({"dirty_build": True}, "dirty_build")):
        poke_meta(**kw)
        e, _ = S.check_input_contract(I.OUT, rev)
        check(f"**_ingest.json 의 {want} 를 어기면 거부**",
              any(want in x for x in e))
        io.open(mp, "wb").write(good_meta)

    poke_meta(input_sha256={})
    e, _ = S.check_input_contract(I.OUT, rev)
    check("input_sha256 가 비면 거부", any("input_sha256" in x for x in e))
    io.open(mp, "wb").write(good_meta)

    # 입력 해시 재계산: 기록된 입력 하나를 실제로 바꾼다
    d = json.loads(good_meta.decode("utf-8"))
    target = next(k for k in d["input_sha256"] if k.endswith("answers.jsonl"))
    tp = os.path.join(PROJ, target)
    orig = io.open(tp, "rb").read()
    try:
        io.open(tp, "ab").write(("{}" + NL).encode())
        e, _ = S.check_input_contract(I.OUT, rev)
        check("**기록된 입력이 바뀌면 거부 (해시 재계산)**",
              any("입력이 바뀌었다" in x for x in e))
    finally:
        io.open(tp, "wb").write(orig)
    e, _ = S.check_input_contract(I.OUT, rev)
    check("되돌리면 통과", e == [])

    for kw, want in (({"eligible_for_adjudication": False},
                      "eligible_for_adjudication"),
                     ({"packet_quarantined": True}, "packet_quarantined"),
                     ({"assignment_revision": rev + 99},
                      "assignment_revision")):
        poke_acc(**kw)
        e, _ = S.check_input_contract(I.OUT, rev)
        check(f"**accepted.jsonl 의 {want} 를 어기면 거부**",
              any(want in x for x in e))
        io.open(ap, "wb").write(good_acc)

    os.remove(mp)
    e, _ = S.check_input_contract(I.OUT, rev)
    check("_ingest.json 이 없으면 거부", any("_ingest.json" in x for x in e))
    io.open(mp, "wb").write(good_meta)

    print()
    print("5) 후보를 실제로 만든다 -- 전부 actionable 이 아니다")
    cands, cerrs, stats, inputs = B.build(I.OUT)
    check("후보가 만들어졌다", cands is not None and len(cands) > 0)
    check("만드는 중 문제 0", cerrs == [])
    check("**actionable 이 하나도 없다**",
          not any(c["actionable"] for c in cands))
    check("**resulting_expr 가 전부 None**",
          all(c["resulting_expr"] is None for c in cands))
    check("**grade_after 가 전부 None**",
          all(c["grade_after"] is None for c in cands))
    check("스키마를 전부 통과한다",
          all(S.validate(c) == [] for c in cands))
    check("site 가 6-튜플", all(len(c["site"]) == 6 for c in cands))
    check("raw op_id 가 정수", all(isinstance(c["site"][2], int) for c in cands))
    check("발행 셀도 기록된다",
          all(len(c["published_cell"]) == 4 for c in cands))
    check("origin 이 기록된다",
          all(c["origin"] in ("raw_slot", "synthesized_norm",
                              "canonical_weight", "ambiguous")
              for c in cands))
    states = {c["state"] for c in cands}
    check(f"상태가 1 차 것뿐이다 {sorted(states)}",
          states <= {"proposed_confirm", "proposed_rename",
                     "pending_cannot_determine",
                     "pending_no_name_adjudication"})
    check("입력 목록에 crosswalk 이 들어간다",
          any("crosswalk" in p for p in inputs))

    print()
    print("6) 입력 계약을 어기면 후보를 만들지 않는다")
    poke_meta(errors=3)
    cands2, cerrs2, _s, _i = B.build(I.OUT)
    check("**후보가 None 이다**", cands2 is None)
    check("이유를 말한다", any("errors" in x for x in cerrs2))
    io.open(mp, "wb").write(good_meta)
    cands3, cerrs3, _s, _i = B.build(I.OUT)
    check("되돌리면 다시 만든다", cands3 is not None and cerrs3 == [])
finally:
    X.LEDGER, I.OUT, I.PACKETS = real
    shutil.rmtree(tmp, ignore_errors=True)

print(NL + f"{len(OK)}/{len(OK) + len(FAIL)} 통과 — "
      "actionable 은 사람 판정 뒤에만 생긴다")
if FAIL:
    print("실패: " + ", ".join(FAIL))
sys.exit(1 if FAIL else 0)
