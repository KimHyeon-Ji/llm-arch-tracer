r"""답 수집기가 **형식 문제를 실제로 잡는가.** 합성 답으로 확인한다.

실제 검토 답이 오기 전에 이 검사가 살아 있어야 한다. 파일럿 결과를 받고 나서 "형식 오류
0 건" 을 보고해도, 검사가 아무것도 안 잡는 것이면 의미가 없다.

임시 반출 위치에 패킷을 만들고 그 안에 합성 `answers.jsonl` 을 놓는다. **실제 대장과
실제 패킷은 건드리지 않는다.**

실행:
    .venv\Scripts\python.exe develop\test_ingest_answers.py
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

import export_shard_packet as X                                  # noqa: E402
import ingest_answers as I                                       # noqa: E402

OK, FAIL = [], []
NL = chr(10)


def check(name, cond):
    (OK if cond else FAIL).append(name)
    print(("  OK   " if cond else "  FAIL ") + name)


def run_export(shard, session, out):
    old = sys.argv
    sys.argv = ["x", "--shard", shard, "--session", session, "--out", out]
    try:
        return X.main() or 0
    except SystemExit as e:
        return e.code
    finally:
        sys.argv = old


tmp = tempfile.mkdtemp()
real_ledger = X.LEDGER
try:
    X.LEDGER = os.path.join(tmp, "led.jsonl")
    out = os.path.join(tmp, "packets")
    am, _ = X.load_assignment()
    shard = am["manifest"][0]["shard"]
    assert run_export(shard, "T-1", out) == 0
    pid = os.listdir(out)[0]
    pdir = os.path.join(out, pid)
    rec = X.read_ledger()[0]
    order = rec["unit_order"]
    units = I.load_units()
    pj = json.load(io.open(os.path.join(pdir, "_packet.json"), encoding="utf-8"))
    src = pj["sources"][0]
    cur = units[order[0]]["current_expr"]

    def write_answers(recs):
        with io.open(os.path.join(pdir, "answers.jsonl"), "w",
                     encoding="utf-8", newline=NL) as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + NL)

    def good(uid, expr=None):
        """온전한 답. 필수 필드를 **전부** 갖춘다."""
        return {"decision_unit_id": uid,
                "proposal": "named" if expr else "cannot_determine",
                **({"proposed_expr": expr} if expr else {}),
                "evidence": ([{"kind": "source", "file": src["path"],
                               "lines": "1-2", "source_sha256": src["sha256"],
                               "claim": "선언부"},
                              {"kind": "trace_or_metamorphic",
                               "artifact": "shape", "claim": "폭이 맞는다"}]
                             if expr else []),
                "assumptions": [], "rejected_candidates": [],
                "confidence": "high"}

    def ev2(file=None, sha=None, lines="1-2", drop=None):
        """근거 두 종류. `drop` 으로 필드 하나를 빼서 검사를 시험한다."""
        a = {"kind": "source", "file": file or src["path"], "lines": lines,
             "source_sha256": sha or src["sha256"], "claim": "선언부"}
        b = {"kind": "trace_or_metamorphic", "artifact": "shape",
             "claim": "폭이 맞는다"}
        for d in (drop or []):
            a.pop(d, None)
        return [a, b]

    def named(uid, expr="d_head", **kw):
        r = {"decision_unit_id": uid, "proposal": "named",
             "proposed_expr": expr, "evidence": ev2(),
             "assumptions": [], "rejected_candidates": [],
             "confidence": "high"}
        r.update(kw)
        return r

    print("1) 온전한 답은 문제 0")
    write_answers([good(u, cur if i == 0 else None)
                   for i, u in enumerate(order)])
    rows, errs, st, pos, _in = I.ingest([pid], out)
    check("모든 단위에 답이 있다", len(rows) == len(order))
    check("문제 0", errs == [])
    check("첫 답이 현재 라벨과 same", rows[0]["comparison"] == "same")
    check("위치가 1 부터 붙는다", rows[0]["position"] == 1)
    check("answer_id 가 붙는다",
          all(r["answer_id"].startswith("A-") for r in rows))
    check("세션이 기록된다", rows[0]["session_id"] == "T-1")
    check("위치별 집계가 있다", len(pos) == len(order))
    check("**전부 유효로 표시된다**",
          all(r["eligible_for_adjudication"] for r in rows))
    check("submission_sha256 가 붙는다",
          all(len(r["submission_sha256"]) == 64 for r in rows))
    check("assignment_revision 이 붙는다",
          all(r["assignment_revision"] is not None for r in rows))
    check("입력 목록이 돌아온다", len(_in) >= 3)

    print()
    print("2) 형식 문제를 하나씩 넣어 본다")

    def one(rec, want):
        write_answers([rec])
        _rows, _errs, _st, _p, _in = I.ingest([pid], out)
        hit = any(want in e for e in _errs)
        check(f"{want!r} 를 잡는다", hit)

    u0 = order[0]
    one({"proposal": "named", "proposed_expr": "d_head", "confidence": "high",
         "evidence": ev2(), "assumptions": [], "rejected_candidates": []},
        "`decision_unit_id` 가 없다")
    one({k: v for k, v in named(u0).items() if k != "proposal"},
        "`proposal` 가 없다")
    one({k: v for k, v in named(u0).items() if k != "confidence"},
        "`confidence` 가 없다")
    one({k: v for k, v in named(u0).items() if k != "assumptions"},
        "`assumptions` 가 없다")
    one({k: v for k, v in named(u0).items() if k != "rejected_candidates"},
        "`rejected_candidates` 가 없다")
    one(named(u0, assumptions="문자열"), "`assumptions` 가 목록이 아니다")
    one(named("없는단위"), "이 패킷에 없는 단위")
    one(named(u0, proposal="maybe"), "모르는 proposal")
    one({k: v for k, v in named(u0).items() if k != "proposed_expr"},
        "named 인데 `proposed_expr` 가 없다")
    one(named(u0, proposal="no_name"), "인데 `proposed_expr` 가 있다")
    one(named(u0, confidence="아주높음"), "모르는 confidence")
    one(named(u0, evidence=[ev2()[0]]), "온전한 근거가 1 종류다")
    one({k: v for k, v in named(u0).items() if k != "evidence"},
        "`evidence` 가 없다")
    one(named(u0, evidence=ev2(file="source/없는파일.py")), "패킷에 없는 source")
    one(named(u0, evidence=ev2(sha="0" * 64)),
        "source_sha256 가 실제 파일과 다르다")
    one(named(u0, evidence=ev2(drop=["claim"])), "source 근거에 `claim` 가 없다")
    one(named(u0, evidence=ev2(drop=["lines"])), "source 근거에 `lines` 가 없다")
    one(named(u0, evidence=ev2(drop=["source_sha256"])),
        "source 근거에 `source_sha256` 가 없다")
    one(named(u0, evidence=[ev2()[0], {"kind": "trace_or_metamorphic",
                                       "claim": "y"}]),
        "trace_or_metamorphic 근거에 `artifact` 가 없다")
    one(named(u0, evidence=[{"kind": "추측", "claim": "x"}, ev2()[1]]),
        "모르는 evidence kind")
    # `no_name` 도 근거 두 종류를 요구한다 -- 그것도 주장이다
    one({"decision_unit_id": u0, "proposal": "no_name", "confidence": "high",
         "evidence": [], "assumptions": [], "rejected_candidates": []},
        "no_name 인데 온전한 근거가 0 종류다")

    print()
    print("3) 빠뜨림과 중복")
    write_answers([good(order[0], cur)])
    rows, errs, st, _p, _in = I.ingest([pid], out)
    check("답이 없는 단위를 센다", any("답이 없는 단위" in e for e in errs))
    check(f"빠진 수가 맞는다 ({len(order) - 1})",
          st["답 빠진 단위"] == len(order) - 1)
    write_answers([good(order[0], cur), good(order[0], cur)])
    rows, errs, st, _p, _in = I.ingest([pid], out)
    check("같은 단위를 두 번 답하면 잡는다",
          any("두 번 답했다" in e for e in errs))
    check("**중복된 두 행 모두 거부된다**",
          not any(r["eligible_for_adjudication"] for r in rows))
    with io.open(os.path.join(pdir, "answers.jsonl"), "a",
                 encoding="utf-8", newline=NL) as f:
        f.write("{망가진 json" + NL)
    rows, errs, st, _p, _in = I.ingest([pid], out)
    check("JSON 이 아닌 줄을 잡는다", any("JSON 이 아니다" in e for e in errs))

    print()
    print("4) 비교는 값을 근거로 쓰지 않는다")
    write_answers([good(order[0], cur)])
    rows, _e, _s, _p, _in = I.ingest([pid], out)
    check("같은 식 -> same", rows[0]["comparison"] == "same")
    m = units[order[0]]["model"]
    import label_universe as U                                    # noqa: E402
    nm, _al = U.universe(m)
    other = next((n for n in sorted(nm)
                  if n not in ("ceil", "round", "min", "max", "roundup")
                  and n != cur), None)
    write_answers([good(order[0], other)])
    rows, _e, _s, _p, _in = I.ingest([pid], out)
    check(f"다른 심볼({other}) -> same 이 아니다",
          rows[0]["comparison"] != "same")
    write_answers([good(order[0], "완전히모르는이름")])
    rows, _e, _s, _p, _in = I.ingest([pid], out)
    check("모르는 이름 -> cannot_determine",
          rows[0]["comparison"] == "cannot_determine")

    print()
    print("4-b) **패킷 단위 격리** -- 문제가 하나면 그 패킷 전부를 격리한다")

    def all_good(bad_at=None, bad=None, drop=None):
        """모든 단위에 답을 채우고, 필요하면 한 자리만 문제 있는 답으로 바꾼다."""
        recs = []
        for i, u in enumerate(order):
            if drop is not None and i == drop:
                continue
            if bad_at is not None and i == bad_at:
                recs.append(bad)
            else:
                recs.append(good(u, cur if i == 0 else None))
        return recs

    write_answers(all_good())
    rows, errs, st, _p, _in = I.ingest([pid], out)
    check("온전하면 전부 유효",
          all(r["eligible_for_adjudication"] for r in rows) and errs == [])
    # 한 자리만 근거를 한 종류로 만든다
    write_answers(all_good(bad_at=1, bad=named(order[1], evidence=[ev2()[0]])))
    rows, errs, st, _p, _in = I.ingest([pid], out)
    acc = [r for r in rows if r["eligible_for_adjudication"]]
    rej = [r for r in rows if not r["eligible_for_adjudication"]]
    check("**한 줄이 문제면 accepted 0**", len(acc) == 0)
    check("전부 거부된다", len(rej) == len(order))
    check("문제 있는 행에 format_errors 가 있다",
          sum(1 for r in rows if r["format_errors"]) == 1)
    check("**나머지 행도 packet_quarantined 로 표시된다**",
          all(r["packet_quarantined"] for r in rows))
    check("집계에 격리가 찍힌다", st.get("격리된 패킷") == 1)
    # 한 자리를 빼먹는다
    write_answers(all_good(drop=2))
    rows, errs, st, _p, _in = I.ingest([pid], out)
    check("**한 단위를 빼먹으면 accepted 0**",
          not any(r["eligible_for_adjudication"] for r in rows))
    # 깨진 JSON 한 줄
    write_answers(all_good())
    with io.open(os.path.join(pdir, "answers.jsonl"), "a", encoding="utf-8",
                 newline=NL) as f:
        f.write("{망가진" + NL)
    rows, errs, st, _p, _in = I.ingest([pid], out)
    check("**깨진 JSON 한 줄이면 accepted 0**",
          not any(r["eligible_for_adjudication"] for r in rows))
    # 레코드가 객체가 아닌 경우 -- 검사기가 죽지 않아야 한다
    write_answers(all_good())
    with io.open(os.path.join(pdir, "answers.jsonl"), "a", encoding="utf-8",
                 newline=NL) as f:
        f.write("[1, 2, 3]" + NL)
    try:
        rows, errs, st, _p, _in = I.ingest([pid], out)
        check("**JSON 배열 입력이 검사기를 죽이지 않는다**", True)
        check("객체가 아니라고 잡는다",
              any("객체가 아니다" in e for e in errs))
        check("그 패킷은 격리된다",
              not any(r["eligible_for_adjudication"] for r in rows))
    except Exception as e:
        check(f"**JSON 배열 입력이 검사기를 죽이지 않는다** ({e})", False)

    print()
    print("4-c) 지정 실행은 **종료 코드가 0 이 아니다**")

    def run_main(*argv):
        # `main()` 은 깨끗한 트리를 요구한다(`require_clean_tree`). 시험은 임시
        # 디렉터리에만 쓰므로 그 가드만 열어 준다 -- 산출물에 dirty_build 가 찍힌다.
        old, oldenv = sys.argv, os.environ.get("ALLOW_DIRTY_BUILD")
        sys.argv = ["ingest_answers.py", *argv]
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

    real_out, real_pk = I.OUT, I.PACKETS
    try:
        I.OUT = os.path.join(tmp, "answers")
        I.PACKETS = out            # **실제 패킷을 읽지 않게** 갈아끼운다
        write_answers([named(u0, evidence=[ev2()[0]])])     # 문제 있는 답
        code = run_main(pid)
        check("**문제가 있으면 exit 1**", code == 1)
        write_answers([good(u, cur if i == 0 else None)
                       for i, u in enumerate(order)])
        code = run_main(pid)
        check("온전하면 exit 0", code == 0)
        acc_p = os.path.join(I.OUT, "accepted.jsonl")
        rej_p = os.path.join(I.OUT, "rejected.jsonl")
        check("accepted.jsonl 이 나온다", os.path.exists(acc_p))
        check("rejected.jsonl 이 나온다", os.path.exists(rej_p))
        na = sum(1 for l in io.open(acc_p, encoding="utf-8") if l.strip())
        nr = sum(1 for l in io.open(rej_p, encoding="utf-8") if l.strip())
        check(f"유효 {len(order)} / 거부 0", na == len(order) and nr == 0)
        m = json.load(io.open(os.path.join(I.OUT, "_ingest.json"),
                              encoding="utf-8"))
        check("입력 해시를 기록한다", len(m.get("input_sha256") or {}) >= 3)
        check("strict 표시가 있다", m.get("strict") is True)
        check("overlay 입력이 accepted 뿐임을 적는다",
              "accepted.jsonl" in str(m.get("overlay_input")))
    finally:
        I.OUT, I.PACKETS = real_out, real_pk

    print()
    print("4-d) 패킷을 **다시 검증**한다 (기록값을 믿지 않는다)")
    write_answers([good(u, cur if i == 0 else None)
                   for i, u in enumerate(order)])
    sp = os.path.join(pdir, "source", os.path.basename(src["path"]))
    orig = io.open(sp, "rb").read()
    try:
        io.open(sp, "ab").write("# 손댐".encode() + NL.encode())
        rows, errs, st, _p, _in = I.ingest([pid], out)
        check("**source 가 바뀌면 잡는다**",
              any("source 가 바뀌었다" in e for e in errs))
        check("그 패킷의 답은 전부 거부된다",
              all(not r["eligible_for_adjudication"] for r in rows))
    finally:
        io.open(sp, "wb").write(orig)
    # shard.md 의 순서를 흔든다
    shp = os.path.join(pdir, "shard.md")
    doc = io.open(shp, encoding="utf-8").read()
    try:
        blocks = doc.split(NL + "### ")
        if len(blocks) > 2:
            swapped = blocks[0] + NL + "### " + (NL + "### ").join(
                [blocks[2], blocks[1]] + blocks[3:])
            io.open(shp, "w", encoding="utf-8", newline=NL).write(swapped)
            rows, errs, st, _p, _in = I.ingest([pid], out)
            check("**shard.md 순서가 대장과 다르면 잡는다**",
                  any("단위 순서가 대장과 다르다" in e for e in errs))
    finally:
        io.open(shp, "w", encoding="utf-8", newline=NL).write(doc)
    # _packet.json 의 packet_id 를 바꾼다
    pjp = os.path.join(pdir, "_packet.json")
    pjd = json.load(io.open(pjp, encoding="utf-8"))
    try:
        json.dump({**pjd, "packet_id": "P-000000000000"},
                  io.open(pjp, "w", encoding="utf-8", newline=NL),
                  ensure_ascii=False)
        rows, errs, st, _p, _in = I.ingest([pid], out)
        check("packet_id 가 대장과 다르면 잡는다",
              any("packet_id 가 대장과 다르다" in e for e in errs))
        json.dump({**pjd, "units": 999},
                  io.open(pjp, "w", encoding="utf-8", newline=NL),
                  ensure_ascii=False)
        rows, errs, st, _p, _in = I.ingest([pid], out)
        check("units 수가 다르면 잡는다",
              any("_packet.json 의 units" in e for e in errs))
    finally:
        json.dump(pjd, io.open(pjp, "w", encoding="utf-8", newline=NL),
                  ensure_ascii=False, indent=1)

    print()
    print("4-e) 패킷 변조 -- **함께 고쳐도** 검출한다")
    write_answers(all_good())
    rows, errs, st, _p, _in = I.ingest([pid], out)
    check("변조 전에는 문제 0", errs == [])
    sp2 = os.path.join(pdir, "source", os.path.basename(src["path"]))
    pjp2 = os.path.join(pdir, "_packet.json")
    orig_src = io.open(sp2, "rb").read()
    orig_pj = io.open(pjp2, "rb").read()
    try:
        # source 를 고치고 _packet.json 의 해시도 **함께** 맞춰 준다
        io.open(sp2, "ab").write("# 손댐".encode() + NL.encode())
        import hashlib
        h = hashlib.sha256(io.open(sp2, "rb").read()).hexdigest()
        d = json.loads(orig_pj.decode("utf-8"))
        for smeta in d["sources"]:
            if smeta["path"] == src["path"]:
                smeta["sha256"] = h
        json.dump(d, io.open(pjp2, "w", encoding="utf-8", newline=NL),
                  ensure_ascii=False, indent=1)
        rows, errs, st, _p, _in = I.ingest([pid], out)
        check("**source 와 _packet.json 을 함께 바꿔도 검출한다**",
              any("패킷 전체 해시가 대장과 다르다" in e for e in errs))
        check("그 패킷은 격리된다",
              not any(r["eligible_for_adjudication"] for r in rows))
    finally:
        io.open(sp2, "wb").write(orig_src)
        io.open(pjp2, "wb").write(orig_pj)
    # shard.md 본문만 바꾼다 (단위 순서는 그대로)
    shp2 = os.path.join(pdir, "shard.md")
    orig_doc = io.open(shp2, "rb").read()
    try:
        io.open(shp2, "ab").write("추가 문장".encode() + NL.encode())
        rows, errs, st, _p, _in = I.ingest([pid], out)
        check("**shard.md 본문만 바꿔도 검출한다**",
              any("패킷 전체 해시가 대장과 다르다" in e for e in errs))
    finally:
        io.open(shp2, "wb").write(orig_doc)
    rows, errs, st, _p, _in = I.ingest([pid], out)
    check("되돌리면 문제 0", errs == [])
    check("입력 해시에 패킷 파일이 들어간다",
          any("shard.md" in p for p in _in) and any("source" in p for p in _in))

    print()
    print("4-f) phase 전용 alias 는 다른 phase 에서 쓰이지 않는다")
    import label_universe as U2                                   # noqa: E402
    import expr_compare as E2                                     # noqa: E402
    ap2 = os.path.join(tmp, "aliases.yaml")
    io.open(ap2, "w", encoding="utf-8", newline=NL).write(
        "aliases:" + NL
        + "  시험별명:" + NL
        + "    expr: n_h" + NL
        + "    model: openai__gpt-oss-20b" + NL
        + "    phase: prefill" + NL)
    check("prefill 에서는 쓰인다",
          E2.load_aliases("openai__gpt-oss-20b", "prefill", ap2)
          .get("시험별명") == "n_h")
    check("**decode 에서는 쓰이지 않는다**",
          "시험별명" not in E2.load_aliases("openai__gpt-oss-20b", "decode", ap2))
    check("**phase 를 모르면 쓰이지 않는다**",
          "시험별명" not in E2.load_aliases("openai__gpt-oss-20b", None, ap2))
    check("**model 을 모르면 아무것도 주지 않는다**",
          E2.load_aliases(None, "prefill", ap2) == {})
    check("universe 가 phase 를 받는다",
          "phase" in __import__("inspect").signature(U2.universe).parameters)
    src_i = __import__("inspect").getsource(I.ingest)
    check("수집기가 (model, phase) 로 캐시한다",
          "u.get(" + chr(34) + "phase" + chr(34) + ")" in src_i
          and "U.universe(key[0], key[1])" in src_i)

    print()
    print("5) 답 파일이 없을 때 -- 지정과 전체 점검을 가른다")
    os.remove(os.path.join(pdir, "answers.jsonl"))
    rows, errs, st, _p, _in = I.ingest([pid], out)
    check("답 없음으로 센다", st.get("답이 아직 없는 패킷") == 1)
    # **지정 실행에서는 오류다.** 조용히 exit 0 을 내면 안 된다(외부 검토 2026-09-25).
    check("**지정 실행에서는 오류다**",
          any("answers.jsonl 이 없다" in e for e in errs))
    rows2, errs2, st2, _p2, _in2 = I.ingest(None, out)
    check("전체 점검에서는 오류가 아니다", errs2 == [])
    check("전체 점검도 답 없음은 센다", st2.get("답이 아직 없는 패킷", 0) >= 1)
finally:
    X.LEDGER = real_ledger
    shutil.rmtree(tmp, ignore_errors=True)

print(NL + f"{len(OK)}/{len(OK) + len(FAIL)} 통과 — "
      "형식 검사가 실제로 발화한다")
if FAIL:
    print("실패: " + ", ".join(FAIL))
sys.exit(1 if FAIL else 0)
