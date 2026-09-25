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
        return {"decision_unit_id": uid,
                "proposal": "named" if expr else "cannot_determine",
                **({"proposed_expr": expr} if expr else {}),
                "evidence": ([{"kind": "source", "file": src["path"],
                               "lines": "1-2", "source_sha256": src["sha256"],
                               "claim": "선언부"},
                              {"kind": "trace_or_metamorphic",
                               "artifact": "shape", "claim": "폭이 맞는다"}]
                             if expr else []),
                "assumptions": [], "confidence": "high"}

    print("1) 온전한 답은 문제 0")
    write_answers([good(u, cur if i == 0 else None)
                   for i, u in enumerate(order)])
    rows, errs, st, pos = I.ingest([pid], out)
    check("모든 단위에 답이 있다", len(rows) == len(order))
    check("문제 0", errs == [])
    check("첫 답이 현재 라벨과 same", rows[0]["comparison"] == "same")
    check("위치가 1 부터 붙는다", rows[0]["position"] == 1)
    check("answer_id 가 붙는다",
          all(r["answer_id"].startswith("A-") for r in rows))
    check("세션이 기록된다", rows[0]["session_id"] == "T-1")
    check("위치별 집계가 있다", len(pos) == len(order))

    print()
    print("2) 형식 문제를 하나씩 넣어 본다")

    def one(rec, want):
        write_answers([rec])
        _rows, _errs, _st, _p = I.ingest([pid], out)
        hit = any(want in e for e in _errs)
        check(f"{want!r} 를 잡는다", hit)

    one({"proposal": "named", "proposed_expr": "d_head"}, "`decision_unit_id` 가 없다")
    one({"decision_unit_id": order[0]}, "`proposal` 가 없다")
    one({"decision_unit_id": "없는단위", "proposal": "no_name"}, "이 패킷에 없는 단위")
    one({"decision_unit_id": order[0], "proposal": "maybe"}, "모르는 proposal")
    one({"decision_unit_id": order[0], "proposal": "named"},
        "named 인데 `proposed_expr` 가 없다")
    one({"decision_unit_id": order[0], "proposal": "no_name",
         "proposed_expr": "d_head"}, "인데 `proposed_expr` 가 있다")
    one({"decision_unit_id": order[0], "proposal": "named",
         "proposed_expr": "d_head", "confidence": "아주높음",
         "evidence": [{"kind": "source", "file": src["path"],
                       "source_sha256": src["sha256"], "claim": "x"},
                      {"kind": "trace_or_metamorphic", "claim": "y"}]},
        "모르는 confidence")
    one({"decision_unit_id": order[0], "proposal": "named",
         "proposed_expr": "d_head",
         "evidence": [{"kind": "source", "file": src["path"],
                       "source_sha256": src["sha256"], "claim": "x"}]},
        "근거가 한 종류다")
    one({"decision_unit_id": order[0], "proposal": "named",
         "proposed_expr": "d_head",
         "evidence": [{"kind": "source", "file": "source/없는파일.py",
                       "source_sha256": src["sha256"], "claim": "x"},
                      {"kind": "trace_or_metamorphic", "claim": "y"}]},
        "패킷에 없는 source")
    one({"decision_unit_id": order[0], "proposal": "named",
         "proposed_expr": "d_head",
         "evidence": [{"kind": "source", "file": src["path"],
                       "source_sha256": "0" * 64, "claim": "x"},
                      {"kind": "trace_or_metamorphic", "claim": "y"}]},
        "source_sha256 가 패킷과 다르다")
    one({"decision_unit_id": order[0], "proposal": "named",
         "proposed_expr": "d_head",
         "evidence": [{"kind": "source", "file": src["path"],
                       "source_sha256": src["sha256"]},
                      {"kind": "trace_or_metamorphic", "claim": "y"}]},
        "source 근거에 `claim` 이 없다")
    one({"decision_unit_id": order[0], "proposal": "named",
         "proposed_expr": "d_head",
         "evidence": [{"kind": "추측", "claim": "x"},
                      {"kind": "trace_or_metamorphic", "claim": "y"}]},
        "모르는 evidence kind")

    print()
    print("3) 빠뜨림과 중복")
    write_answers([good(order[0], cur)])
    rows, errs, st, _p = I.ingest([pid], out)
    check("답이 없는 단위를 센다", any("답이 없는 단위" in e for e in errs))
    check(f"빠진 수가 맞는다 ({len(order) - 1})",
          st["답 빠진 단위"] == len(order) - 1)
    write_answers([good(order[0], cur), good(order[0], cur)])
    rows, errs, st, _p = I.ingest([pid], out)
    check("같은 단위를 두 번 답하면 잡는다",
          any("두 번 답했다" in e for e in errs))
    with io.open(os.path.join(pdir, "answers.jsonl"), "a",
                 encoding="utf-8", newline=NL) as f:
        f.write("{망가진 json" + NL)
    rows, errs, st, _p = I.ingest([pid], out)
    check("JSON 이 아닌 줄을 잡는다", any("JSON 이 아니다" in e for e in errs))

    print()
    print("4) 비교는 값을 근거로 쓰지 않는다")
    write_answers([good(order[0], cur)])
    rows, _e, _s, _p = I.ingest([pid], out)
    check("같은 식 -> same", rows[0]["comparison"] == "same")
    m = units[order[0]]["model"]
    import label_universe as U                                    # noqa: E402
    nm, _al = U.universe(m)
    other = next((n for n in sorted(nm)
                  if n not in ("ceil", "round", "min", "max", "roundup")
                  and n != cur), None)
    write_answers([good(order[0], other)])
    rows, _e, _s, _p = I.ingest([pid], out)
    check(f"다른 심볼({other}) -> same 이 아니다",
          rows[0]["comparison"] != "same")
    write_answers([good(order[0], "완전히모르는이름")])
    rows, _e, _s, _p = I.ingest([pid], out)
    check("모르는 이름 -> cannot_determine",
          rows[0]["comparison"] == "cannot_determine")

    print()
    print("5) 답이 없는 패킷은 오류가 아니다")
    os.remove(os.path.join(pdir, "answers.jsonl"))
    rows, errs, st, _p = I.ingest([pid], out)
    check("답 없음으로 센다", st.get("답이 아직 없는 패킷") == 1)
    check("오류로 세지 않는다", errs == [])
finally:
    X.LEDGER = real_ledger
    shutil.rmtree(tmp, ignore_errors=True)

print(NL + f"{len(OK)}/{len(OK) + len(FAIL)} 통과 — "
      "형식 검사가 실제로 발화한다")
if FAIL:
    print("실패: " + ", ".join(FAIL))
sys.exit(1 if FAIL else 0)
