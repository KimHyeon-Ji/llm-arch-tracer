r"""0-c 순서 1: 수집된 답을 **raw overlay 후보**로 펼친다. 적용하지 않는다.

`accepted.jsonl` 의 답 하나 -> 그 판정 단위의 발행 셀들 -> crosswalk 의 `raw_sites` ->
후보 레코드 여러 개. 키는 **원장 사이트 튜플**이다.

만들어지는 것은 전부 **actionable 이 아니다**(`approved_*` 상태는 사람 판정 뒤에만).
`resulting_expr` 와 `grade_after` 는 `None` 이다.

시작할 때 입력 계약을 **다시** 확인한다 -- `_ingest.json` 의 오류 0·거부 0·strict·
dirty_build, 입력 해시 재계산, `accepted.jsonl` 의 행별 표시와 revision
(외부 검토 2026-09-26). 하나라도 어긋나면 만들지 않는다.

실행:
    .venv\Scripts\python.exe develop\build_overlay_candidates.py
"""
import collections
import gzip
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import _buildguard                                               # noqa: E402
import export_shard_packet as X                                  # noqa: E402
import ingest_answers as I                                       # noqa: E402
import overlay_schema as S                                       # noqa: E402

LAB = X.LAB
CROSS = os.path.join(LAB, "crosswalk")
OUT = os.path.join(LAB, "overlay")


def crosswalk_index(model, phase):
    """`(op_id, field, shape_index, axis) -> (raw_sites, origin)`."""
    p = os.path.join(CROSS, f"{model}.{phase}.jsonl.gz")
    if not os.path.exists(p):
        return None, p
    idx = {}
    with gzip.open(p, "rt", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            idx[(c["op_id"], c["field"], c["shape_index"], c["axis"])] = (
                [tuple(t) for t in c["raw_sites"]], c.get("origin"))
    return idx, p


def build(ingest_dir=None, out_dir=None):
    ingest_dir = ingest_dir or I.OUT
    am, _bman = X.load_assignment()
    rev = am["assignment_revision"]

    # ---- 입력 계약. 어긋나면 만들지 않는다.
    errs, rows = S.check_input_contract(ingest_dir, rev)
    if errs:
        return None, errs, {}, []

    units = {}
    for f in sorted(os.listdir(os.path.join(LAB, "units"))):
        if not f.endswith(".units.jsonl"):
            continue
        for u in (json.loads(l) for l in
                  io.open(os.path.join(LAB, "units", f), encoding="utf-8")):
            units[u["decision_unit_id"]] = u

    inputs = [os.path.join(ingest_dir, "_ingest.json"),
              os.path.join(ingest_dir, "accepted.jsonl"), X.ASSIGN]
    idx_cache, cands, stats = {}, [], collections.Counter()
    for r in rows:
        uid = r["decision_unit_id"]
        u = units.get(uid)
        if u is None:
            errs.append(f"{uid}: 판정 단위를 찾을 수 없다")
            continue
        key = (r["model"], r["phase"])
        if key not in idx_cache:
            idx, p = crosswalk_index(*key)
            if idx is None:
                errs.append(f"crosswalk 이 없다 {p}")
                idx_cache[key] = {}
                continue
            idx_cache[key] = idx
            inputs.append(p)
        idx = idx_cache[key]
        state, verdict = S.state_of(r["proposal"], r.get("comparison"))
        stats["state:" + state] += 1
        for cell in u["published_cells"]:
            ck = tuple(cell)
            got = idx.get(ck)
            if got is None:
                errs.append(f"{uid}: crosswalk 에 없는 발행 셀 {ck}")
                continue
            sites, origin = got
            if not sites:
                errs.append(f"{uid}: 발행 셀 {ck} 에 raw 사이트가 없다")
                continue
            for site in sites:
                rec = {
                    "schema_version": S.SCHEMA_VERSION,
                    "site": [r["model"], r["phase"], site[0], site[1],
                             site[2], site[3]],
                    "model": r["model"], "phase": r["phase"],
                    "decision_unit_id": uid,
                    "question_family_id": r.get("question_family_id"),
                    "answer_id": r["answer_id"],
                    "submission_sha256": r["submission_sha256"],
                    "packet_id": r["packet_id"], "session_id": r["session_id"],
                    "assignment_revision": r["assignment_revision"],
                    "role": r.get("role"),
                    "proposal": r["proposal"],
                    "proposed_expr": r.get("proposed_expr"),
                    "current_expr": u["signature"]["old_expr"],
                    "comparison": r.get("comparison"),
                    "comparison_reason": r.get("comparison_reason"),
                    "verdict_candidate": verdict,
                    "state": state,
                    "actionable": False,
                    # **actionable 이 아니므로 결과를 쓰지 않는다**
                    "resulting_expr": None,
                    "grade_after": None,
                    "grade_before": r.get("grade_before"),
                    "origin": origin,
                    "published_cell": list(cell),
                    "evidence": r.get("evidence") or [],
                    "assumptions": r.get("assumptions") or [],
                    "confidence": r.get("confidence"),
                }
                bad = S.validate(rec)
                if bad:
                    errs.extend(f"{uid} {site}: {b}" for b in bad)
                    stats["스키마 위반"] += 1
                    continue
                cands.append(rec)
                stats["후보"] += 1
                stats["origin:" + str(origin)] += 1
    # 같은 사이트에 후보가 둘 이상이면 중복 배정이다 -- 순서 2 에서 조정한다
    per_site = collections.Counter(tuple(c["site"]) for c in cands)
    stats["사이트"] = len(per_site)
    stats["중복 배정된 사이트"] = sum(1 for v in per_site.values() if v > 1)
    return cands, errs, stats, inputs


def main():
    meta = _buildguard.require_clean_tree()
    _buildguard.stamp(meta, "develop/build_overlay_candidates.py",
                      "develop/overlay_schema.py")
    cands, errs, stats, inputs = build()
    if cands is None:
        print("**입력 계약을 어겼다 -- 후보를 만들지 않는다**", file=sys.stderr)
        for e in errs[:20]:
            print("   " + e, file=sys.stderr)
        return 2
    os.makedirs(OUT, exist_ok=True)
    with io.open(os.path.join(OUT, "candidates.jsonl"), "w", encoding="utf-8",
                 newline=chr(10)) as f:
        for c in cands:
            f.write(json.dumps(c, ensure_ascii=False) + chr(10))
    meta.update({
        "schema_version": S.SCHEMA_VERSION,
        "candidates": len(cands), "errors": len(errs),
        "stats": dict(stats),
        "actionable": sum(1 for c in cands if c["actionable"]),
        "states": S.STATES, "actionable_states": S.ACTIONABLE,
        "note": "적용하지 않았다. actionable 은 사람 판정 뒤에만 생긴다"})
    meta["input_sha256"] = _buildguard.input_manifest(inputs)
    json.dump(meta, io.open(os.path.join(OUT, "_candidates.json"), "w",
                            encoding="utf-8", newline=chr(10)),
              ensure_ascii=False, indent=1)
    print(f"후보 {len(cands):,} 건  (사이트 {stats.get('사이트', 0):,})")
    for k, v in sorted(stats.items()):
        print(f"  {k:<26}{v:,}")
    print(f"  actionable                {meta['actionable']}")
    if errs:
        print()
        print(f"문제 {len(errs)} 건:")
        for e in errs[:20]:
            print("  " + e)
    print()
    print(f"-> {os.path.relpath(OUT, PROJ)}  (적용하지 않았다)")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
