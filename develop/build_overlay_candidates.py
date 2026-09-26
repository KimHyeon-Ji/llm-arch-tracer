r"""0-c 순서 1: 수집된 답을 **raw overlay 후보**로 펼친다. 적용하지 않는다.

`accepted.jsonl` 의 답 하나 -> 그 판정 단위의 발행 셀들 -> crosswalk 의 `raw_sites` ->
`answer_candidate` 레코드 여러 개. 키는 **원장 사이트 튜플**이다.

만들어지는 것은 **`answer_candidate` 뿐**이고 전부 actionable 이 아니다. `final_verdict`·
`resulting_expr`·`grade_after` 는 아예 넣지 않는다 -- 그것은 `unit_adjudication`(순서 2)과
`raw_overlay`(순서 3)의 필드다.

**어떤 실패 경로에서도 공개 후보를 무효화한다.** 전에는 입력 계약·사전 해시 실패에서
곧바로 돌아가 **옛 `candidates.jsonl` 이 그대로 남았고**, 후속 단계가 그 고정 경로를
읽으면 낡은 후보를 쓴다(외부 검토 2026-09-26). 지금은

    성공        임시 파일에 다 쓴 뒤 `os.replace` 로 원자적으로 공개한다
    모든 실패   공개 파일을 **지운다**. 진단은 `_rejected.jsonl` 로만

`units` 와 `crosswalk` 는 읽기 **전에** 배정 manifest 의 해시와 대조한다. 그리고
**기대 집합을 manifest 에서 결정적으로 뽑아** `기대 == 실제 == 검증` 을 확인한다 --
디렉터리에 있는 것만 열거하면 파일이 사라진 것을 못 본다.

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
    """`(op_id, field, shape_index, axis) -> (raw_sites, origin)`.

    **중복 키는 즉시 오류다.** dict 로 조용히 덮어쓰면 어느 행을 읽었는지 알 수 없다
    (외부 검토 2026-09-26).
    """
    p = os.path.join(CROSS, f"{model}.{phase}.jsonl.gz")
    if not os.path.exists(p):
        return None, p, [f"crosswalk 이 없다 {p}"]
    idx, dup, errs = {}, [], []
    with gzip.open(p, "rt", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            k = (c["op_id"], c["field"], c["shape_index"], c["axis"])
            if k in idx:
                dup.append(k)
                continue
            idx[k] = ([tuple(t) for t in c["raw_sites"]], c.get("origin"))
    if dup:
        errs.append(f"crosswalk 에 중복 셀 키 {len(dup)} 개 ({p}): {dup[:3]}")
    return idx, p, errs


def expected_from_manifest(trusted, kind):
    """배정 manifest 의 키에서 **기대 경로 집합**을 결정적으로 뽑는다.

    디렉터리를 열거하면 파일이 사라진 것 자체를 검사하지 못한다(외부 검토 2026-09-26).
    `kind` 는 `"units"` 또는 `"crosswalk"`.
    """
    out = set()
    for rel in trusted:
        if kind == "units" and rel.endswith(".units.jsonl"):
            out.add(os.path.normpath(os.path.join(PROJ, rel)))
        elif kind == "crosswalk" and "/crosswalk/" in rel and rel.endswith(".jsonl.gz"):
            out.add(os.path.normpath(os.path.join(PROJ, rel)))
    return out


def check_set(expected, actual, verified, label):
    """`기대 == 실제 == 검증` 인가. 어긋나는 것을 이름으로 말한다."""
    errs = []
    for name, a, b in (("실제", expected, actual), ("검증", expected, verified)):
        miss = sorted(a - b)
        extra = sorted(b - a)
        if miss:
            errs.append(f"{label}: 기대에 있으나 {name}에 없는 것 {len(miss)} "
                        f"({[os.path.basename(x) for x in miss[:3]]})")
        if extra:
            errs.append(f"{label}: {name}에만 있는 것 {len(extra)} "
                        f"({[os.path.basename(x) for x in extra[:3]]})")
    return errs


def build(ingest_dir=None, out_dir=None):
    ingest_dir = ingest_dir or I.OUT
    am, _bman = X.load_assignment()
    rev = am["assignment_revision"]
    trusted = am.get("input_sha256") or {}

    # ---- 입력 계약. 어긋나면 만들지 않는다.
    errs, rows = S.check_input_contract(ingest_dir, rev)
    if errs:
        return None, errs, {}, []

    # ---- units: **기대 집합 == 실제 == 검증** 을 확인한 뒤 읽는다
    u_exp = expected_from_manifest(trusted, "units")
    u_act = {os.path.normpath(os.path.join(LAB, "units", f))
             for f in os.listdir(os.path.join(LAB, "units"))
             if f.endswith(".units.jsonl")}
    if not u_exp:
        return None, ["배정 manifest 에 units 경로가 없다"], {}, []
    upaths = sorted(u_exp)
    verrs = S.verify_against_manifest(upaths, trusted, "units: ")
    u_ver = {p for p in upaths
             if not any(os.path.basename(p) in e for e in verrs)}
    verrs += check_set(u_exp, u_act, u_ver, "units")
    if verrs:
        return None, verrs, {}, []
    units = {}
    for p in upaths:
        for u in (json.loads(l) for l in io.open(p, encoding="utf-8")):
            units[u["decision_unit_id"]] = u

    inputs = [os.path.join(ingest_dir, "_ingest.json"),
              os.path.join(ingest_dir, "accepted.jsonl"), X.ASSIGN] + upaths
    # ---- crosswalk: 이 실행이 **쓸** 집합을 accepted 행에서 결정적으로 뽑고,
    #      그것이 manifest 기대 집합의 부분집합인지 본다. 실제로 쓴 것과 검증한 것도
    #      맞아야 한다(아래 루프 뒤에서 확인).
    c_exp = expected_from_manifest(trusted, "crosswalk")
    need = {os.path.normpath(os.path.join(
        CROSS, f"{r['model']}.{r['phase']}.jsonl.gz")) for r in rows}
    outside = sorted(need - c_exp)
    if outside:
        return None, [f"crosswalk: manifest 기대 집합 밖의 파일이 필요하다 "
                      f"{[os.path.basename(x) for x in outside[:3]]}"], {}, []

    idx_cache, cands, stats = {}, [], collections.Counter()
    c_used, c_ver = set(), set()
    for r in rows:
        uid = r["decision_unit_id"]
        u = units.get(uid)
        if u is None:
            errs.append(f"{uid}: 판정 단위를 찾을 수 없다")
            continue
        key = (r["model"], r["phase"])
        if key not in idx_cache:
            cp = os.path.join(CROSS, f"{key[0]}.{key[1]}.jsonl.gz")
            # crosswalk 도 **읽기 전에** 신뢰 해시와 대조한다
            cpn = os.path.normpath(cp)
            c_used.add(cpn)
            verrs = S.verify_against_manifest([cp], trusted, "crosswalk: ")
            if verrs:
                errs.extend(verrs)
                idx_cache[key] = {}
                continue
            c_ver.add(cpn)
            idx, p, cerrs = crosswalk_index(*key)
            if cerrs:
                errs.extend(cerrs)
            if idx is None:
                idx_cache[key] = {}
                continue
            idx_cache[key] = idx
            inputs.append(p)
        idx = idx_cache[key]
        state, ckind = S.state_of(r["proposal"], r.get("comparison"))
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
                    "kind": "answer_candidate",
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
                    "candidate_kind": ckind,
                    "state": state,
                    # **answer_candidate 는 절대 actionable 이 아니고, 결과 필드를
                    # 아예 갖지 않는다.** 그것은 조정·최종 레코드의 몫이다.
                    "actionable": False,
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
    # **쓸 것 == 쓴 것 == 검증한 것**
    errs += check_set(need, c_used, c_ver, "crosswalk")
    stats["units 검증"] = len(upaths)
    stats["crosswalk 검증"] = len(c_ver)
    # 같은 사이트에 후보가 둘 이상이면 중복 배정이다 -- 순서 2 에서 조정한다
    per_site = collections.Counter(tuple(c["site"]) for c in cands)
    stats["사이트"] = len(per_site)
    # 중복 조정은 **판정 단위 단위**로 한다(승인된 Q2). 사이트 수는 영향 범위 지표다.
    per_unit = collections.defaultdict(set)
    for c in cands:
        per_unit[c["decision_unit_id"]].add(c["answer_id"])
    stats["단위"] = len(per_unit)
    stats["중복 답이 있는 단위"] = sum(1 for v in per_unit.values() if len(v) > 1)
    return cands, errs, stats, inputs


def main():
    meta = _buildguard.require_clean_tree()
    _buildguard.stamp(meta, "develop/build_overlay_candidates.py",
                      "develop/overlay_schema.py")
    cands, errs, stats, inputs = build()
    pub = os.path.join(OUT, "candidates.jsonl")

    def invalidate(why, code):
        """**어떤 실패에서도 공개 후보를 무효화한다.** 낡은 후보가 쓰이지 않게."""
        os.makedirs(OUT, exist_ok=True)
        if os.path.exists(pub):
            os.remove(pub)
        print(f"**{why} -- 공개 후보를 무효화했다**", file=sys.stderr)
        for e in errs[:20]:
            print("   " + e, file=sys.stderr)
        json.dump({**meta, "errors": len(errs), "error_detail": errs[:50],
                   "candidates_written": False, "invalidated": True,
                   "reason": why},
                  io.open(os.path.join(OUT, "_candidates.json"), "w",
                          encoding="utf-8", newline=chr(10)),
                  ensure_ascii=False, indent=1)
        return code

    if cands is None:
        return invalidate("입력 계약·사전 대조 실패", 2)
    os.makedirs(OUT, exist_ok=True)
    if errs:
        with io.open(os.path.join(OUT, "_rejected.jsonl"), "w",
                     encoding="utf-8", newline=chr(10)) as f:
            for c in cands:
                f.write(json.dumps(c, ensure_ascii=False) + chr(10))
        return invalidate(f"생성 중 문제 {len(errs)} 건", 3)
    # ---- **원자적으로 공개하고 metadata 에 결박한다.** 후보 파일의 digest 와 행 수를
    #      적지 않으면, 성공 뒤 proposal·state 를 바꿔도 다음 단계가 읽는다
    #      (외부 검토 2026-09-26. accepted.jsonl 에서 고친 것과 같은 구멍).
    tmp = pub + ".tmp"
    with io.open(tmp, "w", encoding="utf-8", newline=chr(10)) as f:
        for c in cands:
            f.write(json.dumps(c, ensure_ascii=False) + chr(10))
    os.replace(tmp, pub)
    if os.path.exists(os.path.join(OUT, "_rejected.jsonl")):
        os.remove(os.path.join(OUT, "_rejected.jsonl"))
    meta.update({
        "candidates_written": True,
        "schema_version": S.SCHEMA_VERSION,
        "candidates": len(cands), "errors": len(errs),
        "stats": dict(stats),
        "actionable": sum(1 for c in cands if c["actionable"]),
        "record_kinds": list(S.KINDS),
        "answer_states": list(S.ANSWER_STATES),
        "unit_states": list(S.UNIT_STATES),
        "actionable_states": list(S.ACTIONABLE),
        "note": "적용하지 않았다. actionable 은 사람 판정 뒤에만 생긴다"})
    meta["input_sha256"] = _buildguard.input_manifest(inputs)
    meta["candidates_rows"] = len(cands)
    meta["candidates_sha256"] = _buildguard.sha256_file(pub)
    # metadata 도 임시 파일 뒤 교체한다
    mp = os.path.join(OUT, "_candidates.json")
    mtmp = mp + ".tmp"
    json.dump(meta, io.open(mtmp, "w", encoding="utf-8", newline=chr(10)),
              ensure_ascii=False, indent=1)
    os.replace(mtmp, mp)
    print(f"후보 {len(cands):,} 건  (사이트 {stats.get('사이트', 0):,})")
    for k, v in sorted(stats.items()):
        print(f"  {k:<26}{v:,}")
    print(f"  actionable                {meta['actionable']}")
    print()
    print(f"-> {os.path.relpath(OUT, PROJ)}  (적용하지 않았다)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
