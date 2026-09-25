r"""검토 세션이 낸 `answers.jsonl` 을 받아 **검증하고 지표를 낸다.** 아직 적용하지 않는다.

패킷 디렉터리마다 `answers.jsonl` 이 놓인다. 이 도구는

    1  스키마를 검사한다 (형식 오류를 세고, 어디가 틀렸는지 말한다)
    2  근거 두 종류 이상인지 본다 (`source` + `trace_or_metamorphic`)
    3  `source_sha256` 가 패킷의 frozen source 와 맞는지 본다
    4  패킷에 없는 단위나 빠진 단위를 잡는다
    5  **위치별** 지표를 낸다 -- 대장의 `unit_order` 로 위치를 알 수 있다
    6  현재 라벨과 비교해 `same / alias / different / cannot_determine` 을 붙인다

**적용은 하지 않는다.** 판정과 불일치 조정이 끝나기 전에는 overlay 를 적용하지 않는다는
계약이다(외부 검토 2026-09-25). 이 도구의 산출물은 `work/answers/` 아래에만 쓴다.

현재 라벨과의 비교는 `expr_compare` 가 한다 -- **값을 근거로 쓰지 않는다.**

실행:
    .venv\Scripts\python.exe develop\ingest_answers.py            모든 패킷
    .venv\Scripts\python.exe develop\ingest_answers.py P-xxxx…    하나만
"""
import collections
import hashlib
import hmac
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(PROJ, "src"))
if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import _buildguard                                               # noqa: E402
import expr_compare as E                                         # noqa: E402
import export_shard_packet as X                                  # noqa: E402
import label_universe as U                                       # noqa: E402

LAB = X.LAB
UNITS = os.path.join(LAB, "units")
OUT = os.path.join(LAB, "answers")

PROPOSALS = ("named", "no_name", "cannot_determine")
CONFIDENCE = ("low", "medium", "high")
EVIDENCE_KINDS = ("source", "trace_or_metamorphic")
REQUIRED = ("decision_unit_id", "proposal")


def answer_id(salt, packet_id, unit_id):
    """답 하나의 id. 답 **내용을 넣지 않는다** -- 고쳐 내도 같은 id 다."""
    mac = hmac.new(salt, f"answer:{packet_id}:{unit_id}".encode(),
                   hashlib.sha256)
    return "A-" + mac.hexdigest()[:12]


def load_units():
    """단위 -> (model, phase, 현재 식, grade). **bundle 밖의 자료다.**"""
    out = {}
    for f in sorted(os.listdir(UNITS)):
        if not f.endswith(".units.jsonl"):
            continue
        model, phase = f.replace(".units.jsonl", "").rsplit(".", 1)
        for u in (json.loads(l) for l in io.open(os.path.join(UNITS, f),
                                                 encoding="utf-8")):
            out[u["decision_unit_id"]] = {
                "model": model, "phase": phase,
                "current_expr": u["signature"]["old_expr"],
                "grade": u["grade"],
                "family": u["question_family_id"],
                "cells": u["affects_published_cells"]}
    return out


def check_one(a, i, packet_units, sources):
    """레코드 하나의 형식 문제 목록. **빈 목록이면 통과다.**"""
    errs = []
    if not isinstance(a, dict):
        return [f"{i}: 객체가 아니다"]
    for k in REQUIRED:
        if not a.get(k):
            errs.append(f"{i}: `{k}` 가 없다")
    uid = a.get("decision_unit_id")
    if uid and uid not in packet_units:
        errs.append(f"{i}: 이 패킷에 없는 단위 {uid}")
    p = a.get("proposal")
    if p and p not in PROPOSALS:
        errs.append(f"{i}: 모르는 proposal `{p}`")
    if p == "named" and not a.get("proposed_expr"):
        errs.append(f"{i}: named 인데 `proposed_expr` 가 없다")
    if p != "named" and a.get("proposed_expr"):
        errs.append(f"{i}: {p} 인데 `proposed_expr` 가 있다")
    c = a.get("confidence")
    if c and c not in CONFIDENCE:
        errs.append(f"{i}: 모르는 confidence `{c}`")
    ev = a.get("evidence") or []
    if not isinstance(ev, list):
        errs.append(f"{i}: `evidence` 가 목록이 아니다")
        ev = []
    kinds = {e.get("kind") for e in ev if isinstance(e, dict)}
    for e in ev:
        if not isinstance(e, dict):
            errs.append(f"{i}: evidence 항목이 객체가 아니다")
            continue
        if e.get("kind") not in EVIDENCE_KINDS:
            errs.append(f"{i}: 모르는 evidence kind `{e.get('kind')}`")
        if e.get("kind") == "source":
            f = (e.get("file") or "").replace("\\", "/")
            if f not in sources:
                errs.append(f"{i}: 패킷에 없는 source `{f}`")
            elif e.get("source_sha256") and e["source_sha256"] != sources[f]:
                errs.append(f"{i}: source_sha256 가 패킷과 다르다 ({f})")
            if not e.get("claim"):
                errs.append(f"{i}: source 근거에 `claim` 이 없다")
    # `named` 는 근거 두 종류를 요구한다. 판정을 보류한 답에는 요구하지 않는다.
    if p == "named" and len(kinds & set(EVIDENCE_KINDS)) < 2:
        errs.append(f"{i}: named 인데 근거가 한 종류다 ({sorted(kinds)})")
    return errs


def ingest(packet_ids=None, out_root=None):
    salt = X._salt()
    ledger = {r["packet_id"]: r for r in X.read_ledger()}
    units = load_units()
    uni = {}
    root = out_root or os.path.realpath(
        os.path.join(PROJ, "..", "llm-arch-tracer-review-packets"))
    ids = packet_ids or sorted(ledger)
    rows, errors, stats = [], [], collections.Counter()
    pos_stats = collections.defaultdict(collections.Counter)

    for pid in ids:
        rec = ledger.get(pid)
        if not rec:
            errors.append(f"{pid}: 대장에 없다")
            continue
        pdir = os.path.join(root, pid)
        ap = os.path.join(pdir, "answers.jsonl")
        if not os.path.exists(ap):
            stats["답이 아직 없는 패킷"] += 1
            continue
        pj = json.load(io.open(os.path.join(pdir, "_packet.json"),
                               encoding="utf-8"))
        sources = {s["path"]: s["sha256"] for s in pj["sources"]}
        order = rec["unit_order"]
        seen = set()
        for i, line in enumerate(io.open(ap, encoding="utf-8"), 1):
            if not line.strip():
                continue
            try:
                a = json.loads(line)
            except Exception as e:
                errors.append(f"{pid} {i}: JSON 이 아니다 ({e})")
                stats["json 오류"] += 1
                continue
            errs = check_one(a, i, set(order), sources)
            if errs:
                errors.extend(f"{pid} {e}" for e in errs)
                stats["형식 오류 레코드"] += 1
            uid = a.get("decision_unit_id")
            if uid in seen:
                errors.append(f"{pid} {i}: 같은 단위를 두 번 답했다 {uid}")
                stats["중복 답"] += 1
            seen.add(uid)
            u = units.get(uid) or {}
            pos = order.index(uid) + 1 if uid in order else None
            verdict, why = None, None
            # **`.get` 으로 읽는다.** `named` 인데 `proposed_expr` 가 없는 것은 잡아야
            # 하는 오류이고, 그 오류에서 수집기가 터지면 안 된다(자기검사가 잡았다).
            if (a.get("proposal") == "named" and a.get("proposed_expr")
                    and u.get("current_expr")):
                m = u["model"]
                if m not in uni:
                    uni[m] = U.universe(m)
                nm, al = uni[m]
                verdict, why = E.compare(a["proposed_expr"], u["current_expr"],
                                         nm, al)
            stats["답 총계"] += 1
            stats["proposal:" + str(a.get("proposal"))] += 1
            if verdict:
                stats["비교:" + verdict] += 1
            if pos:
                pos_stats[pos]["답"] += 1
                pos_stats[pos][str(a.get("proposal"))] += 1
                if errs:
                    pos_stats[pos]["형식 오류"] += 1
                if a.get("proposal") == "named":
                    pos_stats[pos]["근거 수"] += len(a.get("evidence") or [])
            rows.append({
                "answer_id": answer_id(salt, pid, uid or f"line{i}"),
                "packet_id": pid, "session_id": rec["session_id"],
                "decision_unit_id": uid, "position": pos,
                "role": rec["roles"].get(uid),
                "model": u.get("model"), "phase": u.get("phase"),
                "proposal": a.get("proposal"),
                "proposed_expr": a.get("proposed_expr"),
                "confidence": a.get("confidence"),
                "evidence_kinds": sorted({e.get("kind") for e in
                                          (a.get("evidence") or [])
                                          if isinstance(e, dict)}),
                "evidence": a.get("evidence") or [],
                "assumptions": a.get("assumptions") or [],
                "rejected_candidates": a.get("rejected_candidates") or [],
                "comparison": verdict, "comparison_reason": why,
                "format_errors": errs,
                "affects_published_cells": u.get("cells"),
                "grade_before": u.get("grade"),
                "question_family_id": u.get("family")})
        missing = [u for u in order if u not in seen]
        if missing:
            errors.append(f"{pid}: 답이 없는 단위 {len(missing)} 개")
            stats["답 빠진 단위"] += len(missing)
    return rows, errors, stats, pos_stats


def main():
    ids = [a for a in sys.argv[1:] if not a.startswith("--")] or None
    meta = _buildguard.require_clean_tree()
    _buildguard.stamp(meta, "develop/ingest_answers.py",
                      "develop/expr_compare.py", "develop/label_universe.py")
    rows, errors, stats, pos = ingest(ids)
    os.makedirs(OUT, exist_ok=True)
    with io.open(os.path.join(OUT, "answers.jsonl"), "w", encoding="utf-8",
                 newline=chr(10)) as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + chr(10))
    meta.update({
        "answers": len(rows), "errors": len(errors),
        "stats": dict(stats),
        "by_position": {str(k): dict(v) for k, v in sorted(pos.items())},
        "note": "적용하지 않았다 -- overlay 는 판정·조정이 끝난 뒤에만 적용한다"})
    json.dump(meta, io.open(os.path.join(OUT, "_ingest.json"), "w",
                            encoding="utf-8", newline=chr(10)),
              ensure_ascii=False, indent=1)
    print(f"답 {len(rows)} 건  형식·정합 문제 {len(errors)} 건")
    for k, v in sorted(stats.items()):
        print(f"  {k:<28}{v}")
    if pos:
        print()
        print(f"{'위치':>4}{'답':>5}{'named':>7}{'no_name':>9}"
              f"{'cannot_det':>12}{'형식오류':>9}{'평균 근거':>10}")
        for k, v in sorted(pos.items()):
            n = v.get("named", 0)
            print(f"{k:>4}{v['답']:>5}{n:>7}{v.get('no_name', 0):>9}"
                  f"{v.get('cannot_determine', 0):>12}{v.get('형식 오류', 0):>9}"
                  f"{(v.get('근거 수', 0) / n if n else 0):>10.1f}")
    if errors:
        print()
        print("문제:")
        for e in errors[:30]:
            print("  " + e)
        if len(errors) > 30:
            print(f"  ... 총 {len(errors)}")
    print()
    print(f"-> {os.path.relpath(OUT, PROJ)}  (적용하지 않았다)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
