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

## fail-closed 다

처음에는 오류를 발견해도 그 레코드를 **같은 파일에** 쓰고 종료 코드 0 을 냈다. 후속
단계가 `format_errors` 필터를 한 번 빠뜨리면 잘못된 답이 overlay 로 들어간다
(외부 검토 2026-09-25). 그래서

    accepted.jsonl   `eligible_for_adjudication: true`  **overlay 입력기는 이것만 받는다**
    rejected.jsonl   거부된 답과 그 이유
    종료 코드         패킷을 **지정**했으면 오류·중복·누락이 있으면 0 이 아니다
                     (지정 없이 전체를 점검할 때는 "답이 아직 없음" 은 예외)

패킷 자체도 다시 검증한다 -- `_packet.json` 의 해시를 믿는 데서 끝내지 않고 **frozen
source 파일을 다시 해시**하고, 대장의 `unit_order`·`assignment_revision`·packet id
결합과 `shard.md` 의 단위 순서를 다시 확인한다.

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
import re
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

BT = chr(96)
BS = chr(92)
LAB = X.LAB
UNITS = os.path.join(LAB, "units")
OUT = os.path.join(LAB, "answers")
# 반출 위치. **모듈 상수로 둔다** -- 자기검사가 실제 패킷을 읽지 않도록 갈아끼운다.
PACKETS = os.path.realpath(os.path.join(PROJ, "..",
                                        "llm-arch-tracer-review-packets"))

PROPOSALS = ("named", "no_name", "cannot_determine")
CONFIDENCE = ("low", "medium", "high")
EVIDENCE_KINDS = ("source", "trace_or_metamorphic")
REQUIRED = ("decision_unit_id", "proposal", "confidence")
# 근거 종류마다 **필수 필드**. 하나라도 없으면 그 근거는 종류로 세지 않는다.
EV_REQUIRED = {"source": ("file", "lines", "source_sha256", "claim"),
               "trace_or_metamorphic": ("artifact", "claim")}
# 근거 두 종류를 요구하는 proposal. `no_name` 도 주장이므로 포함한다.
# `cannot_determine` 은 판정을 보류한 것이어서 요구하지 않는다.
NEEDS_EVIDENCE = ("named", "no_name")
LIST_FIELDS = ("evidence", "assumptions", "rejected_candidates")


def answer_id(salt, packet_id, unit_id):
    """답 하나의 id. 답 **내용을 넣지 않는다** -- 고쳐 내도 같은 id 다."""
    mac = hmac.new(salt, f"answer:{packet_id}:{unit_id}".encode(),
                   hashlib.sha256)
    return "A-" + mac.hexdigest()[:12]


def submission_sha256(a):
    """**제출 내용**의 해시. `answer_id` 는 재제출을 같은 단위로 묶는 키일 뿐이어서
    이력을 구분하지 못한다(외부 검토 2026-09-25). 이 값이 바뀌면 다른 제출이다."""
    return hashlib.sha256(json.dumps(a, ensure_ascii=False,
                                     sort_keys=True).encode()).hexdigest()


def verify_packet(pdir, rec, am):
    """패킷이 **이 배정의 것인가.** `(문제 목록, 실제 source 해시)`.

    `_packet.json` 의 해시를 믿는 데서 끝내면 패킷 안에서 source 를 고쳐도 모른다.
    실제 파일을 다시 해시하고, 대장·배정 manifest 와의 결합도 다시 본다.
    """
    errs = []
    base = os.path.basename(pdir)
    pjp = os.path.join(pdir, "_packet.json")
    if not os.path.exists(pjp):
        return [base + ": _packet.json 이 없다"], {}
    pj = json.load(io.open(pjp, encoding="utf-8"))
    if pj.get("packet_id") != rec["packet_id"]:
        errs.append("packet_id 가 대장과 다르다 (" + str(pj.get("packet_id")) + ")")
    if rec.get("assignment_revision") != am.get("assignment_revision"):
        errs.append("대장의 revision " + str(rec.get("assignment_revision"))
                    + " != 배정 " + str(am.get("assignment_revision")))
    entry = next((e for e in am["manifest"] if e["shard"] == rec["shard"]), None)
    if entry is None:
        errs.append("배정 manifest 에 없는 shard " + str(rec["shard"]))
    elif sorted(entry["unit_ids"]) != sorted(rec["unit_order"]):
        errs.append("대장의 unit_order 가 배정 manifest 의 단위 집합과 다르다")
    if pj.get("units") != len(rec["unit_order"]):
        errs.append("_packet.json 의 units " + str(pj.get("units")) + " != "
                    + str(len(rec["unit_order"])))
    # **실제 파일을 다시 해시한다** -- 기록값을 그대로 믿지 않는다
    sources = {}
    for sm in pj.get("sources") or []:
        fp = os.path.join(pdir, sm["path"])
        if not os.path.exists(fp):
            errs.append("source 파일이 없다 " + sm["path"])
            continue
        real = _buildguard.sha256_file(fp)
        if real != sm.get("sha256"):
            errs.append("source 가 바뀌었다 " + sm["path"] + " (기록 "
                        + str(sm.get("sha256"))[:12] + "... 실제 " + real[:12] + "...)")
        sources[sm["path"]] = real             # 기록값이 아니라 **실제값**
    sp = os.path.join(pdir, "shard.md")
    if os.path.exists(sp):
        doc = io.open(sp, encoding="utf-8").read()
        got = re.findall("^### (" + BS + "S+)",
                         doc.split("## 심볼 정의와 config 값")[0], flags=re.M)
        if got != rec["unit_order"]:
            errs.append("shard.md 의 단위 순서가 대장과 다르다")
    return errs, sources


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
        if a.get(k) in (None, "", [], {}):
            errs.append(str(i) + ": " + BT + k + BT + " 가 없다")
    for k in LIST_FIELDS:
        if k in a and not isinstance(a[k], list):
            errs.append(str(i) + ": " + BT + k + BT + " 가 목록이 아니다")
    for k in ("assumptions", "rejected_candidates"):
        if k not in a:
            errs.append(str(i) + ": " + BT + k + BT
                        + " 가 없다 (없으면 빈 목록으로 적어라)")
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
    ev = a.get("evidence")
    if ev is None:
        errs.append(str(i) + ": " + BT + "evidence" + BT + " 가 없다")
        ev = []
    if not isinstance(ev, list):
        ev = []
    # **필수 필드를 모두 통과한 근거만** 종류로 센다. 빈껍데기 두 개로 요건을 채우지
    # 못하게 한다(외부 검토 2026-09-25).
    good_kinds = set()
    for e in ev:
        if not isinstance(e, dict):
            errs.append(str(i) + ": evidence 항목이 객체가 아니다")
            continue
        kind = e.get("kind")
        if kind not in EVIDENCE_KINDS:
            errs.append(str(i) + ": 모르는 evidence kind " + BT + str(kind) + BT)
            continue
        miss = [k for k in EV_REQUIRED[kind] if not e.get(k)]
        if miss:
            errs.append(str(i) + ": " + kind + " 근거에 "
                        + ", ".join(BT + m + BT for m in miss) + " 가 없다")
        if kind == "source":
            f = (e.get("file") or "").replace(BS, "/")
            if f not in sources:
                errs.append(str(i) + ": 패킷에 없는 source " + BT + f + BT)
                continue
            if e.get("source_sha256") and e["source_sha256"] != sources[f]:
                errs.append(str(i) + ": source_sha256 가 실제 파일과 다르다 ("
                            + f + ")")
                continue
        if not miss:
            good_kinds.add(kind)
    if p in NEEDS_EVIDENCE and len(good_kinds) < 2:
        errs.append(str(i) + ": " + str(p) + " 인데 온전한 근거가 "
                    + str(len(good_kinds)) + " 종류다 (" + str(sorted(good_kinds))
                    + ")")
    return errs


def ingest(packet_ids=None, out_root=None):
    salt = X._salt()
    am, _bman = X.load_assignment()            # 연결 해시·revision 을 다시 확인한다
    inputs = [X.LEDGER, X.ASSIGN]
    ledger = {r["packet_id"]: r for r in X.read_ledger()}
    units = load_units()
    uni = {}
    root = out_root or PACKETS
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
        inputs.append(ap)
        # **패킷을 먼저 검증한다.** source 를 다시 해시하고 대장·배정과의 결합을 본다.
        perrs, sources = verify_packet(pdir, rec, am)
        if perrs:
            errors.extend(pid + ": " + e for e in perrs)
            stats["패킷 검증 실패"] += 1
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
            eligible = (not errs) and uid in order and not perrs
            rows.append({
                "eligible_for_adjudication": eligible,
                "answer_id": answer_id(salt, pid, uid or f"line{i}"),
                "submission_sha256": submission_sha256(a),
                "assignment_revision": rec.get("assignment_revision"),
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
                "question_family_id": u.get("family"),
                "packet_errors": perrs})
            stats["eligible" if eligible else "rejected"] += 1
        missing = [u for u in order if u not in seen]
        if missing:
            errors.append(f"{pid}: 답이 없는 단위 {len(missing)} 개")
            stats["답 빠진 단위"] += len(missing)
    return rows, errors, stats, pos_stats, inputs


def main():
    ids = [a for a in sys.argv[1:] if not a.startswith("--")] or None
    meta = _buildguard.require_clean_tree()
    _buildguard.stamp(meta, "develop/ingest_answers.py",
                      "develop/expr_compare.py", "develop/label_universe.py")
    rows, errors, stats, pos, inputs = ingest(ids)
    os.makedirs(OUT, exist_ok=True)
    # **유효한 답과 거부된 답을 나눈다.** overlay 입력기는 accepted.jsonl 만 읽는다.
    acc = [r for r in rows if r["eligible_for_adjudication"]]
    rej = [r for r in rows if not r["eligible_for_adjudication"]]
    for name, part in (("accepted.jsonl", acc), ("rejected.jsonl", rej)):
        with io.open(os.path.join(OUT, name), "w", encoding="utf-8",
                     newline=chr(10)) as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + chr(10))
    meta.update({
        "answers": len(rows), "accepted": len(acc), "rejected": len(rej),
        "errors": len(errors), "stats": dict(stats),
        "by_position": {str(k): dict(v) for k, v in sorted(pos.items())},
        "packets_requested": ids or "전체 점검",
        "strict": bool(ids),
        "note": "적용하지 않았다 -- overlay 는 판정·조정이 끝난 뒤에만 적용한다",
        "overlay_input": "accepted.jsonl 만 쓴다 (eligible_for_adjudication=true)"})
    meta["input_sha256"] = _buildguard.input_manifest(inputs)
    json.dump(meta, io.open(os.path.join(OUT, "_ingest.json"), "w",
                            encoding="utf-8", newline=chr(10)),
              ensure_ascii=False, indent=1)
    print(f"답 {len(rows)} 건  유효 {len(acc)}  거부 {len(rej)}  "
          f"문제 {len(errors)} 건")
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
    print(f"   accepted.jsonl {len(acc)}  rejected.jsonl {len(rej)}"
          f"   입력 해시 {len(meta['input_sha256'])} 파일")
    # **fail-closed.** 패킷을 지정했으면 문제가 있으면 0 이 아니다. 지정 없이 전체를
    # 점검할 때는 "답이 아직 없음" 만으로 실패시키지 않는다.
    if errors:
        print()
        print(f"**문제 {len(errors)} 건 -- 종료 코드 1**" if ids
              else f"(전체 점검: 문제 {len(errors)} 건, 종료 코드 0)")
        return 1 if ids else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
