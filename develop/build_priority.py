r"""1 단계 검토 우선순위를 정한다. **"상위 N 개" 로 자르면 안 된다.**

승인된 기준(외부 검토 2026-09-25) — 아래 **합집합**:

  1. `open_tie` 전부 + `heuristic` 전부            (위험 등급)
  2. 발행 영향이 있는 family 49 개 각각 최소 1 개
  3. 모델·phase·layer cohort 별 최소 표본
  4. 그 뒤 발행 영향이 큰 순서로 추가해 **발행 셀 95% 이상** 도달

위험 등급만 합쳐도 432 단위인데 발행 셀로는 640 개뿐이다. 그래서 "발행 영향 상위 400" 만
보면 위험한 자리를 놓친다. 그리고 현재 shard 는 층화 셔플돼 있으므로 "상위 400 = 34 shard"
같은 계산도 성립하지 않는다 -- **별도의 priority shard** 가 필요하다.

실행:
    .venv\Scripts\python.exe develop\build_priority.py [coverage] [per_cohort]
"""
import collections
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

LAB = os.path.join(PROJ, "..", "llm-arch-tracer-results-labeled", "work")
UNITS = os.path.join(LAB, "units")
OUT = os.path.join(LAB, "priority")


def main():
    coverage = float(sys.argv[1]) if len(sys.argv) > 1 else 0.95
    per_cohort = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    meta = _buildguard.require_clean_tree()
    _buildguard.stamp(meta, "develop/build_priority.py", "develop/_buildguard.py")

    units, inputs = [], []
    for f in sorted(os.listdir(UNITS)):
        if not f.endswith(".units.jsonl"):
            continue
        p = os.path.join(UNITS, f)
        inputs.append(p)
        phase = f.replace(".units.jsonl", "").rsplit(".", 1)[1]
        for u in (json.loads(l) for l in io.open(p, encoding="utf-8")):
            if u["affects_published_cells"] > 0:
                u["_phase"] = phase
                units.append(u)
    total_cells = sum(u["affects_published_cells"] for u in units)

    chosen, why = {}, collections.defaultdict(list)

    def take(u, reason):
        uid = u["decision_unit_id"]
        chosen[uid] = u
        why[uid].append(reason)

    # 1) 위험 등급 전부
    for u in units:
        if u["grade"] in ("open_tie", "heuristic"):
            take(u, f"risk_grade:{u['grade']}")

    # 2) family 마다 최소 1 개 (영향 큰 것부터)
    by_fam = collections.defaultdict(list)
    for u in units:
        by_fam[u["question_family_id"]].append(u)
    for fam, us in by_fam.items():
        if not any(u["decision_unit_id"] in chosen for u in us):
            take(max(us, key=lambda x: x["affects_published_cells"]), "family_min1")

    # 3) 모델·phase·cohort 별 최소 표본
    by_coh = collections.defaultdict(list)
    for u in units:
        by_coh[(u["model"], u["_phase"],
                u["signature"]["layer_cohort_id"])].append(u)
    for key, us in by_coh.items():
        have = sum(1 for u in us if u["decision_unit_id"] in chosen)
        if have < per_cohort:
            for u in sorted(us, key=lambda x: -x["affects_published_cells"]):
                if u["decision_unit_id"] not in chosen:
                    take(u, "cohort_min")
                    have += 1
                    if have >= per_cohort:
                        break

    # 4) 발행 영향 순으로 coverage 까지
    acc = sum(u["affects_published_cells"] for u in chosen.values())
    for u in sorted(units, key=lambda x: -x["affects_published_cells"]):
        if acc / total_cells >= coverage:
            break
        if u["decision_unit_id"] not in chosen:
            take(u, "impact_coverage")
            acc += u["affects_published_cells"]

    sel = sorted(chosen.values(), key=lambda x: -x["affects_published_cells"])
    picked_cells = sum(u["affects_published_cells"] for u in sel)
    rest = [u for u in units if u["decision_unit_id"] not in chosen]

    os.makedirs(OUT, exist_ok=True)
    with io.open(os.path.join(OUT, "stage1_units.jsonl"), "w",
                 encoding="utf-8", newline=chr(10)) as f:
        for u in sel:
            f.write(json.dumps({
                "decision_unit_id": u["decision_unit_id"],
                "question_family_id": u["question_family_id"],
                "model": u["model"], "phase": u["_phase"],
                "grade": u["grade"],
                "layer_cohort_id": u["signature"]["layer_cohort_id"],
                "affects_published_cells": u["affects_published_cells"],
                "reasons": sorted(set(why[u["decision_unit_id"]]))},
                ensure_ascii=False) + chr(10))
    # 미검토로 남는 것 -- UNKNOWNS.md 에 실을 집계
    pend = collections.Counter()
    pend_cells = collections.Counter()
    for u in rest:
        k = (u["model"], u["_phase"], u["grade"])
        pend[k] += 1
        pend_cells[k] += u["affects_published_cells"]
    with io.open(os.path.join(OUT, "pending_independent_review.jsonl"), "w",
                 encoding="utf-8", newline=chr(10)) as f:
        for k in sorted(pend):
            f.write(json.dumps({"model": k[0], "phase": k[1], "grade": k[2],
                                "units": pend[k], "affects_published_cells": pend_cells[k],
                                "status": "pending_independent_review"},
                               ensure_ascii=False) + chr(10))

    meta.update({
        "coverage_target": coverage, "per_cohort_min": per_cohort,
        "units_total": len(units), "cells_total": total_cells,
        "stage1_units": len(sel), "stage1_cells": picked_cells,
        "stage1_cell_coverage": round(picked_cells / total_cells, 4),
        "pending_units": len(rest),
        "pending_cells": total_cells - picked_cells,
        "reason_counts": dict(collections.Counter(
            r for uid in chosen for r in set(why[uid]))),
        "risk_grade_units": sum(1 for u in units
                                if u["grade"] in ("open_tie", "heuristic")),
        "families_covered": len({u["question_family_id"] for u in sel}),
        "families_total": len(by_fam),
        "cohorts_covered": len({(u["model"], u["_phase"],
                                 u["signature"]["layer_cohort_id"]) for u in sel}),
        "cohorts_total": len(by_coh),
    })
    meta["input_sha256"] = _buildguard.input_manifest(inputs)
    json.dump(meta, io.open(os.path.join(OUT, "_priority.json"), "w",
                            encoding="utf-8", newline=chr(10)),
              ensure_ascii=False, indent=1)
    print(f"1 단계 대상 {len(sel):,} / 전체 {len(units):,} 단위")
    print(f"  발행 셀 {picked_cells:,} / {total_cells:,} "
          f"({100 * picked_cells / total_cells:.1f}%)")
    print(f"  family {meta['families_covered']}/{meta['families_total']}  "
          f"cohort {meta['cohorts_covered']}/{meta['cohorts_total']}")
    print(f"  선정 이유 {json.dumps(meta['reason_counts'], ensure_ascii=False)}")
    print(f"미검토로 남는 것 {len(rest):,} 단위 / "
          f"{total_cells - picked_cells:,} 셀  (pending_independent_review)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
