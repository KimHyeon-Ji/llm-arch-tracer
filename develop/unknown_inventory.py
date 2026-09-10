r"""**모르는 라벨 전수 목록.** 확정이 아닌 축을 질문 단위로 접어 한 표로 낸다.

`full/<phase>.axis_resolution.jsonl`(축 판정 원장)만 읽는다. 등급은 다섯이고, 확정이 아닌
넷이 이 목록의 대상이다.

    scope_inferred   scope 정규식만이 후보를 갈랐다. 근거는 있으나 아무도 검증 안 했다
    heuristic        산술로 지어낸 이름이거나 마지막 수단인 재사용
    open_tie         갈랐어야 하는데 근거가 없다
    unresolved       이름을 붙일 근거 자체가 없다

같은 `(후보, 라벨, 등급)` 은 질문 하나다 -- 수천 축이 한 답으로 채워진다. 그래서 "축이
몇 만" 이어도 물어볼 것은 열 몇 개다.

실행:
    .venv\Scripts\python.exe develop\unknown_inventory.py <산출물 루트> [모델 ...]
"""
import argparse
import collections
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJ, "src"))
if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import axis_ledger as AL                    # noqa: E402

OPEN = (AL.SCOPE_INFERRED, AL.HEURISTIC, AL.OPEN_TIE, AL.UNRESOLVED)


def load(root, model):
    out = {}
    for ph in ("prefill", "decode"):
        p = os.path.join(root, model, "full", f"{ph}.axis_resolution.jsonl")
        if not os.path.exists(p):
            continue
        summary, qs, sites = None, [], []
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if r["kind"] == "summary":
                summary = r
            elif r["kind"] == "question":
                qs.append(r)
            else:
                sites.append(r)
        out[ph] = (summary, qs, sites)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root")
    ap.add_argument("models", nargs="*")
    a = ap.parse_args()
    names = a.models or sorted(d for d in os.listdir(a.root)
                               if os.path.isdir(os.path.join(a.root, d)))
    grand = collections.Counter()
    grand_q = 0
    print("| 모델 | 자리 | 확정 | scope_inferred | heuristic | open_tie | unresolved | 질문 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    rows = []
    for m in names:
        data = load(a.root, m)
        if not data:
            continue
        occ = collections.Counter()
        sites = 0
        qkeys = set()
        for ph, (summary, qs, _s) in data.items():
            occ.update(summary.get("occurrences") or {})
            sites += summary.get("sites", 0)
            for q in qs:
                qkeys.add((q["grade"], q["candidates"], q["label"]))
        grand.update(occ)
        grand_q += len(qkeys)
        rows.append((m, data, qkeys))
        unk = sum(occ.get(g, 0) for g in OPEN)
        print(f"| {m[:34]} | {sites:,} | {occ.get(AL.CONFIRMED,0):,} "
              f"| {occ.get(AL.SCOPE_INFERRED,0):,} | {occ.get(AL.HEURISTIC,0):,} "
              f"| {occ.get(AL.OPEN_TIE,0):,} | {occ.get(AL.UNRESOLVED,0):,} | {len(qkeys)} |")
    tot = sum(grand.values())
    unk = sum(grand.get(g, 0) for g in OPEN)
    print(f"\n합계 자리 {tot:,} / 확정 {grand.get(AL.CONFIRMED,0):,} "
          f"({100.0*grand.get(AL.CONFIRMED,0)/max(tot,1):.1f}%) / "
          f"**모름 {unk:,} ({100.0*unk/max(tot,1):.1f}%)** -> 질문 {grand_q}개")

    for m, data, _qk in rows:
        print(f"\n### {m}")
        merged = collections.Counter()
        for ph, (_s, qs, _si) in data.items():
            for q in qs:
                merged[(q["grade"], q["candidates"], q["label"])] += q["occurrences"]
        for (g, cands, lab), n in merged.most_common():
            c = cands if cands else "—"
            print(f"   {n:8,}축  {g:15} {c:34} -> **{lab}**")
    return 0


if __name__ == "__main__":
    sys.exit(main())
