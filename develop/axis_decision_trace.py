r"""축마다 **어떤 근거로 그 이름이 됐는가**. `full/<phase>.axis_resolution.jsonl` 을 읽는다.

왜 이렇게 바뀌었나
------------------
처음에는 이 도구가 `structure.yaml` 의 심볼과 `rules/symbols.yaml` 의 scope 정규식으로 후보를
**밖에서 재구성**했다. 그래서 진단용 상한밖에 못 냈다 -- 외부 검토(2026-09-11)가 여섯 가지가
빠졌다고 짚었다: 실제 `_ctx_symbols()` 후보가 아님, `_scope_path()` 정규화 없음, scope 매치
깊이/`scope_strict`/group/층 스케줄 미반영, shape 안의 `avoid`/`forbid` 와 head-count 배타
미반영, 최초 판정이 아니라 후처리가 끝난 최종 trace 를 읽음.

그 정보는 전부 리졸버 **안에** 있다. 그래서 밖에서 흉내내지 않고 `src/axis_ledger.py` 가
자리마다 남긴 것을 읽는다. 이 파일은 이제 그 원장을 사람이 읽게 정리할 뿐이다.

사이드카가 없으면 **모른다고 말한다.** 재트레이스해야 생긴다.

실행:
    .venv\Scripts\python.exe develop\axis_decision_trace.py meta-llama__Llama-4-Maverick-17B-128E
    .venv\Scripts\python.exe develop\axis_decision_trace.py --all
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

import axis_ledger as AL                        # noqa: E402

MODELS = os.path.join(PROJ, "models")
ORDER = (AL.CONFIRMED, AL.SCOPE_INFERRED, AL.HEURISTIC, AL.OPEN_TIE, AL.UNRESOLVED)


def read(model, phase):
    p = os.path.join(MODELS, model, "full", f"{phase}.axis_resolution.jsonl")
    if not os.path.exists(p):
        return None
    summary, questions, sites = None, [], 0
    with open(p, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            k = rec.get("kind")
            if k == "summary":
                summary = rec
            elif k == "question":
                questions.append(rec)
            else:
                sites += 1
    return summary, questions, sites


def report(model):
    print(f"\n=== {model}")
    agg = collections.Counter()
    any_phase = False
    for phase in ("prefill", "decode"):
        got = read(model, phase)
        if got is None:
            continue
        any_phase = True
        summary, questions, _ = got
        occ = summary.get("occurrences") or {}
        agg.update(occ)
        total = summary.get("sites", sum(occ.values()))
        flag = "" if summary.get("coverage_ok") else "   **등식 불일치**"
        print(f"   {phase}: 자리 {total:,}  질문 {summary.get('questions', 0)}개{flag}")
        for g in ORDER:
            if occ.get(g):
                pct = 100.0 * occ[g] / max(total, 1)
                print(f"      {g:16} {occ[g]:8,}  {pct:5.1f}%")
        for q in questions[:6]:
            print(f"        {q['occurrences']:7,}축  {q['grade']:14} "
                  f"{q['candidates']} -> **{q['label']}**")
    if not any_phase:
        print("   axis_resolution 사이드카가 없다 -- 재트레이스해야 생긴다.")
    return agg


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", nargs="?")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    names = sorted(os.listdir(MODELS)) if a.all else [a.model]
    agg = collections.Counter()
    have = 0
    for m in names:
        if not os.path.isdir(os.path.join(MODELS, m)):
            continue
        got = report(m)
        if got:
            have += 1
            agg.update(got)
    if a.all:
        tot = sum(agg.values())
        print(f"\n함대 합계 ({have}개 모델에 사이드카 있음): 자리 {tot:,}")
        for g in ORDER:
            if agg.get(g):
                print(f"   {g:16} {agg[g]:9,}  {100.0*agg[g]/max(tot,1):5.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
