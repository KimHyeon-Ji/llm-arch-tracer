r"""축마다 **어떻게 그 이름이 됐는가**. 값 충돌이 조용히 해결된 자리를 드러낸다.

왜 필요한가
-----------
`ambiguous.json` 은 같은 `_pick()` tier 안에 후보가 둘 이상 남았을 때만 기록된다. 그런데
심볼은 `rules/symbols.yaml` 에서 **scope 정규식**으로 미리 갈린다 -- `d_head` 는 attention,
`E` 는 expert/moe. 그래서 값이 똑같이 128 이어도 두 후보가 경쟁조차 안 하고, 충돌은 0 으로
보고된다.

Llama-4-Maverick 이 그 경우다: `d_head = E = 128`, `d_moe = w_local = 8192` 인데 충돌 0 이다.
**이게 "전부 검사해 해결됨" 인지 "검사 자체가 안 됨" 인지 구분이 안 된다**(외부 검토
2026-09-10). scope 배정이 틀리면 아무 경고 없이 틀린 이름이 나온다 -- 이 저장소에서 가장
자주 난 라벨 버그가 정확히 그것이다(`scope-overmatch-label-bug-class`).

그래서 축을 전수로 훑어 등식을 세운다:

    값이 겹치는 축 = scope 로 갈림 + 앵커/식으로 갈림 + 열린 질문 + 남은 것

`scope 만이 가른다` 가 크면 그만큼이 **아무도 검증하지 않은 판정**이다.

**이 도구의 한계.** 아직 앵커/후처리 분기를 보지 않는다. 따라서 "scope 만이 가른다" 는
"scope 가 결정했다" 가 아니라 **"후보가 여럿인데 규칙에서 그 둘을 가르는 것이 scope 뿐이고,
질문으로 기록되지도 않았다"** 는 뜻이다 -- 상한이다. 축이 앵커로 정해졌을 수도 있다.
외부 검토가 요구한 다음 단계는 여기에 `resolved_by_anchor` 를 넣는 것이다.

세는 단위는 **occurrence** 다(prefill+decode, input/output/weight 전부). `ambiguous.json` 의
`axes` 와 기준이 달라 숫자가 크다.

실행:
    .venv\Scripts\python.exe develop\axis_decision_trace.py meta-llama__Llama-4-Maverick-17B-128E
    .venv\Scripts\python.exe develop\axis_decision_trace.py --all
"""
import argparse
import collections
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJ, "src"))
if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import yaml                                     # noqa: E402
import build_table as BT                        # noqa: E402

MODELS = os.path.join(PROJ, "models")
SYMBOLS = os.path.join(PROJ, "rules", "symbols.yaml")


def scopes():
    """{심볼: 컴파일된 scope 정규식 또는 None}. scope 가 없으면 어디서나 후보다."""
    y = yaml.safe_load(open(SYMBOLS, encoding="utf-8")) or {}
    out = {}
    for name, spec in y.items():
        if not isinstance(spec, dict):
            continue
        sc = spec.get("scope")
        out[name] = re.compile(sc) if isinstance(sc, str) else None
    return out


def analyse(model):
    d = os.path.join(MODELS, model)
    st = yaml.safe_load(open(os.path.join(d, "structure.yaml"), encoding="utf-8")) or {}
    sym = {k: v for k, v in (st.get("symbols") or {}).items() if isinstance(v, int)}
    by_value = collections.defaultdict(list)
    for k, v in sym.items():
        by_value[v].append(k)
    sc = scopes()

    counts = collections.Counter()
    detail = collections.Counter()
    for ph in ("prefill", "decode"):
        raw = os.path.join(d, "full", f"{ph}.trace.raw.jsonl")
        if not os.path.exists(raw):
            continue
        conc = BT.load_concrete(d, ph) or {}
        for line in open(raw, encoding="utf-8"):
            r = json.loads(line)
            mp = r.get("module_path") or ""
            c = conc.get(r.get("op_id")) or {}
            for fld in ("input_shape", "output_shape", "weight_shape"):
                symbolic = r.get(fld) or []
                concrete = c.get(fld) or []
                if fld == "weight_shape":
                    symbolic = [symbolic] if symbolic else []
                    concrete = [concrete] if concrete else []
                for si, ssh in enumerate(symbolic):
                    if not isinstance(ssh, list) or si >= len(concrete):
                        continue
                    csh = concrete[si]
                    if not isinstance(csh, list) or len(csh) != len(ssh):
                        continue
                    for lab, val in zip(ssh, csh):
                        cands = by_value.get(val) or []
                        if len(cands) < 2:
                            counts["충돌 아님"] += 1
                            continue
                        counts["값이 겹치는 축"] += 1
                        lab = str(lab)
                        if lab not in cands:
                            # 식이거나(`n_h*d_head`) 후보 밖의 이름. 값 충돌이 아니다.
                            counts["  식/후보 밖"] += 1
                            continue
                        fits = [x for x in cands
                                if sc.get(x) is None or sc[x].search(mp)]
                        if len(fits) == 1:
                            counts["  후보 여럿 · scope 만이 가른다"] += 1
                            detail[(val, tuple(sorted(cands)), lab, "scope")] += 1
                        elif len(fits) > 1:
                            counts["  후보 여럿 · scope 로도 안 갈림"] += 1
                            detail[(val, tuple(sorted(cands)), lab, "tie")] += 1
                        else:
                            counts["  후보 여럿 · scope 에 아무도 안 맞음"] += 1
                            detail[(val, tuple(sorted(cands)), lab, "none")] += 1
    return counts, detail


def report(model):
    counts, detail = analyse(model)
    tot = counts["값이 겹치는 축"]
    print(f"\n=== {model}")
    print(f"   값이 겹치는 축 {tot:,}  (충돌 아닌 축 {counts['충돌 아님']:,})")
    for k in ("  식/후보 밖", "  후보 여럿 · scope 만이 가른다",
              "  후보 여럿 · scope 로도 안 갈림", "  후보 여럿 · scope 에 아무도 안 맞음"):
        if counts[k]:
            print(f"   {k:34} {counts[k]:,}")
    sub = sum(counts[k] for k in ("  식/후보 밖", "  후보 여럿 · scope 만이 가른다",
                                  "  후보 여럿 · scope 로도 안 갈림",
                                  "  후보 여럿 · scope 에 아무도 안 맞음"))
    if sub != tot:
        print(f"   **등식이 안 맞는다**: {sub:,} != {tot:,}")
    top = [(n, k) for k, n in detail.items() if k[3] == "scope"]
    top.sort(reverse=True)
    for n, (val, cands, lab, _) in top[:8]:
        print(f"      {n:7,}축  값 {val}: {list(cands)} -> **{lab}**  (질문으로 기록되지 않음)")
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", nargs="?")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    names = sorted(os.listdir(MODELS)) if a.all else [a.model]
    agg = collections.Counter()
    for m in names:
        if not os.path.isdir(os.path.join(MODELS, m)):
            continue
        try:
            agg.update(report(m))
        except Exception as e:
            print(f"\n=== {m}\n   실패: {e}")
    if a.all:
        print(f"\n함대 합계: 값이 겹치는 축 {agg['값이 겹치는 축']:,} / "
              f"scope 만이 가름 {agg['  후보 여럿 · scope 만이 가른다']:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
