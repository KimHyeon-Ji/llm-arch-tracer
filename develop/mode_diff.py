r"""축 등가류를 모드별로 만들어 **충돌의 원인을 분해한다.**

왜 필요한가
-----------
계보를 켠 뒤 등가류 충돌이 0 -> 1,184 로 늘었다. 저희는 "hybrid 때문이니 모드를 분리하면
내려갈 것"으로 봤지만, 외부 검토가 **그렇게 가정하지 말라**고 했다(2026-09-09). 세 가지가
섞여 있을 수 있다:

  1. `provenance` 에서도 나는 충돌 -- 정확한 새 간선이 **기존의 잠재 라벨 불일치를 드러낸** 것
  2. `hybrid` 에서만 나는 충돌 -- legacy 로 오염된 component 와 provenance component 의 상호작용
  3. `provenance` 에서 hard anchor 까지 충돌 -- **계보 규칙이나 barrier 가 틀린** 것

1 은 고쳐야 할 진짜 결함이고, 2 는 모드 분리로 사라지며, 3 은 제 코드의 버그다. **셋을
구분하지 못하면 어느 것도 고칠 수 없다.**

그래서 같은 자료로 UF 를 네 벌 만들어 대조한다. 재트레이스가 필요 없다 -- 지금 트레이스와
`ports.jsonl`, concrete 사이드카만으로 계산된다.

실행:
    .venv\Scripts\python.exe develop\mode_diff.py                 # 전 모델 요약
    .venv\Scripts\python.exe develop\mode_diff.py --model X -v    # 한 모델, 충돌 경로까지
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
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import axis_classes as ac      # noqa: E402

MODELS = os.path.join(PROJ, "models")
MODES = ("legacy", "provenance", "migration", "hybrid")


def _load(model, phase):
    d = os.path.join(MODELS, model)
    raw = os.path.join(d, "full", f"{phase}.trace.raw.jsonl")
    con = os.path.join(d, "full", f"{phase}.shapes.concrete.jsonl")
    if not (os.path.exists(raw) and os.path.exists(con)):
        return None
    rows = [json.loads(l) for l in open(raw, encoding="utf-8")]
    ac.attach_ports(d, phase, rows)
    conc = {c["op_id"]: c for c in (json.loads(l) for l in open(con, encoding="utf-8"))}
    bars = ac.noop_barriers_of(d, phase)
    return rows, conc, bars


def _conflicts(rows, conc, bars, mode):
    """그 모드의 등가류에서 이름이 둘 이상인 클래스. (이름쌍 -> 자리 수)"""
    uf = ac.build(rows, conc, noop_barriers=bars, mode=mode)
    names = collections.defaultdict(set)
    sites = collections.defaultdict(list)
    for r in rows:
        oid = r.get("op_id")
        for fld, tag in (("input_shape", "i"), ("output_shape", "o")):
            for si, sh in enumerate(r.get(fld) or []):
                if not isinstance(sh, list):
                    continue
                for ax, v in enumerate(sh):
                    root = uf.find((oid, tag, si, ax))
                    names[root].add(str(v))
                    sites[root].append((oid, tag, si, ax, r.get("op_type")))
    out = {}
    for root, ns in names.items():
        if len(ns) > 1:
            out[root] = (tuple(sorted(ns)), sites[root])
    return out


def analyse(model, verbose=False):
    per = {m: collections.Counter() for m in MODES}
    pairs = {m: collections.Counter() for m in MODES}
    witness = []
    for phase in ("prefill", "decode"):
        got = _load(model, phase)
        if not got:
            continue
        rows, conc, bars = got
        confs = {}
        for m in MODES:
            c = _conflicts(rows, conc, bars, m)
            confs[m] = c
            per[m][phase] = len(c)
            for _root, (ns, _s) in c.items():
                pairs[m][ns] += 1
        # provenance 에서도 나는 충돌 = 진짜 고쳐야 할 것
        if verbose:
            for root, (ns, ss) in list(confs["provenance"].items())[:4]:
                witness.append({
                    "phase": phase, "names": list(ns), "n_sites": len(ss),
                    "ops": collections.Counter(x[4] for x in ss).most_common(5),
                })
    return per, pairs, witness


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    names = [a.model] if a.model else sorted(
        d for d in os.listdir(MODELS) if os.path.isdir(os.path.join(MODELS, d)))
    tot = collections.Counter()
    rows_out = []
    allpairs = {m: collections.Counter() for m in MODES}
    for m in names:
        per, pairs, wit = analyse(m, a.verbose)
        s = {k: sum(v.values()) for k, v in per.items()}
        for k in MODES:
            tot[k] += s[k]
            allpairs[k].update(pairs[k])
        if any(s.values()):
            rows_out.append((m, s))
        if a.verbose and wit:
            print(f"--- {m} provenance 충돌 표본")
            for w in wit:
                print(f"    {w['phase']:8} {w['names']} 자리 {w['n_sites']} ops {w['ops']}")
    print(f"{'모델':46} " + " ".join(f"{m:>11}" for m in MODES))
    for m, s in sorted(rows_out, key=lambda r: -r[1]["provenance"])[:14]:
        print(f"{m[:44]:46} " + " ".join(f"{s[k]:>11,}" for k in MODES))
    print(f"{'합계':46} " + " ".join(f"{tot[k]:>11,}" for k in MODES))
    print()
    print("provenance 에서도 나는 충돌의 이름쌍 상위 (= 진짜 고쳐야 할 것):")
    for k, v in allpairs["provenance"].most_common(8):
        print(f"   {v:6}  {list(k)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
