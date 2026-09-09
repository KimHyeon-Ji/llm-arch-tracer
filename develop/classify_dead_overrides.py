r"""발화 0 인 교정을 **자동으로 분류한다.** 자리마다 손으로 보지 않는다.

왜 필요한가
-----------
계보를 켠 뒤 미발화 교정이 0 -> 217 이 됐다. 둘 중 하나다:

  * 규칙이 판정을 **대체**했다 -- 전에도 `split_with_sizes` 교정 39건이 그랬다. 지워도 된다.
  * 계보가 앵커 자리를 옮겨 **놓쳤다** -- 지우면 그 축이 교정 없이 나간다.

217건을 손으로 가르는 것은 작업이 아니라 사고의 원인이다. 외부 검토(2026-09-09)가 준
분류 절차를 그대로 구현한다.

**핵심: "이미 목표 문자열이다" 만으로 `superseded` 라고 하지 않는다.**
새 정책에서는 `target + known + hard proof + member coverage 동일` 이어야 한다.
지금은 상태 층(known/ambiguous/...)이 아직 없으므로 **`already_named_soft` 로 보수적으로
분류**하고, 상태 층이 들어온 뒤 다시 돌린다.

실행:
    .venv\Scripts\python.exe develop\classify_dead_overrides.py
    .venv\Scripts\python.exe develop\classify_dead_overrides.py --model X --list
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
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import axis_classes as ac              # noqa: E402
import build_table as BT               # noqa: E402
from anchors import module_key         # noqa: E402

MODELS = os.path.join(PROJ, "models")

# `label_overrides.apply` 가 보고서 id 에 넣는 필드 순서.
ID_FIELDS = ("module", "from", "to", "expect", "spread", "axis", "rank", "shape",
             "field", "shape_index", "op_type", "nth", "layer_types")

FIELD_KEY = {"i": "input_shape", "o": "output_shape", "w": "weight_shape"}


def _dead(model):
    p = os.path.join(MODELS, model, "full", "label_overrides.json")
    if not os.path.exists(p):
        return []
    out = []
    for x in json.load(open(p, encoding="utf-8")):
        if x.get("applied", 0) == 0:
            out.append(dict(zip(ID_FIELDS, json.loads(x["id"]))))
    return out


def _sites(model, phase):
    d = os.path.join(MODELS, model)
    raw = os.path.join(d, "full", f"{phase}.trace.raw.jsonl")
    if not os.path.exists(raw):
        return None
    rows = [json.loads(l) for l in open(raw, encoding="utf-8")]
    ac.attach_ports(d, phase, rows)
    conc = BT.load_concrete(d, phase) or {}
    return rows, conc, ac.op_ordinals(rows)


def _structural_hits(spec, rows, conc, ordn):
    """**구조적 앵커**로 자리를 찾는다 -- `from` 과 `shape` 조건을 뺀다.

    기존 selector 가 실패한 이유가 "이름이 이미 바뀌어서" 인지 "자리가 사라져서" 인지
    가르려면, 이름에 기대지 않는 조건만으로 먼저 찾아야 한다.
    """
    rx = re.compile(spec["module"])
    fld = FIELD_KEY.get(spec.get("field") or "o", "output_shape")
    si = spec.get("shape_index")
    ax = spec.get("axis")
    hits = []
    for r in rows:
        if spec.get("op_type") and r.get("op_type") != spec["op_type"]:
            continue
        if not rx.search(module_key(r.get("module_path")) or ""):
            continue
        if spec.get("nth") is not None and ordn.get(r.get("op_id")) != spec["nth"]:
            continue
        sv = r.get(fld) or []
        cv = (conc.get(r.get("op_id")) or {}).get(fld) or []
        for i, sh in enumerate(sv):
            if si is not None and i != si:
                continue
            if not isinstance(sh, list) or i >= len(cv) or not isinstance(cv[i], list):
                continue
            if ax is None or ax >= len(sh) or ax >= len(cv[i]):
                continue
            if spec.get("expect") is not None and cv[i][ax] != spec["expect"]:
                continue
            hits.append((r.get("op_id"), str(sh[ax])))
    return hits


def classify(model):
    dead = _dead(model)
    if not dead:
        return {}
    got = {ph: _sites(model, ph) for ph in ("prefill", "decode")}
    out = []
    for spec in dead:
        hits, names = [], collections.Counter()
        for ph, g in got.items():
            if not g:
                continue
            h = _structural_hits(spec, *g)
            hits += h
            names.update(n for _o, n in h)
        if not hits:
            verdict = "site_missing"          # 구조적 앵커 자체가 사라졌다
        elif all(n == str(spec["to"]) for n in names):
            # 전부 이미 목표 이름이다. 그래도 `superseded` 라고 단정하지 않는다 --
            # 상태 층이 없어 hard proof 유무를 모른다.
            verdict = "already_named_soft"
        elif any(n == str(spec["from"]) for n in names):
            verdict = "selector_stale"        # 구조 앵커는 잡히는데 기존 selector 만 실패
        else:
            verdict = "member_rebased"        # 자리는 있는데 이름이 제3의 것
        out.append({"verdict": verdict, "model": model,
                    "module": spec["module"], "op_type": spec.get("op_type"),
                    "from": spec["from"], "to": spec["to"],
                    "hits": len(hits), "names": dict(names.most_common(4))})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model")
    ap.add_argument("--list", action="store_true", help="항목별로 전부 출력")
    a = ap.parse_args()
    names = [a.model] if a.model else sorted(
        d for d in os.listdir(MODELS) if os.path.isdir(os.path.join(MODELS, d)))
    tot = collections.Counter()
    per_model = collections.Counter()
    rows = []
    for m in names:
        r = classify(m)
        for x in (r or []):
            tot[x["verdict"]] += 1
            per_model[m] += 1
            rows.append(x)
    print(f"미발화 교정 {sum(tot.values())}건 분류")
    for k, v in tot.most_common():
        print(f"   {v:5}  {k}")
    print()
    print("모델별:")
    for m, n in per_model.most_common(10):
        print(f"   {n:5}  {m}")
    if a.list:
        print()
        for x in rows:
            print(f"  [{x['verdict']:20}] {x['model'][:34]:36} {str(x['op_type'])[:16]:18} "
                  f"{x['from']} -> {x['to']}  hits={x['hits']} {x['names']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
