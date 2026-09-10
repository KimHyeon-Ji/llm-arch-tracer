r"""라운드 5~7 의 **source-confirmed 314** 를 구조 selector 로 못 박는다.

무엇인가
--------
라운드 5~7 에서 소스로 방향을 확정했고, 교정을 넣으면 등가류가 쪼개져 되돌렸던 자리들이다.
원본은 `develop/verify/provenance_oracle.json` 의 `positive_expand` 에 **11개 그룹 / 314
occurrence** 로 있었다. 그 판정은 *렌더된 라벨*이 다른지를 봤기 때문에, 라벨이 고쳐진 지금은
0 이 나와 **아무것도 검증하지 못한다**(vacuous).

그래서 같은 11개 그룹을 **라벨을 안 보는 구조 selector** 로 다시 적는다. 각 항목은
"이 `expand` 의 비방송 축 `a` 는 입력과 출력이 같은 등가류여야 한다" 는 주장이다.

`develop/expand_contract.py` 와 헷갈리지 마라. 그쪽은 같은 세 모델의 **모든** 비방송 expand
축을 모은 더 넓은 계약(165 자리)이고, 이 파일은 **원본 314 그 자체**다. 외부 검토(2026-09-10)가
둘을 하나로 부른 것을 짚었다.

occurrence 짝짓기
-----------------
자리 이름으로 색인해 `zip` 하지 않는다. **occurrence 하나 = expand 행 하나**이므로 입력 축과
출력 축이 처음부터 같은 행에서 나온다. 한쪽만 개수가 달라지는 일이 구조적으로 불가능하다.
개수는 `expected` 와 **정확히** 같아야 한다 -- 반복 레이어가 하나 사라지거나 늘면 FAIL 이다.

`layer_sig` 는 그 레이어의 `(module_key, op_type)` 집합 해시다. 하이브리드 스택을 config 없이
가른다. Nemotron 의 `mixer|prefill|nth=3|axis=1` 은 지문 없이는 48 occurrence 가 잡히는데,
그중 40 이 mamba 층이고 8 이 attention 층이다. 원본 판정은 40 이었다.

실행:
    .venv\Scripts\python.exe develop\oracle_314.py --build     # 한 번만. 그 뒤 고정.
    .venv\Scripts\python.exe develop\oracle_314.py --check --mode provenance
    .venv\Scripts\python.exe develop\oracle_314.py --selftest
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

import axis_classes as ac                       # noqa: E402
import build_table as BT                        # noqa: E402
from expand_contract import layer_signatures, _LAYER_IDX   # noqa: E402
from anchors import module_key                  # noqa: E402

MODELS = os.path.join(PROJ, "models")
OUT = os.path.join(HERE, "verify", "oracle_314.json")
SRC = os.path.join(HERE, "verify", "provenance_oracle.json")
TOTAL = 314


def _rows(model, phase):
    d = os.path.join(MODELS, model)
    raw = os.path.join(d, "full", f"{phase}.trace.raw.jsonl")
    if not os.path.exists(raw):
        return None
    rows = [json.loads(l) for l in open(raw, encoding="utf-8")]
    ac.attach_ports(d, phase, rows)
    return rows, (BT.load_concrete(d, phase) or {}), ac.noop_barriers_of(d, phase)


def occurrences(sel, rows, conc, ordn, sig):
    """이 selector 가 잡는 (입력자리, 출력자리) 쌍들. 행 하나가 occurrence 하나다."""
    out = []
    ax = sel["axis"]
    for r in rows:
        if r.get("op_type") != "expand" or ordn.get(r.get("op_id")) != sel["nth"]:
            continue
        mp = r.get("module_path") or ""
        if (module_key(mp) or "") != sel["module_key"]:
            continue
        if sel.get("layer_sig"):
            m = _LAYER_IDX.search(mp)
            if not m or sig.get(int(m.group(1)), "") != sel["layer_sig"]:
                continue
        c = conc.get(r["op_id"]) or {}
        ci = (c.get("input_shape") or [None])[0]
        co = (c.get("output_shape") or [None])[0]
        if not (isinstance(ci, list) and isinstance(co, list) and len(ci) == len(co)):
            continue
        if ax >= len(ci) or ci[ax] != co[ax] or ci[ax] == 1:
            continue
        out.append(((r["op_id"], "i", 0, ax), (r["op_id"], "o", 0, ax)))
    return out


def build():
    src = json.load(open(SRC, encoding="utf-8"))["positive_expand"]
    sels = []
    for model, v in src.items():
        for key, n in v["sites"].items():
            mk, ph, nth, ax, arrow = key.split("|")
            sel = {"model": model, "phase": ph, "module_key": mk,
                   "nth": int(nth), "axis": int(ax), "layer_sig": None,
                   "round57_label_hits": n, "round57_arrow": arrow}
            got = _rows(model, ph)
            rows, conc, _ = got
            ordn, sig = ac.op_ordinals(rows), layer_signatures(rows)
            occ = occurrences(sel, rows, conc, ordn, sig)
            if len(occ) != n:
                # 지문으로 갈라 원본 개수와 맞는 층을 고른다. 맞는 것이 없으면 그대로 둔다
                # (그때는 expected 를 실측값으로 적고 차이를 기록한다).
                by = collections.Counter()
                for a, _b in occ:
                    mp = next(r for r in rows if r["op_id"] == a[0]).get("module_path") or ""
                    m = _LAYER_IDX.search(mp)
                    by[sig.get(int(m.group(1)), "") if m else ""] += 1
                pick = [s for s, c in by.items() if c == n]
                if len(pick) == 1:
                    sel["layer_sig"] = pick[0]
                    occ = occurrences(sel, rows, conc, ordn, sig)
            sel["expected"] = len(occ)
            sels.append(sel)
    man = {"note": "rounds 5-7 source-confirmed oracle, re-stated structurally. "
                   "One occurrence = one expand row; input axis and output axis "
                   "must share an equivalence class.",
           "selectors": sels}
    tot = sum(s["expected"] for s in sels)
    man["total_expected"] = tot
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=1)
    print(f"selector {len(sels)}개 / occurrence 합계 {tot} -> {os.path.relpath(OUT, PROJ)}")
    for s in sels:
        flag = "" if s["expected"] == s["round57_label_hits"] else \
            f"   <-- 원본 {s['round57_label_hits']}"
        print(f"   {s['expected']:4}  {s['model'][:26]:28} "
              f"{s['module_key'].split('.*.')[-1][:16]}|{s['phase']}|{s['nth']}|{s['axis']}"
              f"{' sig=' + s['layer_sig'] if s['layer_sig'] else ''}{flag}")
    if tot != TOTAL:
        print(f"\n**경고**: 합계가 {TOTAL} 이 아니라 {tot} 이다. 원본과 어긋난 selector 를 봐라.")
        return 1
    return 0


def evaluate(mode, mutate=None):
    """{model: {ok, broken, count_mismatch}}. `mutate(rows)` 는 자기검사용 결함 주입."""
    if not os.path.exists(OUT):
        return None
    man = json.load(open(OUT, encoding="utf-8"))
    per = collections.defaultdict(lambda: {"ok": 0, "broken": 0, "count_mismatch": 0,
                                           "occurrences": 0})
    cache = {}
    for sel in man["selectors"]:
        k = (sel["model"], sel["phase"])
        if k not in cache:
            got = _rows(*k)
            if not got:
                cache[k] = None
            else:
                rows, conc, bars = got
                if mutate:
                    rows = mutate(rows, sel)
                cache[k] = (rows, conc, ac.op_ordinals(rows), layer_signatures(rows),
                            ac.build(rows, conc, noop_barriers=bars, mode=mode))
        if cache[k] is None:
            per[sel["model"]]["count_mismatch"] += 1
            continue
        rows, conc, ordn, sig, uf = cache[k]
        occ = occurrences(sel, rows, conc, ordn, sig)
        p = per[sel["model"]]
        p["occurrences"] += len(occ)
        if len(occ) != sel["expected"]:
            p["count_mismatch"] += 1
            continue
        for a, b in occ:
            if uf.find(a) == uf.find(b):
                p["ok"] += 1
            else:
                p["broken"] += 1
    return {"selectors": len(man["selectors"]), "total_expected": man["total_expected"],
            "per_model": dict(per)}


def check(mode, ev=None, quiet=False):
    ev = ev or evaluate(mode)
    if ev is None:
        print("oracle 이 없다. --build 먼저.")
        return 1

    def A(k):
        return sum(v[k] for v in ev["per_model"].values())

    if not quiet:
        print(f"mode = {mode}")
        print(f"oracle 314 ({ev['selectors']} selector / {ev['total_expected']} occurrence): "
              f"이어짐 {A('ok')} / 끊김 {A('broken')} / 개수불일치 {A('count_mismatch')}")
        for m, v in sorted(ev["per_model"].items(), key=lambda kv: -kv[1]["broken"]):
            if v["broken"] or v["count_mismatch"]:
                print(f"   끊김 {v['broken']:4} / 개수불일치 {v['count_mismatch']:2}  {m[:46]}")
    return 1 if (A("broken") or A("count_mismatch")) else 0


def _drop_one_layer(rows, sel):
    """반복 레이어 하나를 통째로 지운다. occurrence 가 줄어 개수불일치가 나야 한다."""
    victim = None
    for r in rows:
        if r.get("op_type") == "expand" \
                and (module_key(r.get("module_path")) or "") == sel["module_key"]:
            m = _LAYER_IDX.search(r.get("module_path") or "")
            victim = m.group(1) if m else None
            break
    if victim is None:
        return rows
    return [r for r in rows if f".{victim}." not in (r.get("module_path") or "")]


def selftest(mode="provenance"):
    """검사가 살아있는가. 죽은 검사는 0 을 내고 통과시킨다."""
    ok = True

    def A(e, k):
        return sum(v[k] for v in e["per_model"].values())

    base = evaluate(mode)
    print(f" 1) 기준: 끊김 {A(base,'broken')} / 개수불일치 {A(base,'count_mismatch')} "
          f"/ occurrence {A(base,'occurrences')} (기대 {base['total_expected']})")
    if A(base, "broken") or A(base, "count_mismatch"):
        print("    -> FAIL: 정상 산출물에서 이미 실패한다")
        ok = False
    if A(base, "occurrences") != base["total_expected"]:
        print("    -> FAIL: occurrence 합계가 기대와 다르다")
        ok = False

    leg = evaluate("legacy")
    print(f" 2) 모드를 가르는가: legacy 끊김 {A(leg,'broken')} vs provenance {A(base,'broken')}")
    if not (A(leg, "broken") > 0 and A(base, "broken") == 0):
        print("    -> FAIL: 두 모드가 같은 값을 낸다 (vacuous)")
        ok = False

    mut = evaluate(mode, mutate=_drop_one_layer)
    print(f" 3) 반복 레이어 하나 삭제: 개수불일치 {A(mut,'count_mismatch')}")
    if A(mut, "count_mismatch") == 0:
        print("    -> FAIL: 레이어가 사라져도 통과한다")
        ok = False

    saved = open(OUT, encoding="utf-8").read()
    try:
        man = json.loads(saved)
        man["selectors"][0]["expected"] += 1
        json.dump(man, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        bad = evaluate(mode)
        print(f" 4) 기대 개수를 1 늘림: 개수불일치 {A(bad,'count_mismatch')}")
        if A(bad, "count_mismatch") == 0:
            print("    -> FAIL: 개수가 어긋나도 통과한다")
            ok = False
    finally:
        open(OUT, "w", encoding="utf-8").write(saved)

    print("\n자기검사 " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mode", default="provenance")
    a = ap.parse_args()
    if a.build:
        return build()
    if a.selftest:
        return selftest(a.mode)
    return check(a.mode)


if __name__ == "__main__":
    sys.exit(main())
