"""새 간선을 켜면 **기존 판정이 어디까지 넓어지는지** 잰다. 아무것도 쓰지 않는다.

WHY THIS EXISTS
---------------
`spread: class` 는 판정을 그 축의 등가류 전체에 퍼뜨린다. 그래서 등가류를 넓히는 변경은
**기존 판정의 사정거리도 함께 넓힌다** -- 소스로 확인한 판정 하나가 조용히 훨씬 많은 축을
바꾸게 된다. `applied > 0` 만 보면 이걸 못 본다: 발화 수는 늘기만 하고, 늘어난 자리가
맞는 자리인지는 아무도 안 본다.

그래서 판정 ID 마다 **정확한 자리 집합** `(op_id, field, shape_index, axis)` 를 전후로
비교한다. 외부 검토가 정리한 판정 기준:

    사라진 자리         -> 중단 (판정이 닿던 곳에 못 닿게 됐다)
    다른 판정과 새로 겹침 -> 중단 (서로 다른 이름을 같은 축에 주장하게 된다)
    예상 못 한 확대      -> 검토 (넓어진 자리가 새 singleton 대응축 계보인지 봐야 한다)

그리고 **보호 경계 sentinel**: Zamba2 의 `repeat_kv` 앞뒤는 소스상 뜻이 다른데
(`n_kv` -> `n_h`) n_h == n_kv == 32 이고 `n_rep == 1` 이면 op 자체가 트레이스에 없다.
전치 간선이 바로 이 경계를 넘어 물렸으므로([[transpose-edge-rejected]]), 이 두 좌표가
**계속 다른 등가류에 있는지** 를 따로 고정해 둔다.

실행:
    .venv\\Scripts\\python.exe develop\\touched_diff.py --model Zamba2
    .venv\\Scripts\\python.exe develop\\touched_diff.py
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
MODELS = os.path.join(PROJ, "models")
sys.path.insert(0, os.path.join(PROJ, "src"))

import axis_classes as AC          # noqa: E402
import build_table as BT           # noqa: E402
import label_overrides as LO       # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

_SEL = ("axis", "field", "shape_index", "op_type", "nth")


def _anchor_slots(rows, ordered, spec, ordn):
    """이 판정의 앵커가 짚는 자리들. 렌더된 shape 로 맞춘다(적용기와 같은 방식)."""
    rx = re.compile(spec["module"])
    fld = "input_shape" if spec.get("field") == "i" else "output_shape"
    want = [str(x) for x in (spec.get("shape") or [])]
    si = spec.get("shape_index") or 0
    ax = spec.get("axis")
    out = []
    for row, o in zip(rows, ordered):
        if not rx.search(AC.module_key_of(row.get("module_path") or "") or ""):
            continue
        if spec.get("op_type") and row.get("op_type") != spec["op_type"]:
            continue
        if spec.get("nth") is not None and ordn.get(row["op_id"]) != spec["nth"]:
            continue
        shs = o.get(fld) or []
        if si >= len(shs) or not isinstance(shs[si], list):
            continue
        if want and [str(x) for x in shs[si]] != want:
            continue
        if ax is None or ax >= len(shs[si]):
            continue
        out.append((row["op_id"], spec.get("field") or "o", si, ax))
    return out


def reach(model: str, phase: str):
    """판정 ID -> 그 판정이 `spread: class` 로 닿는 자리 집합. (off, on) 두 벌."""
    d = os.path.join(MODELS, model)
    raw = os.path.join(d, "full", f"{phase}.trace.raw.jsonl")
    if not os.path.exists(raw):
        return None
    ordered = [json.loads(l) for l in io.open(raw, encoding="utf-8")]
    conc = BT.load_concrete(d, phase) or {}
    rows = [dict(conc.get(o["op_id"]) or {}, op_id=o["op_id"], op_type=o.get("op_type"),
                 module_path=o.get("module_path"), depends_on=o.get("depends_on") or [])
            for o in ordered]
    ordn = AC.op_ordinals(ordered)

    specs = [(f"OV:{i}", s) for i, s in enumerate(LO.for_model(model))]
    specs += [(f"CF:{i}", s) for i, s in enumerate(
        [c for c in LO.load_confirmed() if c.get("model") == model])]

    res = {}
    for flag in (False, True):
        uf = AC.build(rows, conc, singleton_edge=flag)
        idx = collections.defaultdict(list)
        for row in rows:
            oid = row["op_id"]
            for fld, tag in (("input_shape", "i"), ("output_shape", "o")):
                for si, sh in enumerate(row.get(fld) or []):
                    if isinstance(sh, list):
                        for ax in range(len(sh)):
                            idx[uf.find((oid, tag, si, ax))].append((oid, tag, si, ax))
        for vid, spec in specs:
            hit = set()
            for slot in _anchor_slots(ordered, ordered, spec, ordn):
                if spec.get("spread") == "class":
                    hit |= set(idx[uf.find(slot)])
                else:
                    hit.add(slot)
            res.setdefault(vid, [None, None])[int(flag)] = hit
    return res, specs


# `repeat_kv` 보호 경계. 소스상 뜻이 바뀌는 자리인데 n_rep == 1 이면 op 이 없어 트레이스로는
# 안 보인다. 두 좌표가 **계속 다른 등가류**여야 한다.
SENTINEL = {"Zyphra__Zamba2-1.2B": [
    ("self_attn", "transpose", 1, "o", 1),      # key: repeat 이전 (n_kv)
    ("self_attn", "transpose", 3, "o", 1),      # key: repeat 이후 (n_h)
]}


def sentinels(model, phase) -> tuple:
    pats = SENTINEL.get(model)
    if not pats:
        return (0, 0)
    d = os.path.join(MODELS, model)
    raw = os.path.join(d, "full", f"{phase}.trace.raw.jsonl")
    if not os.path.exists(raw):
        return (0, 0)
    ordered = [json.loads(l) for l in io.open(raw, encoding="utf-8")]
    conc = BT.load_concrete(d, phase) or {}
    rows = [dict(conc.get(o["op_id"]) or {}, op_id=o["op_id"], op_type=o.get("op_type"),
                 module_path=o.get("module_path"), depends_on=o.get("depends_on") or [])
            for o in ordered]
    ordn = AC.op_ordinals(ordered)
    picks = []
    for mod, op, nth, fld, ax in pats:
        got = [r["op_id"] for r in rows
               if mod in (r.get("module_path") or "") and r.get("op_type") == op
               and ordn.get(r["op_id"]) == nth]
        picks.append([(o, fld, 0, ax) for o in got])
    ok = bad = 0
    for flag in (False, True):
        uf = AC.build(rows, conc, singleton_edge=flag)
        for s1, s2 in zip(*picks):
            if uf.find(s1) == uf.find(s2):
                bad += 1
            else:
                ok += 1
    return (ok, bad)


def main():
    ap = argparse.ArgumentParser(description="새 간선이 기존 판정의 사정거리를 얼마나 넓히나")
    ap.add_argument("--model", default="")
    a = ap.parse_args()
    names = [m for m in sorted(os.listdir(MODELS))
             if os.path.isdir(os.path.join(MODELS, m))
             and (not a.model or a.model.lower() in m.lower())]
    if a.model and not names:
        print(f"필터 '{a.model}' 에 맞는 모델 폴더가 없다.")
        return 2

    shrink = overlap = 0
    for m in names:
        for phase in ("prefill", "decode"):
            got = reach(m, phase)
            if not got:
                continue
            res, specs = got
            byid = dict(specs)
            grew = lost = 0
            for vid, (off, on) in res.items():
                if off is None or on is None:
                    continue
                if off - on:
                    lost += 1
                    shrink += 1
                if on - off:
                    grew += 1
            # 서로 다른 이름을 주장하는 두 판정이 같은 자리에서 만나면 중단이다
            def _lab(s):
                return str(s.get("to") or s.get("label"))
            ids = [v for v in res if res[v][1]]
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    A, B = ids[i], ids[j]
                    if _lab(byid[A]) == _lab(byid[B]):
                        continue
                    if (res[A][1] & res[B][1]) - (res[A][0] & res[B][0]):
                        overlap += 1
            sok, sbad = sentinels(m, phase)
            # **앵커가 몇 건이나 실제로 짚혔는지 반드시 같이 낸다.** 판정의 앵커 `shape` 는
            # 교정 **전** 표기이고 이 도구가 읽는 산출물은 교정 **후** 다. 그래서 교정이
            # 자기 앵커의 shape 을 바꾼 판정은 여기서 안 잡힌다(Zamba2 는 181건 중 10건만
            # 짚혔다). 그걸 안 적으면 "넓어짐 0" 이 안전 신호처럼 보이는데, 실제로는
            # **검사가 눈을 감은 것**이다 -- 이 저장소가 반복해서 잡아온 실패 유형이다.
            resolved = sum(1 for v in res.values() if v[0])
            tag = ""
            if lost:
                tag += f"   ← 사정거리 축소 {lost}건 **중단**"
            if sbad:
                tag += f"   ← 보호 경계 붕괴 {sbad}건 **중단**"
            if resolved < len(res):
                tag += (f"   ← 앵커 미해결 {len(res) - resolved}건: 이 판정들에 대해서는 "
                        f"**아무 말도 하지 않는다**")
            if grew or lost or sok or resolved:
                print(f"{m.split('__')[-1][:26]:<28}{phase:<9}"
                      f"판정 {len(res)}건 중 앵커 해결 {resolved}건 · "
                      f"넓어짐 {grew} · 줄어듦 {lost}"
                      + (f" · 보호경계 {sok}쌍 유지" if sok else "") + tag)
    print("\n" + "=" * 60)
    if shrink or overlap:
        print(f"사정거리 축소 {shrink} / 다른 이름 판정끼리 새 겹침 {overlap} — 여기서 멈춘다")
        return 1
    print("축소 0 · 새 겹침 0 — 넓어진 자리만 남았다(다음: 렌더 diff 로 계보 확인)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
