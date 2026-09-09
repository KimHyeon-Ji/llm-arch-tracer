r"""provenance 작업의 **합격 기준**을 파일로 박고, 언제든 다시 잰다.

왜 필요한가
-----------
`src/axis_classes.py` 에 provenance 를 넣는 것은 47개 모델 전부에 영향이 가는 코어 변경이다.
"좋아졌나?"를 매번 눈으로 판단하면 이번 라운드처럼 결함을 다른 결함으로 바꾸고도 모른다.
그래서 **바꾸기 전에** 통과해야 할 것과 깨지면 안 되는 것을 숫자로 고정한다.

세 가지를 잰다(외부 검토 2026-09-09 의 검증 세트):

1. `positive` — **저절로 풀려야 하는 것.** `expand` 불변식 위반 314건. 방향이 소스로 확정돼
   있고(Granite 216 / Nemotron-Super 80 / Zamba2 18) 교정을 넣으면 등가류가 쪼개져서 되돌린
   자리들이다. provenance 가 제대로 들어갔다면 **교정 없이** 0 이 되어야 한다.

2. `anti_union` — **갈라져 있어야 하는 것.** 값이 겹치는 두 심볼이 한 등가류에 들어가면 안
   된다. positive 만 보면 "고쳐야 할 것이 고쳐졌다"만 검사하고 "합쳐지면 안 되는 것이
   합쳐지지 않았다"는 못 본다.

3. `fixed` — **깨지면 안 되는 것.** 외부 검토가 PASS 를 준 자리들(k_pe = d_rope 339건 등).
   지금 라벨을 그대로 기록해 두고 회귀를 잡는다.

실행:
    .venv\Scripts\python.exe develop\provenance_oracle.py --freeze   # 기준 생성
    .venv\Scripts\python.exe develop\provenance_oracle.py            # 지금 상태를 기준과 대조
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

import build_table as BT              # noqa: E402
import axis_classes as ac             # noqa: E402
from anchors import module_key        # noqa: E402

MODELS = os.path.join(PROJ, "models")
OUT = os.path.join(HERE, "verify", "provenance_oracle.json")

# 한 등가류에 함께 있으면 **틀린** 심볼 쌍. 값이 우연히 같을 뿐 서로 다른 축이다.
# 근거는 develop/HANDOFF_csvjsonl_verification.md 의 "남은 미해결 목록" 과 외부 검토 판정.
ANTI_UNION = {
    "moonshotai__Kimi-K3": [("n_h", "n_h_kda"), ("d_v", "d_head_kda")],
    "ibm-granite__granite-4.0-h-small": [("d_state", "n_h_ssm")],
    "nvidia__NVIDIA-Nemotron-3-Super-120B-A12B-BF16": [("d_state", "n_h_ssm")],
    "Zyphra__Zamba2-1.2B": [("n_h_ssm", "d_head_ssm"), ("n_h", "n_kv")],
    "zai-org__GLM-5.2": [("d_nope+d_rope", "d_v")],
    "Qwen__Qwen3-Next-80B-A3B-Instruct": [("2*n_h*d_head", "d_model")],
    "MiniMaxAI__MiniMax-M2": [("d_model", "2*d_moe")],
    "allenai__OLMoE-1B-7B-0924": [("d_model", "2*d_moe")],
    "deepseek-ai__DeepSeek-V4-Flash": [("d_model", "2*d_moe")],
    "deepseek-ai__DeepSeek-V4-Flash-0731": [("d_model", "2*d_moe")],
}

# 외부 검토가 PASS 를 준 자리. (모델, module_key 접미사, op_type, field, axis) -> 있어야 할 이름.
# 하나라도 다른 이름이 되면 회귀다.
FIXED = [
    ("moonshotai__Kimi-K2-Instruct", "self_attn", "expand", "i", 3, "d_rope"),
    ("moonshotai__Kimi-K2.6", "self_attn", "expand", "i", 3, "d_rope"),
    ("moonshotai__Kimi-K2.7-Code", "self_attn", "expand", "i", 3, "d_rope"),
    ("zai-org__GLM-5.2", "self_attn", "expand", "i", 3, "d_rope"),
    ("moonshotai__Kimi-Linear-48B-A3B-Instruct", "self_attn", "split_with_sizes", "o", 3, "d_v"),
]


def _models():
    return sorted(d for d in os.listdir(MODELS) if os.path.isdir(os.path.join(MODELS, d)))


def _expand_violations(model):
    """`expand` 가 비방송 축의 이름을 바꾼 자리. (module_key, op_type, nth, axis) 로 접는다."""
    d = os.path.join(MODELS, model)
    hits = collections.Counter()
    for ph in ("prefill", "decode"):
        raw = os.path.join(d, "full", f"{ph}.trace.raw.jsonl")
        if not os.path.exists(raw):
            continue
        rows = [json.loads(l) for l in open(raw, encoding="utf-8")]
        conc = BT.load_concrete(d, ph) or {}
        ordn = ac.op_ordinals(rows)
        for r in rows:
            if r.get("op_type") != "expand":
                continue
            c = conc.get(r.get("op_id")) or {}
            ci = (c.get("input_shape") or [None])[0]
            co = (c.get("output_shape") or [None])[0]
            li = (r.get("input_shape") or [None])[0]
            lo = (r.get("output_shape") or [None])[0]
            if not all(isinstance(x, list) for x in (ci, co, li, lo)):
                continue
            if not (len(ci) == len(co) == len(li) == len(lo)):
                continue
            for a in range(len(ci)):
                if ci[a] == co[a] and ci[a] != 1 and str(li[a]) != str(lo[a]):
                    hits[f"{module_key(r.get('module_path'))}|{ph}|{ordn[r['op_id']]}|{a}|"
                         f"{li[a]}->{lo[a]}"] += 1
    return hits


def _class_name_pairs(model):
    """이 모델의 등가류마다 그 안에 나타난 이름 집합."""
    d = os.path.join(MODELS, model)
    out = []
    for ph in ("prefill", "decode"):
        raw = os.path.join(d, "full", f"{ph}.trace.raw.jsonl")
        con = os.path.join(d, "full", f"{ph}.shapes.concrete.jsonl")
        if not (os.path.exists(raw) and os.path.exists(con)):
            continue
        rows = [json.loads(l) for l in open(raw, encoding="utf-8")]
        conc = {c["op_id"]: c for c in (json.loads(l) for l in open(con, encoding="utf-8"))}
        for v in ac.name_conflicts(rows, conc).values():
            out.append(set(map(str, v["names"])))
    return out


def _fixed_labels():
    """FIXED 목록의 자리가 지금 어떤 이름인지."""
    got = {}
    for model, suffix, op, fld, axis, _want in FIXED:
        d = os.path.join(MODELS, model)
        seen = set()
        for ph in ("prefill", "decode"):
            raw = os.path.join(d, "full", f"{ph}.trace.raw.jsonl")
            if not os.path.exists(raw):
                continue
            key = "input_shape" if fld == "i" else "output_shape"
            for r in (json.loads(l) for l in open(raw, encoding="utf-8")):
                if r.get("op_type") != op:
                    continue
                if not (module_key(r.get("module_path")) or "").endswith(suffix):
                    continue
                for sh in (r.get(key) or []):
                    if isinstance(sh, list) and len(sh) > axis:
                        seen.add(str(sh[axis]))
        got[f"{model}|{suffix}|{op}|{fld}|{axis}"] = sorted(seen)
    return got


def _component_health(model):
    """등가류의 크기 분포. **잘못 이은 것을 잡는 진짜 검사다.**

    `anti_union` 은 렌더된 이름을 보는데, 파이프라인이 등가류마다 이름을 하나로 강제하므로
    두 이름이 한 클래스에 함께 보이는 일 자체가 드물다 -- 즉 "값이 겹치는 두 축이 하나로
    묶였다"를 이름만으로는 볼 수 없다. 대신 **클래스가 얼마나 커졌는가**를 본다. 잘못 이으면
    서로 다른 축이 하나로 뭉쳐 최대 클래스가 커지고 클래스 수가 줄어든다.
    (외부 검토 2026-09-09 의 component health.)
    """
    d = os.path.join(MODELS, model)
    out = {}
    for ph in ("prefill", "decode"):
        raw = os.path.join(d, "full", f"{ph}.trace.raw.jsonl")
        con = os.path.join(d, "full", f"{ph}.shapes.concrete.jsonl")
        if not (os.path.exists(raw) and os.path.exists(con)):
            continue
        rows = [json.loads(l) for l in open(raw, encoding="utf-8")]
        conc = {c["op_id"]: c for c in (json.loads(l) for l in open(con, encoding="utf-8"))}
        sizes = sorted((len(v["sites"]) for v in ac.name_conflicts(rows, conc).values()),
                       reverse=True)
        if not sizes:
            continue
        out[ph] = {"classes": len(sizes), "max": sizes[0], "top10": sizes[:10],
                   "sites": sum(sizes)}
    return out


def snapshot():
    pos = {}
    for m in _models():
        h = _expand_violations(m)
        if h:
            pos[m] = {"total": sum(h.values()), "sites": dict(h)}
    anti = {}
    for m, pairs in ANTI_UNION.items():
        if not os.path.isdir(os.path.join(MODELS, m)):
            continue
        classes = _class_name_pairs(m)
        anti[m] = {f"{a} vs {b}": sum(1 for s in classes if a in s and b in s)
                   for a, b in pairs}
    health = {m: _component_health(m) for m in _models()}
    return {"positive_expand": pos, "anti_union": anti, "fixed": _fixed_labels(),
            "component_health": health}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--freeze", action="store_true", help="지금 상태를 기준으로 박는다")
    a = ap.parse_args()
    cur = snapshot()
    tot = sum(v["total"] for v in cur["positive_expand"].values())
    print(f"positive(expand 위반, 0 이 목표): {tot}")
    for m, v in sorted(cur["positive_expand"].items(), key=lambda kv: -kv[1]["total"]):
        print(f"    {v['total']:5}  {m}")
    bad = {f"{m} / {k}": n for m, d in cur["anti_union"].items() for k, n in d.items() if n}
    print(f"anti-union(같은 등가류에 함께 있으면 안 되는 쌍, 0 이 목표): {sum(bad.values())}")
    for k, n in sorted(bad.items(), key=lambda kv: -kv[1]):
        print(f"    {n:5}  {k}")

    if a.freeze:
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(cur, f, ensure_ascii=False, indent=1, sort_keys=True)
        print(f"\n기준을 박았다: {os.path.relpath(OUT, PROJ)}")
        return 0

    if not os.path.exists(OUT):
        print("\n기준 파일이 없다. --freeze 로 먼저 박아라.")
        return 1
    base = json.load(open(OUT, encoding="utf-8"))
    fails = []
    b_tot = sum(v["total"] for v in base["positive_expand"].values())
    if tot > b_tot:
        fails.append(f"positive 퇴행: expand 위반 {b_tot} -> {tot}")
    for m, d in cur["anti_union"].items():
        for k, n in d.items():
            if n > (base["anti_union"].get(m, {}).get(k, 0)):
                fails.append(f"anti-union 퇴행: {m} / {k} 가 한 등가류에 {n}건")
    # 클래스가 커지면 잘못 이은 것이다. 5% 여유를 둔다 -- 정당한 same 간선이 새로 생기면
    # 클래스가 조금 커지는 것은 정상이고, 그때는 위 positive 가 함께 좋아져야 한다.
    for m, phs in cur.get("component_health", {}).items():
        for ph, h in phs.items():
            b = (base.get("component_health", {}).get(m, {}) or {}).get(ph)
            if not b:
                continue
            if h["max"] > b["max"] * 1.05 + 2:
                fails.append(f"등가류 비대: {m}/{ph} 최대 클래스 {b['max']} -> {h['max']}")
    for k, want in base["fixed"].items():
        if cur["fixed"].get(k) != want:
            fails.append(f"fixed 회귀: {k} = {cur['fixed'].get(k)} (기준 {want})")
    print()
    if fails:
        for f in fails:
            print("  FAIL", f)
        return 1
    print(f"기준 대비 이상 없음 (positive {tot}/{b_tot}, fixed {len(base['fixed'])}자리)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
