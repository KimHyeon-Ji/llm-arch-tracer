r"""1 단계의 **의미 판정 단위 38 개**를 고정 manifest 로 만든다 (독립 검토용).

왜 38 개인가: 1 단계 783 판정 단위의 실제 질문은 `(model, 라벨, 모듈, op_type)` 으로
묶으면 38 개뿐이고, 상위 4 개가 690 단위(88%)를 덮는다. 128 개 YAML 항목이나 783 단위가
아니라 **이 38 개를 전수 독립 검토하는 것이 싸고 명확하다**(외부 검토 2026-09-27).

manifest 에 담는 것(검토자에게 주는 것이 아니라 **조정자용 고정 기록**):

    질문 signature              model / 라벨 / 모듈 / op_type / 자리
    영향 범위                   판정 단위 수, 발행 셀 수, raw 자리 수
    현재 판정                   확정인가 교정인가, 결론 라벨
    source revision / SHA-256   무엇을 읽고 판단했는지
    selector 예상·실제 적용 수   앵커가 몇 자리를 짚어야 하고 실제로 몇 자리를 짚었나

**검토자용 패킷은 여기서 만들지 않는다.** 현재 라벨·저자 결론·설명을 숨긴 블라인드
패킷은 기존 `export_shard_packet.py` 경로를 쓴다.

실행:
    .venv\Scripts\python.exe develop\build_question_manifest.py
"""
import collections
import hashlib
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(PROJ, "src"))
if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import yaml                                                      # noqa: E402

import _buildguard                                               # noqa: E402

LAB = os.path.join(PROJ, "..", "llm-arch-tracer-results-labeled", "work")
OUT = os.path.join(LAB, "questions")

# 이번 조사에서 읽은 소스. 인용의 대상이 무엇이었는지 해시로 고정한다.
SOURCES = {
    "moonshotai__Kimi-K3": [
        ("remote", "modeling_kimi_linear.py"),
        ("venv", ".venv/Lib/site-packages/fla/ops/kda/naive.py"),
        ("repo", "src/kda_shim.py"),
    ],
    "deepseek-ai__DeepSeek-V4-Pro": [
        ("venv", ".venv/Lib/site-packages/transformers/models/deepseek_v4/"
                 "modeling_deepseek_v4.py"),
        ("venv", ".venv/Lib/site-packages/transformers/models/deepseek_v4/"
                 "configuration_deepseek_v4.py"),
    ],
    "openai__gpt-oss-120b": [
        ("venv", ".venv/Lib/site-packages/transformers/models/gpt_oss/"
                 "modeling_gpt_oss.py")],
    "openai__gpt-oss-20b": [
        ("venv", ".venv/Lib/site-packages/transformers/models/gpt_oss/"
                 "modeling_gpt_oss.py")],
    "meta-llama__Llama-4-Maverick-17B-128E": [
        ("venv", ".venv/Lib/site-packages/transformers/models/llama4/"
                 "modeling_llama4.py")],
}


def remote_dir(model):
    pv = os.path.join(PROJ, "models", model, "full", "provenance.json")
    if not os.path.exists(pv):
        return None, None
    d = json.load(io.open(pv, encoding="utf-8")) or {}
    rev = d.get("revision_resolved")
    org, name = model.split("__")
    base = os.path.expanduser(
        f"~/.cache/huggingface/hub/models--{org}--{name}/snapshots/{rev}")
    return (base if os.path.isdir(base) else None), rev


def source_meta(model):
    out, (rdir, rev) = [], remote_dir(model)
    for kind, rel in SOURCES.get(model, []):
        p = (os.path.join(rdir, rel) if kind == "remote" and rdir
             else os.path.join(PROJ, rel))
        out.append({"kind": kind, "path": rel,
                    "revision": rev if kind == "remote" else None,
                    "sha256": _buildguard.sha256_file(p),
                    "exists": os.path.exists(p)})
    return out


def load_units():
    raw = {}
    for f in sorted(os.listdir(os.path.join(LAB, "units"))):
        if not f.endswith(".units.jsonl"):
            continue
        phase = f.replace(".units.jsonl", "").rsplit(".", 1)[1]
        for u in (json.loads(l) for l in
                  io.open(os.path.join(LAB, "units", f), encoding="utf-8")):
            u["_phase"] = phase
            raw[u["decision_unit_id"]] = u
    return raw


def verdicts():
    """`(model, label)` -> 이번에 내린 판정. 확정과 교정을 한 표로 모은다."""
    out = collections.defaultdict(lambda: {"confirm": [], "rename": []})
    cf = yaml.safe_load(io.open(os.path.join(PROJ, "rules",
                                             "label_confirmed.yaml"),
                                encoding="utf-8"))["confirmed"]
    ov = yaml.safe_load(io.open(os.path.join(PROJ, "rules",
                                             "label_overrides.yaml"),
                                encoding="utf-8"))["overrides"]
    for o in cf:
        out[(o["model"], o.get("label"))]["confirm"].append(o)
    for o in ov:
        out[(o["model"], o.get("from"))]["rename"].append(o)
    return out


def applied_counts(model):
    """selector 의 **실제** 적용 수. 없으면 `None` (재트레이스 전)."""
    out = {}
    for f, key in (("label_confirmed.json", "matched"),
                   ("label_overrides.json", "applied")):
        p = os.path.join(PROJ, "models", model, "full", f)
        if not os.path.exists(p):
            continue
        for o in json.load(io.open(p, encoding="utf-8")):
            out[o["id"]] = o.get(key, 0)
    return out


def main():
    meta = _buildguard.require_clean_tree()
    _buildguard.stamp(meta, "develop/build_question_manifest.py")
    units = load_units()
    ids = [json.loads(l)["decision_unit_id"] for l in
           io.open(os.path.join(LAB, "priority", "stage1_units.jsonl"),
                   encoding="utf-8")]
    vd = verdicts()
    applied = {m: applied_counts(m) for m in SOURCES}
    smeta = {m: source_meta(m) for m in SOURCES}

    groups = collections.OrderedDict()
    for i in ids:
        u = units[i]
        s = u["signature"]
        k = (u["model"], s["old_expr"], s["module"], s["op_type"])
        g = groups.setdefault(k, {
            "question_id": None, "model": u["model"], "current_label": s["old_expr"],
            "module": s["module"], "op_type": s["op_type"],
            "raw_op": s["raw_op"], "candidates": s.get("candidates"),
            "concrete_value": u.get("concrete_value"),
            "block_types": set(), "phases": set(), "grades": collections.Counter(),
            "decision_units": 0, "published_cells": 0, "raw_sites": 0,
            "slots": set(), "shapes": set(), "families": set()})
        g["decision_units"] += 1
        g["published_cells"] += u["affects_published_cells"]
        g["raw_sites"] += u["represents_raw_sites"]
        g["block_types"].add(s["block_type"])
        g["phases"].add(u["_phase"])
        g["grades"][u["grade"]] += 1
        g["families"].add(u["question_family_id"])
        g["slots"].add(f"{s['field']}[{s['shape_index']}]ax{s['axis']}")
        sh = (s["full_shapes"] or {}).get(s["field"])
        shape = sh if s["field"] == "w" else (
            sh[s["shape_index"]] if sh else None)
        if shape:
            g["shapes"].add(json.dumps([str(x) for x in shape],
                                       ensure_ascii=False))

    rows = []
    for k, g in sorted(groups.items(), key=lambda kv: -kv[1]["decision_units"]):
        model, label = k[0], k[1]
        v = vd.get((model, label), {"confirm": [], "rename": []})
        # 이 질문에 걸린 selector 만 고른다 (모듈·op_type 이 같은 것)
        def mine(lst):
            return [o for o in lst if o.get("op_type") == g["op_type"]]
        cf, ov = mine(v["confirm"]), mine(v["rename"])
        ap = applied[model]
        import label_overrides as LO

        def rid(o, k2):
            return LO._report_id(dict(o, **{"from": o.get(k2) or o.get("from"),
                                            "to": o.get("to") or o.get(k2)}))["id"]
        cf_ap = [ap.get(rid(o, "label")) for o in cf]
        ov_ap = [ap.get(rid(o, "from")) for o in ov]
        qid = "Q-" + hashlib.sha256(
            json.dumps(k, ensure_ascii=False).encode()).hexdigest()[:10]
        rows.append({
            "question_id": qid,
            "model": model, "current_label": label,
            "module": g["module"], "op_type": g["op_type"], "raw_op": g["raw_op"],
            "slots": sorted(g["slots"]), "shapes": sorted(g["shapes"]),
            "concrete_value": g["concrete_value"],
            "candidates": g["candidates"],
            "block_types": sorted(g["block_types"]),
            "phases": sorted(g["phases"]),
            "grades": dict(g["grades"]),
            "decision_units": g["decision_units"],
            "published_cells": g["published_cells"],
            "raw_sites": g["raw_sites"],
            "question_families": sorted(g["families"]),
            "verdict": ("rename" if ov else ("confirm" if cf else "없음")),
            "resulting_label": (ov[0].get("to") if ov else
                                (cf[0].get("label") if cf else None)),
            "selectors": {"confirm": len(cf), "rename": len(ov)},
            "selector_applied": {"confirm": cf_ap, "rename": ov_ap},
            "selector_applied_zero": sum(1 for x in cf_ap + ov_ap
                                         if x == 0 or x is None),
            "source_files": smeta.get(model, []),
        })

    os.makedirs(OUT, exist_ok=True)
    with io.open(os.path.join(OUT, "questions.jsonl"), "w", encoding="utf-8",
                 newline=chr(10)) as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + chr(10))
    meta.update({
        "questions": len(rows),
        "decision_units": sum(r["decision_units"] for r in rows),
        "published_cells": sum(r["published_cells"] for r in rows),
        "raw_sites": sum(r["raw_sites"] for r in rows),
        "by_verdict": dict(collections.Counter(r["verdict"] for r in rows)),
        "selector_zero_questions": sum(1 for r in rows
                                       if r["selector_applied_zero"]),
        "note": "독립 검토 대상. 검토자용 블라인드 패킷은 export_shard_packet.py 로 만든다",
    })
    json.dump(meta, io.open(os.path.join(OUT, "_questions.json"), "w",
                            encoding="utf-8", newline=chr(10)),
              ensure_ascii=False, indent=1)
    print(f"질문 {len(rows)} 개   판정 단위 {meta['decision_units']:,}   "
          f"발행 셀 {meta['published_cells']:,}   raw 자리 {meta['raw_sites']:,}")
    print(f"  판정 {meta['by_verdict']}   selector 미발화 질문 "
          f"{meta['selector_zero_questions']}")
    print()
    print(f"{'질문':<14}{'단위':>5}{'셀':>7}  {'판정':<8}{'모델':<26}{'라벨'}")
    for r in rows:
        print(f"{r['question_id']:<14}{r['decision_units']:>5}"
              f"{r['published_cells']:>7}  {r['verdict']:<8}"
              f"{r['model'].split('__')[-1]:<26}{r['current_label']}"
              + (f" -> {r['resulting_label']}" if r["verdict"] == "rename" else ""))
    print()
    print(f"-> {os.path.relpath(OUT, PROJ)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
