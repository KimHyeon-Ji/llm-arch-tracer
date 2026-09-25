r"""0-b: 질문 셀을 **semantic signature** 로 분할해 판정 단위(decision unit)를 만든다.

왜 필요한가: 원장의 질문 튜플 `(label, grade, candidates)` 는 **최종 판정 단위가 아니다.**
rev 1 에서 현재 라벨이 같은 질문 16 쌍이 한 덩어리로 뭉쳐 있었다. 구조가 다른 자리를 한
질문으로 묶으면 답 하나가 서로 다른 축에 적용된다.

순서(외부 검토 2026-09-25 수정안):

  1. 모든 raw 질문 사이트에 연결된 published cell key 를 붙인다  (crosswalk 가 이미 했다)
  2. semantic signature 로 분할하고 `decision_unit_id` 를 부여한다
  3. 각 unit 에서 **고유 published cell key** 를 집계해 `affects_published_cells` 를 낸다
     -- 같은 발행 셀로 접힌 여러 raw site 는 **한 번만** 센다
  4. `affects_published_cells > 0` 인 unit 만 질문 문서로 만든다

`question_family_id` (원장 튜플 기준, rev 1 의 79 개)와 `decision_unit_id` (signature 분할
결과)를 **각각** 부여한다.

실행:
    .venv\Scripts\python.exe develop\build_decision_units.py
"""
import collections
import gzip
import hashlib
import io
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJ, "src"))
if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

MODELS = os.path.join(PROJ, "models")
LAB = os.path.join(PROJ, "..", "llm-arch-tracer-results-labeled", "work")
CW = os.path.join(LAB, "crosswalk")
OUT = os.path.join(LAB, "units")

PUBLISHED = (
    "deepseek-ai__DeepSeek-V4-Pro",
    "meta-llama__Llama-4-Maverick-17B-128E",
    "moonshotai__Kimi-K3",
    "openai__gpt-oss-120b",
    "openai__gpt-oss-20b",
)


def _rel_module(mp):
    """층 번호와 전문가 번호를 `*` 로 묶은 상대 모듈 경로."""
    if not mp:
        return ""
    mp = re.sub(r"\.(layers|h|blocks)\.\d+", r".\1.*", mp)
    mp = re.sub(r"\.experts\.\d+", ".experts.*", mp)
    return mp


def _concrete(model, phase):
    """`full/<phase>.shapes.concrete.jsonl` -- **raw op_id** 키다."""
    p = os.path.join(MODELS, model, "full", f"{phase}.shapes.concrete.jsonl")
    out = {}
    if os.path.exists(p):
        with io.open(p, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                out[r["op_id"]] = r
    return out


def _concrete_at(conc, raw_site):
    """`(raw_op_id, field, shape_index, axis)` 의 구체 크기.

    **사이드카는 raw op_id 키다.** 발행 op_id 로 조회하면 다른 op 를 읽거나 못 찾는다 --
    crosswalk 를 만들 때 같은 실수를 했고, 여기서 또 했다(2026-09-25).
    """
    oid, fld, si, ax = raw_site
    cr = conc.get(oid)
    if not cr:
        return None
    key = {"i": "input_shape", "o": "output_shape", "w": "weight_shape"}.get(fld)
    if not key:
        return None
    v = cr.get(key)
    if not v:
        return None
    shapes = [v] if (fld == "w" and v and not isinstance(v[0], list)) else v
    try:
        return shapes[si][ax]
    except Exception:                                              # noqa: BLE001
        return None


def _load_cw(model, phase):
    p = os.path.join(CW, f"{model}.{phase}.jsonl.gz")
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _published(model, phase):
    p = os.path.join(MODELS, model, f"{phase}.jsonl")
    with io.open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


FIELD_KEY = {"i": "input_shape", "o": "output_shape", "w": "weight_shape"}


def _operand_shape(prow, field, shape_index):
    """그 셀이 속한 피연산자의 **전체 shape**. signature 의 핵심 구성요소다.

    왜 `op_occurrence` 를 쓰지 않는가: 같은 모듈 안의 반복 연산을 가르려고 서수를 썼더니
    K3 의 KDA 블록이 발행 행 906 개라 자리마다 고유 단위가 됐다(65,248 셀 -> 65,247 단위,
    2026-09-25). 축 등가류로 묶어 봐도 층마다 다른 텐서라 더 쪼개진다(91 셀 -> 218 류).

    **주변 shape 전체**가 옳은 기준이다 -- V4 의 회전/비회전 slice 는 마지막 축이
    `d_rope` / `c_I-d_rope` 로 달라 자연히 갈리고, 구조가 같은 자리는 묶인다.
    실측: K3 65,248 셀 -> 124 단위.
    """
    v = prow.get(FIELD_KEY[field])
    if not v:
        return ()
    shapes = [v] if (field == "w" and v and not isinstance(v[0], list)) else v
    try:
        return tuple(str(x) for x in shapes[shape_index])
    except Exception:                                              # noqa: BLE001
        return ()


def _unit_id(model, phase, sig):
    """signature 로부터 **안정적이고 익명인** id. 정답을 암시하지 않는다."""
    h = hashlib.sha256(json.dumps(sig, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    tag = model.split("__")[-1][:10]
    return f"{tag}-{phase[0].upper()}-{h[:8]}"


def build(model):
    res = {"model": model, "phases": {}}
    for phase in ("prefill", "decode"):
        cw = _load_cw(model, phase)
        pub = _published(model, phase)
        conc = _concrete(model, phase)
        by_pub = {r["op_id"]: r for r in pub}

        units = collections.OrderedDict()
        fam_of_unit = {}
        for c in cw:
            if not c.get("is_question_cell"):
                continue
            prow = by_pub.get(c["op_id"]) or {}
            # --- semantic signature. `old_expr` 도 포함한다 -- 현재 라벨이 다르면 다른
            #     질문이다. (문서에서는 숨기지만 묶는 기준에는 필요하다.)
            sig = {
                "phase": phase,
                "block_type": prow.get("block_type"),
                "module": _rel_module(prow.get("module_path")),
                "op_type": prow.get("op_type"),
                "field": c["field"],
                "shape_index": c["shape_index"],
                "axis": c["axis"],
                # 주변 shape 전체 -- 위 `_operand_shape` docstring 참고
                "operand_shape": list(_operand_shape(prow, c["field"], c["shape_index"])),
                # 아래 셋은 **묶는 기준**이지 문서에 싣는 것이 아니다 (1차 판정은 답을 못 본다)
                "old_expr": c["expr"],
                "candidates": c.get("candidates"),
                "grade": c.get("grade"),
            }
            uid = _unit_id(model, phase, sig)
            u = units.setdefault(uid, {
                "decision_unit_id": uid, "model": model, "signature": sig,
                "published_cells": set(), "raw_sites": set(),
                "candidates": c.get("candidates"), "grade": c.get("grade"),
                "label": c.get("label"), "reason": c.get("reason"),
                "concrete_value": None, "concrete_shapes": None,
                "example_cells": []})
            if u["concrete_value"] is None and c["raw_sites"]:
                rs0 = c["raw_sites"][0]
                u["concrete_value"] = _concrete_at(conc, tuple(rs0))
                u["concrete_shapes"] = {
                    k: (conc.get(rs0[0]) or {}).get(
                        {"i": "input_shape", "o": "output_shape",
                         "w": "weight_shape"}[k])
                    for k in ("i", "o", "w")}
            u["published_cells"].add((c["op_id"], c["field"], c["shape_index"], c["axis"]))
            for rs in c["raw_sites"]:
                u["raw_sites"].add(tuple(rs))
            if len(u["example_cells"]) < 3:
                u["example_cells"].append(c["op_id"])
            # 원장 튜플 기준 family id (rev 1 의 79 개와 대응)
            fam = hashlib.sha256(json.dumps(
                [c.get("label"), c.get("grade"), c.get("candidates")],
                ensure_ascii=False).encode()).hexdigest()[:8]
            fam_of_unit.setdefault(uid, f"{model.split('__')[-1][:10]}-fam-{fam}")

        out = []
        for uid, u in units.items():
            out.append({
                "decision_unit_id": uid,
                "question_family_id": fam_of_unit[uid],
                "model": model,
                "signature": u["signature"],
                # ★ **고유** 발행 셀 키 수. 같은 셀로 접힌 여러 raw site 는 한 번만 센다.
                "affects_published_cells": len(u["published_cells"]),
                "represents_raw_sites": len(u["raw_sites"]),
                "published_cells": sorted(u["published_cells"]),
                "candidates": u["candidates"], "grade": u["grade"],
                "current_label": u["label"], "reason": u["reason"],
                "concrete_value": u["concrete_value"],
                "concrete_shapes": u["concrete_shapes"],
            })
        out.sort(key=lambda x: -x["affects_published_cells"])

        os.makedirs(OUT, exist_ok=True)
        p = os.path.join(OUT, f"{model}.{phase}.units.jsonl")
        with io.open(p, "w", encoding="utf-8", newline="\n") as f:
            for u in out:
                f.write(json.dumps(u, ensure_ascii=False) + "\n")

        fams = {u["question_family_id"] for u in out}
        res["phases"][phase] = {
            "decision_units": len(out),
            "question_families": len(fams),
            "units_with_published_impact": sum(
                1 for u in out if u["affects_published_cells"] > 0),
            "affects_published_cells_total": sum(u["affects_published_cells"] for u in out),
            "represents_raw_sites_total": sum(u["represents_raw_sites"] for u in out),
            "question_cells_in": sum(1 for c in cw if c.get("is_question_cell")),
            "top_units": [{"id": u["decision_unit_id"],
                           "cells": u["affects_published_cells"],
                           "raw": u["represents_raw_sites"],
                           "module": u["signature"]["module"],
                           "op": u["signature"]["op_type"],
                           "field": u["signature"]["field"],
                           "axis": u["signature"]["axis"]} for u in out[:6]],
        }
    return res


def main():
    rep = {"built_from_commit": subprocess.run(
               ["git", "rev-parse", "HEAD"], cwd=PROJ,
               capture_output=True, text=True).stdout.strip(),
           "built_from_tree_clean": not subprocess.run(
               ["git", "status", "--porcelain"], cwd=PROJ,
               capture_output=True, text=True).stdout.strip(),
           "models": []}
    for m in PUBLISHED:
        print(f"=== {m}")
        r = build(m)
        rep["models"].append(r)
        for ph, x in r["phases"].items():
            print(f"   {ph}: 질문 셀 {x['question_cells_in']:,} -> 판정 단위 "
                  f"{x['decision_units']:,} (family {x['question_families']})"
                  f"  발행 영향 있는 단위 {x['units_with_published_impact']:,}"
                  f"  고유 발행 셀 합 {x['affects_published_cells_total']:,}")
    p = os.path.join(OUT, "_units_report.json")
    json.dump(rep, io.open(p, "w", encoding="utf-8", newline="\n"),
              ensure_ascii=False, indent=1)
    print(f"\n보고서 -> {os.path.relpath(p, PROJ)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
