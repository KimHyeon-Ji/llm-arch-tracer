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
import hmac
import io
import json
import os
import re
import secrets
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJ, "src"))
import dim_expr                                                   # noqa: E402
import _buildguard                                               # noqa: E402

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


def _ns(model, phase):
    """발행 식을 평가할 심볼 namespace."""
    pv = os.path.join(MODELS, model, "full", "provenance.json")
    if not os.path.exists(pv):
        return None
    prov = json.load(io.open(pv, encoding="utf-8"))
    sl = prov.get("seq_len_used")
    try:
        ns = dict(dim_expr.namespace(prov, batch=int(prov.get("capture_batch") or 1),
                                     seq_len=sl))
    except Exception:                                              # noqa: BLE001
        return None
    if phase == "decode":
        ns["T"] = sl
    if ns.get("T") and ns.get("d_chunk"):
        ns.setdefault("n_chunk", ns["T"] // ns["d_chunk"])
    return ns


def _concrete_context(prow, ns):
    """**발행 행과 같은 구조**의 concrete input/weight/output shape.

    0-b 는 첫 raw site 가 속한 raw op 의 전체 shape 를 그대로 복사했다. 합성 norm 이나
    canonical weight 에서는 그 raw op 의 전체 shape 가 발행 행의 전체 shape 와 **같지
    않다** -- 실측 10 단위에서 구조가 어긋났다(외부 검토 2026-09-25).

    그래서 복사하지 않고 **발행 행의 각 식을 평가해** 같은 구조로 새로 만든다.
    평가 못 한 자리는 `None` 으로 두고 수를 센다.
    """
    out, bad = {}, 0
    for k, key in (("i", "input_shape"), ("w", "weight_shape"), ("o", "output_shape")):
        v = prow.get(key)
        if not v:
            out[k] = None
            continue
        shapes = [v] if (k == "w" and not isinstance(v[0], list)) else v
        ev = []
        for sh in shapes:
            row = []
            for e in sh:
                x = dim_expr.evaluate(str(e), ns) if ns else None
                if x is None:
                    bad += 1
                row.append(None if x is None else int(x))
            ev.append(row)
        out[k] = ev[0] if (k == "w" and not isinstance(v[0], list)) else ev
    return out, bad


def _param_role(prow):
    """parameter path 를 층·전문가 번호를 지운 **역할**로 정규화한다."""
    return sorted({_rel_module(x) for x in (prow.get("params") or ())})


def _neighbours(prow, by_pub, dependents):
    """직전·직후의 `(op_type, 상대 모듈)` 집합 -- local graph context."""
    prev = sorted({(by_pub[d].get("op_type"), _rel_module(by_pub[d].get("module_path")))
                   for d in (prow.get("depends_on") or []) if d in by_pub})
    nxt = sorted({(r.get("op_type"), _rel_module(r.get("module_path")))
                  for r in dependents.get(prow["op_id"], ())})
    return {"prev": prev, "next": nxt}


def _full_shapes(prow):
    """입력·가중치·출력 **전체** shape. signature 의 핵심."""
    return {k: prow.get(v) for k, v in
            (("i", "input_shape"), ("w", "weight_shape"), ("o", "output_shape"))}


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


def _salt():
    """review bundle 에 **넣지 않는** private salt. 없으면 만든다."""
    p = os.path.join(LAB, "_private", "unit_id_salt.txt")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    if not os.path.exists(p):
        io.open(p, "w", encoding="utf-8").write(secrets.token_hex(32))
    return io.open(p, encoding="utf-8").read().strip().encode()


def _unit_id(salt, model, phase, cells):
    """**비밀과 완전히 독립적인** 익명 id.

    0-b 는 signature 의 SHA-256 을 썼는데, 그 signature 에 `old_expr`·`candidates`·`grade`
    가 들어 있었다. 후보가 둘뿐이고 grade 도 몇 종류뿐이라 **후보를 대입해 hash 를 다시
    계산하면 현재 라벨을 찾을 수 있다**(외부 검토 2026-09-25).

    지금은 **발행 셀 키만**으로 HMAC 한다 -- 비밀이 들어가지 않으므로 역산할 것이 없다.
    salt 는 `work/_private/` 에 두고 review bundle 에 넣지 않는다. 128 비트.
    """
    msg = json.dumps(sorted(cells), sort_keys=True).encode()
    h = hmac.new(salt, msg, hashlib.sha256).hexdigest()[:32]
    return f"{model.split('__')[-1][:10]}-{phase[0].upper()}-{h}"


def build(model):
    res = {"model": model, "phases": {}}
    for phase in ("prefill", "decode"):
        cw = _load_cw(model, phase)
        pub = _published(model, phase)
        conc = _concrete(model, phase)
        ns = _ns(model, phase)
        dependents = collections.defaultdict(list)
        for r in pub:
            for d in (r.get("depends_on") or []):
                dependents[d].append(r)
        by_pub = {r["op_id"]: r for r in pub}

        units = collections.OrderedDict()
        fam_of_unit = {}
        for c in cw:
            if not c.get("is_question_cell"):
                continue
            prow = by_pub.get(c["op_id"]) or {}
            # --- semantic signature. `old_expr` 도 포함한다 -- 현재 라벨이 다르면 다른
            #     질문이다. (문서에서는 숨기지만 묶는 기준에는 필요하다.)
            # ---- signature (0-b.1). 외부 검토(2026-09-25)가 짚은 대로 보강했다.
            #
            # 0-b 는 **판정 대상이 속한 한 operand shape 만** 넣었다. 그래서 전체 I/W/O 가
            # 다른 자리가 한 단위로 합쳐졌다 -- 실측 26 단위. 예: K3 의 두 bmm
            #   [..., d_head_kda] x [..., d_head_kda, 1]           -> [..., 1]
            #   [..., d_head_kda] x [..., d_head_kda, d_head_kda]  -> [..., d_head_kda]
            #
            # `layer_cohort` 도 뺐다가 되돌린다. full layer signature 가 다르다는 것은
            # 주변 계산 구조가 다르다는 뜻이다 -- V4 의 HCA/CSA 와 K3 의 repeat-8/5/1 이
            # 합쳐져 있었다(실측 336 단위). 판정 전에 합치면 안 되고, 독립 판정 결과와
            # 근거가 같을 때 나중에 동치로 묶는다.
            sig = {
                "phase": phase,
                "block_type": prow.get("block_type"),
                "layer_cohort_id": prow.get("layers") or "(비층)",
                "module": _rel_module(prow.get("module_path")),
                "op_type": prow.get("op_type"),
                "raw_op": prow.get("raw_op"),
                "field": c["field"],
                "shape_index": c["shape_index"],
                "axis": c["axis"],
                "full_shapes": _full_shapes(prow),
                "param_role": _param_role(prow),
                "neighbours": _neighbours(prow, by_pub, dependents),
                # 아래 셋은 **묶는 기준**이지 문서에 싣는 것이 아니다 (1차 판정은 답을 못 본다)
                "old_expr": c["expr"],
                "candidates": c.get("candidates"),
                "grade": c.get("grade"),
            }
            uid = json.dumps(sig, ensure_ascii=False, sort_keys=True)   # 임시 키
            u = units.setdefault(uid, {
                "model": model, "signature": sig,
                "published_cells": set(), "raw_sites": set(),
                "candidates": c.get("candidates"), "grade": c.get("grade"),
                "label": c.get("label"), "reason": c.get("reason"),
                "concrete_value": None, "concrete_shapes": None,
                "example_cells": []})
            if u["concrete_value"] is None and c["raw_sites"]:
                u["concrete_value"] = _concrete_at(conc, tuple(c["raw_sites"][0]))
            if u["concrete_shapes"] is None:
                ctx, bad = _concrete_context(prow, ns)
                u["concrete_shapes"] = ctx
                u["concrete_context_unevaluable"] = bad
            u["published_cells"].add((c["op_id"], c["field"], c["shape_index"], c["axis"]))
            for rs in c["raw_sites"]:
                u["raw_sites"].add(tuple(rs))
            if len(u["example_cells"]) < 3:
                u["example_cells"].append(c["op_id"])
            # 원장 튜플 기준 family id (rev 1 의 79 개와 대응)
            fam = hashlib.sha256(json.dumps(
                [model, phase, c.get("label"), c.get("grade"), c.get("candidates")],
                ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:12]
            fam_of_unit.setdefault(uid, f"{model.split('__')[-1][:10]}-{phase[0].upper()}-fam-{fam}")

        salt = _salt()
        out = []
        for uid, u in units.items():
            cells = sorted(u["published_cells"])
            out.append({
                "decision_unit_id": _unit_id(salt, model, phase, cells),
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
                "concrete_context_unevaluable": u.get("concrete_context_unevaluable", 0),
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


def _families(model, phase):
    """원장의 질문 family 전부 (발행 영향 여부와 무관)."""
    p = os.path.join(MODELS, model, "full", f"{phase}.axis_resolution.jsonl")
    out = []
    if not os.path.exists(p):
        return out
    with io.open(p, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("kind") == "question":
                out.append(r)
            elif r.get("kind") == "site":
                break
    return out


def write_family_registry(units_by_key):
    """**원장 79 개 전부**를 담는 registry.

    0-b 는 crosswalk 의 질문 셀만 읽어 만들었으므로, 발행 영향이 없는 family 30 개가
    **산출물에서 아예 사라졌다** -- "빠지는 단위가 없다" 는 결론이 그래서 나왔다
    (외부 검토 2026-09-25). 79 개를 전부 적고 `status` 로 가른다.

    **이 파일은 label 을 담으므로 블라인드 bundle 에 넣지 않는다.**
    """
    rows = []
    for model in PUBLISHED:
        for phase in ("prefill", "decode"):
            for q in _families(model, phase):
                fam = hashlib.sha256(json.dumps(
                    [model, phase, str(q.get("label")), q.get("grade"),
                     q.get("candidates")], ensure_ascii=False,
                    sort_keys=True).encode()).hexdigest()[:12]
                fid = f"{model.split('__')[-1][:10]}-{phase[0].upper()}-fam-{fam}"
                us = units_by_key.get(fid, [])
                cells = sum(u["affects_published_cells"] for u in us)
                rows.append({
                    "model": model, "phase": phase, "question_family_id": fid,
                    "label": str(q.get("label")), "grade": q.get("grade"),
                    "candidates": q.get("candidates"),
                    "raw_occurrences": q.get("occurrences", 0),
                    "has_published_impact": bool(cells),
                    "affects_published_cells": cells,
                    "decision_units": len(us),
                    "status": "stage1" if cells else "raw_only_backlog"})
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "_family_registry.jsonl")
    with io.open(p, "w", encoding="utf-8", newline=chr(10)) as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + chr(10))
    return rows


def main():
    rep = _buildguard.require_clean_tree()
    _buildguard.stamp(rep, "develop/build_decision_units.py",
                      "develop/_buildguard.py")
    rep["models"] = []
    units_by_key = collections.defaultdict(list)
    for m in PUBLISHED:
        print(f"=== {m}")
        r = build(m)
        rep["models"].append(r)
        for phase in ("prefill", "decode"):
            up = os.path.join(OUT, f"{m}.{phase}.units.jsonl")
            if os.path.exists(up):
                for u in (json.loads(l) for l in io.open(up, encoding="utf-8")):
                    units_by_key[u["question_family_id"]].append(u)
        for ph, x in r["phases"].items():
            print(f"   {ph}: 질문 셀 {x['question_cells_in']:,} -> 판정 단위 "
                  f"{x['decision_units']:,} (family {x['question_families']})"
                  f"  발행 영향 있는 단위 {x['units_with_published_impact']:,}"
                  f"  고유 발행 셀 합 {x['affects_published_cells_total']:,}")
    fams = write_family_registry(units_by_key)
    rep["family_registry"] = {
        "total": len(fams),
        "with_published_impact": sum(1 for f in fams if f["has_published_impact"]),
        "raw_only_backlog": sum(1 for f in fams if not f["has_published_impact"])}
    fr = rep["family_registry"]
    print(chr(10) + f"family registry  전체 {fr['total']} / "
          f"발행 영향 {fr['with_published_impact']} / "
          f"backlog {fr['raw_only_backlog']}")
    p = os.path.join(OUT, "_units_report.json")
    json.dump(rep, io.open(p, "w", encoding="utf-8", newline="\n"),
              ensure_ascii=False, indent=1)
    print(f"\n보고서 -> {os.path.relpath(p, PROJ)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
