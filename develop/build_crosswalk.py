r"""0-a: 발행 셀 <-> 원장(raw) 자리의 crosswalk 를 **파이프라인의 실제 provenance 로** 만든다.

왜 역추정하면 안 되는가: `major_ops._collapse_norm` 은 한 norm 행의 입력을 `first` member,
출력을 `last` member, 가중치를 `*.weight` 를 든 member 에서 가져온다 -- **서로 다른 raw op**
이다. 그 위에 major-op 선별, 층 접기, `op_id` 재번호가 두 번 있다. 등장 순서로는 복원할 수
없다(외부 검토 2026-09-24). 그래서 `extract_major` / `collapse_repeats` 에 **쓰기 전용 부수
채널**을 두고, 그 둘이 이미 알고 있는 매핑을 그대로 받아 쓴다.

0-a 완료 조건(2차 검토가 추가한 네 가지 ★ 포함):

  1. 발행 셀 전수 커버
  2. `expr` 일치 (crosswalk <-> 발행 jsonl)
  3. `concrete` 일치 (crosswalk <-> raw 자리의 구체 크기)
  4. ★ 결정 일치 -- 한 발행 셀에 연결된 raw 자리들의 `(label, grade, candidates, reason)`
     **전체 튜플**이 같은가. 발행 `expr` 과 raw `label` 이 다른 경우도 `mixed`.
  5. ★ 하드 게이트 -- 비율이 아니다. **질문 대상 raw site 가 연결된 발행 셀 중
     `origin=ambiguous` 또는 `decision_agreement=mixed` 인 셀이 0**.
  6. ★ 역방향 유일성 -- `origin` 별로 다르게 적용.
  7. `origin` 분포 공개.

실행:
    .venv\Scripts\python.exe develop\build_crosswalk.py            # 5개 모델 전부
    .venv\Scripts\python.exe develop\build_crosswalk.py gpt-oss-20b
"""
import collections
import csv
import hashlib
import io
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJ, "src"))
if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import major_ops                                                 # noqa: E402

MODELS = os.path.join(PROJ, "models")
OUT = os.path.join(PROJ, "..", "llm-arch-tracer-results-labeled", "work", "crosswalk")
FIELD_KEY = {"input_shape": "i", "output_shape": "o", "weight_shape": "w"}


def _sha256(path, limit=None):
    h = hashlib.sha256()
    with io.open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _load_jsonl(path):
    with io.open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _cells(row):
    """발행 행의 축 셀을 (field, shape_index, axis, expr) 로 훑는다."""
    for fld, key in FIELD_KEY.items():
        v = row.get(fld)
        if not v:
            continue
        shapes = [v] if (key == "w" and v and not isinstance(v[0], list)) else v
        for si, sh in enumerate(shapes):
            if not isinstance(sh, list):
                continue
            for ax, e in enumerate(sh):
                yield key, si, ax, str(e)


def _ledger(model, phase):
    """원장 자리: (raw_op_id, field, si, axis) -> 판정 튜플. 질문 목록도 함께."""
    p = os.path.join(MODELS, model, "full", f"{phase}.axis_resolution.jsonl")
    sites, questions = {}, []
    if not os.path.exists(p):
        return sites, questions
    with io.open(p, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            k = r.get("kind")
            if k == "question":
                questions.append(r)
            elif k == "site":
                sites[(r["op_id"], r["field"], r.get("shape_index", 0), r["axis"])] = {
                    "label": str(r.get("label")), "grade": r.get("grade"),
                    "candidates": r.get("candidates"), "reason": r.get("reason")}
    return sites, questions


def build(model):
    report = {"model": model, "phases": {}}
    for phase in ("prefill", "decode"):
        raw_path = os.path.join(MODELS, model, "full", f"{phase}.trace.raw.jsonl")
        pub_path = os.path.join(MODELS, model, f"{phase}.jsonl")
        if not (os.path.exists(raw_path) and os.path.exists(pub_path)):
            report["phases"][phase] = {"skipped": "자료 없음"}
            continue

        print(f"   [{phase}] 적재…", flush=True)
        ordered = _load_jsonl(raw_path)
        published = _load_jsonl(pub_path)
        print(f"   [{phase}] raw {len(ordered):,} / 발행 {len(published):,}", flush=True)

        # --- 파이프라인을 그대로 다시 돌려 provenance 를 받는다 (반환값은 발행본과 같아야 한다)
        p1, p2 = {}, {}
        sigs = major_ops.full_layer_signatures(ordered)
        major = major_ops.extract_major(ordered, prov=p1)
        rebuilt = major_ops.collapse_repeats(major, layer_sigs=sigs, prov=p2)

        # 재구성이 발행본과 같은가 -- 아니면 provenance 를 믿을 수 없다
        mismatch = []
        if len(rebuilt) != len(published):
            mismatch.append(f"행 수 {len(rebuilt)} != 발행 {len(published)}")
        for a, b in zip(rebuilt, published):
            for k in ("op_id", "op_type", "module_path", "input_shape",
                      "output_shape", "weight_shape"):
                if json.dumps(a.get(k), sort_keys=True, default=str) != \
                   json.dumps(b.get(k), sort_keys=True, default=str):
                    mismatch.append(f"op {b.get('op_id')} 의 {k} 불일치")
                    break
        if mismatch:
            report["phases"][phase] = {"rebuild_mismatch": mismatch[:8],
                                       "rebuild_mismatch_total": len(mismatch)}
            continue

        # --- 매핑 뒤집기
        pub_of_major = p2["published_of_major"]
        dropped_to_rep = p2["dropped_to_rep"]
        major_of_raw = p1["major_of_raw"]
        norm_fields = p1["norm_fields"]

        # **발행 id 와 major id 는 다른 번호 공간이다.** 번호가 겹쳐 조용히 동작할 수 있으니
        # 반드시 뒤집어서 쓴다 (gpt-oss 에서 실제로 그렇게 통과했다).
        major_of_pub = {v: k for k, v in pub_of_major.items()}
        # 대표 major -> 그가 대표하는 major 전부 (자신 + 접혀 사라진 층의 같은 자리)
        rep_group = collections.defaultdict(list)
        for m in pub_of_major:
            rep_group[m].append(m)
        for dropped, rep in dropped_to_rep.items():
            if rep in pub_of_major:
                rep_group[rep].append(dropped)

        # major -> raw (norm 이 아닌 경우 1:1; norm 은 member 전체)
        raws_of_major = collections.defaultdict(list)
        for raw, mj in major_of_raw.items():
            raws_of_major[mj].append(raw)

        raw_by_id = {r["op_id"]: r for r in ordered}
        sites, questions = _ledger(model, phase)
        q_labels = {(str(q.get("label")), q.get("grade"), q.get("candidates"))
                    for q in questions}

        print(f"   [{phase}] 원장·매핑 준비 끝, 셀 순회 시작", flush=True)
        rows_out = []
        stat = collections.Counter()
        mixed_list = []
        fanout = collections.defaultdict(collections.Counter)
        concrete_bad = 0
        used_raw = collections.defaultdict(list)          # 역방향 유일성용

        for prow in published:
            P = prow["op_id"]
            M = major_of_pub.get(P)
            majors = rep_group.get(M, []) if M is not None else []
            for key, si, ax, expr in _cells(prow):
                raw_sites, origins = [], set()
                for mj in majors:
                    if mj in norm_fields:
                        src = norm_fields[mj].get(key)
                        origins.add("canonical_weight" if key == "w" else "synthesized_norm")
                        if src is not None:
                            raw_sites.append([src, key, si, ax])
                    else:
                        cands = [r for r in raws_of_major.get(mj, [])]
                        if key == "w":
                            # 원장에 `w` 자리가 없다. weight_pos 가 가리키는 입력 피연산자로
                            # 되돌린다. **축 번호를 그대로 쓰면 안 된다** -- `nn.Linear` 는
                            # 저장형 `[out, in]` 을 `aten.t` 로 전치해 `[in, out]` 으로 넣으므로
                            # 저장 축 0 이 피연산자 축 1 이다(01-main.md 의 weight_pos 규약).
                            # shape 를 맞춰 동일/전치를 판정하고, 둘 다 아니면 ambiguous.
                            wp = prow.get("weight_pos")
                            ws = prow.get("weight_shape") or []
                            ops_ = prow.get("input_shape") or []
                            mapped = None
                            if (isinstance(wp, int) and wp >= 0 and len(cands) == 1
                                    and wp < len(ops_) and isinstance(ops_[wp], list)):
                                a = [str(x) for x in ws]
                                b = [str(x) for x in ops_[wp]]
                                if a == b:
                                    mapped = ax                       # 그대로
                                elif len(a) >= 2 and a[:-2] == b[:-2] and                                         a[-2:] == b[-2:][::-1]:
                                    # 마지막 두 축만 뒤집힌 전치
                                    mapped = (len(a) - 1 if ax == len(a) - 2
                                              else (len(a) - 2 if ax == len(a) - 1 else ax))
                            if mapped is not None:
                                raw_sites.append([cands[0], "i", wp, mapped])
                                origins.add("canonical_weight")
                            else:
                                origins.add("ambiguous")
                        elif len(cands) == 1:
                            raw_sites.append([cands[0], key, si, ax])
                            origins.add("raw_slot")
                        else:
                            origins.add("ambiguous")
                origin = ("ambiguous" if ("ambiguous" in origins or not raw_sites)
                          else (list(origins)[0] if len(origins) == 1 else "ambiguous"))

                # --- ★ 결정 일치: (label, grade, candidates, reason) 전체 튜플
                dec = [sites.get(tuple(rs)) for rs in raw_sites]
                have = [d for d in dec if d]
                agree = {}
                for f in ("label", "grade", "candidates", "reason"):
                    vals = {json.dumps(d.get(f), sort_keys=True) for d in have}
                    agree[f + "_agreement"] = ("uniform" if len(vals) <= 1
                                               else "mixed") if have else "no_site"
                # 발행 expr 과 raw label 이 다르면 mixed
                expr_vs_label = "n/a"
                if have:
                    labs = {d["label"] for d in have}
                    expr_vs_label = "match" if labs == {expr} else "differs"
                    if expr_vs_label == "differs":
                        agree["label_agreement"] = "mixed"
                decision = ("no_site" if not have
                            else ("uniform" if all(v == "uniform" for k, v in agree.items())
                                  else "mixed"))

                rep_dec = have[0] if have else {}
                # 질문 대상인가 (원장 질문 목록의 (label, grade, candidates) 와 일치)
                is_q = bool(have) and (rep_dec["label"], rep_dec["grade"],
                                       rep_dec["candidates"]) in q_labels

                # concrete 대조
                cbad = False
                for rs in raw_sites:
                    rr = raw_by_id.get(rs[0])
                    if not rr:
                        continue
                    fld = {"i": "input_shape", "o": "output_shape"}.get(rs[1])
                    if not fld:
                        continue
                    try:
                        v = (rr.get(fld) or [])[rs[2]][rs[3]]
                    except Exception:                              # noqa: BLE001
                        continue
                    if isinstance(v, int) and str(v) != expr and expr.isdigit():
                        cbad = True
                if cbad:
                    concrete_bad += 1

                stat[origin] += 1
                fanout[origin][len(raw_sites)] += 1
                if origin != "ambiguous":
                    for rs in raw_sites:
                        used_raw[tuple(rs)].append((P, key, si, ax))
                if decision == "mixed":
                    stat["decision_mixed"] += 1
                    if len(mixed_list) < 40:
                        mixed_list.append({"op_id": P, "field": key, "shape_index": si,
                                           "axis": ax, "expr": expr,
                                           "labels": sorted({d["label"] for d in have}),
                                           "grades": sorted({str(d["grade"]) for d in have})})
                if is_q:
                    stat["question_cells"] += 1
                    if origin == "ambiguous":
                        stat["question_cells_ambiguous"] += 1
                    if decision == "mixed":
                        stat["question_cells_mixed"] += 1

                rows_out.append({
                    "phase": phase, "op_id": P, "field": key, "shape_index": si, "axis": ax,
                    "expr": expr, "origin": origin, "raw_sites": raw_sites,
                    "label": rep_dec.get("label"), "grade": rep_dec.get("grade"),
                    "candidates": rep_dec.get("candidates"), "reason": rep_dec.get("reason"),
                    "expr_vs_raw_label": expr_vs_label,
                    "decision_agreement": decision, "is_question_cell": is_q, **agree})

        # --- ★ 역방향 유일성 (origin 별로 다르게)
        rev = {"raw_slot_fanout": 0, "synthesized_norm_fanout": 0}
        by_origin = {(r["op_id"], r["field"], r["shape_index"], r["axis"]): r["origin"]
                     for r in rows_out}
        for rs, users in used_raw.items():
            if len(users) <= 1:
                continue
            kinds = {by_origin.get(u) for u in users}
            if kinds == {"raw_slot"}:
                rev["raw_slot_fanout"] += 1                # 설명 없는 fan-out -> 결함
            elif "synthesized_norm" in kinds:
                rev["synthesized_norm_fanout"] += 1        # first/last 관계면 정상
        # canonical_weight 는 parameter_path + storage shape + axis 로 검증
        wkeys = collections.Counter()
        for prow in published:
            ws = prow.get("weight_shape")
            if not ws:
                continue
            pp = tuple(prow.get("params") or ())
            for ax, _ in enumerate(ws if not isinstance(ws[0], list) else ws[0]):
                wkeys[(pp, tuple(str(x) for x in (ws if not isinstance(ws[0], list) else ws[0])), ax)] += 1
        rev["canonical_weight_distinct_keys"] = len(wkeys)
        rev["canonical_weight_repeated_keys"] = sum(1 for v in wkeys.values() if v > 1)

        os.makedirs(OUT, exist_ok=True)
        cw = os.path.join(OUT, f"{model}.{phase}.jsonl")
        with io.open(cw, "w", encoding="utf-8", newline="\n") as f:
            for r in rows_out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        linked = {tuple(rs) for r in rows_out for rs in r["raw_sites"]}
        pub_cells = sum(1 for prow in published for _ in _cells(prow))
        report["phases"][phase] = {
            "published_cells": pub_cells, "crosswalk_cells": len(rows_out),
            "origin": dict(stat), "reverse_uniqueness": rev,
            "concrete_mismatch": concrete_bad,
            "decision_mixed_examples": mixed_list,
            "fanout": {k: dict(v) for k, v in fanout.items()},
            "ledger_sites": len(sites), "ledger_questions": len(questions),
            "unmatched_ledger_sites": len(set(sites) - linked),
            # 질문 대상 원장 자리 중 발행 셀로 이어지지 않은 것 -- 이게 핵심 수치다.
            # 나머지 unmatched 는 major-op 선별에서 빠진 op 의 자리이므로 정상이다.
            #
            # **`linked` 는 루프 밖에서 한 번만 만든다.** 예전에는 이 집합 축약을 `sites`
            # 순회 **안에서** 다시 만들고 있었다 -- Kimi-K3 는 원장 자리 538 만 x 연결 약
            # 100 만 이라 사실상 끝나지 않았다(실제로 22 시간 멈춰 있었다, 2026-09-24).
            "unmatched_question_sites": sum(
                1 for k, v in sites.items()
                if (v["label"], v["grade"], v["candidates"]) in q_labels and k not in linked),
            "crosswalk_sha256": _sha256(cw),
        }
    return report


# **발행된 5개 모델만.** 이 작업의 대상은 `results` 브랜치에 실린 것이고, models/ 전체를
# 돌면 41 개가 잡혀 시간과 용량만 먹는다(실제로 한 번 그렇게 돌렸다).
PUBLISHED = (
    "deepseek-ai__DeepSeek-V4-Pro",
    "meta-llama__Llama-4-Maverick-17B-128E",
    "moonshotai__Kimi-K3",
    "openai__gpt-oss-120b",
    "openai__gpt-oss-20b",
)


def main():
    filt = sys.argv[1] if len(sys.argv) > 1 else ""
    names = [n for n in PUBLISHED
             if os.path.exists(os.path.join(MODELS, n, "prefill.jsonl"))
             and (not filt or filt.lower() in n.lower())]
    tracer_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJ,
                                   capture_output=True, text=True).stdout.strip()
    out = {"tracer_commit": tracer_commit,
           "major_ops_sha256": _sha256(os.path.join(PROJ, "src", "major_ops.py")),
           "build_crosswalk_sha256": _sha256(os.path.abspath(__file__)),
           "models": []}
    for n in names:
        print(f"=== {n}")
        r = build(n)
        out["models"].append(r)
        for ph, d in r["phases"].items():
            if "rebuild_mismatch" in d:
                print(f"   {ph}: **재구성 불일치** {d['rebuild_mismatch_total']}건 "
                      f"{d['rebuild_mismatch'][:3]}")
                continue
            if "skipped" in d:
                print(f"   {ph}: {d['skipped']}")
                continue
            o = d["origin"]
            print(f"   {ph}: 발행 셀 {d['published_cells']:,} / crosswalk {d['crosswalk_cells']:,}"
                  f"  ambiguous {o.get('ambiguous', 0):,}"
                  f"  질문셀 {o.get('question_cells', 0):,}"
                  f"  (amb {o.get('question_cells_ambiguous', 0)} / mixed {o.get('question_cells_mixed', 0)})")
    rp = os.path.join(OUT, "_report.json")
    json.dump(out, io.open(rp, "w", encoding="utf-8", newline="\n"),
              ensure_ascii=False, indent=1)
    print(f"\n보고서 -> {os.path.relpath(rp, PROJ)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
