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
import gzip
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
import dim_expr                                                  # noqa: E402

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


def _concrete(model, phase):
    """`full/<phase>.shapes.concrete.jsonl` -- raw op_id 별 **실제 구체 크기**."""
    p = os.path.join(MODELS, model, "full", f"{phase}.shapes.concrete.jsonl")
    out = {}
    if not os.path.exists(p):
        return out
    with io.open(p, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["op_id"]] = r
    return out


def _ns(model, phase):
    """발행 라벨을 평가할 심볼 namespace (provenance 의 symbol table + B, T)."""
    pv = os.path.join(MODELS, model, "full", "provenance.json")
    if not os.path.exists(pv):
        return None
    prov = json.load(io.open(pv, encoding="utf-8"))
    b = int(prov.get("capture_batch") or 1)
    sl = prov.get("seq_len_used")
    try:
        ns = dict(dim_expr.namespace(prov, batch=b, seq_len=sl))
    except Exception:                                              # noqa: BLE001
        return None
    if phase == "decode":
        ns["T"] = sl                       # decode 의 캐시 길이 = prefill 길이
    if ns.get("T") and ns.get("d_chunk"):
        ns.setdefault("n_chunk", ns["T"] // ns["d_chunk"])
    return ns


_GROUP = None


def _cells_from_csv(path):
    """**csv 를 jsonl 과 독립적으로** 파싱해 {(op_id, field, si, axis): expr} 를 만든다.
    개수 비교만으로는 부족하다는 지적(외부 검토 2026-09-25)에 따라 키와 식을 정확히 본다."""
    import re as _re
    grp = _re.compile(r"\[([^\[\]]*)\]")
    out = {}
    with io.open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            oid = int(row["op_id"])
            for fld, key in (("input_shape", "i"), ("output_shape", "o"),
                             ("weight_shape", "w")):
                cell = row.get(fld) or ""
                if not cell.strip():
                    continue
                groups = grp.findall(cell)
                # weight_shape 는 단일 shape 이므로 그룹이 하나다
                if key == "w":
                    groups = groups[:1]
                else:
                    # `[[a, b], [c]]` 의 바깥 괄호는 findall 에 안 잡힌다(중첩 제외 패턴)
                    pass
                for si, g in enumerate(groups):
                    for ax, e in enumerate([x.strip() for x in g.split(",") if x.strip()]):
                        out[(oid, key, si, ax)] = e
    return out


def _load_jsonl(path):
    with io.open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


class SchemaError(Exception):
    """shape 스키마가 예상과 다르다. **조용히 넘기지 않는다** (외부 검토 2026-09-25)."""


def _cells(row):
    """발행 행의 축 셀을 (field, shape_index, axis, expr) 로 훑는다.

    스키마가 예상과 다르면 `SchemaError` 를 던진다 -- 예전에는 `continue` 로 넘어가
    셀을 조용히 빠뜨릴 수 있었다.
    """
    for fld, key in FIELD_KEY.items():
        v = row.get(fld)
        if not v:
            continue
        if not isinstance(v, list):
            raise SchemaError(f"op {row.get('op_id')} {fld} 가 list 가 아니다: {type(v)}")
        shapes = [v] if (key == "w" and v and not isinstance(v[0], list)) else v
        for si, sh in enumerate(shapes):
            if not isinstance(sh, list):
                raise SchemaError(f"op {row.get('op_id')} {fld}[{si}] 가 list 가 아니다: {sh!r}")
            for ax, e in enumerate(sh):
                if isinstance(e, (list, dict)):
                    raise SchemaError(f"op {row.get('op_id')} {fld}[{si}][{ax}] 가 중첩됐다")
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

        # 재구성이 발행본과 같은가 -- 아니면 provenance 를 믿을 수 없다.
        # **일부 필드가 아니라 행 전체를 비교한다.** 예전에는 6 개 필드만 봤다.
        # `caveat` 를 파생 열이라고 뺐던 것도 **사실이 아니었다** -- `apply_caveats()` 가
        # raw trace 발행 전에 돌고 그 행을 `extract_major`/`collapse_repeats` 가 이어받는다
        # (실측: K3 decode raw 33,199 행 전부에 `caveat` 키가 있다). 그래서 제외를 없애고
        # **private provenance 키(`_` 접두사)만** 뺀 뒤 전부 비교한다(외부 검토 2026-09-25).
        mismatch = []
        if len(rebuilt) != len(published):
            mismatch.append(f"행 수 {len(rebuilt)} != 발행 {len(published)}")
        for a, b in zip(rebuilt, published):
            ka = {k: v for k, v in a.items() if not str(k).startswith("_")}
            kb = {k: v for k, v in b.items() if not str(k).startswith("_")}
            sa = json.dumps(ka, sort_keys=True, default=str)
            sb = json.dumps(kb, sort_keys=True, default=str)
            if sa != sb:
                diff = sorted(k for k in set(ka) | set(kb)
                              if json.dumps(ka.get(k), sort_keys=True, default=str)
                              != json.dumps(kb.get(k), sort_keys=True, default=str))
                mismatch.append(f"op {b.get('op_id')}: {diff}")
        if mismatch:
            report["phases"][phase] = {"full_row_rebuild_mismatch": len(mismatch),
                                       "examples": mismatch[:8]}
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
        concrete_bad, concrete_uneval, concrete_no_sidecar = 0, 0, 0
        concrete_bad_ex = []
        conc = _concrete(model, phase)
        ns = _ns(model, phase)
        used_raw = collections.defaultdict(list)          # 역방향 유일성용

        for prow in published:
            P = prow["op_id"]
            M = major_of_pub.get(P)
            majors = rep_group.get(M, []) if M is not None else []
            for key, si, ax, expr in _cells(prow):
                raw_sites, origins = [], set()
                for mj in majors:
                    if mj in norm_fields:
                        # `_field_origin` 은 `(raw op_id, field, shape_index)` 다 --
                        # 합성 행의 입력/출력/가중치가 서로 다른 raw op 의 서로 다른
                        # operand 에서 오므로 축을 추측하지 않는다.
                        src = norm_fields[mj].get(key)
                        origins.add("canonical_weight" if key == "w" else "synthesized_norm")
                        if src is not None:
                            r_oid, r_fld, r_si = src
                            raw_sites.append([r_oid, r_fld, r_si, ax])
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
                # **원장에 없는 자리를 버리지 않는다.** 예전에는 `have` 로 걸러 나머지만
                # 같으면 uniform 이라 했다 -- fail-closed 가 아니었다(외부 검토 2026-09-25).
                dec = [sites.get(tuple(rs)) for rs in raw_sites]
                have = [d for d in dec if d]
                n_expected, n_backed = len(raw_sites), len(have)
                n_missing = n_expected - n_backed
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
                # **둘을 가른다.** 원장이 그 축을 아예 기록하지 않은 것(런타임 축 B·T·V 등)과
                # **일부만 기록된 것**은 다르다. 후자가 fail-closed 로 막아야 하는 경우다.
                if n_missing and n_backed == 0:
                    # **원장이 의견이 없는 게 아니라, 검토 원장에서 의도적으로 생략된
                    # 자리다.** `axis_ledger.write()` 는 `grade=confirmed` 사이트를 파일에
                    # 쓰지 않는다(불변식: occurrence = confirmed + scope + open + unresolved).
                    # 그래서 대부분은 이미 확정된 자리이고, 런타임 축도 여기 섞인다
                    # (외부 검토 2026-09-25 정정).
                    decision = "omitted_from_review_ledger"
                    stat["omitted_from_review_ledger"] += 1
                elif n_missing:
                    decision = "partial_ledger"
                    stat["partial_ledger"] += 1

                rep_dec = have[0] if have else {}
                # 질문 대상인가 (원장 질문 목록의 (label, grade, candidates) 와 일치)
                # **gate 이전의 후보를 따로 둔다.** 예전에는 `is_q` 를 먼저
                # `not n_missing` 으로 만든 뒤 `is_q and n_missing` 을 셌기 때문에
                # `question_partial_ledger_coverage` 가 **구조적으로 항상 0** 이었다
                # (외부 검토 2026-09-25). 부분 누락은 적용 대상에서 빼되 조용히 사라지게
                # 하지 않고 `question_conflict` 로 명시 보고한다.
                is_q_cand = bool(have) and (rep_dec["label"], rep_dec["grade"],
                                            rep_dec["candidates"]) in q_labels
                q_partial = is_q_cand and n_missing > 0
                is_q = is_q_cand and n_missing == 0
                if q_partial:
                    stat["question_conflict"] += 1

                # --- concrete 대조: **발행 식을 심볼표로 평가해** 사이드카의 실제 크기와 본다.
                # 예전에는 양쪽이 숫자 리터럴일 때만 비교해서 `d_model`·`B*T` 같은 식을
                # 전혀 검증하지 않았다(외부 검토 2026-09-25). 평가 불가능한 셀은 성공으로
                # 넘기지 않고 `unevaluable` 로 따로 센다.
                ev = dim_expr.evaluate(expr, ns) if ns else None
                if ev is None:
                    concrete_uneval += 1
                else:
                    for rs in raw_sites:
                        cr = conc.get(rs[0])
                        fld = {"i": "input_shape", "o": "output_shape"}.get(rs[1])
                        if not cr or not fld:
                            concrete_no_sidecar += 1
                            continue
                        try:
                            v = (cr.get(fld) or [])[rs[2]][rs[3]]
                        except Exception:                          # noqa: BLE001
                            concrete_no_sidecar += 1
                            continue
                        if int(v) != int(ev):
                            concrete_bad += 1
                            if len(concrete_bad_ex) < 20:
                                concrete_bad_ex.append(
                                    {"op_id": P, "field": key, "shape_index": si,
                                     "axis": ax, "expr": expr, "evaluated": int(ev),
                                     "concrete": int(v), "raw_site": rs})

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
                    "expected_raw_sites": n_expected,
                    "ledger_backed_raw_sites": n_backed,
                    "missing_raw_sites": n_missing,
                    "decision_agreement": decision,
                    "is_question_candidate": is_q_cand,
                    "is_question_cell": is_q,
                    "question_conflict": q_partial,
                    "question_conflict_reason": ("원장이 raw 자리 일부만 기록했다 "
                                                 "(missing_raw_sites > 0)") if q_partial else None,
                    **agree})

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

        # --- **`.jsonl.gz` 로 일관되게 싣는다** (외부 검토 2026-09-25 권고).
        # 생성 산출물이라 사람이 line diff 를 볼 이점보다 저장소 부담이 크다
        # (K3 prefill 비압축 74 MB). `mtime=0` 으로 결정론적 gzip 을 쓰고,
        # **비압축 논리 SHA-256 을 따로 기록**해 압축과 무관하게 내용을 고정한다.
        os.makedirs(OUT, exist_ok=True)
        body = "".join(json.dumps(r, ensure_ascii=False) + chr(10) for r in rows_out)
        logical_sha = hashlib.sha256(body.encode("utf-8")).hexdigest()
        cw = os.path.join(OUT, f"{model}.{phase}.jsonl.gz")
        with io.open(cw, "wb") as _raw:
            with gzip.GzipFile(fileobj=_raw, mode="wb", compresslevel=6, mtime=0) as _gz:
                _gz.write(body.encode("utf-8"))
        # 질문 셀 색인은 평문으로 두되 **축약형**이다. 전체 레코드를 넣으면 K3 가 41 MB 라
        # "작은 색인" 이 아니게 된다(raw_sites 가 대부분을 차지한다). 전체는 위 .gz 에 있다.
        IDX_KEYS = ("op_id", "field", "shape_index", "axis", "expr",
                    "origin", "label", "grade", "candidates")
        idx = os.path.join(OUT, f"{model}.{phase}.questions.jsonl")
        with io.open(idx, "w", encoding="utf-8", newline=chr(10)) as f:
            for r in rows_out:
                if r["is_question_cell"]:
                    f.write(json.dumps({k: r[k] for k in IDX_KEYS},
                                       ensure_ascii=False) + chr(10))
        # --- ★ 셀 키/expr 정확 대조 (csv · 발행 jsonl · crosswalk 셋을 독립 파싱)
        cw_map = {(r["op_id"], r["field"], r["shape_index"], r["axis"]): r["expr"]
                  for r in rows_out}
        dup_keys = len(rows_out) - len(cw_map)
        js_map = {}
        for prow in published:
            for k2, si2, ax2, e2 in _cells(prow):
                js_map[(prow["op_id"], k2, si2, ax2)] = e2
        csv_map = _cells_from_csv(os.path.join(MODELS, model, f"{phase}.csv"))
        key_mismatch = sorted(set(js_map) ^ set(cw_map))
        expr_mismatch = sorted(k for k in set(js_map) & set(cw_map)
                               if js_map[k] != cw_map[k])
        csv_key_mismatch = sorted(set(csv_map) ^ set(js_map))
        csv_expr_mismatch = sorted(k for k in set(csv_map) & set(js_map)
                                   if csv_map[k] != js_map[k])

        # --- ★ 접힌 층의 op 가 대표 층의 같은 위치와 같은 서명인가
        sig_bad = 0
        by_major = {r["op_id"]: r for r in major}
        for dropped, rep in dropped_to_rep.items():
            a, b = by_major.get(dropped), by_major.get(rep)
            # `_op_sig` 는 `_rel_module` 로 층 접두사를 벗기므로 **각 op 의 실제 layer_idx**
            # 를 넘겨야 한다. None 을 넘기면 전체 경로가 남아 층 2 와 층 0 이 늘 달라진다.
            if a and b and (major_ops._op_sig(a, a.get("layer_idx"))
                            != major_ops._op_sig(b, b.get("layer_idx"))):
                sig_bad += 1

        linked = {tuple(rs) for r in rows_out for rs in r["raw_sites"]}
        pub_cells = sum(1 for prow in published for _ in _cells(prow))
        report["phases"][phase] = {
            "published_cells": pub_cells, "crosswalk_cells": len(rows_out),
            "origin": dict(stat), "reverse_uniqueness": rev,
            "concrete_mismatch": concrete_bad,
            "concrete_mismatch_examples": concrete_bad_ex,
            "concrete_unevaluable": concrete_uneval,
            "concrete_no_sidecar": concrete_no_sidecar,
            # raw site 가 없어 비교 대상 자체가 없는 셀 (V4 ambiguous 등).
            # `concrete_unevaluable = 0` 이 "모든 셀을 비교했다" 는 뜻은 아니다.
            "concrete_unmapped_cells": sum(1 for r in rows_out if not r["raw_sites"]),
            "partial_ledger_cells": stat.get("partial_ledger", 0),
            "omitted_from_review_ledger_cells": stat.get("omitted_from_review_ledger", 0),
            # gate 이전 후보 기준 -- 구조적으로 0 이 되지 않는다
            "question_partial_ledger_coverage": sum(
                1 for r in rows_out if r["is_question_candidate"] and r["missing_raw_sites"]),
            "question_conflict_cells": stat.get("question_conflict", 0),
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
            "crosswalk_gz_sha256": _sha256(cw),
            "crosswalk_logical_sha256": logical_sha,
            "crosswalk_uncompressed_bytes": len(body.encode("utf-8")),
            "duplicate_cell_keys": dup_keys,
            "exact_cell_key_or_expr_mismatch": (len(key_mismatch) + len(expr_mismatch)
                                                + len(csv_key_mismatch)
                                                + len(csv_expr_mismatch)),
            "cell_key_mismatch_examples": {
                "jsonl_vs_crosswalk_keys": key_mismatch[:6],
                "jsonl_vs_crosswalk_expr": expr_mismatch[:6],
                "csv_vs_jsonl_keys": csv_key_mismatch[:6],
                "csv_vs_jsonl_expr": csv_expr_mismatch[:6]},
            "folded_layer_op_sig_mismatch": sig_bad,
            "full_row_rebuild_mismatch": 0,
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
    # `committed_in` 은 파일이 자기 자신의 커밋을 가리킬 수 없으므로 기계 보고서에서 뺀다.
    # payload 최초 커밋은 `work/REPORT_0a.md` 에 적는다(외부 검토 2026-09-25).
    out = {"built_from_commit": tracer_commit,
           "built_from_tree_clean": not subprocess.run(
               ["git", "status", "--porcelain"], cwd=PROJ,
               capture_output=True, text=True).stdout.strip(),
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
