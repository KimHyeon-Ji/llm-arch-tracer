r"""+@ overlay 를 적용해 `models/<m>/plus_at/` derived view 를 만든다.

성격
----
`plus_at/` 은 **비권위적 derived view** 다. 권위 있는 출처는 트레이서가 낸
`models/<m>/{prefill,decode}.{csv,jsonl}` 이고 그것은 건드리지 않는다. 이 도구는 원본 +
overlay 로 언제든 같은 결과를 다시 만든다. `plus_at/` 을 손으로 고치면 안 된다.

이 도구가 하는 일
-----------------
1. **사전 승인된** expected footprint 를 읽는다(`develop/plus_at/expected-<모델>.jsonl`).
   적용기가 그것을 다시 만들지 않는다 -- 만들면 자기 결과와 자기가 만든 기대를 비교하는
   순환이 된다(외부 검토 R3 차단 사항 1). 만드는 것은 독립 matcher 의 일이다:
       develop/plus_at_refmatch.py <모델디렉터리> <overlay> <출력>
   overlay 의 `expected_footprint.sha256` 이 그 파일을 고정한다.
2. overlay 의 `match:` 블록을 **일반 해석**해 바꿀 셀을 고른다(actual).
3. V1~V8 과 V9(develop/plus_at_v9.py)를 돈다. **하나라도 실패하면 아무 파일도 안 쓴다.**
4. 임시 디렉터리에 전부 만들고 통과했을 때만 원자적으로 교체한다.
5. MANIFEST.json 에 base/tool 커밋, 입력·overlay·도구·출력·source 해시, V9 결과,
   검토 기록, bundle 계약을 적는다.

release 조건 (하나라도 어기면 status: provisional)
  substitution 전부 status: accepted, point_verified, semantic_evidence_verified
  develop/reviews/ 에 R1·R2·R3 기록 존재
  develop/plus_at/v9_review.yaml 의 모든 게이트 review_status: accepted
  V9 에 not_run(미평가) 게이트가 없음

검증 (외부 검토 2026-09-27 승인본)
---------------------------------
  V1 격리        비대상 canonical cell 전부  original == derived   (바이트 비교가 아니다)
  V2 집합 동일성 actual_cell_set == expected_cell_set,  digest 일치
  V3 sanity      한 shape 에 같은 맨정수가 둘 이상이면 **전체 실패**
  V4 형태 보존   행 수·열 이름·op_id 순서 동일, shape 의 축 개수 동일
  V5 canonical   csv 와 jsonl 을 같은 파서로 배열화해 비교 (문자열 비교 금지)
  V6 점 대입     심볼에 값을 대입하면 원본 구체값이 복원된다
  V7 잔여 보고   바꾸지 않고 남긴 맨정수를 전수 보고
  V8 의미 맥락   바뀐 셀의 (module_path, op_type) 집합 == 선언 집합
  V9 게이트     develop/plus_at_v9.py 가 **실제로 돈다**. 게이트별 decision /
               result(pass|FAIL|not_run) / detail / review_status. deny-by-default --
               분류되지 않거나 not_run 인 게이트가 있으면 release 하지 않는다.

실행:
    .venv/Scripts/python.exe develop/plus_at_apply.py <모델명> [--expected <경로>] [--publish]
`--publish` 없으면 검증만 하고 아무것도 쓰지 않는다(dry run).
"""
import argparse
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import plus_at_canon as C                                      # noqa: E402
import plus_at_refmatch as RM                                  # noqa: E402
import yaml                                                    # noqa: E402

CRLF = chr(13) + chr(10)
NUM = re.compile(r"^\d+$")
FIELDS = ("input_shape", "weight_shape", "output_shape")
OVERLAY_DIR = os.path.join(HERE, "plus_at")

# V9 는 develop/plus_at_v9.py 가 **실제로 돈다.** 예전에는 이 파일에 이름만 적은 표가
# 있었고 출력이 "V9 통과" 라고 말했지만 구현은 일부뿐이었다 -- 외부 검토(2026-09-27, R3)가
# 짚었다. 표는 이제 검사 결과에서 나온다.
#
# release 조건: 필수 게이트 전부 review_status == accepted, 그리고 substitution 의
# point_verified / semantic_evidence_verified 가 참일 때만. 검토 기록이 없으면 provisional.
REVIEW_DIR = os.path.join(HERE, "reviews")
V9_REVIEW = os.path.join(HERE, "plus_at", "v9_review.yaml")
REQUIRED_REVIEWS = ("R1", "R2", "R3")


def sha256_file(p):
    h = hashlib.sha256()
    with io.open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def base_commit(model_dir):
    """원본 표가 나온 커밋. 그 파일을 마지막으로 바꾼 커밋이다."""
    rel = os.path.relpath(os.path.join(model_dir, "prefill.csv"), PROJ)
    try:
        out = subprocess.run(["git", "-C", PROJ, "log", "-1", "--format=%H", "--",
                              rel], capture_output=True).stdout.decode().strip()
        return out or None
    except Exception:                                            # noqa: BLE001
        return None


def git_commit():
    try:
        return subprocess.run(["git", "-C", PROJ, "rev-parse", "HEAD"],
                              capture_output=True).stdout.decode().strip() or None
    except Exception:                                            # noqa: BLE001
        return None


# ------------------------------------------------------------- overlay 해석 (적용기 쪽)
def _match_cells(spec, phase, rows, overlay=None):
    """overlay 의 `match:` 블록을 **일반 해석**해 (key, before, after) 를 낸다.

    `matcher: residual_stage` 는 일반 문법으로 표현할 수 없으므로 전용 경로로 간다.
    그 경로에서도 적용기는 두 가지를 **독립적으로** 확인한다:
      (a) 접힌 행의 모든 층에서 식이 그 값을 내는가 (산술 재확인)
      (b) shape 로 뽑은 후보 전부가 덮였는가 (완전성)
    """
    if spec.get("matcher") == "residual_stage":
        import plus_at_resid as RS
        syms = (overlay or {}).get("symbols") or {}
        R = int(syms["R_res"]["value"])
        L = int(syms["L_layers"]["value"])
        got = list(RS.cells(phase, rows, R, L))      # (a) 를 여기서 예외로 잡는다
        keys = {g[0] for g in got}
        cand = RS.candidates(phase, rows)
        missed = cand - keys
        extra = keys - cand
        if missed or extra:
            raise SystemExit(
                f"residual 후보 완전성 실패 ({phase}): 놓친 셀 {len(missed)} "
                f"{sorted(missed)[:3]}, 후보 밖 셀 {len(extra)} {sorted(extra)[:3]}")
        for key, before, token, _st, _name in got:
            yield key, before, token
        return
    m = spec["match"]
    rx = re.compile(m["module_regex"]) if m.get("module_regex") else None
    want_op = m.get("op_type")
    want_shape = [str(x) for x in m["shape"]] if m.get("shape") else None
    fields = tuple(m.get("fields") or ("input_shape", "output_shape"))
    want_si = m.get("shape_index")
    axis = m["axis"]
    first_only = bool(m.get("shape_first_axis_only"))
    operand_rule = m.get("operand_rule")
    frm = (m.get("from_by_phase") or {}).get(phase) if m.get("from_by_phase") \
        else str(spec["from"])
    if frm is None:
        return
    to_spec = str(spec["to"])

    for r in rows:
        mp = r.get("module_path") or ""
        if rx and not rx.search(mp):
            continue
        if want_op and r.get("op_type") != want_op:
            continue
        for field in fields:
            shapes = C.parse_jsonl_shape(r.get(field), field)
            n = len(shapes)
            for si, sh in enumerate(shapes):
                if want_si is not None and si != want_si:
                    continue
                if want_shape is not None and sh != want_shape:
                    continue
                if axis >= len(sh) or sh[axis] != frm:
                    continue
                if first_only and axis != 0:
                    continue
                if operand_rule == "last_operand_is_last_expert":
                    to = ("n_trace_last" if si == n - 1 else "n_trace_regular")
                elif "|" in to_spec:
                    raise SystemExit(f"{spec['sub_id']}: to 에 | 가 있으면 "
                                     f"operand_rule 이 필요하다")
                else:
                    to = to_spec
                yield (phase, int(r["op_id"]), field, si, axis), frm, to


def select(model_dir, overlay):
    """적용기 쪽 선택 결과. refmatch 와 같아야 한다."""
    subs = overlay["substitutions"]
    ext = {s["sub_id"]: s for s in subs}
    recs, seen = [], {}
    for phase in ("prefill", "decode"):
        jp = os.path.join(model_dir, f"{phase}.jsonl")
        if not os.path.exists(jp):
            continue
        rows = C.read_jsonl_rows(jp)
        by_id = {int(r["op_id"]): r for r in rows}
        for sub_id, spec in ext.items():
            if phase not in (spec.get("phases") or []):
                continue
            for key, before, after in _match_cells(spec, phase, rows, overlay):
                if key in seen:
                    raise SystemExit(f"두 규칙이 같은 셀을 노린다: {key}  "
                                     f"{seen[key]} vs {sub_id}")
                seen[key] = sub_id
                recs.append(RM.canonical_record(key, before, after,
                                                by_id[key[1]], sub_id))
    return recs


# ------------------------------------------------------------------------- 적용
def apply_to_rows(rows, changes, phase, is_json):
    """changes: {(field, si, axis) -> after} per op_id. 제자리에서 바꾼다."""
    n = 0
    by_op = {}
    for k, after in changes.items():
        by_op.setdefault(k[1], {})[(k[2], k[3], k[4])] = after
    for r in rows:
        oid = int(r["op_id"])
        ch = by_op.get(oid)
        if not ch:
            continue
        for field in FIELDS:
            hits = [(si, ax, to) for (f, si, ax), to in ch.items() if f == field]
            if not hits:
                continue
            if is_json:
                shapes = C.parse_jsonl_shape(r.get(field), field)
            else:
                shapes = C.parse_csv_shape(r.get(field), field)
            for si, ax, to in hits:
                shapes[si][ax] = to
                n += 1
            if is_json:
                r[field] = C.unparse_jsonl_shape(shapes, field, r.get(field))
            else:
                r[field] = C.render_csv_shape(shapes, field)
    return n


def write_csv(path, header, rows):
    import csv as _csv
    with io.open(path, "w", encoding="utf-8", newline="") as f:
        w = _csv.writer(f, lineterminator=CRLF)
        w.writerow(header)
        for r in rows:
            w.writerow([r.get(c, "") for c in header])


def write_jsonl(path, rows):
    with io.open(path, "w", encoding="utf-8", newline="") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + chr(10))


# ------------------------------------------------------------------------ 검증
def v6_substitute(overlay, phase, model_dir):
    """symbols 의 expr 를 대입해 {이름: 정수}. cycle 과 음수·비정수를 잡는다.

    기반 환경은 **트레이서 자신의 symbol_table** 이다(full/provenance.json). overlay 가
    새로 선언한 이름만 그 위에 얹는다 -- 아키텍처 값을 +@ 가 다시 적으면 두 출처가
    갈릴 수 있다.
    """
    prov = os.path.join(model_dir, "full", "provenance.json")
    tracer_syms = {}
    if os.path.exists(prov):
        d = json.load(io.open(prov, encoding="utf-8"))
        tracer_syms = {k: v for k, v in (d.get("symbol_table") or {}).items()
                       if isinstance(v, int)}
    sym = overlay["symbols"]
    base = dict(tracer_syms)
    for name, d in sym.items():
        if d.get("kind") == "row_field":
            continue                    # 행마다 다른 값이다. V6 는 행 단위로 따로 본다.
        v = d.get(f"value_{phase}", d.get("value"))
        if v is None:
            raise SystemExit(f"symbols.{name}: {phase} 값이 없다")
        if not isinstance(v, int):
            raise SystemExit(f"symbols.{name}: 값이 정수가 아니다 ({v!r})")
        if v < 0:
            raise SystemExit(f"symbols.{name}: 값이 음수다 ({v})")
        base[name] = v
    # expr 가 있으면 다른 심볼로 계산해 값과 맞는지 본다 (cycle 은 깊이 제한으로 잡는다)
    env = dict(base)
    env.update({"floor": lambda x: int(x), "trace": None})
    for name, d in sym.items():
        expr = d.get(f"expr_{phase}", d.get("expr"))
        if not expr or "trace." in expr or "config." in expr:
            continue
        try:
            got = eval(expr.replace("/", "//"), {"__builtins__": {}}, env)  # noqa: S307
        except Exception as e:                                   # noqa: BLE001
            raise SystemExit(f"symbols.{name}.expr 평가 실패: {expr!r} -- {e}")
        if int(got) != base[name]:
            raise SystemExit(f"symbols.{name}: expr {expr!r} -> {got} 인데 "
                             f"선언값은 {base[name]} 이다")
    return base


def check(model_dir, overlay, expected, actual, out_dir, report):
    """V1~V9. 실패 목록을 반환(빈 목록이면 통과)."""
    bad = []
    # V2 집합 동일성 + digest
    ekeys = {(r["phase"], r["op_id"], r["field"], r["shape_index"], r["axis"])
             for r in expected}
    akeys = {(r["phase"], r["op_id"], r["field"], r["shape_index"], r["axis"])
             for r in actual}
    if ekeys != akeys:
        bad.append(f"V2 집합 불일치 -- expected만 {len(ekeys - akeys)} "
                   f"actual만 {len(akeys - ekeys)}")
        for k in list(ekeys - akeys)[:3]:
            bad.append(f"      expected만: {k}")
        for k in list(akeys - ekeys)[:3]:
            bad.append(f"      actual만:   {k}")
    de = sha256_bytes(RM.canonical_bytes(expected))
    da = sha256_bytes(RM.canonical_bytes(actual))
    if de != da:
        bad.append(f"V2 digest 불일치  expected {de[:16]}  actual {da[:16]}")
    report["footprint_digest"] = da
    report["cells"] = len(actual)

    amap = {(r["phase"], r["op_id"], r["field"], r["shape_index"], r["axis"]):
            (r["before"], r["after"], r["sub_id"], r["module_path"], r["op_type"])
            for r in actual}

    for phase in ("prefill", "decode"):
        op = os.path.join(model_dir, f"{phase}.csv")
        oj = os.path.join(model_dir, f"{phase}.jsonl")
        if not os.path.exists(op):
            continue
        np_ = os.path.join(out_dir, f"{phase}.csv")
        nj = os.path.join(out_dir, f"{phase}.jsonl")

        oc, nc = C.csv_cells(op, phase), C.csv_cells(np_, phase)
        oJ, nJ = C.jsonl_cells(oj, phase), C.jsonl_cells(nj, phase)

        # V5 canonical: csv 와 jsonl 이 같은 cell 집합·값
        if nc != nJ:
            bad.append(f"V5 {phase}: 생성본 csv != jsonl "
                       f"(csv만 {len(set(nc)-set(nJ))}, jsonl만 {len(set(nJ)-set(nc))}, "
                       f"값다름 {sum(1 for k in set(nc)&set(nJ) if nc[k]!=nJ[k])})")
        # V4 형태 보존: cell 키 집합이 동일해야 한다(축 개수·행·피연산자 수 불변)
        if set(oc) != set(nc):
            bad.append(f"V4 {phase}: cell 키 집합이 바뀌었다 "
                       f"(원본만 {len(set(oc)-set(nc))}, 생성본만 {len(set(nc)-set(oc))})")
        # V1 격리: 비대상 셀은 원본과 같아야 한다
        iso = 0
        for k, v in oc.items():
            if k in amap:
                continue
            if nc.get(k) != v:
                iso += 1
                if iso <= 3:
                    bad.append(f"V1 {phase} 비대상 셀이 바뀌었다: {k} {v!r} -> {nc.get(k)!r}")
        if iso:
            bad.append(f"V1 {phase}: 비대상 셀 {iso} 개가 바뀌었다")
        # 대상 셀은 before -> after 로 바뀌어 있어야 한다
        wrong = 0
        for k, (bfr, aft, *_rest) in amap.items():
            if k[0] != phase:
                continue
            if oc.get(k) != bfr or nc.get(k) != aft:
                wrong += 1
                if wrong <= 3:
                    bad.append(f"V2 {phase} 대상 셀이 기대와 다르다: {k} "
                               f"원본 {oc.get(k)!r}(기대 {bfr!r}) "
                               f"생성 {nc.get(k)!r}(기대 {aft!r})")
        if wrong:
            bad.append(f"V2 {phase}: 대상 셀 {wrong} 개가 기대와 다르다")

        # V3 sanity: 한 shape 에 같은 맨정수가 둘 이상 -> 전체 실패
        dup = 0
        _, rows = C.read_csv_rows(op)
        for r in rows:
            for f in FIELDS:
                for sh in C.parse_csv_shape(r.get(f), f):
                    lits = [t for t in sh if NUM.match(t) and t != "1"]
                    if len(lits) != len(set(lits)) and lits:
                        dup += 1
        if dup:
            bad.append(f"V3 {phase}: 한 shape 에 같은 맨정수가 둘 이상인 자리 {dup} "
                       f"-- 전체 적용을 중단한다")

        # V6 점 대입
        env = v6_substitute(overlay, phase, model_dir)
        import plus_at_resid as _RS
        _fml = {v: k for k, v in _RS.TOKEN.items()}
        # 행 단위 식(residual)은 **사이드카로 옮겼다** -- 본표에는 안 들어간다.
        # overlay 에 residual_sidecar 가 있으면 그 값을, 없으면 식 토큰이 나올 수 없다.
        _sc = overlay.get("residual_sidecar") or {}
        _R = int(_sc.get("R_res", {}).get("value") or 0)
        _L = int(_sc.get("L_layers", {}).get("value") or 0)
        _jr = {int(r["op_id"]): r for r in C.read_jsonl_rows(
            os.path.join(model_dir, f"{phase}.jsonl"))}
        for k, (bfr, aft, sub_id, _mp, _ot) in amap.items():
            if k[0] != phase:
                continue
            if aft in _fml:
                # 행 단위 식. **접힌 행의 모든 층**에서 원본 값을 복원해야 한다.
                row = _jr.get(k[1]) or {}
                ls = C.expand_layers(row.get("layers")) or [_L]
                name = _fml[aft]
                if not all(_RS.value(name, l, _R, _L) == int(bfr) for l in ls):
                    bad.append(f"V6 {phase}: {aft} 가 층 {ls[:4]} 에서 {bfr} 를 "
                               f"복원하지 않는다 ({sub_id})")
                    break
                continue
            got = env.get(aft)
            if got is None:
                bad.append(f"V6 {phase}: 심볼 {aft!r} 이 symbols 에 없다 ({sub_id})")
                break
            if str(got) != bfr:
                bad.append(f"V6 {phase}: {aft} = {got} 인데 원본 구체값은 {bfr} "
                           f"({sub_id})")
                break

        # V7 잔여 보고
        left = {}
        for k, v in nc.items():
            if NUM.match(v) and v != "1":
                left[v] = left.get(v, 0) + 1
        report.setdefault("residual_literals", {})[phase] = left

    # V8 의미 맥락: 바뀐 셀의 (module, op_type) 이 선언과 맞는가
    for spec in overlay["substitutions"]:
        sid = spec["sub_id"]
        got = {(r["module_path"], r["op_type"]) for r in actual if r["sub_id"] == sid}
        rx = re.compile(spec["match"]["module_regex"])
        want_op = spec["match"].get("op_type")
        for mp, ot in got:
            if not rx.search(mp):
                bad.append(f"V8 {sid}: 선언 밖 모듈이 바뀌었다 {mp}")
                break
            if want_op and ot != want_op:
                bad.append(f"V8 {sid}: 선언 밖 op 가 바뀌었다 {ot}")
                break
        report.setdefault("per_sub", {})[sid] = len(
            [r for r in actual if r["sub_id"] == sid])

    return bad


def run_v9(model_dir, overlay, actual, out_dir, report):
    """실제 V9 게이트를 돈다. (rows, failed)."""
    import plus_at_v9 as V9
    ctx = {"proj": PROJ, "model_dir": model_dir, "overlay": overlay,
           "actual": actual, "orig_rows": {}, "derived_rows": {},
           "orig_cells": {}, "derived_cells": {}, "env": {},
           "orig_header": None, "derived_header": None,
           "sidecar_records": report.get("_sidecar_records")}
    for phase in ("prefill", "decode"):
        cp = os.path.join(model_dir, f"{phase}.csv")
        if not os.path.exists(cp):
            continue
        ctx["orig_rows"][phase] = C.read_jsonl_rows(
            os.path.join(model_dir, f"{phase}.jsonl"))
        ctx["derived_rows"][phase] = C.read_jsonl_rows(
            os.path.join(out_dir, f"{phase}.jsonl"))
        ctx["orig_cells"][phase] = C.csv_cells(cp, phase)
        ctx["derived_cells"][phase] = C.csv_cells(
            os.path.join(out_dir, f"{phase}.csv"), phase)
        ctx["env"][phase] = v6_substitute(overlay, phase, model_dir)
        h1, _ = C.read_csv_rows(cp)
        h2, _ = C.read_csv_rows(os.path.join(out_dir, f"{phase}.csv"))
        ctx["orig_header"], ctx["derived_header"] = h1, h2
    rows, failed = V9.run(ctx)
    report["v9"] = rows
    return rows, failed


def review_state():
    """develop/reviews/ 의 라운드 기록과 develop/plus_at/v9_review.yaml 을 읽는다.

    라운드는 **파일명이 아니라 문서의 `covers` 필드**로 판단한다. 요청을 합쳐 보내면 답이
    하나인데, 같은 원문을 두 파일로 복제하면 독립 검토처럼 보인다 -- 외부 검토(R3b Q2)가
    그렇게 하지 말라고 했다. 대신 기록이 `covers: [R2, R3]` 로 범위를 밝히고 게이트가
    그것을 읽는다. 파일명 접두사는 `covers` 가 없는 예전 기록의 fallback 이다.
    """
    done = {}
    if os.path.isdir(REVIEW_DIR):
        for fn in sorted(os.listdir(REVIEW_DIR)):
            p = os.path.join(REVIEW_DIR, fn)
            if not os.path.isfile(p):
                continue
            covers = []
            head = io.open(p, encoding="utf-8", errors="replace").read(1200)
            if head.startswith("---"):
                fm = head.split("---", 2)
                if len(fm) >= 3:
                    try:
                        meta = yaml.safe_load(fm[1]) or {}
                        c = meta.get("covers") or meta.get("round")
                        covers = ([c] if isinstance(c, str) else list(c or []))
                    except Exception:                            # noqa: BLE001
                        covers = []
            if not covers:
                m = re.match(r"(R[0-9]+[a-z]?)-", fn)
                covers = [m.group(1)] if m else []
            for c in covers:
                done.setdefault(str(c), []).append(fn)
    v9r = {}
    if os.path.exists(V9_REVIEW):
        v9r = (yaml.safe_load(io.open(V9_REVIEW, encoding="utf-8")) or {}).get(
            "gates") or {}
    return done, v9r


def release_blockers(overlay, v9_rows):
    """정식 release 를 막는 사유 목록. 비어 있어야 released 다."""
    out = []
    for spec in overlay["substitutions"]:
        sid = spec["sub_id"]
        if spec.get("status") != "accepted":
            out.append(f"{sid}: status {spec.get('status')!r} (accepted 아님)")
        v = spec.get("verification") or {}
        if v.get("point_verified") is not True:
            out.append(f"{sid}: point_verified 가 참이 아니다")
        if v.get("semantic_evidence_verified") is not True:
            out.append(f"{sid}: semantic_evidence_verified 가 참이 아니다")
    ef = overlay.get("expected_footprint") or {}
    if ef.get("review_status") != "accepted":
        out.append(f"expected_footprint: review_status "
                   f"{ef.get('review_status')!r} (accepted 아님)")
    done, v9r = review_state()
    for rnd in REQUIRED_REVIEWS:
        if rnd not in done:
            out.append(f"검토 기록 없음: develop/reviews/{rnd}-*")
    for gid, row in v9_rows.items():
        st = (v9r.get(gid) or {}).get("review_status", row.get("review_status"))
        if st != "accepted":
            out.append(f"V9 {gid}: review_status {st!r} (accepted 아님)")
        if row.get("result") == "not_run":
            # not_run 은 "관련 있는데 평가 안 했다" -- release 를 막는다.
            # n/a 는 "overlay 가 그 대상을 선언하지 않았다" -- 막지 않는다.
            out.append(f"V9 {gid}: 평가되지 않았다 ({row['decision']})")
    return out


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--overlay")
    ap.add_argument("--expected", help="사전 승인된 expected footprint 경로")
    ap.add_argument("--publish", action="store_true")
    a = ap.parse_args()

    model_dir = os.path.join(PROJ, "models", a.model)
    ovp = a.overlay or os.path.join(OVERLAY_DIR, f"overlay-{a.model}.yaml")
    overlay = yaml.safe_load(io.open(ovp, encoding="utf-8"))
    print(f"모델 {a.model}\noverlay {os.path.relpath(ovp, PROJ)}")

    # 0) canonical 자기검사 -- 여기서 실패하면 되쓸 수 없다
    bad0 = C.selftest_roundtrip(model_dir)
    if bad0:
        print("**canonical 자기검사 실패**")
        for b in bad0:
            print("  " + b)
        return 1

    # 1) expected 는 **사전 승인된 불변 입력**이다. 적용기가 다시 만들지 않는다.
    #    외부 검토(R3): "현재 적용기는 실행할 때 refmatch 로 expected 를 다시 만들고
    #    actual 과 함께 출력합니다 ... digest 일치는 '두 구현이 일치했다' 는 증거이지
    #    '사전에 승인된 변경 허용 목록을 지켰다' 는 증거가 아닙니다."
    exp_path = a.expected or os.path.join(OVERLAY_DIR, f"expected-{a.model}.jsonl")
    if not os.path.exists(exp_path):
        print(f"**expected footprint 가 없다: {os.path.relpath(exp_path, PROJ)}**")
        print("  먼저 독립 matcher 로 만들고 검토를 받아야 한다:")
        print("    .venv/Scripts/python.exe develop/plus_at_refmatch.py "
              f"models/{a.model} {os.path.relpath(ovp, PROJ)} "
              f"{os.path.relpath(exp_path, PROJ)}")
        return 1
    expected = [json.loads(l) for l in io.open(exp_path, encoding="utf-8")
                if l.strip()]
    exp_sha = sha256_file(exp_path)
    pinned = (overlay.get("expected_footprint") or {}).get("sha256")
    if pinned and pinned != exp_sha:
        print(f"**expected footprint 가 overlay 에 박힌 해시와 다르다**")
        print(f"  overlay: {pinned}")
        print(f"  파일:    {exp_sha}")
        return 1
    if not pinned:
        print(f"  (주의) overlay 에 expected_footprint.sha256 이 없다 -- 고정되지 않았다")
    actual = select(model_dir, overlay)
    print(f"expected {len(expected)} 셀 (사전 승인 입력 {exp_sha[:16]})   "
          f"actual {len(actual)} 셀 (적용기)")

    # 2) 임시 디렉터리에 생성
    out_dir = os.path.join(model_dir, "plus_at")
    tmp = out_dir + ".tmp"
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    changes = {(r["phase"], r["op_id"], r["field"], r["shape_index"], r["axis"]):
               r["after"] for r in actual}
    wrote = {}
    for phase in ("prefill", "decode"):
        cp = os.path.join(model_dir, f"{phase}.csv")
        jp = os.path.join(model_dir, f"{phase}.jsonl")
        if not os.path.exists(cp):
            continue
        header, crows = C.read_csv_rows(cp)
        jrows = C.read_jsonl_rows(jp)
        ph_ch = {k: v for k, v in changes.items() if k[0] == phase}
        nc = apply_to_rows(crows, ph_ch, phase, is_json=False)
        nj = apply_to_rows(jrows, ph_ch, phase, is_json=True)
        if nc != nj or nc != len(ph_ch):
            print(f"  **{phase}: 적용 수가 안 맞는다 csv {nc} jsonl {nj} "
                  f"기대 {len(ph_ch)}**")
            shutil.rmtree(tmp)
            return 1
        write_csv(os.path.join(tmp, f"{phase}.csv"), header, crows)
        write_jsonl(os.path.join(tmp, f"{phase}.jsonl"), jrows)
        wrote[phase] = nc
    with io.open(os.path.join(tmp, "actual_footprint.jsonl"), "wb") as f:
        f.write(RM.canonical_bytes(actual))
    # expected 는 **입력**이다. 여기서 다시 만들지 않고 그대로 복사해 둔다(대조 편의).
    shutil.copyfile(exp_path, os.path.join(tmp, "expected_footprint.jsonl"))
    with io.open(os.path.join(tmp, "symbols.yaml"), "w", encoding="utf-8",
                 newline=chr(10)) as f:
        yaml.safe_dump({"symbols": overlay["symbols"]}, f, allow_unicode=True,
                       sort_keys=False)

    # 사이드카 -- **본표를 바꾸지 않고 식만 기록하는 계열**(residual).
    # 외부 검토(R2/R3)가 행 필드 `l` 을 축 토큰에 넣는 것을 물렸다. 표는 리터럴을 유지하고
    # 여기에 canonical cell 별 stage 와 식을 적는다. B sweep 에는 값이 변하지 않으므로
    # 잃는 것이 없다.
    sc = overlay.get("residual_sidecar")
    if sc:
        import plus_at_resid as RS
        _R, _L = int(sc["R_res"]["value"]), int(sc["L_layers"]["value"])
        recs = []
        for phase in wrote:
            rows = C.read_jsonl_rows(os.path.join(model_dir, f"{phase}.jsonl"))
            by = {int(r["op_id"]): r for r in rows}
            got = RS.cells(phase, rows, _R, _L)     # 식·cardinality 위반은 예외
            keys = {g[0] for g in got}
            cand = RS.candidates(phase, rows)
            if keys != cand:
                print(f"  **사이드카 후보 완전성 실패 ({phase}): 놓친 "
                      f"{len(cand - keys)}, 후보 밖 {len(keys - cand)}**")
                shutil.rmtree(tmp)
                return 1
            for (ph, oid, field, si, ax), before, token, st, name in got:
                row = by[oid]
                recs.append({"phase": ph, "op_id": oid, "field": field,
                             "shape_index": si, "axis": ax, "value": int(before),
                             "stage": st, "formula": name, "expr": token,
                             "layer_idx": row.get("layer_idx"),
                             "layers": row.get("layers"),
                             "module_path": row.get("module_path"),
                             "op_type": row.get("op_type")})
        if len(recs) != int(sc.get("cells") or -1):
            print(f"  **사이드카 셀 수 {len(recs)} != 선언 {sc.get('cells')}**")
            shutil.rmtree(tmp)
            return 1
        side = {"schema_version": 1,
                "what": ("본표를 바꾸지 않는 축의 식. 본표에는 리터럴이 그대로 있고 "
                         "여기에 canonical cell 별 stage 와 식이 있다."),
                "symbols": {"R_res": sc["R_res"], "L_layers": sc["L_layers"],
                            "l": sc["l"]},
                "formulas": sc["formulas"],
                "limitations": sc["limitations"],
                "cells": len(recs), "records": recs}
        with io.open(os.path.join(tmp, "expressions.yaml"), "w", encoding="utf-8",
                     newline=chr(10)) as f:
            yaml.safe_dump(side, f, allow_unicode=True, sort_keys=False)
        _sidecar_cells = len(recs)
        _sidecar_records = recs

    # 3) 검증 V1~V8
    report = {"model": a.model, "applied_per_phase": wrote,
              "sidecar_cells": locals().get("_sidecar_cells"),
              "expected_footprint_sha256": exp_sha,
              "expected_footprint_path":
                  os.path.relpath(exp_path, PROJ).replace(chr(92), "/")}
    bad = check(model_dir, overlay, expected, actual, tmp, report)

    # 3-b) V9 -- 이름만 있는 표가 아니라 실제로 돈다 (외부 검토 R3)
    report["_sidecar_records"] = locals().get("_sidecar_records")
    v9_rows, v9_failed = run_v9(model_dir, overlay, actual, tmp, report)
    report.pop("_sidecar_records", None)
    bad += v9_failed

    # 3-c) release 를 막는 사유. 하나라도 있으면 provisional.
    provisional = release_blockers(overlay, v9_rows)
    report["release_blockers"] = provisional

    print()
    print(f"바뀐 셀 {report.get('cells')}  digest {str(report.get('footprint_digest'))[:16]}")
    for k, v in sorted((report.get("per_sub") or {}).items()):
        print(f"  {k:<18} {v:>6}")
    print(f"남은 맨정수: {json.dumps(report.get('residual_literals'), ensure_ascii=False)}")
    print()
    print("V9 게이트:")
    for gid in sorted(v9_rows):
        r = v9_rows[gid]
        print(f"  {r['result']:<8} {r['decision']:<20} {gid:<36} {r['detail'][:58]}")
    if provisional:
        print()
        print(f"**provisional** -- release 를 막는 사유 {len(provisional)} 건:")
        for p in provisional[:14]:
            print(f"    {p}")
        if len(provisional) > 14:
            print(f"    ... 외 {len(provisional) - 14} 건")

    if bad:
        print()
        print(f"**검증 실패 {len(bad)} 건 -- 아무 파일도 쓰지 않는다**")
        for b in bad:
            print("  " + b)
        shutil.rmtree(tmp)
        return 1
    n_run = sum(1 for r in v9_rows.values() if r["result"] == "pass")
    n_skip = sum(1 for r in v9_rows.values() if r["result"] == "not_run")
    print()
    print(f"검증 통과 -- V1~V8 + V9 {n_run} 종 실행"
          + (f", {n_skip} 종 **미평가**" if n_skip else ""))

    if not a.publish:
        print("(--publish 없음: dry run. 임시 디렉터리를 지운다)")
        shutil.rmtree(tmp)
        return 0

    # 4) MANIFEST + 원자적 교체
    man = {
        "schema_version": 1,
        "kind": "derived_view",
        "authority": "models/<model>/{prefill,decode}.{csv,jsonl} (tracer output)",
        "status": "provisional" if provisional else "released",
        "release_blockers": provisional or None,
        # **두 커밋은 다르다.** base_results_commit 은 원본 표가 나온 커밋이고
        # tool_source_commit 은 이 도구가 들어 있는 커밋이다. 예전에는 둘을 같은
        # 값으로 적었고 그 커밋에는 도구가 아직 없었다 -- 외부 검토(R3)가 짚었다.
        "base_results_commit": base_commit(model_dir),
        "tool_source_commit": git_commit(),
        "inputs": {}, "overlay": {}, "tool": {}, "outputs": {},
        "sources": [], "v9": {},
    }
    import glob as _g
    for phase in wrote:
        for ext in ("csv", "jsonl"):
            rel = f"models/{a.model}/{phase}.{ext}"
            man["inputs"][rel] = sha256_file(os.path.join(PROJ, rel))
        # **검사가 읽는 입력도 pin 한다** (외부 검토 R3b). crosswalk 와 원시 사이드카 없이는
        # axis/reshape 게이트를 재현할 수 없다.
        for rel in (f"models/{a.model}/full/{phase}.trace.raw.jsonl",
                    f"models/{a.model}/full/{phase}.shapes.concrete.jsonl",
                    f"models/{a.model}/full/{phase}.ports.jsonl",
                    f"models/{a.model}/full/{phase}.semantic.jsonl"):
            _p = os.path.join(PROJ, rel)
            if os.path.exists(_p):
                man["inputs"][rel] = sha256_file(_p)
        for _d in ("../llm-arch-tracer-results-labeled/work/crosswalk",):
            for _cw in sorted(_g.glob(os.path.join(PROJ, _d,
                                                   f"{a.model}.{phase}.jsonl*"))):
                man["inputs"][os.path.relpath(_cw, PROJ).replace(chr(92), "/")] = \
                    sha256_file(_cw)
    prov = os.path.join(model_dir, "full", "provenance.json")
    if os.path.exists(prov):
        man["inputs"][f"models/{a.model}/full/provenance.json"] = sha256_file(prov)
    man["overlay"][os.path.relpath(ovp, PROJ).replace("\\", "/")] = sha256_file(ovp)
    man["tool"] = {
        "apply": sha256_file(os.path.join(HERE, "plus_at_apply.py")),
        "refmatch": sha256_file(os.path.join(HERE, "plus_at_refmatch.py")),
        "canon": sha256_file(os.path.join(HERE, "plus_at_canon.py")),
        "v9": sha256_file(os.path.join(HERE, "plus_at_v9.py")),
        "negctl": sha256_file(os.path.join(HERE, "plus_at_negctl.py")),
        "fixture": sha256_file(os.path.join(HERE, "fixtures", "plus_at",
                                            "cases.yaml")),
        "source_commit": git_commit(),
    }
    man["reviews"] = {}
    _done, _v9r = review_state()
    for rnd in REQUIRED_REVIEWS:
        man["reviews"][rnd] = [
            {"file": f, "sha256": sha256_file(os.path.join(REVIEW_DIR, f))}
            for f in _done.get(rnd, [])] or None
    # bundle 계약. **현재 산출물에서 만든다** -- 예전에는 철회된 심볼 이름을 하드코딩해
    # 두어 registry 와 어긋났다(외부 검토 R3b).
    _files = sorted(man["outputs"])
    # **namespace 로 나눈다.** 예전에는 architecture_symbols 가 bundle 전체를 뜻하는
    # 것처럼 보였고, 사이드카의 R_res/L_layers 를 빼먹어 정상적인 사이드카 식도 거부될
    # 계약이 됐다(외부 검토 R3c).
    _sc0 = overlay.get("residual_sidecar") or {}
    _ns = {
        "table_added_symbols": {n: d.get("kind")
                                for n, d in sorted(overlay["symbols"].items())},
        "sidecar_architecture_symbols": sorted(
            k for k in ("R_res", "L_layers") if k in _sc0),
        "sidecar_row_variables": (["l"] if "l" in _sc0 else []),
        "sidecar_allowed_functions": ["ceil"],
        "base_table_symbols": "inherited_from_authority",
    }
    man["bundle_contract"] = {
        "one_bundle": _files,
        "inseparable": (
            "표(csv/jsonl)는 symbols.yaml 과 **분리 불가**하다. 표만 떼어 배포하면 "
            "trace_artifact 심볼이 아키텍처 심볼로 오독된다."),
        "symbol_namespaces": _ns,
        "sidecar": (
            "expressions.yaml 이 있으면 그것도 같은 bundle 이다. 본표에 리터럴로 남은 "
            "residual 누적 폭(2..9)의 stage 와 식이 거기 있다. 사이드카가 없는 소비자는 "
            "숫자 표만 쓸 수 있고 residual recurrence 의미는 복원할 수 없다."
            if "expressions.yaml" in man["outputs"] else None),
        "sidecar_join_key": ["phase", "op_id", "field", "shape_index", "axis"],
        "canonical_cell_rule": (
            "op_id 는 **발행본 표의 번호**다(0.. 로 재번호된 것). 원시 원장 op_id 와 다른 "
            "번호 공간이므로 원시와 잇는 데는 crosswalk 이 필요하다."),
        "reject_unknown_symbol": (
            "**namespace 별로** 적용한다. 본표의 토큰은 트레이서 심볼표 + "
            "table_added_symbols 로 해석돼야 하고, 사이드카의 식은 "
            "sidecar_architecture_symbols + sidecar_row_variables + "
            "sidecar_allowed_functions 로 해석돼야 한다. 모든 심볼이 symbols.yaml 에 "
            "있어야 한다고 보면 정상적인 사이드카 식도 거부된다."),
        "phase_consistency": (
            "prefill 과 decode 의 사이드카 레코드 수와 식 분포가 같아야 한다 -- "
            "V9 의 sidecar_phase_consistency 가 검사한다"),
        "caveat_stays": (
            "caveat 열은 MoE 행에 그대로 남아 있다. 총 expert projection FLOPs 는 "
            "보존되나 전문가별 분포·active expert 수·weight traffic·cache·latency 는 "
            "보존되지 않는다."),
    }
    _ev_owners = list(overlay["substitutions"])
    if overlay.get("residual_sidecar"):
        _sc2 = dict(overlay["residual_sidecar"])
        _sc2["sub_id"] = "k3-residual-sidecar"
        _ev_owners.append(_sc2)
    # provenance 에서 모델 id 와 트레이스에 실제로 쓰인 revision 을 읽는다.
    _mid = _rev = None
    _hub = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    if os.path.exists(prov):
        _pd = json.load(io.open(prov, encoding="utf-8"))
        _mid, _rev = _pd.get("model_id"), _pd.get("revision_resolved")
    man["source_pin"] = {"model_id": _mid, "revision_resolved": _rev,
                         "hub": "~/.cache/huggingface/hub"}
    for spec in _ev_owners:
        for ev in spec.get("evidence") or []:
            p = ev["file"]
            import glob as _glob
            cand = [os.path.join(PROJ, p),
                    os.path.join(PROJ, ".venv", "Lib", "site-packages", p), p]
            # **모델 remote code 는 provenance 의 revision 에 고정한다.**
            # 예전엔 `models--*/snapshots/*` 를 glob 했다 -- 이 캐시에 snapshot 이 셋
            # 있고 glob 이 트레이스에 쓰인 판이 아닌 것을 집었다(외부 검토 R3c. 실제로
            # 9f62e4e9 를 집고 있었다). 고정 snapshot 에 없으면 **다른 snapshot 으로
            # 넘어가지 않고** 아래에서 중단한다.
            if _mid and _rev:
                _slug = "models--" + _mid.replace("/", "--")
                cand.insert(0, os.path.join(_hub, _slug, "snapshots", _rev, p))
            cand += _glob.glob(os.path.join(PROJ, ".venv", "Lib",
                                            "site-packages", "**", p),
                               recursive=True)
            found = next((x for x in cand if os.path.isfile(x)), None)
            if not found:
                # **source SHA-256 은 계약이다.** null 을 적고 넘어가면 안 된다
                # -- 외부 검토(R3 2 차)가 C 의 근거 셋이 null 인 것을 짚었다.
                raise SystemExit(
                    f"근거 파일을 못 찾았다: {p} "
                    f"(sub_id {spec.get('sub_id')}). "
                    f"source SHA-256 없이는 publish 하지 않는다")
            man["sources"].append({
                "file": p, "lines": ev.get("lines"),
                "sha256": sha256_file(found),
                "resolved": os.path.relpath(found, PROJ).replace(chr(92), "/"),
                "sub_id": spec["sub_id"]})
    man["v9"] = v9_rows
    for phase in wrote:
        for ext in ("csv", "jsonl"):
            man["outputs"][f"{phase}.{ext}"] = sha256_file(
                os.path.join(tmp, f"{phase}.{ext}"))
    for f in ("actual_footprint.jsonl", "expected_footprint.jsonl",
              "symbols.yaml", "expressions.yaml"):
        _p = os.path.join(tmp, f)
        if os.path.exists(_p):
            man["outputs"][f] = sha256_file(_p)
    man["report"] = report
    # bundle 계약의 파일 목록은 **outputs 가 채워진 뒤** 정한다. 앞서 만들면 빈 목록이
    # 들어간다 -- 실제로 그렇게 나갔다(2026-09-28).
    man["bundle_contract"]["one_bundle"] = sorted(man["outputs"])
    man["bundle_contract"]["sidecar"] = (
        "expressions.yaml 도 같은 bundle 이다. 본표에 리터럴로 남은 residual 누적 폭"
        "(2..9)의 stage 와 식이 거기 있다. 사이드카가 없는 소비자는 숫자 표만 쓸 수 있고 "
        "residual recurrence 의미는 복원할 수 없다."
        if "expressions.yaml" in man["outputs"] else None)
    with io.open(os.path.join(tmp, "MANIFEST.json"), "w", encoding="utf-8",
                 newline=chr(10)) as f:
        json.dump(man, f, ensure_ascii=False, indent=1, sort_keys=True)

    bak = out_dir + ".bak"
    if os.path.isdir(out_dir):
        if os.path.isdir(bak):
            shutil.rmtree(bak)
        os.replace(out_dir, bak)
    try:
        os.replace(tmp, out_dir)
    except Exception:
        if os.path.isdir(bak):
            os.replace(bak, out_dir)
        raise
    if os.path.isdir(bak):
        shutil.rmtree(bak)
    print(f"published {os.path.relpath(out_dir, PROJ)}"
          f"  (status {man['status']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
