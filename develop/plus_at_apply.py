r"""+@ overlay 를 적용해 `models/<m>/plus_at/` derived view 를 만든다.

성격
----
`plus_at/` 은 **비권위적 derived view** 다. 권위 있는 출처는 트레이서가 낸
`models/<m>/{prefill,decode}.{csv,jsonl}` 이고 그것은 건드리지 않는다. 이 도구는 원본 +
overlay 로 언제든 같은 결과를 다시 만든다. `plus_at/` 을 손으로 고치면 안 된다.

이 도구가 하는 일
-----------------
1. overlay 의 `match:` 블록을 **일반 해석**해 바꿀 셀을 고른다
   (독립 matcher `plus_at_refmatch.py` 는 같은 규칙을 손으로 구현한다 -- 공통 버그 차단).
2. V1~V9 를 돈다. **하나라도 실패하면 아무 파일도 쓰지 않는다.**
3. 임시 디렉터리에 전부 만들고 통과했을 때만 원자적으로 교체한다.
4. MANIFEST.json 에 입력·overlay·도구·출력·source 해시와 V9 승계 표를 적는다.

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
  V9 게이트 승계 게이트별 decision/reason/evidence/review_status, **deny-by-default**

실행:
    .venv\Scripts\python.exe develop\plus_at_apply.py <모델명> [--expected <경로>] [--publish]
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

# V9 -- 게이트별 처분. deny-by-default: base 게이트 목록에 있고 여기 없으면 release 실패.
# `rerun` 은 이 도구가 실제로 다시 도는 것, `inherited_unchanged` 는 V1/V4 가 불변을
# 증명하므로 원본 PASS 를 승계하는 것, `not_applicable` 은 derived table 에 해당하지
# 않는 것(원시 원장·포트 사이드카만 보는 검사). 전부 R3 검토 대상이다.
V9_TABLE = {
    "schema_shape_rank_token_type":
        ("rerun", "라벨 교체가 축 개수나 토큰 종류를 바꾸면 안 된다", "V4+V5"),
    "symbol_declared":
        ("rerun", "표에 새로 넣은 이름이 symbols.yaml 에 선언돼 있어야 한다", "V6 전단계"),
    "expression_no_cycle":
        ("rerun", "symbols 의 expr 가 서로를 순환 참조하면 대입이 안 끝난다", "V6"),
    "substitution_nonneg_integer":
        ("rerun", "대입 결과가 정수이며 음수가 아니다", "V6"),
    "zero_axis_only_initial_residual":
        ("rerun", "`0` 은 선언된 초기 residual 두 자리에서만 허용", "V7 잔여 목록"),
    "batch_seq_head_axis_consistency":
        ("rerun", "배치·시퀀스·head 축 위치 일관성", "V9 구현"),
    "head_scope_exclusive":
        ("rerun", "n_h / n_kv / n_h_kda 가 한 shape 에 공존하지 않는다", "V9 구현"),
    "moe_quotient_remainder_consistency":
        ("rerun", "concat 입력의 regular/last 대응이 전문가 순서와 맞는다", "V9 구현"),
    "prefill_decode_structure":
        ("rerun", "두 phase 의 공통 구조가 같은 방식으로 바뀌었다", "V9 구현"),
    "layers_repeat_consistency":
        ("rerun", "layers 파싱 후 repeat == 펼친 층 수", "V9 구현"),
    "row_metadata_preserved":
        ("rerun", "caveat · unmapped · params · depends_on 보존", "V9 구현"),
    "op_id_dag":
        ("rerun", "op_id 유일성, dependency 존재, DAG 비순환", "V9 구현"),
    "axis_class_consistency":
        ("not_applicable", "등가류는 원시 원장과 포트 사이드카 위에 선다 -- "
                           "발행본 표만으로 재구성할 수 없다", "src/axis_classes.py"),
    "reshape_derivation":
        ("not_applicable", "구체 shape 사이드카가 필요하다 -- 발행본에 없다",
         "src/build_table.reshape_disagreements"),
    "port_coverage":
        ("not_applicable", "포트 사이드카를 본다 -- derived view 의 범위 밖",
         "develop/build_review_bundle.port_coverage"),
    "rules_fingerprint":
        ("inherited_unchanged", "라벨 규칙을 바꾸지 않았다. 원본 PASS 를 승계한다",
         "full/generated.json.label_inputs"),
}


def sha256_file(p):
    h = hashlib.sha256()
    with io.open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def git_commit():
    try:
        return subprocess.run(["git", "-C", PROJ, "rev-parse", "HEAD"],
                              capture_output=True).stdout.decode().strip() or None
    except Exception:                                            # noqa: BLE001
        return None


# ------------------------------------------------------------- overlay 해석 (적용기 쪽)
def _match_cells(spec, phase, rows):
    """overlay 의 `match:` 블록을 **일반 해석**해 (key, before, after) 를 낸다."""
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
            for key, before, after in _match_cells(spec, phase, rows):
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
        if not expr or "trace." in expr:
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
        for k, (bfr, aft, sub_id, _mp, _ot) in amap.items():
            if k[0] != phase:
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

        # V9 일부 -- 행 메타데이터 보존, op_id/DAG, layers·repeat
        ojr = {int(r["op_id"]): r for r in C.read_jsonl_rows(oj)}
        njr = {int(r["op_id"]): r for r in C.read_jsonl_rows(nj)}
        if set(ojr) != set(njr):
            bad.append(f"V9 {phase}: op_id 집합이 바뀌었다")
        meta = ("caveat", "unmapped", "params", "depends_on", "repeat", "layers",
                "module_path", "op_type", "raw_op", "layer_idx", "block_type")
        mb = 0
        for oid, a in ojr.items():
            b = njr.get(oid) or {}
            for key in meta:
                if a.get(key) != b.get(key):
                    mb += 1
                    if mb <= 3:
                        bad.append(f"V9 {phase} op{oid}: {key} 가 바뀌었다")
        if mb:
            bad.append(f"V9 {phase}: 행 메타데이터 {mb} 개가 바뀌었다")
        for oid, r in njr.items():
            for dep in (r.get("depends_on") or []):
                if int(dep) not in njr:
                    bad.append(f"V9 {phase} op{oid}: depends_on {dep} 가 없다")
                    break
                if int(dep) >= oid:
                    bad.append(f"V9 {phase} op{oid}: depends_on {dep} 가 자기 이후다 "
                               f"(DAG 위반)")
                    break
        for oid, r in njr.items():
            ls = C.expand_layers(r.get("layers"))
            rep = r.get("repeat")
            if ls and isinstance(rep, int) and rep != len(ls):
                bad.append(f"V9 {phase} op{oid}: repeat {rep} != 펼친 층 수 {len(ls)} "
                           f"(layers={r.get('layers')!r})")
                break

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


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--overlay")
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

    # 1) expected (독립 matcher) / actual (overlay 해석)
    expected = RM.build(model_dir, ovp)
    actual = select(model_dir, overlay)
    print(f"expected {len(expected)} 셀 (refmatch)   actual {len(actual)} 셀 (적용기)")

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
    with io.open(os.path.join(tmp, "expected_footprint.jsonl"), "wb") as f:
        f.write(RM.canonical_bytes(expected))
    with io.open(os.path.join(tmp, "symbols.yaml"), "w", encoding="utf-8",
                 newline=chr(10)) as f:
        yaml.safe_dump({"symbols": overlay["symbols"]}, f, allow_unicode=True,
                       sort_keys=False)

    # 3) 검증
    report = {"model": a.model, "applied_per_phase": wrote}
    bad = check(model_dir, overlay, expected, actual, tmp, report)

    # 3-b) status: accepted 가 아닌 항목이 있으면 provisional
    st = {s["sub_id"]: s.get("status") for s in overlay["substitutions"]}
    provisional = [k for k, v in st.items() if v != "accepted"]
    report["provisional"] = provisional

    print()
    print(f"바뀐 셀 {report.get('cells')}  digest {str(report.get('footprint_digest'))[:16]}")
    for k, v in sorted((report.get("per_sub") or {}).items()):
        print(f"  {k:<18} {v:>6}")
    print(f"남은 맨정수: {json.dumps(report.get('residual_literals'), ensure_ascii=False)}")
    if provisional:
        print(f"**provisional** -- accepted 아닌 항목: {provisional}")

    if bad:
        print()
        print(f"**검증 실패 {len(bad)} 건 -- 아무 파일도 쓰지 않는다**")
        for b in bad:
            print("  " + b)
        shutil.rmtree(tmp)
        return 1
    print("\n검증 V1~V9 통과")

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
        "provisional_reason": (f"substitutions not accepted: {provisional}"
                               if provisional else None),
        "base_results_commit": git_commit(),
        "inputs": {}, "overlay": {}, "tool": {}, "outputs": {},
        "sources": [], "v9": {},
    }
    for phase in wrote:
        for ext in ("csv", "jsonl"):
            rel = f"models/{a.model}/{phase}.{ext}"
            man["inputs"][rel] = sha256_file(os.path.join(PROJ, rel))
    prov = os.path.join(model_dir, "full", "provenance.json")
    if os.path.exists(prov):
        man["inputs"][f"models/{a.model}/full/provenance.json"] = sha256_file(prov)
    man["overlay"][os.path.relpath(ovp, PROJ).replace("\\", "/")] = sha256_file(ovp)
    man["tool"] = {
        "apply": sha256_file(os.path.join(HERE, "plus_at_apply.py")),
        "refmatch": sha256_file(os.path.join(HERE, "plus_at_refmatch.py")),
        "canon": sha256_file(os.path.join(HERE, "plus_at_canon.py")),
        "git_commit": git_commit(),
    }
    for spec in overlay["substitutions"]:
        for ev in spec.get("evidence") or []:
            p = ev["file"]
            cand = [os.path.join(PROJ, p),
                    os.path.join(PROJ, ".venv", "Lib", "site-packages", p), p]
            found = next((x for x in cand if os.path.isfile(x)), None)
            man["sources"].append({
                "file": p, "lines": ev.get("lines"),
                "sha256": sha256_file(found) if found else None,
                "resolved": os.path.relpath(found, PROJ).replace("\\", "/")
                            if found else None,
                "sub_id": spec["sub_id"]})
    for gate, (dec, reason, ev) in V9_TABLE.items():
        man["v9"][gate] = {"decision": dec, "reason": reason, "evidence": ev,
                           "review_status": "proposed"}
    for phase in wrote:
        for ext in ("csv", "jsonl"):
            man["outputs"][f"{phase}.{ext}"] = sha256_file(
                os.path.join(tmp, f"{phase}.{ext}"))
    for f in ("actual_footprint.jsonl", "expected_footprint.jsonl", "symbols.yaml"):
        man["outputs"][f] = sha256_file(os.path.join(tmp, f))
    man["report"] = report
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
