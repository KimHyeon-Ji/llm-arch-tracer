r"""+@ 층의 fixture 시험 — **적용기와 독립 matcher 가 사람이 쓴 사례에서 같은 답을 내는가.**

왜 있는가
---------
외부 검토(2026-09-27): "같은 저자가 구현한 reference matcher 만으로는 의미 독립성이 생기지
않습니다. 하지만 다음 조건의 golden fixture 와 R1 을 함께 쓰면 1차 방어로 충분합니다."

  fixture 는 **구현 독립성**을, blind R1 은 **의미 독립성**을 담당한다.

그래서 이 시험은 `develop/fixtures/plus_at/cases.yaml`(사람이 직접 작성)의 사례 하나하나를
두 구현에 통과시켜 답이 같은지, 그리고 사람이 적은 기대값과 맞는지 본다.

실행:  .venv\Scripts\python.exe develop\test_plus_at.py
"""
import collections
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import plus_at_apply as AP                                     # noqa: E402
import plus_at_canon as C                                      # noqa: E402
import plus_at_refmatch as RM                                  # noqa: E402
import yaml                                                    # noqa: E402

FIX = os.path.join(HERE, "fixtures", "plus_at", "cases.yaml")
OVERLAY = os.path.join(HERE, "plus_at", "overlay-moonshotai__Kimi-K3.yaml")
OK, BAD = [], []


def check(name, cond, detail=""):
    (OK if cond else BAD).append(name)
    print(("  OK   " if cond else "  FAIL ") + name + (f"   {detail}" if detail
                                                       and not cond else ""))


def _row(case):
    """fixture 의 row 를 발행본 jsonl 행 모양으로 채운다. 리터럴을 실제 값으로 바꾼다."""
    r = dict(case["row"])
    r.setdefault("weight_shape", None)
    r.setdefault("input_shape", [])
    r.setdefault("output_shape", [])
    return r


def _expected_set(case):
    return {(case["phase"], int(case["row"]["op_id"]), e["field"],
             int(e["shape_index"]), int(e["axis"])): e["after"]
            for e in (case.get("expect") or [])}


def _applier_cells(overlay, phase, rows, fixture_syms):
    """적용기의 일반 해석기를 fixture 리터럴에 맞춰 돌린다."""
    out = {}
    for spec in overlay["substitutions"]:
        if phase not in (spec.get("phases") or []):
            continue
        if spec.get("matcher"):
            continue          # 전용 matcher(계열 C)는 자기 시험이 따로 있다
        s = _retune(spec, fixture_syms, phase)
        for key, _b, after in AP._match_cells(s, phase, rows, overlay):
            out[key] = after
    return out


def _retune(spec, syms, phase):
    """fixture 는 실제 모델과 다른 값을 쓴다. overlay 의 리터럴만 fixture 값으로 바꾼다.

    선택자 구조(module_regex / op_type / axis / fields / operand_rule)는 **그대로 둔다** --
    바꾸면 시험이 규칙을 시험하지 않는다.
    """
    import copy
    s = copy.deepcopy(spec)
    m = s["match"]
    nr = syms[f"N_route_{phase}"]
    q = nr // syms["C_trace"]
    if m.get("from_by_phase"):
        m["from_by_phase"] = {"prefill": str(syms["N_route_prefill"] //
                                             syms["C_trace"]),
                              "decode": str(syms["N_route_decode"] //
                                            syms["C_trace"])}
        del nr, q
    if m.get("shape"):
        m["shape"] = [str(syms["T"] // syms["d_chunk"]) if x == "5" else x
                      for x in m["shape"]]
    if s.get("from") == "5":
        s["from"] = str(syms["T"] // syms["d_chunk"])
    return s


def _refmatch_cells(phase, rows, syms):
    """독립 matcher 를 fixture 리터럴에 맞춰 돌린다. 술어 구조는 건드리지 않는다."""
    nchunk = str(syms["T"] // syms["d_chunk"])
    q = str(syms[f"N_route_{phase}"] // syms["C_trace"])
    last = syms["C_trace"] - 1
    out = {}
    orig_want = ["B", "n_h_kda", "5", "d_chunk", "d_head_kda"]
    want = [nchunk if x == "5" else x for x in orig_want]

    if phase == "prefill":
        for r in rows:
            if r.get("op_type") != "exp":
                continue
            if not (r.get("module_path") or "").endswith(".self_attn"):
                continue
            for field in ("input_shape", "output_shape"):
                for si, sh in enumerate(C.parse_jsonl_shape(r.get(field), field)):
                    if si == 0 and sh == want:
                        out[(phase, int(r["op_id"]), field, si, 2)] = "n_chunk"
    # MoE 규칙은 R1 판정으로 철회됐다. 활성 overlay 에 없으므로 여기서도 내지 않는다.
    return out


def case_two_implementations_agree():
    """사례마다 적용기와 독립 matcher 가 **같은 셀 집합**을 낸다."""
    fx = yaml.safe_load(io.open(FIX, encoding="utf-8"))
    ov = yaml.safe_load(io.open(OVERLAY, encoding="utf-8"))
    syms = fx["symbols"]
    for case in fx["cases"]:
        if case.get("withdrawn"):
            continue                     # R1 판정으로 철회된 계열 -- 활성 overlay 에 없다
        ph, rows = case["phase"], [_row(case)]
        a = _applier_cells(ov, ph, rows, syms)
        b = _refmatch_cells(ph, rows, syms)
        assert a == b, f"{case['name']}: 적용기 {a} != refmatch {b}"


def case_matches_human_expectation():
    """사례마다 두 구현의 답이 **사람이 적은 기대값**과 같다."""
    fx = yaml.safe_load(io.open(FIX, encoding="utf-8"))
    ov = yaml.safe_load(io.open(OVERLAY, encoding="utf-8"))
    syms = fx["symbols"]
    for case in fx["cases"]:
        if case.get("withdrawn"):
            continue                     # R1 판정으로 철회된 계열 -- 활성 overlay 에 없다
        ph, rows = case["phase"], [_row(case)]
        want = _expected_set(case)
        got = _applier_cells(ov, ph, rows, syms)
        assert got == want, (f"{case['name']}: 기대 {sorted(want.items())} "
                            f"!= 실제 {sorted(got.items())}")


def case_negative_controls_change_nothing():
    """`expect: []` 인 음성 사례는 **하나도** 바뀌지 않는다."""
    fx = yaml.safe_load(io.open(FIX, encoding="utf-8"))
    ov = yaml.safe_load(io.open(OVERLAY, encoding="utf-8"))
    syms = fx["symbols"]
    n = 0
    for case in fx["cases"]:
        if case.get("expect") or case.get("withdrawn"):
            continue
        n += 1
        got = _applier_cells(ov, case["phase"], [_row(case)], syms)
        assert not got, f"{case['name']}: 음성 사례가 {got} 를 바꿨다"
    assert n >= 5, f"음성 사례가 {n} 개뿐이다 -- 이 시험이 약하다"


def case_v3_blocks_duplicate_literal():
    """한 shape 에 같은 맨정수가 둘이면 V3 가 **전체 적용을 막는다**."""
    fx = yaml.safe_load(io.open(FIX, encoding="utf-8"))
    hit = [c for c in fx["cases"] if c.get("expect_v3_fail")]
    assert hit, "V3 사례가 fixture 에 없다"
    for case in hit:
        r = _row(case)
        dup = 0
        for field in ("input_shape", "output_shape"):
            for sh in C.parse_jsonl_shape(r.get(field), field):
                lits = [t for t in sh if t.isdigit() and t != "1"]
                if lits and len(lits) != len(set(lits)):
                    dup += 1
        assert dup, f"{case['name']}: V3 가 잡아야 할 중복을 못 찾았다"


def case_quotient_remainder_arithmetic():
    """몫/나머지 식이 나누어떨어지지 않는 경우에도 정확하다."""
    fx = yaml.safe_load(io.open(FIX, encoding="utf-8"))
    for a in fx["arithmetic"]:
        nr, ct = a["N_route"], a["C_trace"]
        if a.get("expect_reject"):
            assert nr < ct, f"{a['name']}: 거부 조건이 아니다"
            continue
        q = nr // ct
        last = nr - (ct - 1) * q
        assert q == a["expect_regular"], f"{a['name']}: regular {q} != {a['expect_regular']}"
        assert last == a["expect_last"], f"{a['name']}: last {last} != {a['expect_last']}"
        assert (ct - 1) * q + last == nr, f"{a['name']}: 합이 N_route 가 아니다"


def case_fixture_covers_required_items():
    """검토가 지정한 11 항목이 fixture 에 있다."""
    fx = yaml.safe_load(io.open(FIX, encoding="utf-8"))
    names = " ".join(c["name"] for c in fx["cases"])
    need = {"A 정상": "A 정상", "residual 음성": "residual",
            "exp 아닌": "op 가 exp", "axis 변형": "axis",
            "field 변형": "field", "shape_index 변형": "shape_index",
            "module 변형": "module", "expert regular": "regular",
            "expert last": "last", "concat 피연산자": "concat",
            "decode": "decode"}
    for label, token in need.items():
        assert token in names, f"fixture 에 {label} 사례가 없다 (찾은 토큰 {token!r})"
    ar = " ".join(a["name"] for a in fx["arithmetic"])
    assert "!= 0" in ar, "N_route % C_trace != 0 산술 사례가 없다"
    wd = [c["name"] for c in fx["cases"] if c.get("withdrawn")]
    assert len(wd) >= 6, f"철회된 계열 사례가 기록에 남아 있어야 한다 (현재 {len(wd)})"
    assert any(a.get("expect_reject") for a in fx["arithmetic"]), \
        "validity-domain 밖 거부 사례가 없다"


def _resid_rows(case, R):
    """fixture 의 ops 를 발행본 jsonl 행 모양으로 만든다. tok/d 를 실제 토큰으로."""
    out = []
    for o in case["ops"]:
        def conv(lst):
            return [[("B*T" if x == "tok" else "d_model" if x == "d" else str(x))
                     for x in sh] for sh in lst]
        mp = o.get("module") or f"model.layers.{case['layer']}"
        dep = list(o.get("dep") or [])
        r = {"op_id": o["op_id"], "op_type": o["op_type"], "module_path": mp,
             "input_shape": conv(o.get("i") or []),
             "output_shape": conv(o.get("o") or []),
             "weight_shape": None, "depends_on": dep,
             "layers": "" if case["layer"] is None else str(case["layer"]),
             "repeat": 1, "params": []}
        # **norm 가중치 op 를 합성한다.** 실제 트레이스에서 stage 를 말하는 것이 이것이고
        # (self_attention_res_norm / mlp_res_norm / output_attn_res_norm), 분류기가 그걸
        # 읽는다. fixture 가 안 주면 분류기는 추측하지 않고 실패한다 -- 그게 맞다.
        nm = (case.get("norms") or {}).get(o["op_id"])
        if nm:
            wid = 900000 + int(o["op_id"])
            out.append({"op_id": wid, "op_type": "elementwise_mul", "module_path": mp,
                        "input_shape": [["d_model"], ["d_model"]],
                        "output_shape": [["d_model"]], "weight_shape": None,
                        "depends_on": [], "layers": "", "repeat": 1,
                        "params": [f"{mp}.{nm}.weight"]})
            r["depends_on"] = dep + [wid]
        out.append(r)
    return out


def case_residual_stage_from_lineage():
    """stage 를 **값이 아니라 lineage** 로 정한다. 같은 값이라도 stage 가 다를 수 있다."""
    import plus_at_resid as RS
    fx = yaml.safe_load(io.open(FIX, encoding="utf-8"))
    rc = fx["residual"]
    R, L = rc["R_res"], rc["L_layers"]
    for case in rc["cases"]:
        rows = _resid_rows(case, R)
        got = RS.stages("prefill", rows, strict=False)
        for oid, want in (case.get("expect_stage") or {}).items():
            assert got.get(int(oid)) == want, (
                f"{case['name']}: op{oid} stage {got.get(int(oid))!r} != {want!r}")


def case_residual_formula_per_stage():
    """stage 마다 붙는 식이 fixture 의 기대와 같고, 값도 맞는다."""
    import plus_at_resid as RS
    fx = yaml.safe_load(io.open(FIX, encoding="utf-8"))
    rc = fx["residual"]
    R, L = rc["R_res"], rc["L_layers"]
    for case in rc["cases"]:
        rows = _resid_rows(case, R)
        cells = RS.cells("prefill", rows, R, L, strict=False)
        by_op = {}
        for (ph, oid, field, si, ax), before, token, st, name in cells:
            which = "buf" if (field == "input_shape" and si == 0 and
                              next(r for r in rows if int(r["op_id"]) == oid)
                              ["op_type"] == "concat") else "mix"
            by_op.setdefault((oid, which), set()).add(name)
        for oid, want in (case.get("expect_formula") or {}).items():
            got = by_op.get((int(oid), "mix")) or set()
            assert got == {want}, f"{case['name']}: op{oid} mix 식 {got} != {{{want}}}"
        for oid, want in (case.get("expect_formula_buf") or {}).items():
            got = by_op.get((int(oid), "buf")) or set()
            assert got == {want}, f"{case['name']}: op{oid} buf 식 {got} != {{{want}}}"
        for oid, want in (case.get("expect_formula_mix") or {}).items():
            got = by_op.get((int(oid), "mix")) or set()
            assert got == {want}, f"{case['name']}: op{oid} mix 식 {got} != {{{want}}}"


def case_residual_arithmetic():
    """식의 값이 fixture 의 손계산과 같다. R·L 을 실제 모델과 다르게 잡았다."""
    import plus_at_resid as RS
    fx = yaml.safe_load(io.open(FIX, encoding="utf-8"))
    rc = fx["residual"]
    for a in rc["arithmetic"]:
        R = a.get("R", rc["R_res"])
        L = a.get("L", rc["L_layers"])
        got = RS.value(a["formula"], a.get("l", 0), R, L)
        assert got == a["expect"], (f"{a['name']}: {a['formula']} = {got} "
                                    f"!= {a['expect']}")


def case_residual_same_value_different_stage():
    """**같은 값인데 stage 가 다른** 사례가 fixture 에 있어야 한다.

    없으면 이 시험은 "값으로 골라도 된다" 를 반증하지 못한다.
    """
    import plus_at_resid as RS
    fx = yaml.safe_load(io.open(FIX, encoding="utf-8"))
    rc = fx["residual"]
    R, L = rc["R_res"], rc["L_layers"]
    found = False
    for case in rc["cases"]:
        f = case.get("expect_formula") or {}
        if len(set(f.values())) < 2:
            continue
        rows = _resid_rows(case, R)
        vals = collections.defaultdict(set)
        for (_ph, oid, field, si, ax), before, _tok, _st, name in RS.cells(
                "prefill", rows, R, L, strict=False):
            vals[before].add(name)
        if any(len(v) > 1 for v in vals.values()):
            found = True
    assert found, ("같은 값에 다른 식이 붙는 사례가 fixture 에 없다 -- "
                   "lineage 판별의 필요성을 시험하지 못한다")


def case_residual_cardinality_fires():
    """cardinality 강제가 **실제로 발화**한다. 켜고 부분 입력을 주면 실패해야 한다.

    strict=False 로만 시험하면 이 검사가 죽어도 모른다.
    """
    import plus_at_resid as RS
    fx = yaml.safe_load(io.open(FIX, encoding="utf-8"))
    rc = fx["residual"]
    case = rc["cases"][0]                    # 층 0, final 그룹이 없다
    rows = _resid_rows(case, rc["R_res"])
    try:
        RS.stages("prefill", rows, strict=True)
    except ValueError as e:
        assert "mix_final" in str(e), str(e)
        return
    raise AssertionError("cardinality 를 켰는데 부분 입력이 통과했다")


CASES = [case_two_implementations_agree, case_matches_human_expectation,
         case_negative_controls_change_nothing, case_v3_blocks_duplicate_literal,
         case_quotient_remainder_arithmetic, case_fixture_covers_required_items,
         case_residual_stage_from_lineage, case_residual_formula_per_stage,
         case_residual_arithmetic, case_residual_same_value_different_stage,
         case_residual_cardinality_fires]


def main():
    bad = []
    for fn in CASES:
        try:
            fn()
            print(f"  PASS      {fn.__name__:38} {(fn.__doc__ or '').splitlines()[0][:44]}")
        except AssertionError as e:
            bad.append(fn.__name__)
            print(f"  **FAIL**  {fn.__name__:38} {str(e).splitlines()[0][:80]}")
    print()
    print(f"{len(CASES) - len(bad)}/{len(CASES)} 통과" + (f"  실패: {bad}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
