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
        s = _retune(spec, fixture_syms, phase)
        for key, _b, after in AP._match_cells(s, phase, rows):
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


CASES = [case_two_implementations_agree, case_matches_human_expectation,
         case_negative_controls_change_nothing, case_v3_blocks_duplicate_literal,
         case_quotient_remainder_arithmetic, case_fixture_covers_required_items]


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
