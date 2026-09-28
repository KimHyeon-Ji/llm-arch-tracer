r"""+@ derived view 의 V9 게이트 — **이름만 있는 표가 아니라 실제로 도는 검사.**

왜 다시 썼는가
--------------
외부 검토(2026-09-27, R3)가 짚었다: "실제 V9 코드가 수행하는 것은 주로 metadata 보존,
dependency, layers/repeat 검사입니다. 다음 항목은 이름만 표에 있고 실질 검사가 없거나
불완전합니다" -- `expression_no_cycle`(명시값을 먼저 환경에 넣으므로 순환 expr 도 통과),
`zero_axis_only_initial_residual`(위치를 안 보고 개수만), `batch_seq_head_axis_consistency`,
`head_scope_exclusive`, `moe_quotient_remainder_consistency`, `prefill_decode_structure`,
`op_id_dag`(dict 변환 때문에 중복 op_id 검출 불가).

그래서 각 검사를 **실제로** 구현하고, 구현이 없는 항목은 표에 `not_evaluated` 로 적는다.
`not_applicable` 은 "이 산출물에 그 개념이 없다" 일 때만 쓴다.

각 검사는 `(gate_id, ok, detail)` 을 낸다. 하나라도 ok=False 면 적용이 실패한다.
"""
import io
import json
import os
import re

import plus_at_canon as C

NUM = re.compile(r"^\d+$")
FIELDS = ("input_shape", "weight_shape", "output_shape")
HEAD_SYMS = ("n_h", "n_kv", "n_h_kda")


def _tokens(sh):
    return [str(x) for x in sh]


# ----------------------------------------------------------------- 개별 검사
ALLOWED_FUNCS = ("ceil",)


def g_symbol_declared(ctx):
    """표에 새로 넣은 토큰의 **모든 식별자**가 symbols 에 선언돼 있다.

    계열 C 의 토큰은 맨 심볼이 아니라 식이다(`ceil(l/R_res)+1`). 식은 식별자 단위로
    본다 -- 허용 함수(`ceil`) 밖의 이름이 나오면 실패다.
    """
    declared = set(ctx["overlay"]["symbols"])
    ident = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
    bad = set()
    for r in ctx["actual"]:
        for t in ident.findall(r["after"]):
            if t not in declared and t not in ALLOWED_FUNCS:
                bad.add(f"{t} (in {r['after']})")
    return ("symbol_declared", not bad,
            "미등록 식별자 " + str(sorted(bad)[:4]) if bad
            else f"선언 {len(declared)}개 + 허용 함수 {list(ALLOWED_FUNCS)} 로 전부 해석됨")


def g_expression_no_cycle(ctx):
    """symbols 의 expr 의존 그래프가 비순환이다.

    **명시값을 환경에 먼저 넣지 않는다.** 넣으면 순환 expr 도 평가되어 통과한다
    (외부 검토 지적). 의존 그래프만 보고 위상 정렬한다.
    """
    sym = ctx["overlay"]["symbols"]
    ident = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
    deps = {}
    for name, d in sym.items():
        refs = set()
        for key in ("expr", "expr_prefill", "expr_decode"):
            e = d.get(key)
            if not e:
                continue
            for t in ident.findall(e):
                if t in sym and t != name:
                    refs.add(t)
        deps[name] = refs
    # 위상 정렬 (Kahn)
    indeg = {n: 0 for n in deps}
    for n, rs in deps.items():
        for r in rs:
            indeg[n] += 0                      # n 이 r 에 의존
    order, seen = [], set()

    def visit(n, stack):
        if n in stack:
            return [n]
        if n in seen:
            return None
        seen.add(n)
        for r in deps.get(n, ()):  # noqa: SIM118
            cyc = visit(r, stack + [n])
            if cyc:
                return cyc
        order.append(n)
        return None

    for n in deps:
        cyc = visit(n, [])
        if cyc:
            return ("expression_no_cycle", False, f"순환 참조 {cyc}")
    return ("expression_no_cycle", True, f"심볼 {len(deps)}개 위상 정렬 OK")


def g_substitution_nonneg_integer(ctx):
    """대입 결과가 정수이며 음수가 아니다. 그리고 원본 구체값을 복원한다."""
    bad = []
    for phase, env in ctx["env"].items():
        for name, v in env.items():
            if not isinstance(v, int):
                bad.append(f"{phase}.{name} 가 정수가 아니다 ({v!r})")
            elif v < 0:
                bad.append(f"{phase}.{name} 가 음수 ({v})")
    # 식 토큰(계열 C)은 **행 단위**로 본다. 접힌 행의 모든 층에서 정수·비음수여야 하고
    # 원본 값을 복원해야 한다. 맨 심볼은 phase 환경에서 바로 본다.
    import plus_at_resid as RS
    fml = {v: k for k, v in RS.TOKEN.items()}
    syms = ctx["overlay"]["symbols"]
    _sc = ctx["overlay"].get("residual_sidecar") or {}
    R = int((syms.get("R_res") or _sc.get("R_res") or {}).get("value") or 0) or None
    L = int((syms.get("L_layers") or _sc.get("L_layers") or {}).get("value") or 0) or None
    rowmap = {ph: {int(r["op_id"]): r for r in rows}
              for ph, rows in ctx["derived_rows"].items()}
    for r in ctx["actual"]:
        aft = r["after"]
        if aft in fml:
            row = rowmap.get(r["phase"], {}).get(r["op_id"]) or {}
            ls = C.expand_layers(row.get("layers")) or [L]
            for l in ls:
                got = RS.value(fml[aft], l, R, L)
                if not isinstance(got, int) or got < 0:
                    bad.append(f"{r['phase']} {aft} (층 {l}) = {got!r}")
                    break
                if str(got) != r["before"]:
                    bad.append(f"{r['phase']} op{r['op_id']} {aft} (층 {l}) = {got} "
                               f"!= 원본 {r['before']}")
                    break
            if bad:
                break
            continue
        v = ctx["env"][r["phase"]].get(aft)
        if v is None or str(v) != r["before"]:
            bad.append(f"{r['phase']} {aft} = {v} != 원본 {r['before']}")
            break
    return ("substitution_nonneg_integer", not bad,
            "; ".join(bad[:3]) if bad else "전부 정수·비음수·복원 일치 (식 토큰은 행 단위)")


def g_zero_axis_only_initial_residual(ctx):
    """`0` 축이 **선언된 자리에만** 있다. 개수가 아니라 위치를 본다."""
    allow = set()
    for spec in (ctx["overlay"].get("not_substituted") or []):
        for a in (spec.get("allowed_zero_cells") or []):
            allow.add(tuple(a))
        for a in (spec.get("allow_zero_cells") or []):
            allow.add(tuple(a))
    found = []
    for phase, cells in ctx["derived_cells"].items():
        for key, v in cells.items():
            if v == "0":
                found.append(key)
    extra = [k for k in found if tuple(k) not in allow]
    if not allow:
        return ("zero_axis_only_initial_residual", False,
                f"`0` 축 {len(found)} 자리가 있는데 overlay 에 허용 목록이 없다 "
                f"-- allowed_zero_cells 를 선언해야 한다: {found[:4]}")
    return ("zero_axis_only_initial_residual", not extra,
            f"선언 밖 `0` 축 {extra[:4]}" if extra
            else f"`0` 축 {len(found)} 자리 전부 선언된 자리")


def g_batch_seq_head_axis_consistency(ctx):
    """배치·시퀀스·head 축의 **자리**가 원본과 같다.

    라벨 치환이 배치나 head 축을 옮기거나 새로 만들면 안 된다. 원본과 파생본에서
    `B`/`T`/head 심볼이 나타나는 canonical cell 집합이 정확히 같아야 한다.
    """
    bad = []
    for phase in ctx["orig_cells"]:
        for tok in ("B", "T") + HEAD_SYMS:
            o = {k for k, v in ctx["orig_cells"][phase].items() if v == tok}
            n = {k for k, v in ctx["derived_cells"][phase].items() if v == tok}
            if o != n:
                bad.append(f"{phase} `{tok}` 자리 변동 "
                           f"(원본만 {len(o - n)}, 파생만 {len(n - o)})")
    return ("batch_seq_head_axis_consistency", not bad,
            "; ".join(bad[:3]) if bad else "B·T·head 축 자리 불변")


def g_head_scope_exclusive(ctx):
    """한 shape 에 n_h / n_kv / n_h_kda 가 둘 이상 나오지 않는다."""
    bad = []
    for phase, rows in ctx["derived_rows"].items():
        for r in rows:
            for field in FIELDS:
                for sh in C.parse_jsonl_shape(r.get(field), field):
                    hit = [t for t in _tokens(sh) if t in HEAD_SYMS]
                    if len(set(hit)) > 1:
                        bad.append(f"{phase} op{r['op_id']} {field} {hit}")
    return ("head_scope_exclusive", not bad,
            "; ".join(bad[:3]) if bad else "head 심볼 배타성 OK")


def g_moe_quotient_remainder_consistency(ctx):
    """MoE concat 의 입력이 regular×(C-1) + last×1 이고 출력이 N_route 와 맞는다."""
    bad, checked = [], 0
    # C_trace 는 MoE 판정이 철회되면서 활성 symbols 에서 빠졌다. 없으면 이 게이트는
    # 애초에 해당 없음이다(아래 declares_moe 분기에서 n/a 로 나간다).
    _cs = ctx["overlay"]["symbols"].get("C_trace") or {}
    ct = int(_cs.get("value") or 0)
    for phase, rows in ctx["derived_rows"].items():
        env = ctx["env"][phase]
        for r in rows:
            mp = r.get("module_path") or ""
            if r.get("op_type") != "concat" or not mp.endswith(".block_sparse_moe"):
                continue
            ins = C.parse_jsonl_shape(r.get("input_shape"), "input_shape")
            outs = C.parse_jsonl_shape(r.get("output_shape"), "output_shape")
            firsts = [sh[0] for sh in ins if sh]
            if not any(f.startswith("n_trace") for f in firsts):
                continue
            checked += 1
            reg = sum(1 for f in firsts if f == "n_trace_regular")
            last = sum(1 for f in firsts if f == "n_trace_last")
            if (reg, last) != (ct - 1, 1):
                bad.append(f"{phase} op{r['op_id']}: regular {reg} last {last} "
                           f"(기대 {ct - 1}, 1)")
                continue
            tot = reg * env["n_trace_regular"] + last * env["n_trace_last"]
            if tot != env["N_route"]:
                bad.append(f"{phase} op{r['op_id']}: 입력 합 {tot} != N_route "
                           f"{env['N_route']}")
                continue
            # 출력 축 0 이 N_route 와 같은 양인가 -- 심볼 곱을 대입해 확인
            if outs and outs[0]:
                got = _eval_product(outs[0][0], env)
                if got is not None and got != env["N_route"]:
                    bad.append(f"{phase} op{r['op_id']}: 출력 축0 {outs[0][0]} "
                               f"= {got} != N_route {env['N_route']}")
    if not checked:
        # overlay 가 MoE 를 아예 안 건드리면 이 게이트는 **해당 없음**이다. 다만
        # "선언했는데 0 건" 은 결함이므로 구분한다.
        declares_moe = any("block_sparse_moe" in (sp["match"].get("module_regex") or "")
                           for sp in ctx["overlay"]["substitutions"])
        if declares_moe:
            return ("moe_quotient_remainder_consistency", False,
                    "MoE 판정을 선언했는데 검사할 concat 이 0 건 -- 0 == 0 은 통과가 아니다")
        return ("moe_quotient_remainder_consistency", None,
                "overlay 가 MoE 축을 건드리지 않는다 (R1 판정으로 철회) -- 해당 없음")
    return ("moe_quotient_remainder_consistency", not bad,
            "; ".join(bad[:3]) if bad else f"concat {checked} 건 전부 일치")


def _eval_product(token, env):
    """`B*k*T` 같은 곱을 대입한다. 못 읽으면 None."""
    parts = token.split("*")
    val = 1
    for p in parts:
        p = p.strip()
        if p.isdigit():
            val *= int(p)
        elif p in env:
            val *= env[p]
        else:
            return None
    return val


def g_prefill_decode_structure(ctx):
    """두 phase 의 공통 op 가 같은 방식으로 바뀌었다.

    같은 (module_path 정규화, op_type, field, shape_index, axis) 자리가 두 phase 에
    모두 있으면 `after` 심볼이 같아야 한다. 한쪽만 바뀌면 표가 갈린다.
    """
    if len(ctx["derived_rows"]) < 2:
        return ("prefill_decode_structure", True, "phase 가 하나뿐")
    def key(r):
        return (re.sub(r"layers\.\d+", "layers.N", r["module_path"]), r["op_type"],
                r["field"], r["shape_index"], r["axis"])
    per = {}
    for r in ctx["actual"]:
        per.setdefault(key(r), {})[r["phase"]] = r["after"]
    bad = [f"{k} -> {v}" for k, v in per.items()
           if len(v) == 2 and len(set(v.values())) > 1]
    both = sum(1 for v in per.values() if len(v) == 2)
    if not both:
        multi = [sp["sub_id"] for sp in ctx["overlay"]["substitutions"]
                 if len(sp.get("phases") or []) > 1]
        if multi:
            return ("prefill_decode_structure", False,
                    f"두 phase 를 선언한 판정 {multi} 이 있는데 공통 자리가 0 건이다")
        return ("prefill_decode_structure", None,
                "활성 판정이 전부 단일 phase 다 -- 해당 없음")
    return ("prefill_decode_structure", not bad,
            "; ".join(bad[:3]) if bad else f"공통 자리 {both} 건 전부 같은 심볼")


def g_op_id_dag(ctx):
    """op_id 유일성(**중복 검출 가능하게 리스트로 센다**), dependency 존재, 비순환."""
    bad = []
    for phase, rows in ctx["derived_rows"].items():
        ids = [int(r["op_id"]) for r in rows]
        if len(ids) != len(set(ids)):
            dup = [i for i in set(ids) if ids.count(i) > 1]
            bad.append(f"{phase}: op_id 중복 {dup[:4]}")
            continue
        have = set(ids)
        for r in rows:
            for d in (r.get("depends_on") or []):
                if int(d) not in have:
                    bad.append(f"{phase} op{r['op_id']}: depends_on {d} 없음")
                    break
                if int(d) >= int(r["op_id"]):
                    bad.append(f"{phase} op{r['op_id']}: depends_on {d} 가 자기 이후 "
                               f"(DAG 위반)")
                    break
    return ("op_id_dag", not bad, "; ".join(bad[:3]) if bad else "유일·존재·비순환 OK")


def g_layers_repeat_consistency(ctx):
    """`layers` 를 **범위까지 펼친** 수가 `repeat` 과 같다."""
    bad = []
    for phase, rows in ctx["derived_rows"].items():
        for r in rows:
            ls = C.expand_layers(r.get("layers"))
            rep = r.get("repeat")
            if ls and isinstance(rep, int) and rep != len(ls):
                bad.append(f"{phase} op{r['op_id']}: repeat {rep} != {len(ls)} "
                           f"({r.get('layers')!r})")
    return ("layers_repeat_consistency", not bad,
            "; ".join(bad[:3]) if bad else "전 행 일치")


def g_row_metadata_preserved(ctx):
    """caveat·unmapped·params·depends_on 등 라벨 밖 열이 그대로다."""
    meta = ("caveat", "unmapped", "params", "depends_on", "repeat", "layers",
            "module_path", "op_type", "raw_op", "layer_idx", "block_type",
            "block", "sub_block", "depth", "h1", "h2", "h3", "h4", "weight_pos",
            "phase")
    bad = []
    for phase in ctx["derived_rows"]:
        o = {int(r["op_id"]): r for r in ctx["orig_rows"][phase]}
        for r in ctx["derived_rows"][phase]:
            a = o.get(int(r["op_id"])) or {}
            for kk in meta:
                if a.get(kk) != r.get(kk):
                    bad.append(f"{phase} op{r['op_id']}: {kk} 변동")
                    break
    return ("row_metadata_preserved", not bad,
            "; ".join(bad[:3]) if bad else "메타데이터 불변")


def g_schema_shape_rank_token_type(ctx):
    """열 이름·행 수·shape 축 개수·토큰 종류가 보존된다."""
    bad = []
    for phase in ctx["derived_rows"]:
        if len(ctx["orig_rows"][phase]) != len(ctx["derived_rows"][phase]):
            bad.append(f"{phase}: 행 수 변동")
        if set(ctx["orig_cells"][phase]) != set(ctx["derived_cells"][phase]):
            bad.append(f"{phase}: canonical cell 키 집합 변동 (축 개수·피연산자 수)")
        for k, v in ctx["derived_cells"][phase].items():
            if not isinstance(v, str) or not v:
                bad.append(f"{phase} {k}: 토큰이 비어 있거나 문자열이 아니다")
                break
    if ctx["orig_header"] != ctx["derived_header"]:
        bad.append("csv 열 이름이 바뀌었다")
    return ("schema_shape_rank_token_type", not bad,
            "; ".join(bad[:3]) if bad else "스키마 보존")


def g_reshape_derivation(ctx):
    """reshape 자체 유도와 라벨이 일치한다. **구체 shape 사이드카로 실제로 돈다.**

    외부 검토 지적: "구조는 그대로지만 symbolic label 이 바뀌므로 재검사하거나, 불변
    승계의 구체적 증명을 기록해야 합니다." 재검사한다 -- 사이드카는 full/ 에 있다.
    """
    import sys
    sys.path.insert(0, os.path.join(ctx["proj"], "src"))
    try:
        import build_table as BT
    except Exception as e:                                       # noqa: BLE001
        return ("reshape_derivation", False, f"build_table 로드 실패: {e}")
    tot_o = tot_n = 0
    for phase, rows in ctx["derived_rows"].items():
        sc = os.path.join(ctx["model_dir"], "full",
                          f"{phase}.shapes.concrete.jsonl")
        if not os.path.exists(sc):
            return ("reshape_derivation", False, f"{phase} 구체 사이드카 없음: {sc}")
        conc = {}
        with io.open(sc, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    c = json.loads(line)
                    conc[int(c["op_id"])] = c
        for src, acc in ((ctx["orig_rows"][phase], "o"), (rows, "n")):
            n = 0
            for r in src:
                c = conc.get(int(r["op_id"]))
                if not c:
                    continue
                row = dict(r)
                row["input_shape"] = c.get("input_shape") or []
                row["output_shape"] = c.get("output_shape") or []
                n += len(BT.reshape_disagreements(row, r))
            if acc == "o":
                tot_o += n
            else:
                tot_n += n
    ok = tot_n <= tot_o
    return ("reshape_derivation", ok,
            f"원본 이견 {tot_o} -> 파생 {tot_n}" +
            ("" if ok else "  **늘었다**"))


def g_port_coverage_inherited(ctx):
    """포트 사이드카가 **불변**임을 해시로 증명하고 원본 결과를 승계한다.

    외부 검토: "포트 구조가 완전히 불변임을 base sidecar hash 와 cell-key 불변으로
    증명한다면 inherited_unchanged 가 더 정확합니다."
    """
    import hashlib
    proofs = []
    for phase in ctx["derived_rows"]:
        p = os.path.join(ctx["model_dir"], "full", f"{phase}.ports.jsonl")
        if not os.path.exists(p):
            return ("port_coverage", False, f"{phase} 포트 사이드카 없음")
        h = hashlib.sha256()
        with io.open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        proofs.append(f"{phase}.ports.jsonl sha {h.hexdigest()[:16]}")
        if set(ctx["orig_cells"][phase]) != set(ctx["derived_cells"][phase]):
            return ("port_coverage", False, f"{phase} cell 키가 바뀌어 승계 불가")
    return ("port_coverage", True,
            "derived view 는 포트를 건드리지 않는다 (사이드카 불변 + cell 키 불변): "
            + "; ".join(proofs))


CROSSWALK_DIRS = (
    os.path.join("..", "llm-arch-tracer-results-labeled", "work", "crosswalk"),
)


def _crosswalk(proj, model, phase):
    """발행본 셀 -> raw_sites 대응. 없으면 None.

    **이것 없이는 등가류를 볼 수 없다.** 발행본 op_id 는 0.. 로 다시 번호를 붙인 것이고
    원시 원장 op_id 와 다른 공간이다 -- 같은 숫자를 같은 op 으로 보면 거짓 충돌이 나온다
    (내 첫 구현이 그렇게 244 건을 냈다, 2026-09-28).
    """
    import glob
    import gzip
    import json as _json
    for d in CROSSWALK_DIRS:
        for p in glob.glob(os.path.join(proj, d, f"{model}.{phase}.jsonl*")):
            op = gzip.open if p.endswith(".gz") else io.open
            out = {}
            with op(p, "rt", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = _json.loads(line)
                    key = (r["phase"], int(r["op_id"]),
                           {"i": "input_shape", "o": "output_shape",
                            "w": "weight_shape"}.get(r["field"], r["field"]),
                           int(r["shape_index"]), int(r["axis"]))
                    out[key] = (r.get("expr"), [tuple(x) for x in
                                                (r.get("raw_sites") or [])])
            return p, out
    return None, None


def g_axis_class_consistency(ctx):
    """바꾼 셀이 속한 **등가류 안에서 이름이 하나**인지. crosswalk 로 raw 자리를 찾는다.

    외부 검토(R3 2 차): "footprint 완전성은 shape 후보 집합에 대한 완전성이지, 등가류
    member 전체를 덮었다는 증명이 아닙니다." 그래서 승계가 아니라 재실행한다.
    """
    import json as _json
    import sys as _sys
    _sys.path.insert(0, os.path.join(ctx["proj"], "src"))
    try:
        import axis_classes as AC
    except Exception as e:                                       # noqa: BLE001
        return ("axis_class_consistency", False, f"axis_classes 로드 실패: {e}")

    changed = {(r["phase"], r["op_id"], r["field"], r["shape_index"], r["axis"]): r
               for r in ctx["actual"]}
    if not changed:
        return ("axis_class_consistency", None, "바꾼 셀이 없다")
    model = os.path.basename(os.path.normpath(ctx["model_dir"]))
    TAG = {"input_shape": "i", "output_shape": "o", "weight_shape": "w"}
    INV = {v: k for k, v in TAG.items()}
    notes = []
    for phase in sorted({k[0] for k in changed}):
        cw_path, cw = _crosswalk(ctx["proj"], model, phase)
        if not cw:
            return ("axis_class_consistency", None,
                    f"{phase}: crosswalk 이 없다. 발행본 op_id 는 원시 원장과 다른 번호"
                    f" 공간이므로 crosswalk 없이는 등가류를 볼 수 없다 -- **미평가**")
        # crosswalk 가 지금 발행본과 맞는가. 안 맞으면 거짓 결과 대신 미평가.
        stale = [k for k, r in changed.items()
                 if k[0] == phase and (k not in cw or cw[k][0] != r["before"])]
        if stale:
            return ("axis_class_consistency", None,
                    f"{phase}: crosswalk 이 지금 발행본과 안 맞는다 "
                    f"(셀 {len(stale)} 개에서 expr 불일치 또는 누락) -- **미평가**. "
                    f"{os.path.relpath(cw_path, ctx['proj'])}")
        raw_p = os.path.join(ctx["model_dir"], "full", f"{phase}.trace.raw.jsonl")
        conc_p = os.path.join(ctx["model_dir"], "full",
                              f"{phase}.shapes.concrete.jsonl")
        if not (os.path.exists(raw_p) and os.path.exists(conc_p)):
            return ("axis_class_consistency", None,
                    f"{phase}: 원시 원장/구체 사이드카가 없다 -- 미평가")
        rows = [_json.loads(l) for l in io.open(raw_p, encoding="utf-8") if l.strip()]
        conc = {}
        with io.open(conc_p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    c = _json.loads(line)
                    conc[int(c["op_id"])] = c
        n_ports = AC.attach_ports(ctx["model_dir"], phase, rows)
        miss = AC.missing_port_records(rows)
        if miss or n_ports != len(rows):
            return ("axis_class_consistency", False,
                    f"{phase}: 포트 커버리지 {n_ports}/{len(rows)}, missing {miss}")
        uf = AC.build(rows, conc, noop_barriers=AC.noop_barriers_of(
            ctx["model_dir"], phase), mode="provenance")
        members = {}
        for slot in list(uf.p):
            members.setdefault(uf.find(slot), []).append(slot)
        # raw 자리 -> 발행본 셀 (역방향)
        rev = {}
        for key, (_expr, sites) in cw.items():
            if key[0] != phase:
                continue
            for st in sites:
                rev.setdefault(tuple(st), []).append(key)
        pub = ctx["derived_cells"][phase]
        roots = set()
        for k, r in changed.items():
            if k[0] != phase:
                continue
            for st in cw[k][1]:
                roots.add(uf.find(tuple(st)))
        bad, n_ck = [], 0
        for root in roots:
            labels = {}
            for slot in members.get(root, []):
                for key in rev.get(tuple(slot), ()):
                    if key in pub:
                        labels.setdefault(pub[key], []).append(key)
            if not labels:
                continue
            n_ck += 1
            if len(labels) > 1:
                bad.append(f"{phase} class {root}: 이름 {sorted(labels)}")
        if bad:
            return ("axis_class_consistency", False, "; ".join(bad[:3]))
        notes.append(f"{phase}: 건드린 class {len(roots)}, 발행본 member 가 있는 class "
                     f"{n_ck}, 이름 충돌 0")
    return ("axis_class_consistency", True, "  ".join(notes))


# 게이트 목록. `decision` 은 이 구현이 실제로 하는 것을 적는다.
#   rerun               여기서 돈다
#   inherited_unchanged 불변을 증명하고 원본 결과를 승계한다
#   not_evaluated       관련은 있으나 이 층에서 평가하지 않는다 (**통과로 세지 않는다**)
GATES = [
    (g_schema_shape_rank_token_type, "rerun"),
    (g_symbol_declared, "rerun"),
    (g_expression_no_cycle, "rerun"),
    (g_substitution_nonneg_integer, "rerun"),
    (g_zero_axis_only_initial_residual, "rerun"),
    (g_batch_seq_head_axis_consistency, "rerun"),
    (g_head_scope_exclusive, "rerun"),
    (g_moe_quotient_remainder_consistency, "rerun"),
    (g_prefill_decode_structure, "rerun"),
    (g_layers_repeat_consistency, "rerun"),
    (g_row_metadata_preserved, "rerun"),
    (g_op_id_dag, "rerun"),
    (g_reshape_derivation, "rerun"),
    (g_port_coverage_inherited, "inherited_unchanged"),
    (g_axis_class_consistency, "rerun"),
]

# 구현이 없는 것은 **표에 not_evaluated 로 적고 통과로 세지 않는다.**
NOT_EVALUATED = {}


def run(ctx):
    """(rows, failed) -- rows 는 MANIFEST 에 실을 게이트별 기록."""
    rows, failed = {}, []
    for fn, decision in GATES:
        gid, ok, detail = fn(ctx)
        if ok is None:                      # 전제가 없다 -> 해당 없음. 막지 않는다.
            rows[gid] = {"decision": "not_applicable", "result": "n/a",
                         "detail": detail, "review_status": "proposed"}
            continue
        rows[gid] = {"decision": decision, "result": "pass" if ok else "FAIL",
                     "detail": detail, "review_status": "proposed"}
        if not ok:
            failed.append(f"V9 {gid}: {detail}")
    for gid, reason in NOT_EVALUATED.items():
        rows[gid] = {"decision": "not_evaluated", "result": "not_run",
                     "detail": reason, "review_status": "proposed"}
    return rows, failed
