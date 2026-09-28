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
import math
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
                # **self-edge 를 지우지 않는다.** `x: {expr: x}` 가 통과했다
                # (외부 검토 R3d). DFS 의 stack 검사가 직접 순환도 잡는다.
                if t in sym:
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
    if not allow:
        return ("zero_axis_only_initial_residual", False,
                f"`0` 축 {len(found)} 자리가 있는데 overlay 에 허용 목록이 없다 "
                f"-- allowed_zero_cells 를 선언해야 한다: {found[:4]}")
    # **양방향으로 맞춘다.** found ⊆ allow 만 보면 허용 자리를 선언했는데 실제 `0` 이
    # 0 개여도 통과한다 -- 선언이 산출물과 어긋난 것이고 통과가 아니다 (외부 검토 R3d).
    extra = sorted(k for k in found if tuple(k) not in allow)
    gone = sorted(a for a in allow if a not in {tuple(k) for k in found})
    if extra or gone:
        return ("zero_axis_only_initial_residual", False,
                f"선언 밖 `0` 축 {extra[:4]} / 선언했는데 없는 자리 {gone[:4]}")
    return ("zero_axis_only_initial_residual", True,
            f"`0` 축 {len(found)} 자리 == 선언 {len(allow)} 자리 (양방향 일치)")


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


def _bad_slots(cw, changed, phase, raw_by, conc_by):
    """crosswalk 이 가리킨 raw 자리가 원장에 **실제로 있는가**. 두 게이트가 공유한다.

    외부 검토(R3d): axis 에만 범위 검사가 생기고 reshape 에는 없었다 -- reshape 는 축이
    범위 밖이면 그 변경을 적용하지 않고 조용히 넘어가, 라벨을 안 바꾼 상태로 "이견 0" 을
    냈다. 같은 검사를 한 군데 두고 둘이 같이 쓴다.
    """
    INV = {"i": "input_shape", "o": "output_shape", "w": "weight_shape"}
    bad = []
    for k, r in changed.items():
        if k[0] != phase:
            continue
        for st in cw[k][1]:
            roid, tag, rsi, rax = int(st[0]), st[1], int(st[2]), int(st[3])
            if roid not in raw_by or roid not in conc_by:
                bad.append((roid, "op 없음"))
                continue
            sh = conc_by[roid].get(INV.get(tag, tag))
            if tag == "w":
                ok = bool(sh) and rax < len(sh)
            else:
                ok = bool(sh) and rsi < len(sh) and isinstance(sh[rsi], list) \
                    and rax < len(sh[rsi])
            if not ok:
                bad.append((roid, f"{tag}[{rsi}]ax{rax} 범위 밖"))
    return bad


def g_reshape_derivation(ctx):
    """reshape 자체 유도와 라벨이 일치한다. **crosswalk 로 raw 자리를 찾아** 돈다.

    앞선 구현은 발행본 행의 op_id 로 `shapes.concrete.jsonl`(원시 번호 공간)을 조회했다.
    두 공간이 다르므로 그 "원본 0 -> 파생 0" 은 증명이 아니었다 -- 외부 검토(R3b)가
    axis 게이트와 **같은 착오**가 여기 남아 있다고 짚었다.

    지금은 crosswalk 로 발행본 셀 -> raw 자리를 얻어, 바뀐 셀이 걸린 raw op 만 골라 그 op
    의 reshape 유도를 원본 라벨과 파생 라벨 각각으로 검사한다. crosswalk 이 없거나 낡으면
    **not_evaluated** 로 빠진다(N/A 가 아니다 -- release 를 막는다).
    """
    import sys as _sys
    _sys.path.insert(0, os.path.join(ctx["proj"], "src"))
    try:
        import build_table as BT
    except Exception as e:                                       # noqa: BLE001
        return ("reshape_derivation", False, f"build_table 로드 실패: {e}")
    model = os.path.basename(os.path.normpath(ctx["model_dir"]))
    changed = {(r["phase"], r["op_id"], r["field"], r["shape_index"], r["axis"]): r
               for r in ctx["actual"]}
    INV = {"i": "input_shape", "o": "output_shape", "w": "weight_shape"}
    notes = []
    for phase in sorted(ctx["derived_rows"]):
        cw_path, cw = _crosswalk(ctx["proj"], model, phase)
        if not cw:
            return ("reshape_derivation", "not_evaluated",
                    f"{phase}: crosswalk 이 없다 -- 발행본 op_id 로 원시 사이드카를 "
                    f"조회하면 다른 op 을 본다. 평가하지 않는다")
        stale = [k for k, r in changed.items()
                 if k[0] == phase and (k not in cw or cw[k][0] != r["before"]
                                       or not cw[k][1])]   # 빈 raw_sites 도 거부
        if stale:
            return ("reshape_derivation", "not_evaluated",
                    f"{phase}: crosswalk 이 지금 발행본과 안 맞는다 (셀 {len(stale)})")
        # **crosswalk 이 발행본 전체를 덮는가.** 바뀐 셀만 맞춰 보면 "그 셀만 실린 낡은
        # crosswalk" 도 통과한다. 두 방향으로 집합을 맞춘다 -- 이것이 '번호 공간을
        # 제대로 건넜다' 의 실제 증거다(외부 검토 R3c).
        _cwk = {k for k in cw if k[0] == phase}
        _pubk = set(ctx["derived_cells"][phase])
        if _cwk != _pubk:
            return ("reshape_derivation", "not_evaluated",
                    f"{phase}: crosswalk 이 발행본과 다른 셀 집합이다 "
                    f"(crosswalk 만 {len(_cwk - _pubk)}, 발행본만 {len(_pubk - _cwk)})"
                    f" -- 낡았다")
        sc = os.path.join(ctx["model_dir"], "full", f"{phase}.shapes.concrete.jsonl")
        raw_p = os.path.join(ctx["model_dir"], "full", f"{phase}.trace.raw.jsonl")
        if not (os.path.exists(sc) and os.path.exists(raw_p)):
            return ("reshape_derivation", "not_evaluated",
                    f"{phase}: 원시 원장 또는 구체 사이드카가 없다")
        conc = {}
        with io.open(sc, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    c = json.loads(line)
                    conc[int(c["op_id"])] = c
        raw = {}
        with io.open(raw_p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    raw[int(r["op_id"])] = r
        raw_ops, ren = set(), {}
        for k, rec in changed.items():
            if k[0] != phase:
                continue
            for site in cw[k][1]:
                roid = int(site[0])
                raw_ops.add(roid)
                ren.setdefault(roid, []).append(
                    (site[1], int(site[2]), int(site[3]), rec["after"]))
        tot_o = tot_n = 0
        # **건너뛰지 않는다.** op 이 없거나 축이 범위 밖이면 이 게이트는 그 자리를 못 본
        # 것이고, 못 본 것을 PASS 로 세면 안 된다 (외부 검토 R3c/R3d).
        _bs = _bad_slots(cw, changed, phase, raw, conc)
        if _bs:
            return ("reshape_derivation", "not_evaluated",
                    f"{phase}: crosswalk 이 가리킨 raw slot {len(_bs)} 개를 원장에서 "
                    f"찾을 수 없다 {_bs[:4]}")
        for roid in sorted(raw_ops):
            r = raw[roid]
            c = conc[roid]
            row = dict(r)
            row["input_shape"] = c.get("input_shape") or []
            row["output_shape"] = c.get("output_shape") or []
            tot_o += len(BT.reshape_disagreements(row, r))
            der = json.loads(json.dumps(r))
            for tag, rsi, rax, after in ren.get(roid, ()):
                sh = der.get(INV[tag])
                tgt = sh if tag == "w" else (sh[rsi] if sh and rsi < len(sh) else None)
                if isinstance(tgt, list) and rax < len(tgt):
                    tgt[rax] = after
            tot_n += len(BT.reshape_disagreements(row, der))
        if tot_n > tot_o:
            return ("reshape_derivation", False,
                    f"{phase}: 이견이 늘었다 {tot_o} -> {tot_n}")
        notes.append(f"{phase}: 건드린 raw op {len(raw_ops)}, 이견 {tot_o} -> {tot_n}")
    return ("reshape_derivation", True, "  ".join(notes))


def g_sidecar_expression_integrity(ctx):
    """사이드카 레코드의 식이 **overlay registry 와 같고, 식별자가 전부 선언됐고,
    평가값이 value 와 같은가**.

    외부 검토(R3d): `g_symbol_declared()` 는 ctx["actual"](본표 A 변경)만 본다. 그래서
    사이드카에 `mystery(l)` 같은 미등록 식을 넣어도 통과했다. 게다가 registry 가 두 군데
    있다 -- overlay 의 `residual_sidecar.formulas` 와 `plus_at_resid.TOKEN` 상수. 두 판이
    갈라져도 아무 검사가 안 잡았다. 여기서 **레코드의 expr 를 overlay registry 와** 맞춘다.
    """
    sc = ctx.get("sidecar_records")
    decl = ctx["overlay"].get("residual_sidecar") or {}
    if not decl:
        return ("sidecar_expression_integrity", None, "overlay 가 사이드카를 안 쓴다")
    if not sc:
        # 선언했는데 레코드가 없다. N/A 가 아니다.
        return ("sidecar_expression_integrity", "not_evaluated",
                "overlay 가 residual_sidecar 를 선언했는데 레코드가 없다")
    reg = decl.get("formulas") or {}
    if not reg:
        return ("sidecar_expression_integrity", False,
                "overlay 의 residual_sidecar.formulas 가 비었다")
    ns = {k for k in ("R_res", "L_layers") if k in decl}
    ns |= {"l"} if "l" in decl else set()
    funcs = {"ceil"}
    env = {"ceil": math.ceil,
           "R_res": (decl.get("R_res") or {}).get("value"),
           "L_layers": (decl.get("L_layers") or {}).get("value")}
    if env["R_res"] is None or env["L_layers"] is None:
        return ("sidecar_expression_integrity", "not_evaluated",
                "overlay 에 R_res / L_layers 값이 없다")
    ident = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
    bad, seen = [], {}
    n_declared = decl.get("cells")
    for r in sc:
        key = (r["phase"], r["op_id"], r["field"], r["shape_index"], r["axis"])
        if key in seen:
            bad.append(f"join key 중복 {key}")
            continue
        seen[key] = r
        f = r.get("formula")
        if f not in reg:
            bad.append(f"{key}: formula {f!r} 가 registry 에 없다")
            continue
        if r.get("expr") != reg[f]:
            bad.append(f"{key}: expr {r.get('expr')!r} != registry {reg[f]!r}")
            continue
        unk = sorted({t for t in ident.findall(r["expr"])} - ns - funcs)
        if unk:
            bad.append(f"{key}: 미선언 식별자 {unk}")
            continue
        try:
            got = eval(r["expr"], {"__builtins__": {}},            # noqa: S307
                       dict(env, l=r["layer_idx"]))
        except Exception as e:                                     # noqa: BLE001
            bad.append(f"{key}: 평가 실패 {e}")
            continue
        if got != r["value"]:
            bad.append(f"{key}: 식값 {got} != value {r['value']}")
    if bad:
        return ("sidecar_expression_integrity", False, "; ".join(bad[:3]))
    if n_declared is not None and n_declared != len(sc):
        return ("sidecar_expression_integrity", False,
                f"overlay 가 {n_declared} 셀이라 했는데 레코드 {len(sc)} 개다")
    return ("sidecar_expression_integrity", True,
            f"레코드 {len(sc)} 개: formula 전부 registry 일치, 식별자 {sorted(ns)} + "
            f"{sorted(funcs)} 로 전부 해석, 식값 == value, join key 유일")


def g_base_symbol_coverage(ctx):
    """본표의 **모든 식별자**가 authority + table_added_symbols 로 선언됐는가.

    외부 검토(R3d): bundle 계약의 `base_table_symbols: inherited_from_authority` 가
    dangling pointer 였다 -- 가리킨다고 한 `full/symbol_table.json` 은 없다. 실제
    authority 는 `full/provenance.json` 의 `symbol_table` 이고, 그것과 overlay 가 새로
    넣은 이름의 합집합이 본표를 덮는지 **여기서 센다**. 그러면 계약이 주장이 아니라
    검사된 사실이 된다.
    """
    prov = os.path.join(ctx["model_dir"], "full", "provenance.json")
    if not os.path.exists(prov):
        return ("base_symbol_coverage", "not_evaluated",
                "full/provenance.json 이 없다 -- base symbol authority 를 읽을 수 없다")
    st = (json.load(io.open(prov, encoding="utf-8")).get("symbol_table") or {})
    base = {k for k, v in st.items() if isinstance(v, int)}
    added = set(ctx["overlay"]["symbols"])
    ident = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
    unknown, notes = {}, []
    for phase, cells in ctx["derived_cells"].items():
        seen = set()
        for key, v in cells.items():
            for t in ident.findall(str(v)):
                seen.add(t)
                if t not in base and t not in added:
                    unknown.setdefault(t, key)
        notes.append(f"{phase}: 식별자 {len(seen)}")
    if unknown:
        return ("base_symbol_coverage", False,
                f"선언 밖 식별자 {sorted(unknown)[:6]} (예: {list(unknown.values())[:2]})")
    return ("base_symbol_coverage", True,
            f"authority {len(base)} + table_added {sorted(added)} 로 전부 해석됨.  "
            + "  ".join(notes))


def g_sidecar_phase_consistency(ctx):
    """사이드카의 prefill/decode 가 같은 구조인가 (외부 검토 Q3 승격).

    본표의 활성 판정이 prefill 전용이라 prefill_decode_structure 는 N/A 지만, bundle 에는
    양 phase 의 expressions.yaml 이 들어간다. 그건 따로 봐야 한다.
    """
    sc = ctx.get("sidecar_records")
    if sc is None:
        return ("sidecar_phase_consistency", None, "사이드카가 없다")
    import collections as _c
    per = _c.Counter(r["phase"] for r in sc)
    if len(per) < 2:
        return ("sidecar_phase_consistency", False,
                f"사이드카가 있는데 phase 가 하나뿐이다 ({dict(per)})")
    if len(set(per.values())) != 1:
        return ("sidecar_phase_consistency", False,
                f"phase 별 레코드 수가 다르다 {dict(per)}")
    # **구조 multiset 을 맞춘다.** 수와 식 분포만 보면 서로 다른 셀에 식이 재배치돼도
    # 통과한다 (외부 검토 R3d). op_id 만 제외한다 -- phase 마다 번호가 다르다.
    _K = ("field", "shape_index", "axis", "value", "stage", "formula", "expr",
          "layer_idx", "layers", "module_path", "op_type")
    _ms = {}
    for ph in per:
        _ms[ph] = _c.Counter(tuple(r.get(k) for k in _K)
                             for r in sc if r["phase"] == ph)
    _phs = sorted(_ms)
    _a, _b = _ms[_phs[0]], _ms[_phs[1]]
    if _a != _b:
        _d1, _d2 = list((_a - _b).items())[:2], list((_b - _a).items())[:2]
        return ("sidecar_phase_consistency", False,
                f"구조 multiset 이 다르다 -- {_phs[0]} 만 {_d1}, {_phs[1]} 만 {_d2}")
    dist = {}
    for ph in per:
        dist[ph] = dict(_c.Counter(r["formula"] for r in sc if r["phase"] == ph))
    vals = list(dist.values())
    if any(v != vals[0] for v in vals):
        return ("sidecar_phase_consistency", False, f"식 분포가 다르다 {dist}")
    return ("sidecar_phase_consistency", True,
            f"phase 별 {list(per.values())[0]} 레코드, 식 분포 동일 {vals[0]}")


def g_port_coverage_inherited(ctx):
    """포트 커버리지를 **이름대로** 검사한다. 증거를 둘로 나눠 적는다.

    외부 검토(R3c): "published cell-key 불변" 과 "raw ports coverage" 는 다른 증거다.
    앞선 구현은 앞의 것과 파일 해시만 봤다 -- 이름은 port_coverage 인데 포트를 안 셌다.

    지금은 둘 다 본다:
      raw 증거    raw op-id 집합 == ports op-id 집합, ports 중복 0, schema 단일
      발행본 증거 canonical cell 키 불변 (derived view 가 발행 구조를 안 바꿨다)
    """
    import hashlib
    raw_notes, pub_notes = [], []
    for phase in ctx["derived_rows"]:
        full = os.path.join(ctx["model_dir"], "full")
        pp = os.path.join(full, f"{phase}.ports.jsonl")
        rp = os.path.join(full, f"{phase}.trace.raw.jsonl")
        if not (os.path.exists(pp) and os.path.exists(rp)):
            return ("port_coverage", "not_evaluated",
                    f"{phase}: 포트 또는 원시 원장 사이드카가 없다")
        raw_ids, dup_raw = set(), 0
        with io.open(rp, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                oid = int(json.loads(line)["op_id"])
                if oid in raw_ids:
                    dup_raw += 1
                raw_ids.add(oid)
        port_ids, dup_port, schemas = set(), 0, set()
        h = hashlib.sha256()
        with io.open(pp, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        with io.open(pp, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                oid = int(rec["op_id"])
                if oid in port_ids:
                    dup_port += 1
                port_ids.add(oid)
                schemas.add(rec.get("ports_schema_version"))
        if dup_raw or dup_port:
            return ("port_coverage", False,
                    f"{phase}: 중복 op_id -- 원장 {dup_raw}, 포트 {dup_port}")
        if len(schemas) != 1:
            return ("port_coverage", False,
                    f"{phase}: 포트 schema 가 섞였다 {sorted(schemas)}")
        if raw_ids != port_ids:
            return ("port_coverage", False,
                    f"{phase}: raw op-id 집합 != ports op-id 집합 "
                    f"(원장만 {len(raw_ids - port_ids)}, 포트만 {len(port_ids - raw_ids)})")
        raw_notes.append(f"{phase}: {len(port_ids)}/{len(raw_ids)} schema "
                         f"v{sorted(schemas)[0]} sha {h.hexdigest()[:12]}")
        if set(ctx["orig_cells"][phase]) != set(ctx["derived_cells"][phase]):
            return ("port_coverage", False, f"{phase}: 발행본 cell 키가 바뀌었다")
        pub_notes.append(f"{phase}: cell 키 불변")
    return ("port_coverage", True,
            "raw 포트 커버리지 [" + "; ".join(raw_notes) + "]  "
            "발행본 구조 [" + "; ".join(pub_notes) + "]")


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
    # **결정론.** glob 첫 결과에 기대지 않고 정렬해 첫 것을 쓴다(외부 검토 R3b).
    cands = []
    for d in CROSSWALK_DIRS:
        cands += sorted(glob.glob(os.path.join(proj, d, f"{model}.{phase}.jsonl*")))
    for p in cands:
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
                # **중복 키를 거부한다.** 조용히 마지막 것을 쓰면 어느 판을 읽었는지 모른다.
                if key in out:
                    raise ValueError(f"crosswalk 중복 키: {key} ({p})")
                sites = [tuple(x) for x in (r.get("raw_sites") or [])]
                out[key] = (r.get("expr"), sites)
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
            return ("axis_class_consistency", "not_evaluated",
                    f"{phase}: crosswalk 이 없다. 발행본 op_id 는 원시 원장과 다른 번호"
                    f" 공간이므로 crosswalk 없이는 등가류를 볼 수 없다 -- **미평가**")
        # crosswalk 가 지금 발행본과 맞는가. 안 맞으면 거짓 결과 대신 미평가.
        stale = [k for k, r in changed.items()
                 if k[0] == phase and (k not in cw or cw[k][0] != r["before"]
                                       or not cw[k][1])]   # 빈 raw_sites 도 거부
        if stale:
            return ("axis_class_consistency", "not_evaluated",
                    f"{phase}: crosswalk 이 지금 발행본과 안 맞는다 "
                    f"(셀 {len(stale)} 개에서 expr 불일치 또는 누락) -- **미평가**. "
                    f"{os.path.relpath(cw_path, ctx['proj'])}")
        # **crosswalk 이 발행본 전체를 덮는가.** 바뀐 셀만 맞춰 보면 "그 셀만 실린 낡은
        # crosswalk" 도 통과한다. 두 방향으로 집합을 맞춘다 -- 이것이 '번호 공간을
        # 제대로 건넜다' 의 실제 증거다(외부 검토 R3c).
        _cwk = {k for k in cw if k[0] == phase}
        _pubk = set(ctx["derived_cells"][phase])
        if _cwk != _pubk:
            return ("axis_class_consistency", "not_evaluated",
                    f"{phase}: crosswalk 이 발행본과 다른 셀 집합이다 "
                    f"(crosswalk 만 {len(_cwk - _pubk)}, 발행본만 {len(_pubk - _cwk)})"
                    f" -- 낡았다")
        raw_p = os.path.join(ctx["model_dir"], "full", f"{phase}.trace.raw.jsonl")
        conc_p = os.path.join(ctx["model_dir"], "full",
                              f"{phase}.shapes.concrete.jsonl")
        if not (os.path.exists(raw_p) and os.path.exists(conc_p)):
            return ("axis_class_consistency", "not_evaluated",
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
        # **존재하지 않는 raw slot 을 먼저 거른다.** uf.find 는 모르는 키를 singleton 으로
        # 넣어 버리므로, 없는 자리를 넣으면 member 가 없는 class 가 생겨 조용히 건너뛴다.
        # reshape 게이트와 같은 helper 를 쓴다 (외부 검토 R3d).
        bad_sites = _bad_slots(cw, changed, phase,
                               {int(r["op_id"]): r for r in rows}, conc)
        if bad_sites:
            return ("axis_class_consistency", "not_evaluated",
                    f"{phase}: crosswalk 이 가리킨 raw slot {len(bad_sites)} 개를 "
                    f"원장에서 찾을 수 없다 {bad_sites[:4]}")
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
        # **모든 class 에 발행본 member 가 있어야 한다** (외부 검토 R3c 요구).
        # 다만 솔직히 적는다 -- rev 를 같은 crosswalk 에서 만들므로 roots 의 원소는 항상
        # 발행본 키로 되돌아온다. 즉 이 등식은 지금 구조에서 깨지지 않는 **항등식**이고,
        # 음성 대조로도 발화시키지 못했다(develop/plus_at_negctl.py). 그래서 이 줄은
        # 보험이고, "class 를 조용히 건너뛰지 않았다" 의 실제 증거는 위의 raw slot 존재
        # 검사와 crosswalk 커버리지 검사다.
        if n_ck != len(roots):
            return ("axis_class_consistency", False,
                    f"{phase}: class {len(roots)} 중 발행본 member 가 있는 것이 {n_ck} "
                    f"뿐이다 -- 나머지는 검사되지 않았다")
        notes.append(f"{phase}: 건드린 class {len(roots)} == 검사한 class {n_ck}, "
                     f"이름 충돌 0")
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
    (g_port_coverage_inherited, "rerun"),
    (g_axis_class_consistency, "rerun"),
    (g_sidecar_phase_consistency, "rerun"),
    (g_sidecar_expression_integrity, "rerun"),
    (g_base_symbol_coverage, "rerun"),
]

# 구현이 없는 것은 **표에 not_evaluated 로 적고 통과로 세지 않는다.**
NOT_EVALUATED = {}


def run(ctx):
    """(rows, failed) -- rows 는 MANIFEST 에 실을 게이트별 기록."""
    rows, failed = {}, []
    for fn, decision in GATES:
        gid, ok, detail = fn(ctx)
        if ok == "not_evaluated":
            # **필수 증거가 없어 평가하지 못한 것.** N/A 가 아니다 -- release 를
            # 막는다 (외부 검토 R3b 차단 2).
            rows[gid] = {"decision": "not_evaluated", "result": "not_run",
                         "detail": detail, "review_status": "proposed"}
            continue
        if ok is None:                      # overlay 가 그 대상을 **선언하지 않았다**
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
