"""단일 검증 하네스 — rules/src를 바꾼 뒤 **반드시** 이걸 돌린다.

왜 있는가 (2026-07-27, 실제로 당한 사고들):
  - PowerShell 정규식으로 rules/symbols.yaml을 고쳤다가 심볼명이 깨지고 `scope:` 키가
    중복됐다. yaml.safe_load는 중복 키를 조용히 마지막 것으로 덮어써서 안 잡혔다.
  - docstring 편집이 summarize.py를 SyntaxError로 만들었는데, 재생성을 돌리기 전까지 몰랐다.
  - 심볼라이저에서 "scope 밖 심볼은 버린다"로 바꿨더니 정상 심볼 88,000개가 bare 정수로
    퇴행했다. 우연히 감사 스크립트를 다시 돌려서 발견했다.
  - 감사 스크립트 자체가 틀려서(수식 `2*d_head` 안의 계수 2까지 셈) 잘못된 결론을 보고했다.
공통 원인은 하나다: **"고쳤다"는 주장에 자동 확인이 없었다.** 그래서 주장 대신 종료 코드로
말하게 만든다.

돌리는 법:
    .venv\\Scripts\\python.exe develop\\verify_all.py              # 검사 + 베이스라인 대조
    .venv\\Scripts\\python.exe develop\\verify_all.py --update-baseline   # 검토 후 기준 갱신

검사 항목:
  STATIC   rules/*.yaml 파싱 + 중복 키, src/*.py import, symbols/derived_dims 스키마
  FLEET    모델별 C체크 FAIL, C17, 미해결 유도상수, bare 정수, 미확인 심볼
  EXTERNAL KV cache 카드 vs 공개 갤러리 수치 (develop/verify/references.yaml)
  BASELINE 이전 기준 대비 **퇴행**이면 실패 (개선은 통과시키되 갱신하라고 알림)
"""
import argparse
import importlib
import json
import os
import re
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJ, "src"))

MODELS = os.path.join(PROJ, "models")
REFS = os.path.join(HERE, "verify", "references.yaml")
BASELINE = os.path.join(HERE, "verify", "baseline.json")

# Windows consoles default to cp949 here, which cannot encode the em-dashes/arrows used in the
# Korean messages below. Without this the harness CRASHED with UnicodeEncodeError at the very
# moment it tried to report a FAIL -- i.e. it was silently unable to report the one thing it
# exists for (hit 2026-07-29 when gpt2-xl first went C17=WARN). Force UTF-8 and never let an
# encoding problem swallow a finding.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

failures: list[str] = []
warnings: list[str] = []


def fail(msg):
    failures.append(msg)
    print(f"   FAIL  {msg}")


def warn(msg):
    warnings.append(msg)
    print(f"   WARN  {msg}")


# ---------------------------------------------------------------- STATIC
class _DupCheckLoader(yaml.SafeLoader):
    """yaml.safe_load silently keeps the LAST of duplicate keys -- that is exactly how a
    corrupted symbols.yaml passed unnoticed. Refuse duplicates instead."""


def _no_dup(loader, node, deep=False):
    out = {}
    for k_node, v_node in node.value:
        key = loader.construct_object(k_node, deep=deep)
        if key in out:
            raise ValueError(f"duplicate key {key!r}")
        out[key] = loader.construct_object(v_node, deep=deep)
    return out


_DupCheckLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_dup)


def check_static():
    print("\n[STATIC] rules/*.yaml 파싱 + 중복 키")
    for fn in sorted(os.listdir(os.path.join(PROJ, "rules"))):
        if fn.endswith((".yaml", ".yml")):
            try:
                yaml.load(open(os.path.join(PROJ, "rules", fn), encoding="utf-8"), _DupCheckLoader)
            except Exception as e:
                fail(f"rules/{fn}: {type(e).__name__}: {str(e)[:140]}")

    print("[STATIC] src/*.py import")
    for fn in sorted(os.listdir(os.path.join(PROJ, "src"))):
        if fn.endswith(".py"):
            try:
                importlib.import_module(fn[:-3])
            except Exception as e:
                fail(f"src/{fn}: {type(e).__name__}: {str(e)[:140]}")
    if failures:
        return  # schema checks below need the modules to import

    import summarize

    print("[STATIC] symbols.yaml 스키마")
    S = summarize.load_symbols()
    prios = []
    for name, spec in S.items():
        if not isinstance(spec, dict):
            fail(f"symbols.yaml {name}: mapping이 아님")
            continue
        if not spec.get("aliases") and not spec.get("from"):
            fail(f"symbols.yaml {name}: aliases도 from도 없음")
        if spec.get("dim"):
            if spec.get("priority") is None:
                fail(f"symbols.yaml {name}: dim인데 priority 없음")
            else:
                prios.append(spec["priority"])
        for f in ("scope",):
            if spec.get(f):
                try:
                    re.compile(spec[f])
                except re.error as e:
                    fail(f"symbols.yaml {name}: {f} 정규식 오류 ({e})")
    dupes = {p for p in prios if prios.count(p) > 1}
    if dupes:
        fail(f"symbols.yaml: dim priority 중복 {sorted(dupes)} (렌더 순서가 비결정적이 됨)")

    print("[STATIC] derived_dims.yaml 스키마")
    D = summarize.load_derived_dims()
    for i, r in enumerate(D.get("rules") or []):
        for f in ("expr", "name", "sym"):
            if not r.get(f):
                fail(f"derived_dims rule[{i}] ({r.get('expr')}): '{f}' 없음")
        if r.get("scope"):
            try:
                re.compile(r["scope"])
            except re.error as e:
                fail(f"derived_dims rule[{i}]: scope 정규식 오류 ({e})")


# ---------------------------------------------------------------- FLEET
def _leaves(v, out):
    if isinstance(v, list):
        for x in v:
            _leaves(x, out)
    elif v is not None:
        out.append(v)


def scan_model(name):
    """Metrics for one published model. bare-int counting looks at shape ELEMENTS only --
    a coefficient inside an expression ("2*d_head") is not a bare integer, and miscounting
    those is how an earlier audit produced a wrong conclusion."""
    d = os.path.join(MODELS, name)
    m = {"c_fail": 0, "c17": "?", "unresolved": 0, "bare": 0, "bare_pct": 0.0,
         "unknown_syms": 0, "kv_card": None, "weight_T": 0, "self_contra": 0,
         "label_false": 0, "param_incons": 0, "flow_wrong": 0, "flow_ambig": 0}

    report = os.path.join(d, "full", "report.md")
    if os.path.exists(report):
        for line in open(report, encoding="utf-8"):
            if re.match(r"^C\d+\s+FAIL", line):
                m["c_fail"] += 1
            mm = re.match(r"^C17\s+(\w+)", line)
            if mm:
                m["c17"] = mm.group(1)

    summary = os.path.join(d, "model_summary.md")
    if os.path.exists(summary):
        text = open(summary, encoding="utf-8").read()
        m["unresolved"] = text.count("미해결 — 아래 Tier 3")
        m["unknown_syms"] = len(re.findall(r"^\| (\w+) \| _\(미확인", text, re.M))
        km = re.search(r"KV CACHE / TOKEN \(BF16\) \| ([\d.]+ KiB \([A-Za-z ]+\))", text)
        if km:
            m["kv_card"] = km.group(1)

    raw = os.path.join(d, "full", "prefill.trace.raw.jsonl")
    if os.path.exists(raw):
        bare = sym = 0
        module_paths = set()
        for line in open(raw, encoding="utf-8"):
            r = json.loads(line)
            el = []
            for fld in ("input_shape", "output_shape", "weight_shape"):
                _leaves(r.get(fld), el)
            for e in el:
                if str(e).isdigit():
                    if int(e) > 1:
                        bare += 1
                else:
                    sym += 1
            if r.get("module_path"):
                module_paths.add(r["module_path"])
            # INVARIANT: a weight is allocated from config at load time, so it can never depend
            # on the runtime sequence length. Any T in a weight axis is therefore provably wrong.
            # This is the single strongest automatic check we have on LABEL correctness -- the
            # metrics above (bare counts, C-checks) are all blind to a wrong-but-plausible label,
            # which is exactly why an outside reviewer, not this harness, caught the first one.
            for e in (r.get("weight_shape") or []):
                if re.search(r"\bT\b", str(e)):
                    m["weight_T"] += 1
        m["bare"] = bare
        m["bare_pct"] = round(100 * bare / max(1, bare + sym), 2)

        # INVARIANT: the symbol table must not contradict the model's own trace. Llama-4 shipped
        # E_shared=0 while its trace carried 120 shared_expert modules (config had no such field
        # at all -- the count is hardcoded in modeling code).
        st = os.path.join(d, "structure.yaml")
        if os.path.exists(st):
            syms = (yaml.safe_load(open(st, encoding="utf-8")) or {}).get("symbols", {}) or {}
            for symbol, rx, label in (("E_shared", r"\.shared_expert", "shared expert"),
                                      ("E", r"\.experts?(\.|$)", "routed experts")):
                present = any(re.search(rx, p) for p in module_paths)
                if present and syms.get(symbol) in (0, None):
                    m["self_contra"] += 1

        m["label_false"], m["param_incons"] = _label_checks(d)
        m["flow_wrong"], m["flow_ambig"] = _dataflow_label_check(d)
    return m


def _label_checks(d):
    """(arithmetically-false labels, parameters labelled inconsistently).

    These two are the only automatic checks that grade LABEL CORRECTNESS rather than trace
    well-formedness. Everything else here is blind to a wrong-but-plausible name, which is why an
    outside reviewer -- not this harness -- caught the first round of labelling bugs.

      label_false  : substitute provenance.symbol_table into every rendered label and require it
                     to equal the concrete value recorded in the shapes sidecar. A label that is
                     numerically false for its own model is unambiguously a bug.
      param_incons : one named parameter has one shape, so it must carry one labelling in every
                     op that touches it. Zamba2 rendered the same q_proj weight as
                     `[n_h*d_head, d_attn]` on the matmul and the reverse on the transpose.
    """
    raw = os.path.join(d, "full", "prefill.trace.raw.jsonl")
    conc = os.path.join(d, "full", "prefill.shapes.concrete.jsonl")
    prov = os.path.join(d, "full", "provenance.json")
    if not (os.path.exists(raw) and os.path.exists(conc) and os.path.exists(prov)):
        return 0, 0
    import math
    sys.path.insert(0, os.path.join(PROJ, "src"))
    import summarize

    p = json.load(open(prov, encoding="utf-8"))
    ns = dict(p.get("symbol_table") or {})
    ns["B"] = 1

    class _C:
        def __init__(s, dd):
            for k, v in dd.items():
                setattr(s, k, v)

    ns.update(summarize._derived_vars(_C(p.get("config") or {}), summarize.load_derived_dims()))
    if ns.get("n_h_ssm") and ns.get("d_head_ssm"):
        ns["d_inner"] = ns["n_h_ssm"] * ns["d_head_ssm"]
    if ns.get("n_g_ssm"):
        ns["n_g"] = ns["n_g_ssm"]
    for a, b in (("n_k", "n_h_lin_k"), ("d_k", "d_head_lin_k"), ("n_v", "n_h_lin_v")):
        if ns.get(b):
            ns[a] = ns[b]
    ns.update(ceil=math.ceil, round=round, min=min, max=max,
              roundup=lambda a, b: math.ceil(a / b) * b)

    def ev(expr):
        e = expr.replace("·", "*").replace("−", "-")
        e = re.sub(r"(?<![/*])/(?![/*])", "//", e)   # our formulas mean floor division
        return eval(e, {"__builtins__": {}}, ns)     # noqa: S307 -- our own rendered labels

    cmap = {json.loads(l)["op_id"]: json.loads(l) for l in open(conc, encoding="utf-8")}
    false_n = 0
    by_param, seen_param = {}, 0
    for line in open(raw, encoding="utf-8"):
        r = json.loads(line)
        c = cmap.get(r["op_id"])
        ws, ps = r.get("weight_shape"), (r.get("params") or [])
        if ws and len(ps) == 1:
            by_param.setdefault(ps[0], set()).add(tuple(str(x) for x in ws))
        if not c:
            continue
        for fld in ("input_shape", "output_shape", "weight_shape"):
            sv_all, cv_all = r.get(fld), c.get(fld)
            if sv_all is None or cv_all is None:
                continue
            pairs = [(sv_all, cv_all)] if fld == "weight_shape" else list(zip(sv_all, cv_all))
            for sv, cv in pairs:
                if not isinstance(sv, list) or not isinstance(cv, list):
                    continue
                for s, cc in zip(sv, cv):
                    s = str(s)
                    if s.isdigit():
                        if int(s) != cc:
                            false_n += 1
                        continue
                    try:
                        if ev(s) != cc:
                            false_n += 1
                    except Exception:
                        pass          # label we cannot evaluate -> not evidence of an error
    seen_param = sum(1 for v in by_param.values() if len(v) > 1)
    return false_n, seen_param



def _dataflow_label_check(d):
    """(provably-wrong mismatches, benign-ambiguous mismatches) along the dependency graph.

    FIRST-PRINCIPLES CHECK, and the only one here that can catch a bug class nobody anticipated:
    a tensor flowing from op A's output into op B's input is ONE tensor, so it must read the same
    in both places. No domain knowledge required -- it needs only the graph and the concrete
    shapes, so it does not depend on us having seen the failure mode before. It is what would
    have caught DeepSeek-V4's `g_o` vs `T/m_hca` automatically instead of by eye.

    Two outcomes are separated because only one is a defect:
      wrong  -- one side is a bare integer while the other has a name (information dropped), or
                the same product is spelled two ways (`E*T` vs `T*E`). Always fixable; FAILs.
      ambig  -- both sides carry a real name and the model makes them numerically equal
                (gpt2-xl: d_model == n_h*d_head by construction; gpt-oss: d_model == d_ff ==
                d_moe == 2880). Both names are TRUE for that tensor, so forcing one would destroy
                information rather than add it. Tracked as a number so a sudden jump is visible.
    """
    raw = os.path.join(d, "full", "prefill.trace.raw.jsonl")
    conc = os.path.join(d, "full", "prefill.shapes.concrete.jsonl")
    if not (os.path.exists(raw) and os.path.exists(conc)):
        return 0, 0
    sym = {json.loads(l)["op_id"]: json.loads(l) for l in open(raw, encoding="utf-8")}
    con = {json.loads(l)["op_id"]: json.loads(l) for l in open(conc, encoding="utf-8")}
    wrong = ambig = 0
    for oid, r in sym.items():
        cr = con.get(oid)
        if not cr:
            continue
        for dep in (r.get("depends_on") or []):
            ar, ac = sym.get(dep), con.get(dep)
            if not ar or not ac:
                continue
            for bi_s, bi_c in zip(r.get("input_shape") or [], cr.get("input_shape") or []):
                if not isinstance(bi_c, list) or not bi_c:
                    continue
                for ao_s, ao_c in zip(ar.get("output_shape") or [], ac.get("output_shape") or []):
                    if not isinstance(ao_c, list) or ao_c != bi_c:
                        continue
                    for ls, lo in zip(bi_s, ao_s):
                        ls, lo = str(ls), str(lo)
                        if ls == lo:
                            continue
                        if ls.isdigit() or lo.isdigit():
                            wrong += 1                       # information dropped on one side
                        elif sorted(re.split(r"[*+]", ls)) == sorted(re.split(r"[*+]", lo)):
                            wrong += 1                       # same expression, two spellings
                        else:
                            ambig += 1                       # two true names, equal by construction
    return wrong, ambig

def check_fleet():
    print("\n[FLEET] 모델별 지표")
    names = sorted(n for n in os.listdir(MODELS) if os.path.isdir(os.path.join(MODELS, n)))
    out = {}
    print(f"   {'model':46s} {'Cfail':>5} {'C17':>5} {'unres':>5} {'bare':>7} {'bare%':>6} "
          f"{'?sym':>5} {'wT':>4} {'ctra':>5} {'false':>6} {'pInc':>5} {'flowX':>6} {'flow~':>6}")
    for n in names:
        m = scan_model(n)
        out[n] = m
        print(f"   {n:46s} {m['c_fail']:5d} {m['c17']:>5} {m['unresolved']:5d} "
              f"{m['bare']:7d} {m['bare_pct']:6.2f} {m['unknown_syms']:5d} "
              f"{m['weight_T']:4d} {m['self_contra']:5d} {m['label_false']:6d} "
              f"{m['param_incons']:5d} {m['flow_wrong']:6d} {m['flow_ambig']:6d}")
        if m["c_fail"]:
            fail(f"{n}: C체크 FAIL {m['c_fail']}개")
        if m["c17"] not in ("PASS", "?"):
            fail(f"{n}: C17={m['c17']} (온보딩 미완 — 02-new-module-handling.md Phase 0)")
        if m["unresolved"]:
            fail(f"{n}: 미해결 유도 상수 {m['unresolved']}개")
        if m["weight_T"]:
            fail(f"{n}: 가중치 축에 T가 {m['weight_T']}건 — 정적 파라미터는 시퀀스 길이에 "
                 f"의존할 수 없음(물리적 불변식 위반)")
        if m["self_contra"]:
            fail(f"{n}: 심볼표가 자기 트레이스와 모순 {m['self_contra']}건 "
                 f"(트레이스에 있는 구조를 심볼표가 0/없음으로 표기)")
        if m["label_false"]:
            fail(f"{n}: 산술적으로 거짓인 라벨 {m['label_false']}건 "
                 f"(심볼표 대입값 != 실제 구체값)")
        if m["param_incons"]:
            fail(f"{n}: 같은 파라미터가 op마다 다르게 라벨링됨 {m['param_incons']}건")
        if m["flow_wrong"]:
            fail(f"{n}: 데이터플로우 라벨 불일치 {m['flow_wrong']}건 "
                 f"(같은 텐서인데 한쪽만 정수이거나 표기가 다름)")
    return out


# ---------------------------------------------------------------- EXTERNAL
def _count_attn_layers(d):
    """Layers running softmax attention, by presence of a query projection (same rule as
    summarize.py). Kept independent of the summary text so it checks the pipeline, not itself."""
    raw = os.path.join(d, "full", "prefill.trace.raw.jsonl")
    if not os.path.exists(raw):
        return None
    rx = re.compile(r"\.(q_proj|q_a_proj|q_b_proj|query_key_value|qkv_proj|Wqkv|c_attn)$")
    seen = set()
    for line in open(raw, encoding="utf-8"):
        r = json.loads(line)
        if r.get("layer_idx") is not None and rx.search(r.get("module_path") or ""):
            seen.add(r["layer_idx"])
    return len(seen)


def check_external(fleet):
    print("\n[EXTERNAL] attention 레이어 수 (트레이스 실측 vs 기록)")
    refs = yaml.safe_load(open(REFS, encoding="utf-8"))
    exp = (refs.get("attention_layers") or {}).get("values") or {}
    ok_a = 0
    for name, expected in sorted(exp.items()):
        got = _count_attn_layers(os.path.join(MODELS, name))
        if got is None:
            warn(f"{name}: 트레이스 없음")
        elif got != expected:
            fail(f"{name}: attention 레이어 {got}개 != 기록된 {expected}개 "
                 f"(attention 유무 판정이 뒤집혔을 수 있음)")
        else:
            ok_a += 1
    print(f"   {ok_a}/{len(exp)} 일치")

    print("\n[EXTERNAL] KV cache 카드 vs 공개 갤러리 수치")
    want = refs["kv_cache_per_token"]["values"]
    ok = 0
    for name, expected in sorted(want.items()):
        got = (fleet.get(name) or {}).get("kv_card")
        if got is None:
            warn(f"{name}: KV 카드 없음 (모델 미생성?)")
        elif got != expected:
            fail(f"{name}: KV 카드 '{got}' != 공개값 '{expected}'")
        else:
            ok += 1
    print(f"   {ok}/{len(want)} 일치")


# ---------------------------------------------------------------- BASELINE
def check_baseline(fleet, update):
    print("\n[BASELINE] 이전 기준 대비 퇴행 검사")
    cur = {n: {"bare": m["bare"], "unresolved": m["unresolved"],
               "unknown_syms": m["unknown_syms"], "c_fail": m["c_fail"],
               "flow_ambig": m["flow_ambig"]}
           for n, m in fleet.items()}
    if update:
        os.makedirs(os.path.dirname(BASELINE), exist_ok=True)
        json.dump(cur, open(BASELINE, "w", encoding="utf-8"), indent=2, sort_keys=True)
        print(f"   기준 갱신됨 -> {os.path.relpath(BASELINE, PROJ)}")
        return
    if not os.path.exists(BASELINE):
        warn("기준 파일 없음 — --update-baseline 로 최초 기록 필요")
        return
    old = json.load(open(BASELINE, encoding="utf-8"))
    regressed = improved = 0
    for n, c in sorted(cur.items()):
        o = old.get(n)
        if o is None:
            print(f"   NEW   {n}")
            continue
        for key in ("bare", "unresolved", "unknown_syms", "c_fail", "flow_ambig"):
            if key not in o:
                continue      # metric added after this baseline was written -- nothing to compare
            if c[key] > o[key]:
                fail(f"{n}: {key} 퇴행 {o[key]} -> {c[key]}")
                regressed += 1
            elif c[key] < o[key]:
                improved += 1
    for n in sorted(set(old) - set(cur)):
        warn(f"{n}: 기준에는 있는데 지금 없음 (모델이 사라짐)")
    print(f"   퇴행 {regressed}건 / 개선 {improved}건")
    if improved and not regressed:
        print("   -> 개선만 있음. 검토 후 --update-baseline 로 기준을 올릴 것.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--update-baseline", action="store_true",
                    help="현재 지표를 새 기준으로 기록(검토 후에만)")
    args = ap.parse_args()

    check_static()
    if failures:
        print("\n=== STATIC 단계에서 실패 — 이후 검사 생략 ===")
    else:
        fleet = check_fleet()
        check_external(fleet)
        check_baseline(fleet, args.update_baseline)

    print("\n" + "=" * 72)
    print(f"FAIL {len(failures)}건 / WARN {len(warnings)}건")
    for f in failures:
        print(f"  - {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
