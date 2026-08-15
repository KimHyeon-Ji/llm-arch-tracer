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

# A LayerNorm whose parent is the decoder layer itself -- i.e. one that normalises the residual
# stream. `\.layers\.\d+\.` followed by a single dot-free leaf, so a norm nested one level deeper
# (self_attn.q_a_layernorm, shared_transformer.input_layernorm) deliberately does NOT match: those
# operate on c_q / d_attn, not on the residual stream.
_RESID_NORM = re.compile(r".*\.(?:layers|h|blocks)\.\d+\.[A-Za-z0-9_]*layer_?norm$", re.I)

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

# 지금의 rules/ + src/ 지문. 모델마다 저장된 값과 비교해 "이 산출물이 현재 규칙으로 만들어졌는가"를
# 판정한다 (src/provenance.label_inputs_fingerprint).
def _fingerprint():
    sys.path.insert(0, os.path.join(PROJ, "src"))
    import provenance as _p
    return _p.label_inputs_fingerprint(PROJ)


_CUR_FINGERPRINT = None


def _cur_fingerprint():
    """지연 계산. main() 안에서만 채우면 scan_model 을 직접 부르는 쪽(셀프테스트, 감사 스크립트)이
    None 과 비교해 **전 모델을 낡음으로** 판정한다 -- 검사 자체가 거짓 경보를 내는 형태다."""
    global _CUR_FINGERPRINT
    if _CUR_FINGERPRINT is None:
        _CUR_FINGERPRINT = _fingerprint()
    return _CUR_FINGERPRINT

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
         "label_false": 0, "param_incons": 0, "flow_wrong": 0, "flow_ambig": 0,
         "head_excl": 0, "resid_norm": 0, "batch_excl": 0,
         "heur": 0, "ident_incons": 0, "reshape_incons": 0,
         "matmul_compose": 0, "membership": 0, "membership_notrun": 1,
         "override_dead": 0, "override_axes": 0, "stale": 0, "generated_at": None,
         "weight_operand": 0, "unanswered": 0,
         "uncited": 0, "claim_only": "", "soft_undet": 0, "axis_conflict": 0,
         "unsettled": 0, "bad_stub": 0, "dead_confirm": 0}

    # Module-field membership (src/source_check.membership_gaps), computed at regeneration and
    # persisted so this stays offline. A weight axis may only carry the name of a config field
    # the owning module -- or the module that constructed it -- actually reads. It is the one
    # label check that never looks at a value, so a coincidence of two config numbers cannot
    # hide anything from it. `ran: false` is reported separately: not run is not clean.
    mem = os.path.join(d, "full", "membership.json")
    if os.path.exists(mem):
        _mj = json.load(open(mem, encoding="utf-8"))
        m["membership_notrun"] = 0 if _mj.get("ran") else 1
        m["membership"] = len(_mj.get("gaps") or [])

    # A ④-layer override that matched NOTHING is a stale claim -- the model was re-traced, or the
    # rules improved and the label is already right. Leaving it in rules/label_overrides.yaml would
    # make the file read as "this is corrected" when nothing was corrected. Counted per model.
    # 이 모델의 산출물이 지금의 rules/ + src/ 로 만들어졌는가. 재생성이 조용히 실패하면 여기서
    # 걸린다 -- 실패한 모델은 스탬프를 새로 찍지 못하므로 지문이 옛것으로 남는다.
    # 의뢰서가 낸 질문에 실제로 답했는가 (src/review_ledger.unanswered)
    try:
        sys.path.insert(0, os.path.join(PROJ, "src"))
        import review_ledger as _rl
        m["unanswered_items"] = _rl.unanswered_items(d)
        m["unanswered"] = len(m["unanswered_items"])
        m["uncited"] = _rl.uncited(d)
        m["claim_only"] = _rl.claim_without_change(d, name)
        m["soft_undet"] = _rl.soft_undetermined(d)
        import axis_classes as _ac
        m["axis_conflict"] = _ac.conflicts(d)
        m["unsettled"] = _ac.unsettled_count(d)
        m["bad_stub"] = _ac.bad_stub_count(d)
        m["dead_confirm"] = _ac.dead_confirm_count(d)
    except Exception:
        m["unanswered"], m["unanswered_items"] = 0, []

    stamp = os.path.join(d, "full", "generated.json")
    if not os.path.exists(stamp):
        m["stale"] = 1
    else:
        try:
            _sj = json.load(open(stamp, encoding="utf-8"))
            m["stale"] = 0 if _sj.get("label_inputs") == _cur_fingerprint() else 1
            m["generated_at"] = _sj.get("generated_at")
        except (ValueError, OSError):
            m["stale"] = 1

    ov = os.path.join(d, "full", "label_overrides.json")
    if os.path.exists(ov):
        _ov = json.load(open(ov, encoding="utf-8"))
        m["override_dead"] = sum(1 for o in _ov if not o.get("applied"))
        m["override_axes"] = sum(int(o.get("applied") or 0) for o in _ov)

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

    # BOTH PHASES. Until 2026-08-12 every label check here read `prefill` only, so the decode
    # tables -- half of what we publish, and the half that carries the KV cache, the `T+1` widths
    # and every size-1 axis -- were never checked at all. Running the same checks on decode found
    # MORE disagreements than prefill (872 vs 731) purely because nothing had ever looked.
    # The once-per-model parts (symbol table, heuristic tally) stay outside this loop.
    bare = sym = 0
    module_paths = set()
    import build_table as _bt
    for _phase in ("prefill", "decode"):
      raw = os.path.join(d, "full", f"{_phase}.trace.raw.jsonl")
      if os.path.exists(raw):
        _conc = _bt.load_concrete(d, _phase) or {}
        # Name the residual-stream width carries in rendered labels (see _RESID_NORM below). Only
        # set when the model actually resolved a d_model, so nothing is asserted about a model whose
        # hidden size we could not name.
        st0 = os.path.join(d, "structure.yaml")
        dmodel_sym = None
        if os.path.exists(st0):
            _s0 = (yaml.safe_load(open(st0, encoding="utf-8")) or {}).get("symbols", {}) or {}
            dmodel_sym = "d_model" if _s0.get("d_model") else None
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

            # INVARIANT: one tensor is laid out along query heads OR kv heads, never both. The two
            # counts meet only in repeat_kv, which bridges them with the DERIVED `n_h/n_kv` factor,
            # never with both plain names in one tuple. So the pair co-occurring proves one of them
            # was pasted onto an axis that is not a head-count axis at all. Promoted to the gate
            # after a self-audit found 16,859 such axes across 8 models (Llama-405B/70B, gpt-oss
            # 20b/120b, DeepSeek-V3, ...) -- always a head-SIZE or partial-RoPE axis stolen by a
            # head-COUNT name because the values coincided. See symbolic_shape._HEAD_COUNT_EXCLUSIVE.
            for fld in ("input_shape", "output_shape", "weight_shape"):
                for sh in (r.get(fld) or []):
                    sh = sh if isinstance(sh, list) else [sh]
                    if "n_h" in sh and "n_kv" in sh:
                        m["head_excl"] += 1

            # INVARIANT: a tensor has ONE batch axis, so `B` cannot appear twice in one shape.
            # Same failure family as head_excl: symbolic_shape.dim() answered "B" for EVERY
            # size-1 axis, which is right for the leading one and wrong for every broadcast or
            # reduced singleton after it (`[B,T,B]` from an RMSNorm mean, `[B,n_h,B]`). 200,055
            # of 588,046 rendered shapes (34%) were affected and no metric could see it -- `bare`
            # skips 1 by design. Found by reading the layer-3 review packet, 2026-08-05.
            # Two more forms of the same impossibility: a `B` that comes AFTER the sequence axis
            # (HF layouts always put batch ahead of sequence, so the 1 in a router's
            # `sum(keepdim=True)` -> [T,1] is a reduced axis), and a `B` anywhere in a WEIGHT
            # (a parameter is allocated from config at load time and has no batch dimension --
            # Qwen3-Next's nn.Linear(d_model,1) gate rendered [B,d_model] and _propagate_labels
            # then carried that B onto the matmul output).
            for fld in ("input_shape", "output_shape"):
                for sh in (r.get(fld) or []):
                    sh = [str(x) for x in (sh if isinstance(sh, list) else [sh])]
                    if sh.count("B") > 1 or ("B" in sh and "T" in sh
                                             and sh.index("B") > sh.index("T")):
                        m["batch_excl"] += 1
            ws = r.get("weight_shape")
            if ws:                          # flat list, unlike input/output_shape
                flat = ws if not isinstance(ws[0], list) else [x for s in ws for x in s]
                m["batch_excl"] += sum(1 for x in flat if str(x) == "B")

            # CROSS-CHECK: a reshape's output axes can be derived from its own input axes
            # (see build_table.derive_from_reshape). 97.6% of derivable axes already agree;
            # a disagreement means one of the two accounts of the same tensor is wrong, and is
            # almost always a value collision (two symbols with the same number in one model).
            _c = _conc.get(r.get("op_id"))
            if _c:
                _row = dict(r)
                _row["input_shape"] = _c.get("input_shape") or []
                _row["output_shape"] = _c.get("output_shape") or []
                m["reshape_incons"] += len(_bt.reshape_disagreements(_row, r))

            # INVARIANT: a matmul's labels must COMPOSE. `[.., m, k] @ [.., k, n] -> [.., m, n]`
            # is the definition of the op, so the contraction axes must read the same name and
            # the output must read m and n. Nothing else checked this: every existing check looks
            # at one shape at a time, and a weight operand can be internally plausible while
            # contradicting the very multiplication it takes part in. Found by the layer-3 review,
            # 2026-08-06 -- Qwen2.5-0.5B's biased q_proj renders `[T,d_model] @ [d_model,d_model]`
            # while its output correctly says `n_h*d_head`, i.e. the weight's out axis is wrong.
            # Runtime axes are skipped: a size-1 operand renders `B` or `1` depending on position,
            # which is a rendering convention rather than a disagreement about the tensor.
            if r.get("op_type") in ("matmul", "linear", "mm", "bmm", "batched_matmul"):
                _ins = [x for x in (r.get("input_shape") or [])
                        if isinstance(x, list) and len(x) >= 2]
                _out = ((r.get("output_shape") or [None]) or [None])[0]
                if len(_ins) >= 2 and isinstance(_out, list) and len(_out) >= 2:
                    _a, _b = _ins[-2], _ins[-1]
                    _rt = {"B", "1"}
                    _pairs = ((str(_a[-1]), str(_b[-2])), (str(_a[-2]), str(_out[-2])),
                              (str(_b[-1]), str(_out[-1])))
                    for _x, _y in _pairs:
                        if _x != _y and _x not in _rt and _y not in _rt:
                            m["matmul_compose"] += 1

            # INVARIANT: ONE TENSOR, ONE NAME PER AXIS -- WITHIN a single op. The weight appears
            # twice in every projection row: once in `weight_shape` (as stored) and once inside
            # `input_shape` (as the operand the matmul consumed, usually transposed). Those are the
            # same tensor, so each axis must read the same symbol in both. Nothing checked it:
            # `param_incons` compares a parameter ACROSS ops (consistent here -- it is wrong the
            # same way everywhere), and the matmul-composition check only looks inside
            # `input_shape`. So a fix that corrected the operand and left the stored form behind
            # passed every gate: Llama-3.1-405B/70B read `[T, d_model] @ [d_model, n_kv*d_head]`
            # while the very same weight's `weight_shape` said `[n_kv*d_head, n_h*d_head]` --
            # d_model and n_h*d_head are equal there, so no value check could see it either.
            # Found by an outside review, 2026-08-12. `weight_pos` names the operand.
            _wp, _ws = r.get("weight_pos"), r.get("weight_shape")
            if (isinstance(_wp, int) and _wp >= 0 and isinstance(_ws, list) and _ws
                    and not isinstance(_ws[0], list)):
                _ins = r.get("input_shape") or []
                if _wp < len(_ins) and isinstance(_ins[_wp], list) and len(_ins[_wp]) == len(_ws):
                    _op = [str(x) for x in _ins[_wp]]
                    _st = [str(x) for x in _ws]
                    # stored [out, in]; the operand may be the transpose of the trailing pair
                    _sw = _st[:-2] + _st[-2:][::-1] if len(_st) >= 2 else _st
                    if _op != _st and _op != _sw:
                        m["weight_operand"] += 1

            # INVARIANT: an op that only copies (clone/_to_copy/contiguous/detach) cannot change
            # what an axis MEANS, so its output labels must equal its input labels. Caught
            # DeepSeek-V4-Pro's compressor, where T/m_csa (2048/4) collides with d_head (512) and
            # the axes came out swapped -- DeepSeek-V4-Flash traces the same module at T=1032,
            # where 258 != 512, and shows the true layout [B,T/m_csa,d_head].
            if r.get("op_type") in ("clone", "_to_copy", "contiguous", "detach", "alias",
                                    "copy_"):
                ins, outs = (r.get("input_shape") or []), (r.get("output_shape") or [])
                if len(ins) == 1 and len(outs) == 1                         and isinstance(ins[0], list) and isinstance(outs[0], list)                         and len(ins[0]) == len(outs[0])                         and [str(x) for x in ins[0]] != [str(x) for x in outs[0]]:
                    m["ident_incons"] += 1

            # SAME INVARIANT, wider: an ELEMENTWISE op does not change what an axis means either,
            # and an operand whose CONCRETE shape equals the output's is the same tensor laid out
            # the same way -- so it must read the same. The copy-only version above missed this
            # because it required a single input: DeepSeek-V4-Pro's mHC mixing is
            # `elementwise_add([B,1,4,d_model], [B,1,4,d_model]) -> [B,1,n_hc,d_model]`, two names
            # for one tensor inside one row, and no check saw it (found by a reader of the CSV,
            # 2026-08-10; the cause was `n_hc`'s scope not covering the layer root where mHC mixes).
            # Restricted to shape-PRESERVING elementwise ops and to operands that match the output
            # concretely, so a coincidental shape match on a contraction cannot trip it.
            if r.get("op_type") in ("elementwise_add", "elementwise_mul", "elementwise_sub",
                                    "elementwise_div", "maximum", "minimum", "where",
                                    "masked_fill", "rsqrt", "exp", "neg", "silu", "gelu"):
                outs = r.get("output_shape") or []
                cout = (_conc.get(r.get("op_id")) or {}).get("output_shape") or []
                cins = (_conc.get(r.get("op_id")) or {}).get("input_shape") or []
                if len(outs) == 1 and isinstance(outs[0], list) and len(cout) == 1:
                    for _i, _cin in enumerate(cins):
                        if not isinstance(_cin, list) or list(_cin) != list(cout[0]):
                            continue
                        _sin = (r.get("input_shape") or [])
                        if _i < len(_sin) and isinstance(_sin[_i], list)                                 and [str(x) for x in _sin[_i]] != [str(x) for x in outs[0]]:
                            m["ident_incons"] += 1

            # INVARIANT: a LayerNorm sitting DIRECTLY on the decoder layer normalises the residual
            # stream, so its activation width is d_model. Its leaf name states its POSITION in the
            # block ("post_attention_layernorm"), never what it computes -- but every `attn|attention`
            # scope regex matches that substring, so the whole attention symbol set fired there. On
            # Llama-3.1-70B/405B (d_model == n_h*d_head) the residual stream came out `n_h*d_head`
            # right next to an elementwise_add that called the same tensor `d_model`. Norms NESTED
            # inside a sub-module (self_attn.q_a_layernorm, shared_transformer.input_layernorm) are
            # excluded: those legitimately carry c_q / d_attn. External review, 2026-07-30.
            # `1` is allowed because an RMSNorm's own variance step reduces the width away
            # (`mean`/`rsqrt` -> [B,T,1]). Until 2026-08-05 that axis rendered as `B` and passed
            # through the batch allowance instead; the allowance is now spelled correctly.
            mp = r.get("module_path") or ""
            if _RESID_NORM.match(mp) and dmodel_sym:
                for fld in ("input_shape", "output_shape"):
                    for sh in (r.get(fld) or []):
                        sh = sh if isinstance(sh, list) else [sh]
                        if sh and str(sh[-1]) not in ("B", "1", dmodel_sym):
                            m["resid_norm"] += 1
    m["bare"] = bare
    m["bare_pct"] = round(100 * bare / max(1, bare + sym), 2)

    if module_paths:
        # INVARIANT: the symbol table must not contradict the model's own trace. Llama-4 shipped
        # E_shared=0 while its trace carried 120 shared_expert modules (config had no such field
        # at all -- the count is hardcoded in modeling code).
        st = os.path.join(d, "structure.yaml")
        if os.path.exists(st):
            syms = (yaml.safe_load(open(st, encoding="utf-8")) or {}).get("symbols", {}) or {}
            # How many axes got their name from the arithmetic tail rather than a registered
            # rule. Tracked so a labelling change cannot quietly trade "derived" for "guessed" --
            # the two are indistinguishable in every other metric (both look like a named axis).
            m["heur"] = int(((yaml.safe_load(open(st, encoding="utf-8")) or {})
                             .get("label_provenance") or {}).get("heuristic") or 0)
            for symbol, rx, label in (("E_shared", r"\.shared_expert", "shared expert"),
                                      ("E", r"\.experts?(\.|$)", "routed experts")):
                present = any(re.search(rx, p) for p in module_paths)
                if present and syms.get(symbol) in (0, None):
                    m["self_contra"] += 1

        for _phase in ("prefill", "decode"):
            _lf, _pi = _label_checks(d, _phase)
            _fw, _fa = _dataflow_label_check(d, _phase)
            m["label_false"] += _lf
            m["param_incons"] += _pi
            m["flow_wrong"] += _fw
            m["flow_ambig"] += _fa
    return m


def _label_checks(d, phase="prefill"):
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
    raw = os.path.join(d, "full", f"{phase}.trace.raw.jsonl")
    conc = os.path.join(d, "full", f"{phase}.shapes.concrete.jsonl")
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



def _dataflow_label_check(d, phase="prefill"):
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
    raw = os.path.join(d, "full", f"{phase}.trace.raw.jsonl")
    conc = os.path.join(d, "full", f"{phase}.shapes.concrete.jsonl")
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
            # 어느 출력이 어느 피연산자로 갔는지 **모르면 아무 주장도 하지 않는다.**
            # 생산자의 두 출력이 같은 shape 이면(MLA 의 `split(kv, [nope, v])` 는 두 조각이
            # 같은 폭이다) 검사기는 두 짝을 다 맞춰 보고 **양쪽 다 불일치로 센다** -- 라벨이
            # 소스대로 맞아도 그렇다. 실제로 Kimi 에서 244건이 그 오탐이었다(2026-08-14).
            # `src/axis_classes` 의 간선 조건과 같은 원칙이다: 모호하면 잇지도, 탓하지도 않는다.
            _douts = [x for x in (ac.get("output_shape") or []) if isinstance(x, list)]
            for bi_s, bi_c in zip(r.get("input_shape") or [], cr.get("input_shape") or []):
                if not isinstance(bi_c, list) or not bi_c:
                    continue
                if sum(1 for x in _douts if x == bi_c) != 1:
                    continue
                if sum(1 for x in (cr.get("input_shape") or [])
                       if isinstance(x, list) and x == bi_c) != 1:
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
          f"{'?sym':>5} {'wT':>4} {'ctra':>5} {'false':>6} {'pInc':>5} {'flowX':>6} {'flow~':>6} "
          f"{'hdEx':>5} {'rNrm':>5}")
    for n in names:
        m = scan_model(n)
        out[n] = m
        print(f"   {n:46s} {m['c_fail']:5d} {m['c17']:>5} {m['unresolved']:5d} "
              f"{m['bare']:7d} {m['bare_pct']:6.2f} {m['unknown_syms']:5d} "
              f"{m['weight_T']:4d} {m['self_contra']:5d} {m['label_false']:6d} "
              f"{m['param_incons']:5d} {m['flow_wrong']:6d} {m['flow_ambig']:6d} "
              f"{m['head_excl']:5d} {m['resid_norm']:5d} {m['batch_excl']:6d} "
              f"{m['heur']:6d} {m['ident_incons']:5d} {m['reshape_incons']:6d} "
              f"{m['matmul_compose']:6d}")
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
        if m["matmul_compose"]:
            warn(f"{n}: 행렬곱 라벨이 합성되지 않음 {m['matmul_compose']}건 — "
                 f"[m,k]@[k,n]->[m,n] 관계가 이름으로 성립하지 않는다")
        if m["reshape_incons"]:
            warn(f"{n}: reshape 자체 유도와 라벨이 불일치 {m['reshape_incons']}건 — "
                 f"같은 텐서에 대한 두 설명이 다르다(값 충돌 의심)")
        if m["ident_incons"]:
            warn(f"{n}: 복사 op가 축 라벨을 바꿈 {m['ident_incons']}건 — 값이 겹치는 축의 "
                 f"순서 모호성(01-main.md §10 참고)")
        if m["batch_excl"]:
            fail(f"{n}: 한 shape에 B가 2번 이상 {m['batch_excl']}건 — 텐서의 배치 축은 하나뿐이므로 "
                 f"뒤쪽 크기-1 축은 배치가 아니라 리터럴 1이다")
        if m["head_excl"]:
            fail(f"{n}: 한 shape에 n_h와 n_kv가 동시에 {m['head_excl']}건 — 텐서는 Q head 축이거나 "
                 f"KV head 축이지 둘 다일 수 없음(head 크기 축이 head 개수 이름에 뺏긴 것)")
        if m["resid_norm"]:
            fail(f"{n}: 레이어 직속 LayerNorm의 활성 폭이 d_model이 아님 {m['resid_norm']}건 — "
                 f"잔차 스트림을 정규화하는 모듈이므로 폭은 d_model이어야 함")
        if m["membership"]:
            fail(f"{n}: 가중치 축이 그 모듈이 읽지도 않는 config 필드의 이름을 달고 있음 "
                 f"{m['membership']}건 — 파라미터 shape은 그 모듈이 선언한 것이므로, 그 모듈도 "
                 f"부모도 읽지 않는 필드의 이름은 산술이 맞아도 근거가 없다 "
                 f"(full/membership.json)")
        if m["stale"]:
            fail(f"{n}: 산출물이 현재 rules/ + src/ 로 만들어지지 않았다 "
                 f"(생성 {m['generated_at'] or '기록 없음'}) — 재생성이 실패했거나 규칙이 바뀐 뒤 "
                 f"돌리지 않았다. develop/regen_summaries.py 를 돌릴 것")
        if m["unanswered"]:
            # 개수가 아니라 **무엇이** 안 됐는지 찍는다. 개수만 맞추면 엉뚱한 것에 답하고도
            # 통과할 수 있었고, 실제로 그랬다(Llama-4 의 `E*T`, 2라운드 연속 무응답).
            fail(f"{n}: 의뢰서 항목 {m['unanswered']}건에 대응하는 판정이 없다 — 그 항목의 "
                 f"라벨도 모듈도 언급한 판정이 하나도 없다. review/prompt.md 를 이 모델에 "
                 f"돌릴 것:")
            for line in m.get("unanswered_items") or []:
                fail(f"      {line}")
        if m["dead_confirm"]:
            fail(f"{n}: 더 이상 맞지 않는 확인 기록 {m['dead_confirm']}건 — "
                 f"rules/label_confirmed.yaml 이 \"이 이름이 맞다\"고 적어 뒀는데 그 축의 "
                 f"이름이 그 뒤로 바뀌었다. '예전에 맞았다'는 지금 맞다는 뜻이 아니다 "
                 f"(full/label_confirmed.json)")
        if m["bad_stub"]:
            fail(f"{n}: 지목 불가능한 인계 초안 {m['bad_stub']}건 — 그 초안을 그대로 쓰면 "
                 f"같은 레이어의 다른 등가류까지 바꿔 망가진다. 검토자에게 못 쓰는 초안을 "
                 f"주는 것은 안 주느니만 못하다 (full/*.unsettled.json 의 stub_ambiguous)")
        if m["unsettled"]:
            warn(f"{n}: 규칙이 끝내지 못해 ④층으로 넘긴 축 {m['unsettled']}건 — "
                 f"full/*.unsettled.json 과 review_request.md 0절에 질문과 override 초안이 "
                 f"있다. 이건 결함이 아니라 **인계**다: 값으로 못 가리는 자리를 규칙으로 "
                 f"우기지 않고 소스를 읽는 층에 넘긴 것이다")
        if m["axis_conflict"]:
            fail(f"{n}: 한 축에 이름이 둘 이상인 등가류 {m['axis_conflict']}건 — 같은 텐서의 "
                 f"같은 축인데 자리마다 다른 이름이 붙어 있다. 2026-08-14 부터 이 값은 함대 "
                 f"전체에서 0 이므로(축 등가류 통일) 0 이 아니면 결함이다. "
                 f"full/*.axis_classes.json 에 어느 자리인지 적혀 있다")
        if m.get("claim_only"):
            fail(f"{n}: {m['claim_only']}")
        if m["soft_undet"]:
            fail(f"{n}: 밖을 찾아본 흔적 없는 '확인 못함' 판정 {m['soft_undet']}건 — 캐시된 HF "
                 f"소스에 답이 없으면 저장소 remote code / vLLM / 커널 / 논문 중 어디를 봤는지 "
                 f"URL 로 남겨야 한다. 전수 재검토에서 이 자리 13건 중 10건이 사실은 "
                 f"'알지만 못 넣음' 이었다 (review/prompt.md)")
        if m["uncited"]:
            fail(f"{n}: 소스 인용 없는 교정 주장 {m['uncited']}건 — '이 이름은 틀렸다'는 소스에 "
                 f"대한 주장이므로 무엇을 읽었는지 적어야 한다 (review/prompt.md '근거 없는 "
                 f"판정은 판정이 아니다')")
        if m["weight_operand"]:
            fail(f"{n}: 같은 가중치가 한 행 안에서 두 이름 {m['weight_operand']}건 — "
                 f"`weight_shape`(저장 형태)와 `input_shape`의 그 피연산자가 같은 텐서인데 "
                 f"축 이름이 다르다. 한쪽만 고치고 다른 쪽을 방치한 수정이다")
        if m["override_dead"]:
            fail(f"{n}: 발화하지 않은 라벨 교정 {m['override_dead']}건 — rules/label_overrides.yaml "
                 f"의 항목이 아무 축에도 안 맞았다. 이미 고쳐졌거나 조건이 틀렸다는 뜻이므로 "
                 f"근거를 다시 보고 지우거나 고칠 것 (full/label_overrides.json)")
        elif m["membership_notrun"]:
            warn(f"{n}: 모듈-필드 소속 검사 미수행 — full/module_classes.json 또는 modeling "
                 f"소스가 없다. 통과가 아니다 (develop/backfill_module_classes.py)")
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
def check_documented_literals(refs):
    """Every bare integer we publish must have a WRITTEN reason (references.yaml).

    "이름이 없다" and "이름이 없는 이유를 안다" are different states, and only the second is a
    finished deliverable. Until 2026-08-06 the irreducible_literals list was prose that nothing
    read, so a NEW unexplained constant looked exactly like a documented one. Checking it turns
    the list into a queue: anything not on it is the next thing to research.
    """
    print("\n[EXTERNAL] 이름 없는 정수에 문서화된 사유가 있는가")
    entries = refs.get("irreducible_literals") or []
    universal, per_model, freeform = set(), {}, set()
    for e in entries:
        vals, models = e.get("values"), e.get("models") or []
        if not isinstance(vals, list):
            freeform.update(models)          # e.g. Qwen3-Next's scan boundaries
            continue
        for m in models:
            (universal if m == "all" else per_model.setdefault(m, set())).update(vals)
    undocumented, checked = {}, 0
    for name in sorted(os.listdir(MODELS)):
        d = os.path.join(MODELS, name)
        if not os.path.isdir(d) or name in freeform:
            continue
        allowed = universal | per_model.get(name, set())
        seen = set()
        for phase in ("prefill", "decode"):
            raw = os.path.join(d, "full", f"{phase}.trace.raw.jsonl")
            if not os.path.exists(raw):
                continue
            checked += 1
            for line in open(raw, encoding="utf-8"):
                r = json.loads(line)
                for field in ("input_shape", "output_shape", "weight_shape"):
                    v = r.get(field)
                    if not v:
                        continue
                    for sh in (v if isinstance(v[0], list) else [v]):
                        for x in (sh if isinstance(sh, list) else [sh]):
                            x = str(x)
                            if x.isdigit() and int(x) > 1:
                                seen.add(int(x))
        extra = sorted(seen - allowed)
        if extra:
            undocumented[name] = extra
    if undocumented:
        for name, vals in undocumented.items():
            warn(f"{name}: 사유가 문서화되지 않은 정수 {vals[:8]} "
                 f"-- 확인 후 references.yaml irreducible_literals 에 등재하거나 규칙으로 등록")
    else:
        print(f"   {checked}개 트레이스 전부 문서화된 값만 사용")


def check_modeling_sourced(refs):
    """Symbols whose value was read off the modeling source must still hold in the symbol table."""
    print("\n[EXTERNAL] modeling 소스에서 확인한 심볼값 유지 여부")
    spec = refs.get("modeling_sourced_symbols") or {}
    ok = bad = 0
    for sym, rec in spec.items():
        if not isinstance(rec, dict) or "value" not in rec:
            continue
        for name in rec.get("models") or []:
            st = os.path.join(MODELS, name, "structure.yaml")
            if not os.path.exists(st):
                continue
            syms = (yaml.safe_load(open(st, encoding="utf-8")) or {}).get("symbols") or {}
            if syms.get(sym) == rec["value"]:
                ok += 1
            else:
                bad += 1
                fail(f"{name}: {sym}={syms.get(sym)} 인데 modeling 소스 확인값은 {rec['value']} "
                     f"({str(rec.get('source', ''))[:60]})")
    print(f"   {ok}건 일치" + (f", {bad}건 불일치" if bad else ""))


def check_review_ledger():
    """③ 자유 평가가 지금 산출물에 대해 실제로 수행됐는가 (README 검증 플로우).

    The packet is generated every run; performing the review is not automatic. Nothing recorded
    whether it happened, so an unreviewed model was indistinguishable from one reviewed this
    morning, and a review taken before a labelling change looked current. Same reason C17 exists:
    assert the human step happened rather than assuming it.
    """
    print("\n[REVIEW] ③ 자유 평가 수행 기록")
    import review_ledger
    s = review_ledger.summary(MODELS)
    c = s["counts"]
    print(f"   최신 {c['PASS']} / 만료 {c['STALE']} / 미수행 {c['NONE']}")
    stale = [n for n, (st, _d) in s["models"].items() if st == "STALE"]
    none = [n for n, (st, _d) in s["models"].items() if st == "NONE"]
    if stale:
        warn(f"③ 자유 평가가 만료된 모델 {len(stale)}개 (산출물이 검토 이후 바뀜): "
             f"{', '.join(stale[:4])}{' 외' if len(stale) > 4 else ''}")
    if none:
        warn(f"③ 자유 평가 기록이 없는 모델 {len(none)}개: "
             f"{', '.join(none[:4])}{' 외' if len(none) > 4 else ''}")


def check_baseline(fleet, update):
    print("\n[BASELINE] 이전 기준 대비 퇴행 검사")
    cur = {n: {"bare": m["bare"], "unresolved": m["unresolved"],
               "unknown_syms": m["unknown_syms"], "c_fail": m["c_fail"],
               "flow_ambig": m["flow_ambig"], "heur": m["heur"],
               "ident_incons": m["ident_incons"],
               "reshape_incons": m["reshape_incons"],
               "matmul_compose": m["matmul_compose"]}
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
        for key in ("bare", "unresolved", "unknown_syms", "c_fail", "flow_ambig",
                    "heur", "ident_incons", "reshape_incons", "matmul_compose"):
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
        refs = yaml.safe_load(open(REFS, encoding="utf-8")) if os.path.exists(REFS) else {}
        check_documented_literals(refs)
        check_modeling_sourced(refs)
        check_review_ledger()
        check_baseline(fleet, args.update_baseline)

    print("\n" + "=" * 72)
    print(f"FAIL {len(failures)}건 / WARN {len(warnings)}건")
    for f in failures:
        print(f"  - {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
