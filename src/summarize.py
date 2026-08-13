"""Generates the two higher-level deliverables described in 01-main.md section 11:
  - structure.yaml -- a layer/block-level architecture rollup in common symbols
    (rules/symbols.yaml), derived purely from traced rows + config (deterministic).
  - model_summary.md -- human-readable model summary + extraction methodology +
    cross-reference sources. The architecture part is generated from structure.yaml
    (deterministic); the "sources consulted" section is a Tier 2 research artifact
    (02-new-module-handling.md) and is passed in by whoever is running the pipeline
    (a coding agent doing live research, or a human) -- this module only formats it.
"""
import hashlib
import json
import math
import os
import re

import yaml

_SYMBOLS_PATH = os.path.join(os.path.dirname(__file__), "..", "rules", "symbols.yaml")


def load_symbols(path: str = _SYMBOLS_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _per_layer_scalar(v):
    """Fold a PER-LAYER config field to the one number it holds, or None if the layers disagree.

    Some vendors state a width once per decoder layer instead of once per model: Hunyuan-A13B
    writes `moe_topk: [8, 8, ...]` and `moe_intermediate_size: [3072, ...]`, one entry per layer.
    A list is not an int, so the symbol simply did not resolve -- `k` stayed unknown and the
    routed-slot axis (T x top-k) was taken over by `n_kv`, which happened to be 8 as well. 64 rows
    of self-contradiction, and nothing about the values could show it.

    Only a list whose entries AGREE is folded. If a checkpoint ever varies the width per layer, one
    number would be a lie, and leaving the symbol unresolved is the honest outcome.
    """
    if not isinstance(v, (list, tuple)):
        return v
    nums = [x for x in v if isinstance(x, int) and not isinstance(x, bool)]
    if len(nums) != len(v):
        # NOT a per-layer numeric field. `layer_types` is a list of STRINGS
        # (['sliding_attention', 'full_attention', ...]) and folding it returned None, which wiped
        # `layer_sched` out of every model that has a layer schedule -- DeepSeek-V4, gpt-oss and
        # Llama-4 all reported "해당 없음" for a field their config plainly carries. Found by an
        # outside review, 2026-08-12. A value this function cannot fold is returned untouched.
        return v
    return nums[0] if nums and len(set(nums)) == 1 else None


def _first_attr(cfg, aliases, default=None):
    for a in aliases:
        if hasattr(cfg, a):
            return _per_layer_scalar(getattr(cfg, a))
    return default


def alias_fields(symbols: dict | None = None) -> dict:
    """{symbol: EVERY config field it may stand for}, config-independent.

    `resolved_fields` answers which field a symbol read *in this checkpoint*; this answers which
    fields the rules allow it to mean at all. The membership check needs the second: `d_moe` names
    both `moe_intermediate_size` and `shared_expert_intermediate_size`, and a shared-expert
    projection that reads only the latter is correctly labelled `d_moe`.
    """
    symbols = symbols if symbols is not None else load_symbols()
    out = {}
    for sym, spec in symbols.items():
        flds = set(spec.get("aliases") or [])
        frm = spec.get("from")
        if isinstance(frm, dict) and frm.get("field"):
            flds.add(frm["field"])
        if flds:
            out[sym] = flds
    return out


def resolved_fields(cfg, symbols: dict | None = None) -> dict:
    """{symbol: the config field its value actually came from}.

    resolve_symbols returns values; this returns their provenance, so `src/source_check.py` can
    assert that field really is defined by the model's own configuration class. A symbol resolved
    from a field the config does not declare means an alias matched something incidental.
    """
    symbols = symbols or load_symbols()
    out = {}
    for sym, spec in symbols.items():
        for alias in (spec.get("aliases") or []):
            if _first_attr(cfg, [alias]) is not None:
                out[sym] = alias
                break
        else:
            frm = spec.get("from")
            if isinstance(frm, dict) and _first_attr(cfg, [frm.get("field")]) is not None:
                out[sym] = frm.get("field")
    return out


def resolve_symbols(cfg, symbols: dict | None = None) -> dict:
    """Returns {symbol: value}. A symbol missing from the config (None) means no known
    alias matched -- surface it, don't guess (see rules/symbols.yaml header note)."""
    symbols = symbols or load_symbols()
    # plain dict (insertion-ordered on py3.7+) -- yaml.safe_dump cannot represent OrderedDict,
    # and structure.yaml is a deliverable that must serialize cleanly.
    out = {}
    for sym, spec in symbols.items():
        val = _first_attr(cfg, spec.get("aliases", []))
        if val is None and isinstance(spec.get("from"), dict):
            # value nested in a config dict (DeepSeek-V4 compress_rates[<layer type>])
            holder = _first_attr(cfg, [spec["from"].get("field")])
            if isinstance(holder, dict):
                val = holder.get(spec["from"].get("key"))
        out[sym] = val

    # Deterministic fallbacks. These are NOT guesses: each is the same derivation the modeling
    # code itself does when the config omits the field, so leaving them null would flag a real
    # architectural fact as an unknown (Tier 2 noise). Anything not derivable this way stays None.
    if out.get("d_head") is None and out.get("d_model") and out.get("n_h"):
        out["d_head"] = out["d_model"] // out["n_h"]          # no head_dim field => d_model/n_h
    # GPT-2 leaves the FFN width out of the config and lets the model compute it. Without this
    # d_ff stayed null and the 4x width had to be invented by the arithmetic tail, which named it
    # `4*d_model` -- true, but the axis has a real name. Rule copied from the modeling code:
    #   transformers models/gpt2/modeling_gpt2.py:250
    #   inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size
    if out.get("d_ff") is None and out.get("d_model") and getattr(cfg, "model_type", None) == "gpt2":
        out["d_ff"] = 4 * out["d_model"]
    # Falcon: `num_kv_heads` is NOT the effective KV head count -- a separate `multi_query` flag
    # overrides it to 1. Taking the field at face value made falcon-7b report MHA with n_kv=71
    # when it is MQA with n_kv=1, and inflated its KV-cache card 71x (568 KiB vs 8 KiB).
    # Rule copied from transformers models/falcon/modeling_falcon.py:
    #   num_kv_heads = config.num_kv_heads if (new_decoder_architecture or not multi_query) else 1
    if _first_attr(cfg, ["multi_query"]) and not _first_attr(cfg, ["new_decoder_architecture"]):
        out["n_kv"] = 1
    if out.get("n_kv") is None and out.get("n_h"):
        out["n_kv"] = out["n_h"]                              # no GQA field => MHA (n_kv == n_h)
    # Attention sink: an extra learned logit column appended to the softmax denominator. There is
    # no config field for it -- it lives only in the modeling code -- so the count has to come from
    # the architecture identity, exactly like Falcon's multi_query override above.
    # Source: transformers models/gpt_oss/modeling_gpt_oss.py, GptOssAttention.sinks (one learned
    # scalar per head, concatenated as ONE extra column onto the scores) + the OpenAI gpt-oss model
    # card. See rules/structures/attention/attention-sink.md.
    # Deliberately NOT extended to deepseek_v4, which also has a sink: its score widths are already
    # spelled out inline (`T + T/m_csa + 1`) in derived_dims.yaml, and adding n_sink there would let
    # the generic `T + 1 + n_sink` rule fire on a V4 axis that is not a score width.
    if getattr(cfg, "model_type", None) in ("gpt_oss",):
        out["n_sink"] = 1
    if out.get("E"):
        # A MoE config with no shared-expert field has no shared expert -- that is 0, not unknown.
        if out.get("E_shared") is None:
            out["E_shared"] = 0
        # Pure-MoE stacks (gpt-oss, OLMoE) size their experts with plain `intermediate_size`
        # rather than a separate `moe_intermediate_size`. Read that field DIRECTLY rather than
        # copying d_ff: Llama-4 carries both widths (`intermediate_size_mlp`=16384 for the dense
        # FFN, `intermediate_size`=8192 for the experts), and d_ff now resolves to the former, so
        # copying it would size the experts at twice their real width.
        # The value stays reported as d_ff here -- `intermediate_size` really is that number -- but
        # see symbolic_shape._config_values, which drops it as an AXIS NAME when the two coincide.
        if out.get("d_moe") is None:
            out["d_moe"] = _first_attr(cfg, ["intermediate_size"]) or out.get("d_ff")
    return out


def _layer_clusters(rows: list[dict]):
    """Roll consecutive layers with an identical op-type signature into one range,
    e.g. layers 1-26 all 'MLA + MoE' collapse to a single entry."""
    sigs = {}
    for layer_idx in sorted({r["layer_idx"] for r in rows if r.get("layer_idx") is not None}):
        seq = tuple(r["op_type"] for r in rows if r.get("layer_idx") == layer_idx)
        sigs[layer_idx] = hashlib.sha1(json.dumps(seq).encode()).hexdigest()[:10]

    clusters, cur_start, cur_sig = [], None, None
    for idx in sorted(sigs):
        if cur_sig is None:
            cur_start, cur_sig = idx, sigs[idx]
        elif sigs[idx] != cur_sig:
            clusters.append((cur_start, idx - 1, cur_sig))
            cur_start, cur_sig = idx, sigs[idx]
    if cur_sig is not None:
        clusters.append((cur_start, max(sigs), cur_sig))
    return clusters


def _dominant_blocks(rows, lo, hi):
    blocks = sorted(
        {r["block"] for r in rows if r.get("layer_idx") is not None and lo <= r["layer_idx"] <= hi
         and r.get("block")}
    )
    return blocks


def _trace_shared_expert_count(rows: list[dict]) -> int | None:
    """Distinct shared-expert submodule names seen in the trace (e.g. 'shared_expert',
    'shared_expert_0'), or None if none appear at all.

    Exists because some architectures hardcode a shared expert directly in code with NO config
    field to read at all -- Llama-4: `self.shared_expert = Llama4TextMLP(config)`
    (transformers models/llama4/modeling_llama4.py), unconditional, no count field. Our
    config-based fallback ("no shared-expert field => E_shared=0", see resolve_symbols) is
    wrong for exactly this case: it contradicted our OWN trace, which has 120 distinct
    `shared_expert` module paths for Llama-4-Maverick. Found via external review, 2026-07-29 --
    the fix trusts the trace over an absence-of-config-field assumption, matching how C8 already
    trusts the trace over a possibly-vestigial `num_experts`-like config field."""
    import re as _re
    names = set()
    for r in rows:
        m = _re.search(r"\.(shared_expert(?:s)?(?:_\d+)?)(?:\.|$)", r.get("module_path") or "")
        if m:
            names.add(m.group(1))
    return len(names) or None


def build_structure(rows: list[dict], cfg, model_id: str, revision: str) -> dict:
    symbols = resolve_symbols(cfg)
    if symbols.get("E") and symbols.get("E_shared") == 0:
        trace_n = _trace_shared_expert_count(rows)
        if trace_n:
            symbols["E_shared"] = trace_n
    clusters = _layer_clusters(rows)
    layers = [
        {"range": [lo, hi], "blocks": _dominant_blocks(rows, lo, hi)}
        for lo, hi, _sig in clusters
    ]
    return {
        "model_id": model_id,
        "revision": revision,
        "symbols": symbols,
        "layers": layers,
        "note": (
            "symbols의 값 중 null은 rules/symbols.yaml에 등록된 별칭으로 config에서 "
            "찾지 못한 것 -- 임의 채움 금지(01-main.md P1). 02-new-module-handling.md "
            "Tier 2로 확인 후 별칭을 추가할 것."
        ),
    }


# How much of a model's labelling rests on a REGISTERED rule vs on the arithmetic tail. Until
# 2026-08-05 nothing recorded this, so "every axis has a name" and "every name is derived" were
# indistinguishable -- and the tail is known to invent names that hold at the traced seq_len and
# nowhere else (see 01-main.md, heuristic labels). symbolic_shape.dim() now tallies which branch
# answered; this turns that tally into a published, gate-checkable number.
_STRONG = ("scoped_symbol", "scoped_formula", "derived_formula", "plain_symbol", "runtime")
_WEAK = ("out_of_scope_symbol", "reused_symbol")


def label_provenance(resolver) -> dict:
    """{rule: axes} plus the derived/weak/bare split, from resolver.stats."""
    stats = {k: v for k, v in getattr(resolver, "stats", {}).items() if k != "passthrough"}
    total = sum(stats.values())
    heur = sum(v for k, v in stats.items() if k.startswith("heur"))
    out = {
        "total_axes": total,
        "by_rule": dict(sorted(stats.items(), key=lambda kv: -kv[1])),
        "derived": sum(stats.get(k, 0) for k in _STRONG),
        "weak": sum(stats.get(k, 0) for k in _WEAK),
        "heuristic": heur,
        "bare": stats.get("bare", 0),
    }
    out["heuristic_pct"] = round(100.0 * heur / total, 2) if total else 0.0
    # Where the fabricated names land, so a reviewer has somewhere to look.
    #
    # Filter to heuristics BEFORE taking the top N. Slicing first and filtering after looks
    # equivalent and is not: `weak` also holds every `bare` axis, and bare outnumbers heuristics
    # by an order of magnitude, so the top 12 was all bare and the filter emptied the list. The
    # review request is built from this field, so a model could carry 10,574 invented names
    # (Qwen3-Next did) and still produce an empty request -- the review would be skipped exactly
    # where it was most needed. Found while onboarding Qwen3.5, 2026-08-09.
    heur_only = [kv for kv in getattr(resolver, "weak", {}).items() if kv[0][0].startswith("heur")]
    out["heuristic_examples"] = [
        {"rule": k[0], "module": k[1], "label": k[2], "axes": v}
        for k, v in sorted(heur_only, key=lambda kv: -kv[1])[:12]
    ]
    return out


def write_structure(model_dir: str, structure: dict, fmt: str = "yaml") -> str:
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, f"structure.{fmt}")
    with open(path, "w", encoding="utf-8") as f:
        if fmt == "yaml":
            yaml.safe_dump(structure, f, allow_unicode=True, sort_keys=False)
        else:
            json.dump(structure, f, indent=2, ensure_ascii=False)
    return path


_DERIVED_PATH = os.path.join(os.path.dirname(__file__), "..", "rules", "derived_dims.yaml")


def load_derived_dims(path: str = _DERIVED_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _derived_vars(cfg, spec: dict) -> dict:
    """Scalars that are not symbols themselves but appear in derived-dim formulas: values nested
    inside a config dict (compress_rates[...]) or float coefficients (partial_rotary_factor)."""
    out = {}
    for name, d in (spec.get("vars") or {}).items():
        src = d.get("from") or {}
        val = None
        if src.get("field"):
            holder = _first_attr(cfg, [src["field"]])
            if isinstance(holder, dict):
                val = holder.get(src.get("key"))
        elif src.get("aliases"):
            val = _first_attr(cfg, src["aliases"])
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            out[name] = val
    return out


def _eval_namespace(S: dict, cfg=None, seq_len=None, spec: dict | None = None) -> dict:
    """Symbols + derived_dims `vars` + T + a few safe functions, for evaluating rule expressions."""
    spec = spec if spec is not None else load_derived_dims()
    ns = {k: v for k, v in (S or {}).items() if isinstance(v, int) and not isinstance(v, bool)}
    if cfg is not None:
        ns.update(_derived_vars(cfg, spec))
    if seq_len:
        ns["T"] = int(seq_len)
    ns.update(ceil=math.ceil, round=round, min=min, max=max)
    return ns


def _eval_rule(rule: dict, ns: dict):
    """Rule value, or None when its symbols are missing for this model (rule does not apply)."""
    try:
        # Trusted local rules file; no builtins exposed. A missing symbol raises NameError
        # and the rule is skipped -- exactly the "does not apply to this model" case.
        val = eval(rule["expr"], {"__builtins__": {}}, ns)  # noqa: S307
    except Exception:
        return None
    if isinstance(val, float) and val.is_integer():
        val = int(val)
    if isinstance(val, int) and not isinstance(val, bool) and val > 1:
        return val
    return None


def _known_composites(S: dict, cfg=None, seq_len=None, spec: dict | None = None) -> dict:
    """{concrete value: named expression} for dims that legitimately appear as literals in shapes
    (symbol products the symbolizer intentionally does NOT auto-synthesize, to avoid coincidental
    labels -- see symbolic_shape.py).

    Rules live in rules/derived_dims.yaml, each with a source note. We EVALUATE each rule from
    this model's symbols and match by exact value, so this is not a blind factorization: it
    confirms 'this literal equals that architectural quantity' (P1). A rule whose symbols are
    missing for this model simply does not fire. First matching rule wins, so the file is ordered
    most-specific first.

    Scoped rules (a `scope` regex on the rule) are folded in here too, since the legend is keyed
    by value only; `find_literal_dims` records which modules each value appeared in, which is what
    disambiguates a value that two scoped rules both claim."""
    spec = spec if spec is not None else load_derived_dims()
    ns = _eval_namespace(S, cfg, seq_len, spec)
    out = {}
    for rule in (spec.get("rules") or []):
        val = _eval_rule(rule, ns)
        if val is not None:
            out.setdefault(val, rule["name"])
    return out


def _degenerate_product(expr, ns) -> bool:
    """True when `expr` is a product and one of its factors resolves to 1 (see derived_symbols)."""
    import re as _re2
    if not expr or "*" not in str(expr) or "+" in str(expr) or "-" in str(expr):
        return False
    for tok in _re2.split(r"[*/]", str(expr)):
        tok = tok.strip().strip("()")
        if not tok or tok.isdigit():
            continue
        if ns.get(tok) == 1:
            return True
    return False


def derived_symbols(symbols: dict, cfg=None, seq_len=None, spec: dict | None = None):
    """(global, scoped) maps of {value: compact symbolic expression} -- the `sym` field of the
    same rules that _known_composites reads for its prose `name`. Used by symbolic_shape to
    render a verified formula (e.g. "T+T/m_csa") in the shape cell instead of a bare 2560.

    `scoped` is [(compiled_regex, {value: sym})] for rules that declare a `scope`. A projection
    width like `2*n_kv*d_head` is only meaningful inside attention; without the scope it also
    matched 2048 inside Llama-4's expert block, where the real quantity is E*T."""
    import re as _re
    spec = spec if spec is not None else load_derived_dims()
    ns = _eval_namespace(symbols, cfg, seq_len, spec)
    glob, scoped = {}, {}
    for rule in (spec.get("rules") or []):
        val = _eval_rule(rule, ns)
        if val is None or not rule.get("sym"):
            continue
        # A product with a unit factor is not a distinct dimension -- it IS the other factor, and
        # the plain symbol is the better name. Llama-4 Maverick routes top-1, so `k*T` evaluates
        # to T there and, being a scoped formula, outranked the plain `T`: 48 sequence axes came
        # out `k*T` and flow_ambig went 96 -> 216. A rule is only meaningful where every symbol
        # it multiplies is genuinely > 1.
        if _degenerate_product(rule.get("expr"), ns):
            continue
        # `unless_equals` disarms a formula in the models where it is indistinguishable from a
        # plain symbol. A scoped formula outranks an unscoped plain symbol, so when the two hold
        # the same number the formula wins every axis of that size in the module -- including ones
        # it does not describe. OLMoE's fused gate+up width equals hidden_size, and letting the
        # formula through renamed the residual stream flowing past the experts (flow_ambig
        # 32 -> 64). Where the values differ the formula is unambiguous and fires normally.
        if any(ns.get(other) == val for other in (rule.get("unless_equals") or [])):
            continue
        if rule.get("scope"):
            scoped.setdefault(rule["scope"], {}).setdefault(val, rule["sym"])
        else:
            glob.setdefault(val, rule["sym"])
    return glob, [(_re.compile(p), m) for p, m in scoped.items()]


def find_literal_dims(rows: list[dict], symbols: dict, resolver=None, min_unnamed: int = 128,
                      cfg=None, seq_len=None) -> list[dict]:
    """Dimensions that stayed literal integers after symbolization (fixed, config-derived,
    input-invariant -- the symbolizer left them as honest numbers rather than guess a symbol).
    Returns [{value, expr, where}] so a model's output can explain each 'mystery number'; `expr`
    is None when no rule in rules/derived_dims.yaml explains it, which is the Tier 3 escalation
    signal (see render_model_summary).
    Small values are noise (group sizes, head counts) unless we can name them, so a bare literal
    is reported only if it has a known-composite expression or is >= `min_unnamed`.
    `resolver` symbolizes concrete rows (fresh run); omit it for already-symbolic rows (regen)."""
    comp = _known_composites(symbols, cfg=cfg, seq_len=seq_len)
    # A value that equals a dim symbol is NOT an unreduced literal -- the current symbolizer would
    # render it as that symbol. Listing it here would also invite a wrong label, because several
    # unrelated quantities can share one small integer (DeepSeek-V4: g_o = 16 = T/m_hca = n_hc^2).
    # Regenerated rows were symbolized by an older symbol set, so filter against the CURRENT one.
    _spec = load_symbols()
    symbolic_vals = {v for n, v in (symbols or {}).items()
                     if (_spec.get(n) or {}).get("dim") and isinstance(v, int)
                     and not isinstance(v, bool)}
    # Pair each CONCRETE dim with how it renders, and keep it for the legend only when the render
    # is either a bare integer (genuinely unreduced) or a verified derived formula -- a dim that
    # now shows as "T+T/m_csa" still needs its prose meaning listed. Anything that renders as a
    # plain symbol (d_model, T, T+1, ...) is self-explanatory and must NOT be listed.
    # Without a resolver the rows are already symbolic (legacy path): only bare ints are visible.
    def pairs_of(shape, module_path, is_weight=False):
        if not shape:
            return []
        rendered = (resolver(shape, module_path, is_weight=is_weight) if resolver else shape)
        return list(zip(shape, rendered))

    found = {}
    for r in rows:
        mp = r.get("module_path")
        leaf = (mp or "").rsplit(".", 1)[-1] or "(root)"
        # (shape, is_weight) pairs -- weights must be rendered under the same no-T invariant the
        # tables use, or the legend would disagree with the table it is supposed to explain.
        operands = [(s, False) for s in (r.get("input_shape") or [])]
        operands += [(s, False) for s in (r.get("output_shape") or [])]
        if r.get("weight_shape"):
            operands.append((r["weight_shape"], True))
        for shp, is_w in operands:
            for concrete, shown in pairs_of(shp, mp, is_weight=is_w):
                if not isinstance(concrete, int) or isinstance(concrete, bool) or concrete <= 1:
                    continue
                if str(shown).isdigit() or concrete in comp:
                    found.setdefault(concrete, set()).add(leaf)
    return [{"value": v, "expr": comp.get(v), "where": sorted(found[v])}
            for v in sorted(found)
            if v not in symbolic_vals and (comp.get(v) or v >= min_unnamed)]


# ---- architecture profile (Raschka-gallery style, derived deterministically) --------------
# Everything here comes from config + the trace (op_types, params) -- ground truth, no guessing
# (P1). Facts we cannot derive are marked "?" rather than filled in.

def _hnum(x):
    if not isinstance(x, int):
        return str(x)
    for u, d in (("B", 1_000_000_000), ("M", 1_000_000), ("K", 1_000)):
        if x >= d:
            return f"{x / d:.2f}".rstrip("0").rstrip(".") + u
    return str(x)


def _fmt_symval(val):
    """Human-facing symbol value: collapse a uniform schedule list (e.g. 24x full_attention)
    and cap long heterogeneous lists so the summary stays readable."""
    if val is None:
        return "_(미확인 -- config 별칭 없음, Tier 2 대상)_"
    if isinstance(val, (list, tuple)):
        if len(val) > 1 and len(set(map(str, val))) == 1:
            return f"{len(val)}× {val[0]}"
        if len(val) > 6:
            from collections import Counter
            c = Counter(map(str, val))
            return ", ".join(f"{n}× {t}" for t, n in c.most_common()) + f" (총 {len(val)}층)"
    return str(val)


def derive_architecture(cfg, rows, structure, scale: dict | None = None) -> dict:
    """Human-facing architecture facts derived from config + trace. Attention family, position
    encoding, FFN/MoE, normalization, activation, KV-cache size, param scale."""
    S = structure["symbols"]
    ops = {r.get("op_type") for r in rows}
    params_join = " ".join(p for r in rows for p in r.get("params", []))
    # Layer indices that actually run softmax attention, detected by the presence of a QUERY
    # projection. Both the attention-family label and the KV-cache card key off this rather than
    # off config fields or a layer-type-name whitelist.
    #
    # Why the query projection and not the module name: module naming is wildly inconsistent and
    # every name-based rule tried here broke something. Nemotron-3-Nano calls *every* block
    # `mixer` (attention, Mamba and MLP alike) so no attention name exists at all; Falcon uses
    # `self_attention` while most use `self_attn`; and a substring match on "attn" wrongly counts
    # Qwen3-Next's `linear_attn` (DeltaNet), which keeps a recurrent state and no KV cache. A Q
    # projection, by contrast, exists exactly where softmax attention does. Verified to reproduce
    # every independently-known count: Qwen3-Next 12/48, Zamba2 6/38, Nemotron 4/42, xLSTM 0/32,
    # Falcon 32/32 (audit, 2026-07-30).
    _QPROJ = re.compile(r"\.(q_proj|q_a_proj|q_b_proj|query_key_value|qkv_proj|Wqkv|c_attn)$")
    attn_layers = {r.get("layer_idx") for r in rows
                   if r.get("layer_idx") is not None and _QPROJ.search(r.get("module_path") or "")}
    d_model, n_h = S.get("d_model"), S.get("n_h")
    # MHA models omit num_key_value_heads -> kv defaults to n_h (same convention as C7)
    n_kv = S.get("n_kv") or n_h
    d_head = S.get("d_head") or (d_model // n_h if d_model and n_h else None)
    L = S.get("L")

    kv_lora = _first_attr(cfg, ["kv_lora_rank"])
    d_nope, d_rope, d_v = S.get("d_nope"), S.get("d_rope"), S.get("d_v")
    if kv_lora:
        q_lora = _first_attr(cfg, ["q_lora_rank"])
        attn = f"MLA — KV latent compression (kv_lora_rank={kv_lora}" + (
            f", q_lora_rank={q_lora}" if q_lora else "") + ")"
        # MLA has no single head_dim: q/k = nope+rope, v is separate -- spell it out so the
        # bare config head_dim (which equals qk_rope_head_dim) can't mislead.
        if d_nope and d_rope:
            attn += (f"; 헤드 q/k = nope({d_nope})+rope({d_rope})={d_nope + d_rope}"
                     + (f", v={d_v}" if d_v else "") + (f", n_h={n_h}" if n_h else ""))
    elif n_h and n_kv:
        if n_kv == n_h:
            attn = f"MHA — {n_h} heads (no GQA/MQA)"
        elif n_kv == 1:
            attn = f"MQA — {n_h} query heads, 1 kv head"
        else:
            attn = f"GQA — {n_h} query : {n_kv} kv heads (repeat {n_h // n_kv})"
        if d_head:
            attn += f", d_head={d_head}"
    else:
        attn = "? (no attention-head fields on config — may be attention-free, e.g. SSM/xLSTM)"
    w_local = S.get("w_local")
    if w_local:
        attn += f"; sliding window {w_local}"
        # Only claim a local/global split when the schedule actually names sliding layers.
        # A model can carry `sliding_window` + a `layer_types` schedule and still apply the local
        # window on EVERY layer (DeepSeek-V4: the schedule selects the compressor, not the window).
        if "sliding_attention" in (getattr(cfg, "layer_types", None) or []):
            attn += " on part of layers (hybrid local/global)"
    # Attention sink is part of this model's identity (it is what lets gpt-oss keep a 128-token
    # window without the usual quality loss), but it never shows up as a symbol of its own -- only
    # as a +1 on the score width. Without saying so here, the card gave no sign it existed
    # (external review, 2026-07-30). See rules/structures/attention/attention-sink.md.
    if S.get("n_sink"):
        attn += (f"; attention sink ({S['n_sink']}개 학습형 로짓 열이 softmax 분모에 추가 — "
                 "KV는 늘지 않고 score 폭만 +1)")

    has_rope = ("cos" in ops and "sin" in ops) or bool(_first_attr(cfg, ["rope_theta", "rope_scaling"]))
    learned_pos = any(t in params_join for t in ("wpe", "embed_positions", "position_embeddings"))
    if has_rope:
        rope_scaling = _first_attr(cfg, ["rope_scaling"])
        theta = _first_attr(cfg, ["rope_theta"])
        if not theta and isinstance(rope_scaling, dict):
            theta = rope_scaling.get("rope_theta")
        pos = "RoPE" + (f" (θ={theta})" if theta else "")
        rtype = rope_scaling.get("rope_type") or rope_scaling.get("type") if isinstance(rope_scaling, dict) else None
        if rtype and rtype != "default":
            pos += f", {rtype} scaling"
    elif learned_pos:
        pos = "learned absolute position embeddings"
    else:
        pos = "none observed (NoPE, or position handled implicitly)"

    E, k, e_shared, d_moe = S.get("E"), S.get("k"), S.get("E_shared"), S.get("d_moe")
    act = "SwiGLU (silu·gate)" if "silu" in ops else ("GELU" if "gelu" in ops else "?")
    # A `num_experts`-like config field does NOT prove the model is MoE -- it can be vestigial.
    # Nemotron-3-Nano declares n_routed_experts=8 yet traces ZERO expert params/ops (C8 WARNs
    # about exactly this). The DECODER TYPE card already gated on trace evidence, but this FFN
    # line did not, so one summary printed "DECODER TYPE | Dense" and "FFN | MoE — 8 routed
    # experts" two rows apart -- a self-contradiction inside a single document (found while
    # re-auditing the external review, 2026-07-29). Gate both on the same trace-verified signal.
    real_moe = bool(E) and ("expert" in params_join or "grouped_matmul" in ops)
    if real_moe:
        ffn = f"MoE — {E} routed experts, top-{k}"
        if e_shared:
            ffn += f" + {e_shared} shared"
        ffn += f", expert intermediate {d_moe or S.get('d_ff')}, {act}"
        if "grouped_matmul" in ops:
            ffn += " [grouped_mm]"
    else:
        ffn = f"dense FFN — intermediate {S.get('d_ff')}, {act}"
        if E:
            ffn += (f"  _(config는 {E} expert를 선언하지만 트레이스에 expert 연산·파라미터가 "
                    f"전혀 없음 — vestigial 필드, C8 WARN 참고)_")

    # RMSNorm often traces as decomposed pow/mean/rsqrt/mul (no single 'rmsnorm' op), so rsqrt
    # is the reliable signal; native_layer_norm -> LayerNorm. (Llama names modules *_layernorm
    # but they are RMSNorm classes -- trust the ops, not the name.)
    norms = []
    if "rmsnorm" in ops or "rsqrt" in ops:
        norms.append("RMSNorm")
    if "layernorm" in ops:
        norms.append("LayerNorm")
    norm = " + ".join(norms) or "?"
    if "sdpa" in ops:
        attn_kernel = "sdpa (scaled_dot_product_attention)"
    elif "softmax" in ops:
        attn_kernel = "eager (explicit softmax)"
    else:
        attn_kernel = "? (no softmax/sdpa op — non-softmax attention?)"

    if kv_lora:
        kv_cache = f"compressed MLA latent ≈ kv_lora_rank={kv_lora} (+decoupled RoPE dim) / token / layer"
    elif n_kv and d_head:
        per = 2 * n_kv * d_head
        kv_cache = (f"2·n_kv·d_head = 2·{n_kv}·{d_head} = {per} elems / token / layer"
                    + (f"; all {L} layers ⇒ {per * L} / token" if L else ""))
    else:
        kv_cache = "? (no standard KV cache — attention-free block?)"

    # Context source note: ctx = config max_position_embeddings. With RoPE scaling (e.g. YaRN) the
    # config max is the scaled table size; a vendor's *advertised* usable context can be a rounded,
    # different figure (e.g. gallery lists DeepSeek-V3 as 128K while config max is 163,840). We
    # report the config value (P1, traceable) and note the source so the discrepancy is explained.
    context_note = "config max_position_embeddings"
    _rscale = _first_attr(cfg, ["rope_scaling"])
    if isinstance(_rscale, dict) and _rscale.get("original_max_position_embeddings"):
        _rt = _rscale.get("rope_type") or _rscale.get("type") or "scaled"
        context_note += (f"; {_rt} 스케일(원본 {_rscale['original_max_position_embeddings']}"
                         + (f"×{_rscale['factor']}" if _rscale.get("factor") else "") +
                         ") — 벤더 광고 컨텍스트와 다를 수 있음")

    scale = scale or {}
    total, active = scale.get("total_params"), scale.get("active_params")

    # ---- Raschka-gallery card fields (all derived from config+trace; P1) ----
    # Whether the model attends at all is decided by the TRACE, not by config fields happening to
    # resolve. Config head-count names are reused across families: xLSTM stores `num_heads`=8 /
    # `head_dim`=512 for its mLSTM heads, which is nothing to do with attention, yet that briefly
    # made an attention-free model report "MHA" with a 512 KiB KV cache (audit, 2026-07-30). Same
    # trust-the-trace principle as `real_moe` and C8.
    real_attention = bool(attn_layers) or "sdpa" in ops
    attn_short = ("attention-free" if not real_attention else
                  "MLA" if kv_lora else
                  "MHA" if (n_h and n_kv and n_kv == n_h) else
                  "MQA" if (n_h and n_kv == 1) else
                  "GQA" if (n_h and n_kv and n_kv < n_h) else
                  ("attention-free" if not n_h else "?"))
    # `attn_short` may get a compressed-attention suffix below (DeepSeek-V4); keep the bare family
    # label for the "Related concepts" list so it stays a clean concept name.
    attn_family = attn_short
    # real_moe (trace-verified, not config-declared) is computed once with the FFN description
    # above so the two can never disagree -- see the comment there.
    decoder_type = "Sparse MoE" if real_moe else "Dense"

    dd = cfg.to_dict() if cfg is not None else {}
    sched = dd.get("layer_types") or dd.get("layers_block_type")
    if isinstance(sched, list) and sched:
        from collections import Counter
        cc = Counter(sched)
        # Use the schedule's OWN terms, verbatim, and name the attention family once at the end.
        # Rewriting only `full_attention` into the family (GQA/MLA/...) mixed two vocabularies in
        # one line -- Llama-4 read "36× chunked_attention, 12× GQA" while the `layer_sched` row two
        # tables down read "36× chunked_attention, 12× full_attention". Same file, two stories; an
        # outside review counted three (2026-08-12). Deriving both from `sched` keeps them equal.
        layer_mix = ", ".join(f"{n}× {t}" for t, n in cc.most_common())
        if any(t == "full_attention" for t in cc) and attn_short and attn_short != "?":
            layer_mix += f"  (attention: {attn_short})"
    elif L:
        layer_mix = f"{L}× {attn_short}"
    else:
        layer_mix = "?"
    fk = dd.get("first_k_dense_replace") or dd.get("n_dense_layers")
    if real_moe and L:
        layer_mix += f"  (FFN: {fk} dense + {L - fk} MoE)" if fk else f"  (FFN: {L}× MoE)"

    # KV cache per token in BF16 (2 bytes). Only layers that append to a *growing* cache count --
    # in a hybrid (Mamba/DeltaNet/mlp) the recurrent/FFN layers hold no KV, so count attention
    # layers, not all L. MLA caches the COMPRESSED latent (c_kv + decoupled-rope), not full KV.
    # Bands + formulas follow the gallery's stated method:
    # https://sebastianraschka.com/llm-architecture-gallery/kv-cache-calculations/
    #   standard (separate K and V): 4·n_kv·d_head bytes/layer/token
    #   unified K==V (one tensor):   2·n_kv·d_head
    #   MLA:                         2·(kv_lora_rank + qk_rope_head_dim)
    if attn_layers:
        n_attn = len(attn_layers)
    else:
        attn_layer_types = {"full_attention", "sliding_attention", "chunked_attention"}
        n_attn = (sum(1 for t in sched if t in attn_layer_types)
                  if isinstance(sched, list) and sched else L)

    def _kv_band(kib):
        # Gallery thresholds (upper-inclusive): 0 / 24 / 72 / 160 / 300.
        if kib <= 0:
            return "No cache"
        for hi, name in ((24, "Very low"), (72, "Low"), (160, "Moderate"), (300, "High")):
            if kib <= hi:
                return name
        return "Very high"

    # DeepSeek-V4-style block-compressed attention (CSA/HCA): what grows with context is not the
    # per-token KV but one COMPRESSED entry per `m` tokens, so the amortised cost per layer is
    # d_head/m elems. K and V are the same tensor (`update(kv, kv, ...)`) => 1 tensor, not 2.
    # Layers with no compressor (`sliding_attention`) keep only a bounded `sliding_window` buffer,
    # which does not grow with context and so contributes nothing per token.
    compress_rates = _first_attr(cfg, ["compress_rates"])
    kv_note, kv_related = "", []
    if isinstance(compress_rates, dict) and compress_rates and isinstance(sched, list) and d_head:
        per_tok_bytes = sum(d_head / m * 2 for t in sched
                            if (m := compress_rates.get(t)))
        n_growing = sum(1 for t in sched if compress_rates.get(t))
        kib = per_tok_bytes / 1024
        kv_card = f"{kib:.1f} KiB ({_kv_band(kib)})"
        # Be explicit about what this figure does and does not include -- both exclusions are real
        # memory, just not per-token-growing (P1: state the basis, don't hide it).
        excl = []
        n_bounded = len(sched) - n_growing
        if w_local:
            excl.append(f"고정 크기 sliding 버퍼(window {w_local}, 전 {len(sched)}층"
                        + (f", 그중 {n_bounded}층은 이것만 보유" if n_bounded else "") + ")")
        idx_dim, idx_m = (_first_attr(cfg, ["index_head_dim"]),
                          compress_rates.get("compressed_sparse_attention"))
        if idx_dim and idx_m:
            n_csa = sum(1 for t in sched if t == "compressed_sparse_attention")
            excl.append(f"Lightning Indexer 캐시(+{n_csa * idx_dim / idx_m * 2 / 1024:.2f} KiB/token)")
        kv_note = ("증가하는 압축 엔트리만 계산, K==V 단일 텐서"
                   + ("; 제외: " + " / ".join(excl) if excl else ""))
        # The 아키텍처 특성 table's detail line was computed above from the standard
        # 2·n_kv·d_head assumption, which does not hold here -- restate it on the real basis.
        _abbr = {"compressed_sparse_attention": "CSA", "heavily_compressed_attention": "HCA"}
        kinds = sorted({t for t in sched if compress_rates.get(t)},
                       key=lambda t: -compress_rates[t])
        rate_str = ", ".join(f"{_abbr.get(t, t)} m={compress_rates[t]}" for t in kinds)
        # The bare family label (MQA) is right for the core attention but hides half the design:
        # every layer also runs a compressor branch whose entries are concatenated onto the KV.
        short_kinds = "/".join(_abbr.get(t, t) for t in kinds)
        attn += f" + 블록 압축 분기({rate_str}); sliding window는 전 레이어 적용"
        attn_short = f"{attn_short} + {short_kinds}"
        kv_related = [_abbr[t] for t in kinds if t in _abbr]
        kv_cache = (f"블록 압축 — 압축 레이어당 d_head/m = {d_head}/m elems / token ({rate_str}), "
                    f"K==V 단일 텐서 ⇒ {per_tok_bytes:,.0f} B/token 전체 ({kib:.2f} KiB). "
                    f"sliding 분기는 window={w_local}로 상한이 있어 컨텍스트에 따라 증가하지 않음")
    elif not real_attention:
        # No attention anywhere in the trace => there is no KV cache to size, whatever
        # head-shaped numbers the config happens to expose (xLSTM: mLSTM `num_heads`/`head_dim`).
        kv_card = "N/A — recurrent/SSM state, not KV cache"
        kv_cache = "recurrent/SSM state (no KV cache)"
    else:
        if kv_lora:
            per_tok_layer = kv_lora + (_first_attr(cfg, ["qk_rope_head_dim"]) or 0)
        elif n_kv and d_head:
            per_tok_layer = 2 * n_kv * d_head
        else:
            per_tok_layer = None
        if per_tok_layer and n_attn:
            kib = per_tok_layer * n_attn * 2 / 1024
            span = f" over {n_attn} attn layers" if isinstance(sched, list) and n_attn != L else ""
            kv_card = f"{kib:.1f} KiB ({_kv_band(kib)}){span}"
        elif attn_short == "attention-free":
            kv_card = "N/A — recurrent/SSM state, not KV cache"
        else:
            kv_card = "?"

    related = []
    if "RMSNorm" in norm:
        related.append("RMSNorm")
    if "LayerNorm" in norm:
        related.append("LayerNorm")
    if pos.startswith("RoPE"):
        related.append("RoPE")
    elif "learned" in pos:
        related.append("learned-pos")
    if _first_attr(cfg, ["no_rope_layers"]) is not None:
        related.append("NoPE")
    if attn_family != "?":
        related.append(attn_family)
    related += kv_related
    # Multi-stream residual (mHC): hc_mult>1 keeps the residual as hc_mult parallel streams
    # instead of a single add -- a structural residual variant worth surfacing on the card.
    if (_hc := _first_attr(cfg, ["hc_mult"])) and _hc > 1:
        related.append("mHC")
    if real_moe:
        related.append("MoE")
        if e_shared:
            related.append("shared expert")
        if "sigmoid" in ops:
            related.append("sigmoid-gating")
    if any(t in params_join for t in ("q_norm", "k_norm")):
        related.append("QK-Norm")
    if _first_attr(cfg, ["num_nextn_predict_layers", "num_mtp_layers"], default=0):
        related.append("MTP")
    if "conv1d" in ops:
        related.append("short-conv (SSM/DeltaNet)")
    related = list(dict.fromkeys(related))  # dedupe, keep order

    kd = ["attention-free (recurrent/mLSTM or SSM)" if attn_short == "attention-free"
          else f"{attn_short} attention"]
    if real_moe:
        kd.append(f"Sparse MoE (E={E}, top-{k}" + (f", +{e_shared} shared" if e_shared else "")
                  + (", sigmoid gating/aux-loss-free" if "sigmoid" in ops else "") + ")")
    else:
        kd.append("dense FFN")
    if fk and real_moe:
        kd.append(f"dense-prefix {fk} layer(s)")
    key_detail = "; ".join(kd)

    return {
        "model_type": getattr(cfg, "model_type", "?"),
        "attention": attn,
        "attn_kernel": attn_kernel,
        "d_head": d_head,
        "pos_enc": pos,
        "ffn": ffn,
        "norm": norm,
        "kv_cache": kv_cache,
        "kv_note": kv_note,
        "decode": "autoregressive, 1 token/step, reuses KV cache (prefill builds it)"
                  if kv_lora or (n_kv and d_head) else "1 token/step (recurrent/SSM-style state, no KV cache)",
        "total_params": total,
        "active_params": active,
        "context_note": context_note,
        # gallery card
        "attention_short": attn_short,
        "decoder_type": decoder_type,
        "layer_mix": layer_mix,
        "kv_card": kv_card,
        "related": related,
        "key_detail": key_detail,
    }


# ---- model_summary.md ---------------------------------------------------------
# `sources` is a list of dicts the caller supplies after doing Tier 2 research:
#   {"category": "vLLM 소스", "ref": "vllm/model_executor/models/<file>.py",
#    "checked": "MLA 압축 차원 해석이 추출 결과와 일치하는지 대조", "url": "..."}
# categories worth covering, per 02-new-module-handling.md Tier 2:
#   HF config/model card, 독립 서빙 구현(vLLM/SGLang/TensorRT-LLM), 공식 문서,
#   논문/기술 문서, 공개 벤치마크, (최후 수단) 일반 웹 검색

def render_model_summary(model_id, prov, structure, cfg=None, rows=None, scale=None,
                         checks=None, sources: list[dict] | None = None,
                         literals: list[dict] | None = None, model_dir: str | None = None) -> str:
    sources = sources or []
    rows = rows or []
    if literals is None:
        literals = structure.get("literal_dims") or []
    arch = derive_architecture(cfg, rows, structure, scale) if cfg is not None else {}
    S = structure["symbols"]
    lines = [f"# Model Summary -- {model_id}", ""]

    lines += ["## 기본 정보", ""]
    lines.append(f"- revision: `{prov.get('revision_resolved')}`")
    lines.append(f"- capture backend: {prov.get('capture_backend')} (meta/fake device, 실제 가중치 연산 없음)")
    lines.append(f"- 트레이스 seq_len (T): {prov.get('seq_len_used')}")
    lines.append(f"- attn_implementation: {prov.get('attn_implementation_used')}")
    lines.append(f"- 라이브러리: torch {prov.get('torch_version')}, transformers {prov.get('transformers_version')}")
    lines.append("")

    if arch:
        lines += ["## 요약 정보", ""]
        tot, act = arch.get("total_params"), arch.get("active_params")
        if tot and act and act != tot:
            # State the counting basis. Vendors are NOT consistent about whether "active
            # parameters" includes the embedding and output head, so a bare number invites a
            # false mismatch: GLM-4.5-Air is published as 12B active, which is the body ONLY
            # (ours 13.42B = 12.18B body + 0.62B embed + 0.62B lm_head), while DeepSeek-V3 is
            # published as 37B, which our embedding-inclusive 37.55B matches far better than a
            # body-only 35.7B would. We count everything a single token's forward actually
            # touches -- embedding lookup and output projection included -- and say so.
            scale_str = (f"{_hnum(tot)} total, {_hnum(act)} active ({act / tot * 100:.1f}% active)"
                         "  _(active = 토큰 1개 forward가 실제로 거치는 파라미터. embedding과 "
                         "lm_head 포함 — 벤더 발표치는 본체만 세는 경우가 있어 다를 수 있음)_")
        elif tot:
            scale_str = f"{_hnum(tot)} total (dense)"
        else:
            scale_str = "?"
        ctx = S.get("ctx")
        ctx_str = format(ctx, ",") if isinstance(ctx, int) else "?"
        ctx_note = arch.get("context_note")
        card = [
            ("1", "SCALE", scale_str),
            ("2", "Context (tokens)", ctx_str + (f"  _({ctx_note})_" if ctx_note else "")),
            ("3", "DATE", f"{prov.get('hf_created_at') or '?'}  "
                          "_(HF repo 생성일 — 대략적 출시 시점, 정확한 발표일과 다를 수 있음)_"),
            ("4", "DECODER TYPE", arch["decoder_type"]),
            ("5", "Attention", arch["attention_short"]),
            ("6", "LAYER MIX", arch["layer_mix"]),
            ("7", "KV CACHE / TOKEN (BF16)", arch["kv_card"]
             + (f"  _({arch['kv_note']})_" if arch.get("kv_note") else "")),
            ("8", "KEY DETAIL", arch["key_detail"]),
            ("9", "Related concepts", ", ".join(arch["related"]) or "?"),
        ]
        lines.append("| # | 항목 | 값 |")
        lines.append("|---|---|---|")
        for num, label, val in card:
            cell = str(val).replace("|", "\\|")  # keep any stray pipe from breaking the table
            lines.append(f"| {num} | {label} | {cell} |")
        lines.append("")
        lines.append("_※ (1)(2)(4)(5)(6)(7)(9)은 config·트레이스에서 결정적으로 도출. (3)은 HF repo "
                     "메타데이터. (8)은 도출된 사실 기반 자동 요약이며 편집상 세부는 Tier 2(sources_file)로 "
                     "보강._")
        lines.append("")
        lines.append("ref) 필드 구성은 [Raschka's LLM Architecture Gallery]"
                     "(https://sebastianraschka.com/llm-architecture-gallery/) 카드 형식을 참고. "
                     "(7)은 같은 갤러리의 [KV cache 계산 규약]"
                     "(https://sebastianraschka.com/llm-architecture-gallery/kv-cache-calculations/)을 "
                     "따른다 — BF16 2바이트, 표준 attention은 `4·n_kv·d_head`, K==V 통합이면 `2·n_kv·d_head`, "
                     "MLA는 `2·(kv_lora_rank + qk_rope_head_dim)`, 그리고 **증가하는 캐시를 가진 레이어만** "
                     "합산. 밴드 경계(KiB): 24 / 72 / 160 / 300.")
        lines.append("")

        # Qualitative interpretation only. All raw dimension numbers (L, d_model, head dims, V,
        # ctx, params) live in the 차원·심볼 table below / the SCALE card above -- kept out of here
        # so the two tables don't duplicate (and conflict, as a bare head_dim did for MLA).
        lines += ["## 아키텍처 특성 (정성 요약 — 수치는 아래 차원·심볼 표 참조)", ""]
        lines.append("| 항목 | 값 |")
        lines.append("|---|---|")
        lines.append(f"| 모델 타입 (config) | `{arch['model_type']}` |")
        lines.append(f"| attention | {arch['attention']} |")
        lines.append(f"| attention 커널 | {arch['attn_kernel']} |")
        lines.append(f"| 위치 인코딩 | {arch['pos_enc']} |")
        lines.append(f"| FFN | {arch['ffn']} |")
        lines.append(f"| 정규화 | {arch['norm']} |")
        lines.append(f"| tie embeddings | {_first_attr(cfg, ['tie_word_embeddings'], default='?')} |")
        lines.append(f"| decode 방식 | {arch['decode']} |")
        lines.append(f"| KV cache 크기 | {arch['kv_cache']} |")
        lines.append("")

    lines += ["## 차원·심볼 (공통 심볼, rules/symbols.yaml 기준 — 모든 수치의 단일 출처)", ""]
    lines.append("| symbol | value |")
    lines.append("|---|---|")
    spec = load_symbols()
    # A symbol group (mla / v4_compress / ssm / ...) with NOTHING resolved means the architecture
    # simply has no such mechanism -- that is a fact, not a gap. Only a group that is PARTIALLY
    # resolved has genuinely-unknown members worth a Tier 2 flag. Before this split, DeepSeek-V4
    # showed c_kv/d_nope/d_v as "미확인, Tier 2 대상" when the truth is that V4 is not MLA at all.
    def _group_present(g):
        members = [(n, s) for n, s in spec.items() if s.get("group") == g]
        keys = [n for n, s in members if s.get("group_key")]
        # A group_key symbol is the defining field: if one is declared, only it decides presence
        # (v_head_dim exists on non-MLA configs, so "any member resolved" would misfire).
        probe = keys or [n for n, _s in members]
        return any(structure["symbols"].get(n) is not None for n in probe)

    absent = {g for g in {s.get("group") for s in spec.values() if s.get("group")}
              if not _group_present(g)}
    for sym, val in structure["symbols"].items():
        grp = (spec.get(sym) or {}).get("group")
        if val is None and grp in absent:
            cell = f"—  _(해당 없음: 이 모델은 `{grp}` 계열 구조를 쓰지 않음)_"
        else:
            cell = _fmt_symval(val)
        lines.append(f"| {sym} | {cell} |")
    lines.append("")

    # Config fields this architecture actually uses that rules/symbols.yaml does not know about.
    # Derived by tagging every dim-bearing config field and reading back which module attributes
    # came out WITHOUT a tag (src/symbolic_dims.py). Complementary to the Tier 3 list below: that
    # one says "this NUMBER is unexplained", this one says "this FIELD is unregistered", which is
    # a far more actionable research target. 19 of 25 models come out empty.
    # How each axis label was obtained. This is the honesty column of the whole deliverable:
    # a name from a scoped rule is evidence, a name from the arithmetic tail is a guess that
    # happens to be true at this seq_len. Publishing the split lets a reader weight the tables,
    # and lets the gate refuse a silent rise in guessing.
    lp = structure.get("label_provenance") or {}
    if lp.get("total_axes"):
        _KO = {"scoped_symbol": "이 모듈 스코프의 심볼", "scoped_formula": "이 모듈 스코프의 유도식",
               "plain_symbol": "스코프 없는 심볼", "derived_formula": "derived_dims 유도식",
               "runtime": "런타임 축 (B/T/1)", "out_of_scope_symbol": "스코프가 배제한 심볼",
               "reused_symbol": "같은 shape에서 이미 쓴 심볼 재사용", "bare": "이름 없음 (정수 유지)",
               "heur_product": "휴리스틱: 두 심볼의 곱", "heur_multiple": "휴리스틱: 심볼의 배수",
               "heur_half": "휴리스틱: 심볼의 절반", "heur_plus1": "휴리스틱: 심볼+1"}
        tot = lp["total_axes"]
        lines += ["## 라벨 출처 (이 표의 이름들이 어디서 왔나)", ""]
        lines.append(f"shape 축 **{tot:,}개**를 렌더하면서 어떤 근거로 이름을 붙였는지의 내역이다. "
                     "위쪽 네 줄은 `rules/`에 **등록된 규칙**이 답을 준 경우이고, `휴리스틱`으로 "
                     "시작하는 줄은 등록된 규칙이 없어 **산술적으로 맞는 이름을 지어낸** 경우다. "
                     "후자는 이번 트레이스의 seq_len에서만 참일 수 있으므로 그대로 신뢰하면 안 되고, "
                     "`02-new-module-handling.md` Tier 2로 확인해 규칙으로 승격시켜야 한다.")
        lines.append("")
        lines.append("| 근거 | 축 수 | 비율 |")
        lines.append("|---|---:|---:|")
        for rule, cnt in (lp.get("by_rule") or {}).items():
            lines.append(f"| {_KO.get(rule, rule)} | {cnt:,} | {100.0 * cnt / tot:.2f}% |")
        lines.append("")
        lines.append(f"등록된 규칙 **{lp.get('derived', 0):,}축**, 약한 근거 {lp.get('weak', 0):,}축, "
                     f"휴리스틱 **{lp.get('heuristic', 0):,}축 ({lp.get('heuristic_pct', 0)}%)**, "
                     f"이름 없음 {lp.get('bare', 0):,}축.")
        lines.append("")
        ex = lp.get("heuristic_examples") or []
        if ex:
            lines.append("지어낸 이름이 가장 많이 붙은 자리 (여기부터 확인하면 된다):")
            lines.append("")
            lines.append("| 모듈 | 라벨 | 규칙 | 축 수 |")
            lines.append("|---|---|---|---:|")
            for e in ex:
                lines.append(f"| `{e['module'] or '(모듈 밖)'}` | `{e['label']}` | "
                             f"{_KO.get(e['rule'], e['rule'])} | {e['axes']} |")
            lines.append("")

    unreg = (structure.get("unregistered_fields") or [])
    if unreg:
        lines += ["## 미등록 config 필드 (Tier 2 조사 대상)", ""]
        lines.append("이 아키텍처가 실제로 쓰는 config 필드 중 `rules/symbols.yaml`에 등록되지 "
                     "않은 것들이다. 등록되지 않은 폭은 이름을 붙일 근거가 없으므로 shape 셀에 "
                     "정수로 남는다. `02-new-module-handling.md` Tier 2 절차로 역할을 확인한 뒤 "
                     "`aliases`(같은 개념의 다른 필드명) 또는 `derived_dims.yaml`(계산식)에 "
                     "**출처와 함께** 등록하면 다음 모델부터 자동으로 잡힌다.")
        lines.append("")
        lines.append("| config 필드 | 값 | 쓰는 모듈 수 |")
        lines.append("|---|---|---|")
        for u in unreg[:20]:
            lines.append(f"| `{u['field']}` | {u['value']} | {u['modules']} |")
        lines.append("")

    if literals:
        named = [L for L in literals if L.get("expr")]
        unknown = [L for L in literals if not L.get("expr")]
        lines += ["## 유도 상수 (합성 차원 범례)", ""]
        lines.append("심볼 하나로 안 떨어지고 **여러 심볼의 조합**으로 나오는 고정 차원들이다. "
                     "표·트레이스의 shape 셀에는 검증된 식(`T+T/m_csa` 등)으로 렌더되며, 여기서는 그 식이 "
                     "무슨 뜻인지와 이번 실행에서의 구체값을 함께 준다. 유래는 `rules/derived_dims.yaml`의 "
                     "식을 이 모델 심볼로 **계산해 값이 정확히 일치할 때만** 붙는다(인수분해 추측 아님). "
                     "설명이 안 붙은 값은 정수 그대로 남기고 아래 Tier 3로 넘긴다(P1 — 지어내지 않는다).")
        lines.append("")
        lines.append("| 값 | 유래 | 나타나는 모듈 |")
        lines.append("|---|---|---|")
        for L in named + unknown:
            expr = L.get("expr") or "**미해결 — 아래 Tier 3 확인 필요**"
            where = ", ".join(L.get("where", []))
            lines.append(f"| {L['value']} | {expr} | {where} |")
        lines.append("")
        if unknown:
            # Escalation surface (02-new-module-handling.md Tier 3): an unexplained constant means
            # a module we have not researched yet. Say so loudly with the exact research target
            # rather than leaving a bare number the reader has to reverse-engineer.
            lines += ["### ⚠ 미해결 유도 상수 — 신규 모듈 조사 필요 (Tier 3)", ""]
            lines.append(f"아래 {len(unknown)}개 값은 `rules/derived_dims.yaml`의 어떤 식으로도 설명되지 "
                         "않는다. 거의 항상 **아직 조사하지 않은 모듈**이 있다는 뜻이다. "
                         "`02-new-module-handling.md`의 「신규 모듈 조사 절차」대로 1차 소스(현재 실행 중인 "
                         "modeling 코드) → 독립 서빙 구현(vLLM/SGLang/TensorRT-LLM) → 공식 문서·논문 → "
                         "아키텍처 갤러리 순으로 확인한 뒤, `rules/symbols.yaml`(별칭) 또는 "
                         "`rules/derived_dims.yaml`(식)에 **출처와 함께** 등록할 것. "
                         "확인되지 않으면 추측해서 채우지 말고 사람에게 확인을 요청한다(P1).")
            lines.append("")
            lines.append("| 값 | 나타나는 모듈 | 조사 착안점 |")
            lines.append("|---|---|---|")
            for L in unknown:
                where = ", ".join(L.get("where", []))
                lines.append(f"| {L['value']} | {where} | 해당 모듈의 `__init__` 투영 폭과 "
                             f"forward의 concat/slice 축을 config 필드 조합으로 역산 |")
            lines.append("")

    lines += ["## 레이어 구조", ""]
    for layer in structure["layers"]:
        lo, hi = layer["range"]
        rng = f"{lo}" if lo == hi else f"{lo}-{hi}"
        lines.append(f"- layer {rng}: {', '.join(layer['blocks']) or '(no block info)'}")
    lines.append("")

    if checks:
        lines += ["## 검증 로그 (01-main.md §9 체크리스트)", ""]
        order = {"FAIL": 0, "WARN": 1, "INFO": 2, "SKIP": 3, "PASS": 4}
        overall = "PASS" if not any(s == "FAIL" for s, _ in checks.values()) else "FAIL"
        n_warn = sum(1 for s, _ in checks.values() if s == "WARN")
        repro = dict(checks).get("C13", ("?", ""))[0]
        lines.append(f"- **종합: {overall}** (WARN {n_warn}개, 재현성 C13={repro})")
        lines.append("")
        lines.append("| check | status | detail |")
        lines.append("|---|---|---|")
        for cid in sorted(checks, key=lambda c: int(c[1:])):
            status, detail = checks[cid]
            d = (detail or "").replace("\n", " ")
            if len(d) > 100:
                d = d[:97] + "..."
            lines.append(f"| {cid} | {status} | {d} |")
        lines.append("")

    lines += ["## 추출 방법", ""]
    lines.append(
        "01-main.md Step 1~8에 따라, config.json + 공식 modeling 코드의 실제 forward "
        "실행(meta/fake device)만으로 shape·dependency를 확보했다. 값은 전부 실행 결과에서만 "
        "나오며(P1), shape은 아키텍처 심볼로 렌더된다(§6, 구체 숫자는 provenance.json으로 복원). "
        "아래 소스 중 '교차검증'은 라벨·해석 확인용이지 shape/dependency 값 자체의 출처가 아니다."
    )
    lines.append("")

    lines += ["## 구성 근거 / 소스", ""]
    lines.append("이 요약의 shape·dependency 값은 아래를 **실제 실행**해 얻었다(지어내지 않음, P1):")
    lines.append("")
    lines.append("| 구분 | 소스 | 역할 |")
    lines.append("|---|---|---|")
    lines.append(f"| config (1차) | HF `{model_id}` config.json @ `{prov.get('revision_resolved')}` "
                 f"(sha256 `{(prov.get('config_sha256') or '')[:12]}…`) | 심볼 값의 출처 |")
    lines.append(f"| modeling code (1차) | transformers {prov.get('transformers_version')} 공식 "
                 f"modeling forward (meta device) | op·shape·dependency 캡처 |")
    lines.append(f"| trace (1차) | dispatch(ATen) 레벨, seq_len(T)={prov.get('seq_len_used')} | 표·그래프 생성 근거 |")
    lines.append("")
    lines.append("교차검증(Tier 2 — 라벨·해석용, shape 값의 출처 아님):")
    lines.append("")
    if not sources:
        lines.append(
            "_(추가 교차검증 소스 미첨부 — 프로파일 `sources_file`로 HF model card, "
            "vLLM/SGLang/TensorRT-LLM 독립 구현, 논문/기술 리포트, "
            "[Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/), "
            "공개 벤치마크 순으로 채울 수 있다. 위 1차 소스만으로도 shape·dependency는 확정됨.)_"
        )
    else:
        lines.append("| 항목 | 소스 | 확인한 내용 |")
        lines.append("|---|---|---|")
        for s in sources:
            ref = f"[{s['ref']}]({s['url']})" if s.get("url") else s.get("ref", "")
            lines.append(f"| {s.get('category', '')} | {ref} | {s.get('checked', '')} |")
    lines.append("")
    # The label review is where a SOURCE is actually read, and part of what it establishes can
    # never become a rule -- two config values coincide, a fused parameter's axis order is
    # invisible to the trace, a width is only explained in a paper. Those verdicts belong next to
    # the tables they qualify, not only in a side file nobody opens (③ 라벨 검토 2026-08-09).
    if model_dir:
        import review_notes
        lines += review_notes.summary_section(model_dir)
    return "\n".join(lines)


def write_model_summary(model_dir: str, content: str) -> str:
    path = os.path.join(model_dir, "model_summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path
