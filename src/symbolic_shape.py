"""Symbolic shape rendering (01-main.md section 6 / section 10).

Concrete tensor dimensions depend on run params we pick arbitrarily -- batch size (always
1 here) and the traced seq_len. Rendering them as numbers ties the op tables to one
specific run. Instead every dimension is rendered as an architecture symbol (B, T, d_model,
n_h, d_head, E, k, ...) or a simple symbolic expression (B*T, n_h*d_head, T*k, T+1, ...),
so the deliverables describe the *architecture*, not one run. The concrete seq_len/config
remain in provenance.json (seq_len_used + symbol_table), so numbers are fully recoverable
-- P1 traceability holds, nothing is guessed.

Ambiguity control: if the traced seq_len equals some config dimension (e.g. n_h), a single
concrete value could map to two symbols. resolve_seq_len() picks a seq_len that collides
with no config value, so single dims (and simple products like T*k) resolve unambiguously.
"""
import re

from summarize import _first_attr, derived_symbols, load_symbols, resolve_symbols

# Head/expert-COUNT symbols, as opposed to the head/state-SIZE symbol that (by the universal
# "[..., count, size]" reshape convention) immediately follows one. Used only as a positional
# tie-break in build_resolver.dim() when two symbols coincidentally share a value.
_COUNT_LIKE_SYMS = {"n_h", "n_kv", "n_h_ssm", "n_g_ssm", "n_h_lin_k", "n_h_lin_v", "n_h_I", "E", "k"}

# MUTUAL EXCLUSION: one tensor is laid out along query heads OR kv heads, never both. `repeat_kv`
# is the only place the two counts meet, and it bridges them with the DERIVED `n_h/n_kv` factor
# (`[B, n_kv, n_h/n_kv, T, d_head]`), never with both plain names in one tuple. So `n_h` and `n_kv`
# co-occurring in a single shape is not an ambiguity to rank -- it is proof that one of them was
# pasted onto an axis that is not a head-count axis at all.
#
# Both have very high priority (7, 8), so before this guard ANY axis inside self_attn whose value
# merely coincided with n_h or n_kv got that name. An audit across all 26 models found 16,859 such
# axes in 8 models, every one of them wrong, and all of the same shape -- the head-SIZE or partial-
# RoPE axis of a KV tensor stolen by a head-COUNT name:
#   Llama-3.1-405B  [1,8,16,128]  n_h==d_head==128        -> last axis was `n_h`, is `d_head`
#   Llama-3.1-70B   [1,8,16,64]   n_h==d_head/2==64       -> last axis was `n_h`, is `d_head/2`
#   gpt-oss-20b/120b[1,8,264,64]  n_h==d_head==64         -> last axis was `n_h`, is `d_head`
#   DeepSeek-V3     [1,128,24,128] n_kv==d_nope==128      -> last axis was `n_kv`, is `d_nope`
#                                  (n_kv is meaningless under MLA in the first place)
# The existing defenses could not catch these: `avoid` only blocks reusing the SAME name twice, and
# the count-then-size tie-break only fires when the PRECEDING axis is count-like -- here it is a
# sequence axis (T / T+1 / w_local), so nothing fired. Found by self-audit, 2026-07-30; the external
# review missed this class entirely.
_HEAD_COUNT_EXCLUSIVE = frozenset({"n_h", "n_kv"})

# A LayerNorm leaf names its POSITION in the block, not what it computes. `post_attention_layernorm`
# is not an attention module -- it is the norm applied to the residual stream *after* attention, and
# its width is d_model. But every `attn|attention` scope regex matches the substring "attention"
# inside that name, so the whole attention symbol set (n_h, n_kv, d_head) and every attention-scoped
# derived rule fired there. On Llama-3.1-70B/405B, where d_model == n_h*d_head, that rendered the
# residual stream as `n_h*d_head` in post_attention_layernorm while the elementwise_add feeding it
# said `d_model` -- one tensor, two names, and the break was visible mid-layer (external review,
# 2026-07-30).
#
# Fix: match scopes against the norm's PARENT module. That is the module whose context actually
# determines the width, and it is right in both directions -- `model.layers.N.post_attention_layernorm`
# becomes `model.layers.N` (no attention scope -> d_model), while MLA's
# `model.layers.N.self_attn.q_a_layernorm` becomes `model.layers.N.self_attn`, which still matches
# `attn` and keeps resolving to c_q as it must. Only `*layernorm`/`*layer_norm` leaves are stripped:
# the `_norm` family (q_norm, norm_mlstm, ...) sits inside the module it belongs to, where the leaf
# name carries real information.
_NORM_LEAF = re.compile(r"\.[A-Za-z0-9_]*layer_?norm$", re.I)


def _scope_path(module_path: str | None) -> str | None:
    """Module path to match `scope` regexes against (see _NORM_LEAF)."""
    if not module_path:
        return module_path
    return _NORM_LEAF.sub("", module_path) or module_path


def _dim_symbols(symbols: dict | None = None) -> list[str]:
    """Symbols that actually appear as tensor dimensions, in single-match priority order
    (L/ctx/w_local/layer_sched do not). Both the set and the order come from rules/symbols.yaml
    (`dim: true`, `priority:`) so adding a symbol for a new architecture needs no code change --
    that is the whole point of the new-module process in 02-new-module-handling.md.
    Priority matters when two symbols hold the same value (MLA d_head == d_nope, Mamba
    d_head == d_state): the lower `priority` wins so the more fundamental name is rendered."""
    symbols = symbols if symbols is not None else load_symbols()
    dims = [(spec.get("priority", 999), name)
            for name, spec in symbols.items() if spec.get("dim")]
    return [name for _p, name in sorted(dims)]


def _config_values(cfg, symbols: dict | None = None) -> dict:
    sym = resolve_symbols(cfg, symbols)
    vals = {}
    for name in _dim_symbols(symbols):
        v = sym.get(name)
        if isinstance(v, int) and not isinstance(v, bool):
            vals[name] = v
    if "d_head" not in vals and vals.get("d_model") and vals.get("n_h"):
        vals["d_head"] = vals["d_model"] // vals["n_h"]
    # A pure-MoE stack (gpt-oss, OLMoE, Llama-4) has NO dense FFN: it sizes its experts with plain
    # `intermediate_size` because there is no separate `moe_intermediate_size`. `d_ff` is then not a
    # second dimension that coincides with d_moe -- it is the same field under a name this
    # architecture does not have, so it must not compete for axes. It was: `d_ff`'s scope matches the
    # MoE block itself (`model.layers.N.mlp`) and its priority beats d_model, so gpt-oss rendered
    # that block's input -- the residual stream -- as `d_ff`, while the router one level down said
    # `d_model`. One tensor, two names; caught by the dataflow check right after the router scope was
    # fixed (2026-07-30). Dropping the NAME (the value still stands in the symbol table) lets
    # d_model win outside the experts and d_moe win inside them, which is the real architecture.
    # Models that carry BOTH fields (DeepSeek-V3's dense prefix layers + MoE tail) are untouched.
    if sym.get("E") and vals.get("d_ff") is not None and not _first_attr(cfg, ["moe_intermediate_size"]):
        vals.pop("d_ff", None)
    return vals


def resolve_seq_len(cfg, base: int, symbols: dict | None = None) -> int:
    """Smallest seq_len >= base that equals no config dimension value NOR any pure-config
    derived quantity (e.g. GQA repeat factor n_h/n_kv), so T never collides with either.
    Keeps C14 valid (result is still >= base).

    The derived-value avoidance was added after an external review (2026-07-29) found
    Llama-3.1-405B's traced T=16 exactly equal to its GQA repeat factor n_h/n_kv=128/8=16,
    which mislabeled several activation axes as the repeat-factor formula instead of T. Calling
    derived_symbols with seq_len=None means any rule whose expr references T raises NameError
    and is silently skipped (see summarize._eval_rule) -- only genuinely T-independent, purely
    config-derived values come back, which is exactly the avoidance set we need here."""
    avoid = set(_config_values(cfg, symbols).values())
    glob, scoped = derived_symbols(resolve_symbols(cfg, symbols), cfg=cfg, seq_len=None,
                                   spec=None)
    avoid.update(glob)
    for _rx, m in scoped:
        avoid.update(m)
    t = max(int(base), 16)
    while t in avoid:
        t += 1
    return t


def build_resolver(cfg, seq_len: int, symbols: dict | None = None):
    """Returns resolve_shape(shape_list|None, module_path=None) -> symbolic list. Also exposes
    `.table` (symbol -> concrete value) for provenance.

    `module_path` disambiguates symbols that share a value. A global priority alone is guaranteed
    to be wrong in one of the two contexts: Llama-4 has num_local_experts == d_head == 128, so
    every attention head dim rendered as `E`. Each symbol may declare a `scope` regex in
    rules/symbols.yaml; a symbol whose scope matches the owning module wins over global priority.
    18 of 25 onboarded models have at least one such collision, so this is not an edge case."""
    vals = {}
    vals.update(_config_values(cfg, symbols))
    ordered = [(s, vals[s]) for s in _dim_symbols(symbols) if s in vals and vals[s] >= 2]
    # T goes last: a config dim that happens to equal the traced seq_len should render as the
    # config symbol, and resolve_seq_len() already guarantees no such collision anyway.
    ordered.append(("T", int(seq_len)))
    vals["T"] = int(seq_len)

    # Values that rules/derived_dims.yaml explains exactly, mapped to their VERIFIED compact
    # formula. These beat the "small multiple" heuristic below, which would attach a
    # numerically-true but architecturally-wrong origin: DeepSeek-V4's per-group projection width
    # 4096 is n_h*d_head/g_o, yet it also equals 4*d_g. Rendering the sourced formula keeps the
    # cell self-describing without guessing (the prose meaning stays in the 유도 상수 legend).
    authoritative, authoritative_scoped = derived_symbols(
        resolve_symbols(cfg, symbols), cfg=cfg, seq_len=seq_len)

    spec_all = symbols if symbols is not None else load_symbols()
    scopes = {s: re.compile(spec_all[s]["scope"])
              for s, _v in ordered if (spec_all.get(s) or {}).get("scope")}
    strict = {s: bool((spec_all.get(s) or {}).get("scope_strict")) for s, _v in ordered}
    plain_symbol_names = {s for s, _v in ordered}

    def _ctx_symbols(module_path):
        """Symbol list reordered for this module: in-scope first, unscoped next, out-of-scope last.

        A symbol whose `scope` matches THIS module outranks global priority -- Llama-4 has
        E == d_head == 128, so inside self_attn the head dim must not render as `E`.

        Out-of-scope symbols are DEMOTED, not dropped. Dropping was tried and caused a large
        regression: scope regexes cannot enumerate every naming convention (xLSTM's attention
        equivalent is `mlstm_layer`, which no `attn|attention` scope matches), so dropping turned
        88k correctly-symbolised dims into bare integers. Demotion can only improve a choice
        between equals, never remove a symbol that would otherwise apply.

        The exception is a symbol that opts in with `scope_strict: true`, which IS dropped when out
        of scope. That is for names which are meaningless anywhere else and whose module is certain
        to exist whenever the symbol does -- DeepSeek-V4's block-compression rates m_csa/m_hca live
        only inside `compressor`. Demoting them was not enough: m_csa=4 == n_hc=4, so the demoted
        m_csa still won the mHC stream-mixing matrix `[..., n_hc, n_hc]` (whose second axis has no
        fresh name left and had correctly been reusing n_hc), rewriting 68,974 axes to a rate that
        has nothing to do with hyper-connections. Opting in per symbol keeps the 2026-07-29 lesson
        intact -- the blanket drop is still wrong -- while letting a genuinely local name stay
        local."""
        module_path = _scope_path(module_path)
        if not module_path:
            return [(s, v) for s, v in ordered if not strict.get(s)], [], []
        hit, plain, miss = [], [], []
        for s, v in ordered:
            rx = scopes.get(s)
            if rx is None:
                plain.append((s, v))
            elif rx.search(module_path):
                hit.append((s, v))
            elif not strict.get(s):
                miss.append((s, v))
        return hit, plain, miss

    def dim(n, module_path=None, avoid=None, prev=None, is_weight=False, forbid=None):
        if not isinstance(n, int) or isinstance(n, bool):
            return str(n)
        if n == 1:
            return "B"  # batch (and any genuine singleton, e.g. MQA n_kv=1 -- see C7/symbols)
        hit_syms, plain_syms, miss_syms = _ctx_symbols(module_path)
        ordered_ctx = hit_syms + plain_syms + miss_syms
        if is_weight:
            # HARD PHYSICAL INVARIANT: a weight is a static parameter allocated from config at
            # load time -- it cannot possibly depend on the runtime sequence length. So T (and any
            # expression built from T) is never a valid label for a weight axis, no matter how
            # exactly it matches numerically. Enforcing this at the source is far more robust than
            # hoping resolve_seq_len picks a non-colliding T, because a collision can arise from a
            # PRODUCT of T with something else, which no choice of T can fully avoid.
            #
            # Real violations this catches (found 2026-07-29 while re-auditing the external
            # review's fixes): gemma-2-2b rendered q_proj/o_proj weights as `[T, d_model]`
            # (T=2048 coincided with n_h*d_head=8*256=2048), and gpt2-xl rendered the learned
            # positional-embedding table `wpe` as `[T*d_head, d_model]` (its 1024 max positions
            # coincided with T*d_head=16*64). Both are physically impossible and would mislead
            # anyone reading the tables as workload descriptions.
            ordered_ctx = [(s, v) for s, v in ordered_ctx if s != "T"]
            hit_syms = [(s, v) for s, v in hit_syms if s != "T"]
            plain_syms = [(s, v) for s, v in plain_syms if s != "T"]
            miss_syms = [(s, v) for s, v in miss_syms if s != "T"]

        # Same invariant as above: a derived formula containing T cannot describe a weight axis.
        def _ok(formula):
            return not (is_weight and re.search(r"\bT\b", formula))

        def _pick(cands):
            """Exact matches within one tier, with the count-then-size positional tie-break.

            When several symbols tie on value (Llama-3.1-405B has n_h == d_head == 128) and the
            axis immediately before this one resolved to a head/expert-COUNT symbol, prefer a
            SIZE symbol here: a count axis is essentially always followed by that unit's own dim,
            never by a second unrelated count. Leaves gpt-oss's attention-sink shape
            `[B, n_h, T, T+1]` (also n_h == d_head == 64) alone -- there the preceding axis is
            `B`, so the tie-break never fires."""
            ms = [s for s, v in cands if n == v and (not avoid or s not in avoid)
                  and (not forbid or s not in forbid)]
            if not ms:
                return None
            if prev in _COUNT_LIKE_SYMS:
                sized = [s for s in ms if s not in _COUNT_LIKE_SYMS]
                if sized:
                    return sized[0]
            return ms[0]

        # Resolution order is SCOPE RELEVANCE first, then simplicity -- not simplicity alone.
        # A symbol or formula whose declared scope matches THIS module is evidence about this
        # module; one whose scope explicitly excludes it is not, however simple it looks.
        #
        # An earlier pass (2026-07-29) tried "any plain symbol beats any formula" and regressed
        # DeepSeek-V4: inside `self_attn.compressor` the HCA compressed-entry count (T/m_hca=16,
        # a scoped formula that genuinely applies there) lost to `g_o`=16, the grouped-output-
        # projection group count whose scope is o_a_proj/o_b_proj -- a different module entirely.
        # The result contradicted itself inside a single op: the concat inputs said `g_o` while
        # that same concat's output said `T+T/m_hca`.
        r = _pick(hit_syms)                        # 1. plain symbol scoped TO this module
        if r:
            return r
        scope_path = _scope_path(module_path)
        if scope_path:                             # 2. derived formula scoped TO this module
            for rx, m in authoritative_scoped:
                if n in m and rx.search(scope_path) and _ok(m[n]):
                    return m[n]
        r = _pick(plain_syms) or _pick(miss_syms)  # 3. unscoped, then out-of-scope plain symbol
        if r:
            return r
        # A scoped derived rule that did NOT match this module must not fire globally -- that is
        # the whole point of the scope (Llama-4: 2*n_kv*d_head is meaningless inside `experts`,
        # where 2048 is really E*T and the T-product rule below gets it right).
        if n in authoritative and not any(n in m for _rx, m in authoritative_scoped) \
                and _ok(authoritative[n]):
            return authoritative[n]      # derived_dims explains it exactly -> use that formula
        # A plain symbol we skipped only because it was already used elsewhere in this same
        # shape tuple (see resolve_shape) is still a better answer than falling through to a
        # heuristic guess or a bare literal -- reuse is allowed once no fresh alternative exists.
        # `forbid` is NOT relaxed here the way `avoid` is: reusing a name already spent in this
        # tuple is merely redundant, but a mutually-exclusive name is affirmatively false.
        if avoid:
            for s, v in ordered_ctx:
                if n == v and (not forbid or s not in forbid):
                    return s
        for s, v in ordered_ctx:          # symbol + 1 (e.g. cache length T+1)
            if n == v + 1 and v >= 16:    # small symbols would give noise like "g_o+1"
                return f"{s}+1"
        # Small multiple (e.g. 2*d_moe for gate+up). T is EXCLUDED: a constant weight axis that
        # happens to equal 2x the traced seq_len (DeepSeek-V4 n_h*d_head/g_o = 4096 at T=2048)
        # would otherwise render "2*T" and fabricate a sequence dependency on a fixed dim -- the
        # exact thing P1 forbids. resolve_seq_len() only de-collides single dims, not multiples,
        # so the guard has to be here. A genuine 2*T stays an honest literal instead.
        for c in (2, 3, 4):
            for s, v in ordered_ctx:
                if s != "T" and n == c * v:
                    return f"{c}*{s}"
        # product of two symbols, but ONLY when one factor is T (the runtime dim): T*k routed
        # tokens, B*T flattened, etc. Pure structural×structural products (e.g. c_kv*k) are
        # almost always coincidental collisions, not real derived dims -- skip them so a static
        # weight dim like n_h*(d_nope+d_rope)=3072 stays an honest literal instead of "c_kv*k".
        # For a weight this whole rule is unreachable by construction (T was dropped from
        # ordered_ctx above), which is exactly what killed gpt2-xl's bogus `wpe = [T*d_head, ...]`.
        for i, (s1, v1) in enumerate(ordered_ctx):
            for s2, v2 in ordered_ctx[i:]:
                if v1 * v2 == n and "T" in (s1, s2):
                    # Canonical operand order (T last), so the SAME product never gets two
                    # spellings. `ordered_ctx` is reordered per module scope, so without this the
                    # identical tensor came out `E*T` in one op and `T*E` in the next -- caught by
                    # the dataflow-consistency audit, 2026-07-30.
                    return f"{s2}*{s1}" if s1 == "T" else f"{s1}*{s2}"
        for s, v in ordered_ctx:          # half dim (RoPE inv_freq / rotate_half use d_head/2)
            if s != "T" and v % 2 == 0 and n == v // 2:
                return f"{s}/2"
        return str(n)                     # irreducible -> keep the number (do not fabricate)

    def resolve_shape(shape, module_path=None, is_weight=False, pin=None):
        """`pin=(index, label)` forces one axis and resolves it FIRST.

        Used for a linear/matmul weight's contraction axis, which is by definition the same
        physical dimension as the input activation's last axis and must therefore carry the same
        label. Resolving it first also lets the duplicate-avoidance below give the OTHER axis a
        different name, which is what fixes square weights: Zamba2's `q_proj` is
        nn.Linear(attention_hidden_size=4096, n_h*head_dim=4096), so both axes are 4096 and the
        left-to-right pass labelled it `[d_attn, n_h*d_head]` -- exactly backwards from the real
        [out, in] order. Pinning `in` to the activation's `d_attn` yields `[n_h*d_head, d_attn]`.
        """
        if shape is None:
            return None
        # Track plain symbols already used WITHIN this one shape tuple. Without this, two
        # sibling axes that happen to share a value both render as the same highest-priority
        # symbol -- e.g. a [B,T,n_h,d_head] reshape rendered as [B,T,n_h,n_h] wherever a model's
        # head count equals its head dim (Llama-3.1-405B: n_h=128=d_head), because each axis was
        # resolved independently with no memory of what the previous axis already claimed. Found
        # via external review, 2026-07-29. Only plain symbol names are tracked (not expressions
        # like "n_h*d_head" or "T+1"), and reuse is still allowed as a last resort (see `avoid`
        # handling in dim()) rather than falling back to a worse (heuristic/bare-number) render.
        used, prev = set(), None
        # Names that are now affirmatively WRONG for every remaining axis, because a
        # mutually-exclusive sibling already claimed one (see _HEAD_COUNT_EXCLUSIVE).
        banned = set()

        def _claim(label):
            if label in plain_symbol_names:
                used.add(label)
            if label in _HEAD_COUNT_EXCLUSIVE:
                banned.update(_HEAD_COUNT_EXCLUSIVE - {label})

        out = [None] * len(shape)
        if pin is not None:
            pi, plabel = pin
            if 0 <= pi < len(shape):
                out[pi] = plabel
                _claim(plabel)
        for i, x in enumerate(shape):
            if out[i] is not None:          # pinned axis already decided
                prev = out[i] if out[i] in plain_symbol_names else None
                continue
            r = dim(x, module_path, avoid=used, prev=prev, is_weight=is_weight, forbid=banned)
            out[i] = r
            prev = r if r in plain_symbol_names else None
            _claim(r)
        return out

    resolve_shape.table = {"B": 1, **{s: v for s, v in ordered}}
    return resolve_shape
