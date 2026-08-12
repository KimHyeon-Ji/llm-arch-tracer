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
import collections
import re

from summarize import _first_attr, derived_symbols, load_symbols, resolve_symbols

# Head/expert-COUNT symbols, as opposed to the head/state-SIZE symbol that (by the universal
# "[..., count, size]" reshape convention) immediately follows one. Used only as a positional
# tie-break in build_resolver.dim() when two symbols coincidentally share a value.
_COUNT_LIKE_SYMS = {"n_h", "n_kv", "n_h_ssm", "n_g_ssm", "n_h_lin_k", "n_h_lin_v", "n_h_I", "E", "k"}

# How many entries a stage SELECTS -- top-k routing, indexer top-k. Unlike a head count these are
# not a layout of anything: no parameter is allocated at this width and no module declares it as a
# dimension, so a scope match must not let one outrank a genuine width (see _ctx_symbols).
_SELECTION_SYMS = {"k", "k_I"}

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
_HAS_T = re.compile(r"\bT\b")
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

# `model.layers.7.mixer.in_proj` -> 7. Same stack names scope.py knows about.
_LAYER_IDX = re.compile(r"\.(?:layers|h|blocks|block|layer)\.(\d+)(?:\.|$)")


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
    # The test is simply "do the two names hold the same number?" -- that is exactly when d_ff is
    # not a second dimension but the same field wearing a name this architecture does not have.
    # Llama-4 has BOTH widths (dense 16384 / expert 8192), so its d_ff survives and correctly
    # labels the dense FFN.
    if sym.get("E") and vals.get("d_ff") is not None and vals.get("d_ff") == vals.get("d_moe"):
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

    # WHICH KIND OF BLOCK IS THIS LAYER. A hybrid stack interleaves block types, and several of
    # them name their module identically -- Nemotron-H calls every block `mixer`, whether it holds
    # Mamba, attention or an FFN. So a module-path regex cannot separate them, and where the two
    # architectures' widths collide (Nemotron-3-Super: head_dim == ssm_state_size ==
    # mamba_num_heads == 128) the attention names took over Mamba axes with nothing to reveal it.
    # The config states the schedule outright (`layers_block_type` / `layer_types`), so a symbol
    # can declare the block kinds it belongs to and be demoted everywhere else.
    _sched = None
    for _f in ("layers_block_type", "layer_types"):
        _v = getattr(cfg, _f, None)
        if isinstance(_v, (list, tuple)) and _v:
            _sched = [str(x) for x in _v]
            break
    layer_kinds = {s: set(spec_all[s]["not_layer_types"])
                   for s, _v in ordered if (spec_all.get(s) or {}).get("not_layer_types")}
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
        # A symbol that declares block kinds is DEMOTED in a layer of the wrong kind, exactly like
        # a scope miss. Demotion, not removal: the schedule names vary by vendor and a name we
        # cannot place is still better than a bare integer (the same lesson that keeps scope
        # misses demoted rather than dropped).
        wrong_kind = set()
        if _sched and layer_kinds:
            mi = _LAYER_IDX.search(module_path)
            if mi:
                li = int(mi.group(1))
                if 0 <= li < len(_sched):
                    kind = _sched[li]
                    wrong_kind = {s for s, ks in layer_kinds.items() if kind in ks}

        hit, plain, miss = [], [], []
        for s, v in ordered:
            if s in wrong_kind:
                if not strict.get(s):
                    miss.append((s, v))
                continue
            rx = scopes.get(s)
            if rx is None:
                plain.append((s, v))
            else:
                m = rx.search(module_path)
                if m:
                    hit.append((m.start(), s, v))
                elif not strict.get(s):
                    miss.append((s, v))
        # Among symbols that all match this module, the one matching DEEPER in the path wins.
        # Scope regexes match anywhere in `model.layers.*.self_attn.compressor.indexer.scorer.
        # weights_proj`, so a symbol scoped to the enclosing module (`n_h`, scope `attn|attention`,
        # matching at `self_attn`) beat one scoped to the module actually holding the parameter
        # (`n_h_I`, scope `indexer|scorer`) purely on global priority. The source says
        # `weights_proj = nn.Linear(hidden_size, config.index_n_heads)` -- and index_n_heads ==
        # num_attention_heads == 64 in DeepSeek-V4, so nothing about the value could reveal it.
        # Nesting IS the evidence: the innermost scope that claims a module is the one that built
        # it. Global priority still breaks ties at equal depth.
        #
        # A SELECTION count is never promoted this way. `k_I` (index_topk) is scoped to `indexer`
        # and therefore sits deeper than `c_q` (q_lora_rank, scoped to `attn`), and on depth alone
        # it took over the in-features of GLM-5.2's `indexer.wq_b`, where the two values coincide
        # -- 156 axes worse. How many entries a stage keeps is a routing quantity: it can describe
        # what a stage produced, never how wide the parameter feeding it is. Depth is evidence
        # about WHICH MODULE a name belongs to, and it cannot promote a name that does not belong
        # to that kind of axis in the first place.
        hit.sort(key=lambda t: -t[0])
        return [(s, v) for _d, s, v in hit], plain, miss

    stats: "collections.Counter[str]" = collections.Counter()
    weak: "collections.Counter[tuple]" = collections.Counter()
    ties: "collections.Counter[tuple]" = collections.Counter()

    def _dim_core(n, module_path=None, avoid=None, prev=None, is_weight=False,
                  forbid=None, t_dep=None):
        """Every `return` goes through `_r(kind, label)`, which records WHICH rule produced the
        name. Without that the output cannot distinguish a name DERIVED from a scoped rule from
        one that merely happens to match a number -- and the arithmetic tail below is known to
        fabricate names that are true at this seq_len and false at any other. See `.stats`."""
        def _r(kind, label):
            stats[kind] += 1
            if kind.startswith("heur") or kind == "bare":
                weak[(kind, module_path or "", str(label))] += 1
            return label

        if not isinstance(n, int) or isinstance(n, bool):
            return _r("passthrough", str(n))
        if n == 1:
            return _r("runtime", "B")  # batch (and any genuine singleton, MQA n_kv=1 -- C7/symbols)
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
            #
            # Selection counts are NOT excluded here, though the argument above nearly extends to
            # them. It does not: DeepSeek-V4's router carries `tid2eid`, a real parameter of shape
            # `[V, num_experts_per_tok]`, and dropping `k` from weight axes turned it into a bare
            # `[V, 6]`. Their one dangerous case -- winning a value tie on depth alone -- is
            # handled where it actually arises, in _pick.
            ordered_ctx = [(s, v) for s, v in ordered_ctx if s != "T"]
            hit_syms = [(s, v) for s, v in hit_syms if s != "T"]
            plain_syms = [(s, v) for s, v in plain_syms if s != "T"]
            miss_syms = [(s, v) for s, v in miss_syms if s != "T"]

        # Same invariant as above: a derived formula containing T cannot describe a weight axis.
        #
        # `t_dep` extends the same idea to ACTIVATIONS, on measured evidence rather than a
        # physical argument: src/tdep.py compares this axis between the prefill and decode
        # traces. If the size moved, the axis depends on the sequence length and no fixed
        # config symbol can name it; if it did not move, no T-bearing expression can. That
        # settles a value collision one trace cannot -- GLM-4.5-Air's routed-slot axis is 128
        # in prefill and 8 in decode, so it is `k*T`, not `E` (128 in both). None = no
        # evidence, so no filter.
        def _t_ok(label):
            has_t = bool(re.search(r"\bT\b", str(label)))
            if t_dep is True:
                return has_t
            # The `did not move` verdict is deliberately NOT enforced. It is sound in principle
            # (an axis with the same size in both phases cannot be named with T) but measuring it
            # fleet-wide cost DeepSeek-V4-Flash 900 axes to bare and moved flow_ambig on five
            # models: rejecting a T-bearing name leaves nothing registered to put in its place,
            # so an honest-but-nameless integer replaces a name that reads correctly at this
            # seq_len. Enforcing it needs the missing rules first. Only the positive verdict --
            # which REPLACES a wrong name with a right one -- is acted on.
            return True

        def _ok(formula):
            if is_weight and re.search(r"\bT\b", formula):
                return False
            return _t_ok(formula)

        def _pick(cands):
            """Exact matches within one tier, with the count-then-size positional tie-break.

            When several symbols tie on value (Llama-3.1-405B has n_h == d_head == 128) and the
            axis immediately before this one resolved to a head/expert-COUNT symbol, prefer a
            SIZE symbol here: a count axis is essentially always followed by that unit's own dim,
            never by a second unrelated count. Leaves gpt-oss's attention-sink shape
            `[B, n_h, T, T+1]` (also n_h == d_head == 64) alone -- there the preceding axis is
            `B`, so the tie-break never fires."""
            ms = [s for s, v in cands if n == v and (not avoid or s not in avoid)
                  and (not forbid or s not in forbid) and _t_ok(s)]
            if not ms:
                return None
            # AMBIGUITY, RECORDED. Two symbols with the same value in the same tier means the rules
            # have no way to choose -- whichever wins does so by global priority, which is a
            # convention, not evidence. The label may well be right, but nothing here KNOWS that,
            # and until now the output looked exactly like a confident one. Every such choice is
            # logged so the ④-layer review can be pointed straight at it (review/04-full-inventory).
            if len(ms) > 1:
                ties[(module_path or "", n, tuple(sorted(ms)))] += 1
            # A SELECTION count never wins a value tie against a symbol the rules rank above it.
            # Depth (see _ctx_symbols) is evidence about which module a name belongs to, and it
            # correctly moved DeepSeek-V4's indexer onto `n_h_I`/`c_I`. But it also let `k_I`
            # (index_topk, scoped to `indexer`) outrank `c_q` (q_lora_rank, scoped to `attn`) on
            # GLM-5.2, where index_topk == q_lora_rank == 2048 -- and a compressed-Q latent width
            # is not "how many entries the indexer kept". Global priority already encodes which of
            # two colliding names is the more fundamental, so a selection count may only win when
            # nothing better-ranked is in the running. DeepSeek-V3's `k` (priority 22) still beats
            # `n_grp` (38) in the router, which is the right answer there.
            if ms[0] in _SELECTION_SYMS and len(ms) > 1:
                order = {s: i for i, (s, _v) in enumerate(ordered)}
                better = [s for s in ms[1:]
                          if s not in _SELECTION_SYMS
                          and order.get(s, 10 ** 6) < order.get(ms[0], 10 ** 6)]
                if better:
                    ms = better + [m for m in ms if m not in better]
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
            return _r("scoped_symbol", r)
        scope_path = _scope_path(module_path)
        if scope_path:                             # 2. derived formula scoped TO this module
            for rx, m in authoritative_scoped:
                if n in m and rx.search(scope_path) and _ok(m[n]):
                    return _r("scoped_formula", m[n])
        r = _pick(plain_syms)                      # 3. unscoped plain symbol
        if r:
            return _r("plain_symbol", r)
        # 3b. symbol whose scope EXCLUDES this module. Kept as a last resort because scopes are
        # imperfect, but NOT for symbols that declare a `group`: a group tag says the symbol
        # belongs to one architecture family, so letting it name an axis in another family is
        # never a near-miss, it is a category error. Found by review 2026-08-09 -- Nemotron-3's
        # Mamba chunk-state axis (2) was named `k`, the MoE experts-per-token count, purely
        # because the numbers matched and no other symbol was left.
        r = _pick([(s, v) for s, v in miss_syms if not (spec_all.get(s) or {}).get("group")])
        if r:
            return _r("out_of_scope_symbol", r)
        # A scoped derived rule that did NOT match this module must not fire globally -- that is
        # the whole point of the scope (Llama-4: 2*n_kv*d_head is meaningless inside `experts`,
        # where 2048 is really E*T and the T-product rule below gets it right).
        if n in authoritative and not any(n in m for _rx, m in authoritative_scoped) \
                and _ok(authoritative[n]):
            return _r("derived_formula", authoritative[n])  # derived_dims explains it exactly
        # A plain symbol we skipped only because it was already used elsewhere in this same
        # shape tuple (see resolve_shape) is still a better answer than falling through to a
        # heuristic guess or a bare literal -- reuse is allowed once no fresh alternative exists.
        # `forbid` is NOT relaxed here the way `avoid` is: reusing a name already spent in this
        # tuple is merely redundant, but a mutually-exclusive name is affirmatively false.
        if avoid:
            for s, v in ordered_ctx:
                if n == v and (not forbid or s not in forbid) and _t_ok(s):
                    return _r("reused_symbol", s)
        # ---- 아래는 전부 휴리스틱(등록된 규칙이 아니다) ----
        # 여기서는 **스코프 밖 심볼을 쓰지 않는다**. 평범한 매칭에서 out-of-scope를 드롭하지 않고
        # 강등만 하는 이유는 스코프 정규식이 모든 작명 관례를 담을 수 없기 때문인데(xLSTM의
        # `mlstm_layer`), 그건 "이름을 그대로 붙일 때"의 이야기다. 스코프가 명시적으로 배제한
        # 심볼로 **식을 지어내는** 것은 그보다 훨씬 약한 근거이고, 실제로 산술적으로만 참인
        # 이름을 대량 생산했다: self_attn 안에서 `4*k`(k=expert top-k), `4*E`(E=expert 수),
        # `k/2`가 RoPE·DeltaNet 축에 붙었다. 게이트는 값이 맞으니 전부 통과시켰고 자유 평가에서
        # 나왔다(2026-07-31). 근거가 약한 자리에서는 이름을 짓기보다 정수로 남기는 게 정직하다.
        heur_ctx = hit_syms + plain_syms
        for s, v in heur_ctx:             # symbol + 1 (e.g. cache length T+1)
            if n == v + 1 and v >= 16 and _t_ok(f"{s}+1"):    # small symbols would give noise like "g_o+1"
                return _r("heur_plus1", f"{s}+1")
        # Small multiple (e.g. 2*d_moe for gate+up). T is EXCLUDED: a constant weight axis that
        # happens to equal 2x the traced seq_len (DeepSeek-V4 n_h*d_head/g_o = 4096 at T=2048)
        # would otherwise render "2*T" and fabricate a sequence dependency on a fixed dim -- the
        # exact thing P1 forbids. resolve_seq_len() only de-collides single dims, not multiples,
        # so the guard has to be here. A genuine 2*T stays an honest literal instead.
        for c in (2, 3, 4):
            for s, v in heur_ctx:
                if s != "T" and n == c * v and _t_ok(f"{c}*{s}"):
                    return _r("heur_multiple", f"{c}*{s}")
        # product of two symbols, but ONLY when one factor is T (the runtime dim): T*k routed
        # tokens, B*T flattened, etc. Pure structural×structural products (e.g. c_kv*k) are
        # almost always coincidental collisions, not real derived dims -- skip them so a static
        # weight dim like n_h*(d_nope+d_rope)=3072 stays an honest literal instead of "c_kv*k".
        # For a weight this whole rule is unreachable by construction (T was dropped from
        # ordered_ctx above), which is exactly what killed gpt2-xl's bogus `wpe = [T*d_head, ...]`.
        for i, (s1, v1) in enumerate(heur_ctx):
            for s2, v2 in heur_ctx[i:]:
                # T*T excluded: a SINGLE axis of size T² is essentially always a coincidence, not a
                # real dim -- a genuine quadratic quantity (an attention score matrix) appears as
                # TWO axes `[..., T, T]`, never one. xLSTM's per-head qk width 256 came out `T*T`
                # purely because the traced T was 16; it would be false at any other seq_len, which
                # is exactly the fabricated sequence-dependence P1 forbids (free-form review,
                # 2026-07-31).
                if v1 * v2 == n and "T" in (s1, s2) and not (s1 == "T" and s2 == "T") \
                        and t_dep is not False:
                    # Canonical operand order (T last), so the SAME product never gets two
                    # spellings. `ordered_ctx` is reordered per module scope, so without this the
                    # identical tensor came out `E*T` in one op and `T*E` in the next -- caught by
                    # the dataflow-consistency audit, 2026-07-30.
                    return _r("heur_product", f"{s2}*{s1}" if s1 == "T" else f"{s1}*{s2}")
        # Half dim (RoPE inv_freq / rotate_half use d_head/2). Same `v >= 16` floor as the +1
        # rule above, and for the same reason: halving a small symbol names a small number after
        # something it has nothing to do with. Zamba2's `d_conv` is 4, so every literal 2 in the
        # Mamba block came out `d_conv/2` -- including the 2 of a `concat` of two [B,1,...]
        # tensors, which is a count of operands, not a width. 4,468 axes across d_conv/2,
        # m_csa/2, d_conv_lin/2 and k_grp/2, all of them 4 -> 2. The genuine cases are untouched:
        # d_rope/2 (64) and d_head/2 (128/256/512) are the real rotate_half split.
        for s, v in heur_ctx:
            if s != "T" and v >= 16 and v % 2 == 0 and n == v // 2 and _t_ok(f"{s}/2"):
                return _r("heur_half", f"{s}/2")
        if t_dep is True:
            # "this axis moved between the phases" is a PREFERENCE, not a licence to give up. If
            # no T-bearing name fits, the evidence still stands but we have no vocabulary for it,
            # and a bare integer is worse than the ordinary answer. Retry unfiltered. (The
            # opposite verdict IS a hard filter: an axis that did not move cannot be named with
            # T, and there `bare` is the honest outcome.) Without this fallback the filter cost
            # DeepSeek-V4-Flash 900 axes to bare and raised heur on five models.
            return _dim_core(n, module_path, avoid=avoid, prev=prev, is_weight=is_weight,
                             forbid=forbid, t_dep=None)
        return _r("bare", str(n))         # irreducible -> keep the number (do not fabricate)


    def dim(n, module_path=None, avoid=None, prev=None, is_weight=False, forbid=None,
            t_dep=None):
        """Ordinary resolution, except where the phase evidence CONTRADICTS the answer.

        Narrow on purpose. `t_dep is True` means this axis changed size between the prefill and
        decode traces, so no fixed config symbol can be its name -- but that verdict is acted on
        only when the ordinary answer is exactly such a symbol AND a T-bearing alternative
        exists. Requiring a T-bearing name for every axis that moved was measured and rejected:
        it forces a name where the rules have no vocabulary, moving flow_ambig on five models
        (DeepSeek-V4-Pro 183 -> 935) and pushing 900 DeepSeek-V4-Flash axes to bare. Here a name
        changes only when a wrong one can be replaced by a right one -- the case the evidence was
        built for, GLM-4.5-Air's routed-slot axis reading `E` when it is `k*T`.
        """
        snap_s, snap_w = collections.Counter(stats), collections.Counter(weak)
        plain = _dim_core(n, module_path, avoid=avoid, prev=prev, is_weight=is_weight,
                          forbid=forbid, t_dep=None)
        if t_dep is not True or plain in ("T", "B") or not str(plain).isidentifier():
            return plain          # no verdict, or the answer is not a bare config symbol
        mid_s, mid_w = collections.Counter(stats), collections.Counter(weak)
        forced = _dim_core(n, module_path, avoid=avoid, prev=prev, is_weight=is_weight,
                           forbid=forbid, t_dep=True)
        keep_plain = not _HAS_T.search(str(forced))
        # exactly one of the two probes is the published answer, so only its tally survives
        chosen_s = (mid_s - snap_s) if keep_plain else (collections.Counter(stats) - mid_s)
        chosen_w = (mid_w - snap_w) if keep_plain else (collections.Counter(weak) - mid_w)
        stats.clear(); stats.update(snap_s); stats.update(chosen_s)
        weak.clear(); weak.update(snap_w); weak.update(chosen_w)
        return plain if keep_plain else forced

    def resolve_shape(shape, module_path=None, is_weight=False, pin=None, t_dep=None):
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
        # A tensor has ONE batch axis. dim() answers "B" for every size-1 axis, which is right for
        # the leading one and wrong for every broadcast singleton after it -- 34% of all rendered
        # shapes came out with `B` twice or more (`[B,T,B]` from a mean's reduced axis,
        # `[B,n_h,B]`, `[B,T,n_hc,B]`). Those later ones are literal 1s, so say 1. Found by
        # reading the layer-3 review packet, 2026-08-05.
        # ...and a `B` that comes AFTER the sequence axis is not batch either: every HF layout
        # puts batch ahead of sequence ([B,T,d], [B,n_h,T,d]), so the trailing 1 in a router's
        # `sum(...,keepdim=True)` -> [T,1] is a reduced axis, not a batch. 2,301 more shapes.
        seen_batch = "B" in (out or [])
        seen_seq = "T" in (out or [])
        for i, x in enumerate(shape):
            if out[i] is not None:          # pinned axis already decided
                prev = out[i] if out[i] in plain_symbol_names else None
                continue
            r = dim(x, module_path, avoid=used, prev=prev, is_weight=is_weight,
                    forbid=banned, t_dep=(t_dep or {}).get(i))
            if r == "B":
                if seen_batch or seen_seq:
                    r = "1"
                else:
                    seen_batch = True
            elif r == "T":
                seen_seq = True
            out[i] = r
            prev = r if r in plain_symbol_names else None
            _claim(r)
        # Enforce the same invariant on the finished shape, so a PINNED axis cannot smuggle a
        # second `B` past the loop above (Qwen3-Next did: a pinned [T,B] on 48 matmul/sigmoid rows).
        # A WEIGHT has no batch axis at all -- same physical argument as "a weight axis cannot be
        # T": the parameter is allocated from config at load time. Qwen3-Next's shared_expert_gate
        # is nn.Linear(d_model, 1), so its out_features rendered `B`, and _propagate_labels then
        # carried that B onto the matmul output ([T,B] on 96 rows).
        seen_b, seen_t = is_weight, False
        for i, lab in enumerate(out):
            if lab == "B":
                if seen_b or seen_t:
                    out[i] = "1"
                else:
                    seen_b = True
            elif lab == "T":
                seen_t = True
        return out

    resolve_shape.table = {"B": 1, **{s: v for s, v in ordered}}
    resolve_shape.stats = stats            # rule -> how many axes it named
    resolve_shape.weak = weak              # (rule, module_path, label) -> count, heuristics only
    resolve_shape.cfg = cfg                # the layer schedule, for label_overrides' block filter
    resolve_shape.ties = ties              # (module, value, candidates) -> count, arbitrary picks

    def _label_of(value, module_path=None):
        """The label this resolver publishes for one axis. For REPORTING only -- it deliberately
        restores the tally afterwards so asking a question does not change the provenance counts."""
        keep_stats, keep_weak, keep_ties = (collections.Counter(stats), collections.Counter(weak),
                                            collections.Counter(ties))
        try:
            return _dim_core(value, module_path)
        finally:
            stats.clear(); stats.update(keep_stats)
            weak.clear(); weak.update(keep_weak)
            ties.clear(); ties.update(keep_ties)

    resolve_shape.label_of = _label_of
    return resolve_shape
