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

from summarize import derived_symbols, load_symbols, resolve_symbols


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
    return vals


def resolve_seq_len(cfg, base: int, symbols: dict | None = None) -> int:
    """Smallest seq_len >= base that equals no config dimension value, so T (and T*k, T+1,
    ...) never collide with a config symbol. Keeps C14 valid (result is still >= base)."""
    avoid = set(_config_values(cfg, symbols).values())
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

    def _ctx_symbols(module_path):
        """Symbol list reordered for this module: in-scope first, unscoped next, out-of-scope last.

        A symbol whose `scope` matches THIS module outranks global priority -- Llama-4 has
        E == d_head == 128, so inside self_attn the head dim must not render as `E`.

        Out-of-scope symbols are DEMOTED, not dropped. Dropping was tried and caused a large
        regression: scope regexes cannot enumerate every naming convention (xLSTM's attention
        equivalent is `mlstm_layer`, which no `attn|attention` scope matches), so dropping turned
        88k correctly-symbolised dims into bare integers. Demotion can only improve a choice
        between equals, never remove a symbol that would otherwise apply."""
        if not module_path:
            return ordered
        hit, plain, miss = [], [], []
        for s, v in ordered:
            rx = scopes.get(s)
            if rx is None:
                plain.append((s, v))
            elif rx.search(module_path):
                hit.append((s, v))
            else:
                miss.append((s, v))
        return hit + plain + miss

    def dim(n, module_path=None):
        if not isinstance(n, int) or isinstance(n, bool):
            return str(n)
        if n == 1:
            return "B"  # batch (and any genuine singleton, e.g. MQA n_kv=1 -- see C7/symbols)
        ordered_ctx = _ctx_symbols(module_path)
        if module_path:
            for rx, m in authoritative_scoped:
                if n in m and rx.search(module_path):
                    return m[n]
        for s, v in ordered_ctx:          # exact single symbol
            if n == v:
                return s
        # A scoped derived rule that did NOT match this module must not fire globally -- that is
        # the whole point of the scope (Llama-4: 2*n_kv*d_head is meaningless inside `experts`,
        # where 2048 is really E*T and the T-product rule below gets it right).
        if n in authoritative and not any(n in m for _rx, m in authoritative_scoped):
            return authoritative[n]      # derived_dims explains it exactly -> use that formula
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
        for i, (s1, v1) in enumerate(ordered_ctx):
            for s2, v2 in ordered_ctx[i:]:
                if v1 * v2 == n and "T" in (s1, s2):
                    return f"{s1}*{s2}"
        for s, v in ordered_ctx:          # half dim (RoPE inv_freq / rotate_half use d_head/2)
            if s != "T" and v % 2 == 0 and n == v // 2:
                return f"{s}/2"
        return str(n)                     # irreducible -> keep the number (do not fabricate)

    def resolve_shape(shape, module_path=None):
        if shape is None:
            return None
        return [dim(x, module_path) for x in shape]

    resolve_shape.table = {"B": 1, **{s: v for s, v in ordered}}
    return resolve_shape
