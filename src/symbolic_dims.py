"""Read each module's declared widths as a config EXPRESSION, by running the model's own code.

WHY
---
`symbolic_shape.dim()` receives a bare integer and searches config for something equal to it.
When two config fields hold the same number, `priority:` decides -- and is then guaranteed wrong
in one of the two contexts. Llama-3.1-405B has d_model == n_h*d_head == 16384; Llama-4 has
E == d_head == 128. Every labelling bug this project has found is that one design.

A human does not have the problem, because they read the construction:

    self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim)

Parsing that source with `ast` is the obvious way to automate it and it is brittle: the
expression is rarely inline (`self.head_dim = config.head_dim or hidden // heads`), the attribute
name is not the runtime module path, and every repo writes it differently.

So instead: RUN the code and let the arithmetic carry its own name. Each dim-bearing config field
is replaced by `Dim(value, symbol)` -- an `int` subclass whose operators propagate an expression.
`nn.Linear.__init__` stores whatever it was handed, so afterwards

    model.layers.0.self_attn.q_proj.out_features.expr == "n_h*d_head"
    model.layers.0.self_attn.q_proj.in_features.expr  == "d_model"     # both are 4096

is read straight off the built module. Being an `int` subclass, torch and transformers treat it
as an ordinary integer -- the model code is unchanged and unaware.

WHAT IT DOES NOT DO
-------------------
Only modules that DECLARE their widths are covered (nn.Linear, nn.Embedding, and any module that
caches a config int as an attribute). That is ~1.8% of all traced axes -- the anchors. Everything
else is intermediate activation, which `anchors.propagate` spreads to along the dataflow.

Where a tag is lost (an explicit `int(x)`, or arithmetic that happens in C), the value falls back
to a plain integer and the existing value matching handles it. Degrading to today's behaviour is
the worst case; a wrong name is not.
"""
import torch.nn as nn

from summarize import derived_symbols, load_symbols, resolve_symbols


def _e(x):
    """Expression for a value: its tag if it has one, else the bare number."""
    return getattr(x, "expr", None) or str(int(x))


# Symbols that resolve_symbols DERIVED rather than read from a config field -- `d_head` when the
# config has no `head_dim` and summarize computes `d_model // n_h`. Set by tag_config().
_DERIVED: dict = {}


def _atomize(value: int, expr: str) -> str:
    """Prefer a derived symbol's own NAME over the arithmetic that reconstructs it.

    Without this, a model that computes `self.head_dim = hidden_size // num_heads` makes q_proj's
    width come out `n_h*(d_model/n_h)` -- which sympy then happily simplifies to `d_model`. That
    is semantically WRONG: a Q projection's output is the packed head layout, not the residual
    stream. They are numerically equal, so no amount of later simplification can tell them apart;
    the distinction has to be kept here, before the arithmetic collapses.

    Restricted to symbols we know are derived, so this is not general value matching: `d_head`
    was defined as `d_model // n_h` by summarize.resolve_symbols itself, and we are only choosing
    its declared name over its definition.
    """
    if not _DERIVED:
        return expr
    for sym, val in _DERIVED.items():
        if val == value and expr != sym:
            return sym
    return expr


class Dim(int):
    """An int that remembers which config expression produced it.

    Subclasses `int` deliberately: every isinstance check, `torch.empty(...)` call and strict
    dataclass validator keeps working. Only the arithmetic operators are overridden, to carry
    the name along -- so `config.num_attention_heads * self.head_dim` arrives at nn.Linear
    already knowing it is `n_h*d_head`.
    """

    def __new__(cls, value, expr):
        o = super().__new__(cls, value)
        o.expr = expr
        return o

    def _bin(self, other, op, sym, swap=False):
        try:
            v = op(int(self), int(other))
        except Exception:
            return NotImplemented
        a, b = (_e(other), _e(self)) if swap else (_e(self), _e(other))
        if sym == "*" and b == "1":
            return Dim(v, a)
        expr = f"({a}{sym}{b})" if sym in "+-" else f"{a}{sym}{b}"
        return Dim(v, _atomize(v, expr))

    def __mul__(self, o):       return self._bin(o, lambda x, y: x * y, "*")
    def __rmul__(self, o):      return self._bin(o, lambda x, y: y * x, "*", swap=True)
    def __add__(self, o):       return self._bin(o, lambda x, y: x + y, "+")
    def __radd__(self, o):      return self._bin(o, lambda x, y: y + x, "+", swap=True)
    def __sub__(self, o):       return self._bin(o, lambda x, y: x - y, "-")
    def __rsub__(self, o):      return self._bin(o, lambda x, y: y - x, "-", swap=True)
    def __floordiv__(self, o):  return self._bin(o, lambda x, y: x // y, "/")
    def __rfloordiv__(self, o): return self._bin(o, lambda x, y: y // x, "/", swap=True)
    def __truediv__(self, o):   return self._bin(o, lambda x, y: x // y, "/")

    # transformers deep-copies the config (to_dict / generation config) and the default int
    # reconstructor calls __new__ with only the value, which would drop the tag.
    def __reduce__(self):
        return (Dim, (int(self), self.expr))

    def __deepcopy__(self, memo):
        return Dim(int(self), self.expr)

    def __repr__(self):
        return f"{int(self)}<{self.expr}>"


def tag_config(cfg, symbols: dict | None = None) -> dict:
    """Replace每 dim-bearing config field with a named `Dim`. Returns {field: symbol}.

    The symbol -> value mapping comes from `summarize.resolve_symbols`, NOT from walking
    `aliases:` directly. That matters: resolve_symbols applies the deterministic corrections
    that live in Python rather than in the YAML (Llama-4 keeps its expert width in plain
    `intermediate_size`, and summarize.py falls back to it when `moe_intermediate_size` is
    absent). Reading aliases directly missed that and left 8192 untagged, while resolve_symbols
    also yields every architecture-specific symbol the model actually has (c_kv, d_nope, m_csa,
    g_o, n_hc, d_state ...) -- which is the single biggest lever on coverage.

    Crucially it also returns None for symbols a model does NOT have, so a dense model never
    acquires a spurious `d_moe`.
    """
    spec = symbols if symbols is not None else load_symbols()
    values = resolve_symbols(cfg, spec)
    tagged = {}
    # A symbol with a resolved value but NO config field carrying it was derived by
    # resolve_symbols (d_head = d_model // n_h when there is no head_dim field). Record those so
    # _atomize can keep the declared name instead of the arithmetic -- see its docstring.
    _DERIVED.clear()
    ints = _config_ints(cfg)
    for sym, val in values.items():
        if not isinstance(val, int) or isinstance(val, bool) or val < 2:
            continue
        if not (spec.get(sym) or {}).get("dim"):
            continue
        if not any(getattr(cfg, a, None) == val
                   for a in ((spec.get(sym) or {}).get("aliases") or [])):
            if list(ints.values()).count(val) == 0:
                _DERIVED[sym] = int(val)
    # Two passes, most-specific claim first. Some configs make two fields THE SAME OBJECT:
    # DeepSeek-V4's `intermediate_size` and `moe_intermediate_size` satisfy `is`, so tagging one
    # tags both, and the second symbol then finds a Dim already in place and skips. d_ff matched
    # `intermediate_size` (its 2nd alias) while d_moe matched `moe_intermediate_size` (its 1st and
    # only), so d_ff won and every shared expert was labelled `d_ff`. Letting a first-alias match
    # go first hands the field to its most specific owner -- and agrees with the existing
    # convention in symbolic_shape._config_values, where d_moe wins when the two coincide.
    def _claim_rank(sym):
        aliases = (spec.get(sym) or {}).get("aliases") or []
        return 0 if (aliases and getattr(cfg, aliases[0], None) is not None) else 1

    for sym, val in sorted(values.items(), key=lambda kv: _claim_rank(kv[0])):
        if not isinstance(val, int) or isinstance(val, bool) or val < 2:
            continue
        if not (spec.get(sym) or {}).get("dim"):
            continue                     # L / ctx / layer_sched are not tensor dimensions
        # Tag EVERY alias carrying this value, not just the first. A config can name the same
        # quantity twice -- Qwen3-Next has `moe_intermediate_size` and
        # `shared_expert_intermediate_size` both at 512, and the shared-expert MLP is built from
        # the second, so stopping at the first left that module's width unnamed. Requiring the
        # value to equal the resolved symbol keeps it safe: an alias holding a different number
        # (a model whose shared expert really is a different width) is skipped.
        hit = None
        for alias in ((spec.get(sym) or {}).get("aliases") or []):
            cur = getattr(cfg, alias, None)
            if isinstance(cur, int) and not isinstance(cur, bool) and not isinstance(cur, Dim) \
                    and int(cur) == int(val):
                setattr(cfg, alias, Dim(int(val), sym))
                tagged[alias] = sym
                hit = alias
        if hit is None:
            # resolve_symbols knows the VALUE (via a correction that lives in Python) but the
            # alias list does not say which FIELD carries it -- Llama-4 keeps the expert width in
            # plain `intermediate_size`, which is not one of d_moe's aliases. Locate the field by
            # value, but ONLY when exactly one config field holds it. This is not blind value
            # matching: the value is already established by the authoritative resolver and we are
            # merely finding its home. Where several fields collide (Llama-4 has
            # intermediate_size == attention_chunk_size == floor_scale == 8192) we refuse and let
            # unregistered_fields() report it, because guessing here is how wrong names are born.
            # ...but never claim a field that is another symbol's DECLARED alias. DeepSeek-V4 has
            # no `intermediate_size`, so d_ff resolves to the same 2048 as d_moe; without this
            # guard d_ff grabbed `moe_intermediate_size` -- d_moe's own alias -- and the shared
            # expert's width came out `d_ff`. An explicit alias always outranks a value match.
            # Exclude only fields ALREADY TAGGED, not every field some symbol merely declares.
            # The stricter version excluded the right answer: Llama-4's `intermediate_size` is
            # declared by d_ff, but d_ff actually resolved to `intermediate_size_mlp`, so the
            # field was free -- and refusing it left `floor_scale` (a RoPE constant that happens
            # to be 8192) as the sole candidate, which was then tagged `d_moe`.
            cand = [f for f, cur in _config_ints(cfg).items()
                    if cur == int(val) and f not in tagged and not _is_non_dim(f)]
            if len(cand) == 1:
                hit = cand[0]
        # Symbols whose value lives inside a config DICT rather than a top-level field
        # (`from: {field: compress_rates, key: ...}`). DeepSeek-V4's compressor reads
        # `config.compress_rates["heavily_compressed_attention"]` straight into an attribute, so
        # tagging the dict entry is the only way that width can carry a name.
        frm = (spec.get(sym) or {}).get("from") or {}
        if hit is None and frm.get("field") and frm.get("key") is not None:
            container = getattr(cfg, frm["field"], None)
            if isinstance(container, dict):
                cur = container.get(frm["key"])
                if isinstance(cur, int) and not isinstance(cur, bool)                         and not isinstance(cur, Dim) and int(cur) == int(val):
                    container[frm["key"]] = Dim(int(val), sym)
                    tagged[f'{frm["field"]}[{frm["key"]}]'] = sym
                    # tagged IN PLACE; the field itself is a dict and must never be replaced by a
                    # Dim -- the config is a strict dataclass and rejects the type outright.
                    continue
        if hit is not None and not isinstance(getattr(cfg, hit, None), Dim):
            setattr(cfg, hit, Dim(int(val), sym))
            tagged[hit] = sym

    # Fields whose value a VERIFIED derived formula explains exactly. DeepSeek-V3 precomputes
    # `qk_head_dim = qk_nope_head_dim + qk_rope_head_dim` in its config, so no symbol alias can
    # reach it -- but rules/derived_dims.yaml already knows that sum. Reusing those rules keeps
    # the "only a checked formula, never a guess" property (P1); an unexplained value stays an
    # untagged integer and shows up in the unregistered-field report.
    # Scoped rules count too. `scope:` restricts where a formula may name an AXIS; here we are
    # identifying what a CONFIG FIELD means, which is not module-local. DeepSeek-V3's 192 is only
    # in the scoped maps, so consulting the global one alone found nothing. To stay safe the value
    # is used only when every rule that explains it agrees on one formula.
    try:
        from summarize import derived_symbols
        # NOTE: no `spec=` here. That parameter is a structure spec, not the symbols table;
        # passing the symbols table silently returned zero scoped rules.
        glob_map, scoped = derived_symbols(values, cfg=cfg, seq_len=None)
        glob_map = dict(glob_map)
        agree = {}
        for _rx, m in scoped:
            for v, f in m.items():
                agree.setdefault(v, set()).add(f)
        for v, fs in agree.items():
            if len(fs) == 1 and v not in glob_map:
                glob_map[v] = next(iter(fs))
    except Exception:
        glob_map = {}
    if glob_map:
        for field, cur in _config_ints(cfg).items():
            if cur in glob_map and not isinstance(getattr(cfg, field, None), Dim):
                setattr(cfg, field, Dim(int(cur), glob_map[cur]))
                tagged[field] = glob_map[cur]
    return tagged


def _is_non_dim(name: str) -> bool:
    """Field names that are counts or scalar constants rather than tensor dimensions."""
    return any(t in name for t in _NOT_A_DIM)


def _config_ints(cfg) -> dict:
    d = cfg.to_dict() if hasattr(cfg, "to_dict") else vars(cfg)
    return {k: v for k, v in d.items()
            if isinstance(v, int) and not isinstance(v, bool) and not isinstance(v, Dim) and v >= 2}


# Attribute names that are not architecture dimensions. Excluding them keeps the "unregistered
# field" report honest -- layer_idx/padding_idx were never candidates for a symbol.
# Attribute names that are not architecture DIMENSIONS. Excluding them keeps the "unregistered
# field" report actionable: an iteration count, a RoPE scaling constant or a count of memory
# blocks is never a tensor axis, so reporting it as a naming gap is noise. Verified per model:
# DeepSeek-V4 `hc_sinkhorn_iters` (Sinkhorn iterations), Llama-4 `floor_scale` (RoPE), Zamba2
# `num_fwd_mem_blocks` (`for i in range(...)` building LoRA adapters, modeling_zamba2.py:276).
_NOT_A_DIM = ("idx", "_id", "rank", "seed", "version", "kernel_size", "groups",
              "stride", "dilation", "padding", "iters", "_scale", "mem_blocks")

# nn.Linear/nn.Embedding copy their widths onto these attributes, so an untagged width shows up
# once as the config field that caused it AND again here. Reporting both makes the research list
# noisier without adding a target -- the actionable name is the config field.
_DERIVED_ATTRS = ("in_features", "out_features", "num_embeddings", "embedding_dim")


def normalize(expr: str | None) -> str | None:
    """Algebraically tidy an expression, keeping its meaning.

    `n_h*(((d_nope+d_rope)-d_rope)+d_v)` -> `n_h*(d_nope + d_v)`, `(c_kv+d_rope)` ->
    `c_kv + d_rope`, `1*d_head` -> `d_head`. The RAW form is kept alongside (see
    module_expressions) because it is the evidence -- it records the arithmetic the model code
    actually performed.

    Safe because sympy is given no VALUES: the symbols are free variables, so it can only apply
    valid algebra and can never decide that `d_model` and `n_h*d_head` are interchangeable. The
    one collapse that would be valid algebra -- `n_h*(d_model/n_h)` -> `d_model` -- is prevented
    upstream by _atomize, which keeps `d_head` from being expanded into its definition in the
    first place. Order matters: atomize during construction, simplify only afterwards.
    """
    if not expr:
        return expr
    try:
        import sympy
        e = sympy.sympify(expr, evaluate=True)
        s = _fmt(sympy.simplify(e))
        # a sum is parenthesised so it composes (`n_h*(d_nope+d_v)`), but at top level the
        # existing labels write it bare (`T+1`, not `(T+1)`)
        if s.startswith("(") and s.endswith(")") and s.count("(") == 1:
            s = s[1:-1]
        return s
    except Exception:
        return expr                              # unparseable -> keep the raw text, never guess


def _fmt(e) -> str:
    """Render a sympy expression in this project's shape-cell convention.

    sympy orders a product alphabetically (`d_head*n_h`), while every existing label writes the
    COUNT first (`n_h*d_head`, `n_kv*d_head`, `2*d_moe`) and uses no spaces. Rendering the two
    conventions side by side in one table would read as two different quantities, so normalise
    to the established one -- a numeric coefficient first, then a count symbol, then the rest.
    """
    import sympy
    from symbolic_shape import _COUNT_LIKE_SYMS

    # symbolic_shape's set covers head/expert counts; a few more symbols count *groups* and read
    # the same way (`g_o*d_g`, not `d_g*g_o`). Extended only for rendering order.
    counts = set(_COUNT_LIKE_SYMS) | {"g_o", "n_hc", "n_g", "n_g_ssm", "E_shared", "n_k", "n_v"}

    def rank(a):
        if a.is_Number:
            return (0, str(a))
        return (1 if str(a) in counts else 2, str(a))

    if isinstance(e, sympy.Mul):
        # sympy stores `a/b` as Mul(a, Pow(b, -1)), which a naive factor join renders as
        # `a*1/b`. Split the fraction so it comes out `a/b` like every existing label
        # (`n_h*d_head/g_o`, `d_head/2`).
        num, den = sympy.fraction(sympy.together(e))
        if den != 1:
            return f"{_fmt(num)}/{_fmt(den)}"
        args = list(e.as_ordered_factors())
        return "*".join(_fmt(a) for a in sorted(args, key=rank))
    if isinstance(e, sympy.Add):
        args = sorted(e.as_ordered_terms(), key=lambda a: (a.is_Number, str(a)))
        return "(" + "+".join(_fmt(a) for a in args) + ")"
    return str(e).replace(" ", "")


def module_expressions(model) -> dict:
    """{module_path: {"in": expr|None, "out": expr|None, "attrs": {name: expr}}}

    Read AFTER building a model whose config was tagged. `in_features`/`out_features` cover
    nn.Linear; `attrs` catches everything else a module cached (num_experts, hidden_size,
    expert_dim ...), which is how 3-D expert parameters get explained -- their axes never keep a
    tag of their own, because torch.Size stores plain ints.
    """
    out = {}
    for name, mod in model.named_modules():
        rec = {"in": None, "out": None, "attrs": {}}
        i, o = getattr(mod, "in_features", None), getattr(mod, "out_features", None)
        if i is None and o is None:
            i, o = getattr(mod, "embedding_dim", None), getattr(mod, "num_embeddings", None)
        if isinstance(i, Dim):
            rec["in_raw"] = i.expr
            rec["in"] = normalize(i.expr)
        if isinstance(o, Dim):
            rec["out_raw"] = o.expr
            rec["out"] = normalize(o.expr)
        for k, v in vars(mod).items():
            if k.startswith("_") or not isinstance(v, int) or isinstance(v, bool) or v < 2:
                continue
            if any(t in k for t in _NOT_A_DIM):
                continue
            if isinstance(v, Dim):
                rec["attrs"][k] = v.expr
        if rec["in"] or rec["out"] or rec["attrs"]:
            out[name] = rec
    return out


def _derived_candidates(values) -> dict:
    """{value: {formula}} from rules/derived_dims.yaml, evaluated for THIS model.

    Recovers widths the tag could not reach. Zamba2 computes its Mamba inner width as
    `int(mamba_expand * hidden_size)` -- a float multiply, so `int()` is unavoidable and the tag
    dies there. The width is still `n_h_ssm * d_head_ssm`, and that is a registered, verified
    formula, so naming it from the rule is reading a rule rather than guessing (P1).
    """
    try:
        from summarize import derived_symbols
        glob_map, scoped = derived_symbols(values, cfg=None, seq_len=None)
    except Exception:
        return {}
    return {"global": glob_map or {}, "scoped": scoped or []}


def _candidates_for(derived, module_path: str) -> dict:
    """{value: {formula}} valid IN THIS MODULE.

    A `scope:` on a derived rule says where the formula may name an axis, so it has to be honoured
    here: 4096 inside Zamba2's `mamba` is d_inner, but the same number is also n_h*d_head, whose
    rule is scoped to attention. Merging every scoped map made all four candidates compete and the
    uniqueness test correctly refused -- which is safe but leaves the axis unnamed for a reason
    that scope already answers.
    """
    if not derived:
        return {}
    out = {}
    for v, f in (derived.get("global") or {}).items():
        out.setdefault(v, set()).add(f)
    for rx, m in (derived.get("scoped") or []):
        if rx.search(module_path or ""):
            for v, f in m.items():
                out.setdefault(v, set()).add(f)
    return out


# "an unnamed DIMENSION", not "any digit": a small coefficient (`2*d_state`) is normal notation,
# while a large or standalone number is a naming gap. Same distinction as anchors.tag_is_usable --
# getting it wrong once already made the half-tagged `(4096+2*d_state)` beat the complete
# `d_inner+2*n_g*d_state`, because both contain the character `2`.
_BARE_INT = __import__("re").compile(
    r"(?<![\w.])(?:1[6-9]|[2-9]\d|\d{3,})(?![\w.])|[+\-]\s*\d+(?!\s*[*\w.])")


def _simplest(formulas):
    """The one formula with strictly fewest operators, else None.

    `d_inner` and `d_inner/n_g` both evaluate to 4096 when n_g == 1 -- the second is the first
    with a degenerate divisor, not a competing reading. Preferring the simpler one resolves that
    without inventing anything; a genuine tie (two unrelated names) still refuses.
    """
    def key(f):
        # An unnamed integer inside the expression is a naming GAP, so a fully symbolic form
        # always wins however long it is. Ranking by length alone picked Zamba2's half-tagged
        # `(4096+2*d_state)` over the complete `d_inner+2*n_g*d_state`.
        return (1 if _BARE_INT.search(f) else 0, sum(f.count(o) for o in "*+-/"), len(f))

    ranked = sorted(formulas, key=key)
    if len(ranked) == 1:
        return ranked[0]
    return ranked[0] if key(ranked[0]) < key(ranked[1]) else None


def param_axis_expressions(model, derived: dict | None = None) -> dict:
    """{param_name: [expr | None, ...]} -- a name for each axis of every rank>=2 parameter.

    Covers the case `module_expressions` cannot: a module that holds a raw `nn.Parameter` instead
    of an nn.Linear has no in_features/out_features to read, and `torch.Size` stores plain ints so
    the tag never survives into the shape. MoE experts are all like this -- gpt-oss, Qwen3, Llama-4
    and every DeepSeek keep `gate_up_proj [E, d_model, 2*d_moe]` and `down_proj [E, d_moe, d_model]`
    as bare Parameters (~400 modules across the fleet).

    But the module DID cache the widths it was built from, and those attributes ARE tagged:
    Qwen3-30B's expert module carries num_experts->E, hidden_dim->d_model, intermediate_dim->d_moe.
    So each axis is matched against THAT module's own tagged attributes (and small multiples of
    them, since a fused gate+up axis is 2x the expert width).

    This is value matching, but over a candidate set of three or four semantically related numbers
    rather than the whole config -- and it refuses when the match is not unique, which is what
    stops gpt-oss (d_model == d_ff == 2880) from getting a coin-flip answer.
    """
    out = {}
    for mod_name, mod in model.named_modules():
        cands = {}
        for k, v in vars(mod).items():
            if k.startswith("_") or not isinstance(v, Dim):
                continue
            if any(t in k for t in _NOT_A_DIM) or k in _DERIVED_ATTRS:
                continue
            cands.setdefault(int(v), set()).add(v.expr)
            for c in (2, 3, 4):
                cands.setdefault(int(v) * c, set()).add(f"{c}*{v.expr}")
        # verified derived formulas, restricted to those whose scope covers this module
        for v, fs in _candidates_for(derived, mod_name).items():
            cands.setdefault(v, set()).update(fs)
        if not cands:
            continue
        for pname, p in mod.named_parameters(recurse=False):
            if p is None or p.dim() < 2:
                continue
            axes = []
            for d in p.shape:
                hit = cands.get(int(d))
                axes.append((next(iter(hit)) if len(hit) == 1 else _simplest(hit)) if hit else None)
            if any(axes):
                out[f"{mod_name}.{pname}" if mod_name else pname] = axes
    return out


def _explained_values(cfg) -> set:
    """Every width the rule set can already name for this model -- plain symbols and derived."""
    vals = {v for v in resolve_symbols(cfg).values()
            if isinstance(v, int) and not isinstance(v, bool)}
    glob, scoped = derived_symbols(resolve_symbols(cfg), cfg=cfg, seq_len=None)
    vals.update(glob)
    for _rx, m in scoped:
        vals.update(m)
    return vals


def unregistered_fields(model) -> dict:
    """{(attribute, value): count} for module integer attributes that carry NO tag.

    These are exactly the config fields this architecture uses that `rules/symbols.yaml` does not
    know about -- the tagger cannot name a width it was never told about. Today that gap is
    invisible, because value matching silently picks *something*; here it is reported. On
    Llama-3.1-8B the list is empty apart from `layer_idx`; on Llama-4 it surfaced
    `intermediate_size`/`expert_dim` = 8192, i.e. the real d_moe gap.

    Feeding this list into Tier 2 research (02-new-module-handling.md) turns "what should I even
    look up?" into a finite, checkable list.
    """
    gaps = {}
    for _name, mod in model.named_modules():
        for k, v in vars(mod).items():
            if k.startswith("_") or not isinstance(v, int) or isinstance(v, bool) or v < 2:
                continue
            if any(t in k for t in _NOT_A_DIM) or k in _DERIVED_ATTRS or isinstance(v, Dim):
                continue
            gaps[(k, int(v))] = gaps.get((k, int(v)), 0) + 1
    return gaps


def probe(model_id, revision=None, config_overrides=None) -> dict:
    """{"expressions": {...}, "unregistered": [{field, value, count}], "tagged": {field: symbol}}

    Builds a THROWAWAY tagged model. Deliberately a second, separate build rather than tagging the
    config the tracer uses: a Dim is an int subclass and should be inert, but the trace is the one
    artifact that must not be perturbed by a labelling experiment. On meta there are no weights,
    so the extra build is cheap.

    Returns {} on any failure -- this is a reporting aid, and it must never cost a completed run.
    """
    try:
        import loader
        import provenance
        cfg, _prov = provenance.snapshot(model_id, revision, config_overrides=config_overrides)
        tagged = tag_config(cfg)
        model = loader.load_meta(cfg, trust_remote_code=provenance.needs_remote_code(cfg))
        gaps = unregistered_fields(model)
        return {
            "tagged": tagged,
            "expressions": module_expressions(model),
            # `spec` was an undefined name here, so this line raised NameError on EVERY model and
            # the bare except below turned the whole probe into {"error": ...}. Everything it
            # feeds went silently empty: param_axes (so anchors rule 1 could never fire),
            # expressions (the tag-based labels), and unregistered (so all 26 models reported a
            # clean "no unregistered config fields" that was really a crash). Found 2026-08-06.
            "param_axes": param_axis_expressions(model, _derived_candidates(resolve_symbols(cfg))),
            # A field whose VALUE the rule set already explains is not a naming gap. Llama-4 has
            # `intermediate_size` and `expert_dim` both = 8192, which d_moe already resolves, and
            # Zamba2 has `intermediate_size` = `group_size` = 4096, which the registered
            # `n_h_ssm * d_head_ssm` (64*64) already derives -- reporting them sent the review
            # after a question that was answered. What matters is a width with NO name available,
            # not a second config field for a width that has one (③ 라벨 검토 2026-08-09).
            "unregistered": [{"field": f, "value": v, "modules": n}
                             for (f, v), n in sorted(gaps.items(), key=lambda x: -x[1])
                             if v not in _explained_values(cfg)],
        }
    except Exception as e:                       # noqa: BLE001 -- never lose a run over a report
        return {"error": f"{type(e).__name__}: {str(e)[:160]}"}
