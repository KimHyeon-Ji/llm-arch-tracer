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

from summarize import load_symbols, resolve_symbols


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
    for sym, val in values.items():
        if not isinstance(val, int) or isinstance(val, bool) or val < 2:
            continue
        if not (spec.get(sym) or {}).get("dim"):
            continue                     # L / ctx / layer_sched are not tensor dimensions
        hit = None
        for alias in ((spec.get(sym) or {}).get("aliases") or []):
            cur = getattr(cfg, alias, None)
            if isinstance(cur, int) and not isinstance(cur, bool) and not isinstance(cur, Dim) \
                    and int(cur) == int(val):
                hit = alias
                break
        if hit is None:
            # resolve_symbols knows the VALUE (via a correction that lives in Python) but the
            # alias list does not say which FIELD carries it -- Llama-4 keeps the expert width in
            # plain `intermediate_size`, which is not one of d_moe's aliases. Locate the field by
            # value, but ONLY when exactly one config field holds it. This is not blind value
            # matching: the value is already established by the authoritative resolver and we are
            # merely finding its home. Where several fields collide (Llama-4 has
            # intermediate_size == attention_chunk_size == floor_scale == 8192) we refuse and let
            # unregistered_fields() report it, because guessing here is how wrong names are born.
            cand = [f for f, cur in _config_ints(cfg).items() if cur == int(val)]
            if len(cand) == 1:
                hit = cand[0]
        if hit is not None:
            setattr(cfg, hit, Dim(int(val), sym))
            tagged[hit] = sym
    return tagged


def _config_ints(cfg) -> dict:
    d = cfg.to_dict() if hasattr(cfg, "to_dict") else vars(cfg)
    return {k: v for k, v in d.items()
            if isinstance(v, int) and not isinstance(v, bool) and not isinstance(v, Dim) and v >= 2}


# Attribute names that are not architecture dimensions. Excluding them keeps the "unregistered
# field" report honest -- layer_idx/padding_idx were never candidates for a symbol.
_NOT_A_DIM = ("idx", "_id", "rank", "seed", "version", "kernel_size", "groups",
              "stride", "dilation", "padding")

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
            "unregistered": [{"field": f, "value": v, "modules": n}
                             for (f, v), n in sorted(gaps.items(), key=lambda x: -x[1])],
        }
    except Exception as e:                       # noqa: BLE001 -- never lose a run over a report
        return {"error": f"{type(e).__name__}: {str(e)[:160]}"}
