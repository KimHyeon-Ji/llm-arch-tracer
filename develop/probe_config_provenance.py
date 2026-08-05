"""PROTOTYPE (not yet wired into the pipeline) -- identify WHICH config field produced each
module dimension, by construction rather than by value matching.

The problem it solves: symbolic_shape names an axis by searching config values for one that
equals the number. When two config fields hold the same number the name is decided by
`priority:`, which is guaranteed wrong in one of the two contexts. Llama-3.1-405B has
d_model == n_h*d_head == 16384; Llama-4 has E == d_head == 128.

A human does not have this problem, because they read the construction:

    self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim)

This recovers the same fact without parsing source. Perturb ONE config field, rebuild on meta
(no weights, cheap), and see which module dims move and by how much. A module dim is a
multilinear function of the config fields, so the partial derivatives are the cofactors:

    dim = n_h * d_head   =>   d(dim)/d(d_head) = n_h ,  d(dim)/d(n_h) = d_head

Every reading is then CHECKED against the dim's actual value, so a shape that does not fit
is reported UNRESOLVED instead of guessed. Verified on the two known collision models:

    Llama-3.1-405B  q_proj  out=16384 -> d_head*n_h    in=16384 -> d_model
                    o_proj  out=16384 -> d_model       in=16384 -> d_head*n_h
    Llama-4         router  out=128   -> E             (not d_head)

Cost is one meta rebuild per candidate field (~7-10 per model). Intended use is to compute the
map ONCE per model and store it in provenance.json, so tracing stays cheap.

Run:  .venv/Scripts/python.exe develop/probe_config_provenance.py <hf-model-id>
"""
import copy
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import loader          # noqa: E402  (installs the transformers 5.x shims)
import provenance      # noqa: E402
import summarize       # noqa: E402


def declared(model):
    """{module_path: (out_features, in_features)} for every module that declares them."""
    out = {}
    for name, mod in model.named_modules():
        i = getattr(mod, "in_features", None)
        o = getattr(mod, "out_features", None)
        if isinstance(i, int) and isinstance(o, int):
            out[name] = (o, i)
        elif isinstance(getattr(mod, "num_embeddings", None), int):
            out[name] = (mod.num_embeddings, mod.embedding_dim)
    return out


def probe(model_id, revision=None, fields=None, delta=1):
    cfg, _ = provenance.snapshot(model_id, revision)
    base_model = loader.load_meta(cfg, trust_remote_code=True)
    base = declared(base_model)
    del base_model

    syms = summarize.load_symbols()
    # candidate config fields = every alias any dim symbol declares, that this config actually has
    cand = {}
    for sym, spec in syms.items():
        if not spec.get("dim"):
            continue
        for alias in (spec.get("aliases") or []):
            if hasattr(cfg, alias) and isinstance(getattr(cfg, alias), int):
                cand.setdefault(alias, sym)
                break
    if fields:
        cand = {f: cand.get(f, f) for f in fields}

    vals = {sym: getattr(cfg, f) for f, sym in cand.items()}
    print(f"# {model_id}")
    print(f"  symbol values: {vals}")
    resp = {}                     # module -> {axis: {sym: coefficient}}
    for field, sym in cand.items():
        c2 = copy.deepcopy(cfg)
        try:
            setattr(c2, field, getattr(cfg, field) + delta)
            m2 = loader.load_meta(c2, trust_remote_code=True)
            d2 = declared(m2)
            del m2
        except Exception as e:
            print(f"  !! {field}: rebuild failed ({type(e).__name__}) -- skipped")
            continue
        moved = 0
        for mod, (o, i) in base.items():
            if mod not in d2:
                continue
            o2, i2 = d2[mod]
            for axis, (a, b) in (("out", (o, o2)), ("in", (i, i2))):
                if b != a:
                    resp.setdefault(mod, {}).setdefault(axis, {})[sym] = (b - a) / delta
                    moved += 1
        print(f"  {field:28s} -> {sym:12s} moved {moved} module axes")
    return base, resp, vals


def interpret(response, value, vals):
    """Turn {symbol: d(dim)/d(symbol)} + the dim's own value into a symbolic expression.

    The dim is a multilinear function of the config fields, so the partial derivatives ARE the
    cofactors:  dim = n_h*d_head  =>  d/d(d_head) = n_h  and  d/d(n_h) = d_head.
    Every reading below is CHECKED against the actual value, so a shape that does not fit the
    form is reported as unresolved rather than guessed."""
    if not response:
        return None
    items = sorted(response.items())
    if len(items) == 1:
        s, c = items[0]
        c = int(c)
        if vals.get(s) is not None and c * vals[s] == value:
            return s if c == 1 else f"{c}*{s}"
    if len(items) == 2:
        (s1, c1), (s2, c2) = items
        if vals.get(s1) and vals.get(s2) and vals[s1] * vals[s2] == value \
                and int(c1) == vals[s2] and int(c2) == vals[s1]:
            return f"{s1}*{s2}"
    return "UNRESOLVED " + ", ".join(f"d/d{s}={int(c)}" for s, c in items)


def render(base, resp, vals, limit=14):
    print()
    print("  %-40s %-11s %-18s %-11s %s"
          % ("module", "out(val)", "out (확인)", "in(val)", "in (확인)"))
    for mod in list(base)[:limit]:
        r = resp.get(mod, {})
        o, i = base[mod]
        print("  %-40s %-11s %-18s %-11s %s"
              % (mod[:40], o, interpret(r.get("out"), o, vals),
                 i, interpret(r.get("in"), i, vals)))


if __name__ == "__main__":
    mid = sys.argv[1]
    b, r, v = probe(mid)
    render(b, r, v)
