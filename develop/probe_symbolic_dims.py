"""PROTOTYPE -- recover each module dimension's config EXPRESSION by running the model code.

This supersedes develop/probe_config_provenance.py (the perturbation probe): same goal, one
build instead of N, and it returns whole expressions instead of only products.

The goal is what a human does by hand: open modeling_llama.py, read

    self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim)

and write down `in = d_model`, `out = n_h*d_head`. That is reading, not guessing -- which is
exactly what symbolic_shape cannot do, because it only ever sees the NUMBER 4096 and has to
search config for something equal to it. When two fields hold the same number (Llama-3.1-405B:
d_model == n_h*d_head == 16384) the search cannot possibly decide correctly.

Parsing the source with `ast` is the obvious way to "read the code" and it is brittle: the
expression is usually not inline (`self.head_dim = config.head_dim or hidden // heads`), the
attribute name is not the runtime module path, and every repo writes it differently.

So instead: RUN the code and let the arithmetic carry its own name. Replace each dim-bearing
config field with `Dim(value, symbol)` -- an int subclass whose operators propagate an
expression -- then build on meta. `nn.Linear.__init__` stores whatever it was handed, so

    model.layers.0.self_attn.q_proj.out_features.expr == "n_h*d_head"

is read straight off the built module. Verified end-to-end:

    Llama-3.1-8B    q_proj  out=n_h*d_head              in=d_model
                    o_proj  out=d_model                 in=n_h*d_head     (both are 4096)
    DeepSeek-V2-Lite q_proj out=n_h*(d_nope+d_rope)     in=d_model
                    kv_b_proj out=n_h*(((d_nope+d_rope)-d_rope)+d_v)  in=c_kv

Known gaps before this can replace value matching:
  * expressions come out unsimplified (`((d_nope+d_rope)-d_rope)+d_v` is `d_nope+d_v`)
  * only modules exposing in_features/out_features (nn.Linear, nn.Embedding) are covered;
    3-D expert weights built as nn.Parameter(torch.empty(...)) lose the tag inside torch.Size
  * any code path doing int(x) or arithmetic in C strips the tag -> falls back to a bare number
    (degrades to today's behaviour, never to a wrong name)

Run:  .venv/Scripts/python.exe develop/probe_symbolic_dims.py <hf-model-id> [rows]
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import loader          # noqa: E402
import provenance      # noqa: E402
import summarize       # noqa: E402


def _e(x):
    return getattr(x, "expr", None) or str(int(x))


class Dim(int):
    """An int that remembers the config expression it came from. Subclasses int, so every
    isinstance check, torch.empty(...) call and dataclass validator treats it as a plain int --
    it only adds a name that survives arithmetic."""

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
        # keep the rendering close to how the symbols file writes them
        if sym == "*" and b == "1":
            return Dim(v, a)
        return Dim(v, f"({a}{sym}{b})" if sym in "+-" else f"{a}{sym}{b}")

    def __mul__(self, o):        return self._bin(o, lambda x, y: x * y, "*")
    def __rmul__(self, o):       return self._bin(o, lambda x, y: y * x, "*", swap=True)
    def __add__(self, o):        return self._bin(o, lambda x, y: x + y, "+")
    def __radd__(self, o):       return self._bin(o, lambda x, y: y + x, "+", swap=True)
    def __sub__(self, o):        return self._bin(o, lambda x, y: x - y, "-")
    def __rsub__(self, o):       return self._bin(o, lambda x, y: y - x, "-", swap=True)
    def __floordiv__(self, o):   return self._bin(o, lambda x, y: x // y, "/")
    def __rfloordiv__(self, o):  return self._bin(o, lambda x, y: y // x, "/", swap=True)
    def __truediv__(self, o):    return self._bin(o, lambda x, y: x // y, "/")

    # transformers deep-copies the config (to_dict / generation config), and the default int
    # reconstructor calls __new__ with only the value -- so the tag has to survive pickling.
    def __reduce__(self):
        return (Dim, (int(self), self.expr))

    def __deepcopy__(self, memo):
        return Dim(int(self), self.expr)

    def __repr__(self):
        return f"{int(self)}<{self.expr}>"


def tag_config(cfg, symbols):
    """Replace each dim-bearing config field with a named Dim. Returns {field: symbol}."""
    tagged = {}
    for sym, spec in symbols.items():
        if not spec.get("dim"):
            continue
        for alias in (spec.get("aliases") or []):
            v = getattr(cfg, alias, None)
            if isinstance(v, int) and not isinstance(v, bool) and not isinstance(v, Dim):
                setattr(cfg, alias, Dim(v, sym))
                tagged[alias] = sym
                break
    return tagged


def read_off(model):
    out = {}
    for name, mod in model.named_modules():
        i, o = getattr(mod, "in_features", None), getattr(mod, "out_features", None)
        if i is None and o is None:
            i, o = getattr(mod, "embedding_dim", None), getattr(mod, "num_embeddings", None)
        if isinstance(i, int) or isinstance(o, int):
            out[name] = (o, i)
    return out


if __name__ == "__main__":
    mid = sys.argv[1]
    cfg, _ = provenance.snapshot(mid)
    tagged = tag_config(cfg, summarize.load_symbols())
    print(f"# {mid}")
    print(f"  tagged config fields: {tagged}")
    model = loader.load_meta(cfg, trust_remote_code=provenance.needs_remote_code(cfg))
    d = read_off(model)
    print()
    print("  %-46s %-26s %s" % ("module", "out", "in"))
    for name in list(d)[: int(sys.argv[2]) if len(sys.argv) > 2 else 16]:
        o, i = d[name]
        print("  %-46s %-26s %s" % (name[:46], _e(o) if o is not None else "-",
                                    _e(i) if i is not None else "-"))
