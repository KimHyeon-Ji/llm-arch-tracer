"""PROTOTYPE v2 -- the settings that actually make symbolic shape capture work.
Supersedes probe_symbolic_forward.py. Background and measurements: 03-labeling-roadmap.md 5-6~5-8.

Goal: label the ~98% of trace axes that have no nn.Linear to read an answer off. Instead of
hand-writing a propagation rule per aten op (measured 65.9%, and every rule is a new place to be
wrong), build the parameters as SymInt-sized FakeTensors and let PyTorch's own meta kernels
compute every intermediate shape as a sympy expression.

THE THREE SETTINGS, each established by measurement, not by reading docs:

  specialize_zero_one=False    batch=1 collapses to literal 1 otherwise, losing the B axis
  duck_shape=False             belt-and-braces; we create symbols explicitly via ConstantSource
                               so equal values already do not share a symbol, but this makes it
                               impossible for an auto-created symbol to alias one
  prefer_deferred_runtime_asserts_over_guards=True
                               ** the one that matters **. Without it a guard FREEZES a symbol to
                               a constant and the whole trace degrades. DeepSeek-V2-Lite went
                               17% -> 63% symbolic, 7 frozen symbols -> 0. The frozen ones were
                               T->16 and d_model->2048, i.e. exactly the axes we most need.

Also required:
  * BACKED symbols with hints. Unbacked ones die on GuardOnDataDependentSymNode the moment model
    code branches on a shape (`if q_len > 1`).
  * Swap parameters AFTER construction -- transformers 5.x configs are strict dataclasses and
    reject a SymInt field.
  * EVERY parameter and buffer axis, not just nn.Linear (59% -> 72%). Widths we recognise get a
    named symbol; the rest get a DISTINCT opaque symbol -- never share one because two widths
    happen to be equal.
  * Some models need bf16 (DeepSeek, OLMoE); this is the existing `use_bf16` remedy in
    rules/error_remedies.yaml, reused unchanged.

Measured with all of the above (single forward, FakeTensorMode + ShapeEnv + TorchDispatchMode):

    Llama-3.1-8B     73% of axes symbolic, 0 frozen
    Qwen3-30B-A3B    73%, 1 frozen (s42->2*s6 -- a symbol RELATION, not a constant, harmless)
    OLMoE-1B-7B      73%, 0 frozen
    DeepSeek-V2-Lite 63%, 0 frozen

And the residual is not a gap: on Llama-3.1-8B every non-symbolic axis is one of just two values
-- `1` (x3174, genuine broadcast singletons from unsqueeze/mask) and `4` (x128, the GQA repeat
factor n_h/n_kv). 96% of the "missing" axes are axes whose value really is 1.

NOT wired into the pipeline. Reads models only; writes nothing.
Run:  .venv/Scripts/python.exe develop/probe_symbolic_forward_v2.py <hf-model-id> [...]
"""
import collections
import sys
import os

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from torch.fx.experimental.symbolic_shapes import ShapeEnv, DimDynamic   # noqa: E402
from torch._subclasses.fake_tensor import FakeTensorMode                 # noqa: E402
from torch._dynamo.source import ConstantSource                          # noqa: E402
from torch.utils._python_dispatch import TorchDispatchMode               # noqa: E402
import provenance                                                        # noqa: E402
DT=[torch.float32]
import loader                                                            # noqa: E402


def run(mid, verbose=False, dtype=None, defer=False, concrete_T=False, chunk=None):
    cfg, _ = provenance.snapshot(mid)
    m = loader.load_meta(cfg, trust_remote_code=provenance.needs_remote_code(cfg))

    DT[0] = dtype or torch.float32
    se = ShapeEnv(specialize_zero_one=False, duck_shape=False, prefer_deferred_runtime_asserts_over_guards=defer)
    fm = FakeTensorMode(shape_env=se, allow_non_fake_inputs=True)
    reg = {}                                  # symbol name -> SymInt

    def sym(name, val):
        if name in reg:
            return reg[name]
        s = se.create_symbol(val, ConstantSource(name), dynamic_dim=DimDynamic.DYNAMIC)
        reg[name] = se.create_symintnode(s, hint=val)
        return reg[name]

    rows = []

    class Rec(TorchDispatchMode):
        def __torch_dispatch__(self, f, t, args=(), kw=None):
            o = f(*args, **(kw or {}))
            outs = [x for x in (o if isinstance(o, (tuple, list)) else [o])
                    if isinstance(x, torch.Tensor)]
            if outs:
                rows.append((str(f), tuple(outs[0].shape)))
            return o

    with fm:
        hs = cfg.hidden_size
        nh = cfg.num_attention_heads
        nkv = getattr(cfg, "num_key_value_heads", nh) or nh
        hd = getattr(cfg, "head_dim", None) or hs // nh
        # named symbols for the config dims we know
        NAMED = {hs: sym("d_model", hs), cfg.vocab_size: sym("V", cfg.vocab_size)}
        for nm, v in (("n_h", nh), ("n_kv", nkv), ("d_head", hd),
                      ("d_ff", getattr(cfg, "intermediate_size", None)),
                      ("d_moe", getattr(cfg, "moe_intermediate_size", None)),
                      ("E", getattr(cfg, "num_experts", None) or
                       getattr(cfg, "n_routed_experts", None) or
                       getattr(cfg, "num_local_experts", None))):
            if v:
                NAMED.setdefault(v, sym(nm, v))
        # composite widths that appear as a single parameter axis
        for v, nm in ((nh * hd, "n_h*d_head"), (nkv * hd, "n_kv*d_head")):
            NAMED.setdefault(v, sym(nm, v))

        opaque = {}

        def axis_sym(v):
            """named symbol if we know this width, else a distinct OPAQUE symbol.
            Never reuse a symbol just because two widths are numerically equal."""
            if v in NAMED:
                return NAMED[v]
            if v not in opaque:
                opaque[v] = sym(f"OPAQUE_{v}", v)
            return opaque[v]

        # ---- symbolise EVERY parameter and buffer, every axis
        n_par = 0
        for mod_name, mod in m.named_modules():
            for pname, p in list(mod.named_parameters(recurse=False)):
                if p is None:
                    continue
                shape = [axis_sym(int(d)) for d in p.shape]
                setattr(mod, pname, nn.Parameter(torch.empty(*shape, device="meta", dtype=DT[0]),
                                                 requires_grad=False))
                n_par += 1
            for bname, b in list(mod.named_buffers(recurse=False)):
                if b is None or b.dim() == 0:
                    continue
                shape = [axis_sym(int(d)) for d in b.shape]
                try:
                    mod.register_buffer(bname, torch.empty(*shape, device="meta", dtype=DT[0]),
                                        persistent=False)
                    n_par += 1
                except Exception:
                    pass
            if isinstance(mod, nn.Linear):
                mod.in_features = axis_sym(mod.in_features)
                mod.out_features = axis_sym(mod.out_features)
            for attr in ("head_dim", "num_heads", "num_key_value_heads", "hidden_size",
                         "intermediate_size", "num_experts", "expert_dim"):
                if isinstance(getattr(mod, attr, None), int):
                    setattr(mod, attr, axis_sym(int(getattr(mod, attr))))

        # concrete_T: SSM/scan models compute a chunk PAD size from seq_len, and
        # F.pad rejects symbolic sizes ("SymIntArrayRef expected to contain only
        # concrete integers"). Keeping only the runtime axis concrete still leaves
        # every architecture dim symbolic, which is the part we actually need.
        if chunk:
            # SSM chunked scan computes pad = (chunk - T % chunk) % chunk and hands it to F.pad,
            # which rejects symbolic sizes. Expressing T as chunk*N_chunk makes T % chunk
            # simplify to 0 symbolically, so the pad vanishes and F.pad accepts it -- while T
            # itself stays fully symbolic. Verified: F.pad returns [1, 256*s85, 64].
            T = sym("N_chunk", 4) * chunk
        elif concrete_T:
            T = 16
        else:
            T = sym("T", 16)
        B = sym("B", 1)
        ids = torch.empty(B, T, dtype=torch.long, device="meta")
        with Rec():
            m(input_ids=ids, use_cache=False)

    tot = sum(len(s) for _, s in rows)
    ns = sum(1 for _, s in rows for x in s if not isinstance(x, int))
    spec = {str(k): str(v) for k, v in se.replacements.items()}
    named_specialized = [n for n, s in reg.items()
                         if str(getattr(s, "node", None) and s.node.expr) in spec]
    print("%-40s%s params=%4d ops=%5d axes=%6d symbolic=%3.0f%% specialized=%d"
          % (mid.split("/")[-1], " [defer]" if defer else "        ", n_par, len(rows), tot, 100 * ns / tot, len(spec)))
    if verbose and spec:
        inv = {str(v.node.expr): k for k, v in reg.items() if hasattr(v, "node")}
        print("     고정된 심볼:", ", ".join(
            f"{inv.get(k, k)}->{v}" for k, v in list(spec.items())[:8]))
    return rows


if __name__ == "__main__":
    for mid in sys.argv[1:]:
        try:
            run(mid, verbose=True, defer=True)
        except Exception as e:
            if "BF16" in str(e) or "bfloat16" in str(e):
                try:
                    run(mid, verbose=True, dtype=torch.bfloat16, defer=True)
                    continue
                except Exception as e2:
                    e = e2
            print("%-40s FAILED %s: %s" % (mid.split("/")[-1], type(e).__name__, str(e)[:100]))
