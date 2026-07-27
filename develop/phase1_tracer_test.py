"""Phase 1 -- tracer/scope unit test WITHOUT any HF model (04-verification-plan.md).

Two parts:
  (1) general: a small nn.Module (Linears + attention-like ops + norm) traced on meta.
      Verify op capture, depends_on has no cycle, weight_shape attribution survives views.
  (2) MLA-isolated: a block that compresses KV to a low-rank latent then re-expands
      (down/up projection) + a decoupled-RoPE-style split. Traced in isolation so that
      when Phase 6 (DeepSeek-V2-Lite) hits MLA+MoE together we already trust MLA alone.

Run: .venv\\Scripts\\python.exe develop\\phase1_tracer_test.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F

from scope import ScopeLabeler
from tracer import OpGraphTracer


# ---------------------------------------------------------------- helpers
def has_cycle(rows):
    graph = {r["op_id"]: r.get("depends_on", []) for r in rows}
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {i: WHITE for i in graph}

    def visit(n, stack):
        if color[n] == BLACK:
            return None
        if color[n] == GRAY:
            return stack + [n]
        color[n] = GRAY
        for dep in graph.get(n, []):
            c = visit(dep, stack + [n])
            if c:
                return c
        color[n] = BLACK
        return None

    for i in graph:
        c = visit(i, [])
        if c:
            return c
    return None


def trace_module(model, inputs):
    scope = ScopeLabeler(model)
    tracer = OpGraphTracer(model, scope)
    with torch.no_grad(), tracer:
        model(**inputs)
    scope.remove()
    return tracer.rows


def check(cond, label):
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}")
    return cond


# ---------------------------------------------------------------- part 1
class TinyAttn(nn.Module):
    def __init__(self, d, nh):
        super().__init__()
        self.nh, self.dh = nh, d // nh
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        B, S, D = x.shape
        h = self.norm(x)
        q = self.q(h).view(B, S, self.nh, self.dh).transpose(1, 2)
        k = self.k(h).view(B, S, self.nh, self.dh).transpose(1, 2)
        v = self.v(h).view(B, S, self.nh, self.dh).transpose(1, 2)
        attn = torch.softmax(q @ k.transpose(-1, -2) / (self.dh ** 0.5), dim=-1)
        o = (attn @ v).transpose(1, 2).reshape(B, S, D)
        return x + self.o(o)


class TinyMLP(nn.Module):
    def __init__(self, d, dff):
        super().__init__()
        self.up = nn.Linear(d, dff, bias=False)
        self.gate = nn.Linear(d, dff, bias=False)
        self.down = nn.Linear(dff, d, bias=False)

    def forward(self, x):
        return x + self.down(F.silu(self.gate(x)) * self.up(x))


class TinyLayer(nn.Module):
    def __init__(self, d, nh, dff):
        super().__init__()
        self.self_attn = TinyAttn(d, nh)
        self.mlp = TinyMLP(d, dff)

    def forward(self, x):
        return self.mlp(self.self_attn(x))


class TinyBackbone(nn.Module):
    def __init__(self, d, nh, dff, nlayers):
        super().__init__()
        self.layers = nn.ModuleList([TinyLayer(d, nh, dff) for _ in range(nlayers)])

    def forward(self, x):
        for lyr in self.layers:
            x = lyr(x)
        return x


class TinyModel(nn.Module):
    # real HF AutoModelForCausalLM always nests the decoder stack under a backbone
    # (model.layers.N... for Llama-family), so scope paths carry a leading `model.`.
    def __init__(self, d=32, nh=4, dff=64, nlayers=2):
        super().__init__()
        self.model = TinyBackbone(d, nh, dff, nlayers)

    def forward(self, x):
        return self.model(x)


def part1_general():
    print("Part 1 -- general tracer/scope on a handcrafted module")
    with torch.device("meta"):
        model = TinyModel()
        model.eval()
        x = torch.randn(1, 6, 32)
    rows = trace_module(model, {"x": x})

    ok = True
    ok &= check(len(rows) > 0, f"captured {len(rows)} ops")
    ok &= check(has_cycle(rows) is None, "depends_on graph is acyclic")

    # weight_shape attribution: every linear (addmm/mm) op must carry a weight_shape,
    # AND the param name must survive the internal weight.t() view (TRIVIAL propagation)
    lin_rows = [r for r in rows if r["raw_op"] in
                ("aten.addmm.default", "aten.mm.default", "aten.bmm.default")
                and any("weight" in p for p in r["params"])]
    ok &= check(len(lin_rows) > 0, f"{len(lin_rows)} matmul-family ops attributed to a weight param")
    ok &= check(all(r["weight_shape"] is not None for r in lin_rows),
                "every weight-bearing matmul carries a weight_shape (view-through attribution)")

    # layer labeling: scope must assign layer_idx 0 and 1, block self_attn / mlp
    layer_idxs = {r["layer_idx"] for r in rows if r["layer_idx"] is not None}
    ok &= check(layer_idxs == {0, 1}, f"scope labeled layers {sorted(layer_idxs)} (expected 0,1)")
    blocks = {r["block"] for r in rows if r["block"]}
    ok &= check({"self_attn", "mlp"} <= blocks, f"scope saw blocks {sorted(blocks)}")

    # softmax must be captured (functional op inside forward -- the whole point of dispatch)
    ok &= check(any("_softmax" in r["raw_op"] for r in rows),
                "softmax captured (functional op, not a module boundary)")
    return ok


# ---------------------------------------------------------------- part 2
class MLAIsolated(nn.Module):
    """Mimics MLA: KV compressed to a low-rank latent (kv_lora_rank) then re-expanded to
    per-head k/v, plus a decoupled RoPE dim carried separately. Not DeepSeek's exact code
    -- just the structural pattern (down-proj -> latent -> up-proj + split) in isolation."""

    def __init__(self, d=64, nh=4, dh=16, kv_lora=16, rope_dim=8):
        super().__init__()
        self.nh, self.dh, self.rope_dim = nh, dh, rope_dim
        self.q_proj = nn.Linear(d, nh * (dh + rope_dim), bias=False)
        self.kv_down = nn.Linear(d, kv_lora, bias=False)          # compress
        self.kv_up = nn.Linear(kv_lora, nh * dh * 2, bias=False)  # re-expand to k and v
        self.k_rope = nn.Linear(d, rope_dim, bias=False)          # decoupled shared rope key
        self.o_proj = nn.Linear(nh * dh, d, bias=False)

    def forward(self, x):
        B, S, D = x.shape
        q = self.q_proj(x).view(B, S, self.nh, self.dh + self.rope_dim)
        q_nope, q_rope = q.split([self.dh, self.rope_dim], dim=-1)
        latent = self.kv_down(x)                                  # compressed KV
        kv = self.kv_up(latent).view(B, S, self.nh, self.dh * 2)
        k_nope, v = kv.split([self.dh, self.dh], dim=-1)
        k_rope = self.k_rope(x).view(B, S, 1, self.rope_dim).expand(B, S, self.nh, self.rope_dim)
        k = torch.cat([k_nope, k_rope], dim=-1)
        q_full = torch.cat([q_nope, q_rope], dim=-1)
        attn = torch.softmax(
            (q_full.transpose(1, 2) @ k.transpose(1, 2).transpose(-1, -2))
            / ((self.dh + self.rope_dim) ** 0.5), dim=-1)
        o = (attn @ v.transpose(1, 2)).transpose(1, 2).reshape(B, S, self.nh * self.dh)
        return self.o_proj(o)


class _MLABackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([MLAIsolated()])

    def forward(self, x):
        return self.layers[0](x)


class MLAWrapper(nn.Module):
    # nested under `model.` like a real HF CausalLM -> scope path model.layers.0...
    def __init__(self):
        super().__init__()
        self.model = _MLABackbone()

    def forward(self, x):
        return self.model(x)


def part2_mla():
    print("Part 2 -- MLA isolated (KV compress/expand + decoupled RoPE dim)")
    with torch.device("meta"):
        model = MLAWrapper()
        model.eval()
        x = torch.randn(1, 6, 64)
    rows = trace_module(model, {"x": x})

    ok = True
    ok &= check(len(rows) > 0, f"captured {len(rows)} ops")
    ok &= check(has_cycle(rows) is None, "depends_on graph is acyclic")

    # the down-proj -> up-proj chain must exist as a dependency path (compression captured).
    # Pick the data-carrying matmul, NOT the weight-transpose op that also carries the param.
    _MM = ("aten.mm.default", "aten.addmm.default", "aten.bmm.default")
    down = next((r for r in rows if r["raw_op"] in _MM and any("kv_down" in p for p in r["params"])), None)
    up = next((r for r in rows if r["raw_op"] in _MM and any("kv_up" in p for p in r["params"])), None)
    ok &= check(down is not None and up is not None, "kv_down and kv_up both captured as matmul ops")
    if down and up:
        # up-proj must (transitively) depend on down-proj's output
        graph = {r["op_id"]: r.get("depends_on", []) for r in rows}
        seen, frontier = set(), [up["op_id"]]
        while frontier:
            c = frontier.pop()
            if c in seen:
                continue
            seen.add(c)
            frontier.extend(graph.get(c, []))
        ok &= check(down["op_id"] in seen, "kv_up transitively depends on kv_down (compression path intact)")
        # the latent (down output) must be lower-rank than the re-expanded kv
        down_out = down["output_shape"][0]
        up_out = up["output_shape"][0]
        ok &= check(down_out[-1] < up_out[-1],
                    f"latent dim {down_out[-1]} < expanded kv dim {up_out[-1]} (compression is real)")

    # cat ops for the decoupled-rope concat must be captured
    ok &= check(any(r["raw_op"] == "aten.cat.default" for r in rows),
                "decoupled-rope concat (aten.cat) captured")
    return ok


if __name__ == "__main__":
    r1 = part1_general()
    print()
    r2 = part2_mla()
    print()
    print(f"Phase 1 overall: {'PASS' if (r1 and r2) else 'FAIL'}")
    sys.exit(0 if (r1 and r2) else 1)
