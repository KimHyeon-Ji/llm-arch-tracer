"""Make KDA (Kimi Delta Attention) traceable by routing it through torch, not Triton.

WHY THIS EXISTS
---------------
Moonshot's Kimi-Linear and Kimi-K3 implement their linear-attention layers with kernels from
`fla` (flash-linear-attention), and their modeling file imports those kernels unconditionally at
module import time:

    from fla.modules import FusedRMSNormGated, ShortConvolution
    from fla.ops.kda import chunk_kda, fused_recurrent_kda
    from fla.ops.kda.gate import fused_kda_gate
    ...

Every one of those pulls in `triton`, so on any machine without it the model cannot even be
imported -- and `triton` has no wheel for this platform.

Installing triton would not help. **A Triton kernel is invisible to `TorchDispatchMode`**: it is
one opaque launch, not a sequence of ATen ops, so the KDA layers would come out of the trace as a
hole no matter what hardware we had. Reading inside them requires a torch implementation. That is
the same reason Mamba2 (Nemotron-3, Zamba2), Gated DeltaNet (Qwen3-Next, Qwen3.5/3.6) and xLSTM
are traced through their torch paths -- the difference is only that those models ship the fallback
themselves, while Kimi does not.

WHAT IS SUBSTITUTED
-------------------
Three of the five come from `fla` ITSELF -- the library ships torch references next to its
kernels, which is what a reference implementation is for:

    chunk_kda            -> fla.ops.kda.naive.naive_chunk_kda
    fused_recurrent_kda  -> fla.ops.kda.naive.naive_recurrent_kda
    fused_kda_gate       -> fla.ops.kda.gate.naive_kda_gate

Those files import cleanly without triton (they are pure torch + einops); they are loaded here by
path so that importing them cannot drag in `fla/__init__.py`.

Two are small nn.Modules whose torch equivalent is unambiguous and is written out here:

    ShortConvolution     -> causal depthwise conv1d, the same form Mamba's torch path uses
    FusedRMSNormGated    -> RMSNorm(x) * act(gate)

DISCLOSURE
----------
A trace produced with this shim describes the REFERENCE implementation of KDA, not the Triton
kernel the model runs on GPU. They compute the same function and should agree on every shape --
that is what makes it a reference -- but this is a substitution and must be visible in the
artifact. `install()` records itself in the adaptation log, and the model's `review_findings.json`
must carry it as an open note so `model_summary.md` shows it next to the tables.

STATUS (2026-08-10)
-------------------
Verified this far on Kimi-Linear-48B-A3B: the fla imports resolve, the config loads, and the
model BUILDS on meta. The forward then hits a third, unrelated version drift in the same remote
file -- it calls  while transformers 5.x spells that
argument . That is a rename, not a semantic change, but it is one more patch on
top of two, and each one widens the gap between what we trace and what the repo ships. Finish it
deliberately: alias the kwarg, then trace, then check the result against the config's own
declaration (head_dim 128, num_heads 32, short_conv_kernel_size 4, 20 kda_layers / 7 full).
"""
import importlib.util
import os
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

_FLA_ROOT = None


def _fla_dir() -> str | None:
    """Directory of the installed `fla` package, without importing it (that needs triton)."""
    global _FLA_ROOT
    if _FLA_ROOT is None:
        spec = importlib.util.find_spec("fla")
        _FLA_ROOT = os.path.dirname(spec.origin) if spec and spec.origin else ""
    return _FLA_ROOT or None


def _load_by_path(name: str, rel: str):
    """Load one fla source file directly, so `fla/__init__.py` (and triton) never runs."""
    root = _fla_dir()
    if not root:
        return None
    path = os.path.join(root, rel)
    if not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:                      # noqa: BLE001 -- a missing reference is not fatal here
        return None
    return mod


class ShortConvolution(nn.Module):
    """Causal depthwise conv1d over the sequence -- `fla.modules.ShortConvolution` in torch.

    Same shape contract as the original: `(x=[B, T, C], cache, output_final_state, cu_seqlens)`
    returns `(y=[B, T, C], final_state | None)`. The activation is applied after the convolution.
    """

    def __init__(self, hidden_size, kernel_size, activation=None, bias=False, **_kw):
        super().__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.activation = activation
        self.conv = nn.Conv1d(hidden_size, hidden_size, kernel_size,
                              groups=hidden_size, padding=kernel_size - 1, bias=bias)

    def forward(self, x, cache=None, output_final_state=False, cu_seqlens=None, **_kw):
        y = self.conv(x.transpose(1, 2))[..., : x.shape[1]].transpose(1, 2)
        if self.activation == "silu":
            y = F.silu(y)
        elif self.activation == "swish":
            y = F.silu(y)
        elif self.activation == "gelu":
            y = F.gelu(y)
        state = None
        if output_final_state:
            # the kernel keeps the last (kernel_size - 1) inputs to continue the convolution
            state = x[:, -(self.kernel_size - 1):, :].transpose(1, 2)
        return y, state


class FusedRMSNormGated(nn.Module):
    """`RMSNorm(x) * act(gate)` -- `fla.modules.FusedRMSNormGated` in torch."""

    def __init__(self, hidden_size, eps=1e-5, activation="sigmoid", **_kw):
        super().__init__()
        self.eps = eps
        self.activation = activation
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x, gate=None, **_kw):
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        h = h.to(x.dtype) * self.weight
        if gate is None:
            return h
        g = torch.sigmoid(gate) if self.activation == "sigmoid" else F.silu(gate)
        return h * g.to(h.dtype)


def _wrap_kda(naive_fn):
    """Adapt the reference signature to the kernel's call site.

    The model calls `chunk_kda(q=..., k=..., v=..., g=..., beta=..., initial_state=...,
    output_final_state=True, use_qk_l2norm_in_kernel=True, cu_seqlens=...)`. The reference takes
    no `use_qk_l2norm_in_kernel` (the kernel folds that normalisation in, so it has to be applied
    here) and no `cu_seqlens` (that is the varlen-packing path; our traces are a single unpacked
    sequence, so it is None and nothing is dropped).
    """
    def call(q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False,
             use_qk_l2norm_in_kernel=False, cu_seqlens=None, **_kw):
        if use_qk_l2norm_in_kernel:
            q = F.normalize(q, dim=-1, p=2)
            k = F.normalize(k, dim=-1, p=2)
        return naive_fn(q=q, k=k, v=v, g=g, beta=beta, scale=scale,
                        initial_state=initial_state, output_final_state=output_final_state)
    return call


def _wrap_gate(naive_gate):
    """`fused_kda_gate(g, A_log, head_dim, g_bias=...)` -> `naive_kda_gate(g, A_log, dt_bias=...)`.

    The kernel takes the head width positionally and folds the reshape in; the reference wants
    `g` already laid out as `[..., H, K]`.
    """
    def call(g, A_log, head_dim=None, g_bias=None, **_kw):
        if head_dim:
            g = g.view(*g.shape[:-1], g.shape[-1] // head_dim, head_dim)
        return naive_gate(g, A_log, dt_bias=g_bias)
    return call


def _mod(name: str, **attrs) -> types.ModuleType:
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


class _OutputRecorder:
    """Stub for `transformers.utils.generic.OutputRecorder`, removed upstream.

    Kimi's modeling file imports it and uses it in exactly one place -- a `_can_record_outputs`
    declaration naming which module's output to capture for `router_logits`. It is metadata for
    the output-recording plumbing and takes no part in the computation, so a stub cannot change a
    single shape. Without it the file does not import at all on transformers 5.x.
    """

    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs


def patch_transformers_compat() -> list:
    """Fill in names Kimi's remote code imports that this transformers no longer exports."""
    added = []
    from transformers.utils import generic as _generic
    if not hasattr(_generic, "OutputRecorder"):
        _generic.OutputRecorder = _OutputRecorder
        added.append("transformers.utils.generic.OutputRecorder")
    return added


def available() -> bool:
    """True when fla's torch references can be loaded (i.e. the shim can be installed)."""
    return bool(_load_by_path("_kda_naive_probe", os.path.join("ops", "kda", "naive.py")))


def install() -> dict | None:
    """Seed `sys.modules` with the six names Kimi's modeling file imports.

    Pre-seeding rather than stubbing `triton`: the model's `from fla... import X` then finds our
    module and the real `fla` package is never touched. Returns an adaptation-log entry, or None
    if the references could not be loaded (in which case the caller must not claim a trace).
    """
    naive = _load_by_path("_fla_kda_naive", os.path.join("ops", "kda", "naive.py"))
    gate = _load_by_path("_fla_kda_gate", os.path.join("ops", "kda", "gate.py"))
    if not naive or not getattr(naive, "naive_chunk_kda", None):
        return None
    naive_gate = getattr(gate, "naive_kda_gate", None) if gate else None

    def _identity_cache(fn):
        return fn

    def _lens_from_mask(mask):
        return mask.sum(-1).to(torch.int32) if mask is not None else None

    def _cu_seqlens_from_mask(mask):
        lens = _lens_from_mask(mask)
        if lens is None:
            return None
        return torch.cat([lens.new_zeros(1), lens.cumsum(0)]).to(torch.int32)

    mods = {
        "fla": _mod("fla"),
        "fla.modules": _mod("fla.modules", ShortConvolution=ShortConvolution,
                            FusedRMSNormGated=FusedRMSNormGated),
        "fla.ops": _mod("fla.ops"),
        "fla.ops.kda": _mod("fla.ops.kda",
                            chunk_kda=_wrap_kda(naive.naive_chunk_kda),
                            fused_recurrent_kda=_wrap_kda(naive.naive_recurrent_kda)),
        "fla.ops.kda.gate": _mod("fla.ops.kda.gate",
                                 fused_kda_gate=_wrap_gate(naive_gate) if naive_gate else None),
        "fla.ops.utils": _mod("fla.ops.utils"),
        "fla.ops.utils.index": _mod("fla.ops.utils.index",
                                    prepare_lens_from_mask=_lens_from_mask,
                                    prepare_cu_seqlens_from_mask=_cu_seqlens_from_mask),
        "fla.utils": _mod("fla.utils", tensor_cache=_identity_cache),
    }
    for name, mod in mods.items():
        sys.modules.setdefault(name, mod)
    # make the submodules reachable as attributes too (`import fla.ops.kda` style access)
    sys.modules["fla"].modules = mods["fla.modules"]
    sys.modules["fla"].ops = mods["fla.ops"]
    sys.modules["fla"].utils = mods["fla.utils"]
    mods["fla.ops"].kda = mods["fla.ops.kda"]
    mods["fla.ops"].utils = mods["fla.ops.utils"]
    mods["fla.ops.kda"].gate = mods["fla.ops.kda.gate"]
    mods["fla.ops.utils"].index = mods["fla.ops.utils.index"]

    return {
        "tier": 1,
        "remedy": "kda_torch_reference",
        "detail": ("KDA traced through fla's OWN torch reference (naive_chunk_kda / "
                   "naive_recurrent_kda / naive_kda_gate) plus torch equivalents of "
                   "ShortConvolution and FusedRMSNormGated. The Triton kernel the model runs on "
                   "GPU is opaque to TorchDispatchMode, so no trace of it is possible; shapes "
                   "here describe the reference implementation."),
    }
