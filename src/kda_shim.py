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

RESULT (2026-08-10) -- KDA is traceable; the MoE dispatch is not
---------------------------------------------------------------
Everything this module was built for works. On BOTH Kimi-Linear-48B and Kimi-K3 the run now
loads the remote code, builds on meta, falls back to FakeTensor, and reaches the KDA reference
implementation with real shapes (`q = [1, 320, 96, 128]` on K3). Seven obstacles were in the
way; six are bridged:

  1. fla -> triton import                       SOLVED (this module)
  2. `OutputRecorder` removed upstream          SOLVED (stub; declaration-only, no shapes)
  3. `create_causal_mask` arg renamed/dropped   SOLVED (alias + pass only accepted params)
  4. repo Cache lacks 5.x methods               SOLVED (the library's own definitions)
  5. constructor forces flash_attention_2       SOLVED (reset_attn_implementation, post-build)
  6. `.item()` / chunk length                   SOLVED (meta->fake remedy; profile seq_len,
                                                which run.py had never read, now pins T=320
                                                because `naive_chunk_kda` asserts T % 64 == 0)
  7. MoE dispatch reads routed counts on the    NOT SOLVABLE -- see below
     HOST and loops in Python

(7) is not a version drift and not something a shim should paper over. `KimiSparseMoeBlock`
computes its expert assignment and then does

    tokens_per_expert = tokens_per_expert.cpu().numpy()
    for i, num_tokens in enumerate(tokens_per_expert): ...   # modeling_kimi.py:754

-- the number of tokens routed to each expert is pulled to the host and drives Python control
flow. That number does not exist for us: there are no weights, so there is no routing, and a
FakeTensor has no value to read. The model offers no other path (`forward` raises
`NotImplementedError("Training mode is not supported")`).

This is a property of the REPO'S implementation, not of the architecture. Kimi-K2 is the proof:
same vendor, same MoE, 384 experts -- and it traces cleanly with zero new rules, because it runs
through the maintained `deepseek_v3` implementation, whose routing stays on-device
(scatter/gather + grouped matmul, no host transfer). The day `kimi_linear` / `kimi_k3` land in
transformers proper, or the repo's dispatch stops crossing to the host, both models trace with
what is already here -- the KDA part is done.
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


def _extract_func(rel: str, func: str):
    """Pull ONE function's source out of an fla file and run just that.

    `fla/ops/kda/gate.py` holds the torch reference `naive_kda_gate` next to `@triton.jit`
    kernels, so importing the module needs triton even though the reference itself does not.
    Slicing the function out keeps us running fla's own code -- transcribing the formula by hand
    would make it our claim instead of theirs.
    """
    root = _fla_dir()
    if not root:
        return None
    path = os.path.join(root, rel)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        src = f.read()
    start = src.find(f"def {func}(")
    if start < 0:
        return None
    rest = src[start:]
    # end at the next top-level statement (a line that starts in column 0 and is not a decorator
    # continuation of this function)
    end = len(rest)
    for i, line in enumerate(rest.splitlines(keepends=True)[1:], start=1):
        if line[:1] not in (" ", "\t", "\n", "\r") and not line.startswith(")"):
            end = sum(len(x) for x in rest.splitlines(keepends=True)[:i])
            break
    ns = {"torch": torch, "F": F, "nn": nn, "math": __import__("math")}
    try:
        exec(compile(rest[:end], f"<fla:{rel}:{func}>", "exec"), ns)   # noqa: S102 -- fla's own source
    except Exception:                          # noqa: BLE001
        return None
    return ns.get(func)


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
    """Fill the gaps between Kimi's remote code and the installed transformers.

    Both entries are pure API drift: a name that was removed and an argument that was renamed.
    Neither changes what is computed, which is why they are safe to bridge -- but they are the
    reason this model needs a compatibility layer at all, and they are listed in the adaptation
    log so the artifact says so.
    """
    added = []
    from transformers.utils import generic as _generic
    if not hasattr(_generic, "OutputRecorder"):
        _generic.OutputRecorder = _OutputRecorder
        added.append("transformers.utils.generic.OutputRecorder")

    # `create_causal_mask` drifted twice: `input_embeds` was renamed `inputs_embeds`, and
    # `cache_position` was dropped (5.x derives the offset from past_key_values/position_ids).
    # Rename what was renamed, then pass only what the installed signature accepts -- rather than
    # guessing which extras still matter, let the library's own parameter list decide. Everything
    # dropped this way is something this version computes for itself.
    import inspect
    import transformers.masking_utils as _mu
    _orig = getattr(_mu, "create_causal_mask", None)
    if _orig is not None and not getattr(_orig, "_kda_shim_aliased", False):
        _accepted = set(inspect.signature(_orig).parameters)

        def create_causal_mask(*args, **kwargs):
            if "input_embeds" in kwargs and "inputs_embeds" not in kwargs:
                kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
            return _orig(*args, **{k: v for k, v in kwargs.items() if k in _accepted})

        create_causal_mask._kda_shim_aliased = True
        _mu.create_causal_mask = create_causal_mask
        added.append("transformers.masking_utils.create_causal_mask (인자 개명 + 미지원 인자 제거)")

    # The repo ships its own `KimiDynamicCache`, written against the older Cache interface, and
    # 5.x's mask builder now asks it for `get_query_offset`. Take the definition from the library
    # rather than inventing one -- `transformers.cache_utils.Cache.get_query_offset` is exactly
    # `get_seq_length(layer_idx)` (the MTP caches are the only exception, and this is not one).
    from transformers.cache_utils import Cache as _Cache
    if not hasattr(_Cache, "_kda_shim_offset_backfilled"):
        _Cache._kda_shim_offset_backfilled = True

        def _backfill(cache_cls):
            hit = False
            if not hasattr(cache_cls, "get_query_offset") and hasattr(cache_cls, "get_seq_length"):
                cache_cls.get_query_offset = lambda self, layer_idx=0: self.get_seq_length(
                    layer_idx=layer_idx)
                hit = True
            # `get_mask_sizes` changed its first argument from a `cache_position` tensor to a
            # plain `q_length`. The repo's body only ever reads `cache_position.shape[0]`, so the
            # two spellings carry the same number -- accept either and keep the repo's own
            # formula (`kv_length = query_length + past_seen_tokens`, `kv_offset = 0`).
            orig = getattr(cache_cls, "get_mask_sizes", None)
            if orig is not None and not getattr(orig, "_kda_shim_aliased", False):
                def get_mask_sizes(self, q, layer_idx=0, _orig=orig):
                    if isinstance(q, int):
                        return q + self.get_seq_length(layer_idx), 0
                    return _orig(self, q, layer_idx)
                get_mask_sizes._kda_shim_aliased = True
                cache_cls.get_mask_sizes = get_mask_sizes
                hit = True
            return hit

        _kda_backfill_cache.append(_backfill)
        added.append("저장소 캐시 클래스 보정 (get_query_offset / get_mask_sizes 시그니처)")
    return added


# populated by patch_transformers_compat; applied to the repo's cache class once it is imported
_kda_backfill_cache: list = []


def reset_attn_implementation(model, want: str = "eager") -> str | None:
    """Undo the constructor's flash-attention override, after the model is built.

    `KimiLinearModel.__init__` overwrites whatever backend it is handed --
    `config._attn_implementation = "flash_attention_2"` (modeling_kimi.py:912-919) -- and declares
    support under the pre-5.x name `_supports_flash_attn_2`, which this transformers no longer
    reads, so it then refuses to run at all. Setting the backend BEFORE construction cannot win;
    the attention modules read `self.config._attn_implementation` at FORWARD time, so putting it
    back afterwards is both effective and the smallest possible intervention.

    flash-attn is not installed and could not run on meta/fake tensors regardless; `eager` is the
    path that materialises the attention math as ATen ops, which is what we are here to record.
    """
    cfg = getattr(model, "config", None)
    if cfg is None:
        return None
    was = getattr(cfg, "_attn_implementation", None)
    if was == want:
        return None
    for c in (cfg, getattr(cfg, "text_config", None)):
        if c is not None and hasattr(c, "_attn_implementation"):
            try:
                c._attn_implementation = want
            except Exception:                  # noqa: BLE001 -- some configs guard the setter
                return None
    return f"{was} -> {want}"


def backfill_cache_class(model) -> bool:
    """Give the repo's cache class the methods 5.x expects, after the module has been imported.

    The class only exists once the remote modeling file has been loaded, so this runs on the built
    model rather than at patch time.
    """
    done = False
    for mod_name, mod in list(sys.modules.items()):
        if "modeling_kimi" not in mod_name:
            continue
        for attr in dir(mod):
            obj = getattr(mod, attr, None)
            if isinstance(obj, type) and attr.endswith("Cache"):
                for fn in _kda_backfill_cache:
                    done = fn(obj) or done
    return done


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
    if not naive or not getattr(naive, "naive_chunk_kda", None):
        return None
    # gate.py cannot be imported (its Triton kernels sit in the same file), so take just the
    # reference function's source -- see _extract_func.
    naive_gate = _extract_func(os.path.join("ops", "kda", "gate.py"), "naive_kda_gate")

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
