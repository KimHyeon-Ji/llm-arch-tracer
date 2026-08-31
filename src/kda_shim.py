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

RESULT -- both KDA and the MoE dispatch are traceable (updated 2026-08-31; this section
originally said the MoE dispatch was NOT solvable as of 2026-08-10 -- that was superseded by
`patch_moe_infer` below and this text had gone stale)
---------------------------------------------------------------
Everything this module was built for works. On BOTH Kimi-Linear-48B and Kimi-K3 the run now
loads the remote code, builds on meta, falls back to FakeTensor, and reaches the KDA reference
implementation with real shapes (`q = [1, 320, 96, 128]` on K3). Seven obstacles were in the
way; all seven are bridged:

  1. fla -> triton import                       SOLVED (this module)
  2. `OutputRecorder` removed upstream          SOLVED (stub; declaration-only, no shapes)
  3. `create_causal_mask` arg renamed/dropped   SOLVED (alias + pass only accepted params)
  4. repo Cache lacks 5.x methods               SOLVED (the library's own definitions)
  5. constructor forces flash_attention_2       SOLVED (reset_attn_implementation, post-build)
  6. `.item()` / chunk length                   SOLVED (meta->fake remedy; profile seq_len,
                                                which run.py had never read, now pins T=320
                                                because `naive_chunk_kda` asserts T % 64 == 0)
  7. MoE dispatch reads routed counts on the    SOLVED (`patch_moe_infer` below --
     HOST and loops in Python                   disclosed even-split substitution)

(7) is not a version drift and not something a shim should silently paper over. `KimiSparseMoeBlock`
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
(scatter/gather + grouped matmul, no host transfer). `patch_moe_infer` (below) works around the
host read for `kimi_k3`/`kimi_linear` specifically -- it does NOT read the routing values at all;
it substitutes a disclosed even split (see that function's docstring for exactly what is and is
not faithful about the result). The day `kimi_linear`/`kimi_k3` land in transformers proper, or
the repo's dispatch stops crossing to the host, the shim becomes unnecessary but nothing here
needs to change to keep working.
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
        xt = x.transpose(1, 2)                          # [B, C, T]
        if cache is not None:
            # `cache` is the previous (kernel_size - 1) real inputs, stored in this same [B, C,
            # K-1] layout by the `state` computation below. The padding-then-slice trick just
            # below is only a causal conv when there IS no real history -- decode's whole point is
            # that there is one, so prepend it and run the conv with no padding instead. Until
            # 2026-08-20 this branch did not exist: `cache` was accepted and silently ignored, so
            # every decode step convolved against implicit zero history instead of the cached
            # tokens (found via external/Codex review of commits since 3c955a3a).
            xt = torch.cat([cache, xt], dim=-1)          # [B, C, K-1+T]
            y = F.conv1d(xt, self.conv.weight, self.conv.bias,
                         groups=self.conv.groups)[..., -x.shape[1]:].transpose(1, 2)
        else:
            y = self.conv(xt)[..., : x.shape[1]].transpose(1, 2)
        if self.activation == "silu":
            y = F.silu(y)
        elif self.activation == "swish":
            y = F.silu(y)
        elif self.activation == "gelu":
            y = F.gelu(y)
        state = None
        if output_final_state:
            # the kernel keeps the last (kernel_size - 1) inputs to continue the convolution --
            # `xt` already includes the incoming cache (if any), so this keeps sliding the window
            # forward across calls instead of resetting it to just the newest tokens.
            state = xt[..., -(self.kernel_size - 1):]
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


def _wrap_kda(naive_fn, naive_gate=None, naive_lowerbound_gate=None):
    """Adapt the reference signature to the kernel's call site.

    The model calls `chunk_kda(q=..., k=..., v=..., g=..., beta=..., A_log=..., dt_bias=...,
    use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True,
    safe_gate=..., lower_bound=..., transpose_state_layout=True, ...)`.
    **The kernel folds preprocessing steps in that the reference does not**, so they have to be
    applied here or they simply do not happen:

        use_qk_l2norm_in_kernel   -> L2-normalise q and k
        use_gate_in_kernel        -> the KDA gate (plain or safe/lower-bound, see below)
        use_beta_sigmoid_in_kernel-> beta = beta.sigmoid()

    The gate one matters most: it is what makes this **Kimi Delta Attention** rather than a plain
    gated DeltaNet -- a per-channel forget gate. Dropping `A_log`/`dt_bias` into `**_kw` (which is
    what this wrapper did until 2026-08-17) left that computation out of the trace entirely, and
    the two parameters then contributed to no op at all -- which is exactly how the gate's
    parameter-coverage check (C10) found it.

    Two gate formulas exist in `fla.ops.kda.gate` (`naive_kda_gate` / `naive_kda_lowerbound_gate`,
    and the kernel picks between them via `safe_gate`/`lower_bound` -- see
    `fla/ops/kda/chunk.py`'s `chunk_kda` docstring: `safe_gate=True` "changes the gate activation
    from [...] to lower_bound * sigmoid(exp(A_log) * (g + dt_bias))". Kimi-K3's own config sets
    `linear_attn_config.gate_lower_bound = -5.0`, so `self.gate_lower_bound is not None` makes
    every real call pass `safe_gate=True, lower_bound=-5.0` -- this is not a hypothetical branch,
    it is the one this model actually takes. Until 2026-08-20 those three kwargs fell into
    `**_kw` and were silently dropped, so the trace always used the plain gate formula even though
    the model requests the lower-bound one (same K/V widths either way, so no shape check could
    catch it -- found via external/Codex review of commits since 3c955a3a).

    `transpose_state_layout` is a Triton-kernel memory-layout choice for `recurrent_state`
    (`fla/ops/kda/chunk.py` renames it to `state_v_first` and warns); the naive torch reference has
    no equivalent layout knob, so it is accepted and dropped -- nothing about the computed values
    depends on it.

    `cu_seqlens` is still dropped: that is the varlen-packing path, and our traces are a single
    unpacked sequence, so it is None and nothing is lost.
    """
    def call(q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False,
             use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
             use_beta_sigmoid_in_kernel=False, A_log=None, dt_bias=None,
             safe_gate=False, lower_bound=None, cu_seqlens=None, **_kw):
        if use_qk_l2norm_in_kernel:
            q = F.normalize(q, dim=-1, p=2)
            k = F.normalize(k, dim=-1, p=2)
        if use_gate_in_kernel and A_log is not None:
            if safe_gate and lower_bound is not None and naive_lowerbound_gate is not None:
                g = naive_lowerbound_gate(g, A_log, dt_bias=dt_bias, lower_bound=lower_bound)
            elif safe_gate and lower_bound is not None:
                # 참조 게이트를 못 얻었을 때도 식은 소스에 적혀 있다 (fla/ops/kda/chunk.py)
                gb = g if dt_bias is None else g + dt_bias.view(g.shape[-2:])
                g = lower_bound * torch.sigmoid(A_log.float().exp().unsqueeze(-1) * gb.float())
            elif naive_gate is not None:
                g = naive_gate(g, A_log, dt_bias=dt_bias)
            else:                      # 참조 게이트를 못 얻었을 때도 식은 소스에 적혀 있다
                gb = g if dt_bias is None else g + dt_bias.view(g.shape[-2:])
                g = -A_log.float().exp().unsqueeze(-1) * F.softplus(gb.float())
        if use_beta_sigmoid_in_kernel:
            beta = beta.sigmoid()
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
    # reference functions' source -- see _extract_func. Two gate formulas exist: the plain one and
    # the safe/lower-bound one the kernel switches to when `safe_gate=True` (see _wrap_kda).
    naive_gate = _extract_func(os.path.join("ops", "kda", "gate.py"), "naive_kda_gate")
    naive_lowerbound_gate = _extract_func(
        os.path.join("ops", "kda", "gate.py"), "naive_kda_lowerbound_gate")

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
                            chunk_kda=_wrap_kda(
                                naive.naive_chunk_kda, naive_gate, naive_lowerbound_gate),
                            fused_recurrent_kda=_wrap_kda(
                                naive.naive_recurrent_kda, naive_gate, naive_lowerbound_gate)),
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
                   "naive_recurrent_kda / naive_kda_gate / naive_kda_lowerbound_gate) plus torch "
                   "equivalents of ShortConvolution and FusedRMSNormGated. The Triton kernel the "
                   "model runs on GPU is opaque to TorchDispatchMode, so no trace of it is "
                   "possible; shapes here describe the reference implementation. "
                   "NOTE (2026-08-25, external/Codex review of commits since 3c955a3a): "
                   "`_wrap_kda` branches on `safe_gate`/`lower_bound` (Kimi-K3's own config sets "
                   "`gate_lower_bound=-5.0`) and `ShortConvolution.forward` prepends a supplied "
                   "`cache` instead of ignoring it. Verified with an actual re-trace (not the "
                   "label-only `develop/regen_summaries.py`, which never re-runs the model and so "
                   "cannot exercise either branch): prefill's `chunk_kda` call correctly gets "
                   "`safe_gate=True` and uses the lower-bound gate formula, and decode's "
                   "`ShortConvolution` correctly concatenates the cached tokens before "
                   "convolving. Decode's OWN `fused_recurrent_kda` call site in the model's source "
                   "never passes `safe_gate` (only `lower_bound`), so it always takes the plain "
                   "gate formula there -- that is the model's own code, not a shim gap. Neither "
                   "fix changes any tensor SHAPE. The currently-promoted `models/` output still "
                   "reflects the pre-fix trace, since promoting the new one requires a full "
                   "re-onboarding pass (op_ids shift once decode gains the cache-prepend op, so "
                   "every positional rule/override needs re-verifying against the new raw trace) "
                   "-- left as separate future work; see review/06-open-renames.md A61."),
    }


_MOE_PATCH_CACHE: dict = {}   # cls -> adaptation-log entry, so a second (decode-phase) call
                              # can still return it instead of None (see the guard below).


def patch_moe_infer(model) -> dict | None:
    """Make `KimiSparseMoeBlock.moe_infer` traceable — it drives its loop off ROUTING VALUES.

    The repo's dispatch is:

        tokens_per_expert = cnts.sum(dim=0).cpu().numpy()   # <- values, not shapes
        for i, num_tokens in enumerate(tokens_per_expert):
            expert_out = self.experts[i](sorted_tokens[start:start + num_tokens])

    `.numpy()` raises on a fake tensor, and even if it did not, **there are no routing values to
    read** — nothing was computed. This is not a Triton problem like KDA; it is data-dependent
    control flow, the same class the tracer already meets in other MoE models. The others get away
    with it because transformers dispatches every expert in one batched matmul; this repo's file
    keeps the older per-expert Python loop.

    WHAT IS SUBSTITUTED, AND WHAT THAT COSTS
    ----------------------------------------
    Two departures, both recorded in `provenance.adaptation_log` so a reader is never misled.

    1. **Even split instead of the real counts.** The op STRUCTURE stays faithful -- every expert
       traced still runs its own gate/up/down on a `[n, d_model]` slice, which is what the table
       describes. The per-expert token COUNT does not; that is runtime data which changes with
       every input, so no single trace could report it anyway.

    2. **Only the first `cap` experts are traced** (default 4). Kimi-K3 has **896 experts per
       layer across 93 layers**; running them all produced a 483 MB prefill trace whose labelling
       pass had not finished after half an hour. The experts are structurally identical -- the
       same three projections on the same widths, differing only in weight values, which a
       weightless trace cannot see anyway. `E` in `structure.yaml` still comes from the config and
       still reads 896; what the op table shows is `cap` instances of a repeated structure.
       Set `KDA_SHIM_EXPERT_CAP=0` to trace every expert.

    Everything around the loop -- the scatter, the argsort, the gather back, the weighting -- is
    the model's own code and is traced unchanged.
    """
    import torch as _t

    _CAP = int(os.environ.get("KDA_SHIM_EXPERT_CAP", "4"))
    blocks = [m for m in model.modules() if type(m).__name__ == "KimiSparseMoeBlock"]
    if not blocks:
        return None
    cls = type(blocks[0])
    if getattr(cls, "_moe_infer_traceable", False):
        # Already patched (e.g. the decode-phase reload) -- return the SAME entry again
        # rather than None, so a caller that reloads per phase still sees it every time.
        return _MOE_PATCH_CACHE.get(cls)

    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]

        n, n_exp = sorted_tokens.shape[0], len(self.experts)
        traced = n_exp if _CAP <= 0 else min(_CAP, n_exp)
        per = max(1, n // traced)
        outputs, start = [], 0
        for i in range(traced):
            end = n if i == traced - 1 else min(n, start + per)
            if end <= start:
                break                     # 토큰보다 전문가가 많으면 남은 전문가는 안 돈다
            outputs.append(self.experts[i + self.ep_rank * self.experts_per_rank](
                sorted_tokens[start:end]))
            start = end
        outs = _t.cat(outputs, dim=0) if outputs else sorted_tokens.new_empty(0)

        new_x = _t.empty_like(outs)
        new_x[idxs] = outs
        return (new_x.view(*topk_ids.shape, -1)
                .type(topk_weight.dtype)
                .mul_(topk_weight.unsqueeze(dim=-1))
                .sum(dim=1)
                .type(new_x.dtype))

    cls.moe_infer = _t.no_grad()(moe_infer)
    cls._moe_infer_traceable = True
    n_exp = len(blocks[0].experts)
    entry = {
        "tier": 1,
        "remedy": "moe_infer_even_split",
        # Structured fields (not just prose) so validate.c10_coverage can compute exactly which
        # params were deliberately left untraced, instead of a human re-reading the detail text.
        "expert_cap": None if _CAP <= 0 else min(_CAP, n_exp),
        "experts_per_layer": n_exp,
        "detail": ("KimiSparseMoeBlock.moe_infer drives its expert loop off "
                   "`tokens_per_expert.cpu().numpy()`, i.e. off routing VALUES, which a "
                   "shape-only trace does not have. Replaced with an even split of the sorted "
                   "tokens across experts: the op structure (per-expert gate/up/down on "
                   "[n, d_model]) is faithful, the per-expert token COUNT is not -- that is "
                   "runtime data no single trace could report. Experts traced: "
                   f"{'all' if _CAP <= 0 else _CAP} of {n_exp} per layer; the "
                   "rest are structurally identical (same three projections, same widths) and "
                   "differ only in weight values, which a weightless trace cannot observe. "
                   "`E` in structure.yaml still comes from the config."),
    }
    _MOE_PATCH_CACHE[cls] = entry
    return entry
