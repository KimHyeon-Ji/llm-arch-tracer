"""Step 2 -- meta-device load. No weights are ever fetched or computed."""
import torch
from transformers import AutoModelForCausalLM

# transformers 5.x compat shim: hybrid Mamba models (e.g. Nemotron-H) derive layer_types that
# include cache-less layer kinds like 'mlp' (FFN-only layers). DynamicCache(config=...) dispatches
# EVERY layer type through DYNAMIC_LAYER_TYPE_MAPPING and KeyErrors on 'mlp' (no cache class),
# crashing the forward when use_cache=True. mlp layers never touch the cache, so mapping 'mlp' to
# a plain (empty) DynamicLayer placeholder keeps layer indexing valid and is harmless. setdefault
# => only added if absent; no effect on any other model. (Root cause found via Tier 2 source read.)
try:
    from transformers.cache_utils import DYNAMIC_LAYER_TYPE_MAPPING, DynamicLayer
    DYNAMIC_LAYER_TYPE_MAPPING.setdefault("mlp", DynamicLayer)
except Exception:
    pass

# transformers 5.x compat shim #2: `OutputRecorder` MOVED from transformers.utils.generic to
# transformers.modeling_utils. Repo modeling files written against 4.5x import it from the old
# path and die at import time (Kimi-K3's modeling_kimi_linear.py). Re-export the SAME class under
# its old name -- not a reimplementation, so it cannot drift from what the repo code expects.
# Only added if genuinely absent, so a future transformers that restores the name wins.
try:
    from transformers.utils import generic as _tf_generic
    if not hasattr(_tf_generic, "OutputRecorder"):
        from transformers.modeling_utils import OutputRecorder as _OutputRecorder
        _tf_generic.OutputRecorder = _OutputRecorder
except Exception:
    pass


def _kda_prepare(cfg) -> list:
    """Route KDA through torch when this architecture needs it -- see src/kda_shim.py.

    Triggered by the config itself: `linear_attn_config` is Kimi's own marker for the KDA layers,
    and their modeling file imports Triton kernels at module scope, so the shim has to be in place
    BEFORE `from_config` loads that file. Returns adaptation-log entries (empty for every other
    architecture, which never touches this path).
    """
    if not getattr(cfg, "linear_attn_config", None) and not getattr(
            getattr(cfg, "text_config", None), "linear_attn_config", None):
        return []
    import kda_shim
    log = []
    entry = kda_shim.install()
    if entry:
        log.append(entry)
    compat = kda_shim.patch_transformers_compat()
    if compat:
        log.append({"tier": 1, "remedy": "transformers_compat",
                    "detail": "저장소 remote code 가 구버전 transformers 를 전제해 보정: "
                              + ", ".join(compat)})
    return log


def _from_config(cfg, trust_remote_code, dtype):
    """AutoModelForCausalLM.from_config, optionally with a non-default parameter dtype. dtype only
    affects meta-tensor dtypes (shapes are identical either way); it's a Tier 1 remedy for kernels
    that ASSERT a specific dtype -- e.g. gpt-oss's MoE grouped-matmul rejects fp32 and wants BF16.
    set_default_dtype is kwarg-name-agnostic across the torch_dtype->dtype rename."""
    if dtype is None:
        return AutoModelForCausalLM.from_config(cfg, trust_remote_code=trust_remote_code)
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        return AutoModelForCausalLM.from_config(cfg, trust_remote_code=trust_remote_code)
    finally:
        torch.set_default_dtype(prev)


def load_meta(cfg, trust_remote_code: bool = True, dtype=None):
    _kda_prepare(cfg)
    with torch.device("meta"):
        model = _from_config(cfg, trust_remote_code, dtype)
    model.eval()
    import kda_shim as _ks
    _ks.backfill_cache_class(model)          # the repo's Cache class exists only after the load
    _ks.reset_attn_implementation(model)     # the constructor may have forced flash-attn
    return model


def load_fake(cfg, trust_remote_code: bool = True, dtype=None):
    """Tier 1 remedy target: some ops lack meta kernels. FakeTensorMode is the fallback --
    still zero real compute, just a different backend for shape inference."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    _kda_prepare(cfg)
    fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
    with fake_mode:
        model = _from_config(cfg, trust_remote_code, dtype)
    model.eval()
    import kda_shim as _ks
    _ks.backfill_cache_class(model)
    _ks.reset_attn_implementation(model)
    return model, fake_mode
