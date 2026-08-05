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
    with torch.device("meta"):
        model = _from_config(cfg, trust_remote_code, dtype)
    model.eval()
    return model


def load_fake(cfg, trust_remote_code: bool = True, dtype=None):
    """Tier 1 remedy target: some ops lack meta kernels. FakeTensorMode is the fallback --
    still zero real compute, just a different backend for shape inference."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
    with fake_mode:
        model = _from_config(cfg, trust_remote_code, dtype)
    model.eval()
    return model, fake_mode
