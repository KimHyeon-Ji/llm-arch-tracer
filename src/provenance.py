"""Step 1 -- revision/config snapshot + architecture support gate. See 01-main.md Step 1."""
import json
import hashlib
import torch
import transformers
from huggingface_hub import model_info
from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING


def needs_remote_code(cfg) -> bool:
    """Whether we must fall back to the repo's remote modeling code. Native transformers
    implementation is PREFERRED whenever the architecture is builtin -- remote modeling files
    are often pinned to an older transformers and break on a newer major (e.g. DeepSeek remote
    code imports `is_torch_fx_available`, removed in transformers 5.x). Only architectures with
    no native support genuinely need trust_remote_code."""
    return getattr(cfg, "model_type", None) not in CONFIG_MAPPING


def snapshot(model_id: str, revision: str | None = None, config_overrides: dict | None = None):
    """Fetch config.json (not weights) and pin revision to a commit hash. Prefer the native
    config class (trust_remote_code=False); fall back to remote only if there's no native one.

    config_overrides (from the profile) are passed as kwargs to from_pretrained, overriding
    config.json values BEFORE the config class validates them. Needed when a model's config.json
    selects options the installed library rejects but that don't change the architecture -- e.g.
    xLSTM's config.json asks for a triton kernel (chunkwise--triton_xl_chunk) that (a) the native
    config's strict dataclass rejects and (b) can't run on meta anyway; override to the native
    kernel. Recorded in provenance for reproducibility."""
    overrides = dict(config_overrides or {})
    info = model_info(model_id, revision=revision)
    resolved = info.sha  # actual commit hash -- this is the source-of-truth revision
    if overrides:
        # Patch the config DICT before the config class validates it. Passing overrides as
        # from_pretrained kwargs is too late for strict-dataclass configs (they validate the
        # config.json value in __init__, before kwargs apply) -- e.g. xLSTM rejects its own
        # config.json triton kernel. Fetch dict -> update -> from_dict (native class).
        config_dict, _unused = PretrainedConfig.get_config_dict(model_id, revision=resolved)
        mt = config_dict.get("model_type")
        config_dict.update(overrides)
        if mt in CONFIG_MAPPING:
            cfg = CONFIG_MAPPING[mt].from_dict(config_dict)
        else:
            cfg = AutoConfig.from_pretrained(model_id, revision=resolved, trust_remote_code=True, **overrides)
    else:
        try:
            cfg = AutoConfig.from_pretrained(model_id, revision=resolved, trust_remote_code=False)
        except Exception:
            cfg = AutoConfig.from_pretrained(model_id, revision=resolved, trust_remote_code=True)
    cfg_dict = cfg.to_dict()
    prov = {
        "model_id": model_id,
        "revision_requested": revision,
        "revision_resolved": resolved,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        # HF repo creation date -- a real, verifiable proxy for the model's release date (used as
        # the gallery card DATE; not exact release, labeled as such). Not fabricated (P1).
        "hf_created_at": (str(getattr(info, "created_at", "") or "")[:10] or None),
        "trust_remote_code": needs_remote_code(cfg),  # False => using native transformers impl
        "config_overrides": overrides,  # profile-supplied config field overrides applied at load
        "config_sha256": hashlib.sha256(
            json.dumps(cfg_dict, sort_keys=True).encode()
        ).hexdigest(),
        "config": cfg_dict,
        "capture_backend": None,       # filled in later by adapt.py
        "seq_len_used": None,
        "attn_implementation_used": None,
        "adaptation_log": [],          # Tier 0/1/2/3 actions taken, see 02-new-module-handling.md
    }
    # Multimodal/composite configs (e.g. Llama-4 'llama4') nest the LLM under text_config; we
    # trace the text decoder, so hand back the text sub-config. get_text_config() returns the
    # config itself for text-only models -> no-op for every other model. Provenance above still
    # records the ORIGINAL full config (config_sha256/config), the faithful repo artifact.
    if hasattr(cfg, "get_text_config"):
        try:
            text_cfg = cfg.get_text_config()
            # A nested sub-config carries its own `auto_map` but an EMPTY `_name_or_path`, and
            # AutoModelForCausalLM.from_config resolves remote code against that field -- so a
            # composite repo whose text decoder needs trust_remote_code failed with
            # "Repo id must use alphanumeric chars ... : ''" (Kimi-K3, whose text tower is a
            # separate `kimi_linear` modeling file). The parent knows the repo; hand it down.
            if not getattr(text_cfg, "_name_or_path", None):
                text_cfg._name_or_path = getattr(cfg, "_name_or_path", None) or model_id
            cfg = text_cfg
        except Exception:
            pass
    return cfg, prov


def support_gate(cfg):
    """Tier 0 check: does the installed transformers know this architecture at all?"""
    mt = getattr(cfg, "model_type", None)
    builtin = mt in CONFIG_MAPPING
    has_remote = bool(getattr(cfg, "auto_map", None))
    if not builtin and not has_remote:
        raise RuntimeError(
            f"model_type '{mt}' unsupported: upgrade transformers, or the repo needs "
            f"trust_remote_code with a custom modeling file."
        )
    return {"model_type": mt, "builtin": builtin, "remote_code": has_remote}


def write_provenance(path, prov):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2, default=str)
