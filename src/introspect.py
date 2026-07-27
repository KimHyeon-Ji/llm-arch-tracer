"""Step 3 -- architecture introspection. Derive safe run parameters from config instead
of hardcoding them in a profile. See 01-main.md Step 3 and 02-new-module-handling.md
(seq_len-too-small is a Tier 1 failure this directly prevents)."""
import re

# config keys whose numeric values constrain how long seq_len must be (top-k selection,
# sliding windows, compression ratios, etc.). If seq_len < these, ops like topk/gather
# will error at trace time.
_SEQ_CONSTRAINT_KEYS = re.compile(
    r"(sliding_window|window|top_?k|index_topk|attention_top_k|"
    r"compress_rate|compress_rates|n_group|selected)",
    re.I,
)

_EXTRA_ENTRYPOINT_PATTERNS = re.compile(
    r"(mtp|nextn|multi_token|draft|speculativ|medusa|eagle)", re.I
)


def _numbers(x):
    if isinstance(x, bool):
        return
    if isinstance(x, int):
        yield x
    elif isinstance(x, dict):
        for v in x.values():
            yield from _numbers(v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from _numbers(v)


def derive_min_seq_len(cfg, margin: int = 8, cap: int = 2048) -> int:
    d = cfg.to_dict()
    bound = 1
    for k, v in d.items():
        if _SEQ_CONSTRAINT_KEYS.search(k):
            for n in _numbers(v):
                bound = max(bound, int(n))
    return min(max(bound * 2 + margin, 16), cap)


def layer_schedule(cfg):
    """Heterogeneous per-layer attention schedule, if the model has one (e.g. sliding vs
    global, CSA vs HCA). None means the model is architecturally uniform across layers."""
    return getattr(cfg, "layer_types", None)


def find_extra_entrypoints(model):
    """Modules that live outside the main forward() call graph (MTP heads, draft models,
    etc.) -- these need a separate trace call, see entrypoints handling in run.py."""
    found = []
    for name, mod in model.named_modules():
        if name == "":
            continue
        if _EXTRA_ENTRYPOINT_PATTERNS.search(name) or _EXTRA_ENTRYPOINT_PATTERNS.search(
            type(mod).__name__
        ):
            found.append((name, type(mod).__name__))
    # keep only top-level matches, drop nested duplicates
    return [
        f for f in found if not any(f[0] != g[0] and f[0].startswith(g[0] + ".") for g in found)
    ]
