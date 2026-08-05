"""Decide whether a shape axis depends on the sequence length, by comparing the two phases.

A value collision between a config symbol and a T-bearing expression cannot be settled from one
trace: GLM-4.5-Air flattens `[T=16, k=8]` to 128 and its expert count E is also 128, so both
names fit. But the model was traced TWICE, at different sequence lengths, and the two concrete
sidecars are already in every model directory:

    prefill  view -> [128]        decode  view -> [8]

E is a config field and cannot change between phases; `k*T` must. One comparison decides it, with
no new tracing and no guessing. The same evidence settles `T/m_csa` vs `d_head` in DeepSeek-V4-Pro
and every other collision of this shape.

Ops are matched across phases by (module_path, op_type, n-th occurrence). That is exact for
standard transformer blocks (93-95% of ops pair up) and partial for models whose prefill runs many
more scan iterations than decode (xLSTM 14%, Qwen3-Next 22%). Where an op does not pair, this
module says nothing and the ordinary rendering stands -- absence of evidence is not evidence.
"""
import collections
import json
import os

_FIELDS = ("input_shape", "output_shape")


def _index(model_dir: str, phase: str) -> dict:
    """{(module_path, op_type, ordinal): concrete record} for one phase."""
    conc_path = os.path.join(model_dir, "full", f"{phase}.shapes.concrete.jsonl")
    raw_path = os.path.join(model_dir, "full", f"{phase}.trace.raw.jsonl")
    if not (os.path.exists(conc_path) and os.path.exists(raw_path)):
        return {}
    conc = {}
    for line in open(conc_path, encoding="utf-8"):
        r = json.loads(line)
        conc[r["op_id"]] = r
    ident = {}
    for line in open(raw_path, encoding="utf-8"):
        r = json.loads(line)
        ident[r.get("op_id")] = (r.get("module_path") or "", r.get("op_type"))
    seen, out = collections.Counter(), {}
    for op_id in sorted(conc, key=lambda x: x if isinstance(x, int) else 0):
        key = ident.get(op_id)
        if not key:
            continue
        seen[key] += 1
        out[(key[0], key[1], seen[key])] = (op_id, conc[op_id])
    return out


def build(model_dir: str) -> dict:
    """{phase: {op_id: {(field, shape_index, axis): bool}}} -- True where the axis moved.

    Only axes that PAIR across the two phases get an entry. `True` means the size changed with
    the sequence length, `False` means it did not; a missing entry means the two traces gave no
    comparable pair and nothing should be inferred.
    """
    pre, dec = _index(model_dir, "prefill"), _index(model_dir, "decode")
    if not pre or not dec:
        return {}
    # A module is usable only if EVERY one of its prefill ops found a decode partner. Partial
    # coverage is worse than none: relabelling some ops of a region and not others leaves one
    # tensor with two names, which is what a reader cannot resolve. Models whose prefill runs
    # more scan iterations than decode (Nemotron, Zamba2, xLSTM, Qwen3-Next) fail this for their
    # scan modules and keep their ordinary rendering there -- measured, after the ungated version
    # moved flow_ambig on exactly those models (Nemotron 0 -> 210, DeepSeek-V4-Pro 183 -> 935).
    seen, paired = collections.Counter(), collections.Counter()
    for key in pre:
        seen[key[0]] += 1
        if key in dec:
            paired[key[0]] += 1
    usable = {mod for mod, n in seen.items() if paired.get(mod, 0) == n}

    out = {"prefill": {}, "decode": {}}
    for key in set(pre) & set(dec):
        if key[0] not in usable:
            continue
        (p_id, p_rec), (d_id, d_rec) = pre[key], dec[key]
        for field in _FIELDS:
            p_shapes, d_shapes = p_rec.get(field) or [], d_rec.get(field) or []
            for si, (ps, ds) in enumerate(zip(p_shapes, d_shapes)):
                if not isinstance(ps, list) or not isinstance(ds, list) or len(ps) != len(ds):
                    continue        # rank changed -- not the same axis layout, say nothing
                for ax, (pv, dv) in enumerate(zip(ps, ds)):
                    if not isinstance(pv, int) or not isinstance(dv, int):
                        continue
                    if pv == 1 or dv == 1:
                        continue    # singletons are governed by the batch-axis invariant
                    moved = pv != dv
                    out["prefill"].setdefault(p_id, {})[(field, si, ax)] = moved
                    out["decode"].setdefault(d_id, {})[(field, si, ax)] = moved
    return out


def axis_hints(tmap: dict, phase: str, op_id) -> dict:
    """{(field, shape_index, axis): bool} for one op, or {} when nothing paired."""
    return ((tmap or {}).get(phase) or {}).get(op_id) or {}
