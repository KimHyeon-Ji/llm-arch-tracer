"""Derive the "major operator" view from a full symbolic op trace (01-main.md section 6.1).

The full <phase>.trace.raw.jsonl captures every ATen primitive, including layout/view plumbing
that costs no FLOPs and moves no real bytes. For inference-latency reasoning we want only the
operators that dominate compute or memory. This module filters a full trace down to those,
contracts the dependency graph *through* the pruned nodes (so depends_on still forms a valid
DAG over the survivors), and renumbers op_id sequentially from 0.

Selection criteria (agreed with the user):
  * DROP whole position-embedding precompute modules (module_path names "rotary"/"rope"):
    cos/sin are computed once per forward and broadcast to every layer -- amortized, negligible.
  * ROLL UP each normalization module (leaf name contains "norm", or GPT-2 "ln_1/ln_2/ln_f")
    into ONE rmsnorm/layernorm row -- RMSNorm otherwise decomposes into pow/mean/add/rsqrt/mul
    and floods the view with five rows per norm.
  * KEEP (always) the compute / attention / activation core:
      linear, matmul, batched_matmul, grouped_matmul, sdpa, conv1d, embedding,
      softmax, silu, gelu, relu, sigmoid, tanh, exp, layernorm, rmsnorm
  * KEEP (size-gated) elementwise_add, elementwise_mul, concat, sum -- only when they touch a
    "wide" activation dim (d_model / d_ff / d_moe). This keeps residual adds, GLU gating and MoE
    combine, and drops the small RoPE-application muls, rotate_half / KV-append concats and the
    attention-mask add (all d_head- or T-scale, small byte).
  * DROP everything else (view/transpose/expand/slice/select/clone/copy/cast, RoPE trig, and MoE
    routing plumbing: topk/sort/gather/scatter/cumsum/where/...).

After filtering, collapse_repeats() further folds structurally identical decoder layers into a
single representative block tagged with how many times (and which indices) it repeats, so the
major view shows the basic blocks once instead of every unrolled layer. The full trace under
full/ stays fully unrolled.

Shapes are already symbolic when this runs (build_table symbolizes before calling us), so the
size gate reads dimension symbols like "d_model" directly.
"""
import json
import re

ALWAYS_KEEP = {
    "linear", "matmul", "batched_matmul", "grouped_matmul", "sdpa", "conv1d", "embedding",
    "softmax", "silu", "gelu", "relu", "sigmoid", "tanh", "exp", "layernorm", "rmsnorm",
}
SIZE_GATED = {"elementwise_add", "elementwise_mul", "concat", "sum"}
WIDE_TOKENS = ("d_model", "d_ff", "d_moe")

_LN_RE = re.compile(r"^ln(_?\d+|_f)?$")
_ROPE_RE = re.compile(r"(^|[._])rope([._]|$)")


def _leaf(module_path):
    return (module_path or "").rsplit(".", 1)[-1].lower()


def _is_rotary(row):
    mp = (row.get("module_path") or "").lower()
    return "rotary" in mp or bool(_ROPE_RE.search(mp))


def _norm_key(row):
    """The module_path if this row lives in a normalization module, else None."""
    leaf = _leaf(row.get("module_path"))
    if "norm" in leaf or _LN_RE.match(leaf):
        return row.get("module_path")
    return None


def _last_dims(row):
    dims = []
    for operand in (row.get("input_shape") or []):
        if operand:
            dims.append(str(operand[-1]))
    for shp in (row.get("output_shape") or []):
        if shp:
            dims.append(str(shp[-1]))
    return dims


def _touches_wide(row):
    return any(any(tok in d for tok in WIDE_TOKENS) for d in _last_dims(row))


def _keep(row):
    ot = row.get("op_type")
    if ot in ALWAYS_KEEP:
        return True
    if ot in SIZE_GATED:
        return _touches_wide(row)
    return False


def _short_raw(raw):
    parts = str(raw).split(".")
    return parts[1] if len(parts) > 1 and parts[0] == "aten" else str(raw)


def _collapse_norm(members):
    """One synthesized rmsnorm/layernorm row standing in for a whole norm module's ops."""
    members = sorted(members, key=lambda m: m["op_id"])
    first, last = members[0], members[-1]
    optypes = {m.get("op_type") for m in members}
    mp_low = (first.get("module_path") or "").lower()
    kind = "rmsnorm" if ("rsqrt" in optypes or "rms" in mp_low) else "layernorm"

    params = []
    for m in members:
        for p in (m.get("params") or []):
            if p not in params:
                params.append(p)
    # the norm weight is the 1-D operand of whichever op carries a *.weight param (not fabricated
    # -- it is an actually-traced operand shape, e.g. ["d_model"]).
    weight_shape = None
    for m in members:
        if any(str(p).endswith(".weight") for p in (m.get("params") or [])):
            for operand in (m.get("input_shape") or []):
                if isinstance(operand, list) and len(operand) == 1:
                    weight_shape = operand
                    break
            if weight_shape:
                break

    row = dict(first)
    row["op_type"] = kind
    row["input_shape"] = first.get("input_shape")
    row["output_shape"] = last.get("output_shape")
    row["weight_shape"] = weight_shape
    row["params"] = params
    row["raw_op"] = "+".join(_short_raw(m.get("raw_op")) for m in members)
    row["unmapped"] = False
    row["_members"] = members
    return row


def extract_major(rows):
    """rows: full trace as ordered dicts with SYMBOLIC shapes (build_table._ordered_row output).
    Returns the major-operator rows, renumbered 0..N-1 with contracted depends_on."""
    by_id = {r["op_id"]: r for r in rows}

    # 1. group the ops of each normalization module (rotary modules are dropped wholesale)
    norm_groups = {}
    for r in rows:
        if _is_rotary(r):
            continue
        k = _norm_key(r)
        if k is not None:
            norm_groups.setdefault(k, []).append(r)
    member_to_rep = {}
    for members in norm_groups.values():
        rep = min(m["op_id"] for m in members)
        for m in members:
            member_to_rep[m["op_id"]] = rep

    # 2. survivors: one synthesized row per norm module + each kept standalone op
    survivors = {}
    for members in norm_groups.values():
        survivors[min(m["op_id"] for m in members)] = _collapse_norm(members)
    for r in rows:
        if _is_rotary(r) or r["op_id"] in member_to_rep:
            continue
        if _keep(r):
            survivors[r["op_id"]] = dict(r)
    survivor_ids = set(survivors)

    def rep_of(oid):
        if oid in member_to_rep:
            return member_to_rep[oid]
        if oid in survivor_ids:
            return oid
        return None

    # 3. contract dependencies through pruned nodes: for any original id, the set of nearest
    # surviving ancestors reachable by walking depends_on backward. Iterative post-order with
    # memoization (traces can be deep, so no recursion).
    memo = {}

    def resolve(start):
        if start in memo:
            return memo[start]
        stack = [(start, False)]
        while stack:
            node, done = stack.pop()
            if node in memo:
                continue
            rp = rep_of(node)
            if rp is not None:
                memo[node] = {rp}
                continue
            if not done:
                stack.append((node, True))
                for p in (by_id.get(node, {}).get("depends_on") or []):
                    if p not in memo:
                        stack.append((p, False))
            else:
                acc = set()
                for p in (by_id.get(node, {}).get("depends_on") or []):
                    acc |= memo.get(p, set())
                memo[node] = acc
        return memo[start]

    def raw_deps(rep_id):
        row = survivors[rep_id]
        members = row.get("_members")
        if members:  # a collapsed norm: external deps only (drop intra-group edges)
            ids = {m["op_id"] for m in members}
            return {x for m in members for x in (m.get("depends_on") or []) if x not in ids}
        return set(row.get("depends_on") or [])

    # 4. renumber survivors in trace order and remap depends_on to the new ids
    order = sorted(survivor_ids)
    new_id = {old: i for i, old in enumerate(order)}
    out = []
    for old in order:
        row = {k: v for k, v in survivors[old].items() if k != "_members"}
        deps = set()
        for d in raw_deps(old):
            deps |= {s for s in resolve(d) if s != old}
        row["op_id"] = new_id[old]
        row["depends_on"] = sorted(new_id[s] for s in deps if s in new_id)
        out.append(row)
    return out


# -- repeat folding: collapse structurally identical decoder layers to one block ---------------

_STACK_NAMES = ("layers", "h", "blocks", "block", "layer")  # mirrors scope._STACK_NAMES


def _rel_module(mp, layer_idx):
    """module_path with the `.<stack>.<layer_idx>` prefix stripped, so the same sub-module in
    different layers compares equal (e.g. model.layers.7.self_attn.q_proj -> self_attn.q_proj;
    a layer-root op like model.layers.7 -> '')."""
    if not mp:
        return ""
    stacks = "|".join(_STACK_NAMES)
    m = re.search(rf"\.(?:{stacks})\.{layer_idx}(?:\.|$)", mp)
    return mp[m.end():] if m else mp


def _op_sig(o, idx):
    return (o.get("op_type"), _rel_module(o.get("module_path"), idx),
            json.dumps(o.get("input_shape")), json.dumps(o.get("weight_shape")),
            json.dumps(o.get("output_shape")))


def _layer_sig(ops):
    """Structural signature of one decoder layer: order-sensitive tuple over its ops of
    (op_type, layer-relative module, symbolic input/weight/output shapes). Two layers with the
    same signature are the same block regardless of layer index or absolute op_id."""
    return tuple(_op_sig(o, o.get("layer_idx")) for o in ops)


def full_layer_signatures(full_rows):
    """{layer_idx: signature} computed over the FULL trace (all ops, before major filtering).
    collapse_repeats() groups blocks by this so that two decoder layers count as distinct blocks
    whenever their full structure differs -- even if the difference is only in ops the major view
    drops (e.g. a NoPE layer vs a RoPE layer, which differ solely in rotary-application ops). This
    keeps the block count aligned with the real per-layer architecture."""
    layers = {}
    for r in full_rows:
        li = r.get("layer_idx")
        if li is not None:
            layers.setdefault(li, []).append(r)
    return {idx: tuple(_op_sig(o, idx) for o in ops) for idx, ops in layers.items()}


def _block_composition(ops):
    """Best-effort human label for a decoder block from its ops: mixer part (attn / MLA / SSM /
    linAttn / xLSTM) + FFN part (FFN / MoE), e.g. 'MLA+MoE', 'attn+FFN', 'SSM'. Falls back to
    'block' when nothing is recognized -- the op rows themselves carry the detail."""
    types = {o.get("op_type") for o in ops}
    ml = " ".join((o.get("module_path") or "") for o in ops).lower()
    mixer = None
    if "softmax" in types or "sdpa" in types:
        mixer = "MLA" if ("q_a_proj" in ml or "kv_a_proj" in ml or "kv_a_layernorm" in ml) else "attn"
    elif "conv1d" in types:
        mixer = "SSM"
    elif any(k in ml for k in ("linear_attn", "deltanet", "linear_attention")):
        mixer = "linAttn"
    elif any(k in ml for k in ("mlstm", "xlstm", "slstm")):
        mixer = "xLSTM"
    ffn = None
    if "grouped_matmul" in types or "experts" in ml:
        ffn = "MoE"
    elif any(k in ml for k in ("gate_proj", "up_proj", "down_proj", "gate_up_proj", "feed_forward")):
        ffn = "FFN"
    parts = [p for p in (mixer, ffn) if p]
    return "+".join(parts) if parts else "block"


def _free_block_type(row):
    """block_type tag for a non-layer (one-off) op: embed / norm / head, else '-'."""
    ot = row.get("op_type")
    if ot == "embedding":
        return "embed"
    if ot in ("layernorm", "rmsnorm"):  # op_type is reliable even when the leaf is 'ln_f'
        return "norm"
    if "lm_head" in (row.get("module_path") or "").lower():
        return "head"
    return "-"


def _compact_ranges(idxs):
    """[0,1,2,3,23] -> '0-3,23'."""
    idxs = sorted(set(idxs))
    parts, start, prev = [], idxs[0], idxs[0]
    for x in idxs[1:]:
        if x == prev + 1:
            prev = x
        else:
            parts.append(f"{start}-{prev}" if start != prev else f"{start}")
            start = prev = x
    parts.append(f"{start}-{prev}" if start != prev else f"{start}")
    return ",".join(parts)


def collapse_repeats(mrows, layer_sigs=None):
    """Fold repeated decoder layers in a major-op list. Keeps ops with layer_idx=None (embedding,
    final norm, lm_head) as-is; for layer ops, keeps only the first layer of each distinct
    signature and tags every row with `repeat` (how many layers that block stands for) and
    `layers` (which indices). depends_on that pointed into a dropped repeat layer is redirected to
    the matching op of the representative block. op_id is renumbered 0..M.

    `layer_sigs` ({layer_idx: signature}, from full_layer_signatures) is the grouping key when
    given -- so blocks split on the FULL per-layer structure, not just the surviving major ops
    (see full_layer_signatures). Without it, the signature falls back to the major ops."""
    layers = {}  # layer_idx -> [ops] (insertion order = trace order)
    for r in mrows:
        li = r.get("layer_idx")
        if li is not None:
            layers.setdefault(li, []).append(r)

    # group layer indices by signature; representative = first index seen for that signature
    sig_rep, idx_rep, rep_idxs = {}, {}, {}
    for idx, ops in layers.items():
        sig = layer_sigs[idx] if (layer_sigs and idx in layer_sigs) else _layer_sig(ops)
        rep = sig_rep.setdefault(sig, idx)
        idx_rep[idx] = rep
        rep_idxs.setdefault(rep, []).append(idx)

    # dropped op -> the representative block's op at the same position (for dep redirection)
    op_to_rep = {}
    for idx, ops in layers.items():
        rep = idx_rep[idx]
        if rep != idx:
            for pos, o in enumerate(ops):
                op_to_rep[o["op_id"]] = layers[rep][pos]["op_id"]

    # human-readable composition label per representative block (attn+FFN, MLA+MoE, SSM, ...)
    comp = {rep: _block_composition(layers[rep]) for rep in rep_idxs}

    # emit free ops + representative-layer ops, in original order
    emitted = []
    for r in mrows:
        li = r.get("layer_idx")
        if li is None:
            emitted.append((r, 1, "", _free_block_type(r)))
        elif idx_rep[li] == li:  # representative layer only
            emitted.append((r, len(rep_idxs[li]), _compact_ranges(rep_idxs[li]), comp[li]))

    new_id = {row["op_id"]: i for i, (row, *_ ) in enumerate(emitted)}

    def remap(d):
        if d in new_id:
            return new_id[d]
        rd = op_to_rep.get(d)  # dropped repeat-layer op -> its representative
        return new_id.get(rd) if rd is not None else None

    out = []
    for row, repeat, lays, block_type in emitted:
        nr = dict(row)
        nr["op_id"] = new_id[row["op_id"]]
        deps = []
        for d in (row.get("depends_on") or []):
            nd = remap(d)
            if nd is not None and nd != nr["op_id"] and nd not in deps:
                deps.append(nd)
        nr["depends_on"] = sorted(deps)
        nr["block_type"] = block_type
        nr["repeat"] = repeat
        nr["layers"] = lays
        out.append(nr)
    return out
