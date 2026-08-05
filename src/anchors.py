"""Module-declared dimensions as the anchor for shape labelling.

WHY THIS EXISTS
---------------
The trace knows, exactly and without inference, which parameter each op consumed
(`tracer.param_origin`, tensor identity) and what that parameter's shape is. An
`nn.Linear` therefore *declares* its two widths: `weight.shape == [out_features,
in_features]`. That is ground truth about the architecture.

The symbolizer nevertheless threw it away and re-derived meaning from the bare integer:
`dim(5120, module_path)` searched every config value for a match and used `scope:` regexes
to break ties. Every labelling bug this project has found is that one design -- a number
that coincides with a second symbol's value, resolved by priority instead of by structure:

  * Llama-3.1-405B `n_h == d_head == 128`   -> KV head-size axes labelled `n_h` (16,859 axes)
  * Llama-4        `E  == d_head == 128`    -> attention head dim labelled `E`
  * Zamba2 `q_proj` is Linear(4096, 4096)   -> weight rendered `[d_attn, n_h*d_head]`, reversed
  * gpt-oss `d_model == d_ff == 2880`       -> residual stream labelled `d_ff`

None of those are ambiguous to the module itself. `k_proj.out_features` is the KV
projection width *because of which module it is*, whatever number it holds.

WHAT THIS MODULE DOES
---------------------
1. `declared_dims`  -- recover every parameter-bearing module's declared widths from the
   trace. Uses the CONCRETE shapes, so it works offline on already-published models
   (`full/<phase>.shapes.concrete.jsonl`); no re-trace is needed to adopt anchoring.
2. `build_anchors`  -- resolve each module's widths to a label ONCE, and reuse that label
   at every occurrence. One module -> one labelling, in every layer and every op.
3. Factor split    -- when a module's output width resolves to a product `A*B`, a
   downstream reshape that splits exactly that width into two axes gets `A` and `B` by
   *derivation from the anchor*, not by matching each axis against config values
   independently. This is what makes n_h/n_kv structurally impossible to confuse.

It does NOT invent names. Naming still comes from `rules/symbols.yaml` through the same
resolver; anchoring only decides WHICH axis a name applies to, and pins it everywhere.
"""
import re

# `.layers.0.` and `.layers.31.` are the same module in different layers, and `experts.7.`
# is one expert of the same expert module. Anchors are per-ARCHITECTURE-position, so the
# indices are normalised away and every layer contributes to (and agrees on) one entry.
_IDX = re.compile(r"\.(\d+)(?=\.|$)")


def module_key(module_path: str | None) -> str | None:
    """Layer/expert-index-free module identity, e.g. `model.layers.7.self_attn.k_proj`
    -> `model.layers.*.self_attn.k_proj`."""
    if not module_path:
        return None
    return _IDX.sub(".*", module_path)


def _param_module(params) -> str | None:
    """The module that owns a single `*.weight`/`*.bias` parameter."""
    if not params or len(params) != 1:
        return None
    name = str(params[0])
    return name.rsplit(".", 1)[0] if "." in name else None


def declared_dims(rows, concrete: dict) -> dict:
    """{module_key: {"shape": concrete weight shape, "in": v, "out": v, "path": sample path}}

    `in` is the CONTRACTED width -- the axis the incoming activation is multiplied along --
    and `out` is what the module produces. They are read off the op, not assumed from axis
    order: a Linear hands `aten.mm` the transposed weight, while a batched expert weight goes
    in as stored, so a fixed [0]=out/[1]=in convention is wrong for one of them. The
    contracted axis is the one whose size equals the activation's last axis, which is a fact
    of the multiplication rather than a convention.
    """
    out = {}
    for row in rows:
        mod = _param_module(row.get("params"))
        if not mod:
            continue
        # Keyed by PARAMETER, not module. A module can hold several raw Parameters of different
        # shapes -- gpt-oss's `mlp.experts` has gate_up_proj [E, d_model, 2*d_moe] and down_proj
        # [E, d_moe, d_model] -- and keying by module applied the first one's axis roles to the
        # second, labelling a 2880-wide axis `2*d_moe` (=5760). Caught by the arithmetic gate.
        key = module_key((row.get("params") or [""])[0])
        # Prefer the op that actually CONTRACTS with the weight. A parameter is touched by a bare
        # `aten.t` before its `mm`, and that transpose has no activation operand -- so an anchor
        # taken from it has no incoming tensor to trace upstream, and the dataflow correction
        # below silently does nothing. Keep looking until a contracting op for this module shows
        # up; fall back to whatever we have if none ever does.
        if out.get(key, {}).get("contracting"):
            continue
        conc = concrete.get(row.get("op_id")) or {}
        w = conc.get("weight_shape")
        ins = conc.get("input_shape") or []
        if not w:
            # The tracer only fills weight_shape for rank>=2 params, so a norm's 1-D scale vector
            # never lands there -- but it IS one of the operands of the `mul` that consumes it.
            # Recover it: these are the anchors that pin the residual stream, and without them the
            # projections have nothing upstream to chain off.
            if not str((row.get("params") or [""])[0]).endswith(".weight"):
                continue
            cand = [o for o in ins if isinstance(o, list) and len(o) == 1
                    and isinstance(o[0], int) and o[0] >= 2]
            if len(cand) != 1:
                continue
            w = cand[0]
        if len(w) == 1:
            # A norm's scale vector declares the width of the stream it normalises, exactly and
            # with no axis ambiguity at all. These are the cheapest anchors available and they
            # pin the residual stream, which is what the projections then chain off.
            out[key] = {"shape": list(w), "path": mod, "in": w[0], "out": w[0],
                        "in_axis": 0, "out_axis": 0, "op_id": row.get("op_id"),
                        "param": (row.get("params") or [""])[0],
                        "contracting": True, "rank1": True}
            continue
        ins = conc.get("input_shape") or []
        wp = row.get("weight_pos")
        # activation operand = the one that is NOT the weight
        act = None
        for i, operand in enumerate(ins):
            if i != wp and operand:
                act = operand
        entry = {"shape": list(w), "path": mod, "in": None, "out": None,
                 "in_axis": None, "out_axis": None, "op_id": row.get("op_id"),
                 "param": (row.get("params") or [""])[0],
                 "contracting": act is not None}
        n = len(w)
        if act and isinstance(act[-1], int):
            contracted = act[-1]
            # the weight axis matching the contracted width is `in`; the other trailing axis is
            # `out`. Ambiguous only when the weight is square, where in == out anyway.
            if w[n - 1] == contracted:
                entry["in_axis"], entry["out_axis"] = n - 1, n - 2
            elif w[n - 2] == contracted:
                entry["in_axis"], entry["out_axis"] = n - 2, n - 1
        if entry["in_axis"] is None:
            entry["out_axis"], entry["in_axis"] = n - 2, n - 1   # [out, in] as stored (Linear)
        entry["in"], entry["out"] = w[entry["in_axis"]], w[entry["out_axis"]]
        out[key] = entry
    return out


def build_anchors(rows, concrete: dict, resolver, canon: dict | None = None,
                  tags: dict | None = None, param_axes: dict | None = None) -> dict:
    """{module_key: {"in": label, "out": label, "split": (count_label, size_label) | None}}

    The label for each width is produced by the ordinary resolver, but computed ONCE per
    module and then reused. That alone removes the "one tensor, two names" class: the same
    parameter can no longer be rendered `[n_h*d_head, d_attn]` at its transpose and
    `[d_attn, n_h*d_head]` at its matmul.

    `split` is set when the output width resolves to a two-factor product of plain symbols
    (`n_h*d_head`, `n_kv*d_head`, `2*d_moe`). A downstream reshape of exactly that width into
    two axes is then labelled from the factors instead of by independent value matching --
    the fix for the head-count/head-size confusion.
    """
    dims = declared_dims(rows, concrete)
    table = dict(getattr(resolver, "table", None) or {})
    _TABLE["t"] = table          # relabel() checks per-axis expressions against it
    anchors = {}
    for key, entry in dims.items():
        path = entry["path"]
        w = entry["shape"]
        labels = None
        if canon:
            hit = canon.get(path)
            if hit and tuple(hit[0]) == tuple(w):
                labels = list(hit[1])
        if labels is None:
            labels = resolver(w, path, is_weight=True)
        rec = {"in": None, "out": None, "split": None, "path": path}
        # axis indices were decided in declared_dims from the multiplication itself, so the
        # label mapping is positional and cannot mis-assign a square weight's two sides
        ia, oa = entry.get("in_axis"), entry.get("out_axis")
        if ia is not None and 0 <= ia < len(labels):
            rec["in"] = labels[ia]
        if oa is not None and 0 <= oa < len(labels):
            rec["out"] = labels[oa]
        rec["out_value"] = entry["out"]
        rec["in_value"] = entry["in"]
        # NOT carried: `in_axis`/`out_axis`. relabel() rule 1 reads them to decide which weight
        # axis to write, and because they were never on the record that rule has always been a
        # no-op -- every anchor correction to date comes from rule 2, the activation pin.
        #
        # Reviving it was tried on 2026-08-05 and REVERTED. Writing the anchor's in/out onto the
        # weight axes produced 396 arithmetically-false labels across Zamba2 and DeepSeek-V4 and
        # doubled gpt-oss's dataflow disagreement, because the axis roles derived from the
        # contraction do not always line up with the module's own in/out view (Zamba2's
        # `linear_q_adapter` has them reversed). An arithmetic guard on the tag did not help,
        # which is the evidence that the fault is in the axis mapping rather than in the names.
        # Fixing it needs per-parameter axis roles that are verified against the concrete widths,
        # not just carried over -- a separate change with its own audit.
        # The config tag beats the resolver where it speaks: it records the expression the model's
        # own __init__ computed, so it can tell `d_model` from `n_h*d_head` when the two are the
        # same number. Applied only when the expression names every dimension it uses -- see
        # tag_is_usable. src/symbolic_dims.py.
        # Per-axis expressions for THIS parameter, from the module's own cached widths
        # (symbolic_dims.param_axis_expressions). Unlike in_axis/out_axis these need no guess
        # about which side is which -- each axis was matched against the module's attributes and
        # is verified against the concrete width below.
        rec["axes"] = list((param_axes or {}).get(entry.get("param") or "") or [])
        t = (tags or {}).get(path) or {}
        for side in ("in", "out"):
            cand = t.get(side)
            if tag_is_usable(cand) and _evaluates_to(cand, entry[side], table):
                rec[side] = cand
        anchors[key] = rec
    # Does this parameter's module own exactly one parameter? relabel() needs it for rule 2.
    per_module = {}
    for rec in anchors.values():
        per_module[rec["path"]] = per_module.get(rec["path"], 0) + 1
    for rec in anchors.values():
        rec["sole_param"] = per_module.get(rec["path"], 0) == 1

    _apply_dataflow(anchors, dims, rows, concrete)
    for rec in anchors.values():          # split depends on the FINAL out label
        rec["split"] = _factor_split(rec["out"])
    return anchors


def _upstream_module(rows_by_id, concrete, start_op, act_shape, dims, max_hops=12):
    """module_key of the nearest anchored module upstream of `start_op` along the dataflow.

    Walks `depends_on` (tensor identity, not guesswork) past the view/transpose plumbing that
    sits between two real modules. Only follows edges whose producer OUTPUT is the activation
    we came in on, so it cannot wander into the weight's own `aten.t` branch."""
    seen, frontier = set(), [(start_op, act_shape)]
    for _hop in range(max_hops):
        nxt = []
        for op_id, want in frontier:
            row = rows_by_id.get(op_id)
            if row is None:
                continue
            for dep in (row.get("depends_on") or []):
                if dep in seen:
                    continue
                seen.add(dep)
                drow = rows_by_id.get(dep)
                if drow is None:
                    continue
                douts = (concrete.get(dep) or {}).get("output_shape") or []
                if want is not None and not any(list(o or []) == list(want) for o in douts):
                    # not the tensor we are tracing; could be the weight branch
                    if not any((list(o or [])[-1:] == list(want)[-1:]) for o in douts if o):
                        continue
                key = module_key(drow.get("module_path"))
                if key in dims and key != module_key(rows_by_id[start_op].get("module_path")):
                    return key
                nxt.append((dep, want))
        if not nxt:
            break
        frontier = nxt
    return None


def _apply_dataflow(anchors: dict, dims: dict, rows, concrete: dict) -> int:
    """Force every module's `in` label to equal its producer's `out` label. Returns the count
    of corrections.

    The width entering a module and the width leaving its producer are ONE tensor, so one label
    must serve both. Resolving them independently is what let Llama-3.1-405B render
    `q_proj.in` as `n_h*d_head`: that model has d_model == n_h*d_head == 16384, so the value
    alone cannot say whether the axis is the residual stream or the packed head layout. The
    producer (`input_layernorm`, width d_model and unambiguous) settles it."""
    rows_by_id = {r.get("op_id"): r for r in rows}
    fixed = 0
    for key, rec in anchors.items():
        entry = dims.get(key) or {}
        op_id = entry.get("op_id")
        if op_id is None or rec.get("in") is None:
            continue
        # A rank-1 (norm) anchor has ONE width, so its `in` and `out` are the same tensor and
        # must keep the same label. Correcting its `in` from the producer split the two apart:
        # OLMo-2 has d_model == n_h*d_head == 4096, so `post_attention_layernorm` took
        # `n_h*d_head` from o_proj's (itself unresolved) output and contradicted its own `out`.
        # Norms are the anchors that pin the residual stream -- they are corrected BY nothing.
        if entry.get("rank1"):
            continue
        conc = concrete.get(op_id) or {}
        wp = rows_by_id.get(op_id, {}).get("weight_pos")
        act = None
        for i, operand in enumerate(conc.get("input_shape") or []):
            if i != wp and operand:
                act = operand
        up = _upstream_module(rows_by_id, concrete, op_id, act, dims)
        if not up:
            continue
        up_rec = anchors.get(up) or {}
        if up_rec.get("out") and up_rec.get("out_value") == rec.get("in_value") \
                and up_rec["out"] != rec["in"]:
            rec["in"] = up_rec["out"]
            fixed += 1
    return fixed


# Ops that genuinely re-lay-out a packed dimension into [count, size]. A transpose/slice/expand
# moves or trims axes that already exist and cannot create a factorisation.
# Tried and REMOVED 2026-08-05: propagating a label across a reshape that only merges leading
# dims (same trailing width, same element count). It changed nothing measurable, because the
# disagreements it was meant to close sit between two NON-anchored positions -- OLMo-2's
# `elementwise_add` output and a `view` input, both 4096, neither authoritative. No propagation
# rule can settle that without inventing a tie-break; it needs distance-to-anchor ranking.
_RESHAPE_OPS = frozenset({"view", "reshape", "_unsafe_view"})
_MENTIONS_T = re.compile(r"\bT\b")

# See rule 3 in relabel(): the factor-split rule measured as a net regression and is disabled
# pending dataflow provenance. Flipping this back on without that work will re-introduce the
# coincidental-product rewrites documented there.
_ENABLE_SPLIT = False


def _block_anchors(anchors: dict, key: str | None):
    """Anchors for this module and for every module nested under it. A reshape inside
    `self_attn` is labelled from the projections that live in `self_attn`, which are the only
    modules that could have produced the tensor it is reshaping."""
    if not key:
        return []
    return [rec for k, rec in anchors.items() if k == key or k.startswith(key + ".")]


# Copying the corrected activation label onto the weight's contraction axis is right in
# principle -- they are the same physical dimension -- but measured it costs more than it buys:
# +2 dataflow disagreements on xLSTM and +3 on OLMo-2, because the transpose that produced the
# operand, and everything upstream of it, still carry the resolver's name. Making it pay off
# needs the weight branch to be propagated as a unit, not one axis at a time. Until then the
# within-op mismatch stands: DeepSeek-V2-Lite's q_proj reads `[T, d_model]` for its activation
# while the weight it multiplies says `n_h*d_v` on the shared axis (both 2048).
_ENABLE_REPIN = False


def _repin_weight(row, rendered: dict, width, label):
    """Give the weight's CONTRACTION axis the same label the incoming activation just got.

    build_table._contraction_pin already does this, but it runs while the row is first rendered --
    before the config tag has corrected the activation -- so the two ended up disagreeing inside a
    single op: DeepSeek-V2-Lite's q_proj read `[T, d_model]` for its input while the weight it
    multiplies said `n_h*d_v` on the axis those two share (both 2048).

    Only the contracted axis is touched, and only when its concrete width matches. That axis and
    the activation's last axis are the same physical dimension, so copying the label between them
    cannot introduce a claim that was not already made.
    """
    wp = row.get("weight_pos")
    if wp is None or wp < 0 or not label:
        return None
    conc_ins = row.get("input_shape") or []
    rend_ins = rendered.get("input_shape") or []
    if not (0 <= wp < len(conc_ins) and wp < len(rend_ins)):
        return None
    wc, wl = conc_ins[wp], rend_ins[wp]
    if not wc or not wl or len(wc) != len(wl):
        return None
    for i in (len(wc) - 1, len(wc) - 2):          # the contracted axis is one of the trailing two
        if 0 <= i < len(wc) and wc[i] == width and wl[i] != label:
            wl[i] = label
            # Marked authoritative so propagate() carries it to the `aten.t` that produced this
            # operand. Without that the transpose kept the old name and the same tensor read two
            # ways across one edge, which is precisely what the dataflow check counts.
            return ("input_shape", wp, i)
    return None


def relabel(row, rendered: dict, anchors: dict) -> int:
    """Overwrite rendered axis labels with the module-declared answer, in place.

    Anchors are strictly stronger evidence than value matching -- they come from the
    parameter the op actually consumed -- so where an anchor speaks it wins. Where it is
    silent or ambiguous the ordinary rendering stands untouched, so this can only replace a
    guess with a fact, never remove information.

    Returns (axes changed, [(field, shape_index, axis), ...]) -- the second element marks the
    positions an anchor decided, which propagate() then carries to every other op that sees
    the same tensor.
    """
    key = module_key(row.get("module_path"))
    # the anchor belongs to the PARAMETER this op consumed (see declared_dims)
    own = anchors.get(module_key((row.get("params") or [""])[0]))
    block = _block_anchors(anchors, key)
    if not own and not block:
        return 0, []
    # "the width entering this module is its in_features" holds for the op that actually applies
    # the parameter -- not for every op that happens to run inside the module. DeepSeek-V4-Pro's
    # compressor calls `new_zeros([1, 512, 8, 512])`, an allocation whose trailing 512 coincides
    # with the module's input width; pinning it relabelled 1,440 compressed-sequence axes
    # (`T/m_csa`, correct) as `d_head`. Restricting to the parameter-consuming op keeps the claim
    # true by construction.
    # Rule 2 ("the width entering this module is its in_features") presumes the module HAS one
    # incoming width. A module holding several parameters does not: gpt-oss's `mlp.experts` takes
    # the residual stream into gate_up_proj and the intermediate into down_proj, and since
    # d_model == d_moe == 2880 there, pinning both left the same 2880-wide axis reading `d_model`
    # at one op and `d_moe` at the next -- each correct locally, contradictory across the edge,
    # and it doubled the model's dataflow-disagreement count. Weight labelling (rule 1) is still
    # per-parameter and unaffected; only the activation pin needs a single-parameter module.
    applies = (own is not None and _param_module(row.get("params")) is not None
               and own.get("sole_param", True))
    changed = 0
    fixed = []          # (field, shape_index, axis) an anchor decided -> authoritative

    def _put(labels, i, label, where=None):
        nonlocal changed
        if label and 0 <= i < len(labels):
            if labels[i] != label:
                labels[i] = label
                changed += 1
            if where:
                fixed.append((where[0], where[1], i))

    # 1. the weight itself. REVIVED on a different footing than the attempt that failed earlier:
    #    that one wrote the anchor's in/out at axis positions derived from the multiplication, and
    #    those roles do not always match the module's own view (Zamba2's linear_q_adapter has them
    #    reversed), producing 396 arithmetically-false labels. Here each axis carries its own
    #    expression, taken from the module's cached widths, and every one is checked against the
    #    concrete width before it is written -- there is no axis-role inference left to be wrong.
    if _ENABLE_AXIS_LABELS and own and own.get("axes") and rendered.get("weight_shape")             and row.get("weight_shape"):
        wc, wl = list(row["weight_shape"]), rendered["weight_shape"]
        ax = own["axes"]
        if len(wc) == len(wl) == len(ax):
            for i, expr in enumerate(ax):
                if expr and _evaluates_to(expr, wc[i], _TABLE.get("t")):
                    _put(wl, i, expr, ("weight_shape", 0))
    if applies and rendered.get("weight_shape") and row.get("weight_shape"):
        w = list(row["weight_shape"])
        labels = rendered["weight_shape"]
        if len(w) == len(labels):
            ia, oa = own.get("in_axis"), own.get("out_axis")
            if ia is not None and own.get("in"):
                _put(labels, ia, own["in"])
            if oa is not None and own.get("out"):
                _put(labels, oa, own["out"])

    # 2. activations: the width entering this module is its `in`, the width leaving is its
    #    `out`. Only the LAST axis is pinned -- the leading axes are batch/sequence, which the
    #    module declares nothing about.
    for fld in ("input_shape", "output_shape"):
        conc_all, rend_all = row.get(fld) or [], rendered.get(fld) or []
        for si, (conc, labels) in enumerate(zip(conc_all, rend_all)):
            if not conc or not labels or len(conc) != len(labels):
                continue
            if applies and row.get("weight_pos") != si:
                last = conc[-1]
                if own.get("in_value") == last and own.get("in"):
                    _put(labels, len(labels) - 1, own["in"], (fld, si))
                    if _ENABLE_REPIN:
                        pinned = _repin_weight(row, rendered, last, own["in"])
                        if pinned:
                            fixed.append(pinned)
                elif own.get("out_value") == last and own.get("out"):
                    _put(labels, len(labels) - 1, own["out"], (fld, si))
            # 3. count/size split -- OFF. Kept because the reasoning is worth not re-deriving.
            #
            #    The idea: two axes whose product is exactly one module's output width are that
            #    width's [count, size] factorisation, so `k_proj`'s reshape is [n_kv, d_head]
            #    however badly d_head collides with n_h.
            #
            #    Measured across all 26 models it was NEGATIVE value. Where it is safe it only
            #    re-derives labels that were already right (the head-count/head-size class is
            #    handled at the source by symbolic_shape._HEAD_COUNT_EXCLUSIVE); where the
            #    architecture is unusual it rewrote correct labels from coincidental products --
            #    successively 30k, then 7.9k, then 4.5k axes as each guard was added, and DeepSeek
            #    -V4's compressor/indexer still lost ~1.7k after all three. "Product matches" is a
            #    coincidence detector, not evidence.
            #
            #    To turn it on it needs real provenance: follow `depends_on` to the module that
            #    PRODUCED the tensor and use only that module's split, rather than any anchor
            #    that happens to sit in the same block. Until then the ordinary rendering stands.
            if not _ENABLE_SPLIT:
                continue
            #
            #    Scoped to modules NESTED IN THE SAME BLOCK and only when exactly one of them
            #    can explain the product. An earlier version scoped it to "this module and
            #    everything under it", which at the model root (`module_path == "model"`) meant
            #    every anchor in the network: DeepSeek-V4-Pro then had 7,323 `T` axes rewritten
            #    to `n_h` on coincidental products. `.*.` requires the row to sit inside a
            #    decoder layer, which is the only place a projection's reshape can appear.
            #
            #    ONLY the trailing two axes are eligible. A reshape splits the packed FEATURE
            #    dimension, which is always last -- the leading axes are batch/heads/sequence.
            #    Scanning every adjacent pair instead rewrote correct labels wherever a
            #    (count, sequence) pair coincided with a projection width: Llama-3.1-70B's
            #    `[B, n_h, T, d_head]` has n_h*T == 64*16 == 1024 == n_kv*d_head, so `n_h` became
            #    `n_kv` and `T` became `d_head` on 7,872 axes. The product of a head count and a
            #    sequence length is never a projection width; only the tail can be.
            #
            #    Restricted further to ops that ACTUALLY reshape, and to axes that are not the
            #    sequence. Llama-3.1-70B's RoPE `slice` produces `[B, n_h, T, d_head/2]` whose
            #    trailing pair is 16*64 == 1024 == n_kv*d_head, and a pure product test rewrote
            #    `T`->`n_kv` and `d_head/2`->`d_head` on 4,488 axes. A slice does not split a
            #    packed dimension, and a feature split never consumes the sequence axis.
            if ".*." not in (key or "") or len(conc) < 2 or row.get("op_type") not in _RESHAPE_OPS:
                continue
            i = len(conc) - 2
            a, b = conc[i], conc[i + 1]
            if not isinstance(a, int) or not isinstance(b, int) or a < 2 or b < 2:
                continue
            if any(_MENTIONS_T.search(str(labels[j])) for j in (i, i + 1)):
                continue
            hits = {r["split"] for r in block
                    if r.get("split") and r.get("out_value") == a * b}
            if len(hits) != 1:
                continue                  # ambiguous or unknown -> leave the normal rendering
            cnt, size = hits.pop()
            _put(labels, i, cnt, (fld, si))
            _put(labels, i + 1, size, (fld, si))
    return changed, fixed


def propagate(rows, ordered, authoritative: dict, passes: int = 4) -> int:
    """Spread anchor-decided labels along the dataflow, overwriting named labels.

    relabel() fixes the tensor where the module declares it, but the SAME tensor is also
    rendered on the producing op's output and on every consumer's input, and those were left
    reading the old name. That is not cosmetic: the gate's dataflow check counts one tensor
    with two names, and correcting only one side turned Zamba2 from 0 to 36 such mismatches.

    Distinct from build_table._propagate_labels, which is deliberately monotone (bare integers
    only) so it can never overwrite a considered choice. Here the overwrite is the point --
    an anchor IS the considered choice, and it outranks whatever value matching produced. The
    label only travels between shapes that are concretely IDENTICAL, so it stays a statement
    about one tensor.
    """
    by_id = {r.get("op_id"): (r, o) for r, o in zip(rows, ordered)}
    total = 0
    for _p in range(passes):
        moved = 0
        for row, out in zip(rows, ordered):
            oid = row.get("op_id")
            for dep in (row.get("depends_on") or []):
                pair = by_id.get(dep)
                if not pair:
                    continue
                p_row, p_out = pair
                for bi, (bc, bs) in enumerate(zip(row.get("input_shape") or [],
                                                  out.get("input_shape") or [])):
                    if not isinstance(bc, list) or not bc:
                        continue
                    for ai, (ac, as_) in enumerate(zip(p_row.get("output_shape") or [],
                                                       p_out.get("output_shape") or [])):
                        if not isinstance(ac, list) or list(ac) != list(bc):
                            continue
                        for axis in range(min(len(bs), len(as_))):
                            mine = ("input_shape", bi, axis) in authoritative.get(oid, set())
                            theirs = ("output_shape", ai, axis) in authoritative.get(dep, set())
                            if mine and not theirs and as_[axis] != bs[axis]:
                                as_[axis] = bs[axis]
                                authoritative.setdefault(dep, set()).add(
                                    ("output_shape", ai, axis))
                                moved += 1
                            elif theirs and not mine and bs[axis] != as_[axis]:
                                bs[axis] = as_[axis]
                                authoritative.setdefault(oid, set()).add(
                                    ("input_shape", bi, axis))
                                moved += 1
        total += moved
        if not moved:
            break
    return total


# An unnamed dimension leaking into a tag expression means rules/symbols.yaml has no name for
# that width, so the tag is WORSE than the resolver's rendering there: DeepSeek-V3's q_b_proj
# comes out `192*n_h` (192 is the unregistered qk_head_dim) and Zamba2's mamba in_proj carries a
# bare 8192. Small coefficients are fine and common (`2*d_moe`, `2*n_h*d_head`); what disqualifies
# a tag is a literal standing in for a DIMENSION -- large, or added as its own term.
_BIG_LITERAL = re.compile(r"(?<![\w.])(?:1[6-9]|[2-9]\d|\d{3,})(?![\w.])")
# A literal added as its OWN term (`...+8192`, `(n_h+2)`) is an unnamed dimension. A literal that
# is followed by `*` is a coefficient of the next term (`...+2*n_h_lin_v*d_head_lin_v`) and is
# perfectly normal, so it must not disqualify the expression.
_ADDED_LITERAL = re.compile(r"[+\-]\s*\d+(?!\s*[*\w.])|(?<![\w.*])\d+(?=\s*[+\-])")


def _evaluates_to(expr: str | None, value, table: dict | None) -> bool:
    """Whether `expr` really equals `value` under this model's symbol table.

    The tag describes a module's in_features/out_features, while the anchor decides WHICH weight
    axis is which from the multiplication. Those two views normally agree, but not always --
    Zamba2's `linear_q_adapter` had them reversed, so writing the tag at the anchor's axis put
    `r_lora` (128) on the 4096 axis, and DeepSeek-V4's compressor put `2*d_head` (1024) on a
    512-wide one. Both were caught by the arithmetic gate AFTER publishing; checking here means a
    tag is adopted only when it is numerically true at the axis it lands on.

    A tag that cannot be evaluated (an unregistered symbol) is rejected, not assumed.
    """
    if not expr or value is None or not table:
        return False
    try:
        return int(eval(str(expr), {"__builtins__": {}}, dict(table))) == int(value)
    except Exception:
        return False


def tag_is_usable(expr: str | None) -> bool:
    """Whether a config-tag expression is a better label than value matching.

    Rejecting on an unnamed literal is what keeps this honest: the tag records exactly what the
    model code computed, so a number left in it is a rules gap at that position, not a naming
    choice. Better to keep the existing label than to publish a half-named formula.
    """
    if not expr:
        return False
    return not (_BIG_LITERAL.search(expr) or _ADDED_LITERAL.search(expr))


# symbol table of the model currently being rendered; set by build_anchors so relabel() can
# verify a per-axis expression without threading it through every call site
_TABLE: dict = {}

# Writing weight-axis labels from symbolic_dims.param_axis_expressions -- OFF, measured 2026-08-05.
#
# The idea is sound and the data is genuinely useful (it explains 74% of parameter axes, including
# GPT-2's Conv1D and every MoE expert's 3-D Parameter, which nothing else reaches). But matching an
# axis against the widths a module happens to cache says only that the NUMBERS agree, not that the
# meanings do, and the uniqueness test only rejects two competing explanations -- never a single
# wrong one. Measured over all 26 models it changed 3,365 axes with derived formulas in the
# candidate set (`d_moe -> E*k` x696, `d_moe -> E` x384: an expert's width is not the expert
# count) and still 830 with module attributes alone, including a regression where the complete
# `d_inner+2*n_g*d_state` was replaced by the half-tagged `(4096+2*d_state)`.
#
# So param_axes stays PUBLISHED (it is good evidence for a reader and for Tier 2 research) but
# does not drive labelling. Turning this on needs a check that the axis MEANS the thing, not just
# that it measures the same -- the dataflow role of the axis, not its size.
_ENABLE_AXIS_LABELS = False

_PRODUCT = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\*([A-Za-z_][A-Za-z0-9_]*)$")


def _factor_split(label):
    """('n_h', 'd_head') for 'n_h*d_head'. None when the label is not a plain two-symbol
    product -- numeric multiples like `2*d_moe` are a gate+up CONCATENATION, not a
    count/size reshape, so splitting them would be false."""
    if not isinstance(label, str):
        return None
    m = _PRODUCT.match(label)
    if not m:
        return None
    return (m.group(1), m.group(2))
