"""Step 8 -- assemble traced rows into the deliverable tables.
See 01-main.md section 6 (table schema) and section 2 (deliverables). depends_on is a column
in every file, so no separate dependency-graph file is emitted.

Two views are written per phase (01-main.md section 6.1):
  <model_dir>/<phase>.csv, <phase>.jsonl          -- MAJOR operators only (latency-relevant)
  <model_dir>/full/<phase>.csv, <phase>.trace.raw.jsonl  -- the complete trace
The major view is derived from the full one by major_ops.extract_major (filter + norm rollup +
dependency-graph contraction). Everything else the run produces (provenance.json, report.md)
also lives under full/; only the major tables, structure.yaml and model_summary.md sit at the
top of the model folder -- see run.py.

Shapes are written SYMBOLICALLY (B, T, d_model, E, k, ...) via a resolver, never as raw
numbers -- the concrete seq_len/config live in provenance.json for recovery (section 10).

The concrete dims are ALSO kept verbatim in a sidecar `full/<phase>.shapes.concrete.jsonl`
(op_id + the three shape fields only). Rationale: symbolization is lossy, so before the sidecar
existed a fix to the symbolizer could not be applied to already-published models -- the only
way to re-render was a full re-trace of every model (see the "2*T" fabrication fixed 2026-07-27,
which stranded 9 models). With the sidecar, `develop/regen_summaries.py` re-symbolizes offline.
The sidecar is an input to regeneration, not a deliverable to read directly.

Column order (all files share it): op_id, then the module-hierarchy columns (h1, h2, ...),
then op_type and the rest. Hierarchy is up front so each row reads "structure first" -- op_id
+ where in the module tree -- before the op detail. The in-memory rows keep concrete shapes
(validate.py needs numbers); symbolization happens only on the written copies (the resolver is
idempotent on already-symbolic shapes, so regenerating from an existing jsonl is safe).
"""
import csv
import json
import os

import anchors as anchors_mod
import major_ops

FULL_SUBDIR = "full"

# canonical column order: "op_id" + <hierarchy h1..hN> + _AFTER_HIER + _TAIL
_AFTER_HIER = ["op_type", "input_shape", "weight_shape", "weight_pos", "output_shape",
               "depends_on", "layer_idx", "block", "sub_block", "depth"]
_TAIL = ["module_path", "raw_op", "params", "phase", "unmapped"]
_JSON_FIELDS = ("input_shape", "output_shape", "weight_shape", "depends_on", "params")


def _levels_of(row: dict) -> list:
    """Module-hierarchy chain for a row: from `levels` (fresh trace) or, when regenerating from
    an already-written jsonl, reconstructed from its h1..hN columns."""
    if row.get("levels"):
        return list(row["levels"])
    hs, i = [], 1
    while f"h{i}" in row:
        hs.append(row[f"h{i}"])
        i += 1
    while hs and hs[-1] in ("", None):
        hs.pop()
    return hs


_CONTRACTING_OPS = {"linear", "matmul", "batched_matmul"}


def _contraction_pin(row: dict, in_concrete: list, in_labels: list):
    """(axis_index, label) forcing a weight's contraction axis to match the input activation.

    For a linear/matmul, the weight's in_features axis and the input's last axis are the SAME
    physical dimension, so they must carry the same label. Resolving them independently let them
    disagree whenever a model made the two numerically equal: Zamba2's `q_proj` is
    nn.Linear(attention_hidden_size=4096, n_h*head_dim=4096), and the weight came out
    `[d_attn, n_h*d_head]` -- the reverse of the real [out, in] order, while the input activation
    (correctly) said `d_attn`. Found by an audit that cross-checked the two, 2026-07-30.

    Only fires when the CONCRETE values match unambiguously, so it cannot invent an agreement
    that is not physically there. Returns None on the regen-without-sidecar path, where shapes
    are already symbolic strings rather than ints.
    """
    if row.get("op_type") not in _CONTRACTING_OPS:
        return None
    w = row.get("weight_shape")
    if not w or len(w) < 2 or not in_concrete or not in_concrete[0] or not in_labels:
        return None
    act_last = in_concrete[0][-1]
    if not isinstance(act_last, int):
        return None
    # weight is [out, in] normally, [in, out] once transposed -- pick whichever axis actually
    # equals the activation's contracted dim.
    for idx in (len(w) - 1, len(w) - 2):
        if isinstance(w[idx], int) and w[idx] == act_last:
            return (idx, in_labels[0][-1])
    return None


def weight_pos_candidates(weight_shape, input_shapes) -> list:
    """Indices of the operands whose shape IS weight_shape -- as stored, or with the last two axes
    swapped (nn.Linear keeps [out, in] and hands `aten.mm` the transpose; a batched expert weight
    is transposed the same way on its trailing pair)."""
    if not weight_shape:
        return []
    w = list(weight_shape)
    swapped = w[:-2] + [w[-1], w[-2]] if len(w) >= 2 else w
    return [i for i, o in enumerate(input_shapes or [])
            if isinstance(o, list) and list(o) in (w, swapped)]


def derive_weight_pos(weight_shape, input_shapes, op_type=None) -> int | None:
    """Index into input_shape of the operand that IS weight_shape; -1 if no operand carries that
    shape; None when the op has no weight.

    Fallback for rows the tracer did not record it on -- every model published before 2026-08-04
    regenerates through here. Shape alone cannot always separate the weight from the activation, so
    when several operands match, the op's calling convention breaks the tie: a contracting op is
    `mm(activation, weight)` / `addmm(bias, activation, weight)` / `bmm(activation, weight)`, i.e.
    the weight is the RIGHTMOST match, while a gather or a norm (`embedding(weight, idx)`,
    `native_layer_norm(x, weight, bias)`) puts it leftmost. Qwen3-Next's `shared_expert_gate` is
    why this matters: out_features=1 makes its weight [B, d_model] -- byte-identical to the
    activation it multiplies -- and a left-to-right scan confidently returned the activation."""
    cands = weight_pos_candidates(weight_shape, input_shapes)
    if not weight_shape:
        return None
    if not cands:
        return -1
    return cands[-1] if op_type in _CONTRACTING_OPS else cands[0]


def _propagate_labels(rows: list[dict], ordered: list[dict]) -> None:
    """Carry a resolved axis label along the dataflow, in place.

    A tensor that flows from op A's output into op B's input is ONE tensor, so it must read the
    same in both places. Each op is otherwise labelled in isolation, which lets one side name an
    axis while the other leaves a bare integer: Nemotron-3-Nano names every block `mixer`, so the
    `n_kv*d_head` rule (scoped to attention-ish module names) fired inside `mixer.k_proj` but not
    in the parent `mixer`, and the same 1024-wide tensor came out `n_kv*d_head` then `1024`.

    Deliberately MONOTONE: only a bare integer is ever replaced, and only by the producer's label
    for the identical concrete tensor. It can add information but never overwrite a considered
    choice, so it cannot introduce the kind of regression a re-prioritisation can. Genuine
    two-concepts-one-value ambiguities (gpt-oss d_model/d_ff/d_moe all 2880) are left alone --
    those are documented, not guessable. Found by the dataflow audit, 2026-07-30."""
    by_id = {r.get("op_id"): (r, o) for r, o in zip(rows, ordered)}
    for _pass in range(3):        # a fill-in can enable the next one; converges quickly
        changed = False
        for row, out in zip(rows, ordered):
            for dep in (row.get("depends_on") or []):
                prod = by_id.get(dep)
                if not prod:
                    continue
                p_row, p_out = prod
                for bi_c, bi_s in zip(row.get("input_shape") or [],
                                      out.get("input_shape") or []):
                    if not isinstance(bi_c, list) or not bi_c:
                        continue
                    for ao_c, ao_s in zip(p_row.get("output_shape") or [],
                                          p_out.get("output_shape") or []):
                        if not isinstance(ao_c, list) or ao_c != bi_c:
                            continue      # not the same tensor
                        # Bidirectional: whichever side is still a bare integer takes the other
                        # side's name. Either end can be the unresolved one -- a scope can match
                        # the child module but not the parent, or the reverse.
                        for i, (mine, theirs) in enumerate(zip(bi_s, ao_s)):
                            if str(mine).isdigit() and not str(theirs).isdigit():
                                bi_s[i] = theirs
                                changed = True
                            elif str(theirs).isdigit() and not str(mine).isdigit():
                                ao_s[i] = mine
                                changed = True
        if not changed:
            break


def _canonical_weight_labels(rows: list[dict], resolver) -> dict:
    """{param_name: (concrete_shape, labels)} taken from the op that CONTRACTS with that weight.

    A parameter has exactly one shape, so it must get exactly one labelling everywhere it appears.
    Only the contracting op (linear/matmul) can pin the in_features axis against the activation
    (see _contraction_pin), so its rendering is the authoritative one; a bare `t` on the same
    parameter has no activation to check against and would otherwise order the axes arbitrarily.
    Zamba2 showed both spellings of the same q/k/v_proj weight in one trace -- `[n_h*d_head,
    d_attn]` on the matmul and the reverse on the transpose (audit, 2026-07-30)."""
    canon = {}
    for row in rows:
        if row.get("op_type") not in _CONTRACTING_OPS:
            continue
        params, w = row.get("params") or [], row.get("weight_shape")
        if len(params) != 1 or not w or params[0] in canon:
            continue
        mp = row.get("module_path")
        in_shapes = row.get("input_shape") or []
        pin = _contraction_pin(row, in_shapes, [resolver(s, mp) for s in in_shapes])
        if pin:
            canon[params[0]] = (tuple(w), resolver(w, mp, is_weight=True, pin=pin))
    return canon


def _ordered_row(row: dict, resolver, hier_cols: list, canon: dict | None = None) -> dict:
    """Row as an ordered dict in the canonical column order. Shapes rendered symbolically via
    resolver (idempotent on symbolic shapes). Does not mutate the input row."""
    levels = _levels_of(row)
    out = {"op_id": row.get("op_id")}
    for i, col in enumerate(hier_cols):
        out[col] = levels[i] if i < len(levels) else ""
    out["op_type"] = row.get("op_type")
    # module_path lets the resolver disambiguate symbols that share a value (see build_resolver):
    # 128 is d_head inside self_attn but E inside the MoE block.
    mp = row.get("module_path")
    in_shapes = row.get("input_shape") or []
    out["input_shape"] = [resolver(s, mp) for s in in_shapes]
    # is_weight=True enforces "a static parameter cannot depend on runtime seq len" -- see
    # build_resolver.dim(). Without it, a weight axis whose size coincides with T (or a T
    # product) rendered as a sequence-dependent symbol, which is physically impossible.
    w = row.get("weight_shape")
    params = row.get("params") or []
    hit = canon.get(params[0]) if (canon and len(params) == 1) else None
    if hit and w and tuple(w) == hit[0]:
        out["weight_shape"] = list(hit[1])   # one parameter -> one labelling, everywhere
    else:
        out["weight_shape"] = resolver(w, mp, is_weight=True,
                                       pin=_contraction_pin(row, in_shapes, out["input_shape"]))
    # Position is a property of the operand list, so it is derived from the SYMBOLIC shapes that
    # were just written -- the two columns are then consistent by construction even where
    # symbolization collapsed two distinct concrete dims onto one name.
    wp = row.get("weight_pos")
    out["weight_pos"] = derive_weight_pos(out["weight_shape"], out["input_shape"],
                                          out["op_type"]) if wp is None else wp
    out["output_shape"] = [resolver(s, mp) for s in (row.get("output_shape") or [])]
    out["depends_on"] = row.get("depends_on", [])
    out["layer_idx"] = row.get("layer_idx")
    out["block"] = row.get("block")
    out["sub_block"] = row.get("sub_block")
    out["depth"] = row.get("depth")
    out["module_path"] = row.get("module_path")
    out["raw_op"] = row.get("raw_op")
    out["params"] = row.get("params", [])
    out["phase"] = row.get("phase")
    out["unmapped"] = row.get("unmapped")
    return out


def _columns_for(ordered_rows: list[dict]) -> tuple[list, list]:
    """(hier_cols, full column order) sized to the deepest module hierarchy present."""
    max_depth = max((len(_levels_of(r)) for r in ordered_rows), default=0)
    hier_cols = [f"h{i}" for i in range(1, max_depth + 1)]
    return hier_cols, ["op_id"] + hier_cols + _AFTER_HIER + _TAIL


def _plain(v):
    """Human-readable rendering of a nested shape/list for the CSV: bare symbols, no quotes.
    [["V","d_model"],["B","T"]] -> [[V, d_model], [B, T]]; None -> "". The dim symbols never
    contain commas, so this stays unambiguous. The CSV is the eyeball format; the .jsonl keeps
    proper JSON for programmatic parsing."""
    if v is None:
        return ""
    if isinstance(v, list):
        return "[" + ", ".join(_plain(x) for x in v) + "]"
    return str(v)


def _emit(csv_path: str, jsonl_path: str, ordered_rows: list[dict], columns: list):
    """Write already-ordered, already-symbolic rows to CSV + JSONL, projected to `columns`.
    CSV renders shape/list fields as bare, quote-free text (readability); JSONL stays JSON."""
    proj = [{c: r.get(c) for c in columns} for r in ordered_rows]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for out in proj:
            row = dict(out)
            for k in _JSON_FIELDS:
                row[k] = _plain(row.get(k))
            writer.writerow(row)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for out in proj:
            f.write(json.dumps(out, default=str) + "\n")


CONCRETE_SUFFIX = "shapes.concrete.jsonl"


def concrete_path(model_dir: str, phase: str) -> str:
    return os.path.join(model_dir, FULL_SUBDIR, f"{phase}.{CONCRETE_SUFFIX}")


def _dims_are_concrete(rows: list[dict]) -> bool:
    """True only if these rows still carry integer dims, i.e. they came from a live trace rather
    than from a re-read of an already-symbolized jsonl."""
    seen_int = False
    for row in rows:
        shapes = list(row.get("input_shape") or []) + list(row.get("output_shape") or [])
        w = row.get("weight_shape")
        if w:
            shapes.append(w)
        for shp in shapes:
            for d in (shp or []):
                if isinstance(d, str):
                    return False
                if isinstance(d, int):
                    seen_int = True
    return seen_int


def _write_concrete(model_dir: str, phase: str, rows: list[dict]):
    """Persist the pre-symbolization dims so the trace can be re-rendered without re-tracing.
    Only op_id + shapes: everything else in the row is unaffected by symbolization.

    NO-OP when `rows` are already symbolic. write_outputs is called both from a live trace
    (concrete) and from regeneration (symbolic, read back from full/<phase>.trace.raw.jsonl), and
    the regen path was overwriting the sidecar with the very strings the sidecar exists to let us
    recompute -- destroying the only copy of the concrete dims and forcing a full re-trace to get
    them back. Hit for real on 2026-08-04 by develop/regen_tables.py; recovered from git. The
    sidecar is append-only in spirit: a trace may create it, nothing else may degrade it."""
    if not _dims_are_concrete(rows):
        return
    with open(concrete_path(model_dir, phase), "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps({
                "op_id": row.get("op_id"),
                "input_shape": row.get("input_shape") or [],
                "weight_shape": row.get("weight_shape"),
                "output_shape": row.get("output_shape") or [],
                # trace-time fact, not a rendering: the tracer resolves it against tensor identity,
                # which no later pass can reconstruct. Kept here so regeneration restores it
                # instead of falling back to the shape-only heuristic (see derive_weight_pos).
                #
                # Derived when the row has none. Writing a bare None here made the value STICK at
                # None: regeneration strips a row's weight_pos when the sidecar has none, so the
                # first regen after this field was added wrote None for every published model and
                # every later regen re-read it. Anything downstream that needs the operand index
                # (anchors._repin_weight) then silently did nothing.
                "weight_pos": row.get("weight_pos") if row.get("weight_pos") is not None
                else derive_weight_pos(row.get("weight_shape"), row.get("input_shape"),
                                       row.get("op_type")),
            }, default=str) + "\n")


def load_concrete(model_dir: str, phase: str) -> dict:
    """{op_id: {shape fields}} from the sidecar, or {} when the model predates it."""
    path = concrete_path(model_dir, phase)
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            out[rec["op_id"]] = rec
    return out


def write_outputs(model_dir: str, phase: str, rows: list[dict], resolver, tags: dict | None = None,
                  param_axes: dict | None = None):
    """Write both the full trace (under full/) and the derived major-operator view (top level).
    No separate .graph.json: the dependency graph is recoverable from the depends_on column."""
    os.makedirs(model_dir, exist_ok=True)
    for row in rows:
        row.setdefault("phase", phase)

    hier_cols, columns = _columns_for(rows)  # _levels_of reads `levels` or reconstructs from h*
    full_dir = os.path.join(model_dir, FULL_SUBDIR)
    os.makedirs(full_dir, exist_ok=True)
    # sidecar first, from the still-concrete rows (resolver has not touched them yet)
    _write_concrete(model_dir, phase, rows)
    canon = _canonical_weight_labels(rows, resolver)
    ordered = [_ordered_row(row, resolver, hier_cols, canon) for row in rows]  # symbolic, ordered

    # Module-declared dimensions override value matching wherever they speak (see anchors.py).
    # Requires concrete rows: on the regen-from-symbolic path (develop/regen_tables.py) the ints
    # are already gone, so anchoring is skipped there and the stored labels pass through.
    if _dims_are_concrete(rows):
        conc = {r.get("op_id"): r for r in rows}
        anch = anchors_mod.build_anchors(rows, conc, resolver, canon, tags, param_axes)
        authoritative = {}
        for row, out in zip(rows, ordered):
            # the anchor pass reads weight_pos off the CONCRETE row; _ordered_row has just
            # resolved it, so backfill rather than leave the row's copy missing
            if row.get("weight_pos") is None:
                row["weight_pos"] = out.get("weight_pos")
            _n, fixed = anchors_mod.relabel(row, out, anch)
            if fixed:
                authoritative[row.get("op_id")] = set(fixed)
        # an anchor names a TENSOR, so every op that sees that tensor must read the same name
        anchors_mod.propagate(rows, ordered, authoritative)

    _propagate_labels(rows, ordered)

    _emit(os.path.join(full_dir, f"{phase}.csv"),
          os.path.join(full_dir, f"{phase}.trace.raw.jsonl"), ordered, columns)

    # group repeated blocks by FULL per-layer structure (from `ordered`, the complete trace), so
    # layers differing only in major-dropped ops (e.g. NoPE vs RoPE) still count as distinct blocks
    layer_sigs = major_ops.full_layer_signatures(ordered)
    major = major_ops.collapse_repeats(major_ops.extract_major(ordered), layer_sigs=layer_sigs)
    _, base_columns = _columns_for(major)
    # block_type (attn+FFN / MLA+MoE / SSM ...), repeat (how many layers the block stands for),
    # and layers (which indices) go right after op_id so the block structure reads up front.
    major_columns = base_columns[:1] + ["block_type", "repeat", "layers"] + base_columns[1:]
    csv_path = os.path.join(model_dir, f"{phase}.csv")
    _emit(csv_path, os.path.join(model_dir, f"{phase}.jsonl"), major, major_columns)
    return csv_path
