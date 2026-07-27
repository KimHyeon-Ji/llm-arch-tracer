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

import major_ops

FULL_SUBDIR = "full"

# canonical column order: "op_id" + <hierarchy h1..hN> + _AFTER_HIER + _TAIL
_AFTER_HIER = ["op_type", "input_shape", "weight_shape", "output_shape", "depends_on",
               "layer_idx", "block", "sub_block", "depth"]
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


def _ordered_row(row: dict, resolver, hier_cols: list) -> dict:
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
    out["input_shape"] = [resolver(s, mp) for s in (row.get("input_shape") or [])]
    out["weight_shape"] = resolver(row.get("weight_shape"), mp)
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


def _write_concrete(model_dir: str, phase: str, rows: list[dict]):
    """Persist the pre-symbolization dims so the trace can be re-rendered without re-tracing.
    Only op_id + shapes: everything else in the row is unaffected by symbolization."""
    with open(concrete_path(model_dir, phase), "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps({
                "op_id": row.get("op_id"),
                "input_shape": row.get("input_shape") or [],
                "weight_shape": row.get("weight_shape"),
                "output_shape": row.get("output_shape") or [],
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


def write_outputs(model_dir: str, phase: str, rows: list[dict], resolver):
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
    ordered = [_ordered_row(row, resolver, hier_cols) for row in rows]  # symbolic, ordered

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
