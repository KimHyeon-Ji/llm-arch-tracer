"""Read-only arithmetic/export audit of the four round-2 published models."""
import csv
import itertools
import json
import re
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import dim_expr as DE

MODELS = ["meta-llama__Llama-4-Maverick-17B-128E", "openai__gpt-oss-20b",
          "openai__gpt-oss-120b", "deepseek-ai__DeepSeek-V4-Pro"]


def rows(path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def claims_for(model, phase):
    selected_model = selected_phase = None
    found = []
    for line in (ROOT / "develop/codex_ask_four_models_claims.md").read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            selected_model = line[3:].replace("/", "__")
        if line.startswith("### "):
            selected_phase = line[4:].split()[0]
        if selected_model != model or selected_phase != phase:
            continue
        match = re.match(r"^\s*(\d+)\s+(\S+)\s+(\S+)\s+(\[\[.*\]\]) -> (\[\[.*\]\])(?:  w=(\[.*\]))?$", line)
        if match:
            oid, op, module, ins, outs, weight = match.groups()
            found.append({"op_id": int(oid), "op_type": op, "input_shape": ins,
                          "output_shape": outs, "weight_shape": weight})
    return found


def audit(model):
    folder = ROOT / "models" / model
    prov = DE.load_provenance(str(folder))
    ns = DE.namespace(prov, prov["capture_batch"])
    out = {"revision": prov["revision_resolved"], "B": ns["B"], "T": ns["T"], "phases": {}}
    if model == "deepseek-ai__DeepSeek-V4-Pro":
        from symbolic_shape import build_resolver
        resolver = build_resolver(SimpleNamespace(**prov["config"]), ns["T"], batch=ns["B"])
        probes = [([3], "model.layers.0.attn_hc", {"is_weight": True}, ["3"]),
                  ([1344], "model.layers.0.self_attn", {}, ["B*(d_head-d_rope)"]),
                  ([2176, 3, 512], "model.layers.0.self_attn", {}, ["T", "B", "d_head"])]
        out["resolver_counterexamples"] = [{"concrete": shape, "module": module, "options": options,
                                              "actual": resolver(shape, module, **options), "expected": expected}
                                             for shape, module, options, expected in probes]
    for phase in ("prefill", "decode"):
        summary = list(rows(folder / f"{phase}.jsonl"))
        claims = claims_for(model, phase)
        claim_errors = []
        if len(claims) != len(summary):
            claim_errors.append(["row_count", len(claims), len(summary)])
        by_id = {r["op_id"]: r for r in summary}
        for claim in claims:
            actual = by_id[claim["op_id"]]
            for field in ("op_type", "input_shape", "output_shape", "weight_shape"):
                expected = actual.get(field)
                if field != "op_type" and expected is not None:
                    expected = json.dumps(expected, ensure_ascii=False).replace('"', '')
                if claim[field] != expected:
                    claim_errors.append([claim["op_id"], field, claim[field], expected])
        with (folder / f"{phase}.csv").open(encoding="utf-8", newline="") as f:
            csv_rows = list(csv.DictReader(f))
        assert len(summary) == len(csv_rows)
        export_errors = []
        for j, c in zip(summary, csv_rows):
            for field in ("input_shape", "output_shape", "weight_shape", "params", "depends_on"):
                # CSV removes quotes around string leaves, but preserves order and expressions.
                expected = "" if j.get(field) is None else json.dumps(j[field], ensure_ascii=False).replace('"', '')
                if c[field] != expected:
                    export_errors.append([j["op_id"], field, c[field], expected])
            for field in ("op_id", "block_type", "repeat", "layers", "module_path", "op_type", "raw_op"):
                expected = "" if j.get(field) is None else str(j[field])
                if c[field] != expected:
                    export_errors.append([j["op_id"], field, c[field], expected])
        templates = Counter((r.get("block_type"), r.get("repeat"), r.get("layers")) for r in summary)
        axes, bad, unknown, weight_runtime, scale_rows, arity_errors = 0, [], [], [], [], []
        op_count = 0
        rawfile = folder / "full" / f"{phase}.trace.raw.jsonl"
        concretefile = folder / "full" / f"{phase}.shapes.concrete.jsonl"
        for r, c in itertools.zip_longest(rows(rawfile), rows(concretefile)):
            assert r is not None and c is not None
            assert r["op_id"] == c["op_id"]
            op_count += 1
            if any(p.endswith(".scale") for p in r.get("params", [])):
                if len(scale_rows) < 5:
                    scale_rows.append(r)
            for field in ("input_shape", "output_shape", "weight_shape"):
                s, t = r.get(field), c.get(field)
                if s is None or t is None:
                    if s != t:
                        arity_errors.append([r["op_id"], field, s, t])
                    continue
                if field == "weight_shape":
                    s, t = [s], [t]
                if len(s) != len(t):
                    arity_errors.append([r["op_id"], field, s, t])
                for si, (sh, actual) in enumerate(zip(s, t)):
                    if len(sh) != len(actual):
                        arity_errors.append([r["op_id"], field, si, sh, actual])
                    for axis, (label, value) in enumerate(zip(sh, actual)):
                        anchor = [r["op_id"], field, si, axis, label, value, r["module_path"]]
                        evaluated = DE.evaluate(label, ns)
                        if evaluated is None:
                            unknown.append(anchor)
                        else:
                            axes += 1
                            if evaluated != value:
                                bad.append(anchor + [evaluated])
                        if field == "weight_shape" and re.search(r"\b(?:B|T)\b", str(label)):
                            weight_runtime.append(anchor)
        published = {r["op_id"]: r for r in rows(rawfile)}
        ledger_pairs = Counter()
        ledger_examples = []
        ledger_sites = 0
        for site in rows(folder / "full" / f"{phase}.axis_resolution.jsonl"):
            if site.get("kind") != "site":
                continue
            ledger_sites += 1
            field = {"i": "input_shape", "o": "output_shape", "w": "weight_shape"}[site["field"]]
            rec = published[site["op_id"]]
            sh = rec[field] if field == "weight_shape" else rec[field][site["shape_index"]]
            current = str(sh[site["axis"]])
            if current != str(site["label"]):
                key = (site["grade"], str(site["label"]), current)
                ledger_pairs[key] += 1
                if len(ledger_examples) < 12:
                    ledger_examples.append({**site, "published_label": current, "module_path": rec["module_path"]})
        out["phases"][phase] = {"summary_rows": len(summary), "raw_ops": op_count, "checked_axes": axes,
                                 "unknown": unknown, "numeric_errors": bad, "arity_errors": arity_errors,
                                 "weight_runtime": weight_runtime, "export_errors": export_errors,
                                 "claims_rows": len(claims), "claims_errors": claim_errors,
                                 "templates": [[*k, v] for k, v in templates.items()], "scale_rows": scale_rows,
                                 "ledger_sites": ledger_sites, "ledger_label_mismatches": sum(ledger_pairs.values()),
                                 "ledger_mismatch_pairs": [[*k, v] for k, v in ledger_pairs.items()],
                                 "ledger_mismatch_examples": ledger_examples,
                                 "add_rows": [r for r in summary if r["op_type"] == "add_"]}
    return out


if __name__ == "__main__":
    output = {model: audit(model) for model in MODELS}
    destination = ROOT / "develop/four_models_round2_evidence.json"
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for model, result in output.items():
        for phase, p in result["phases"].items():
            print(model, phase, json.dumps({k: len(v) if isinstance(v, list) else v for k, v in p.items()
                                           if k not in ("scale_rows", "add_rows", "templates")}))
