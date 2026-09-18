"""Small CPU lowering reproduction and saved layer-0 label audit; no model fetch."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "develop")]
import dim_expr as DE
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from transition_diff import classify_unmatched


class Capture(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.bmm = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if str(func) == "aten.bmm.default":
            self.bmm.append([list(args[0].shape), list(args[1].shape), list(out.shape)])
        return out


def lowering():
    torch.set_num_threads(2)
    gen = torch.Generator().manual_seed(137)
    x = torch.randn(4, 96, 64, 128, generator=gen, dtype=torch.float64)
    y = torch.randn(4, 96, 128, generator=gen, dtype=torch.float64)
    result = {}
    for b in (1, 2, 3, 4):
        with Capture() as cap:
            z = torch.einsum("... c d, ... d -> ... c", x[:b], y[:b])
        reference = (x[:b] * y[:b].unsqueeze(-2)).sum(-1)
        singles = torch.cat([torch.einsum("... c d, ... d -> ... c", x[i:i+1], y[i:i+1]) for i in range(b)])
        torch.testing.assert_close(z, reference, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(z, singles, rtol=1e-12, atol=1e-12)
        result[str(b)] = {"bmm": cap.bmm, "numeric_check": "PASS"}
    return result


def unsafe_shape_checker():
    old = [{"raw_op": "aten.alias.default", "input_shape": [[4]], "output_shape": [[4]]}]
    new = [
        {"raw_op": "aten.slice.Tensor", "input_shape": [[4]], "output_shape": [[2]]},
        {"raw_op": "aten.cat.default", "input_shape": [[2], [2]], "output_shape": [[4]]},
    ]
    verdict, reason = classify_unmatched(old, new, {}, {})
    x = torch.arange(4)
    duplicate = torch.cat([x[:2], x[:2]])
    assert not torch.equal(x, duplicate)
    return {"classifier": verdict, "reason": reason, "old_values": x.tolist(), "new_values": duplicate.tolist()}


def rows_until(path, limit):
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["op_id"] > limit:
                break
            yield row


def saved_audit():
    full = ROOT / "develop/out/moonshotai__Kimi-K3/full"
    prov = DE.load_provenance(str(full.parent))
    ns = DE.namespace(prov, 3)
    # Scoped source identity; do not register a global value-matched dimension.
    ns["n_chunk"] = ns["T"] // ns["d_chunk"]
    result = {}
    for phase in ("prefill", "decode"):
        symbolic = {r["op_id"]: r for r in rows_until(full / f"{phase}.trace.raw.jsonl", 19000)
                    if (r.get("module_path") or "").startswith("model.layers.0.self_attn")}
        if not symbolic:
            continue
        concrete = {r["op_id"]: r for r in rows_until(full / f"{phase}.shapes.concrete.jsonl", max(symbolic))}
        checked, unknown, errors, suspicious = 0, 0, [], []
        for oid, r in symbolic.items():
            c = concrete[oid]
            for field in ("input_shape", "output_shape", "weight_shape"):
                ss, cs = r.get(field), c.get(field)
                if ss is None or cs is None:
                    continue
                if field == "weight_shape":
                    ss, cs = [ss], [cs]
                assert len(ss) == len(cs), (oid, field)
                for si, (s, t) in enumerate(zip(ss, cs)):
                    assert len(s) == len(t), (oid, field, si)
                    for axis, (label, actual) in enumerate(zip(s, t)):
                        value = DE.evaluate(label, ns)
                        anchor = {"op_id": oid, "module": r["module_path"], "raw_op": r["raw_op"],
                                  "field": field, "shape_index": si, "axis": axis,
                                  "label": label, "evaluated": value, "concrete": actual}
                        if value is None:
                            unknown += 1
                        else:
                            checked += 1
                            if value != actual:
                                errors.append(anchor)
                        if label == "n_h_kda/2":
                            suspicious.append(anchor)
        anchors = {r["op_id"] for r in errors + suspicious}
        ports = {r["op_id"]: r.get("scalar_args") for r in rows_until(full / f"{phase}.ports.jsonl", max(symbolic))
                 if r["op_id"] in anchors}
        result[phase] = {"scope": "layer 0 self_attn and descendants", "ops": len(symbolic),
                         "checked_axes": checked, "unevaluable_axes": unknown,
                         "numeric_errors": errors, "head_half_labels_requiring_source_review": suspicious,
                         "scalar_args": ports}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = {"torch": torch.__version__, "torch_git": torch.version.git_version,
              "lowering": lowering(), "shape_checker_counterexample": unsafe_shape_checker(),
              "saved_candidate_audit": saved_audit()}
    serialized = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(serialized, encoding="utf-8")
    else:
        print(serialized)
