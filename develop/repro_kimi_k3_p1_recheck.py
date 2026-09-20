r"""Read-only review probes for codex_ask_kimi_k3_p1_recheck.md.

Run from the repository root:
    .venv\Scripts\python.exe -X utf8 develop/repro_kimi_k3_p1_recheck.py

Prints observations, not a release approval. Real model/rule/candidate files are
only read; synthetic proof/audit fixtures live in a temporary directory.
"""
import ast
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "develop")]

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
import kda_shim as KS
from scope import ScopeLabeler
from tracer import OpGraphTracer
import transition_release as TR


def gates():
    path = os.path.join("ops", "kda", "gate.py")
    plain = KS._extract_func(path, "naive_kda_gate")
    bounded = KS._extract_func(path, "naive_kda_lowerbound_gate")
    naive = KS._load_by_path("_codex_p1_recheck_naive", os.path.join("ops", "kda", "naive.py"))
    assert plain and bounded and naive
    q = torch.full((1, 1, 1, 128), 1e-4)
    z, a, bias = torch.zeros_like(q), torch.zeros(1), torch.zeros(128)
    kw = dict(q=q, k=q, v=torch.ones_like(q), g=z, beta=torch.zeros(1, 1, 1),
              A_log=a, dt_bias=bias, use_gate_in_kernel=True, lower_bound=-5.)
    selected = {}
    for needs in (True, False):
        for safe in (False, True):
            call = KS._wrap_kda(lambda **v: v, plain, bounded, gate_needs_safe=needs)
            selected[f"needs_safe={needs},safe_gate={safe}"] = call(**kw, safe_gate=safe)["g"].flatten()[0].item()
    normalized = q.float() / torch.sqrt((q.float() ** 2).sum(-1, keepdim=True) + 1e-6)
    call = KS._wrap_kda(lambda **v: v, plain, bounded, gate_needs_safe=False)
    current = call(**kw, use_qk_l2norm_in_kernel=True)["q"]
    recurrent = KS._wrap_kda(naive.naive_recurrent_kda, plain, bounded, gate_needs_safe=False)
    actual, _ = recurrent(**kw, use_qk_l2norm_in_kernel=True,
                          use_beta_sigmoid_in_kernel=True, output_final_state=True)
    expected, _ = naive.naive_recurrent_kda(
        q=normalized, k=normalized, v=kw["v"], g=bounded(z, a, bias, -5.),
        beta=torch.full((1, 1, 1), .5), output_final_state=True)

    # Execute only the installed chunk Python forwarding function. Stop at its
    # gate call, before any Triton kernel. This checks the actual forwarding
    # branch rather than using the wrapper's own condition as an oracle.
    source = ROOT / ".venv/Lib/site-packages/fla/ops/kda/chunk_fwd.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "chunk_kda_fwd")
    fn.decorator_list, fn.returns = [], None
    for arg in fn.args.args + fn.args.kwonlyargs:
        arg.annotation = None
    forwarded = {}

    class Captured(Exception):
        pass

    def capture_gate(**args):
        forwarded.update({"lower_bound": args["lower_bound"],
                          "safe_gate_forwarded_to_gate": "safe_gate" in args})
        raise Captured

    ns = {"kda_gate_chunk_cumsum": capture_gate, "RCP_LN2": 1.4426950408889634}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(source), "exec"), ns)
    try:
        ns["chunk_kda_fwd"](q=q, k=q, v=q, g=z, beta=kw["beta"], scale=1.,
                            initial_state=None, output_final_state=False,
                            use_gate_in_kernel=True, safe_gate=False, lower_bound=-5., A_log=a)
    except Captured:
        pass
    return {"wrapper_gate_selection": selected, "installed_chunk_forward_safe_false": forwarded,
            "normalization": {"q_element": 1e-4, "head_dim": 128,
                              "actual_element": current.flatten()[0].item(),
                              "expected_element": normalized.flatten()[0].item(),
                              "actual_recurrent_output_element": actual.flatten()[0].item(),
                              "expected_recurrent_output_element": expected.flatten()[0].item()}}


def trace_probe(device, kind):
    model = torch.nn.Identity()
    scope = ScopeLabeler(model)
    tracer = OpGraphTracer(model, scope)
    mode = FakeTensorMode() if device == "fake" else contextlib.nullcontext()
    with mode, torch.no_grad(), tracer:
        x = torch.zeros(2, 3, device="cpu" if device == "fake" else "meta")
        a, b = x[0], x[1]  # Both views are created before the first write.
        p = torch.ones(3, device=x.device)
        q = torch.full((3,), 2., device=x.device)
        if kind == "two_old_views":
            a.copy_(p)
            b.copy_(q)
            y = x.clone()
        elif kind == "base_write_old_view":
            x.add_(1)
            y = a.clone()
        else:
            a.copy_(p)
            y = x.clone()
    scope.remove()
    rows = tracer.rows
    by_id = {r["op_id"]: r for r in rows}
    seen, todo = set(), [rows[-1]["op_id"]]
    while todo:
        oid = todo.pop()
        if oid not in seen:
            seen.add(oid)
            todo.extend(by_id[oid]["depends_on"])
    writes = [r["op_id"] for r in rows if r["raw_op"] in ("aten.copy_.default", "aten.add_.Tensor")]
    ref = rows[-1]["input_sources"][0]
    producer = by_id[ref["op_id"]]
    return {"writes": writes, "ancestors": sorted(seen),
            "all_writes_reachable": all(i in seen for i in writes),
            "last_input_shape": rows[-1]["input_shape"][0],
            "claimed_source_output_shape": producer["output_shape"][ref["output_slot"]],
            "rows": [{k: r[k] for k in ("op_id", "raw_op", "depends_on", "input_sources",
                                        "input_tensor_ids", "output_tensor_ids")}
                     for r in rows]}


def exceptions():
    model = "moonshotai__Kimi-K3"
    actual = json.loads((ROOT / "models" / model / "full/lowering_proof.json").read_text(encoding="utf-8"))
    corrections = TR._corrections(model)
    cases = {"actual": actual}
    doubled = copy.deepcopy(actual)
    v = doubled["phases"]["decode"]
    v["templates"].append(copy.deepcopy(v["templates"][0]))
    v["templates"][1]["new_ops"]["aten.tanh.default"] = 1  # A different, unreviewed template.
    for key in ("records_total", "records_paired", "records_failed_template"):
        v[key] *= 2
    cases["two_templates_552_records"] = doubled
    other_scope = copy.deepcopy(actual)
    other_scope["phases"]["decode"]["templates"][0]["module_leaves"] = ["mlp", "self_attn"]
    cases["scope_violation_not_in_first_four_paths"] = other_scope
    unsupported = copy.deepcopy(actual)
    unsupported["phases"]["decode"]["templates"][0]["why"] = {"unsupported op: no replay performed": 69}
    cases["no_numerical_replay"] = unsupported
    result = {}
    with tempfile.TemporaryDirectory(prefix="kimi-p1-review-") as tmp:
        base = Path(tmp).resolve()
        proof = base / "proof.json"
        for name, data in cases.items():
            proof.write_text(json.dumps(data), encoding="utf-8")
            result[name] = TR._corrections_cover(str(proof), corrections)
        cand, old = base / "out" / model, base / "models" / model
        (cand / "full").mkdir(parents=True)
        old.mkdir(parents=True)
        (cand / "full/report.md").write_text("C1 PASS\n", encoding="utf-8")
        (cand / "full/lowering_proof.json").write_text(json.dumps(actual), encoding="utf-8")
        calls = [(0, "pass"), (0, "semantic_topology_change 276\n"),
                 (2, "proof process crashed before rewriting its JSON")]
        with patch.object(TR, "OUT", str(base / "out")), patch.object(TR, "MODELS", str(base / "models")), \
                patch.object(TR, "_run", side_effect=calls), contextlib.redirect_stdout(io.StringIO()):
            rc = TR.audit(model, "mock-profile.yaml")
        manifest = json.loads((cand / "audit_manifest.json").read_text(encoding="utf-8"))
        result["stale_proof_and_process_exit_2"] = {
            "audit_rc": rc, "approved": manifest["approved"],
            "proof_exit_code": manifest["lowering_proof"]["exit_code"],
            "claim": manifest["lowering_proof"].get("claim")}
    return result


if __name__ == "__main__":
    print(json.dumps({"gates": gates(),
                      "tracer": {f"{device}:{kind}": trace_probe(device, kind)
                                 for device in ("meta", "fake")
                                 for kind in ("narrow_port", "two_old_views", "base_write_old_view")},
                      "corrections": exceptions()}, ensure_ascii=False, indent=2))
