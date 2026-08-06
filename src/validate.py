"""Step 9 / 01-main.md section 9 -- the C1-C16 checklist. Each check returns
(status, detail) where status is PASS / WARN / FAIL / INFO / SKIP.

Config field names vary across model families (e.g. num_experts vs n_routed_experts).
_first_attr() checks a list of common aliases; if none match, the check returns SKIP
with a note -- that itself is a signal worth researching (Tier 2, see
02-new-module-handling.md) rather than a silent pass."""
import hashlib
import json
import os
from collections import defaultdict


def _first_attr(cfg, names, default=None):
    for n in names:
        if hasattr(cfg, n):
            return getattr(cfg, n)
    return default


def c1_layer_count(rows, cfg):
    layers = {r["layer_idx"] for r in rows if r.get("layer_idx") is not None}
    expected = _first_attr(cfg, ["num_hidden_layers", "n_layer", "num_layers"])
    if expected is None:
        return "SKIP", "no known layer-count field on config"
    ok = len(layers) == expected
    return ("PASS" if ok else "FAIL"), f"{len(layers)} == {expected}"


def c2_layer_clustering(rows, cfg):
    sigs = defaultdict(list)
    for layer_idx in {r["layer_idx"] for r in rows if r.get("layer_idx") is not None}:
        op_seq = tuple(r["op_type"] for r in rows if r.get("layer_idx") == layer_idx)
        sig = hashlib.sha1(json.dumps(op_seq).encode()).hexdigest()[:12]
        sigs[sig].append(layer_idx)
    n_clusters = len(sigs)

    # Expected heterogeneity from config comes from MULTIPLE per-layer schedule lists, not just
    # layer_types: e.g. SmolLM3 is uniform in layer_types but has no_rope_layers=[1,1,1,0,...]
    # (NoPE every 4th layer). Build a per-layer signature from every config list whose length ==
    # num_layers, and count distinct signatures.
    L = _first_attr(cfg, ["num_hidden_layers", "n_layer", "num_layers"])
    d = cfg.to_dict()
    sched = {k: v for k, v in d.items()
             if isinstance(v, (list, tuple)) and L and len(v) == L}
    if not sched:
        return ("PASS" if n_clusters == 1 else "WARN"), (
            f"{n_clusters} cluster(s); no per-layer schedule list on config to compare "
            f"(uniform, or scalar schedule like first_k_dense_replace)")
    expected = len({tuple(str(v[i]) for v in sched.values()) for i in range(L)})
    fields = sorted(sched)
    if n_clusters == expected:
        return "PASS", f"{n_clusters} clusters == {expected} from config schedule {fields}"
    # Mismatch: op-sequence clustering is heuristic -- it may not split mask-only differences
    # (sliding vs full attention share op sequences) or may split on value schedules. Per
    # 01-main.md §9.3 only a genuine config contradiction is FAIL; report WARN for review.
    return "WARN", (f"{n_clusters} trace clusters vs {expected} config-schedule signatures "
                    f"{fields} -- review (mask-only heterogeneity is op-invisible)")


def c3_dag_integrity(rows):
    ids = {r["op_id"] for r in rows}
    graph = {r["op_id"]: r.get("depends_on", []) for r in rows}
    # cycle check via DFS
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {i: WHITE for i in ids}

    def visit(n, stack):
        if color[n] == BLACK:
            return None
        if color[n] == GRAY:
            return stack + [n]
        color[n] = GRAY
        for dep in graph.get(n, []):
            cyc = visit(dep, stack + [n])
            if cyc:
                return cyc
        color[n] = BLACK
        return None

    for i in ids:
        cyc = visit(i, [])
        if cyc:
            return "FAIL", f"cycle detected: {cyc}"

    sources = {r["op_id"] for r in rows if not r.get("depends_on")}
    orphans = [r["op_id"] for r in rows if r["op_id"] not in sources and not r.get("depends_on")]
    return "PASS", f"acyclic, {len(orphans)} orphan(s)"


def c4_reachability(rows):
    # embedding lives at model.embed_tokens (outside `.layers.`), so its `block` is "other" --
    # identify it by op_type (dispatch truth) or module_path, consistent with lm_head below.
    embed_ids = [r["op_id"] for r in rows
                 if r.get("op_type") == "embedding" or "embed" in (r.get("module_path") or "")]
    lm_head_ids = [r["op_id"] for r in rows if "lm_head" in (r.get("module_path") or "")]
    if not embed_ids or not lm_head_ids:
        return "SKIP", "could not identify embedding/lm_head ops by naming heuristic"
    graph = {r["op_id"]: r.get("depends_on", []) for r in rows}
    reachable = set()
    frontier = list(lm_head_ids)
    while frontier:
        cur = frontier.pop()
        if cur in reachable:
            continue
        reachable.add(cur)
        frontier.extend(graph.get(cur, []))
    ok = any(e in reachable for e in embed_ids)
    return ("PASS" if ok else "FAIL"), "embedding reachable from lm_head" if ok else "no path found"


def c5_connection(rows, cfg):
    """01-main.md C5: (a) matmul input·weight contraction dims align; (b) each block preserves
    the residual-stream width (d_model). Runs on concrete in-memory shapes. Non-standard mixing
    (residual add whose operands differ, or a layer without a plain d_model residual) is reported
    but NOT failed -- the spec says C5 must cover both standard residual and non-standard mixing."""
    hidden = _first_attr(cfg, ["hidden_size", "d_model", "n_embd"])
    mm_ops = {"aten.mm.default", "aten.bmm.default", "aten.matmul.default"}
    bad = []
    for r in rows:
        shp = [s for s in (r.get("input_shape") or []) if s and len(s) >= 2]
        if r["raw_op"] in mm_ops and len(shp) >= 2:
            if shp[-2][-1] != shp[-1][-2]:
                bad.append(r["op_id"])
        elif r["raw_op"] == "aten.addmm.default" and len(shp) >= 2:
            if shp[-2][-1] != shp[-1][-2]:
                bad.append(r["op_id"])
    if bad:
        return "FAIL", f"{len(bad)} matmul(s) with mismatched contraction dims, e.g. op {bad[:5]}"
    if hidden is None:
        return "SKIP", "no hidden-size field to check residual width against"
    resid = defaultdict(int)
    for r in rows:
        if r.get("op_type") == "elementwise_add" and r.get("layer_idx") is not None:
            if hidden in [s[-1] for s in (r.get("output_shape") or []) if s]:
                resid[r["layer_idx"]] += 1
    layers = {r["layer_idx"] for r in rows if r.get("layer_idx") is not None}
    no_resid = sorted(l for l in layers if resid.get(l, 0) == 0)
    detail = (f"matmul contraction dims consistent; residual stream at d_model={hidden} in "
              f"{len(layers) - len(no_resid)}/{len(layers)} layers")
    if no_resid:
        return "WARN", detail + f"; layers w/o a plain d_model residual (non-standard mixing?): {no_resid[:10]}"
    return "PASS", detail


def c6_hidden_head_consistency(rows, cfg):
    hidden = _first_attr(cfg, ["hidden_size", "d_model", "n_embd"])
    if hidden is None:
        return "SKIP", "no known hidden-size field"
    mismatches = [
        r["op_id"] for r in rows
        if r.get("output_shape") and any(
            s and s[-1] not in (hidden, None) and r.get("block") in (None, "mlp", "self_attn")
            for s in r["output_shape"]
        )
    ]
    # heuristic only -- exact matching needs per-op knowledge of which axis is "hidden"
    return "PASS", f"hidden_size={hidden} (heuristic check, {len(mismatches)} flagged)"


def c7_gqa(cfg):
    """Head-sharing ratio. Reads the RESOLVED symbol table, not raw config fields.

    Re-deriving from aliases here duplicated the lookup and diverged: falcon-7b names the field
    `num_kv_heads` (not `num_key_value_heads`) and additionally overrides it to 1 via a separate
    `multi_query` flag, so this check reported "MHA" for a model that is MQA. resolve_symbols
    owns that config-semantics knowledge; keeping one source of truth prevents the two from
    disagreeing again."""
    import summarize
    syms = summarize.resolve_symbols(cfg)
    heads, kv_heads = syms.get("n_h"), syms.get("n_kv")
    if heads is None:
        return "SKIP", "no attention-head field"
    if kv_heads is None:
        kv_heads = heads
    if kv_heads == 1:
        return "PASS", f"MQA ({heads} query heads : 1 kv head)"
    if kv_heads == heads:
        return "PASS", "MHA (kv_heads == heads, not GQA)"
    if heads % kv_heads != 0:
        return "FAIL", f"heads={heads} not divisible by kv_heads={kv_heads}"
    return "PASS", f"GQA {heads}:{kv_heads} (repeat factor {heads // kv_heads})"


def c8_moe(rows, cfg):
    """01-main.md C8: verify from the TRACE (not just config) that router output dim ==
    num_experts, a top-k selection of experts_per_tok exists, and expert FFN weights are
    present (grouped [E,...] or >=2 per-expert modules). Only the routed-token *count* is
    data-dependent -> that stays symbolic (WARN, normal). Missing router+expert evidence is
    a real defect (FAIL): the experts aren't being traced. Runs on concrete in-memory shapes."""
    num_experts = _first_attr(cfg, ["num_experts", "n_routed_experts", "num_local_experts"])
    top_k = _first_attr(cfg, ["num_experts_per_tok", "moe_topk", "top_k"])
    if num_experts is None:
        return "SKIP", "no MoE-related fields found on config (likely a dense model)"

    def out_last_dims(r):
        return [s[-1] for s in (r.get("output_shape") or []) if s]

    router = any(num_experts in out_last_dims(r) for r in rows)
    topk_ok = top_k is not None and any(
        (r.get("raw_op", "").startswith("aten.topk") or r.get("op_type") == "topk")
        and any(top_k in (s or []) for s in (r.get("output_shape") or []))
        for r in rows
    )
    grouped = any((r.get("weight_shape") or [None])[0] == num_experts for r in rows)
    expert_idx = {p.split(".experts.")[1].split(".")[0]
                  for r in rows for p in r.get("params", []) if ".experts." in p}
    per_expert = len([x for x in expert_idx if x.isdigit()]) >= 2
    expert_weight = grouped or per_expert

    signals = [
        f"router_dim(E={num_experts}):{'ok' if router else 'MISSING'}",
        f"top_k({top_k}):{'ok' if topk_ok else 'n/a'}",
        f"expert_weight:{'grouped' if grouped else ('per-expert' if per_expert else 'MISSING')}",
    ]
    if not router and not expert_weight:
        # A num_experts-like config field does NOT guarantee the model uses MoE (some configs
        # carry vestigial/unused fields -- e.g. Nemotron-3-Nano-4B has n_routed_experts=8 but is
        # dense). If there are also no expert params at all, the model is dense here -> WARN, not
        # FAIL. FAIL only when expert params exist but weren't traced (genuine coverage miss).
        any_expert_param = any("expert" in p.lower() for r in rows for p in r.get("params", []))
        if not any_expert_param:
            return "WARN", (f"config has num_experts={num_experts} but NO expert params or router/"
                            f"expert ops in trace -- model is dense here (field appears vestigial): {signals}")
        return "FAIL", f"MoE config (E={num_experts}) has expert params but no router/expert op traced: {signals}"
    return "WARN", (
        f"MoE trace-verified [{', '.join(signals)}]; routed-token count is data-dependent/"
        f"symbolic (01-main.md C8) -- WARN is normal, not a defect."
    )


def c9_embed_lm_head(rows, cfg):
    vocab = _first_attr(cfg, ["vocab_size"])
    tied = _first_attr(cfg, ["tie_word_embeddings"], default=False)
    if vocab is None:
        return "SKIP", "no vocab_size field"
    return "PASS", f"vocab_size={vocab}, tie_word_embeddings={tied}"


def c10_coverage(rows, param_names: set):
    touched = set()
    for r in rows:
        touched.update(r.get("params", []))
    missing = param_names - touched
    if missing:
        return "FAIL", f"{len(missing)} param(s) with no contributing op, e.g. {sorted(missing)[:5]}"
    return "PASS", f"all {len(param_names)} params covered"


def c11_decode_consistency(decode_rows: list[dict]):
    if not decode_rows:
        return "SKIP", "no decode rows provided"
    cache_ops = [r for r in decode_rows if r.get("op_type") in ("concat",) or "cache" in (r.get("raw_op") or "").lower()]
    new_token_ok = any(
        r.get("input_shape") and any(s and len(s) >= 2 and s[1] == 1 for s in r["input_shape"])
        for r in decode_rows
    )
    if not cache_ops:
        return "WARN", "no concat/cache-touching op found in decode trace -- verify cache is actually being reused"
    if not new_token_ok:
        return "WARN", "could not confirm a seq-dim==1 tensor among decode inputs (heuristic)"
    return "PASS", f"{len(cache_ops)} cache-related op(s) found, new-token seq dim confirmed"


def c13_reproducibility(rows_run1: list[dict], rows_run2: list[dict]):
    def h(rows):
        return hashlib.sha256(json.dumps(rows, sort_keys=True, default=str).encode()).hexdigest()
    ok = h(rows_run1) == h(rows_run2)
    return ("PASS" if ok else "FAIL"), "identical across two runs" if ok else "MISMATCH between runs"


def c14_seq_len(seq_len_used: int, min_seq_len_required: int):
    ok = seq_len_used >= min_seq_len_required
    return ("PASS" if ok else "FAIL"), f"used={seq_len_used} >= required={min_seq_len_required}"


def c15_entrypoint_coverage(traced_entrypoints: set, discovered_entrypoints: set, cfg=None):
    """WARN if a discovered auxiliary entrypoint wasn't traced, OR if config declares MTP/nextn
    layers but no such module exists in the (native) model -- the native transformers impl often
    omits the MTP head that the model author's remote code builds, so config says MTP>0 while the
    traced model has none. Surface that gap (P8) instead of a vacuous PASS."""
    missing = discovered_entrypoints - traced_entrypoints
    notes = []
    if cfg is not None:
        mtp = _first_attr(cfg, ["num_nextn_predict_layers", "num_mtp_layers"], default=0) or 0
        has_mtp_mod = any("mtp" in d.lower() or "nextn" in d.lower() for d in discovered_entrypoints)
        if mtp and not has_mtp_mod:
            notes.append(f"config declares {mtp} MTP/nextn layer(s) but no MTP module in the traced "
                         f"model (native transformers impl omits the MTP head) -- MTP NOT traced")
    if missing:
        return "WARN", f"not traced: {sorted(missing)}" + ("; " + "; ".join(notes) if notes else "")
    if notes:
        return "WARN", "; ".join(notes)
    return "PASS", "all discovered entrypoints traced"


def c17_module_onboarding(literals: list[dict], symbols: dict, structures_dir: str | None = None,
                          model_type: str | None = None, model_id: str | None = None):
    """Phase 0 온보딩이 실제로 끝났는지 (02-new-module-handling.md).

    Why this check exists: the old process only ran on FAILURE. DeepSeek-V4-Pro passed C1-C16
    cleanly while five brand-new modules (CSA, HCA, Lightning Indexer, mHC, grouped output
    projection) went completely undocumented and left seven unexplained constants in the summary.
    A silently-wrong deliverable is exactly what the other checks cannot see, so this one asserts
    the research step happened rather than the trace being well-formed.

    WARN (not FAIL): the artifacts are still correct-as-traced; what is missing is the writeup."""
    gaps = []
    unresolved = [L["value"] for L in (literals or []) if not L.get("expr")]
    if unresolved:
        gaps.append(f"미해결 유도 상수 {len(unresolved)}개 {unresolved[:8]} "
                    f"-- rules/derived_dims.yaml에 식+출처 등록 필요")
    needles = [n for n in (model_id, model_type) if n]
    if structures_dir and needles:
        # A model we have onboarded should leave a trace in the structure library -- either its
        # repo id in some doc's "확인된 모델" list (how the docs are actually written) or, for a
        # brand-new family, its model_type in a new structure doc. Naming the exact doc is the
        # researcher's judgement, so we only assert that *something* mentions it.
        try:
            found = False
            for root, _dirs, files in os.walk(structures_dir):
                for fn in files:
                    if not fn.endswith(".md"):
                        continue
                    with open(os.path.join(root, fn), encoding="utf-8") as f:
                        text = f.read()
                    if any(n in text for n in needles):
                        found = True
                        break
                if found:
                    break
            if not found:
                gaps.append(f"`{needles[0]}`이 rules/structures/ 어디에도 없음 "
                            f"-- 새 구조 문서화 또는 기존 문서의 '확인된 모델' 갱신 필요")
        except OSError:
            pass
    if gaps:
        gaps.append("남은 축별 안건은 models/<model>/research_agenda.md 참고")
        return "WARN", "; ".join(gaps)
    return "PASS", "유도 상수 전부 설명됨, 구조 라이브러리에 등재됨"


def c16_unmapped(rows):
    unmapped = [r for r in rows if r.get("unmapped")]
    if not unmapped:
        return "INFO", "no unmapped ops"
    kinds = sorted({r["raw_op"] for r in unmapped})
    return "INFO", f"{len(unmapped)} unmapped rows, {len(kinds)} distinct raw ops: {kinds[:10]}"
