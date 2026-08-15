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
import collections
import json
import os

import anchors as anchors_mod
import axis_classes
import label_overrides
import tdep
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
    # The ACTIVATION operand, not operand 0. `aten.linear` passes the bias first, so on a biased
    # projection operand 0 is the bias -- a 1-D tensor of out_features. Its last axis equals the
    # contraction width only when in == out, which is exactly the square projection where nothing
    # else can tell the two apart, so the pin quietly wrote the BIAS's label onto the weight's
    # contraction axis: Qwen2.5-0.5B's q_proj came out `[T,d_model] @ [d_model,d_model]` while its
    # own output said `n_h*d_head`. Found by the matmul-composition check, 2026-08-06.
    act_i = None
    for i, c in enumerate(in_concrete):
        if not isinstance(c, list) or len(c) < 2:
            continue                      # a bias is rank 1 and can never be the activation
        if list(c) == list(w) or list(c) == list(w)[:-2] + list(w)[-2:][::-1]:
            continue                      # that is the weight itself (stored or transposed)
        act_i = i
    if act_i is None or act_i >= len(in_labels) or not in_labels[act_i]:
        return None
    act_last = in_concrete[act_i][-1]
    if not isinstance(act_last, int):
        return None
    # weight is [out, in] normally, [in, out] once transposed -- pick whichever axis actually
    # equals the activation's contracted dim.
    for idx in (len(w) - 1, len(w) - 2):
        if isinstance(w[idx], int) and w[idx] == act_last:
            return (idx, in_labels[act_i][-1])
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
    _IDENTITY = ("clone", "_to_copy", "contiguous", "detach", "alias", "copy_")
    for _pass in range(3):        # a fill-in can enable the next one; converges quickly
        changed = False
        for row, out in zip(rows, ordered):
            # An op that only copies cannot rename an axis, so a bare integer on one side takes
            # the other side's name. Same monotone rule as the cross-op fill below, applied
            # WITHIN the op -- xLSTM's backbone copies an axis already named `d_model*qk_f/n_h`
            # into a buffer and the output kept a bare 256.
            if row.get("op_type") in _IDENTITY:
                for ci, si in zip(row.get("input_shape") or [], out.get("input_shape") or []):
                    for co, so in zip(row.get("output_shape") or [], out.get("output_shape") or []):
                        if not (isinstance(ci, list) and isinstance(co, list) and ci == co):
                            continue
                        if not (isinstance(si, list) and isinstance(so, list)
                                and len(si) == len(so)):
                            continue
                        for i, (a, b) in enumerate(zip(si, so)):
                            if str(b).isdigit() and not str(a).isdigit():
                                so[i] = a
                                changed = True
                            elif str(a).isdigit() and not str(b).isdigit():
                                si[i] = b
                                changed = True
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


# A reshape carries its own evidence: `view [T,k] -> [T*k]` says that output axis IS the product
# of those two input axes, with no value matching involved. Deriving it independently and comparing
# is the cheapest cross-check we have on intermediate-tensor labels -- the ones with no module to
# ask. Measured across 26 models: 97.6% of derivable axes AGREE with the labels already there,
# which is strong evidence for both; the disagreements are where to look. The first two inspected
# were real (DeepSeek-V4-Flash's indexer projections, where d_model=4096 collides with
# n_h*d_head/g_o=4096 and the scoped formula won over the residual stream -- DeepSeek-V4-Pro runs
# the same module at d_model=7168 with no collision and says d_model).
#
# Used as an AUDITOR, not a labeller: adopting the derived names outright would have changed only
# 1,024 axes (0.4%), so it earns its place by finding errors, not by filling gaps.
_VIEW_FAMILY = {"view", "reshape", "_unsafe_view", "flatten", "unflatten",
                "clone", "_to_copy", "contiguous", "detach", "alias", "squeeze", "unsqueeze",
                # `copy_` too: xLSTM's backbone copies an axis already named `d_model*qk_f/n_h`
                # into a buffer and the name was dropped, leaving a bare 256.
                "copy_"}


def _canon_product(labels: list) -> str | None:
    """Join factors in this project's convention (T last, so one tensor gets one spelling).

    Returns None where the product would not be a credible name, so the cross-check reports a
    disagreement only when it has something worth saying. Three ways it fails, all measured on
    the fleet: an operand that is itself an expression (parenthesising is its own problem), a
    REPEATED symbol (`k*4*k` from DeepSeek-V3's gate -- a dimension is never one symbol squared),
    and an unnamed integer that is not a leading coefficient (`n_g_ssm*5`). The last is the same
    rule the symbolizer already applies when ranking competing explanations: an expression
    carrying an unnamed literal marks a rules gap, not an answer.
    """
    labels = [str(x) for x in labels]
    if any(("+" in x) or ("/" in x) or ("*" in x) for x in labels):
        return None
    named = [x for x in labels if not x.isdigit()]
    if len(named) != len(set(named)):
        return None                       # a symbol multiplied by itself is not a dimension
    if any(x.isdigit() for x in labels[1:]):
        return None                       # trailing literal -> unnamed factor, so no claim
    ts = [x for x in labels if x == "T"]
    return "*".join([x for x in labels if x != "T"] + ts)


def derive_from_reshape(cin: list, sin: list, cout: list, merges: bool = True) -> dict:
    """{output axis index: label derived from the INPUT axes}, for one reshape.

    Two-pointer walk over the concrete sizes. Only the MERGE direction is derivable -- several
    input axes collapsing into one output axis means that axis is their product. A SPLIT says
    nothing about which factor is which, so it is left to the symbol table. Size-1 axes are
    skipped entirely: they are runtime singletons already governed by the batch-axis invariant.
    """
    out: dict = {}
    i = j = 0
    while i < len(cin) and j < len(cout):
        a, b = cin[i], cout[j]
        if a == 1:
            i += 1
            continue
        if b == 1:
            j += 1
            continue
        if a == b:
            out.setdefault(j, str(sin[i]))          # 1:1 carry
            i, j = i + 1, j + 1
            continue
        if a < b:                                   # merge inputs until the product matches
            p, k, labs = a, i, [sin[i]]
            while p < b and k + 1 < len(cin):
                k += 1
                p *= cin[k]
                labs.append(sin[k])
            if p != b:
                return out
            lab = _canon_product(labs) if merges else None
            if lab:
                out[j] = lab
            i, j = k + 1, j + 1
            continue
        p, k = b, j                                 # split -- skip the group
        while p < a and k + 1 < len(cout):
            k += 1
            p *= cout[k]
        if p != a:
            return out
        i, j = i + 1, k + 1
    return out


_ALT_SPELLINGS = None


def _alt_spellings() -> dict:
    """{registered symbol: {its factorisation}} from rules/derived_dims.yaml.

    `d_inner` and `d_head_ssm*n_h_ssm` are the SAME quantity -- the second is literally the
    registered rule that defines the first. A reshape that spells it factored is not disagreeing
    with a label that uses the compact registered name, so reporting it as a contradiction is a
    false alarm (59 across Zamba2 and Nemotron). Note this must not swallow the real cases: `E`
    vs `k*T` looks similar but `k*T` is not a factorisation of `E` -- it is a different quantity
    that happens to share a value, and no rule says otherwise.
    """
    global _ALT_SPELLINGS
    if _ALT_SPELLINGS is None:
        import summarize
        _ALT_SPELLINGS = {}
        for rule in (summarize.load_derived_dims().get("rules") or []):
            sym, expr = rule.get("sym"), str(rule.get("expr") or "")
            if not sym or "+" in expr or "-" in expr or "/" in expr:
                continue
            factors = sorted(t.strip() for t in expr.split("*") if t.strip())
            if len(factors) > 1:
                _ALT_SPELLINGS.setdefault(sym, set()).add("*".join(factors))
    return _ALT_SPELLINGS


def _weight_agrees_with_operand(rows: list[dict], ordered: list[dict]) -> int:
    """One tensor, one name per axis -- WITHIN the op. Make `weight_shape` read like the operand.

    A projection row carries its weight twice: `weight_shape` as stored (`[out, in]`) and, inside
    `input_shape`, the operand the matmul actually consumed (usually the transpose). Those are the
    same tensor. They were rendered independently, so a correction applied to one silently left the
    other behind -- Llama-3.1-405B/70B read `[T, d_model] @ [d_model, n_kv*d_head]` while the very
    same weight's `weight_shape` said `[n_kv*d_head, n_h*d_head]`. Both are arithmetically true
    there (d_model == n_h*d_head), so no value check could see it, and no existing check compared
    the two spellings at all. 4,406 rows across 25 models; found by an outside review 2026-08-12.

    The OPERAND wins. Its contraction axis is pinned to the activation that flows into it
    (`_contraction_pin`), which is evidence from the dataflow; the stored form is resolved from the
    number alone. Only the axes that disagree are touched, and only when the concrete shapes line
    up as stored-or-transposed, so this cannot invent an agreement that is not physically there.
    """
    changed = 0
    for row, out in zip(rows, ordered):
        wp, cw = row.get("weight_pos"), row.get("weight_shape")
        if not (isinstance(wp, int) and wp >= 0 and isinstance(cw, list) and cw
                and not isinstance(cw[0], list)):
            continue
        cins, sins = row.get("input_shape") or [], out.get("input_shape") or []
        sw = out.get("weight_shape")
        if wp >= len(cins) or wp >= len(sins) or not isinstance(sw, list):
            continue
        cop, sop = cins[wp], sins[wp]
        if not (isinstance(cop, list) and isinstance(sop, list)
                and len(cop) == len(cw) == len(sw) == len(sop)):
            continue
        ident = list(range(len(cw)))
        swap = (list(range(len(cw) - 2)) + [len(cw) - 1, len(cw) - 2]) if len(cw) >= 2 else ident
        cwt = list(cw)[:-2] + list(cw)[-2:][::-1] if len(cw) >= 2 else list(cw)
        # A SQUARE weight matches both readings, and picking identity there reversed q_proj:
        # `nn.Linear(hidden, n_h*d_head)` stores `[out, in]`, the matmul consumes the transpose, and
        # for Llama d_model == n_h*d_head so the concrete shapes cannot tell them apart. The result
        # read `[d_model, n_h*d_head]` -- [in, out] -- while k_proj next to it correctly read
        # `[n_kv*d_head, d_model]`. The op TYPE settles it: a contracting op is fed the transpose.
        order = ([swap, ident] if row.get("op_type") in _CONTRACTING_OPS else [ident, swap])
        mapping = None
        for cand in order:
            want = cwt if cand is swap else list(cw)
            if list(cop) == want:
                mapping = cand
                break
        if mapping is None:
            continue                                            # not the same layout; say nothing
        for i, j in enumerate(mapping):
            a_lab, b_lab = str(sop[i]), str(sw[j])
            if a_lab == b_lab:
                continue
            # A weight is a static parameter: no batch axis, no dependence on the sequence length.
            # The operand may render a size-1 weight axis as `B`, and copying that across would
            # plant a batch axis inside a parameter (1,414 rows on the first attempt). When one
            # side violates a hard invariant, the OTHER side is the answer -- in both directions.
            a_bad = a_lab in ("B", "T") or bool(_HAS_T_TOKEN.search(a_lab))
            b_bad = b_lab in ("B", "T") or bool(_HAS_T_TOKEN.search(b_lab))
            if a_bad and not b_bad:
                sop[i] = b_lab
            elif not a_bad:
                sw[j] = a_lab
            else:
                continue
            changed += 1
    return changed


def _resid_stream_wins(rows: list[dict], ordered: list[dict], d_model: int | None) -> int:
    """On a dependency edge, the residual-stream name wins a tie.

    `d_model` is the one width every block reads and writes, and the rules already treat it as the
    most fundamental name (several derived rules carry `unless_equals: [d_model]` for exactly this
    reason). When a tensor flows from one op to the next and one end calls that axis `d_model` while
    the other calls it something equal-valued, the residual stream is the right answer.

    This is what lets `o_proj` come out correct without a special case for it: the residual
    `elementwise_add` downstream already reads `d_model` on both operands, so the projection's
    output takes that name, and _matmul_compose_enforce then carries it back onto the weight's
    out-features. Guarded on the concrete size actually being d_model, so it cannot spread the name
    to an axis that is not the residual width.
    """
    if not isinstance(d_model, int):
        return 0
    by_id = {r.get("op_id"): (r, o) for r, o in zip(rows, ordered)}
    changed = 0
    for _pass in range(3):
        moved = 0
        for row, out in zip(rows, ordered):
            for dep in (row.get("depends_on") or []):
                prod = by_id.get(dep)
                if not prod:
                    continue
                p_row, p_out = prod
                for bi_c, bi_s in zip(row.get("input_shape") or [], out.get("input_shape") or []):
                    if not isinstance(bi_c, list) or not isinstance(bi_s, list):
                        continue
                    for ao_c, ao_s in zip(p_row.get("output_shape") or [],
                                          p_out.get("output_shape") or []):
                        if not isinstance(ao_c, list) or list(ao_c) != list(bi_c):
                            continue
                        if not isinstance(ao_s, list) or len(ao_s) != len(bi_s):
                            continue
                        for i, v in enumerate(bi_c):
                            if v != d_model:
                                continue
                            x, y = str(bi_s[i]), str(ao_s[i])
                            if x == y or {x, y} & {"B", "T"}:
                                continue
                            if x == "d_model":
                                ao_s[i] = "d_model"
                            elif y == "d_model":
                                bi_s[i] = "d_model"
                            else:
                                continue
                            moved += 1
        # ...and ACROSS a reshape, within the op. The residual name otherwise stops at the first
        # `_unsafe_view`: `o_proj` emits `[T, n_h*d_head]`, a view turns it into `[B, T, ·]`, and
        # only THAT tensor meets the residual add. Aligning the non-1 axes carries the name back
        # through, which is what finally lets o_proj's out-features read `d_model`.
        for row, out in zip(rows, ordered):
            if row.get("op_type") not in _RESHAPE_FAMILY:
                continue
            cins, sins = row.get("input_shape") or [], out.get("input_shape") or []
            couts, souts = row.get("output_shape") or [], out.get("output_shape") or []
            if not (cins and couts and sins and souts):
                continue
            ci, co = cins[0], couts[0]
            si, so = sins[0], souts[0]
            if not all(isinstance(x, list) for x in (ci, co, si, so)):
                continue
            if len(si) != len(ci) or len(so) != len(co):
                continue
            ai = [(k, v) for k, v in enumerate(ci) if isinstance(v, int) and v != 1]
            ao = [(k, v) for k, v in enumerate(co) if isinstance(v, int) and v != 1]
            if len(ai) != len(ao) or [v for _k, v in ai] != [v for _k, v in ao]:
                continue
            for (ki, v), (ko, _v) in zip(ai, ao):
                if v != d_model:
                    continue
                x, y = str(si[ki]), str(so[ko])
                if x == y or {x, y} & {"B", "T"}:
                    continue
                if x == "d_model":
                    so[ko] = "d_model"
                elif y == "d_model":
                    si[ki] = "d_model"
                else:
                    continue
                moved += 1
        changed += moved
        if not moved:
            break
    return changed


def _transpose_swaps_names(rows: list[dict], ordered: list[dict]) -> int:
    """전치의 출력 라벨은 입력 라벨을 그 축만큼 맞바꾼 것이다. 추론이 아니라 정의다.

    `aten.t` 는 rank-2 전용이라 축 (0,1) 을 맞바꾼다 -- 함대의 t 14,202건이 전부 rank 2 다.
    `transpose`/`permute` 는 어느 축을 바꿨는지 행에 남아 있지 않으므로, **구체 shape 이 순열을
    유일하게 결정할 때만** 적용한다(같은 크기 축이 둘 이상이면 손대지 않는다).

    왜 필요한가: q_proj 가 `nn.Linear(d_model, n_h*d_head)` 이고 두 값이 같은 모델
    (Llama-3.1 전 크기, OLMo-2, OLMoE, SmolLM3, Qwen2.5 …)에서 가중치가 `[n_h*d_head, d_model]`
    로 바르게 렌더된 뒤에도 그 전치가 **입력과 똑같은 이름**을 내고 있었다. 바로 아래 matmul 은
    전치형 `[d_model, n_h*d_head]` 로 읽으므로 한 텐서에 두 이름이 됐고, 그것이 이번 세션에
    flow_ambig 이 오른 주된 원인이다(Llama-3.1-8B 128 -> 256 을 실측으로 추적, 2026-08-14).
    정사각이라 값으로는 영원히 못 잡는다 -- 전치라는 연산의 뜻으로만 잡힌다.
    """
    changed = 0
    for row, out in zip(rows, ordered):
        op = row.get("op_type")
        if op not in ("t", "transpose", "permute"):
            continue
        ins, outs = row.get("input_shape") or [], row.get("output_shape") or []
        lins, louts = out.get("input_shape") or [], out.get("output_shape") or []
        if len(ins) != 1 or len(outs) != 1 or len(lins) != 1 or len(louts) != 1:
            continue
        src, dst, lsrc, ldst = ins[0], outs[0], lins[0], louts[0]
        if not all(isinstance(x, list) for x in (src, dst, lsrc, ldst)):
            continue
        if len(src) != len(dst) or len(src) != len(lsrc) or len(dst) != len(ldst):
            continue
        if op == "t":
            if len(src) != 2:
                continue
            perm = [1, 0]
        else:
            # 구체 크기로 순열을 복원한다. 같은 크기가 둘 이상이면 유일하지 않으므로 포기.
            if len(set(src)) != len(src):
                continue
            pos = {v: i for i, v in enumerate(src)}
            if any(v not in pos for v in dst):
                continue
            perm = [pos[v] for v in dst]
        want = [lsrc[i] for i in perm]
        if want != ldst:
            for i, v in enumerate(want):
                if ldst[i] != v:
                    ldst[i] = v
                    changed += 1
    return changed


_SQUEEZE_VIEW_OPS = frozenset({"view", "reshape", "_unsafe_view", "unsqueeze", "squeeze", "alias"})



def _squeeze_view_keeps_names(rows: list[dict], ordered: list[dict]) -> int:
    """A view that only adds or drops LEADING size-1 axes keeps every other axis it had.

    `_unsafe_view([T, d_moe]) -> [B, T, 3072]` is the same tensor with a batch axis put back on.
    The trailing widths are identical, so the trailing names are identical -- there is nothing to
    infer. But `_propagate_labels` matches tensors by full shape, so a rank change hides the edge
    from it, and the name is simply lost: Hunyuan's shared MLP came out `[T, d_moe]` at the matmul
    and `[B, T, 3072]` one op later, and the whole silu/mul chain after it inherited the bare
    integer. Found while auditing what the "unnamed" axes actually are, 2026-08-13 -- 62% of them
    are size-1 axes and ~35% are loop counters (both correctly nameless), and this was most of
    what remained.

    Monotone on purpose: it only fills an axis rendered as a BARE INTEGER, never overwrites a
    name. Filling cannot introduce a disagreement -- it removes one -- while overwriting could
    carry a wrong name across the edge, which is how `_carry_reshape_labels` failed.
    """
    changed = 0
    for row, out in zip(rows, ordered):
        if row.get("op_type") not in _SQUEEZE_VIEW_OPS:
            continue
        ins, outs = row.get("input_shape") or [], row.get("output_shape") or []
        lins, louts = out.get("input_shape") or [], out.get("output_shape") or []
        if len(ins) < 1 or len(outs) != 1 or len(louts) != 1 or not lins:
            continue
        src, dst, lsrc, ldst = ins[0], outs[0], lins[0], louts[0]
        if not all(isinstance(x, list) for x in (src, dst, lsrc, ldst)):
            continue
        if len(src) != len(lsrc) or len(dst) != len(ldst):
            continue
        # identical once leading 1s are stripped from both sides
        a = [(i, v) for i, v in enumerate(src) if not (v == 1 and i < len(src) - len(dst))]
        s_core = [v for v in src if isinstance(v, int)]
        d_core = [v for v in dst if isinstance(v, int)]
        while s_core and s_core[0] == 1 and len(s_core) > len(d_core):
            s_core.pop(0)
        while d_core and d_core[0] == 1 and len(d_core) > len(s_core):
            d_core.pop(0)
        if not s_core or s_core != d_core:
            continue
        del a
        # walk both from the right; the cores line up by construction
        for k in range(1, len(s_core) + 1):
            si, di = len(src) - k, len(dst) - k
            if si < 0 or di < 0 or src[si] != dst[di]:
                break
            # NEVER a size-1 axis. It carries no information, and copying across one produced
            # `[B, 1, 3072]` -> `[B, B, 3072]` in decode: two batch axes, which the gate's
            # batch_excl invariant forbids outright. The trailing 1s of the two shapes can line up
            # by accident; the widths that matter are the ones > 1.
            if src[si] == 1:
                continue
            if str(ldst[di]).isdigit() and not str(lsrc[si]).isdigit():
                ldst[di] = lsrc[si]
                changed += 1
            elif str(lsrc[si]).isdigit() and not str(ldst[di]).isdigit():
                lsrc[si] = ldst[di]
                changed += 1
    return changed


def _gather_keeps_features(rows: list[dict], ordered: list[dict]) -> int:
    """A gather along dim 0 selects ROWS. It cannot rename the trailing axes.

    `index(x[T, d_model], idx[k*T]) -> [k*T, d_model]` is a definition: the op picks rows out of
    `x`, so every axis after the first is the same axis it was, whatever its width collides with.
    gpt-oss has d_model == d_moe == 2880 and the routed-token gather came out
    `[T, d_model], [k*T] -> [k*T, d_moe]` -- the expert intermediate width pasted onto the residual
    stream at the moment it enters the experts, and then carried down the whole gate/up chain. No
    value can catch that (the numbers are equal) and no rule was looking, because the axis is not
    declared by any module: it is produced by an indexing op. Found by outside review, 2026-08-13.

    Deliberately narrow: dim-0 gather only (same rank in and out, identical trailing concrete
    widths, index operand of rank 1). A gather on a later axis, or one that changes rank, says
    nothing about which axis survived and is left alone.
    """
    changed = 0
    fixed = {}          # op_id -> the corrected rendered output shape
    for row, out in zip(rows, ordered):
        if row.get("op_type") != "index":
            continue
        ins, outs = row.get("input_shape") or [], row.get("output_shape") or []
        louts = out.get("output_shape") or []
        lins = out.get("input_shape") or []
        if len(ins) < 2 or len(outs) != 1 or len(louts) != 1 or len(lins) != len(ins):
            continue
        src, idx, dst = ins[0], ins[1], outs[0]
        if not (isinstance(src, list) and isinstance(idx, list) and isinstance(dst, list)):
            continue
        if len(idx) != 1 or len(src) != len(dst) or len(src) < 2:
            continue
        if list(src[1:]) != list(dst[1:]):      # trailing widths must be untouched
            continue
        lsrc, ldst = lins[0], louts[0]
        if not isinstance(lsrc, list) or not isinstance(ldst, list) or len(lsrc) != len(ldst):
            continue
        hit = False
        for i in range(1, len(ldst)):
            if ldst[i] != lsrc[i]:
                ldst[i] = lsrc[i]
                changed += 1
                hit = True
        if hit and row.get("op_id") is not None:
            fixed[row["op_id"]] = (list(dst), list(ldst))
    # Carry it to the consumers of that exact tensor. _propagate_labels is monotone -- it fills
    # integers and never overwrites a name -- so without this the gather's output read `d_model`
    # while the very next op read `d_moe` on the same tensor (gpt-oss: 72 axes on 120b, 48 on 20b,
    # all a `masked_fill_` applying the routing mask to the gathered hidden states). Renaming one
    # end and leaving the other is the defect, not the fix.
    if fixed:
        for row, out in zip(rows, ordered):
            deps = [d for d in (row.get("depends_on") or []) if d in fixed]
            if not deps:
                continue
            for oid in deps:
                cshape, lshape = fixed[oid]
                for cv, sv in zip(row.get("input_shape") or [], out.get("input_shape") or []):
                    if not isinstance(cv, list) or not isinstance(sv, list):
                        continue
                    if list(cv) != cshape or len(sv) != len(lshape):
                        continue
                    for i in range(1, len(sv)):
                        if sv[i] != lshape[i]:
                            sv[i] = lshape[i]
                            changed += 1
    return changed


def _unify_axis_classes(rows: list[dict], ordered: list[dict]) -> int:
    """한 축(등가류)에는 이름이 하나다. 만든 자리가 정하고 나머지는 따른다.

    이 프로젝트의 라벨 결함은 거의 전부 한 가지였다 -- 이름이 **텐서가 아니라 칸**에 붙어 있고,
    한 텐서가 여러 칸에 흩어져 각각 값으로 이름을 얻는다는 것. 그래서 한 자리를 고치면 옆자리가
    옛 이름을 유지하고, 그것을 꿰매려고 사후 패스가 열세 개까지 늘었으며 서로 싸웠다.

    이 패스가 그 순서를 바로 세운다. `src/axis_classes` 가 "어느 칸들이 같은 축인가"를 보수적으로
    계산하고(모호하면 잇지 않는다), 여기서 등가류마다 **이름을 하나만** 골라 모든 칸에 쓴다.

    승자: 그 축을 **가장 먼저 만들어 낸 출력 자리**의 이름. 근거는 구조에서 나온다 -- 행렬곱
    합성, 모듈이 선언한 폭, split 조각, 전치의 순열은 전부 텐서를 만드는 쪽에 있고, 소비자의
    피연산자 이름은 같은 숫자를 다시 값 매칭한 결과다. 출력 자리가 없으면(외부에서 들어온 텐서)
    가장 이른 입력 자리를 쓴다.

    정수는 이름을 이기지 못한다: 후보는 이름뿐이고, 이름이 하나도 없으면 손대지 않는다
    (정수 채우기는 `_propagate_labels` 의 일이다). 크기-1 축은 아예 제외한다 -- 정보가 없어서
    `B` 와 리터럴 `1` 이 서로 덮어쓰는 사고가 난다.
    """
    conc = {r.get("op_id"): r for r in rows}
    uf = axis_classes.build(rows, conc)
    # 등가류 -> [(op_id, is_output, ordered_row, field, shape_index, axis)]
    members = {}
    for row, out in zip(rows, ordered):
        oid = row.get("op_id")
        crow = conc.get(oid) or {}
        for fld, tag in (("output_shape", "o"), ("input_shape", "i")):
            cvals = crow.get(fld) or []
            for si, sh in enumerate(out.get(fld) or []):
                if not isinstance(sh, list):
                    continue
                csh = cvals[si] if si < len(cvals) and isinstance(cvals[si], list) else None
                for ax in range(len(sh)):
                    if csh is not None and ax < len(csh) and csh[ax] == 1:
                        continue                       # 크기-1 축은 제외
                    members.setdefault(uf.find((oid, tag, si, ax)), []).append(
                        (oid, tag == "o", sh, ax))

    changed = 0
    for sites in members.values():
        names = {str(sh[ax]) for _, _, sh, ax in sites}
        if len(names) < 2:
            continue
        named = [(oid, is_out, sh, ax) for oid, is_out, sh, ax in sites
                 if not str(sh[ax]).isdigit()]
        if not named:
            continue
        outs = [x for x in named if x[1]]
        winner = min(outs or named, key=lambda x: x[0])
        want = str(winner[2][winner[3]])
        for _, _, sh, ax in sites:
            # 정수도 채운다. 처음에는 "정수 채우기는 _propagate_labels 의 일"이라며 건너뛰었는데,
            # 그러면 한 등가류에 이름과 정수가 함께 남아 "한 축 한 이름"이 깨진다 -- concat
            # 통과 간선을 넣어 등가류가 커지자 Nemotron 에서 `['2', 'n_kv']` 로 16/24건 드러났다
            # (2026-08-14). 이름이 정수를 이기는 것은 두 패스가 같은 방향이라 싸우지 않고,
            # 뒤에 도는 `_unname_refilled_operands` 가 루프 계단에 대해 여전히 마지막 말을 한다.
            if str(sh[ax]) != want:
                sh[ax] = want
                changed += 1
    return changed


def _producer_wins(rows: list[dict], ordered: list[dict], passes: int = 3) -> int:
    """소비자의 피연산자 이름은 그 텐서를 **만든 op** 의 출력 이름을 따른다.

    지금까지 이 방향이 없었다. `_propagate_labels` 는 단조라 정수만 메우고 이름은 못 덮으므로,
    생산자가 더 나은 근거로 이름을 정해도 소비자는 자기가 값으로 맞춘 이름을 그대로 들고 있었다.
    Qwen3-Next 의 `bmm(q, v)` 출력이 `d_head_lin_v` 로 바로잡힌 뒤에도 바로 다음 `_unsafe_view`
    가 `d_head_lin_k` 를 유지한 것이 그 예다(549축, 2026-08-14).

    왜 생산자가 이기는가: 생산자의 출력 이름은 그 텐서를 **만든 구조**에서 나온다 -- 행렬곱
    합성, 모듈이 선언한 폭, split 조각, 전치의 순열. 소비자의 피연산자 이름은 같은 숫자를 다시
    값 매칭한 결과다. 둘이 다르면 전자가 더 나은 증거다.

    보수적인 부분:
      * `src/axis_classes` 와 **같은 간선 조건**을 쓴다 -- 생산자의 그 출력과 shape 이 일치하는
        피연산자가 **정확히 하나**일 때만. 모호하면 손대지 않는다.
      * 크기-1 축은 건드리지 않는다. `B` 와 리터럴 `1` 이 서로 덮어쓰는 것을 막는다.
      * 이름이 있는 자리만 덮는다. 정수를 이름으로 채우는 것은 `_propagate_labels` 의 일이고,
        여기서 하면 두 패스가 같은 자리를 두고 싸운다.
    """
    by_id = {r.get("op_id"): (r, o) for r, o in zip(rows, ordered)}
    changed = 0
    for _ in range(passes):
        moved = 0
        for row, out in zip(rows, ordered):
            ins, sins = row.get("input_shape") or [], out.get("input_shape") or []
            for dep in (row.get("depends_on") or []):
                pr = by_id.get(dep)
                if not pr:
                    continue
                prow, pout = pr
                douts, souts = prow.get("output_shape") or [], pout.get("output_shape") or []
                for oi, po in enumerate(douts):
                    if not isinstance(po, list) or not po or oi >= len(souts):
                        continue
                    if sum(1 for x in douts if x == po) != 1:
                        continue                      # 생산자 쪽 모호
                    match = [k for k, x in enumerate(ins) if x == po]
                    if len(match) != 1:
                        continue                      # 소비자 쪽 모호
                    i = match[0]
                    src, dst = souts[oi], (sins[i] if i < len(sins) else None)
                    if not (isinstance(src, list) and isinstance(dst, list)
                            and len(src) == len(dst) == len(po)):
                        continue
                    for ax, size in enumerate(po):
                        if size == 1:
                            continue
                        a, b = str(src[ax]), str(dst[ax])
                        if a == b or a.isdigit() or b.isdigit():
                            continue
                        dst[ax] = a
                        moved += 1
        changed += moved
        if not moved:
            break
    return changed


def _matmul_compose_enforce(rows: list[dict], ordered: list[dict]) -> int:
    """`[.., m, k] @ [.., k, n] -> [.., m, n]` -- make the three names agree. Definition, not a rule.

    The contraction axis is ONE axis: it is the second operand's `k` and the first operand's `k`,
    and the output's `n` is the second operand's `n`. Rendering them independently let them drift
    apart wherever two config widths coincide, and `o_proj` is the standing example -- Llama's
    output projection is `nn.Linear(n_h*d_head, hidden_size)`, so its out-features is `d_model`, but
    every axis of that row read `n_h*d_head` because the two are equal. An outside review called it
    "가장 오래 방치" (2026-08-12), and it is a structural error even where the numbers hide it: a
    model that sizes them differently would render a wrong name, not just a redundant one.

    Only fires where the CONCRETE shapes really compose, and only copies a name onto an axis that
    disagrees -- it cannot invent a composition the tensors do not have. Runtime axes (`B`, `T`) are
    left alone: a size-1 operand renders `B` by convention, which is not a disagreement about width.
    """
    changed = 0
    for row, out in zip(rows, ordered):
        if row.get("op_type") not in ("matmul", "linear", "mm", "bmm", "batched_matmul"):
            continue
        cins, sins = row.get("input_shape") or [], out.get("input_shape") or []
        couts, souts = row.get("output_shape") or [], out.get("output_shape") or []
        ci = [i for i, x in enumerate(cins) if isinstance(x, list) and len(x) >= 2]
        if len(ci) < 2 or not couts or not isinstance(couts[0], list) or len(couts[0]) < 2:
            continue
        ia, ib = ci[-2], ci[-1]
        if ia >= len(sins) or ib >= len(sins) or not souts:
            continue
        ca, cb, co = cins[ia], cins[ib], couts[0]
        sa, sb, so = sins[ia], sins[ib], souts[0]
        if not all(isinstance(x, list) for x in (sa, sb, so)):
            continue
        if len(sa) != len(ca) or len(sb) != len(cb) or len(so) != len(co):
            continue
        # (a[-1], b[-2]) contract; a[-2] -> out[-2]; b[-1] -> out[-1]
        for (xs, xi, ys, yi) in ((sa, -1, sb, -2), (sa, -2, so, -2), (sb, -1, so, -1)):
            cx = (ca if xs is sa else cb)[xi]
            cy = (cb if ys is sb else co)[yi]
            if not (isinstance(cx, int) and isinstance(cy, int) and cx == cy):
                continue
            lx, ly = str(xs[xi]), str(ys[yi])
            if lx == ly or {lx, ly} & {"B", "T"}:
                continue
            # a bare integer takes the name; otherwise WHICH side is authoritative depends on
            # whether this multiplication has a declared weight.
            #
            #   * with a parameter -- the output IS the module's declared out_features, so it wins
            #     (that is the o_proj case this pass exists for);
            #   * without one -- `attn @ v`, a pure activation product -- the output is not
            #     declared by anything. Its name came from value matching, while the operand's came
            #     from the tensor that produced it. Letting the output win there pushed the
            #     CONTRACTION's name onto the operand's free axis: Qwen3-Next's
            #     `bmm([48,64,128], [48,128,128])` had v's feature axis (`d_head_lin_v`, from the
            #     view above it) overwritten with `d_head_lin_k`, 549 axes across five models --
            #     the largest remaining class conflict on 2026-08-14. head_k_dim == head_v_dim so
            #     no value can separate them; only provenance can.
            declared = bool(anchors_mod.weight_param(row.get("params")))
            if lx.isdigit():
                xs[xi] = ly
            elif ly.isdigit():
                ys[yi] = lx
            elif ys is so and declared:
                xs[xi] = ly
            elif ys is so:
                ys[yi] = lx
            else:
                ys[yi] = lx
            changed += 1
    return changed


def _resync_param_labels(rows: list[dict], ordered: list[dict]) -> int:
    """One parameter, one labelling -- across every op that touches it.

    Making `weight_shape` follow the operand fixes the row but splits the parameter: the CONTRACTING
    op learns `d_model` from the activation it multiplies, while the bare `t` that only transposes
    the same weight has no activation to learn from and keeps whatever the number matched
    (`n_h*d_head`). Same parameter, two labellings, 3,018 rows.

    So pick one and apply it everywhere. The contracting op's version wins for exactly the reason it
    is better evidence -- its contraction axis was pinned to the tensor that flows in, not resolved
    from a number that two config fields happen to share.
    """
    # `len(params) == 1` 이 아니라 anchors.weight_param 을 쓴다. 편향이 있는 투영은 op 에
    # 파라미터를 **둘**(weight + bias) 넘기므로 예전 조건은 그런 모듈을 통째로 건너뛰었다 --
    # Qwen2.5 · falcon · GPT-2 · SmolLM3 처럼 attention bias 를 쓰는 모델 전부다. 그래서 그
    # 모델들에서 `q_proj` 의 `t` 가 `[d_model, d_model]` 로 남고 그 아래 `linear` 은
    # `[n_h*d_head, d_model]` 로 읽는, 한 파라미터 두 표기가 계속 있었다(등가류 감사가
    # Qwen2.5-0.5B 144건 · falcon-7b 128건으로 짚었다, 2026-08-14). 편향은 out_features 를
    # 한 번 더 말할 뿐 폭에 대한 정보가 없으므로 anchors 쪽은 이미 같은 이유로 고쳐져 있었다.
    best = {}
    for row, out in zip(rows, ordered):
        pname, sw = anchors_mod.weight_param(row.get("params")), out.get("weight_shape")
        if not pname or not isinstance(sw, list) or not sw or isinstance(sw[0], list):
            continue
        if row.get("op_type") in _CONTRACTING_OPS:
            best.setdefault(pname, [str(x) for x in sw])

    changed = 0
    for row, out in zip(rows, ordered):
        pname = anchors_mod.weight_param(row.get("params"))
        if not pname or pname not in best:
            continue
        want, cw = best[pname], row.get("weight_shape")
        sw = out.get("weight_shape")
        if not (isinstance(sw, list) and len(sw) == len(want)):
            continue
        for i, lab in enumerate(want):
            if lab in ("B", "T") or _HAS_T_TOKEN.search(lab):
                continue
            if str(sw[i]) != lab:
                sw[i] = lab
                changed += 1
        # and the operand that IS this weight, in the same op
        wp = row.get("weight_pos")
        cins, sins = row.get("input_shape") or [], out.get("input_shape") or []
        if not (isinstance(wp, int) and 0 <= wp < len(sins) and wp < len(cins)):
            continue
        cop, sop = cins[wp], sins[wp]
        if not (isinstance(cop, list) and isinstance(sop, list) and len(cop) == len(cw) == len(want)):
            continue
        cwt2 = list(cw)[:-2] + list(cw)[-2:][::-1] if len(cw) >= 2 else list(cw)
        swapped = want[:-2] + want[-2:][::-1] if len(want) >= 2 else want
        if row.get("op_type") in _CONTRACTING_OPS and list(cop) == cwt2:
            src = swapped                                       # 수축 op 은 전치를 먹는다
        elif list(cop) == list(cw):
            src = want
        elif list(cop) == cwt2:
            src = swapped
        else:
            continue
        for i, lab in enumerate(src):
            if str(sop[i]) != lab:
                sop[i] = lab
                changed += 1
        # ...and the OUTPUT, when this op merely re-lays-out the weight. A bare `t` produces the
        # transposed parameter, so its output is the same tensor a third time; leaving it behind
        # made the consumer disagree with its producer (`shared_expert_gate`'s out_features=1 axis
        # read `1` on the operand and `B` on the transpose that produced it, 296 rows).
        couts, souts = row.get("output_shape") or [], out.get("output_shape") or []
        for co, so in zip(couts, souts):
            if not (isinstance(co, list) and isinstance(so, list) and len(co) == len(want)):
                continue
            # 전치 계열은 **연산의 뜻**으로 먼저 판정한다. 정사각 가중치는 `co == cw` 와
            # `co == cw[전치]` 가 동시에 참이라, shape 비교만 하면 "안 바뀜" 가지가 먼저
            # 걸려 전치가 이름을 그대로 물려준다. 전치는 정의상 축을 맞바꾼다.
            if row.get("op_type") in ("t", "transpose", "permute") and len(want) >= 2:
                tgt = want[:-2] + want[-2:][::-1]
            elif list(co) == list(cw):
                tgt = want
            elif len(want) >= 2 and list(co) == list(cw)[:-2] + list(cw)[-2:][::-1]:
                tgt = want[:-2] + want[-2:][::-1]
            else:
                continue
            for i, lab in enumerate(tgt):
                if str(so[i]) != lab:
                    so[i] = lab
                    changed += 1
    return changed


import re as _re
_HAS_T_TOKEN = _re.compile(r"T")

_REGISTERED_SYMS = None


def _is_registered(label) -> bool:
    """True if `label` is a name rules/derived_dims.yaml declares (not one we inferred).

    Used to stop an inference from overwriting a sourced rule. Bare integers and plain symbols are
    not "registered" in this sense -- a plain symbol is exactly what an inference is entitled to
    refine into a composite.
    """
    global _REGISTERED_SYMS
    if _REGISTERED_SYMS is None:
        import summarize
        _REGISTERED_SYMS = {str(r.get("sym")) for r in (summarize.load_derived_dims().get("rules")
                                                        or []) if r.get("sym")}
    return str(label) in _REGISTERED_SYMS


def reshape_disagreements(row: dict, ordered: dict) -> list:
    """[(axis, current label, derived label)] where the two accounts of a reshape differ.

    Reports a DISAGREEMENT, not a verdict: it says the two accounts of one tensor cannot both be
    right, never which one is wrong. DeepSeek's MLA shows why -- `view [B,T,n_h,d_nope] ->
    [B,T,n_h*d_v]` is flagged correctly, but there the OUTPUT is right (an attention result's head
    width is d_v by definition) and the INPUT lost `d_v` to `d_nope` because both are 128.
    """
    if row.get("op_type") not in _VIEW_FAMILY:
        return []
    cin_all, sin_all = row.get("input_shape") or [], ordered.get("input_shape") or []
    if not cin_all or not sin_all or not isinstance(cin_all[0], list):
        return []
    cin, sin = cin_all[0], sin_all[0]
    if not isinstance(sin, list) or len(sin) != len(cin):
        return []
    bad = []
    for sout, cout in zip(ordered.get("output_shape") or [], row.get("output_shape") or []):
        if not isinstance(cout, list) or not isinstance(sout, list):
            continue
        for idx, lab in derive_from_reshape(cin, sin, cout).items():
            if idx >= len(sout):
                continue
            cur = str(sout[idx])
            if cur == lab or cur.isdigit():
                continue
            canon = "*".join(sorted(lab.split("*")))
            if canon in _alt_spellings().get(cur, ()):
                continue                  # same quantity, registered compact name vs factors
            bad.append((idx, cur, lab))
    return bad


def _carry_reshape_labels(rows: list[dict], ordered: list[dict]) -> int:
    """A view cannot change what an axis MEANS -- carry the input's label onto the output.

    Only the 1:1 axes are carried (an input axis that passes through a reshape untouched), never
    the merged ones: a merge product is derivable but the existing label may be a better-attested
    spelling of the same thing. Forward direction, because the input label came from the producer,
    which sits closer to the tensor's origin (the residual stream, an nn.Linear's declared width).

    What this fixes: wherever `d_model == n_h*d_head` -- Qwen2.5-0.5B (896 = 14*64), SmolLM3-3B
    (2048 = 16*128), and every other model with that coincidence -- the flatten in front of q/k/v_proj
    re-labelled the RESIDUAL STREAM as the packed head layout, because inside a `self_attn` scope
    the head formula outranks the plain symbol. The input side had it right all along.
    """
    n = 0
    for row, out in zip(rows, ordered):
        if row.get("op_type") not in _VIEW_FAMILY:
            continue
        cin_all, sin_all = row.get("input_shape") or [], out.get("input_shape") or []
        if not cin_all or not sin_all or not isinstance(cin_all[0], list):
            continue
        cin, sin = cin_all[0], sin_all[0]
        if not isinstance(sin, list) or len(sin) != len(cin):
            continue
        for sout, cout in zip(out.get("output_shape") or [], row.get("output_shape") or []):
            if not isinstance(cout, list) or not isinstance(sout, list):
                continue
            for idx, lab in derive_from_reshape(cin, sin, cout, merges=False).items():
                if idx < len(sout) and str(sout[idx]) != lab and not lab.isdigit():
                    sout[idx] = lab
                    n += 1
    return n


def _apply_merge_derivation(rows: list[dict], ordered: list[dict], authoritative: dict) -> int:
    """Label an axis that a reshape CREATED by merging input axes, and mark it authoritative.

    `view [T,k] -> [T*k]` does not merely happen to equal T*k -- that axis IS the flattened
    [token, slot] pair, and no value match can outrank the op's own operands. Only merges are
    taken (>=2 input axes collapsing into one), never the 1:1 carries: those tie into the
    residual-stream chain, where relabelling one end without its consumers doubled flow_ambig
    (see the note at the call site).

    What this fixes: MoE routing widths, where the routed-slot count collides with a config
    symbol. GLM-4.5-Air flattens [T=16, k=8] to 128 and E is also 128, so the slot list read `E`;
    DeepSeek-V4-Pro flattens [2048, 6] to 12288 and `4*d_moe` is also 12288. Both then carried
    the wrong name through sort/index/gather. gpt-oss got `k*T` right only because nothing there
    collided. Marked authoritative so anchors.propagate carries it to every op seeing the tensor.
    """
    n = 0
    for row, out in zip(rows, ordered):
        if row.get("op_type") not in _VIEW_FAMILY:
            continue
        cin_all, sin_all = row.get("input_shape") or [], out.get("input_shape") or []
        if not cin_all or not sin_all or not isinstance(cin_all[0], list):
            continue
        cin, sin = cin_all[0], sin_all[0]
        if not isinstance(sin, list) or len(sin) != len(cin):
            continue
        if any(str(x).isdigit() for x in sin):
            continue                      # an unnamed operand cannot license a derived product
        for si, (sout, cout) in enumerate(zip(out.get("output_shape") or [],
                                              row.get("output_shape") or [])):
            if not isinstance(cout, list) or not isinstance(sout, list):
                continue
            plain = derive_from_reshape(cin, sin, cout, merges=False)
            for idx, lab in derive_from_reshape(cin, sin, cout).items():
                if idx in plain or idx >= len(sout) or "*" not in lab:
                    continue              # merges only -- plain carries stay out of this
                if str(sout[idx]) != lab:
                    sout[idx] = lab
                    n += 1
                authoritative.setdefault(row.get("op_id"), set()).add(
                    ("output_shape", si, idx))
    return n


def _carry_authoritative(rows: list[dict], ordered: list[dict], authoritative: dict) -> int:
    """Carry an AUTHORITATIVE axis label across a reshape, within the one op.

    anchors.propagate syncs a tensor between the op that produced it and the ops that consume
    it, but never between an op's own input and output -- so a derived name stopped at the
    reshape's edge and the next `_to_copy` still read the old one (ident_incons 0 -> 58 on
    DeepSeek-V3). Restricted to axes an anchor or a merge derivation already decided: that is
    what keeps this from repeating the failure of the unrestricted carry, which moved ordinary
    value-matched labels around the residual stream and doubled flow_ambig.
    """
    n = 0
    for row, out in zip(rows, ordered):
        oid = row.get("op_id")
        marks = authoritative.setdefault(oid, set())
        cin_all, sin_all = row.get("input_shape") or [], out.get("input_shape") or []

        # ANY op whose output is concretely the same shape as one of its inputs keeps that
        # tensor's axes (sort, floor_divide, index-with-1D-index, the elementwise family). The
        # routed-slot list runs through a dozen of these between the reshape that creates it and
        # the expert matmul, and stopping at the reshape left every one of them on the old name.
        for si, (sout, cout) in enumerate(zip(out.get("output_shape") or [],
                                              row.get("output_shape") or [])):
            if not isinstance(cout, list) or not isinstance(sout, list):
                continue
            for bi, (cshape, sshape) in enumerate(zip(cin_all, sin_all)):
                if not isinstance(cshape, list) or list(cshape) != list(cout):
                    continue
                if not isinstance(sshape, list) or len(sshape) != len(sout):
                    continue
                for ax in range(len(sout)):
                    if ("input_shape", bi, ax) in marks                             and ("output_shape", si, ax) not in marks:
                        if sout[ax] != sshape[ax]:
                            sout[ax] = sshape[ax]
                            n += 1
                        marks.add(("output_shape", si, ax))
                break

        if row.get("op_type") not in _VIEW_FAMILY:
            continue
        if not cin_all or not sin_all or not isinstance(cin_all[0], list):
            continue
        cin, sin = cin_all[0], sin_all[0]
        if not isinstance(sin, list) or len(sin) != len(cin):
            continue
        for si, (sout, cout) in enumerate(zip(out.get("output_shape") or [],
                                              row.get("output_shape") or [])):
            if not isinstance(cout, list) or not isinstance(sout, list):
                continue
            # index of the input axis each 1:1 output axis came from
            src = {}
            i = j = 0
            while i < len(cin) and j < len(cout):
                if cin[i] == 1:
                    i += 1
                elif cout[j] == 1:
                    j += 1
                elif cin[i] == cout[j]:
                    src[j] = i
                    i, j = i + 1, j + 1
                else:
                    break
            for oj, ii in src.items():
                in_auth = ("input_shape", 0, ii) in marks
                out_auth = ("output_shape", si, oj) in marks
                if in_auth and not out_auth and sout[oj] != sin[ii]:
                    sout[oj] = sin[ii]
                    marks.add(("output_shape", si, oj))
                    n += 1
                elif out_auth and not in_auth and sin[ii] != sout[oj]:
                    sin[ii] = sout[oj]
                    marks.add(("input_shape", 0, ii))
                    n += 1
    return n


def _unname_loop_indices(rows: list[dict], ordered: list[dict], resolver=None) -> int:
    """Strip names from axes that are Python loop counters, not architecture dimensions.

    A chunked scan written as `for i in range(1, chunk_size)` slices `[..., i, i]` once per
    iteration, so ONE axis position emits a whole ladder of sizes -- 1, 2, 3, ... -- from the same
    module and op. Every one of them is a loop counter with no name, but value matching names each
    integer that happens to equal a config field, so Qwen3-Next came out with `n_kv` (2),
    `d_conv_lin` (4), `k` (10), `3*n_kv` (6), `n_h/n_kv` (8) ... scattered through the ladder while
    1, 3, 5, 7, 9 stayed bare. The names are arithmetically true and architecturally meaningless.

    The ladder itself is the evidence: an axis position whose observed sizes form a long run of
    near-consecutive integers is an iteration index. A real dimension takes ONE value per position
    within a phase. Measured across all 26 models, exactly one architecture matches (Qwen3-Next's
    gated delta net); nothing else comes close to the threshold (③ 라벨 검토 2026-08-09).
    """
    seen = collections.defaultdict(set)
    for row, out in zip(rows, ordered):
        mk = anchors_mod.module_key(row.get("module_path"))
        for fld in ("input_shape", "output_shape"):
            for si, conc in enumerate(row.get(fld) or []):
                if not isinstance(conc, list):
                    continue
                for ai, v in enumerate(conc):
                    if isinstance(v, int):
                        # NOT keyed by operand index. The loop slices the same axis of several
                        # operands, so splitting per operand cut one 65-value ladder into pieces
                        # that each missed the threshold -- Qwen3.5/3.6 kept `3*n_kv`, `n_h+1`,
                        # `n_h_lin_v+1` and friends on values (6, 8, 12, 25, 33, 48, 49) that are
                        # plainly rungs of that same ladder (2026-08-10).
                        # NOT keyed by field either. The same axis of the same op appears once as
                        # an input and once as an output; keying them apart stripped the name from
                        # the output and left the input carrying the old fabricated one, so a
                        # single `elementwise_add` read `n_kv` going in and `2` coming out
                        # (1,008 rows on Qwen3-Next, found by the new elementwise check 2026-08-10).
                        seen[(mk, row.get("op_type"), ai, len(conc))].add(v)
    ladders = set()
    for pos, vals in seen.items():
        if len(vals) >= 8 and (max(vals) - min(vals)) <= 2 * len(vals):
            ladders.add(pos)
    if not ladders:
        return 0
    changed = 0
    for row, out in zip(rows, ordered):
        mk = anchors_mod.module_key(row.get("module_path"))
        for fld in ("input_shape", "output_shape"):
            concs, labs = row.get(fld) or [], out.get(fld) or []
            for si, (conc, lab) in enumerate(zip(concs, labs)):
                if not isinstance(conc, list) or not isinstance(lab, list):
                    continue
                for ai, v in enumerate(conc):
                    if (mk, row.get("op_type"), ai, len(conc)) not in ladders:
                        continue
                    if isinstance(v, int) and str(lab[ai]) != str(v):
                        # Provenance must describe what we PUBLISH. The resolver counted this axis
                        # as a heuristic when it named it, and stripping the name afterwards left
                        # the tally saying "10,574 invented names" for a table that no longer has
                        # them -- and the review request, built from that tally, kept asking about
                        # labels nobody could find in the CSV (2026-08-10). Move the count to
                        # `bare`, which is what the axis now is.
                        _demote(resolver, row.get("module_path"), str(lab[ai]))
                        lab[ai] = str(v)
                        changed += 1
    return changed


# Ops that cannot change what an axis MEANS: elementwise arithmetic, and the copy family. The copy
# ops belong here for the same reason `_propagate_labels` treats them specially -- `clone` moving a
# loop rung left 504 rows reading a name going in and an integer coming out, exactly the elementwise
# case one op class over.
# 축을 재배치할 뿐 무엇도 계산하지 않는 op — 이름이 통과해야 하는 자리다.
_RESHAPE_FAMILY = ("view", "reshape", "_unsafe_view", "flatten", "squeeze", "unsqueeze",
                   "permute", "transpose", "t", "clone", "_to_copy", "contiguous", "alias",
                   "expand")

_SHAPE_PRESERVING = ("elementwise_add", "elementwise_mul", "elementwise_sub", "elementwise_div",
                     "maximum", "minimum", "where", "masked_fill", "rsqrt", "exp", "neg",
                     "silu", "gelu",
                     "clone", "_to_copy", "contiguous", "detach", "alias", "copy_")


def _unname_refilled_operands(rows: list[dict], ordered: list[dict], resolver=None) -> int:
    """Strip a name from an operand axis whose OUTPUT axis is a bare integer, in the same op.

    `_unname_loop_indices` runs before `_propagate_labels` and must: moving it after traded 1,008
    in/out mismatches for 43,000 dataflow ones (see the call site). But propagation is monotone --
    it puts names back -- so on Qwen3-Next a single `elementwise_add` read `n_kv` going in and `2`
    coming out, 1,008 times. Two names for one tensor inside one row.

    This runs LAST and settles only that contradiction, in the one direction that cannot invent
    anything:

      * shape-preserving elementwise ops only, and only an operand whose CONCRETE shape equals the
        output's -- same tensor, same layout, so the two must read alike;
      * only where the output axis is already a bare integer. Naming the output from the operand
        would be a guess; unnaming the operand asserts nothing that was not already published.

    Unnaming an operand only moves the contradiction one hop, so it runs to a FIXPOINT along the
    dependency graph: the producer of a tensor whose axis is now bare must drop that name too.
    Following tensor identity (`depends_on` + equal concrete shape) is what makes this safe where
    a value-based sweep is not -- inside `linear_attn`, 4 is a loop rung on one tensor and the real
    `d_conv_lin` on another, and only the edges tell them apart.
    """
    by_id = {r.get("op_id"): (r, o) for r, o in zip(rows, ordered)}
    changed = total = 0

    def _clear(row, labs, ai, v):
        _demote(resolver, row.get("module_path"), str(labs[ai]))
        labs[ai] = str(v)

    for _pass in range(6):
        changed = 0
        # (a) within one shape-preserving elementwise op: the output axis is already a bare
        #     integer, an operand of the identical concrete shape still carries a name.
        for row, out in zip(rows, ordered):
            if row.get("op_type") not in _SHAPE_PRESERVING:
                continue
            couts, souts = row.get("output_shape") or [], out.get("output_shape") or []
            cins, sins = row.get("input_shape") or [], out.get("input_shape") or []
            if len(couts) != 1 or not isinstance(couts[0], list) or len(souts) != 1:
                continue
            cout, sout = couts[0], souts[0]
            if not isinstance(sout, list) or len(sout) != len(cout):
                continue
            for i, cin in enumerate(cins):
                if not isinstance(cin, list) or list(cin) != list(cout):
                    continue
                if i >= len(sins) or not isinstance(sins[i], list) or len(sins[i]) != len(cout):
                    continue
                for ai, v in enumerate(cout):
                    if not isinstance(v, int) or str(sout[ai]) != str(v):
                        continue      # output axis carries a name -> nothing decided here
                    if str(sins[i][ai]) != str(v):
                        _clear(row, sins[i], ai, v)
                        changed += 1
        # (b) across the edge: consumer says bare, producer says a name, same tensor.
        for row, out in zip(rows, ordered):
            for dep in (row.get("depends_on") or []):
                prod = by_id.get(dep)
                if not prod:
                    continue
                p_row, p_out = prod
                for bi_c, bi_s in zip(row.get("input_shape") or [], out.get("input_shape") or []):
                    if not isinstance(bi_c, list) or not bi_c:
                        continue
                    for ao_c, ao_s in zip(p_row.get("output_shape") or [],
                                          p_out.get("output_shape") or []):
                        if not isinstance(ao_c, list) or ao_c != bi_c:
                            continue          # not the same tensor
                        for ai, v in enumerate(bi_c):
                            if not isinstance(v, int):
                                continue
                            mine, theirs = str(bi_s[ai]), str(ao_s[ai])
                            if mine == str(v) and theirs != str(v):
                                _clear(p_row, ao_s, ai, v)
                                changed += 1
                            elif theirs == str(v) and mine != str(v):
                                _clear(row, bi_s, ai, v)
                                changed += 1
        total += changed
        if not changed:
            break
    return total


def _demote(resolver, module_path: str | None, label: str) -> None:
    """Move one axis from its heuristic bucket to `bare` in the resolver's tally."""
    weak = getattr(resolver, "weak", None)
    stats = getattr(resolver, "stats", None)
    if weak is None or stats is None:
        return
    for kind, mp, lab in list(weak):
        if lab != label or mp != (module_path or "") or not kind.startswith("heur"):
            continue
        if weak[(kind, mp, lab)] <= 0:
            continue
        weak[(kind, mp, lab)] -= 1
        if weak[(kind, mp, lab)] == 0:
            del weak[(kind, mp, lab)]
        stats[kind] = max(0, stats.get(kind, 0) - 1)
        stats["bare"] = stats.get("bare", 0) + 1
        weak[("bare", mp, label)] += 1
        return


_SPLIT_OPS = {"split", "split_with_sizes", "chunk", "unbind"}
_PRODUCT_OPS = _CONTRACTING_OPS | {"grouped_matmul"}


def _weight_out_from_output(rows: list[dict], ordered: list[dict], authoritative: dict) -> int:
    """A weight's out_features axis and its op's output last axis are the same dimension.

    The mirror of _contraction_pin, which already ties the weight's IN axis to the activation's
    last axis. Without the out side, a fused parameter keeps whatever value matching gave it even
    after the activation chain has been settled by other evidence: OLMoE's expert weight stayed
    `[E, d_model, d_model]` while the matmul it feeds had already been corrected to produce
    `[k*T, 2*d_moe]` -- one tensor described two ways, one op apart.

    Only the trailing pair is eligible (`[..., out, in]`, or its transpose), and only when exactly
    one of the two axes carries the output's width -- where out == in numerically nothing but the
    axis order distinguishes them and the ordinary rendering stands.
    """
    changed = 0
    for row, out in zip(rows, ordered):
        if row.get("op_type") not in _PRODUCT_OPS:
            continue
        w, wl = row.get("weight_shape"), out.get("weight_shape")
        couts, souts = row.get("output_shape") or [], out.get("output_shape") or []
        if not w or not wl or len(w) != len(wl) or len(w) < 2 or not couts:
            continue
        co, so = couts[0], (souts[0] if souts else None)
        if not isinstance(co, list) or not isinstance(so, list) or not co or len(so) != len(co):
            continue
        val, lab = co[-1], str(so[-1])
        if not isinstance(val, int) or lab.isdigit():
            continue
        hits = [i for i in (len(w) - 2, len(w) - 1) if w[i] == val]
        if len(hits) != 1:
            continue
        i = hits[0]
        if wl[i] != lab:
            wl[i] = lab
            changed += 1
            authoritative.setdefault(row.get("op_id"), set()).add(("weight_shape", 0, i))
    return changed


def _merge_from_split(rows: list[dict], ordered: list[dict], authoritative: dict) -> int:
    """An axis that splits into n equally-sized, equally-named parts IS n times that name.

    The inverse of _split_from_authoritative, and evidence of the same kind: the factorisation
    comes from THIS op, not from a value that happens to match. A `split` states outright how the
    axis it consumed was composed -- if the two halves are both `d_moe`, the axis was `2*d_moe`,
    whatever else that number happens to equal.

    This is the only evidence available for a FUSED PARAMETER. `nn.Linear` declares which axis is
    out_features and which is in, so anchors can read the width off the module; `nn.Parameter`
    declares nothing -- `nn.Parameter(torch.empty(num_experts, 2 * intermediate, hidden))` is just
    a 3-D tensor by the time we see it. OLMoE and DeepSeek-V4-Flash size their experts so that
    2*intermediate == hidden (2*1024 == 2048, 2*2048 == 4096), and with the module silent, value
    matching had no way to prefer either name and rendered the gate+up projection `d_model`. The
    split one op later says what it really is:
        split [[k*T, d_model]] -> [[k*T, d_moe], [k*T, d_moe]]
    -- a self-contradiction inside a single row (found by review 2026-08-09).

    Marked authoritative so propagate() carries it to the op that produced the tensor.
    """
    changed = 0
    for row, out in zip(rows, ordered):
        if row.get("op_type") not in _SPLIT_OPS:
            continue
        cin_all, sin_all = row.get("input_shape") or [], out.get("input_shape") or []
        couts, souts = row.get("output_shape") or [], out.get("output_shape") or []
        if not cin_all or not isinstance(cin_all[0], list) or len(couts) < 2:
            continue
        cin, sin = cin_all[0], sin_all[0] if sin_all else None
        if not isinstance(sin, list) or len(sin) != len(cin):
            continue
        # every part identical in shape, and differing from the input in exactly one axis
        if any(not isinstance(c, list) or len(c) != len(cin) for c in couts):
            continue
        if any(c != couts[0] for c in couts[1:]):
            continue
        diff = [j for j in range(len(cin)) if cin[j] != couts[0][j]]
        if len(diff) != 1:
            continue
        j = diff[0]
        n = len(couts)
        if not isinstance(cin[j], int) or cin[j] != n * couts[0][j] or couts[0][j] < 2:
            continue
        parts = {str(s[j]) for s in souts if isinstance(s, list) and len(s) == len(cin)}
        if len(parts) != 1:
            continue
        lab = parts.pop()
        # a bare integer part carries no information, and a compound one would need to be
        # re-parenthesised to stay true (`n*(a+b)` is not `n*a+b`)
        if not lab.isidentifier():
            continue
        merged = f"{n}*{lab}"
        # A REGISTERED name for this axis outranks the inferred multiple. The n-equal-parts
        # inference is only sound when the parts really are the same quantity, and it cannot tell:
        # MLA splits `[.., n_h, d_nope+d_v]` into two 128-wide pieces that are a QK width and a
        # VALUE width, and because qk_nope_head_dim == v_head_dim the parts both rendered `d_nope`
        # -- so this rule "confirmed" the parent as `2*d_nope` and overwrote `d_nope+d_v`, which
        # rules/derived_dims.yaml already explains from the source. 195 rows across DeepSeek-V2/V3,
        # tiny-deepseek-v3 and three Kimi checkpoints (2026-08-12). When a registered formula
        # already names the axis, that formula is the better-attested account and stands.
        if sin[j] != merged and str(sin[j]) != merged and not _is_registered(sin[j]):
            sin[j] = merged
            changed += 1
            authoritative.setdefault(row.get("op_id"), set()).add(("input_shape", 0, j))
    return changed


def _split_from_registered_sum(rows: list[dict], ordered: list[dict], authoritative: dict,
                               table: dict) -> int:
    """A split whose input axis carries a registered `A+B` names its two pieces A and B, in order.

    The registered rule is not just an arithmetic fact, it is a transcription of the source: the
    reason `d_nope + d_v` exists in rules/derived_dims.yaml is
    `torch.split(kv_nope, [qk_nope_head_dim, v_head_dim], dim=-1)` -- so the ORDER of its operands
    says which piece is which. That is the only evidence there is when the two are the same size,
    which MLA makes them: DeepSeek-V2/V3, Kimi K2/K2.6/K2.7 all have
    qk_nope_head_dim == v_head_dim == 128, and both pieces rendered `d_nope`, carrying that name
    down the whole value path into `o_proj`. Same collision on the q/k side, where
    d_head == d_rope == 64 turned the RoPE piece into `d_head`.

    Only fires when both concrete sizes match the two symbols' values, so it cannot impose an
    order the numbers contradict.
    """
    changed = 0
    for row, out in zip(rows, ordered):
        if row.get("op_type") not in _SPLIT_OPS:
            continue
        cin_all, sin_all = row.get("input_shape") or [], out.get("input_shape") or []
        couts, souts = row.get("output_shape") or [], out.get("output_shape") or []
        if not cin_all or not isinstance(cin_all[0], list) or len(couts) != 2:
            continue
        cin, sin = cin_all[0], (sin_all[0] if sin_all else None)
        if not isinstance(sin, list) or len(sin) != len(cin):
            continue
        if any(not isinstance(c, list) or len(c) != len(cin) for c in couts):
            continue
        diff = [j for j in range(len(cin))
                if couts[0][j] != cin[j] or couts[1][j] != cin[j]]
        if len(diff) != 1:
            continue
        j = diff[0]
        a, b = couts[0][j], couts[1][j]
        if not (isinstance(a, int) and isinstance(b, int) and a + b == cin[j]):
            continue
        lab = str(sin[j])
        if not _is_registered(lab):
            continue
        parts = [p.strip() for p in lab.split("+")]
        if len(parts) != 2 or not all(p.isidentifier() for p in parts):
            continue
        if table.get(parts[0]) != a or table.get(parts[1]) != b:
            continue
        for si, (sout, want) in enumerate(zip(souts, parts)):
            if isinstance(sout, list) and len(sout) == len(cin) and str(sout[j]) != want:
                sout[j] = want
                changed += 1
                authoritative.setdefault(row.get("op_id"), set()).add(("output_shape", si, j))
    return changed


def _split_from_authoritative(rows: list[dict], ordered: list[dict], authoritative: dict,
                              table: dict) -> int:
    """Push an AUTHORITATIVE product label back onto the input axes the reshape merged.

    This is the narrow, evidence-backed version of the split rule that anchors._ENABLE_SPLIT
    could not make safe. That one looked for any anchor in the same block whose output width
    happened to factor the same way -- "product matches" is a coincidence detector. Here the
    factorisation must come from THIS op's own output axis, and that axis must already be
    authoritative, i.e. decided by the module that consumes the tensor rather than by value
    matching. Each factor is then checked against the concrete width of the axis it lands on.

    What it fixes: DeepSeek MLA. `view [B,T,n_h,d_nope] -> [B,T,n_h*d_v]` merges the attention
    result's head layout; o_proj's declared in_features makes the OUTPUT authoritative as
    `n_h*d_v`, but the input's last axis had lost `d_v` to `d_nope` (both 128), so one op gave two
    accounts of one tensor. The reshape preserves axis order, so the factors map positionally.
    """
    changed = 0
    for row, out in zip(rows, ordered):
        if row.get("op_type") not in _VIEW_FAMILY:
            continue
        cin_all, sin_all = row.get("input_shape") or [], out.get("input_shape") or []
        if not cin_all or not sin_all or not isinstance(cin_all[0], list):
            continue
        cin, sin = cin_all[0], sin_all[0]
        if not isinstance(sin, list) or len(sin) != len(cin):
            continue
        marks = authoritative.get(row.get("op_id")) or set()
        if not marks:
            continue
        for si, (sout, cout) in enumerate(zip(out.get("output_shape") or [],
                                              row.get("output_shape") or [])):
            if not isinstance(cout, list) or not isinstance(sout, list):
                continue
            for oj, lab in enumerate(sout):
                if ("output_shape", si, oj) not in marks:
                    continue
                factors = str(lab).split("*")
                if len(factors) != 2 or any(f.isdigit() or not f.isidentifier() for f in factors):
                    continue
                vals = [table.get(f) for f in factors]
                if any(v is None for v in vals):
                    continue
                # the contiguous input run whose product is this output axis, same length
                target = cout[oj]
                for i in range(len(cin) - 1):
                    if cin[i] * cin[i + 1] != target:
                        continue
                    if [cin[i], cin[i + 1]] != vals:
                        continue        # factors must sit on axes of matching width, in order
                    for k, f in ((i, factors[0]), (i + 1, factors[1])):
                        if sin[k] != f:
                            sin[k] = f
                            changed += 1
                        authoritative.setdefault(row.get("op_id"), set()).add(
                            ("input_shape", 0, k))
                    break
    return changed


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
        # A bias must not disqualify the module here either -- see anchors.weight_param.
        wp, w = anchors_mod.weight_param(row.get("params")), row.get("weight_shape")
        if not wp or not w or wp in canon:
            continue
        params = [wp]
        mp = row.get("module_path")
        in_shapes = row.get("input_shape") or []
        pin = _contraction_pin(row, in_shapes, [resolver(s, mp) for s in in_shapes])
        if pin:
            canon[params[0]] = (tuple(w), resolver(w, mp, is_weight=True, pin=pin))
    return canon


def _ordered_row(row: dict, resolver, hier_cols: list, canon: dict | None = None,
                 hints: dict | None = None) -> dict:
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
    # Per-axis sequence-length verdicts measured across the two phases (src/tdep.py). They act
    # exactly like `is_weight`: a filter on which names are admissible for THIS axis.
    def _hint(field, si):
        if not hints:
            return None
        h = {ax: v for (f, i, ax), v in hints.items() if f == field and i == si}
        return h or None

    out["input_shape"] = [resolver(s, mp, t_dep=_hint("input_shape", i))
                          for i, s in enumerate(in_shapes)]
    # is_weight=True enforces "a static parameter cannot depend on runtime seq len" -- see
    # build_resolver.dim(). Without it, a weight axis whose size coincides with T (or a T
    # product) rendered as a sequence-dependent symbol, which is physically impossible.
    w = row.get("weight_shape")
    params = row.get("params") or []
    # A bias must not disqualify the lookup (see anchors.weight_param) -- with it, every biased
    # projection fell back to the ordinary rendering and lost the canonical labelling.
    _wp_name = anchors_mod.weight_param(params)
    hit = canon.get(_wp_name) if canon else None
    if hit and w and tuple(w) == hit[0]:
        out["weight_shape"] = list(hit[1])   # one parameter -> one labelling, everywhere
    else:
        out["weight_shape"] = resolver(w, mp, is_weight=True,
                                       pin=_contraction_pin(row, in_shapes, out["input_shape"]))
    # Position is a property of the operand list, so it is derived from the SYMBOLIC shapes that
    # were just written -- the two columns are then consistent by construction even where
    # symbolization collapsed two distinct concrete dims onto one name.
    # The same parameter also appears as an OPERAND (a bare `aten.t` takes it as input, and a
    # matmul takes the transposed view). Give that operand the canonical labelling too, or the
    # stored weight keeps whatever value matching produced: on any model with
    # d_model == n_h*d_head, Llama-3.1-8B's q_proj input read `[n_h*d_head, n_h*d_head]` while
    # its own weight_shape column said `[n_h*d_head, d_model]` -- one parameter, two accounts.
    #
    # Accepted with Zamba2 flow_ambig 0 -> 18 (2026-08-06), inspected first. Its shared q/k/v_proj
    # is Linear(d_attn=4096, n_h*d_head=4096), so nothing but the axis ORDER can tell the two
    # apart, and the order is not in doubt: nn.Linear stores [out, in] and aten.t yields [in, out].
    # Before, the `t` read `[d_attn, n_h*d_head] -> [d_attn, d_attn]` -- the transposed order on
    # the stored weight, and an output carrying no information. After: `[n_h*d_head, d_attn] ->
    # [d_attn, n_h*d_head]`, both correct. The 18 are the boundary with consumers that value
    # matching still names the old way.
    if hit:
        for _i, (_c, _s) in enumerate(zip(in_shapes, out["input_shape"])):
            if isinstance(_c, list) and tuple(_c) == hit[0] and len(_s) == len(hit[1]):
                out["input_shape"][_i] = list(hit[1])

    wp = row.get("weight_pos")
    out["weight_pos"] = derive_weight_pos(out["weight_shape"], out["input_shape"],
                                          out["op_type"]) if wp is None else wp
    out["output_shape"] = [resolver(s, mp, t_dep=_hint("output_shape", i))
                           for i, s in enumerate(row.get("output_shape") or [])]
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


def write_outputs(model_dir: str, phase: str, rows: list[dict], resolver, tags: dict | None = None, tdep_map: dict | None = None,
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
    # label_provenance must describe the rows we PUBLISH. _canonical_weight_labels renders each
    # weight an extra time to find its authoritative spelling, and those throwaway renders were
    # landing in resolver.stats -- when the bias fix let biased modules into this pass, gpt2-xl's
    # heuristic count jumped 5,328 -> 5,812 with no label actually changing. Discard the tally
    # from this pass; the rendering loop below is the one that counts.
    _stats_pre = collections.Counter(getattr(resolver, "stats", {}))
    canon = _canonical_weight_labels(rows, resolver)
    if hasattr(resolver, "stats"):
        resolver.stats.clear()
        resolver.stats.update(_stats_pre)
    ordered = [_ordered_row(row, resolver, hier_cols, canon,
                           tdep.axis_hints(tdep_map, phase, row.get("op_id")))
               for row in rows]  # symbolic, ordered

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
        # _apply_merge_derivation + _carry_authoritative are DELIBERATELY NOT CALLED. Measured
        # 2026-08-06: together they take reshape_incons 773 -> 8, and the labels they write are
        # right (GLM-4.5-Air's routed-slot list really is k*T=128, not E=128) -- but flow_ambig
        # rises 2,015 -> 2,187 because the MoE routing region contains ops where the tensor
        # genuinely changes identity (`sort` emits new tensors, `index` gathers), and value
        # matching re-labels those independently. Propagation only moves the frontier; extending
        # it to same-shape outputs moved it again. A partially-corrected region is worse to read
        # than a consistently-wrong one, so this needs the fix at the source -- see 01-main.md
        # §10.2 for why a registered rule cannot win today (`E` is scoped, so it outranks any
        # derived formula) and what would settle it.
        # an anchor names a TENSOR, so every op that sees that tensor must read the same name
        anchors_mod.propagate(rows, ordered, authoritative)
        # An anchor's verdict has to cross reshapes to reach the axes it explains: o_proj's
        # declared in_features marks `n_h*d_v` on the matmul, but two `view`s sit between that and
        # the attention result whose head layout we want to name. propagate() crosses op
        # boundaries, _carry_authoritative crosses a reshape INSIDE one op, and the split rule
        # then factors the product back onto the axes the reshape merged. Alternated to a fixed
        # point. Only axes an anchor decided ever move -- the unrestricted carry was measured and
        # rejected (see _carry_reshape_labels).
        for _ in range(4):
            moved = _carry_authoritative(rows, ordered, authoritative)
            moved += _merge_from_split(rows, ordered, authoritative)
            moved += _weight_out_from_output(rows, ordered, authoritative)
            moved += _split_from_authoritative(rows, ordered, authoritative,
                                               getattr(resolver, "table", {}) or {})
            moved += anchors_mod.propagate(rows, ordered, authoritative)
            if not moved:
                break
        # A loop counter is not a dimension, and no evidence above can make it one.
        #
        # Placed BEFORE `_propagate_labels`, deliberately. Moving it after was tried (2026-08-10)
        # to stop propagation from refilling the stripped axes, and it traded 1,008 in/out
        # mismatches for 43,000 dataflow mismatches: the ops AROUND the ladder still carry the
        # fabricated names, so an op reading `2` next to one reading `n_kv` is inconsistent either
        # way. Fixing it properly means stripping every op that touches those tensors, which a
        # value-based sweep cannot do safely -- inside `linear_attn`, 4 is both a loop rung and the
        # real `d_conv_lin`. Recorded as an open finding instead of guessed at.
        _unname_loop_indices(rows, ordered, resolver)
        # Last: a loop counter is not a dimension, and no evidence above can make it one.

    # _carry_reshape_labels is DELIBERATELY NOT CALLED -- see its docstring. Measured 2026-08-05:
    # it corrects the view itself but doubles flow_ambig fleet-wide (Llama-3.1-70B 160 -> 400,
    # 405B 504 -> 882), because renaming one end of a tensor leaves every consumer on the old name
    # and _propagate_labels is monotone by design. The mislabel is not local to the reshape: inside
    # a `self_attn` scope the head formula outranks the plain symbol for EVERY op there, so the fix
    # belongs in that priority decision, not in a post-pass. Kept as an auditor (reshape_incons).
    _propagate_labels(rows, ordered)
    # ...and undo the one thing propagation is known to get wrong: refilling an axis that the
    # loop-counter pass had just emptied, so an op reads a name going in and an integer coming out.
    if resolver is not None:
        _unname_refilled_operands(rows, ordered, resolver)
    # 한 행 안에서 같은 가중치가 두 이름을 갖지 않게 한다. 반드시 마지막에 — 그 앞의 어떤 패스가
    # 피연산자 쪽을 고치더라도 저장 형태가 따라온다.
    # 잔차 스트림이 먼저 자리를 잡고 -> 행렬곱 합성이 그걸 가중치까지 나른다 -> 가중치와
    # 피연산자가 한 이름으로 합의한다. 순서가 중요하다: 합성이 먼저면 나를 이름이 없다.
    _tbl = getattr(resolver, "table", {}) or {}
    _resid_stream_wins(rows, ordered, _tbl.get("d_model"))
    _gather_keeps_features(rows, ordered)
    _squeeze_view_keeps_names(rows, ordered)
    _transpose_swaps_names(rows, ordered)
    _matmul_compose_enforce(rows, ordered)
    _weight_agrees_with_operand(rows, ordered)
    _resync_param_labels(rows, ordered)
    _weight_agrees_with_operand(rows, ordered)
    # 그리고 소비자를 생산자에 맞춘다. 위 패스들이 텐서를 만든 자리의 이름을 바로잡았으므로,
    # 그것을 그 텐서의 모든 소비자에게 전한다 -- 지금까지 이 방향이 비어 있었다.
    _unify_axis_classes(rows, ordered)
    _weight_agrees_with_operand(rows, ordered)
    # 위 패스들이 새로 정한 이름을 이웃이 아직 정수로 들고 있을 수 있다. 단조 채움을 한 번 더
    # 돌려 그 자리만 메우고(이름을 덮어쓰지 않는다), 그 과정에서 되채워진 루프 계단을 다시 뗀다.
    _propagate_labels(rows, ordered)
    # ...and once more AFTER that propagation. The rank-changing view is the one edge propagation
    # cannot see, so it has to run on both sides of it: the first call catches views whose input
    # was already named, this one catches the ones whose input only got its name just now
    # (Hunyuan's `_unsafe_view([T, d_moe]) -> [B, T, 3072]` was in the second group -- the matmul
    # above it is what names `d_moe`, and that happens in the propagation immediately before).
    # Safe to repeat: the pass only fills bare integers and never overwrites a name, and
    # _unname_refilled_operands below still gets the last word on loop counters.
    _squeeze_view_keeps_names(rows, ordered)
    _transpose_swaps_names(rows, ordered)
    # ...and carry the name the view just recovered to everything downstream of it. Without this
    # the view alone was named while its `silu`/`mul`/`down_proj` chain stayed bare, and
    # _unname_refilled_operands -- which walks that disagreement backwards to a fixpoint -- simply
    # stripped the name again. Measured on Hunyuan: 576 axes named, then all 576 taken back.
    _propagate_labels(rows, ordered)
    if resolver is not None:
        _unname_refilled_operands(rows, ordered, resolver)
    # A parameter has no batch axis -- that is an invariant, not an inference, so it gets the last
    # word. The three calls above are all followed by `_propagate_labels`, which is monotone and
    # therefore treats the `1` this pass just wrote as an EMPTY slot and refills it from the
    # activation on the other side of the op: `B`. Falcon-H1 multiplies its MuP vector (a real
    # buffer, shape `[1, 1, 2*d_inner+2*n_g*d_state+n_h_ssm]`) against `[B, 1, ...]`, and all 44
    # layers shipped the buffer's own operand reading `B` while its stored form read `1` -- the
    # two spellings of one tensor disagreeing, which is exactly what this pass exists to stop.
    # Found by the blind onboarding test, 2026-08-15.
    _weight_agrees_with_operand(rows, ordered)

    # LAST, after every inference: the ④-layer verdicts. A reader with the source open sometimes
    # knows what no rule can decide from a number, and this is where that knowledge lands in the
    # published tables instead of stopping at review_findings.json. Each override carries a source
    # citation and the size it expects, and the gate fails on one that matched nothing.
    # See src/label_overrides.py and review/05-overrides.md.
    # Axes where two symbols held the same value and the winner was decided by convention rather
    # than evidence. The label may be right; nothing here knows that. Written out so the ④-layer
    # review can be aimed at exactly these instead of re-reading everything.
    _ties = getattr(resolver, "ties", None)
    if _ties:
        folded = collections.Counter()
        for (mp, val, cands), cnt in _ties.items():
            folded[(anchors_mod.module_key(mp) or "(root)", val, cands)] += cnt
        with open(os.path.join(full_dir, "ambiguous.json"), "w", encoding="utf-8") as f:
            # `chosen` は the label that actually shipped. It was written as
            # `c[0] if len(c) == 1 else None` -- but an entry only exists when there are TWO or
            # more candidates, so the field was None in every row of every model and told a
            # reviewer nothing about what the table says (outside review, 2026-08-12).
            # `_pick` returns ms[0] of the value-filtered, context-ordered list, which is what
            # `candidates` is sorted from -- so record the resolver's own answer instead.
            json.dump([{"module": mk, "value": v, "candidates": list(c), "axes": n,
                        "chosen": (resolver.label_of(v, mk)
                                   if hasattr(resolver, "label_of") else None)}
                       for (mk, v, c), n in folded.most_common()], f,
                      ensure_ascii=False, indent=1)

    _settled = set()
    _model_name = os.path.basename(os.path.normpath(model_dir))
    ov_report = label_overrides.apply(rows, ordered, _model_name,
                                      cfg=getattr(resolver, "cfg", None), touched=_settled)
    # 확인 기록(고칠 게 없다는 판정)도 그 축을 종결시킨다. 라벨은 건드리지 않는다.
    cf_report = label_overrides.confirm(rows, ordered, _model_name, touched=_settled)
    cf_path = os.path.join(full_dir, "label_confirmed.json")
    if cf_report:
        prev_cf = {}
        if os.path.exists(cf_path) and phase != "prefill":
            with open(cf_path, encoding="utf-8") as f:
                prev_cf = {o.get("id"): o.get("matched", 0) for o in (json.load(f) or [])}
        for o in cf_report:
            o["matched"] += prev_cf.get(o.get("id"), 0)
        with open(cf_path, "w", encoding="utf-8") as f:
            json.dump(cf_report, f, ensure_ascii=False, indent=1)
    elif os.path.exists(cf_path):
        os.remove(cf_path)
    if not ov_report:
        # 이 모델의 교정이 **하나도 없다면** 옛 보고서를 지운다. 남겨 두면 이미 삭제한 항목을
        # 계속 주장하는 파일이 되고, 게이트도 검토자도 그 거짓을 읽는다 -- 실제로 이번 세션에
        # 지운 교정 3건의 보고서가 그대로 남아 있었다(2026-08-14, 외부 검토를 따라가다 발견).
        _stale = os.path.join(full_dir, "label_overrides.json")
        if os.path.exists(_stale):
            os.remove(_stale)
    if ov_report:
        # ACCUMULATE across phases. write_outputs runs once per phase and this file was being
        # overwritten each time, so an override that only applies to prefill (a chunk-scan axis
        # that decode does not reach) read as "matched nothing" from the decode pass and the gate
        # called it a stale claim. The question is whether it fired AT ALL.
        ov_path = os.path.join(full_dir, "label_overrides.json")
        prev = {}
        if os.path.exists(ov_path) and phase != "prefill":
            with open(ov_path, encoding="utf-8") as f:
                # 항목 신원은 `id` -- 매칭에 쓰이는 선택자 전부를 담는다. 예전에는
                # (module, from, to) 셋뿐이라, 그 셋이 같고 앵커만 다른 두 교정이 서로를
                # 덮어써 prefill 발화 기록이 decode 누적에서 사라졌다(override_dead 오탐).
                prev = {o.get("id", (o["module"], o["from"], o["to"])): o.get("applied", 0)
                        for o in (json.load(f) or [])}
        for o in ov_report:
            o["applied"] += prev.get(o.get("id", (o["module"], o["from"], o["to"])), 0)
        with open(ov_path, "w", encoding="utf-8") as f:
            json.dump(ov_report, f, ensure_ascii=False, indent=1)

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
    # 축 등가류 감사. 지금은 **보고만** 한다 -- 이름을 등가류 단위로 결정하는 것은 다음 단계이고,
    # 그 전에 "무엇이 어긋나 있는지"를 기준선으로 박아 두어야 개선이 측정된다.
    # src/axis_classes.py 참고. 발행된 jsonl 을 다시 읽으므로 중간 상태가 아니라 산출물을 잰다.
    try:
        axis_classes.write_audit(model_dir, phase)
        # 규칙이 끝내지 못한 축을 ④층으로 넘긴다. 정규식을 더 비트는 대신 "여기까지"라고
        # 선언하고, 소스를 읽어야 풀리는 것은 소스를 읽는 층에 맡긴다.
        axis_classes.write_unsettled(model_dir, phase, ordered,
                                     {r.get("op_id"): r for r in rows},
                                     getattr(resolver, "ties", None),
                                     getattr(resolver, "weak", None), settled=_settled)
    except Exception:
        pass
    return csv_path
