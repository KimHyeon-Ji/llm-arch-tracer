"""Apply the ④-layer verdicts to the rendered tables.

WHY THIS EXISTS
---------------
Layers ①-③ decide a label from the rules. Layer ④ -- a reader with the source open -- sometimes
knows better, and until now that knowledge stopped at `review_findings.json`: the summary card said
"지금 렌더 / 소스가 말하는 것", while `full/*.csv` kept the wrong name. A judgement nobody can act
on is half a judgement.

The obvious fix -- change the rule so the labeller gets it right -- was tried twice and reverted
both times. It fails for the same reason each time: a value collision is not local. Renaming one
op leaves every neighbour on the old name, and the dataflow checks light up (DeepSeek MLA:
reshape_incons 61 -> 122, flow_ambig 0 -> 122). The rules decide per axis from a number; when two
config fields hold the same number there is no number to decide from.

So this does not re-derive anything. It REWRITES a specific label, everywhere it occurs under a
declared module, after all inference is done. That is safe precisely because it is not inference.

WHAT KEEPS IT HONEST
--------------------
An override is a claim, and every claim here has to pay for itself:

  * `source` is REQUIRED and must name the file and line it came from. No citation, no override.
  * `expect` is REQUIRED: the concrete size the axis must have. If the real tensor is not that
    size, the override does not fire -- it cannot silently paste a name onto the wrong axis.
  * The gate FAILS on an override that matched nothing. A claim that no longer applies (the model
    was re-traced, the rules improved, the label is already right) is a stale claim, and stale
    claims are how a table starts lying.
  * `layer_types` narrows to a block kind, for hybrid stacks where the same module name holds a
    Mamba block in one layer and attention in the next.
  * `spread: class` makes the rename follow the TENSOR instead of stopping at the module. See below.
  * `axis` narrows to one axis POSITION (negative counts from the right) and `rank` to shapes of
    one length. Needed when the same number means two different things inside one shape and a
    blanket rename would destroy the half that is already right: DeepSeek-V4-Pro's CSA compressor
    builds `new_zeros((batch, n_windows, 2 * ratio, head_dim))` where n_windows == head_dim == 512,
    so `[B, ·, 2*m_csa, ·]` needs `T/m_csa` at axis 1 and `d_head` at axis -1 -- one rename applied
    to both would collapse them onto a single name. `rank` separates `[B, n_windows, head_dim]`
    from `[B, 1, n_windows, head_dim]`, where the window sits at a different index.

CLASS MODE (`spread: class`)
---------------------------
An override without it renames labels **under a module** and stops at the module boundary, so the
same tensor keeps the old name at the op next door and the dataflow check lights up. That is the
single reason 58 source-confirmed verdicts could not be applied.

With `spread: class` the entry instead names the **axis equivalence class** (`src/axis_classes.py`)
that the matched axis belongs to, and every site in that class -- producer output, every consumer's
operand, however many modules away -- takes the same name. One tensor, one name, by construction
rather than by a chain of patch passes.

Still gated the same way: `expect` must match the measured width at the site that anchors the
class, `source` is required, and the gate fails an entry that matched nothing. The class is built
from concrete shapes only and conservatively (ambiguous edges are not joined), so a class can be
too SMALL -- which just means the override reaches less far -- but not wrong.

What it still cannot express: a collision distinguishable only by where the tensor came from when
the two candidates live in the SAME class (MLA's `d_nope` vs `d_v` on the same axis of the same
tensor). Those stay `open` with their source line, which is the truthful outcome.
"""
import os
import re

import yaml

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "rules", "label_overrides.yaml")
_CACHE = None


def load(path: str = _PATH) -> list:
    global _CACHE
    if _CACHE is None:
        if not os.path.exists(path):
            _CACHE = []
        else:
            with open(path, encoding="utf-8") as f:
                _CACHE = (yaml.safe_load(f) or {}).get("overrides") or []
    return _CACHE


def for_model(model_dir_name: str, path: str = _PATH) -> list:
    return [o for o in load(path) if o.get("model") == model_dir_name]


_LAYER_IDX = re.compile(r"\.(?:layers|h|blocks|block|layer)\.(\d+)(?:\.|$)")

# An override is hand-written, so it can be wrong in ways a rule cannot: it names an axis by
# POSITION, and a position that holds a sequence axis in one tensor holds a static width in
# another. Writing `T/m_csa` at axis 1 of DeepSeek-V4-Pro's compressor was right for every
# activation and wrong for `position_bias`, an `nn.Parameter(compress_rate, head_dim)` whose
# axis 1 is head_dim -- 31 weight axes came out claiming a static parameter is sized by the
# runtime sequence length. The gate's weight_T invariant caught it, but a layer that can only
# be corrected after the fact is a layer that will ship the error once. Refuse it here instead:
# no override may put a T-derived name into a weight, whatever the entry says.
_MENTIONS_T = re.compile(r"\bT\b")


def _schedule(cfg):
    for f in ("layers_block_type", "layer_types"):
        v = getattr(cfg, f, None)
        if isinstance(v, (list, tuple)) and v:
            return [str(x) for x in v]
    return None


def apply(rows: list, ordered: list, model_dir_name: str, cfg=None, path: str = _PATH) -> list:
    """Rewrite labels in `ordered` per the declared overrides. Returns one report dict each.

    `rows` carries the CONCRETE shapes and `ordered` the rendered ones, index-aligned -- the same
    pairing every other pass in build_table uses. The concrete side is what `expect` is checked
    against, so an override can never fire on an axis of the wrong size.
    """
    ovs = for_model(model_dir_name, path)
    if not ovs:
        return []
    sched = _schedule(cfg) if cfg is not None else None
    prepared = []
    for o in ovs:
        prepared.append({
            "spec": o,
            "rx": re.compile(o["module"]),
            "kinds": set(o.get("layer_types") or ()),
            "n": 0,
            "vetoed": 0,
        })

    from anchors import module_key
    # 등가류 모드가 하나라도 있으면 축 등가류를 한 번 만들어 둔다. 구체 shape 으로만 잇고
    # 모호하면 잇지 않으므로(src/axis_classes) 클래스가 작을 수는 있어도 틀리지는 않는다.
    uf = idx = None
    if any(p["spec"].get("spread") == "class" for p in prepared):
        import axis_classes
        conc = {r.get("op_id"): r for r in rows}
        uf = axis_classes.build(rows, conc)
        idx = {}                       # root -> [(ordered_row, field, shape_index, axis)]
        for row, out in zip(rows, ordered):
            oid = row.get("op_id")
            for fld, tag in (("input_shape", "i"), ("output_shape", "o")):
                for si, sh in enumerate(out.get(fld) or []):
                    if not isinstance(sh, list):
                        continue
                    for a in range(len(sh)):
                        idx.setdefault(uf.find((oid, tag, si, a)), []).append((out, fld, si, a))

    def _spread(p, row, out, fld, si, axis, to):
        """이 축이 속한 등가류의 모든 자리에 같은 이름을 쓴다. 바뀐 자리 수를 반환."""
        if uf is None or fld == "weight_shape":
            return 0
        tag = "i" if fld == "input_shape" else "o"
        n = 0
        for o2, f2, s2, a2 in idx.get(uf.find((row.get("op_id"), tag, si, axis))) or []:
            sh = (o2.get(f2) or [None] * (s2 + 1))[s2]
            if isinstance(sh, list) and a2 < len(sh) and str(sh[a2]) != to:
                sh[a2] = to
                n += 1
        return n

    for row, out in zip(rows, ordered):
        mk = module_key(row.get("module_path")) or "(root)"
        kind = None
        if sched:
            m = _LAYER_IDX.search(row.get("module_path") or "")
            if m and 0 <= int(m.group(1)) < len(sched):
                kind = sched[int(m.group(1))]
        for p in prepared:
            if not p["rx"].search(mk):
                continue
            if p["kinds"] and kind not in p["kinds"]:
                continue
            frm, to, want = str(p["spec"]["from"]), str(p["spec"]["to"]), p["spec"]["expect"]
            ax, rank = p["spec"].get("axis"), p["spec"].get("rank")
            for fld in ("input_shape", "output_shape", "weight_shape"):
                cvals, svals = row.get(fld), out.get(fld)
                if cvals is None or svals is None:
                    continue
                if fld == "weight_shape" and _MENTIONS_T.search(to):
                    p["vetoed"] += 1
                    continue
                pairs = ([(cvals, svals)] if fld == "weight_shape"
                         else list(zip(cvals, svals)))
                # The same parameter appears twice per op: once as `weight_shape` and once as the
                # operand at `weight_pos` inside `input_shape`. Vetoing only the former left
                # DeepSeek-V4-Pro's `position_bias` operand renamed and its weight not, which is
                # the "one tensor, two names" defect the gate calls weight_operand -- 31 rows.
                if _MENTIONS_T.search(to) and isinstance(row.get("weight_pos"), int):
                    wp = row["weight_pos"]
                    if fld == "input_shape" and 0 <= wp < len(pairs):
                        p["vetoed"] += 1
                        pairs = [x for i, x in enumerate(pairs) if i != wp]
                for cv, sv in pairs:
                    if not isinstance(cv, list) or not isinstance(sv, list) or len(cv) != len(sv):
                        continue
                    if rank is not None and len(sv) not in (
                            rank if isinstance(rank, (list, tuple)) else (rank,)):
                        continue
                    want_i = None if ax is None else (ax if ax >= 0 else len(sv) + ax)
                    for i, (c, s) in enumerate(zip(cv, sv)):
                        if want_i is not None and i != want_i:
                            continue
                        if str(s) == frm and isinstance(c, int) and c == want:
                            sv[i] = to
                            p["n"] += 1
                            if p["spec"].get("spread") == "class":
                                # 이 축이 속한 텐서 전체를 같은 이름으로. 모듈 경계에서
                                # 멈추지 않는 유일한 경로다.
                                si = pairs.index((cv, sv)) if (cv, sv) in pairs else 0
                                p["n"] += _spread(p, row, out, fld, si, i, to)
    return [{"from": p["spec"]["from"], "to": p["spec"]["to"], "module": p["spec"]["module"],
             "expect": p["spec"]["expect"], "source": p["spec"].get("source", ""),
             "applied": p["n"], "vetoed": p["vetoed"]} for p in prepared]
