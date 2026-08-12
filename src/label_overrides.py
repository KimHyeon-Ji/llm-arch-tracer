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

Scope, deliberately: this renames a label under a module. It does NOT follow a tensor along the
dataflow, so a collision that is only distinguishable by where the tensor came from (MLA's
`d_nope` vs `d_v`, both 128, both inside `self_attn`, both on `[B, n_h, T, ·]`) still cannot be
expressed. Those stay `open` with their source line, which is the truthful outcome.
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
        })

    from anchors import module_key
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
            for fld in ("input_shape", "output_shape", "weight_shape"):
                cvals, svals = row.get(fld), out.get(fld)
                if cvals is None or svals is None:
                    continue
                pairs = ([(cvals, svals)] if fld == "weight_shape"
                         else list(zip(cvals, svals)))
                for cv, sv in pairs:
                    if not isinstance(cv, list) or not isinstance(sv, list) or len(cv) != len(sv):
                        continue
                    for i, (c, s) in enumerate(zip(cv, sv)):
                        if str(s) == frm and isinstance(c, int) and c == want:
                            sv[i] = to
                            p["n"] += 1
    return [{"from": p["spec"]["from"], "to": p["spec"]["to"], "module": p["spec"]["module"],
             "expect": p["spec"]["expect"], "source": p["spec"].get("source", ""),
             "applied": p["n"]} for p in prepared]
