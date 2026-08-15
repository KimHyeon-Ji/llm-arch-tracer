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
  * `shape` pins the entry to one exact rendered shape, which is how an ANCHOR is made unique when
    a module holds the same name at the same width in several different axis classes.
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

`from` may equal `to` -- ANCHOR MODE. Then the entry does not rename the site it matches; it
uses that site only to identify the class, and pins every OTHER site in the class to that name.
Needed when the correct name is already present somewhere and the wrong one sits outside the
module that could be matched safely: DeepSeek-V4's `o_a_proj` already reads `g_o` throughout,
while the `_unsafe_view` one level up in `self_attn` reads `T/m_hca` -- and `T/m_hca` is a REAL
axis elsewhere in `self_attn`, so matching on the wrong name there would destroy it.

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


# 항목의 **신원**. 보고서와 phase 누적이 이걸로 항목을 구분한다.
#
# 예전에는 `(module, from, to)` 셋만 썼다. 앵커 선택자가 생기면서 그 셋이 같고 나머지가 다른
# 항목이 생길 수 있게 됐고 -- 같은 `self_attn` / `d_nope` -> `d_v` 인데 앵커가 다른 두 교정 --
# 그러면 prefill 에서 발화한 기록을 decode 누적이 덮어써서 실제로 동작한 교정이
# `override_dead` 로 걸린다. 외부 검토(Codex, 2026-08-14)가 코드로 짚었다.
#
# 매칭에 영향을 주는 필드를 **전부** 넣는다. 하나라도 빠지면 두 항목이 같은 것으로 보인다.
_ID_FIELDS = ("module", "from", "to", "expect", "spread", "axis", "rank", "shape",
              "field", "shape_index", "op_type", "nth", "layer_types")


def _report_id(spec: dict) -> dict:
    import json as _json
    return {"id": _json.dumps([spec.get(k) for k in _ID_FIELDS],
                              ensure_ascii=False, sort_keys=True)}


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
    import axis_classes as _ac
    ordinals = _ac.op_ordinals(rows)
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
            spec = p["spec"]
            # 앵커를 정확히 짚는 선택자. `shape`/`axis` 만으로는 부족하다 -- Kimi 의
            # `self_attn` 에서 `[B, n_h, T, d_nope]` 의 축 3 은 **366개 등가류**에 걸쳐 있고
            # (q 의 q_pass, KV 의 k_nope, value_states …), 그중 하나만 고쳐야 한다.
            # 외부 검토(Codex, 2026-08-14)가 실측으로 짚었다: 976 자리 / 366 등가류.
            if spec.get("op_type") and row.get("op_type") != spec["op_type"]:
                continue
            if spec.get("nth") is not None and ordinals.get(row.get("op_id")) != spec["nth"]:
                continue
            frm, to, want = str(spec["from"]), str(spec["to"]), spec["expect"]
            ax, rank = p["spec"].get("axis"), p["spec"].get("rank")
            shape = spec.get("shape")
            want_field, want_si = spec.get("field"), spec.get("shape_index")
            for fld in ("input_shape", "output_shape", "weight_shape"):
                if want_field and fld != {"i": "input_shape", "o": "output_shape",
                                          "w": "weight_shape"}.get(want_field, want_field):
                    continue
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
                for _si, (cv, sv) in enumerate(pairs):
                    if want_si is not None and _si != want_si:
                        continue
                    if not isinstance(cv, list) or not isinstance(sv, list) or len(cv) != len(sv):
                        continue
                    if rank is not None and len(sv) not in (
                            rank if isinstance(rank, (list, tuple)) else (rank,)):
                        continue
                    # `shape`: 렌더된 shape 전체가 이것과 같을 때만. 등가류를 유일하게 지목해야
                    # 할 때 쓴다 -- 한 모듈 안에 같은 이름·같은 크기의 축이 여러 등가류에 걸쳐
                    # 있으면 module/from/expect 만으로는 어느 것인지 못 가른다(Kimi 의
                    # `self_attn` 은 크기 64 `n_h` 축이 여섯 등가류에 나뉘어 있다).
                    if shape is not None and [str(x) for x in sv] != [str(x) for x in shape]:
                        continue
                    want_i = None if ax is None else (ax if ax >= 0 else len(sv) + ax)
                    for i, (c, s) in enumerate(zip(cv, sv)):
                        if want_i is not None and i != want_i:
                            continue
                        if str(s) == frm and isinstance(c, int) and c == want:
                            # 실제로 바뀐 것만 센다. `from == to`(앵커 모드)에서는 이 자리가
                            # 안 바뀌고 등가류의 나머지가 바뀌므로, 무조건 세면 "발화했다"가
                            # 거짓이 된다 -- 게이트의 발화 검사가 그 거짓을 통과시킨다.
                            if str(sv[i]) != to:
                                sv[i] = to
                                p["n"] += 1
                            if p["spec"].get("spread") == "class":
                                # 이 축이 속한 텐서 전체를 같은 이름으로. 모듈 경계에서
                                # 멈추지 않는 유일한 경로다.
                                si = pairs.index((cv, sv)) if (cv, sv) in pairs else 0
                                p["n"] += _spread(p, row, out, fld, si, i, to)
    return [{"id": _report_id(p["spec"])["id"],
             "from": p["spec"]["from"], "to": p["spec"]["to"],
             "module": p["spec"]["module"], "expect": p["spec"]["expect"],
             "source": p["spec"].get("source", ""),
             "applied": p["n"], "vetoed": p["vetoed"]} for p in prepared]
