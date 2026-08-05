"""OpGraphTracer -- the actual op/shape/dependency capture. TorchDispatchMode intercepts
every ATen op dispatched during forward, on meta/fake tensors (zero real compute).
depends_on is derived from tensor identity, not static analysis or guessing (01-main.md
P1, P3)."""
import itertools
import torch
from torch.utils._pytree import tree_flatten
from torch.utils._python_dispatch import TorchDispatchMode

import build_table          # for weight_pos_candidates -- one definition of "this operand IS the
                            # weight", shared with the regeneration path so the two cannot drift
from scope import ScopeLabeler

# view/transpose/etc: not "real" ops for weight attribution purposes, but param origin
# should propagate through them (e.g. weight.t() should still count as touching `weight`)
TRIVIAL = {
    "aten.t.default",
    "aten.transpose.int",
    "aten.view.default",
    "aten.reshape.default",
    "aten.permute.default",
    "aten.expand.default",
    "aten._unsafe_view.default",
    "aten.detach.default",
    "aten.contiguous.default",
    "aten.alias.default",
    "aten._to_copy.default",
}


def _shape(t):
    return list(t.shape) if isinstance(t, torch.Tensor) else None


class OpGraphTracer(TorchDispatchMode):
    def __init__(self, model, scope: ScopeLabeler):
        super().__init__()
        self.scope = scope
        self.rows = []
        self._id = itertools.count()
        self.producer = torch.utils.weak.WeakTensorKeyDictionary()
        self.param_origin = torch.utils.weak.WeakTensorKeyDictionary()
        self.param_shape = {}
        for n, p in itertools.chain(model.named_parameters(), model.named_buffers()):
            self.param_origin[p] = n
            self.param_shape[n] = list(p.shape)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = str(func)
        flat, _ = tree_flatten((args, kwargs))
        tensors_in = [a for a in flat if isinstance(a, torch.Tensor)]

        deps, param_names = [], []
        for t in tensors_in:
            pid = self.producer.get(t)
            if pid is not None:
                deps.append(pid)
            origin = self.param_origin.get(t)
            if origin is not None:
                param_names.append(origin)

        out = func(*args, **kwargs)  # meta/fake -> shape only, no real compute

        op_id = next(self._id)
        outs = [o for o in tree_flatten(out)[0] if isinstance(o, torch.Tensor)]
        for o in outs:
            self.producer[o] = op_id
            if name in TRIVIAL and param_names:
                self.param_origin[o] = param_names[0]

        weight_shape, weight_name = None, None
        for w in sorted(set(param_names)):
            s = self.param_shape.get(w)
            if s and len(s) >= 2:
                weight_shape, weight_name = s, w
                break

        # Which input_shape entry IS that weight. input_shape lists every tensor the op received,
        # so for a Linear the weight is in there twice over: once as input_shape[i] (the operand,
        # already transposed by aten.t) and once as weight_shape (the as-stored parameter). Without
        # this index a consumer cannot tell the two apart -- it would double-count the weight in a
        # bytes/FLOPs model, or charge activation dtype to a quantized weight. See 01-main.md 6.2.
        #
        # Deliberately the SHAPE-verbatim answer, cross-checked against tensor identity rather than
        # taken from it. Identity alone would also point at an operand that is a reshape or a slice
        # of the parameter (DeepSeek-V4's o_a_proj feeds a bmm a 3-D view of a 2-D weight), whose
        # shape is NOT weight_shape -- so the column would have meant one thing on freshly traced
        # models and another on regenerated ones. Identity only breaks ties between operands that
        # already match, which is what makes a square q_proj [5120, 5120] unambiguous here and not
        # in build_table.derive_weight_pos.
        weight_pos = None
        if weight_name is not None:
            shapes_in = [_shape(t) for t in tensors_in]
            cands = build_table.weight_pos_candidates(weight_shape, shapes_in)
            weight_pos = -1        # weight_shape appears on no operand: fused, reshaped or sliced
            for i in cands:
                if self.param_origin.get(tensors_in[i]) == weight_name:
                    weight_pos = i
                    break
            else:
                if cands:
                    weight_pos = cands[-1] if len(shapes_in) > 1 else cands[0]

        row = {
            "op_id": op_id,
            "raw_op": name,
            "input_shape": [_shape(t) for t in tensors_in],
            "weight_shape": weight_shape,
            "weight_pos": weight_pos,
            "output_shape": [_shape(o) for o in outs],
            "depends_on": sorted(set(d for d in deps if d != op_id)),
            "params": sorted(set(param_names)),
        }
        row.update(self.scope.current())
        self.rows.append(row)
        return out
