"""OpGraphTracer -- the actual op/shape/dependency capture. TorchDispatchMode intercepts
every ATen op dispatched during forward, on meta/fake tensors (zero real compute).
depends_on is derived from tensor identity, not static analysis or guessing (01-main.md
P1, P3)."""
import itertools
import torch
from torch.utils._pytree import tree_flatten
from torch.utils._python_dispatch import TorchDispatchMode

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

        weight_shape = None
        for w in sorted(set(param_names)):
            s = self.param_shape.get(w)
            if s and len(s) >= 2:
                weight_shape = s
                break

        row = {
            "op_id": op_id,
            "raw_op": name,
            "input_shape": [_shape(t) for t in tensors_in],
            "weight_shape": weight_shape,
            "output_shape": [_shape(o) for o in outs],
            "depends_on": sorted(set(d for d in deps if d != op_id)),
            "params": sorted(set(param_names)),
        }
        row.update(self.scope.current())
        self.rows.append(row)
        return out
