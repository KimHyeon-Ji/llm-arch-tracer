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


def _scalar_args(args, kwargs):
    """텐서가 아닌 인자만 남긴다 -- dim / sizes / repeats 같은 것.

    `TorchDispatchMode` 에는 이것들이 평범한 `int` 나 `list[int]` 로 도착한다. 지금 파이프라인은
    이걸 버리고 나중에 구체 shape 으로 순열을 역산하는데, 크기가 겹치는 순간 역산이 불가능해진다
    (`_transpose_swaps_names` 가 `len(set(src)) != len(src)` 면 포기하는 이유). 기록만 해 두면
    축 계보를 op 정의 그대로 놓을 수 있다.

    텐서·모듈 같은 무거운 것은 담지 않는다. 스칼라와 스칼라 리스트만.
    """
    def _ok(v):
        if isinstance(v, bool) or v is None:
            return True
        if isinstance(v, (int, float, str)):
            return True
        if isinstance(v, (list, tuple)) and len(v) <= 16:
            return all(isinstance(x, (int, float, bool)) or x is None for x in v)
        return False

    pos = [v if not isinstance(v, tuple) else list(v)
           for v in args if _ok(v) and not isinstance(v, torch.Tensor)]
    kw = {k: (list(v) if isinstance(v, tuple) else v)
          for k, v in (kwargs or {}).items() if _ok(v) and not isinstance(v, torch.Tensor)}
    if not pos and not kw:
        return None
    return {"pos": pos, "kw": kw} if kw else {"pos": pos}


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
        # 텐서 하나에 안정된 id 를 준다. 같은 텐서가 여러 op 를 거치는 것을 행 사이에서
        # 이을 수 있어야 계보가 성립한다(WeakTensorKeyDictionary 라 수명은 텐서를 따른다).
        self.tensor_uid = torch.utils.weak.WeakTensorKeyDictionary()
        self._tid = itertools.count(1)
        for n, p in itertools.chain(model.named_parameters(), model.named_buffers()):
            self.param_origin[p] = n
            self.param_shape[n] = list(p.shape)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = str(func)
        flat, _ = tree_flatten((args, kwargs))
        tensors_in = [a for a in flat if isinstance(a, torch.Tensor)]

        deps, param_names = [], []
        # 입력 슬롯별 생산자 포트. `depends_on` 은 집합으로 접히므로 여기서 따로 남긴다 --
        # 어느 입력이 어느 op 의 몇 번째 출력에서 왔는지가 계보의 기본 단위다.
        input_sources = []
        input_tensor_ids = []
        for t in tensors_in:
            src = self.producer.get(t)
            input_sources.append(list(src) if src is not None else None)
            input_tensor_ids.append(self.tensor_uid.get(t))
            if src is not None:
                deps.append(src[0])
            origin = self.param_origin.get(t)
            if origin is not None:
                param_names.append(origin)

        out = func(*args, **kwargs)  # meta/fake -> shape only, no real compute

        op_id = next(self._id)
        outs = [o for o in tree_flatten(out)[0] if isinstance(o, torch.Tensor)]
        for slot, o in enumerate(outs):
            # **(op_id, output_slot)** 로 담는다. op_id 만 담으면 split 처럼 출력이 여럿인 op
            # 에서 "어느 조각이었는가" 가 사라지고, 그걸 잃으면 축 계보를 정확한 포트로
            # 이을 수 없다(외부 검토 2026-09-09, provenance 설계 1단계).
            self.producer[o] = (op_id, slot)
            self.tensor_uid[o] = self.tensor_uid.get(o) or next(self._tid)
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

        # **텐서를 만들지 않는 조회는 연산이 아니다.** `prim.device.default` 같은 것은 fake
        # tensor 모드에서만 디스패치에 올라오는데(meta 모드에서는 속성 접근으로 끝난다),
        # 표에 넣으면 "출력이 없는 op" 이라는 없는 개념이 생긴다. 실측: Kimi-K3 는 KDA 참조
        # 구현이 루프마다 `device=q.device` 를 읽어 **트레이스의 절반 이상**(30만 행 중 16만)이
        # 이것이었다. 다른 모델에는 하나도 없다 -- fake 모드로 도는 모델에서만 나온다.
        # 출력이 없고 이름이 `prim.` 으로 시작하는 것만 뺀다(aten 은 건드리지 않는다).
        if not outs and name.startswith("prim."):
            return out

        row = {
            "op_id": op_id,
            "raw_op": name,
            # **비텐서 인자를 버리지 않는다.** 지금은 `_transpose_swaps_names` 가 구체 shape
            # 으로 순열을 역산하는데, 크기가 겹치면 역산이 불가능하다. TorchDispatchMode 에는
            # dim/sizes 가 평범한 int 로 도착하므로 그대로 남긴다 -- transpose 의 dim0/dim1,
            # permute 의 dims, view 의 target sizes, cat 의 dim, split 의 dim/sizes 등.
            # (외부 검토 2026-09-09)
            "scalar_args": _scalar_args(args, kwargs),
            "input_shape": [_shape(t) for t in tensors_in],
            "weight_shape": weight_shape,
            "weight_pos": weight_pos,
            "output_shape": [_shape(o) for o in outs],
            "depends_on": sorted(set(d for d in deps if d != op_id)),
            "input_sources": input_sources,
            "input_tensor_ids": input_tensor_ids,
            "output_tensor_ids": [self.tensor_uid.get(o) for o in outs],
            "params": sorted(set(param_names)),
        }
        row.update(self.scope.current())
        self.rows.append(row)
        return out
