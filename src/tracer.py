"""OpGraphTracer -- the actual op/shape/dependency capture. TorchDispatchMode intercepts
every ATen op dispatched during forward, on meta/fake tensors (zero real compute).
depends_on is derived from tensor identity, not static analysis or guessing (01-main.md
P1, P3)."""
import itertools
import torch
from torch.utils._pytree import tree_flatten
from torch.utils._python_dispatch import TorchDispatchMode

import build_table      # for weight_pos_candidates -- one definition of "this operand IS the
                       # weight", shared with the regeneration path so the two cannot drift
import noderef
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
    def __init__(self, model, scope: ScopeLabeler, phase: str = "?"):
        super().__init__()
        self.scope = scope
        self.rows = []
        self._id = itertools.count()
        # **물리 생산자와 논리 출처는 다른 맵이다.** `depends_on` 은 물리 쪽에서만 나온다 --
        # 하나로 합쳐 의미 노드로 덮으면 기존 ATen 의존성이 바뀌어 legacy 산출물이 흔들린다
        # (외부 검토 2026-09-10).
        self.physical_producer = torch.utils.weak.WeakTensorKeyDictionary()
        # **쓰기 의존은 값 출처와 다른 것이다.** 뷰에 제자리로 쓰면 베이스의 내용이 바뀌므로
        # 베이스를 읽는 op 은 그 쓰기에 **의존**한다. 그렇다고 베이스의 **값이** 그 뷰의
        # 출력인 것은 아니다 -- 뷰는 베이스의 일부라 rank 도 크기도 다르다. 예전에는
        # `physical_producer[base]` 를 뷰의 포트로 덮어써서 `input_sources` 가 rank 4 출력을
        # rank 5 입력의 출처로 가리켰다(외부 검토 2026-09-20). 의존은 여기 모으고 값 출처는
        # 건드리지 않는다. 누적이라 **미리 만들어 둔 뷰에 여러 번 쓴 경우**도 전부 남는다.
        self.write_deps = torch.utils.weak.WeakTensorKeyDictionary()
        self.param_origin = torch.utils.weak.WeakTensorKeyDictionary()
        # 그 텐서가 **어떤 종류의** 외부 입력인가. `param_origin` 하나로는 parameter 와
        # buffer 를 구분할 수 없었다(둘 다 같은 맵에 넣고 있었다).
        self.origin_kind = torch.utils.weak.WeakTensorKeyDictionary()
        # 그래프 입력의 **kwargs 경로 이름**. parameter/buffer 는 `param_origin` 이 든다.
        self.external_name = torch.utils.weak.WeakTensorKeyDictionary()
        self.param_shape = {}
        # 텐서 하나에 안정된 id 를 준다. 같은 텐서가 여러 op 를 거치는 것을 행 사이에서
        # 이을 수 있어야 계보가 성립한다(WeakTensorKeyDictionary 라 수명은 텐서를 따른다).
        self.tensor_uid = torch.utils.weak.WeakTensorKeyDictionary()
        self._tid = itertools.count(1)
        # 논리 version. 같은 텐서가 제자리 연산으로 내용이 바뀌거나 의미 경계를 지나면
        # 올라간다. `logical_source[(uid, version)]` 가 그 시점의 출처다.
        self.version = {}
        self.logical_source = {}
        self.phase = phase
        for n, p in model.named_parameters():
            self.param_origin[p] = n
            self.origin_kind[p] = "parameter"
            self.param_shape[n] = list(p.shape)
        for n, b in model.named_buffers():
            self.param_origin[b] = n
            self.origin_kind[b] = "buffer"
            self.param_shape[n] = list(b.shape)

    def register_graph_inputs(self, kwargs):
        """모델 호출 인자를 `graph_input` 으로 등록한다. 트레이스 **시작 전에** 부른다.

        종류만이 아니라 **kwargs 경로 이름**(`input_ids`, `past_key_values.0.key` …)까지
        남긴다. 종류만 적으면 그래프 입력이 여럿일 때 어느 것인지 알 수 없다.
        """
        def walk(obj, path):
            if isinstance(obj, torch.Tensor):
                if obj not in self.origin_kind:
                    self.origin_kind[obj] = "graph_input"
                    self.external_name[obj] = path
                return
            if isinstance(obj, dict):
                for k, v in obj.items():
                    walk(v, f"{path}.{k}" if path else str(k))
            elif isinstance(obj, (list, tuple)):
                for i, v in enumerate(obj):
                    walk(v, f"{path}.{i}" if path else str(i))
            else:
                for attr in ("key_cache", "value_cache", "layers"):
                    sub = getattr(obj, attr, None)
                    if sub is not None:
                        walk(sub, f"{path}.{attr}" if path else attr)

        walk(kwargs, "")

    def _logical_source(self, t, phys):
        """그 텐서의 **논리 출처**. 지금은 물리 생산자와 같다.

        의미 노드로 다시 묶는 것(SemanticPort)은 다음 단계다 -- 지금 켜면 전역 시간 barrier
        와 새 방식이 동시에 살아 있는 중간 상태가 된다(외부 검토 2026-09-10).
        """
        uid = self.tensor_uid.get(t)
        if uid is not None:
            ref = self.logical_source.get((uid, self.version.get(uid, 0)))
            if ref is not None:
                return ref
        if phys is not None:
            return noderef.SourceRef(noderef.OpNode(phys[0]), phys[1])
        kind = self.origin_kind.get(t) or "unknown_external"
        name = self.param_origin.get(t) if kind in ("parameter", "buffer")             else self.external_name.get(t)
        return noderef.SourceRef(noderef.ExtNode(kind, name), 0)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = str(func)
        flat, _ = tree_flatten((args, kwargs))
        tensors_in = [a for a in flat if isinstance(a, torch.Tensor)]

        deps, param_names = [], []
        # 쓰기 의존은 `depends_on` 에도 들어가지만(그게 참이니까) **값 등가류를 이을 때는
        # 빼야 한다.** `axis_classes` 는 생산자의 출력 shape 과 소비자의 입력 shape 을
        # 맞춰 축을 잇는데, 쓰기 의존의 "생산자" 는 **뷰** 라서 shape 이 우연히 맞으면
        # 엉뚱한 축끼리 한 등가류가 된다 -- 하드 불변식(한 축에 이름 하나)이 138 건
        # 깨졌다(2026-09-20). 어느 간선이 쓰기 의존인지 따로 적어 그쪽에서 거른다.
        write_dep_ids = set()
        # 입력 슬롯별 생산자 포트. `depends_on` 은 집합으로 접히므로 여기서 따로 남긴다 --
        # 어느 입력이 어느 op 의 몇 번째 출력에서 왔는지가 계보의 기본 단위다.
        input_sources = []
        input_tensor_ids = []
        for t in tensors_in:
            phys = self.physical_producer.get(t)
            # `depends_on` 은 **물리 생산자**에서만 나온다. 의미 노드가 여기 섞이면 기존
            # ATen 의존성이 바뀐다.
            if phys is not None:
                deps.append(phys[0])
            wd = self.write_deps.get(t) or ()
            deps.extend(wd)
            write_dep_ids.update(wd)
            input_sources.append(noderef.encode(self._logical_source(t, phys)))
            input_tensor_ids.append(self.tensor_uid.get(t))
            origin = self.param_origin.get(t)
            if origin is not None:
                param_names.append(origin)

        out = func(*args, **kwargs)  # meta/fake -> shape only, no real compute

        op_id = next(self._id)
        outs = [o for o in tree_flatten(out)[0] if isinstance(o, torch.Tensor)]
        bumped = set()
        for slot, o in enumerate(outs):
            # **(op_id, output_slot)** 로 담는다. op_id 만 담으면 split 처럼 출력이 여럿인 op
            # 에서 "어느 조각이었는가" 가 사라지고, 그걸 잃으면 축 계보를 정확한 포트로
            # 이을 수 없다(외부 검토 2026-09-09, provenance 설계 1단계).
            self.physical_producer[o] = (op_id, slot)
            self.tensor_uid[o] = self.tensor_uid.get(o) or next(self._tid)
            # **제자리 연산.** 입력과 같은 객체가 돌아오면 uid 는 같은데 내용이 바뀌었다.
            # version 을 올려 두지 않으면 그 텐서에 걸려 있던 옛 출처(나중에는 SemanticPort)
            # 가 그대로 남아, 뒤 op 가 이 변경을 건너뛴 것처럼 보인다(외부 검토 2026-09-10).
            # uid 하나당 이 op 에서 한 번만 올린다.
            uid = self.tensor_uid.get(o)
            if uid not in bumped and any(o is t for t in tensors_in):
                bumped.add(uid)
                self.version[uid] = self.version.get(uid, 0) + 1
            ref = noderef.SourceRef(noderef.OpNode(op_id), slot)
            self.logical_source[(uid, self.version.get(uid, 0))] = ref
            if name in TRIVIAL and param_names:
                self.param_origin[o] = param_names[0]

            # **뷰에 쓰면 베이스도 바뀐다.** `o[:, i] = einsum(...)` 은 `select` 가 만든
            # 뷰에 `copy_` 하는데, 뷰와 베이스는 서로 다른 텐서 객체다. 베이스의 생산자를
            # 갱신하지 않으면 나중에 베이스를 읽는 op 이 이 쓰기를 못 보고, 그 결과
            # KDA 의 attention 결과가 `o_norm` 까지 이어지지 않았다 -- 표만 그래프로 읽으면
            # V projection 과 output gate 만으로 출력이 만들어지는 것처럼 보였다
            # (외부 검토 2026-09-20). 원래 생산자 간선은 사라지지 않는다: 이 op 이
            # 뷰(=`select` 출력)에 의존하고 그 `select` 가 옛 생산자에 의존하므로 사슬로 남는다.
            if any(o is t for t in tensors_in):
                base, seen = getattr(o, "_base", None), set()
                while base is not None and id(base) not in seen:
                    seen.add(id(base))
                    # **의존만 쌓는다.** 값 출처(`physical_producer`)는 건드리지 않는다 --
                    # 베이스의 값은 여전히 베이스를 만든 op 의 것이고, 이 op 은 그 일부를
                    # 바꿨을 뿐이다.
                    cur = self.write_deps.get(base)
                    if cur is None:
                        cur = []
                        self.write_deps[base] = cur
                    if op_id not in cur:
                        cur.append(op_id)
                    buid = self.tensor_uid.get(base)
                    if buid is None:
                        buid = next(self._tid)
                        self.tensor_uid[base] = buid
                    # 내용이 바뀌었으므로 version 은 올린다. 다만 이 version 의 논리 출처를
                    # 뷰의 포트로 적지는 않는다 -- 적으면 같은 오류가 그쪽으로 옮겨간다.
                    if buid not in bumped:
                        bumped.add(buid)
                        self.version[buid] = self.version.get(buid, 0) + 1
                    base = getattr(base, "_base", None)

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
            "write_deps": sorted(d for d in write_dep_ids if d != op_id),
            "input_sources": input_sources,
            "ports_schema_version": noderef.SCHEMA_VERSION,
            "input_tensor_ids": input_tensor_ids,
            "output_tensor_ids": [self.tensor_uid.get(o) for o in outs],
            "params": sorted(set(param_names)),
        }
        row.update(self.scope.current())
        self.rows.append(row)
        return out
