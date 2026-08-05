"""PROTOTYPE -- label the 98% of axes that have NO module to ask, using PyTorch's own symbolic
shape engine instead of hand-written propagation rules.

The gap this addresses (03-labeling-roadmap.md section 5): module-declared dimensions are only
1.8% of all axes. Everything else -- attention scores [B, n_h, T, T], the RoPE half-dim, the
repeat_kv expansion, the T+1 cache length -- has no nn.Linear to read the answer off.

Writing a propagation rule per aten op was the obvious plan and it is the wrong one: there are
hundreds of ops, and a hand-written rule is another place to be subtly wrong. PyTorch already
computes exactly this. Build the parameters as FakeTensors whose sizes are SymInts, run the
real forward under a ShapeEnv, and every intermediate shape comes back as a sympy expression
computed by torch's own meta kernels -- exact by construction, not by rule.

    embedding   (1, s47, s90)          = [B, T, d_model]
    view        (1, s47, s7, s42)      = [B, T, n_h, d_head]     <- the reshape SPLIT, free
    scores      (1, s7, s47, s47)      = [B, n_h, T, T]
    cat         (1, s47 + 1)           = T+1  (the KV cache length)

Two things make it work on a real model:
  * BACKED symbols with hints (`create_symintnode(..., hint=val)`), not unbacked. Model code
    branches on shapes (`if q_len > 1`); unbacked symints raise GuardOnDataDependentSymNode,
    backed ones evaluate against the hint and merely record a guard.
  * Parameters are swapped AFTER construction. transformers 5.x configs are strict dataclasses
    and reject a SymInt field outright, but nn.Module attributes have no such validation.

Measured (first pass, only 7 Linear leaf names + embedding + 1-D norms symbolised):
    Qwen2.5-0.5B    2494 ops   59% of axes symbolic
    Llama-3.1-8B    3302 ops   59%
    gemma-2-2b      3158 ops   58%
    DeepSeek-V2-Lite  FAILED -- MLA's q_proj is n_h*(d_nope+d_rope), not n_h*d_head

That failure is the point: the per-module expression must come from develop/probe_symbolic_dims.py
(config tagging), not from a hardcoded MAP like the one below. The two compose -- tagging says
WHAT each parameter axis is, this says where it GOES.

Run:  .venv/Scripts/python.exe develop/probe_symbolic_forward.py <hf-model-id> [...]
"""
import sys,torch,torch.nn as nn,collections
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from torch.fx.experimental.symbolic_shapes import ShapeEnv, DimDynamic
from torch._subclasses.fake_tensor import FakeTensorMode
from torch._dynamo.source import ConstantSource
from torch.utils._python_dispatch import TorchDispatchMode
import provenance, loader

def run(mid):
    cfg,_=provenance.snapshot(mid)
    m=loader.load_meta(cfg, trust_remote_code=provenance.needs_remote_code(cfg))
    se=ShapeEnv(); fm=FakeTensorMode(shape_env=se, allow_non_fake_inputs=True)
    def sym(n,v): return se.create_symintnode(se.create_symbol(v,ConstantSource(n),dynamic_dim=DimDynamic.DYNAMIC),hint=v)
    rows=[]
    class Tr(TorchDispatchMode):
        def __torch_dispatch__(self,f,t,args=(),kw=None):
            o=f(*args,**(kw or {}))
            outs=[x for x in (o if isinstance(o,(tuple,list)) else [o]) if isinstance(x,torch.Tensor)]
            if outs: rows.append((str(f), tuple(outs[0].shape)))
            return o
    with fm:
        hs=cfg.hidden_size; nh=cfg.num_attention_heads
        nkv=getattr(cfg,'num_key_value_heads',nh); hd=getattr(cfg,'head_dim',None) or hs//nh
        d_model=sym('d_model',hs); n_h=sym('n_h',nh); n_kv=sym('n_kv',nkv); d_head=sym('d_head',hd)
        d_ff=sym('d_ff',cfg.intermediate_size); V=sym('V',cfg.vocab_size); T=sym('T',16)
        MAP={'q_proj':(n_h*d_head,d_model),'k_proj':(n_kv*d_head,d_model),'v_proj':(n_kv*d_head,d_model),
             'o_proj':(d_model,n_h*d_head),'gate_proj':(d_ff,d_model),'up_proj':(d_ff,d_model),'down_proj':(d_model,d_ff)}
        for name,mod in m.named_modules():
            leaf=name.split('.')[-1]
            if isinstance(mod,nn.Linear) and leaf in MAP:
                o,i=MAP[leaf]
                mod.weight=nn.Parameter(torch.empty(o,i,device='meta'),requires_grad=False)
                if getattr(mod,'bias',None) is not None: mod.bias=nn.Parameter(torch.empty(o,device='meta'),requires_grad=False)
                mod.in_features,mod.out_features=i,o
            elif isinstance(mod,nn.Embedding): mod.weight=nn.Parameter(torch.empty(V,d_model,device='meta'),requires_grad=False)
            elif isinstance(mod,nn.Linear) and leaf=='lm_head': mod.weight=nn.Parameter(torch.empty(V,d_model,device='meta'),requires_grad=False)
            elif getattr(mod,'weight',None) is not None and mod.weight.dim()==1:
                mod.weight=nn.Parameter(torch.empty(d_model,device='meta'),requires_grad=False)
            if hasattr(mod,'head_dim'): mod.head_dim=d_head
        ids=torch.empty(1,T,dtype=torch.long,device='meta')
        with Tr(): m(input_ids=ids, use_cache=False)
    tot=sum(len(s) for _,s in rows); ns=sum(1 for _,s in rows for x in s if not isinstance(x,int))
    print('%-44s ops=%5d axes=%6d symbolic=%.0f%%' % (mid, len(rows), tot, 100*ns/tot))
    return rows

if __name__=='__main__':
    for mid in sys.argv[1:]:
        try: run(mid)
        except Exception as e:
            print('%-44s FAILED %s: %s' % (mid, type(e).__name__, str(e)[:110]))
