"""ScopeLabeler -- module hook, LABELING ONLY. Never used as op evidence (see 01-main.md
Step 4: hooks miss functional ops like softmax/RoPE that live inside a module's forward,
not at a module boundary)."""
# decoder-stack container names across families: Llama/Qwen/DeepSeek use `layers`, GPT-2 uses
# `h` (transformer.h.N), others use `blocks`/`block`. Extend as new naming shows up.
_STACK_NAMES = ("layers", "h", "blocks", "block", "layer")


def parse_scope(path: str) -> dict:
    """Decompose a module path into scope labels + an explicit hierarchy.

    `levels` is the module-nesting chain used to build per-level columns (h1, h2, ...) so the
    op table shows structure, not just a flat op sequence (01-main.md section 6). For a
    decoder-layer op the chain is everything under `<stack>.<idx>.` (e.g. self_attn -> q_proj);
    for other ops (embedding, final norm, lm_head) it is the full path. block/sub_block are
    kept (= first/second level) for the validators and structure rollup. The decoder stack is
    found by the first `<stackname>.<digit>` (outermost), so per-expert indices like
    `...experts.3.gate_proj` don't get mistaken for the layer index.
    """
    parts = path.split(".") if path else []
    layer_idx, levels = None, []
    for i in range(len(parts) - 1):
        if parts[i] in _STACK_NAMES and parts[i + 1].isdigit():
            layer_idx = int(parts[i + 1])
            levels = parts[i + 2:]
            break
    if layer_idx is not None:
        block = levels[0] if levels else None
        sub_block = levels[1] if len(levels) > 1 else None
    else:
        block = "other"
        sub_block = None
        levels = parts  # full component chain for non-layer modules
    return {
        "layer_idx": layer_idx,
        "block": block,
        "sub_block": sub_block,
        "levels": levels,
        "depth": len(levels),
        "module_path": path,
    }


class ScopeLabeler:
    def __init__(self, model):
        self.stack = []
        self.handles = []
        for name, mod in model.named_modules():
            if name == "":
                continue
            self.handles.append(mod.register_forward_pre_hook(self._pre(name)))
            self.handles.append(mod.register_forward_hook(self._post()))

    def _pre(self, name):
        def hook(*_args, **_kwargs):
            self.stack.append(name)

        return hook

    def _post(self):
        def hook(*_args, **_kwargs):
            if self.stack:
                self.stack.pop()

        return hook

    def current(self) -> dict:
        path = self.stack[-1] if self.stack else ""
        return parse_scope(path)

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []
