"""Step 4 -- input construction. arange token ids (never zeros) + forward-signature
filtering. See 01-main.md Step 4."""
import inspect
import torch


def _filter_kwargs(model, kwargs: dict) -> dict:
    sig = set(inspect.signature(model.forward).parameters)
    return {k: v for k, v in kwargs.items() if k in sig}


# We deliberately do NOT pass an attention_mask. For a single unpadded sequence an
# all-ones padding mask is redundant, and transformers 5.x value-inspects it (masking_utils
# fast_all -> tensor.sum()==numel -> .item()) to decide whether it can skip the causal mask.
# That value inspection is impossible on meta tensors ("Tensor.item() cannot be called on
# meta tensors") and it fails identically under both sdpa and eager -- so it is NOT an
# attn_implementation issue. Omitting the mask makes the model build a pure causal mask from
# cache_position/position_ids, which is exactly what we want for prefill/decode of one
# unpadded sequence. Verified empirically: with the mask both sdpa+eager FAIL on meta; without
# it both succeed and still produce a usable past_key_values cache. (01-main.md P2 meta-first.)
def build_inputs(model, cfg, phase: str, seq_len: int, past=None) -> dict:
    vocab = cfg.vocab_size
    if phase == "prefill":
        ids = (torch.arange(seq_len) % vocab).view(1, -1).to("meta")
        kw = dict(
            input_ids=ids,
            position_ids=torch.arange(seq_len).view(1, -1).to("meta"),
            cache_position=torch.arange(seq_len).to("meta"),
            use_cache=True,  # so decode gets a model-generated cache, see 01-main.md Step 6
        )
    elif phase == "decode":
        p = seq_len
        ids = torch.tensor([p % vocab]).view(1, 1).to("meta")
        kw = dict(
            input_ids=ids,
            position_ids=torch.tensor([[p]]).to("meta"),
            cache_position=torch.tensor([p]).to("meta"),
            past_key_values=past,
            use_cache=True,
        )
    else:
        raise ValueError(f"unknown phase: {phase}")
    return _filter_kwargs(model, kw)
