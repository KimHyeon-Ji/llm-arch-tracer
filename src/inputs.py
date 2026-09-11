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
def _input_device(model) -> str:
    """Where to build the inputs: "meta" normally, "cpu" once the model is on FakeTensors.

    The two backends need different input tensors. `meta` carries shape only, which is the point
    -- but a model that reads a VALUE out of an input (`cache_position[0]`, as Kimi-Linear does
    to decide whether it is prefilling) cannot run there at all, which is why the meta_to_fake
    remedy exists. FakeTensorMode is built to accept real tensors alongside fake ones
    (`allow_non_fake_inputs=True`), so on that backend the inputs must be ordinary CPU tensors --
    they are 17 integers, nothing is materialised. Handing it meta inputs instead reproduced the
    exact error the remedy was applied to fix, so the retry loop spun six times and gave up
    (2026-08-10).
    """
    try:
        p = next(model.parameters())
    except StopIteration:
        return "meta"
    return "meta" if getattr(p, "device", None) is not None and p.device.type == "meta" else "cpu"


def build_inputs(model, cfg, phase: str, seq_len: int, past=None, batch: int = 1) -> dict:
    """`batch` 는 **배치 축 판별 프로브**를 위해 열어 둔 것이다. 기본값 1 이 발행 경로다.

    크기 1 축은 배치인지 방송 싱글턴인지 값으로 구분할 수 없다 -- `dim()` 이 크기 1 이면
    무조건 `B` 를 답하므로, "값이 1 이 아닌 B 가 없다" 는 식의 검사는 반례를 찾을 수 없다
    (2026-09-11 에 그걸 반례 검사라고 내세웠다가 외부 검토에 걸렸다). 같은 모델을 B=2 로
    한 번 더 트레이스해 **크기가 따라 변하는 축**을 봐야 진짜 배치 축을 안다.
    """
    vocab = cfg.vocab_size
    dev = _input_device(model)
    if phase == "prefill":
        ids = (torch.arange(seq_len) % vocab).view(1, -1).repeat(batch, 1).to(dev)
        kw = dict(
            input_ids=ids,
            position_ids=torch.arange(seq_len).view(1, -1).repeat(batch, 1).to(dev),
            cache_position=torch.arange(seq_len).to(dev),
            use_cache=True,  # so decode gets a model-generated cache, see 01-main.md Step 6
        )
    elif phase == "decode":
        p = seq_len
        ids = torch.tensor([p % vocab]).view(1, 1).repeat(batch, 1).to(dev)
        kw = dict(
            input_ids=ids,
            position_ids=torch.tensor([[p]]).repeat(batch, 1).to(dev),
            cache_position=torch.tensor([p]).to(dev),
            past_key_values=past,
            use_cache=True,
        )
    else:
        raise ValueError(f"unknown phase: {phase}")
    return _filter_kwargs(model, kw)
