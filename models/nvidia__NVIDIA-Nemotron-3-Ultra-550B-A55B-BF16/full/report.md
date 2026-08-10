# Extraction Report -- nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 @ 624ba927cfbef0427354998700de3d51173c8c04

C1   PASS   108 == 108
C2   PASS   3 clusters == 3 from config schedule ['layers_block_type']
C3   PASS   acyclic, 0 orphan(s)
C4   PASS   embedding reachable from lm_head
C5   PASS   matmul contraction dims consistent; residual stream at d_model=8192 in 108/108 layers
C6   PASS   hidden_size=8192 (heuristic check, 0 flagged)
C7   PASS   GQA 64:2 (repeat factor 32)
C8   WARN   MoE trace-verified [router_dim(E=512):ok, top_k(22):ok, expert_weight:grouped]; routed-token count is data-dependent/symbolic (01-main.md C8) -- WARN is normal, not a defect.
C9   PASS   vocab_size=131072, tie_word_embeddings=False
C10  PASS   all 879 params covered
C11  PASS   72 cache-related op(s) found, new-token seq dim confirmed
C13  SKIP   pass --check-repro to actually run twice and verify
C14  PASS   used=24 >= required=24
C15  WARN   config declares 1 MTP/nextn layer(s) but no MTP module in the traced model (native transformers impl omits the MTP head) -- MTP NOT traced
C16  INFO   8523 unmapped rows, 41 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', 'aten.alias.default', 'aten.arange.default', 'aten.bitwise_not.default', 'aten.clamp.default', 'aten.clamp_.default', 'aten.clone.default', 'aten.constant_pad_nd.default', 'aten.copy_.default']
C17  PASS   유도 상수 전부 설명됨, 구조 라이브러리에 등재됨
