# Extraction Report -- Qwen/Qwen3.5-397B-A17B @ 8472618112abcbd45acbcdc58436aff4233c23f7

C1   PASS   60 == 60
C2   PASS   2 clusters == 2 from config schedule ['layer_types']
C3   PASS   acyclic, 0 orphan(s)
C4   PASS   embedding reachable from lm_head
C5   PASS   matmul contraction dims consistent; residual stream at d_model=4096 in 60/60 layers
C6   PASS   hidden_size=4096 (heuristic check, 3645 flagged)
C7   PASS   GQA 32:2 (repeat factor 16)
C8   WARN   MoE trace-verified [router_dim(E=512):ok, top_k(10):ok, expert_weight:grouped]; routed-token count is data-dependent/symbolic (01-main.md C8) -- WARN is normal, not a defect.
C9   PASS   vocab_size=248320, tie_word_embeddings=False
C10  PASS   all 1038 params covered
C11  PASS   136 cache-related op(s) found, new-token seq dim confirmed
C13  SKIP   pass --check-repro to actually run twice and verify
C14  PASS   used=17 >= required=16
C15  PASS   all discovered entrypoints traced
C16  INFO   41679 unmapped rows, 38 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', 'aten.alias.default', 'aten.arange.default', 'aten.clamp_.default', 'aten.clone.default', 'aten.constant_pad_nd.default', 'aten.copy_.default', 'aten.div_.Tensor', 'aten.empty_like.default']
C17  PASS   유도 상수 전부 설명됨, 구조 라이브러리에 등재됨
