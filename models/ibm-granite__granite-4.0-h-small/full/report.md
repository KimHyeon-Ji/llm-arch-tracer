# Extraction Report -- ibm-granite/granite-4.0-h-small @ b8c0982bab7fde4eb48110f5a069527c008fab39

C1   PASS   40 == 40
C2   PASS   2 clusters == 2 from config schedule ['layer_types']
C3   PASS   acyclic, 0 orphan(s)
C4   PASS   embedding reachable from lm_head
C5   PASS   matmul contraction dims consistent; residual stream at d_model=4096 in 40/40 layers
C6   PASS   hidden_size=4096 (heuristic check, 168 flagged)
C7   PASS   GQA 32:8 (repeat factor 4)
C8   WARN   MoE trace-verified [router_dim(E=72):ok, top_k(10):ok, expert_weight:grouped]; routed-token count is data-dependent/symbolic (01-main.md C8) -- WARN is normal, not a defect.
C9   PASS   vocab_size=100352, tie_word_embeddings=True
C10  PASS   all 586 params covered
C11  PASS   44 cache-related op(s) found, new-token seq dim confirmed
C13  SKIP   pass --check-repro to actually run twice and verify
C14  PASS   used=16 >= required=16
C15  PASS   all discovered entrypoints traced
C16  INFO   5459 unmapped rows, 39 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', 'aten.alias.default', 'aten.arange.default', 'aten.bitwise_not.default', 'aten.clamp.default', 'aten.clamp_.default', 'aten.clone.default', 'aten.constant_pad_nd.default', 'aten.copy_.default']
C17  PASS   유도 상수 전부 설명됨, 구조 라이브러리에 등재됨
