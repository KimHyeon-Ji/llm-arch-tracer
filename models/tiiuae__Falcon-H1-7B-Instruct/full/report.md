# Extraction Report -- tiiuae/Falcon-H1-7B-Instruct @ 41e72f27effbab80cd45b6e884688452253a3686

C1   PASS   44 == 44
C2   PASS   1 cluster(s); no per-layer schedule list on config to compare (uniform, or scalar schedule like first_k_dense_replace)
C3   PASS   acyclic, 0 orphan(s)
C4   PASS   embedding reachable from lm_head
C5   PASS   matmul contraction dims consistent; residual stream at d_model=3072 in 44/44 layers
C6   PASS   hidden_size=3072 (heuristic check, 2816 flagged)
C7   PASS   GQA 12:2 (repeat factor 6)
C8   SKIP   no MoE-related fields found on config (likely a dense model)
C9   PASS   vocab_size=130049, tie_word_embeddings=False
C10  PASS   all 751 params covered
C11  PASS   221 cache-related op(s) found, new-token seq dim confirmed
C13  SKIP   pass --check-repro to actually run twice and verify
C14  PASS   used=17 >= required=16
C15  PASS   all discovered entrypoints traced
C16  INFO   7431 unmapped rows, 29 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', 'aten.alias.default', 'aten.arange.default', 'aten.bitwise_not.default', 'aten.clamp.default', 'aten.clone.default', 'aten.constant_pad_nd.default', 'aten.copy_.default', 'aten.expand.default']
C17  PASS   유도 상수 전부 설명됨, 구조 라이브러리에 등재됨
