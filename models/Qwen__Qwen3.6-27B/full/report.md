# Extraction Report -- Qwen/Qwen3.6-27B @ 6a9e13bd6fc8f0983b9b99948120bc37f49c13e9

C1   PASS   64 == 64
C2   PASS   2 clusters == 2 from config schedule ['layer_types']
C3   PASS   acyclic, 0 orphan(s)
C4   PASS   embedding reachable from lm_head
C5   PASS   matmul contraction dims consistent; residual stream at d_model=5120 in 64/64 layers
C6   PASS   hidden_size=5120 (heuristic check, 2208 flagged)
C7   PASS   GQA 24:4 (repeat factor 6)
C8   SKIP   no MoE-related fields found on config (likely a dense model)
C9   PASS   vocab_size=248320, tie_word_embeddings=False
C10  PASS   all 851 params covered
C11  PASS   145 cache-related op(s) found, new-token seq dim confirmed
C13  SKIP   pass --check-repro to actually run twice and verify
C14  PASS   used=17 >= required=16
C15  PASS   all discovered entrypoints traced
C16  INFO   42962 unmapped rows, 27 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', 'aten.alias.default', 'aten.clone.default', 'aten.constant_pad_nd.default', 'aten.copy_.default', 'aten.expand.default', 'aten.eye.default', 'aten.masked_fill.Scalar', 'aten.ones.default']
C17  PASS   유도 상수 전부 설명됨, 구조 라이브러리에 등재됨
