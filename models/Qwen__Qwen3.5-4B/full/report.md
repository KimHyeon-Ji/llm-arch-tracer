# Extraction Report -- Qwen/Qwen3.5-4B @ 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a

C1   PASS   32 == 32
C2   PASS   2 clusters == 2 from config schedule ['layer_types']
C3   PASS   acyclic, 0 orphan(s)
C4   PASS   embedding reachable from lm_head
C5   PASS   matmul contraction dims consistent; residual stream at d_model=2560 in 32/32 layers
C6   PASS   hidden_size=2560 (heuristic check, 1104 flagged)
C7   PASS   GQA 16:4 (repeat factor 4)
C8   SKIP   no MoE-related fields found on config (likely a dense model)
C9   PASS   vocab_size=248320, tie_word_embeddings=True
C10  PASS   all 426 params covered
C11  PASS   73 cache-related op(s) found, new-token seq dim confirmed
C13  SKIP   pass --check-repro to actually run twice and verify
C14  PASS   used=17 >= required=16
C15  PASS   all discovered entrypoints traced
C16  INFO   21498 unmapped rows, 27 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', 'aten.alias.default', 'aten.clone.default', 'aten.constant_pad_nd.default', 'aten.copy_.default', 'aten.expand.default', 'aten.eye.default', 'aten.masked_fill.Scalar', 'aten.ones.default']
C17  PASS   유도 상수 전부 설명됨, 구조 라이브러리에 등재됨
