# Extraction Report -- Zyphra/Zamba2-1.2B @ 6b05bf29d1bb4ca71a36d12f7da4d3120dcde7fe

C1   PASS   38 == 38
C2   PASS   2 clusters == 2 from config schedule ['layers_block_type']
C3   PASS   acyclic, 0 orphan(s)
C4   PASS   embedding reachable from lm_head
C5   PASS   matmul contraction dims consistent; residual stream at d_model=2048 in 38/38 layers
C6   PASS   hidden_size=2048 (heuristic check, 0 flagged)
C7   PASS   MHA (kv_heads == heads, not GQA)
C8   SKIP   no MoE-related fields found on config (likely a dense model)
C9   PASS   vocab_size=32000, tie_word_embeddings=True
C10  PASS   all 406 params covered
C11  PASS   69 cache-related op(s) found, new-token seq dim confirmed
C13  PASS   identical across two runs
C14  PASS   used=16 >= required=16
C15  PASS   all discovered entrypoints traced
C16  INFO   4468 unmapped rows, 28 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', 'aten.alias.default', 'aten.bitwise_not.default', 'aten.clamp.default', 'aten.clone.default', 'aten.constant_pad_nd.default', 'aten.copy_.default', 'aten.expand.default', 'aten.masked_fill.Scalar']
C17  PASS   유도 상수 전부 설명됨, 구조 라이브러리에 등재됨
