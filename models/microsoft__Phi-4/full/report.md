# Extraction Report -- microsoft/Phi-4 @ 2db69c1c3e91a05d2c64a3185acfbaf36f744e25

C1   PASS   40 == 40
C2   PASS   1 cluster(s); no per-layer schedule list on config to compare (uniform, or scalar schedule like first_k_dense_replace)
C3   PASS   acyclic, 0 orphan(s)
C4   PASS   embedding reachable from lm_head
C5   PASS   matmul contraction dims consistent; residual stream at d_model=5120 in 40/40 layers
C6   PASS   hidden_size=5120 (heuristic check, 2800 flagged)
C7   PASS   GQA 40:10 (repeat factor 4)
C8   SKIP   no MoE-related fields found on config (likely a dense model)
C9   PASS   vocab_size=100352, tie_word_embeddings=False
C10  PASS   all 243 params covered
C11  PASS   241 cache-related op(s) found, new-token seq dim confirmed
C13  SKIP   pass --check-repro to actually run twice and verify
C14  PASS   used=16 >= required=16
C15  PASS   all discovered entrypoints traced
C16  INFO   2439 unmapped rows, 15 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', 'aten.alias.default', 'aten.arange.default', 'aten.clone.default', 'aten.expand.default', 'aten.le.Tensor', 'aten.scalar_tensor.default', 'aten.slice.Tensor', 'aten.split.Tensor']
C17  PASS   유도 상수 전부 설명됨, 구조 라이브러리에 등재됨
