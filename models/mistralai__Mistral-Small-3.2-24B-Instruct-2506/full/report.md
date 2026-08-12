# Extraction Report -- mistralai/Mistral-Small-3.2-24B-Instruct-2506 @ 95a6d26c4bfb886c58daf9d3f7332c857cb27b43

C1   PASS   40 == 40
C2   PASS   1 cluster(s); no per-layer schedule list on config to compare (uniform, or scalar schedule like first_k_dense_replace)
C3   PASS   acyclic, 0 orphan(s)
C4   PASS   embedding reachable from lm_head
C5   PASS   matmul contraction dims consistent; residual stream at d_model=5120 in 40/40 layers
C6   PASS   hidden_size=5120 (heuristic check, 2800 flagged)
C7   PASS   GQA 32:8 (repeat factor 4)
C8   SKIP   no MoE-related fields found on config (likely a dense model)
C9   PASS   vocab_size=131072, tie_word_embeddings=False
C10  PASS   all 363 params covered
C11  PASS   161 cache-related op(s) found, new-token seq dim confirmed
C13  SKIP   pass --check-repro to actually run twice and verify
C14  PASS   used=16 >= required=16
C15  PASS   all discovered entrypoints traced
C16  INFO   2235 unmapped rows, 14 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', 'aten.alias.default', 'aten.arange.default', 'aten.clone.default', 'aten.expand.default', 'aten.le.Tensor', 'aten.scalar_tensor.default', 'aten.slice.Tensor', 'aten.t.default']
C17  PASS   유도 상수 전부 설명됨, 구조 라이브러리에 등재됨
