# Extraction Report -- HuggingFaceTB/SmolLM3-3B-Base @ d78a42f79198603e614095753484a04c10c2b940

C1   PASS   36 == 36
C2   PASS   2 clusters == 2 from config schedule ['layer_types', 'no_rope_layers']
C3   PASS   acyclic, 0 orphan(s)
C4   PASS   embedding reachable from lm_head
C5   PASS   matmul contraction dims consistent; residual stream at d_model=2048 in 36/36 layers
C6   PASS   hidden_size=2048 (heuristic check, 2520 flagged)
C7   PASS   GQA 16:4 (repeat factor 4)
C8   SKIP   no MoE-related fields found on config (likely a dense model)
C9   PASS   vocab_size=128256, tie_word_embeddings=True
C10  PASS   all 326 params covered
C11  PASS   127 cache-related op(s) found, new-token seq dim confirmed
C13  PASS   identical across two runs
C14  PASS   used=17 >= required=16
C15  PASS   all discovered entrypoints traced
C16  INFO   2449 unmapped rows, 14 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', 'aten.alias.default', 'aten.clone.default', 'aten.expand.default', 'aten.ones.default', 'aten.scalar_tensor.default', 'aten.slice.Tensor', 'aten.t.default', 'aten.transpose.int']
C17  PASS   유도 상수 전부 설명됨, 구조 라이브러리에 등재됨
