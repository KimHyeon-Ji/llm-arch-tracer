# Extraction Report -- NX-AI/xLSTM-7b @ 9dc507bd0939cf372a4a4f667335651d8e49dddb

C1   PASS   32 == 32
C2   PASS   1 cluster(s); no per-layer schedule list on config to compare (uniform, or scalar schedule like first_k_dense_replace)
C3   PASS   acyclic, 0 orphan(s)
C4   PASS   embedding reachable from lm_head
C5   PASS   matmul contraction dims consistent; residual stream at d_model=4096 in 32/32 layers
C6   PASS   hidden_size=4096 (heuristic check, 0 flagged)
C7   SKIP   no attention-head field
C8   SKIP   no MoE-related fields found on config (likely a dense model)
C9   PASS   vocab_size=50304, tie_word_embeddings=False
C10  PASS   all 483 params covered
C11  WARN   no concat/cache-touching op found in decode trace -- verify cache is actually being reused
C13  PASS   identical across two runs
C14  PASS   used=16 >= required=16
C15  PASS   all discovered entrypoints traced
C16  INFO   22501 unmapped rows, 20 distinct raw ops: ['aten._unsafe_view.default', 'aten.abs.default', 'aten.add_.Tensor', 'aten.alias.default', 'aten.clone.default', 'aten.copy_.default', 'aten.div.Tensor', 'aten.expand.default', 'aten.log_sigmoid_forward.default', 'aten.maximum.default']
C17  PASS   유도 상수 전부 설명됨, 구조 라이브러리에 등재됨
