# Extraction Report -- meta-llama/Llama-3.1-405B @ b906e4dc842aa489c962f9db26554dcfdde901fe

C1   PASS   126 == 126
C2   PASS   1 cluster(s); no per-layer schedule list on config to compare (uniform, or scalar schedule like first_k_dense_replace)
C3   PASS   acyclic, 0 orphan(s)
C4   PASS   embedding reachable from lm_head
C5   PASS   matmul contraction dims consistent; residual stream at d_model=16384 in 126/126 layers
C6   PASS   hidden_size=16384 (heuristic check, 9324 flagged)
C7   PASS   GQA 128:8 (repeat factor 16)
C8   SKIP   no MoE-related fields found on config (likely a dense model)
C9   PASS   vocab_size=128256, tie_word_embeddings=False
C10  PASS   all 1137 params covered
C11  PASS   505 cache-related op(s) found, new-token seq dim confirmed
C13  PASS   identical across two runs
C14  PASS   used=17 >= required=16
C15  PASS   all discovered entrypoints traced
C16  INFO   8713 unmapped rows, 14 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', 'aten.alias.default', 'aten.clone.default', 'aten.expand.default', 'aten.ones.default', 'aten.scalar_tensor.default', 'aten.slice.Tensor', 'aten.t.default', 'aten.transpose.int']
C17  PASS   유도 상수 전부 설명됨, 구조 라이브러리에 등재됨
