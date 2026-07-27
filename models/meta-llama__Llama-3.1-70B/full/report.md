# Extraction Report -- meta-llama/Llama-3.1-70B @ 349b2ddb53ce8f2849a6c168a81980ab25258dac

C1   PASS   80 == 80
C2   PASS   1 cluster(s); no per-layer schedule list on config to compare (uniform, or scalar schedule like first_k_dense_replace)
C3   PASS   acyclic, 0 orphan(s)
C4   PASS   embedding reachable from lm_head
C5   PASS   matmul contraction dims consistent; residual stream at d_model=8192 in 80/80 layers
C6   PASS   hidden_size=8192 (heuristic check, 5920 flagged)
C7   PASS   GQA 64:8 (repeat factor 8)
C8   SKIP   no MoE-related fields found on config (likely a dense model)
C9   PASS   vocab_size=128256, tie_word_embeddings=False
C10  PASS   all 723 params covered
C11  PASS   321 cache-related op(s) found, new-token seq dim confirmed
C13  PASS   identical across two runs
C14  PASS   used=16 >= required=16
C15  PASS   all discovered entrypoints traced
C16  INFO   5539 unmapped rows, 14 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', 'aten.alias.default', 'aten.clone.default', 'aten.expand.default', 'aten.ones.default', 'aten.scalar_tensor.default', 'aten.slice.Tensor', 'aten.t.default', 'aten.transpose.int']
C17  PASS   유도 상수 전부 설명됨, 구조 라이브러리에 등재됨
