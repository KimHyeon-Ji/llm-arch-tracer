# Extraction Report -- Qwen/Qwen3-30B-A3B @ ad44e777bcd18fa416d9da3bd8f70d33ebb85d39

C1   PASS   48 == 48
C2   PASS   1 cluster(s); no per-layer schedule list on config to compare (uniform, or scalar schedule like first_k_dense_replace)
C3   PASS   acyclic, 0 orphan(s)
C4   PASS   embedding reachable from lm_head
C5   PASS   matmul contraction dims consistent; residual stream at d_model=2048 in 48/48 layers
C6   PASS   hidden_size=2048 (heuristic check, 5472 flagged)
C7   PASS   GQA 32:4 (repeat factor 8)
C8   WARN   MoE trace-verified [router_dim(E=128):ok, top_k(8):ok, expert_weight:grouped]; routed-token count is data-dependent/symbolic (01-main.md C8) -- WARN is normal, not a defect.
C9   PASS   vocab_size=151936, tie_word_embeddings=False
C10  PASS   all 531 params covered
C11  PASS   193 cache-related op(s) found, new-token seq dim confirmed
C13  PASS   identical across two runs
C14  PASS   used=104 >= required=104
C15  PASS   all discovered entrypoints traced
C16  INFO   4531 unmapped rows, 26 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', 'aten.alias.default', 'aten.arange.default', 'aten.clamp_.default', 'aten.clone.default', 'aten.div_.Tensor', 'aten.empty_like.default', 'aten.expand.default', 'aten.floor_divide.default']
C17  PASS   유도 상수 전부 설명됨, 구조 라이브러리에 등재됨
