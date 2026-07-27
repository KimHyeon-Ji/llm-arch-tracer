# Extraction Report -- tiiuae/falcon-7b @ ec89142b67d748a1865ea4451372db8313ada0d8

C1   PASS   32 == 32
C2   PASS   1 cluster(s); no per-layer schedule list on config to compare (uniform, or scalar schedule like first_k_dense_replace)
C3   PASS   acyclic, 0 orphan(s)
C4   PASS   embedding reachable from lm_head
C5   PASS   matmul contraction dims consistent; residual stream at d_model=4544 in 32/32 layers
C6   PASS   hidden_size=4544 (heuristic check, 160 flagged)
C7   PASS   MQA (71 query heads : 1 kv head)
C8   SKIP   no MoE-related fields found on config (likely a dense model)
C9   PASS   vocab_size=65024, tie_word_embeddings=True
C10  PASS   all 195 params covered
C11  PASS   129 cache-related op(s) found, new-token seq dim confirmed
C13  PASS   identical across two runs
C14  PASS   used=16 >= required=16
C15  PASS   all discovered entrypoints traced
C16  INFO   1698 unmapped rows, 21 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', 'aten.add_.Tensor', 'aten.alias.default', 'aten.arange.default', 'aten.bitwise_and.Tensor', 'aten.clone.default', 'aten.expand.default', 'aten.index.Tensor', 'aten.le.Tensor']
C17  PASS   유도 상수 전부 설명됨, 구조 라이브러리에 등재됨
