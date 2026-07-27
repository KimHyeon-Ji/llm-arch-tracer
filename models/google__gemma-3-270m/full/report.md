# Extraction Report -- google/gemma-3-270m @ 9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1

C1   PASS   18 == 18
C2   PASS   2 clusters == 2 from config schedule ['layer_types']
C3   PASS   acyclic, 0 orphan(s)
C4   PASS   embedding reachable from lm_head
C5   PASS   matmul contraction dims consistent; residual stream at d_model=640 in 18/18 layers
C6   PASS   hidden_size=640 (heuristic check, 1752 flagged)
C7   PASS   MQA (4 query heads : 1 kv head)
C8   SKIP   no MoE-related fields found on config (likely a dense model)
C9   PASS   vocab_size=262144, tie_word_embeddings=True
C10  PASS   all 236 params covered
C11  PASS   74 cache-related op(s) found, new-token seq dim confirmed
C13  PASS   identical across two runs
C14  PASS   used=1032 >= required=1032
C15  PASS   all discovered entrypoints traced
C16  INFO   1550 unmapped rows, 21 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', 'aten.alias.default', 'aten.arange.default', 'aten.bitwise_and.Tensor', 'aten.clone.default', 'aten.expand.default', 'aten.gt.Tensor', 'aten.le.Tensor', 'aten.lift_fresh.default']
C17  PASS   유도 상수 전부 설명됨, 구조 라이브러리에 등재됨
