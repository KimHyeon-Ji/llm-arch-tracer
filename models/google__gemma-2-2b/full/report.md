# Extraction Report -- google/gemma-2-2b @ c5ebcd40d208330abc697524c919956e692655cf

C1   PASS   26 == 26
C2   PASS   2 clusters == 2 from config schedule ['layer_types']
C3   PASS   acyclic, 0 orphan(s)
C4   PASS   embedding reachable from lm_head
C5   PASS   matmul contraction dims consistent; residual stream at d_model=2304 in 26/26 layers
C6   PASS   hidden_size=2304 (heuristic check, 1950 flagged)
C7   PASS   GQA 8:4 (repeat factor 2)
C8   SKIP   no MoE-related fields found on config (likely a dense model)
C9   PASS   vocab_size=256000, tie_word_embeddings=True
C10  PASS   all 288 params covered
C11  PASS   105 cache-related op(s) found, new-token seq dim confirmed
C13  PASS   identical across two runs
C14  PASS   used=2048 >= required=2048
C15  PASS   all discovered entrypoints traced
C16  INFO   1628 unmapped rows, 16 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', 'aten.alias.default', 'aten.clone.default', 'aten.div.Tensor', 'aten.expand.default', 'aten.lift_fresh.default', 'aten.ones.default', 'aten.scalar_tensor.default', 'aten.slice.Tensor']
C17  PASS   유도 상수 전부 설명됨, 구조 라이브러리에 등재됨
