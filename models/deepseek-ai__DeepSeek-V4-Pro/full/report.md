# Extraction Report -- deepseek-ai/DeepSeek-V4-Pro @ b5968e9190ef611bbf34a7229255be88a0e937c1

C1   PASS   61 == 61
C2   PASS   4 clusters == 4 from config schedule ['layer_types', 'mlp_layer_types']
C3   PASS   acyclic, 0 orphan(s)
C4   PASS   embedding reachable from lm_head
C5   PASS   matmul contraction dims consistent; residual stream at d_model=7168 in 61/61 layers
C6   PASS   hidden_size=7168 (heuristic check, 24299 flagged)
C7   PASS   MQA (128 query heads : 1 kv head)
C8   WARN   MoE trace-verified [router_dim(E=384):ok, top_k(6):ok, expert_weight:grouped]; routed-token count is data-dependent/symbolic (01-main.md C8) -- WARN is normal, not a defect.
C9   PASS   vocab_size=129280, tie_word_embeddings=False
C10  PASS   all 1772 params covered
C11  PASS   426 cache-related op(s) found, new-token seq dim confirmed
C13  PASS   identical across two runs
C14  PASS   used=2048 >= required=2048
C15  WARN   config declares 1 MTP/nextn layer(s) but no MTP module in the traced model (native transformers impl omits the MTP head) -- MTP NOT traced
C16  INFO   31440 unmapped rows, 48 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', 'aten.alias.default', 'aten.arange.default', 'aten.bitwise_and.Tensor', 'aten.clamp.default', 'aten.clamp_.default', 'aten.clone.default', 'aten.copy_.default', 'aten.div.Tensor']
C17  PASS   유도 상수 전부 설명됨, 구조 라이브러리에 등재됨
