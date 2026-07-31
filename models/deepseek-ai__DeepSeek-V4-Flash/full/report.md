# Extraction Report -- deepseek-ai/DeepSeek-V4-Flash @ 60d8d70770c6776ff598c94bb586a859a38244f1

C1   PASS   43 == 43
C2   PASS   4 clusters == 4 from config schedule ['layer_types', 'mlp_layer_types']
C3   PASS   acyclic, 0 orphan(s)
C4   PASS   embedding reachable from lm_head
C5   PASS   matmul contraction dims consistent; residual stream at d_model=4096 in 43/43 layers
C6   PASS   hidden_size=4096 (heuristic check, 16671 flagged)
C7   PASS   MQA (64 query heads : 1 kv head)
C8   WARN   MoE trace-verified [router_dim(E=256):ok, top_k(6):ok, expert_weight:grouped]; routed-token count is data-dependent/symbolic (01-main.md C8) -- WARN is normal, not a defect.
C9   PASS   vocab_size=129280, tie_word_embeddings=False
C10  PASS   all 1242 params covered
C11  PASS   340 cache-related op(s) found, new-token seq dim confirmed
C13  PASS   identical across two runs
C14  PASS   used=1032 >= required=1032
C15  WARN   config declares 1 MTP/nextn layer(s) but no MTP module in the traced model (native transformers impl omits the MTP head) -- MTP NOT traced
C16  INFO   22037 unmapped rows, 48 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', 'aten.alias.default', 'aten.arange.default', 'aten.bitwise_and.Tensor', 'aten.clamp.default', 'aten.clamp_.default', 'aten.clone.default', 'aten.copy_.default', 'aten.div.Tensor']
C17  PASS   유도 상수 전부 설명됨, 구조 라이브러리에 등재됨
