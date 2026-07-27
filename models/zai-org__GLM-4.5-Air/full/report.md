# Extraction Report -- zai-org/GLM-4.5-Air @ a24ceef6ce4f3536971efe9b778bdaa1bab18daa

C1   PASS   46 == 46
C2   WARN   2 cluster(s); no per-layer schedule list on config to compare (uniform, or scalar schedule like first_k_dense_replace)
C3   PASS   acyclic, 0 orphan(s)
C4   PASS   embedding reachable from lm_head
C5   PASS   matmul contraction dims consistent; residual stream at d_model=4096 in 46/46 layers
C6   PASS   hidden_size=4096 (heuristic check, 5800 flagged)
C7   PASS   GQA 96:8 (repeat factor 12)
C8   WARN   MoE trace-verified [router_dim(E=128):ok, top_k(8):ok, expert_weight:grouped]; routed-token count is data-dependent/symbolic (01-main.md C8) -- WARN is normal, not a defect.
C9   PASS   vocab_size=151552, tie_word_embeddings=False
C10  PASS   all 690 params covered
C11  PASS   277 cache-related op(s) found, new-token seq dim confirmed
C13  PASS   identical across two runs
C14  PASS   used=16 >= required=16
C15  WARN   config declares 1 MTP/nextn layer(s) but no MTP module in the traced model (native transformers impl omits the MTP head) -- MTP NOT traced
C16  INFO   5267 unmapped rows, 31 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', 'aten.alias.default', 'aten.arange.default', 'aten.bitwise_not.default', 'aten.clamp_.default', 'aten.clone.default', 'aten.div_.Tensor', 'aten.empty_like.default', 'aten.expand.default']
C17  PASS   유도 상수 전부 설명됨, 구조 라이브러리에 등재됨
