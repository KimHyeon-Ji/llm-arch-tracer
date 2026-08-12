# Extraction Report -- baidu/ERNIE-4.5-21B-A3B-PT @ 87db95487941cb39592ee0abca3b9155a6d19c5c

C1   PASS   28 == 28
C2   WARN   2 cluster(s); no per-layer schedule list on config to compare (uniform, or scalar schedule like first_k_dense_replace)
C3   PASS   acyclic, 0 orphan(s)
C4   PASS   embedding reachable from lm_head
C5   PASS   matmul contraction dims consistent; residual stream at d_model=2560 in 28/28 layers
C6   PASS   hidden_size=2560 (heuristic check, 3136 flagged)
C7   PASS   GQA 20:4 (repeat factor 5)
C8   WARN   MoE trace-verified [router_dim(E=64):ok, top_k(6):ok, expert_weight:grouped]; routed-token count is data-dependent/symbolic (01-main.md C8) -- WARN is normal, not a defect.
C9   PASS   vocab_size=103424, tie_word_embeddings=True
C10  PASS   all 362 params covered
C11  PASS   57 cache-related op(s) found, new-token seq dim confirmed
C13  SKIP   pass --check-repro to actually run twice and verify
C14  PASS   used=16 >= required=16
C15  WARN   config declares 1 MTP/nextn layer(s) but no MTP module in the traced model (native transformers impl omits the MTP head) -- MTP NOT traced
C16  INFO   3007 unmapped rows, 29 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', 'aten.alias.default', 'aten.arange.default', 'aten.clamp.default', 'aten.clamp_.default', 'aten.clone.default', 'aten.div.Tensor', 'aten.empty_like.default', 'aten.expand.default']
C17  PASS   유도 상수 전부 설명됨, 구조 라이브러리에 등재됨
