# Extraction Report -- MiniMaxAI/MiniMax-M2 @ 757303d492a50514c312788b5247a4f696a4c6a3

C1   PASS   62 == 62
C2   PASS   1 clusters == 1 from config schedule ['attn_type_list']
C3   PASS   acyclic, 0 orphan(s)
C4   PASS   embedding reachable from lm_head
C5   PASS   matmul contraction dims consistent; residual stream at d_model=3072 in 62/62 layers
C6   PASS   hidden_size=3072 (heuristic check, 6944 flagged)
C7   PASS   GQA 48:8 (repeat factor 6)
C8   WARN   MoE trace-verified [router_dim(E=256):ok, top_k(8):ok, expert_weight:grouped]; routed-token count is data-dependent/symbolic (01-main.md C8) -- WARN is normal, not a defect.
C9   PASS   vocab_size=200064, tie_word_embeddings=False
C10  PASS   all 685 params covered
C11  PASS   373 cache-related op(s) found, new-token seq dim confirmed
C13  SKIP   pass --check-repro to actually run twice and verify
C14  PASS   used=16 >= required=16
C15  PASS   all discovered entrypoints traced
C16  INFO   5681 unmapped rows, 26 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', 'aten.alias.default', 'aten.arange.default', 'aten.clamp_.default', 'aten.clone.default', 'aten.div_.Tensor', 'aten.empty_like.default', 'aten.expand.default', 'aten.floor_divide.default']
C17  PASS   유도 상수 전부 설명됨, 구조 라이브러리에 등재됨
