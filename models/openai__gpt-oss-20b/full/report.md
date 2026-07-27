# Extraction Report -- openai/gpt-oss-20b @ 6cee5e81ee83917806bbde320786a8fb61efebee

C1   PASS   24 == 24
C2   PASS   2 clusters == 2 from config schedule ['layer_types']
C3   PASS   acyclic, 0 orphan(s)
C4   PASS   embedding reachable from lm_head
C5   PASS   matmul contraction dims consistent; residual stream at d_model=2880 in 24/24 layers
C6   PASS   hidden_size=2880 (heuristic check, 2208 flagged)
C7   PASS   GQA 64:8 (repeat factor 8)
C8   WARN   MoE trace-verified [router_dim(E=32):ok, top_k(4):ok, expert_weight:grouped]; routed-token count is data-dependent/symbolic (01-main.md C8) -- WARN is normal, not a defect.
C9   PASS   vocab_size=201088, tie_word_embeddings=False
C10  PASS   all 411 params covered
C11  PASS   120 cache-related op(s) found, new-token seq dim confirmed
C13  PASS   identical across two runs
C14  PASS   used=264 >= required=264
C15  PASS   all discovered entrypoints traced
C16  INFO   2152 unmapped rows, 32 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', 'aten.add_.Tensor', 'aten.alias.default', 'aten.arange.default', 'aten.bitwise_and.Tensor', 'aten.clamp.default', 'aten.clamp_.default', 'aten.clone.default', 'aten.empty_like.default']
C17  PASS   유도 상수 전부 설명됨, 구조 라이브러리에 등재됨
