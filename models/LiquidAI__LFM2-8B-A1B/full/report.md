# Extraction Report -- LiquidAI/LFM2-8B-A1B @ c1c44ff9fc00db3ebf4516970563f5f383d23670

C1   PASS   24 == 24
C2   WARN   3 trace clusters vs 2 config-schedule signatures ['layer_types'] -- review (mask-only heterogeneity is op-invisible)
C3   PASS   acyclic, 0 orphan(s)
C4   PASS   embedding reachable from lm_head
C5   PASS   matmul contraction dims consistent; residual stream at d_model=2048 in 24/24 layers
C6   PASS   hidden_size=2048 (heuristic check, 444 flagged)
C7   PASS   GQA 32:8 (repeat factor 4)
C8   WARN   MoE trace-verified [router_dim(E=32):ok, top_k(4):ok, expert_weight:grouped]; routed-token count is data-dependent/symbolic (01-main.md C8) -- WARN is normal, not a defect.
C9   PASS   vocab_size=65536, tie_word_embeddings=True
C10  PASS   all 212 params covered
C11  PASS   43 cache-related op(s) found, new-token seq dim confirmed
C13  SKIP   pass --check-repro to actually run twice and verify
C14  PASS   used=16 >= required=16
C15  PASS   all discovered entrypoints traced
C16  INFO   1351 unmapped rows, 29 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', 'aten.alias.default', 'aten.arange.default', 'aten.clamp_.default', 'aten.clone.default', 'aten.constant_pad_nd.default', 'aten.copy_.default', 'aten.div.Tensor', 'aten.empty_like.default']
C17  PASS   유도 상수 전부 설명됨, 구조 라이브러리에 등재됨
