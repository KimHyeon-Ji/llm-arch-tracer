# Extraction Report -- deepseek-ai/DeepSeek-V2-Lite @ 604d5664dddd88a0433dbae533b7fe9472482de0

C1   PASS   27 == 27
C2   WARN   2 cluster(s); no per-layer schedule list on config to compare (uniform, or scalar schedule like first_k_dense_replace)
C3   PASS   acyclic, 0 orphan(s)
C4   PASS   embedding reachable from lm_head
C5   PASS   matmul contraction dims consistent; residual stream at d_model=2048 in 27/27 layers
C6   PASS   hidden_size=2048 (heuristic check, 2864 flagged)
C7   PASS   MHA (kv_heads == heads, not GQA)
C8   WARN   MoE trace-verified [router_dim(E=64):ok, top_k(6):ok, expert_weight:grouped]; routed-token count is data-dependent/symbolic (01-main.md C8) -- WARN is normal, not a defect.
C9   PASS   vocab_size=102400, tie_word_embeddings=False
C10  PASS   all 351 params covered
C11  PASS   108 cache-related op(s) found, new-token seq dim confirmed
C13  PASS   identical across two runs
C14  PASS   used=17 >= required=16
C15  PASS   all discovered entrypoints traced
C16  INFO   2744 unmapped rows, 29 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', 'aten.alias.default', 'aten.arange.default', 'aten.clamp_.default', 'aten.clone.default', 'aten.empty_like.default', 'aten.expand.default', 'aten.floor_divide.default', 'aten.ge.Scalar']
C17  PASS   유도 상수 전부 설명됨, 구조 라이브러리에 등재됨
