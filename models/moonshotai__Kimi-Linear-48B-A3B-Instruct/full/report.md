# Extraction Report -- moonshotai/Kimi-Linear-48B-A3B-Instruct @ e1df551a447157d4658b573f9a695d57658590e9

C1   PASS   27 == 27
C2   WARN   3 cluster(s); no per-layer schedule list on config to compare (uniform, or scalar schedule like first_k_dense_replace)
C3   PASS   acyclic, 0 orphan(s)
C4   PASS   embedding reachable from lm_head
C5   PASS   matmul contraction dims consistent; residual stream at d_model=2304 in 27/27 layers
C6   PASS   hidden_size=2304 (heuristic check, 12813 flagged)
C7   PASS   MHA (kv_heads == heads, not GQA)
C8   WARN   MoE trace-verified [router_dim(E=256):ok, top_k(None):n/a, expert_weight:grouped]; routed-token count is data-dependent/symbolic (01-main.md C8) -- WARN is normal, not a defect.
C9   PASS   vocab_size=163840, tie_word_embeddings=False
C10  WARN   all params covered except 19656 excluded by a documented remedy (see adaptation_log) -- not a coverage miss
C11  PASS   114 cache-related op(s) found, new-token seq dim confirmed
C13  SKIP   pass --check-repro to actually run twice and verify
C14  PASS   used=16 >= required=16
C15  PASS   all discovered entrypoints traced
C16  INFO   11786 unmapped rows, 39 distinct raw ops: ['aten._local_scalar_dense.default', 'aten._to_copy.default', 'aten._unsafe_view.default', 'aten.add_.Tensor', 'aten.alias.default', 'aten.arange.default', 'aten.bitwise_not.default', 'aten.clamp_min.default', 'aten.clone.default', 'aten.copy_.default']
C17  PASS   유도 상수 전부 설명됨, 구조 라이브러리에 등재됨
