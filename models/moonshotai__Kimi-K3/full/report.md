# Extraction Report -- moonshotai/Kimi-K3 @ a590ce090cb049c93a33dfe8c208ec652aa20503

C1   PASS   93 == 93
C2   WARN   4 cluster(s); no per-layer schedule list on config to compare (uniform, or scalar schedule like first_k_dense_replace)
C3   PASS   acyclic, 0 orphan(s)
C4   PASS   embedding reachable from lm_head
C5   PASS   matmul contraction dims consistent; residual stream at d_model=7168 in 93/93 layers
C6   PASS   hidden_size=7168 (heuristic check, 610622 flagged)
C7   PASS   MHA (kv_heads == heads, not GQA)
C8   WARN   MoE trace-verified [router_dim(E=896):ok, top_k(None):n/a, expert_weight:grouped]; routed-token count is data-dependent/symbolic (01-main.md C8) -- WARN is normal, not a defect.
C9   PASS   vocab_size=163840, tie_word_embeddings=False
C10  WARN   all params covered except 246194 excluded by a documented remedy (see adaptation_log) -- not a coverage miss
C11  PASS   1050 cache-related op(s) found, new-token seq dim confirmed
C13  SKIP   pass --check-repro to actually run twice and verify
C14  PASS   used=320 >= required=16
C15  PASS   all discovered entrypoints traced
C16  INFO   526144 unmapped rows, 42 distinct raw ops: ['aten._local_scalar_dense.default', 'aten._to_copy.default', 'aten._unsafe_view.default', 'aten.add_.Tensor', 'aten.alias.default', 'aten.arange.default', 'aten.clamp_min.default', 'aten.clone.default', 'aten.copy_.default', 'aten.div.Tensor']
C17  WARN   미해결 유도 상수 1개 [1280] -- rules/derived_dims.yaml에 식+출처 등록 필요; 남은 축별 안건은 models/<model>/research_agenda.md 참고
