# Extraction Report -- tencent/Hunyuan-A13B-Instruct @ 290ddb9a56ed23c2c83a1c8081533e58925df952

C1   PASS   32 == 32
C2   PASS   1 clusters == 1 from config schedule ['moe_intermediate_size', 'moe_topk', 'num_shared_expert']
C3   PASS   acyclic, 0 orphan(s)
C4   PASS   embedding reachable from lm_head
C5   PASS   matmul contraction dims consistent; residual stream at d_model=4096 in 32/32 layers
C6   PASS   hidden_size=4096 (heuristic check, 3552 flagged)
C7   PASS   GQA 32:8 (repeat factor 4)
C8   WARN   MoE trace-verified [router_dim(E=64):ok, top_k([8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]):n/a, expert_weight:grouped]; routed-token count is data-dependent/symbolic (01-main.md C8) -- WARN is normal, not a defect.
C9   PASS   vocab_size=128167, tie_word_embeddings=True
C10  PASS   all 450 params covered
C11  PASS   129 cache-related op(s) found, new-token seq dim confirmed
C13  SKIP   pass --check-repro to actually run twice and verify
C14  PASS   used=24 >= required=24
C15  PASS   all discovered entrypoints traced
C16  INFO   6046 unmapped rows, 31 distinct raw ops: ['aten._local_scalar_dense.default', 'aten._to_copy.default', 'aten._unsafe_view.default', 'aten.alias.default', 'aten.arange.default', 'aten.clamp_.default', 'aten.clone.default', 'aten.div_.Tensor', 'aten.empty_like.default', 'aten.expand.default']
C17  PASS   유도 상수 전부 설명됨, 구조 라이브러리에 등재됨
