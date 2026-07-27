# Extraction Report -- meta-llama/Llama-4-Maverick-17B-128E @ 10751cb97a4d7c90f7ed89196b98eb8220cfa1c2

C1   PASS   48 == 48
C2   WARN   3 trace clusters vs 2 config-schedule signatures ['layer_types', 'no_rope_layers'] -- review (mask-only heterogeneity is op-invisible)
C3   PASS   acyclic, 0 orphan(s)
C4   PASS   embedding reachable from lm_head
C5   PASS   matmul contraction dims consistent; residual stream at d_model=5120 in 48/48 layers
C6   PASS   hidden_size=5120 (heuristic check, 3156 flagged)
C7   PASS   GQA 40:8 (repeat factor 5)
C8   WARN   MoE trace-verified [router_dim(E=128):ok, top_k(1):ok, expert_weight:grouped]; routed-token count is data-dependent/symbolic (01-main.md C8) -- WARN is normal, not a defect.
C9   PASS   vocab_size=202048, tie_word_embeddings=False
C10  PASS   all 507 params covered
C11  PASS   96 cache-related op(s) found, new-token seq dim confirmed
C13  PASS   identical across two runs
C14  PASS   used=16 >= required=16
C15  PASS   all discovered entrypoints traced
C16  INFO   4076 unmapped rows, 30 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', 'aten.add_.Tensor', 'aten.alias.default', 'aten.arange.default', 'aten.clone.default', 'aten.div.Tensor', 'aten.expand.default', 'aten.floor.default', 'aten.full_like.default']
C17  PASS   유도 상수 전부 설명됨, 구조 라이브러리에 등재됨
