# Extraction Report -- Qwen/Qwen3-Next-80B-A3B-Instruct @ 9c7f2fbe84465e40164a94cc16cd30b6999b0cc7

C1   PASS   48 == 48
C2   PASS   2 clusters == 2 from config schedule ['layer_types']
C3   PASS   acyclic, 0 orphan(s)
C4   PASS   embedding reachable from lm_head
C5   PASS   matmul contraction dims consistent; residual stream at d_model=2048 in 48/48 layers
C6   PASS   hidden_size=2048 (heuristic check, 3000 flagged)
C7   PASS   GQA 16:2 (repeat factor 8)
C8   WARN   MoE trace-verified [router_dim(E=512):ok, top_k(10):ok, expert_weight:grouped]; routed-token count is data-dependent/symbolic (01-main.md C8) -- WARN is normal, not a defect.
C9   PASS   vocab_size=151936, tie_word_embeddings=False
C10  PASS   all 759 params covered
C11  PASS   145 cache-related op(s) found, new-token seq dim confirmed
C13  PASS   identical across two runs
C14  PASS   used=17 >= required=16
C15  PASS   all discovered entrypoints traced
C16  INFO   33812 unmapped rows, 37 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', 'aten.alias.default', 'aten.arange.default', 'aten.clamp_.default', 'aten.clone.default', 'aten.constant_pad_nd.default', 'aten.copy_.default', 'aten.div_.Tensor', 'aten.empty_like.default']
C17  PASS   유도 상수 전부 설명됨, 구조 라이브러리에 등재됨
