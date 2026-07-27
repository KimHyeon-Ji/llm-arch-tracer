# Extraction Report -- openai-community/gpt2-xl @ 15ea56dee5df4983c59b2538573817e1667135e2

C1   PASS   48 == 48
C2   PASS   1 cluster(s); no per-layer schedule list on config to compare (uniform, or scalar schedule like first_k_dense_replace)
C3   PASS   acyclic, 0 orphan(s)
C4   PASS   embedding reachable from lm_head
C5   PASS   matmul contraction dims consistent; residual stream at d_model=1600 in 48/48 layers
C6   PASS   hidden_size=1600 (heuristic check, 528 flagged)
C7   PASS   MHA (kv_heads == heads, not GQA)
C8   SKIP   no MoE-related fields found on config (likely a dense model)
C9   PASS   vocab_size=50257, tie_word_embeddings=True
C10  PASS   all 580 params covered
C11  PASS   96 cache-related op(s) found, new-token seq dim confirmed
C13  PASS   identical across two runs
C14  PASS   used=16 >= required=16
C15  PASS   all discovered entrypoints traced
C16  INFO   1638 unmapped rows, 12 distinct raw ops: ['aten._unsafe_view.default', 'aten.alias.default', 'aten.clone.default', 'aten.expand.default', 'aten.ones.default', 'aten.scalar_tensor.default', 'aten.split.Tensor', 'aten.t.default', 'aten.transpose.int', 'aten.tril.default']
C17  PASS   유도 상수 전부 설명됨, 구조 라이브러리에 등재됨
