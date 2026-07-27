# Extraction Report -- nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 @ dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f

C1   PASS   42 == 42
C2   PASS   3 clusters == 3 from config schedule ['layers_block_type']
C3   PASS   acyclic, 0 orphan(s)
C4   PASS   embedding reachable from lm_head
C5   PASS   matmul contraction dims consistent; residual stream at d_model=3136 in 42/42 layers
C6   PASS   hidden_size=3136 (heuristic check, 0 flagged)
C7   PASS   GQA 40:8 (repeat factor 5)
C8   WARN   config has num_experts=8 but NO expert params or router/expert ops in trace -- model is dense here (field appears vestigial): ['router_dim(E=8):MISSING', 'top_k(2):n/a', 'expert_weight:MISSING']
C9   PASS   vocab_size=131072, tie_word_embeddings=False
C10  PASS   all 263 params covered
C11  PASS   29 cache-related op(s) found, new-token seq dim confirmed
C13  PASS   identical across two runs
C14  PASS   used=24 >= required=24
C15  PASS   all discovered entrypoints traced
C16  INFO   2741 unmapped rows, 28 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', 'aten.alias.default', 'aten.bitwise_not.default', 'aten.clamp.default', 'aten.clone.default', 'aten.constant_pad_nd.default', 'aten.copy_.default', 'aten.expand.default', 'aten.masked_fill.Scalar']
C17  PASS   유도 상수 전부 설명됨, 구조 라이브러리에 등재됨
