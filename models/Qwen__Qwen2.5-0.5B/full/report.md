# Extraction Report -- Qwen/Qwen2.5-0.5B @ 060db6499f32faf8b98477b0a26969ef7d8b9987

C1   PASS   24 == 24
C2   PASS   1 clusters == 1 from config schedule ['layer_types']
C3   PASS   acyclic, 0 orphan(s)
C4   PASS   embedding reachable from lm_head
C5   PASS   matmul contraction dims consistent; residual stream at d_model=896 in 24/24 layers
C6   PASS   hidden_size=896 (heuristic check, 1776 flagged)
C7   PASS   GQA 14:2 (repeat factor 7)
C8   SKIP   no MoE-related fields found on config (likely a dense model)
C9   PASS   vocab_size=151936, tie_word_embeddings=True
C10  PASS   all 290 params covered
C11  PASS   97 cache-related op(s) found, new-token seq dim confirmed
C13  PASS   identical across two runs
C14  PASS   used=56 >= required=56
C15  PASS   all discovered entrypoints traced
C16  INFO   1675 unmapped rows, 14 distinct raw ops: ['aten._to_copy.default', 'aten._unsafe_view.default', 'aten.alias.default', 'aten.clone.default', 'aten.expand.default', 'aten.ones.default', 'aten.scalar_tensor.default', 'aten.slice.Tensor', 'aten.t.default', 'aten.transpose.int']
C17  PASS   유도 상수 전부 설명됨, 구조 라이브러리에 등재됨
