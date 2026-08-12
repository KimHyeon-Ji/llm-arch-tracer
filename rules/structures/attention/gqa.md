# Grouped-Query Attention (GQA)

## 정의
K/V head 수(`n_kv`)를 Q head 수(`n_h`)보다 적게 둬서 KV cache 크기를 줄이는 방식.
`n_h / n_kv`개의 Q head가 K/V head 하나를 공유(`repeat_kv`로 확장).

## 관련 심볼 (rules/symbols.yaml)
`n_h`, `n_kv`, `d_head`. repeat 배수 = `n_h / n_kv`.

## 트레이스에서 식별하는 방법
- k_proj/v_proj weight_shape 1축 = `n_kv * d_head` (q_proj보다 작음)
- Q와 K/V 사이에 `expand`/`repeat_kv` 계열 op이 SDPA 직전에 존재
- `01-main.md` §9.2 C7이 이 정합성을 자동 검사

## 확인된 모델 (계속 추가)
- `Qwen/Qwen2.5-0.5B` — 14 Q heads : 2 KV heads (7:1)
- `meta-llama/Llama-3.1-8B` — 32 Q heads : 8 KV heads (4:1)
- `google/gemma-2-2b` — 8 Q heads : 4 KV heads (2:1), 26 layers.
  sliding/full 하이브리드지만 major view에서는 한 블록으로 접힌다(sliding과 full은 **mask만**
  다르고 op 시퀀스가 같다 — 지연시간 관점에서 올바른 축약)
- `google/gemma-3-270m` — 4 Q heads : 1 KV head (4:1), 18 layers, `d_head`=256, `d_model`=640.
  `_sliding_window_pattern`=6 → sliding 5층마다 full 1층, `w_local`=512.
  유도 상수 511 = `w_local − 1`(sliding mask 밴드 폭)

예약 최종테스트(2026-07-23/27):
- `meta-llama/Llama-3.1-70B` — 64:8 (8:1), 80 layers
- `meta-llama/Llama-3.1-405B` — 128:8 (16:1), 126 layers, `d_model`=16384.
  8B와 동일 아키텍처의 순수 스케일업이라 GQA 비율만 커진다
- `meta-llama/Llama-4-Maverick-17B-128E` — 40:8 (5:1), `d_head`=128
- `openai/gpt-oss-20b` / `-120b` — 64:8 (8:1), `d_head`=64.
  sliding/full 교대 + [attention-sink.md](attention-sink.md)

**MQA(n_kv=1)는 GQA의 극단**이며 별도 문서를 두지 않았다. `deepseek-ai/DeepSeek-V4-Pro`/`-Flash`가
여기 해당하는데, V4는 단순 MQA가 아니라 **shared-KV MQA(K와 V가 같은 텐서) + 블록 압축 분기**
조합이다 → [csa.md](csa.md), [hca.md](hca.md). C7은 repeat 배수 = `n_h`로 통과한다.

## 참고 소스
- 각 모델 Hugging Face config.json (`num_attention_heads`, `num_key_value_heads`)
- Raschka's LLM Architecture Gallery — GQA 도입 시점·채택 모델 비교

- **`mistralai/Mistral-Small-3.2-24B-Instruct-2506`** (Phase 31): 40 layers, `n_h`=32, `n_kv`=8,
  `d_head`=128, `d_ff`=32768. 순수 GQA dense — 새 규칙 0개로 들어왔다.
- **`microsoft/Phi-4`** (Phase 38): 40 layers, `n_h`=40, `n_kv`=10, `d_head`=128, `d_ff`=17920.
  `phi3` 네이티브 구현, fused QKV. 새 규칙 0개.
- **`MiniMaxAI/MiniMax-M2`** (Phase 32): 62 layers, `n_h`=48, `n_kv`=8, `d_head`=128 + MoE
  (`E`=256, `k`=8). 새 벤더 계열인데 새 규칙 0개.
- **`tencent/Hunyuan-A13B-Instruct`** (Phase 35): 32 layers, `n_h`=32, `n_kv`=8, `E`=64.
  config 가 `moe_topk`/`num_experts` 를 **레이어별 리스트**로 적는다 — src/run.py 의 활성
  파라미터 추정이 리스트를 나눠 죽던 것을 평균으로 접어 고쳤다(2026-08-12).
- **`LiquidAI/LFM2-8B-A1B`** (Phase 36): 24 layers, `n_h`=32, `n_kv`=8, `d_head`=64, MoE
  (`E`=32, `k`=4). short conv + attention 하이브리드. 새 규칙 0개.
