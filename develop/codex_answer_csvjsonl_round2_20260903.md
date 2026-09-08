# Codex 2차 라운드 답변 (2026-09-03) — 원문 보존

1차 라운드 수정 반영 후 재검토 결과. 아래는 받은 원문을 그대로 보존한 것이며,
검증 판정은 develop/verdicts_round2_*.md 에 별도로 기록한다.

## 이상 없음으로 통과한 모델 (1차 수정이 옳았음을 확인)

MiniMaxAI__MiniMax-M2, NX-AI__xLSTM-7b, allenai__OLMo-2-1124-7B-Instruct,
allenai__OLMoE-1B-7B-0924, bzantium__tiny-deepseek-v3, deepseek-ai__DeepSeek-V2-Lite,
openai-community__gpt2-xl, deepseek-ai__DeepSeek-V3, meta-llama__Llama-3.1-405B,
tencent__Hunyuan-A13B-Instruct, tiiuae__Falcon-H1-7B-Instruct

## 신규 지적 요약 (상세는 아래 원문)

| 배치 | 건수 | 성격 |
|---|---|---|
| A | 2 | Qwen3-Next: decode 의 fused-Q split, recurrent sum/select/copy_ |
| B | 3 | Qwen3.5-397B decode fused-Q, Qwen3.5-4B recurrent(prefill+decode) |
| C | 4 | Qwen3.6-27B/35B recurrent + 35B decode fused-Q |
| D | 7 | Zamba2 attention n_h→n_kv, Mamba head/feature 축 교차, tiny-Llama |
| E | 7 | Granite Mamba 전 구간 head/state 축 교차 |
| F·G | 8 | V4-Flash 2종 RoPE 잔여 자리 |
| H | 8 | Kimi 3종 k_rot, **Nemotron-Nano 신규 3건** |
| I | 11 | Nemotron Super/Ultra mixer 전 구간 |
| J | 5 | GLM-5.2, **J1 은 신규(query projection)** |

**공통 패턴: 1차 수정이 옳았으나 같은 결함의 나머지 자리에 닿지 못했다.**

---

(원문은 대화 기록 참조 — 소스 근거 태그 Q-Attn / Q-GDN / Z-Attn / Z-SSM / L-KV /
G-SSM / V4-RoPE / MLA-RoPE / N-SSM / N-KV / GLM-MLA / GLM-I 와 배치별 표)
