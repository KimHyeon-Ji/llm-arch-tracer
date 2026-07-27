# Attention Sink (head별 학습형 softmax 분모 바이어스)

## 정의
head마다 **학습 가능한 스칼라 하나**를 attention score 행렬에 한 열로 더 붙여, softmax가
"아무 토큰에도 주의를 주지 않는" 선택지를 갖게 하는 장치. OpenAI gpt-oss 모델 카드 표현으로는
*"learned attention sink per-head, where the denominator of the softmax has an additional additive
value"* — 즉 분모에만 항이 하나 더 붙는다.

```
scores      [B, n_h, T, kv_len]
sinks       [n_h]                    ← 학습 파라미터
  → view/expand [B, n_h, T, 1]
  → cat(dim=-1) [B, n_h, T, kv_len+1]
  → softmax
  → 마지막 열(sink)은 출력 가중합에서 버림
```

sink 열은 값을 싣지 않으므로(대응하는 V가 없음) softmax 확률질량 중 sink로 간 몫은 그대로
**버려진다**. 결과적으로 각 head의 출력이 "합이 1인 가중평균"이 아니라 **합 ≤ 1**이 되어,
attend할 게 없을 때 억지로 어딘가에 질량을 몰아주는 현상(attention sink 문제)을 완화한다.

기존 "첫 토큰이 sink 역할을 하더라" 라는 *관찰*을 아키텍처로 **명시화**한 것이라고 보면 된다.

## 관련 심볼 (rules/symbols.yaml)
`n_h` (sink 파라미터 길이), `w_local`, `layer_sched`. sink 자체는 차원이 아니라 파라미터라
별도 심볼이 없다 — 트레이스에서는 `kv_len + 1`이라는 **+1**로만 드러난다.

## 트레이스에서 식별하는 방법
`self_attn.sinks` 파라미터를 params에 달고 있는 `view` → `expand` → `cat` 3연속이 지문이다.
`cat`의 출력 마지막 축이 입력보다 정확히 **1 크다**.

**gpt-oss-20b 실측** (`aten.cat.default`, params=`model.layers.N.self_attn.sinks`):
| op_id | op | shape |
|---|---|---|
| 146 | `view` | `[n_h]` → `[B, n_h, B, B]` |
| 147 | `expand` | → `[B, n_h, T, B]` |
| 148 | `concat` | `[B,n_h,T,T] ⊕ [B,n_h,T,B]` → **`[B, n_h, T, T+1]`** |

DeepSeek-V4는 압축 KV를 먼저 concat하므로 `+1`이 `T`가 아니라 압축 포함 길이에 붙는다:
`[B,n_h,T,2560] → [B,n_h,T,2561]` (= `T + T/m_csa + 1`). 그래서 V4의 유도 상수에
2065·2561 같은 홀수가 나온다 — [csa.md](csa.md), [hca.md](hca.md) 참고.

**주의**: `T+1`은 decode 단계의 KV 캐시 길이로도 나온다. sink인지 구분하려면 그 op의
`params`에 `sinks`가 달렸는지를 봐야 한다.

## KV cache 영향
없다. sink는 **레이어당 `n_h`개 스칼라**로 끝이고 토큰마다 쌓이지 않는다. 갤러리의
KV cache/token 계산에도 들어가지 않는다.

## 확인된 모델 (계속 추가)
- **`openai/gpt-oss-20b`** (예약 최종테스트, 2026-07-23): 24층, `n_h`=64, `n_kv`=8(GQA 8:1),
  `d_head`=64, `d_model`=2880, sliding/full **교대**(`layer_types` = sliding 12 + full 12),
  `w_local`=128, MoE `E`=32/top-4, `d_moe`=2880, swiglu_limit=7.0.
  전 레이어에 `self_attn.sinks` `[64]` 존재.
- **`openai/gpt-oss-120b`** (예약 최종테스트, 2026-07-23): 36층(sliding 18 + full 18),
  나머지 attention 제원은 20b와 동일, MoE `E`=128/top-4.
- **`deepseek-ai/DeepSeek-V4-Pro` / `-Flash`** (예약 최종테스트, 2026-07-23): 동일 기법.
  transformers `deepseek_v4` 구현 주석이 *"Per-head learnable attention sink like gpt OSS"*로
  직접 계보를 밝힌다. 압축 KV concat 뒤에 붙는 점만 다르다.

## 참고 소스
- **1차** transformers 5.14.1 `models/gpt_oss/modeling_gpt_oss.py`, `models/deepseek_v4/` —
  트레이스로 직접 관측
- **1차** [gpt-oss-120b & gpt-oss-20b Model Card (arXiv:2508.10925)](https://arxiv.org/html/2508.10925v1)
- [Hugging Face — Welcome GPT OSS](https://huggingface.co/blog/welcome-openai-gpt-oss):
  *"Alternate attention layers: full-context, and sliding 128-token window"*,
  *"Learned attention sink per-head, where the denominator of the softmax has an additional
  additive value"*
- [vLLM Blog — vLLM Now Supports gpt-oss](https://vllm.ai/blog/2025-08-05-gpt-oss): sink를 지원하는
  전용 커널(FlashAttention 3 / FlashInfer)이 따로 필요하다는 점 — 독립 구현 관점의 교차검증
