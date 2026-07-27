# Multi-Head Attention (MHA)

## 정의
Q/K/V head 수가 모두 같은 표준 attention. GQA/MQA의 특수(기저) 케이스로 봐도 된다
(`n_kv == n_h`).

## 관련 심볼 (rules/symbols.yaml)
`n_h`, `n_kv`(= `n_h`), `d_head`.

## 트레이스에서 식별하는 방법
q/k/v projection의 `weight_shape` out 축이 전부 `n_h * d_head`로 동일(GPT-2는 c_attn 하나로
qkv를 합쳐 뽑은 뒤 `split`). `repeat_kv`류 확장 op이 없음(있으면 GQA). config에
`num_key_value_heads`가 없으면 kv=heads로 간주(C7·model_summary 동일 규칙).

## 확인된 모델 (계속 추가)
- **`openai-community/gpt2-xl`** (Phase 10): 48 layers, 순수 MHA `n_h`=25, `d_head`=64,
  GQA/MQA 아님. 학습형 절대 위치 임베딩(`wpe`, RoPE 아님 → [../position_encoding/](../position_encoding/)),
  qkv 합침(`c_attn`)+`split`, gelu-tanh 근사 FFN(단일 gelu op 아님 → activation 미표기).
  scope 파싱은 decoder stack이 `transformer.h.N`(‘layers’ 아님)이라 `_STACK_NAMES`에 `h`를
  넣어 지원(Phase 10에서 scope.py 일반화). tie embeddings. C1 48==48, C6/C7 PASS, C13 repro.
  revision `15ea56dee5df4983c59b2538573817e1667135e2`.

## 참고 소스
- Hugging Face transformers `models/gpt2` 구현 — 트레이스로 직접 관측
- Raschka's LLM Architecture Gallery (GQA/MQA와의 비교 도표)
