# RMSNorm (표준)

## 정의
평균을 빼지 않고 제곱평균으로만 정규화하는 정규화 방식. LayerNorm보다 계산이
간단해 대부분의 현대 LLM이 기본으로 채택.

## 관련 심볼 (rules/symbols.yaml)
없음(정규화 자체는 `d_model` 차원에 적용되는 elementwise 연산이라 별도 구조 심볼
불필요).

## 트레이스에서 식별하는 방법
`pow` → `mean` → `add`(eps) → `rsqrt` → `mul`(정규화) → `mul`(학습 가능 scale) 순서의
op 시퀀스. 이 시퀀스가 attention/FFN 앞뒤에 반복해서 나타남(pre-norm 구조가 흔함).

## 확인된 모델 (계속 추가)
`models/` 24개 중 **23개가 RMSNorm**. 유일한 예외는 `openai-community/gpt2-xl`(LayerNorm,
트레이스에 `native_layer_norm`으로 잡힘). 예약 최종테스트 7개(Llama-3.1 70B/405B,
gpt-oss 20b/120b, DeepSeek-V4 Pro/Flash, Llama-4-Maverick) 전부 RMSNorm이다.

변형 메모:
- **weightless RMSNorm**: DeepSeek-V4의 `DeepseekV4UnweightedRMSNorm`(mHC 입력 정규화와
  q_b_norm에 사용). `pow/mean/rsqrt/mul`은 나오는데 대응하는 weight 파라미터가 **없다** —
  C10 커버리지에서 이상이 아니다. → [../residual/mhc.md](../residual/mhc.md)
- **QK-Norm**(q/k에 별도 RMSNorm): OLMo-2 → [qk-norm.md](qk-norm.md)

## 참고 소스
- Hugging Face transformers 공통 RMSNorm 구현
