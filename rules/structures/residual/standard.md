# 표준 잔차 연결 (elementwise add)

## 정의
블록의 출력을 입력에 그대로 더하는 가장 기본적인 residual connection.
`hidden = hidden + block(hidden)` 형태.

## 관련 심볼 (rules/symbols.yaml)
없음(구조 자체가 `d_model` 차원의 단일 add).

## 트레이스에서 식별하는 방법
블록 경계에서 `elementwise_add` 단일 op으로 잔차가 처리됨. 두 피연산자의 shape이
`d_model` 기준으로 동일(`01-main.md` C5). 여러 스트림을 섞는 믹싱 op이 대신
등장하면 표준이 아닌 변형(develop/에서 만나면 별도 항목으로 추가).

## 확인된 모델 (계속 추가)
`models/` 24개 중 **22개가 표준 잔차**(`h = h + sublayer(norm(h))` 단일 elementwise add).
예약 최종테스트 중에서는 Llama-3.1 70B/405B, gpt-oss 20b/120b, Llama-4-Maverick이 여기 해당.

**예외 2개**:
- `deepseek-ai/DeepSeek-V4-Pro` / `-Flash` — mHC(다중 스트림 잔차 + Sinkhorn 이중확률 사영).
  잔차가 `[B,T,d_model]`이 아니라 `[B,T,n_hc,d_model]`이다 → [mhc.md](mhc.md)
- (SSM/xLSTM 계열은 블록 내부 구조가 다르지만 블록 간 잔차 자체는 표준 add다)

## 참고 소스
- Hugging Face transformers 공통 decoder layer 구현
