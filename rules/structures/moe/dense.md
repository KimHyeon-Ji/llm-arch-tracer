# Dense FFN (MoE 아닌 기본형)

## 정의
라우팅 없이 모든 토큰이 같은 FFN(up-projection → 활성화 → down-projection, 필요시
gate 계열로 게이트) 하나를 통과하는 기본형. MoE는 이 블록을 여러 개 두고 라우팅을
추가한 변형으로 볼 수 있다.

## 관련 심볼 (rules/symbols.yaml)
`d_ff`.

## 트레이스에서 식별하는 방법
`mlp`/`ffn` block 안에 라우팅 관련 op(topk, gather, scatter 등)이 전혀 없고,
linear → 활성화 → linear 패턴만 반복. 이게 없으면(즉 라우팅 op이 있으면) MoE.

## 확인된 모델 (계속 추가)
- (develop/ 진행에 따라 추가)

## 참고 소스
- Hugging Face transformers 공통 FFN 구현
