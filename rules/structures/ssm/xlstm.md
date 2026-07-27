# xLSTM / mLSTM (행렬 메모리 순환) — attention 없는 시퀀스 믹서

`ssm/` 카테고리(비-attention 순환 시퀀스 믹서)에 함께 둔다. 엄밀히는 상태공간(SSM)이 아니라
**행렬 메모리 LSTM(mLSTM)** 이지만, "attention이 아닌 순환 믹서"라는 축을 공유한다.

## 정의
self-attention이 **아예 없고**, 지수 게이팅(exponential input/forget gate)을 쓰는 mLSTM
블록만으로 시퀀스를 처리한다. KV cache 대신 **행렬 메모리 상태**를 굴린다. 이 파이프라인의
attention·KV-cache 가정이 "우아하게 degrade"하는지 보는 견고성 기준점(Phase 18).

## 트레이스에서 식별 (Phase 18 NX-AI/xLSTM-7b 실측)
- 블록명 `mlstm_layer`(+ `ffn`, `norm_mlstm`, `norm_ffn`). `self_attn`/softmax/`sdpa` **전무**.
- ops: `exp`(지수 게이트), `maximum`+`sub`(로그공간 수치 안정화, max-tracking), `batched_matmul`
  (행렬 메모리 K·V 상호작용), `select`/`unsqueeze`/`view` 다수(순환), `elementwise_mul/add`(게이팅),
  `log_sigmoid_forward`/`abs`. RMSNorm(pow/mean/rsqrt).
- scope: decoder stack이 `.blocks.N`이라 `_STACK_NAMES`의 `blocks`로 정상 라벨링(C1 32==32).

## 파이프라인 견고성 결과 (Phase 18의 진짜 목적)
attention 없는 모델에서도 **크래시·오탐 FAIL 없이** 동작:
- **C7 SKIP** "no attention-head field" — num_attention_heads 부재를 SKIP(정상).
- **C11 WARN** "no concat/cache-touching op in decode" — KV concat 기반 decode 가정이 안 맞음을
  FAIL이 아닌 WARN으로 표면화(mLSTM은 순환 상태 decode).
- **C5/C6/C10 PASS** — 잔차 스트림 d_model=4096 32/32층 유지, 483 params 전부 커버.
- model_summary의 "attention" 항목은 `? (attention-free, e.g. SSM/xLSTM)`, KV cache 항목도
  `? (attention-free)`로 정직하게 표기(derive_architecture의 attention-free 분기).

## 확인된 모델
- **`NX-AI/xLSTM-7b`** (Phase 18): 32 mLSTM 블록, hidden 4096, attention 없음. C1 32==32,
  C5 32/32, C10 483 params, C7 SKIP, C11 WARN, C13 repro. config가 triton 커널을 지정해 native
  config가 거부 → 프로파일 `config_overrides`로 native 커널(chunkwise--native_autograd 등) 강제
  (triton은 meta에서 못 돌고 아키텍처 불변). revision `9dc507bd0939cf372a4a4f667335651d8e49dddb`.

## 참고 소스
- transformers `models/xlstm` 구현(native 커널) — 트레이스로 직접 관측
- xLSTM 논문, Raschka's LLM Architecture Gallery(비-attention 계보; 교차검증용)
