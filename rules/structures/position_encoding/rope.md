# RoPE (Rotary Position Embedding, 표준)

## 정의
Q/K에 위치에 따른 회전 변환을 적용해 상대적 위치 정보를 주입하는 방식. 별도의
학습 가능한 위치 임베딩 테이블 없이, `rope_theta` 기반으로 계산되는 cos/sin을
곱해서 적용한다.

## 관련 심볼 (rules/symbols.yaml)
아직 공통 심볼에 없음(모델마다 `rope_theta`, `rope_scaling` 필드명은 비교적
일관되지만 아직 §10 표에 추가 안 함 — 필요해지면 추가 검토).

## 트레이스에서 식별하는 방법
Q/K projection 직후, SDPA 이전에 `cos`/`sin` 곱셈과 회전(`mul` + `add`/`cat` 조합)이
매 레이어 반복 등장. 레이어마다 이 연산이 없으면(NoPE) 또는 압축 경로와 별도
경로로 나뉘어 있으면(decoupled RoPE, MLA류) — 둘 다 이 문서의 범위를 벗어나는
변형이라 develop/에서 만나면 별도 항목으로 추가.

## 확인된 모델 (계속 추가)
`models/` 24개 중 **22개가 RoPE**다. 예외 2개: `openai-community/gpt2-xl`(학습형 절대 위치
임베딩 `wpe` → [../attention/mha.md](../attention/mha.md)), `NX-AI/xLSTM-7b`(순환 구조라 위치
인코딩 자체가 없음). 표준형이 아닌 변형은 아래로 갈라지므로 여기엔 계보만 적는다.

| 변형 | 모델 |
|---|---|
| 표준 RoPE 전층 | Llama-3.1 8B/70B/405B, Qwen2.5, Qwen3-30B, Gemma-2/3, OLMo-2, OLMoE, GLM-4.5-Air, DeepSeek-V2-Lite/V3, Nemotron-3-Nano |
| **부분 RoPE**(head 차원 일부만 회전) | `Qwen/Qwen3-Next-80B`(`partial_rotary_factor`=0.25 → 256 중 64), `deepseek-ai/DeepSeek-V4-Pro`/`-Flash`(0.125 → 512 중 64) |
| **NoPE 인터리브** | SmolLM3, Llama-4-Maverick → [nope.md](nope.md) |
| **decoupled RoPE**(MLA 전용 소수 차원) | DeepSeek-V2-Lite / V3 → [../attention/mla.md](../attention/mla.md) |
| **rope-type별 이중 RoPE** | DeepSeek-V4: 코어 attention은 `main`(θ=10000), 압축 분기는 `compress`(θ=160000 + YaRN). `rope_parameters`가 `layer_types`가 아니라 **rope-type 라벨**(`main`/`compress`)로 키잉되는 첫 사례 |

부분 RoPE의 비회전 통과분(`d_head − d_rope`)은 `rules/derived_dims.yaml`에 식으로 등록돼 있어
유도 상수 표에 이름이 붙는다(V4 448, Qwen3-Next 192).

## 참고 소스
- Hugging Face transformers 공통 RoPE 구현
- RoFormer 논문(원 설계)
