# QK-Norm + norm 배치(pre/post) 변형

두 개의 독립적인 정규화 축을 함께 다룬다(Phase 11 OLMo-2에서 동시 등장).

## (1) QK-Norm
### 정의
attention 계산 전에 **Q와 K에 각각 정규화(주로 RMSNorm)**를 적용해 학습 안정성을 높이는
기법. 표준 attention은 q_proj/k_proj 출력을 바로 쓰지만, QK-Norm은 그 사이에 norm을 끼운다.

### 트레이스에서 식별
self_attn 블록 안에 `q_norm`/`k_norm`(또는 `q_layernorm`/`k_layernorm`) 모듈이 있고, q/k
projection 직후 RMSNorm op 시퀀스(`pow`/`mean`/`rsqrt`/`mul` 또는 `rms_norm`)가 head 차원에
걸린다. 결과적으로 layer당 RMSNorm 인스턴스가 표준(2개: 입력/FFN)보다 많아진다(OLMo-2는 4개:
q_norm, k_norm, + post-attn, post-ffn).

## (2) norm 배치: pre-norm vs post-norm(-inside-residual)
### 정의
- **pre-norm**(대부분): `x + sublayer(norm(x))` — 서브레이어 입력에 norm(모듈명 흔히
  `input_layernorm`, `post_attention_layernorm`이 "다음 블록의 pre-norm" 역할).
- **post-norm(OLMo-2형)**: `x + norm(sublayer(x))` — 서브레이어 **출력**에 norm을 걸어 residual에
  더한다. 모듈명이 `post_attention_layernorm`/`post_feedforward_layernorm`이고 `input_layernorm`이
  없다.

### 트레이스에서 식별 / C5와의 관계
norm op이 sublayer의 matmul들 **뒤**, residual `elementwise_add` **앞**에 위치한다. 어느
배치든 residual 스트림 폭은 `d_model`로 보존되므로 **C5(연결 불변식)는 두 배치 모두 PASS**한다
(Phase 11: post-norm인데도 residual d_model=4096 32/32층 유지). 배치 차이는 op 순서로만 드러나며,
C2 클러스터링/구조 롤업의 op 시퀀스에 반영된다.

## 관련 심볼
정규화는 별도 shape 심볼이 없다(가중치는 `[d_model]` 또는 head 차원 `[d_head]`). RMSNorm 자체는
[rmsnorm.md](rmsnorm.md) 참고.

## 확인된 모델
- **`allenai/OLMo-2-1124-7B-Instruct`** (Phase 11): 32 layers, 순수 MHA(`n_h`=32), **QK-Norm**
  (`q_norm`/`k_norm`) + **post-norm**(`post_attention_layernorm`/`post_feedforward_layernorm`,
  `input_layernorm` 없음), 전부 RMSNorm. C5/C6/C7 PASS, C13 repro.
  revision 확인: develop/logs/phase11_olmo2-7b.log.

## 참고 소스
- transformers `models/olmo2` 구현 — 트레이스로 직접 관측(q_norm/k_norm, post_*_layernorm 모듈)
- OLMo-2 tech report / Raschka's LLM Architecture Gallery(정규화 배치 계보 비교, 교차검증용)
