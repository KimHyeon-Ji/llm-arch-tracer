# 검토 요청 — Kimi-K3 후속: 수정 확인 + 반박 판단 + 남은 2건

지난 검토(9개 지적)를 저희 쪽에서 하나씩 소스/트레이스로 직접 재확인한 뒤 반영했습니다.
결과를 요약하니, (1) 반영한 3건이 맞는지, (2) 반박한 2건에 대한 재반박이 있는지, (3) 아직
못 고친 2건에 실마리를 주실 수 있는지 봐주시면 좋겠습니다. 이번에도 **파이프라인 코드는
보지 마시고**, 아래 서술과 공개 소스만으로 판단해 주세요.

## 1. 반영한 것 (직접 검증 후 수정 완료)

### RoPE → NoPE
말씀하신 대로 `mla_use_nope=True`가 실제로 회전을 끄고 있다는 걸 **트레이스로 직접 재확인**
했습니다 — MLA 레이어(config 4번째 레이어) 전체에서 `cos`/`sin` 연산이 단 한 번도 없고,
`d_rope` 조각은 분리 → head 차원으로 broadcast → 그대로 concat될 뿐, 그 사이에 회전에
해당하는 곱셈이 없습니다. `rope_theta=10000.0`은 config에 있지만 실제로 안 읽히는 죽은
값으로 결론 내렸습니다.

### SwiGLU → SiTU-GLU
`hidden_act="situ"`, `activation_situ_beta=4.0`, `activation_situ_linear_beta=25.0` 전부
확인했습니다. 저희 쪽 활성함수 판정 로직이 트레이스 전체에서 "silu 연산이 있는가"를
전역으로 검사하는데, FFN과 무관한 다른 곳(KDA의 자체 게이트)이 silu를 쓰고 있어서
FFN도 SwiGLU로 잘못 판정됐던 것으로 확인했습니다. FFN 자체(dense 레이어든 MoE
전문가든)는 실제로 tanh+sigmoid 연산만 씁니다.

### KV cache 104.6 → 27.0 KiB
정확히 지적하신 원인을 코드에서 찾았습니다 — "진짜 attention 레이어" 판정이 `q_proj`라는
서브모듈 이름의 유무로 되어 있는데, **KDA 레이어도 자체적으로 `q_proj`라는 이름의
프로젝션을 갖고 있어서**(`model.layers.0.self_attn.q_proj`, layer 0은 KDA 확정) 93개
레이어 전부가 "진짜 attention"으로 잘못 세어지고 있었습니다. 24개로 정정했고, 결과가
`27.0 KiB (Low) over 24 attn layers`로 나와 말씀하신 계산과 정확히 일치합니다.

## 2. 반박한 것 — 재반박 있으면 알려주세요

### "attention 커널 = eager" 그대로 둠
이 필드는 "이 모델이 실제 배포에서 어떤 커널을 쓰는가"가 아니라 **저희 트레이스 방식이
관찰한 것**을 뜻합니다. meta/fake device 트레이스는 항상 explicit softmax 경로로
캡처되기 때문에, 26개 모델 **전부**가 예외 없이 "eager (explicit softmax)"라고 씁니다.
Kimi-K3만 "MLA: FlashAttention2, KDA: FLA chunk_kda"로 바꾸면 이 필드의 의미 자체가
모델마다 달라지는 셈이라 반영하지 않았습니다.

### "Attention: MLA" 헤드라인을 "Hybrid: KDA+MLA"로 안 바꿈
이미 있는 다른 하이브리드 모델들의 관례를 확인했습니다 — Qwen3-Next는 linear_attention이
섞여 있어도 "Attention: GQA"만 씁니다, Zamba2도 linear_attention이 다수인데 "Attention:
MHA"만 씁니다. 두 모델 다 하이브리드 비율은 별도의 LAYER MIX 행이 담당합니다(저희도
"LAYER MIX: 69× KDA, 24× MLA"로 이미 명시). 기존 관례와 일관성을 위해 헤드라인은
그대로 뒀습니다.

## 3. 아직 못 고친 것 — 도와주시면 좋겠습니다

### `d_head=74`가 유령 값이고, 파생값이 진짜 KDA 축을 잘못 설명함

config에 `head_dim=74`라는 필드가 있긴 한데, 저희 트레이스 **전체**에서 `d_head`(=74)라는
이름이 실제 축에 단 한 번도 안 씁니다 — 완전히 미사용 필드로 보입니다. 문제는 여기서
파생된 두 표현이 **실제로 존재하는 축**을 설명하는 데 쓰이고 있다는 겁니다:

- 값 10 (= 74−64, "d_head − d_rope"라고 저희가 잘못 표시) — 실측 위치:
  `model.layers.0.self_attn`(KDA 확정 레이어), shape `[B, n_h_kda, 5, 10]`
- 값 37 (= 74/2, "d_head/2"라고 저희가 잘못 표시) — 비슷하게 self_attn 안

**질문**: KDA(`KimiDeltaAttention`)의 실제 forward 코드에서, 이 10과 37이라는 폭이 진짜로
뭘 의미하나요? (`b_proj`/`g_proj`/`f_a_proj`/`f_b_proj` 같은 KDA 자체 프로젝션들의 코드를
봐주시면 좋겠습니다.) 정확한 이름을 모르고 추측으로 넣으면 오늘 c_I/2 때와 똑같은 실수를
반복하는 거라, 확인 전엔 등록하지 않기로 했습니다.

### `E_shared*d_moe`(6144)와 `n_h*d_rope`(값이 같은 다른 자리)를 사람이 읽는 요약 한 줄로
어떻게 구분해야 하나

실제 표(CSV)의 축별 라벨은 이미 모듈별로 정확히 구분돼 있습니다(shared-expert는
`E_shared*d_moe`, MLA attention은 `n_h*d_v`, KDA attention은 `n_h_kda*d_head_kda`). 다만
사람이 읽는 요약 문서의 "유도 상수" 범례 표는 **같은 숫자 값 하나당 설명 하나만** 보여주게
설계돼 있어서, 여러 모듈이 우연히 같은 값을 가지면 그중 하나의 설명만 대표로 뜨고 나머지
모듈 이름도 그 옆에 나열됩니다(오해 소지가 있는 표시 방식이지, 실제 데이터가 틀린 건
아닙니다). 이건 이번엔 구조 변경 없이 미해결로 남겼습니다 — 혹시 더 나은 표시 방법이
있다면 제안 부탁드립니다(필수는 아닙니다).

## 첨부: 수정된 `model_summary.md`의 관련 부분만

```
| 5 | Attention | MLA |
| 6 | LAYER MIX | 69× KDA, 24× MLA  (FFN: 1 dense + 92 MoE) |
| 7 | KV CACHE / TOKEN (BF16) | 27.0 KiB (Low) over 24 attn layers |
| 8 | KEY DETAIL | MLA attention; Sparse MoE (E=896, top-16, +2 shared, sigmoid gating/aux-loss-free); dense-prefix 1 layer(s) |
| 9 | Related concepts | RMSNorm, MLA, MoE, shared expert, sigmoid-gating, short-conv (SSM/DeltaNet) |
...
| 위치 인코딩 | none observed (NoPE, or position handled implicitly) |
| FFN | MoE — 896 routed experts, top-16 + 2 shared, expert intermediate 3072, SiTU-GLU (tanh+sigmoid gate, β=4.0, β_linear=25.0) |
```

## 답변 형식

- 반영한 3건: 문제없으면 짧게 "확인" 정도로 충분합니다.
- 반박한 2건: 동의하시면 짧게, 재반박 있으면 근거와 함께.
- 남은 2건: 아는 만큼만 주셔도 됩니다.
