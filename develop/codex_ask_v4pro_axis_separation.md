DeepSeek-V4-Pro 한 모델의 축 분리 문제에 대해 **방법**을 여쭙습니다.
라벨 목록이 아니라 "이 두 축을 무엇으로 가를 수 있는가"에 대한 조언을 구합니다.

## 상황

`deepseek-ai__DeepSeek-V4-Pro` 는 prefill 에서 `d_head == T/m_csa == 512` 입니다
(head_dim=512, T=2048, compress_rate=4 → n_windows=512).

지난 라운드에 주신 지적(compressor/indexer/scorer 의 window 축이 `d_head` 로 붙어 있다)은
소스로 확인했습니다. `modeling_deepseek_v4.py:646-657`:

    n_windows = chunk_kv.shape[1] // self.compress_rate
    chunk_kv  = chunk_kv.view(batch, n_windows, ratio, -1)
    new_kv    = chunk_kv.new_zeros((batch, n_windows, 2 * ratio, self.head_dim))

축 1 = n_windows, 마지막 축 = head_dim 이 맞습니다.

## 저희 쪽에서 무슨 일이 있었나

교정을 넣었더니 **HCA/KV 의 feature 축까지 번져** 원래 맞던 `d_head` 가 `T/m_csa` 로
바뀌었습니다(이것이 지난 라운드에 지적해 주신 6·7·10·11번입니다). 되돌렸고, 되돌린 뒤
decode 는 이미 정확합니다:

    op 327 unsqueeze compressor: [B, T/m_hca, d_head]   구체 [1, 16, 512]
    op 328 concat    self_attn : [B,1,w_local,d_head] + [B,1,T/m_hca,d_head]
                              -> [B,1,w_local+T/m_hca,d_head]

(참고로 6·7·10·11 에 제안해 주신 `T/m_csa`(=512)는 이 축들의 실제 정수가 16 이라
수치가 맞지 않습니다. 맞는 이름은 `T/m_hca`(=16)이고 현재 그렇게 돼 있습니다.)

## 왜 못 고쳤나 — 도구 쪽 한계

저희 도구는 "축 등가류"를 만들어 같은 축인 자리들을 하나로 묶고, 등가류마다 이름을 한 번만
정합니다. 등가류 간선은 **구체 shape 이 일치하는 생산자→소비자** 관계로 놓입니다.

`d_head == T/m_csa == 512` 이므로 **구조가 다른 두 텐서가 같은 shape 으로 보이고**,
CSA window 축과 KV feature 축이 하나의 등가류로 묶입니다. 그래서:

- 등가류를 따라 고치면 → 반대쪽(원래 맞던 `d_head`)이 오염됩니다.
- 등가류를 무시하고 자리별로 고치면 → "한 축에 이름이 둘"이라는 게이트 검사가 FAIL 합니다.

즉 **트레이스에 남은 정보만으로는 두 축을 가를 수 없습니다.** prefill 의 compressor 안
`d_head` 라벨 13,710 자리는 전부 구체값 512 로 수치적으로는 참입니다.

## 질문

이 두 축을 갈라줄 **소스 수준의 판별 근거**가 있을까요? 구체적으로:

1. `n_windows` 축과 `head_dim` 축이 **서로 다른 크기가 되는 config** 가 V4 계열에 있습니까?
   (있다면 그 config 로 한 번 더 트레이스해서 두 축을 분리한 뒤, 그 결과를 512 config 에
   옮길 수 있습니다. 저희가 prefill/decode 두 phase 를 대조해 라우팅 슬롯 충돌을 푼 것과
   같은 방법입니다.)

2. 그게 없다면, **둘 중 한쪽만 통과하는 연산**이 있습니까?
   예를 들어 window 축만 `arange`/`position_bias`/`softmax(dim=2)` 를 거친다든지,
   feature 축만 `kv_norm` 이나 rotary 를 거친다든지 — 모듈·op 로 스코프를 좁힐 수 있는
   비대칭이면 충분합니다.

3. 1·2 가 모두 없다면, 이 자리는 트레이스 정보만으로 원리적으로 판정 불가능하다고 보고
   저희가 **미해결로 문서화**하는 것이 맞을까요?

## 참고 — 같은 성격의 미해결 3건

같은 뿌리(등가류가 서로 다른 두 축을 묶음)의 문제가 다른 모델에도 있습니다.
혹시 공통으로 쓸 수 있는 판별법이 보이면 함께 말씀해 주시면 좋겠습니다.

| 모델 | 묶이는 두 축 | 값 |
|---|---|---|
| deepseek-ai__DeepSeek-V4-Pro | CSA window ↔ HCA/KV feature | 512 |
| zai-org__GLM-5.2 | 캐시 key ↔ value | 256 |
| Zyphra__Zamba2-1.2B | mamba head ↔ head feature | 64 |
| MiniMax-M2 / OLMoE / V4-Flash 2종 | experts gate-up 원본 ↔ 전치본 | d_model = 2*d_moe |
