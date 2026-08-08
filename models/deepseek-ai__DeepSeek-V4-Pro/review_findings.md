# ③ 자유 평가 결과 (자동)

조사 안건 최대 항목(attn_hc/ffn_hc 의 [B,T,n_hc,n_hc], 28,496축)을 modeling 소스로 확인했다. 오류가 아니라 mHC 스트림 혼합 행렬이며, 정사각이므로 축 이름이 두 번 나오는 것이 정상이다. 이 결과로 research.py 의 중복-이름 탐지기에 정사각 행렬 예외를 넣었고, 함대의 코드-조사 대상이 52,687축 -> 1,752축으로 줄었다.

> 아래는 LLM 이 소스를 대조해 낸 판단이다. **재현 전에는 반영하지 않는다** — 근거 URL 을 직접 열어 확인한 뒤 규칙으로 승격한다.

| 모듈 | 축 | 현재 라벨 | 판정 | 제안 | 확신 | 근거 |
|---|---|---|---|---|---|---|
| `model.layers.*.attn_hc / ffn_hc` | `[B, T, n_hc, n_hc] 의 뒤 두 축` | `n_hc (두 번)` | current_label_correct | `-` | high | DeepseekV4HyperConnection.forward: comb_logits = comb_w.view(*comb_w.shape[:-1], hc, hc) * comb_scale + comb_b.view(hc,  (https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v4/modeling_deepseek_v4.py) |
