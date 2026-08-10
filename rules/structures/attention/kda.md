# KDA — Kimi Delta Attention

Moonshot의 선형 어텐션. Gated DeltaNet 계열이며, 고정 크기 상태 행렬을 delta rule로 굴리고
KV 캐시를 쓰지 않는다. 레이어 스케줄이 이종이라 full attention 층과 섞인다.

## config에서 식별하는 방법

`linear_attn_config` 딕셔너리가 이 아키텍처의 표식이다:

```json
{"kda_layers": [1,2,3,5,...], "full_attn_layers": [4,8,12,...],
 "num_heads": 32, "head_dim": 128, "short_conv_kernel_size": 4}
```

| 모델 | 총 층 | KDA | full |
|---|---|---|---|
| `moonshotai/Kimi-Linear-48B-A3B-Instruct` | 27 | 20 | 7 |
| `moonshotai/Kimi-K3` | 93 | 69 | 24 |

## 트레이스 상태 — **미확보**, 이유는 MoE 디스패치

두 모델 다 `models/` 에 없다. KDA 자체는 문제가 아니다 — `src/kda_shim.py` 로 fla의 **자체 torch
참조 구현**(`naive_chunk_kda` / `naive_recurrent_kda` / `naive_kda_gate`)을 태워 실제 shape까지
도달하는 것을 확인했다(K3에서 `q = [1, 320, 96, 128]`). 막는 것은 MoE 쪽이다:

```python
tokens_per_expert = tokens_per_expert.cpu().numpy()
for i, num_tokens in enumerate(tokens_per_expert): ...     # modeling_kimi.py:754
```

**전문가별로 배정된 토큰 수를 호스트로 가져와 파이썬 루프의 경계로 쓴다.** 가중치가 없으면
라우팅 결과가 없고, FakeTensor에는 읽을 값이 없다. 모델 코드에 다른 경로도 없다 —
`forward` 의 학습 분기는 `NotImplementedError`.

**이건 아키텍처의 성질이 아니라 그 저장소 구현의 성질이다.** 같은 벤더의 `Kimi-K2-Instruct` 는
전문가 384개 MoE인데도 **새 규칙 0개, 휴리스틱 0.00%** 로 깨끗이 트레이스된다 — 유지되는
`deepseek_v3` 구현으로 돌고, 그쪽 라우팅은 scatter/gather + grouped matmul 로 **디바이스 안에서**
끝나기 때문이다.

### 언제 풀리나

- `kimi_linear` / `kimi_k3` 가 transformers 본체에 들어오면 (본체 구현은 관례상 on-device 라우팅)
- 또는 저장소 구현이 호스트 전송을 걷어내면

둘 중 하나가 되면 **지금 있는 것만으로 트레이스된다.** KDA 쪽 준비는 끝나 있다.

## 참고

- 트레이스가 되면 shape은 Triton 커널이 아니라 **fla 참조 구현**에서 나온 것이므로,
  `review_findings.json` 에 `status:"open"` 으로 남겨 요약 카드에 표시해야 한다.
- Triton 커널은 `TorchDispatchMode` 에 보이지 않는다(커널 런치 하나로 지나간다). 그래서 참조
  구현으로 도는 것은 우회가 아니라 **내부를 볼 수 있는 유일한 방법**이다 — Mamba2·Gated
  DeltaNet·xLSTM 도 같은 이유로 torch 경로를 탄다.
