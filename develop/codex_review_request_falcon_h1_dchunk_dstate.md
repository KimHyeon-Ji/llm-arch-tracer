# Codex 검토 요청 — Falcon-H1 Mamba2 `d_chunk`/`d_state` 축 역추론 검증

## 배경

`tiiuae/Falcon-H1-7B-Instruct`는 `config.mamba_chunk_size == config.mamba_d_state == 256`이라
값만으로는 두 심볼을 가릴 수 없다. `review_request.md`의 0절(실제 조치 필요 표)에 이 tie로
남아있던 18행을 오늘 소스(`modeling_falcon_h1.py`의 `torch_forward`, Mamba2 SSD naive scan
구현)와 실측 트레이스 op_id 대조로 전부 판정해서 `rules/label_overrides.yaml`에 9건,
`rules/label_confirmed.yaml`에 16건을 추가했다. `develop/verify_all.py`는 FAIL 0 / 퇴행 0을
확인했지만, 이 게이트는 **판정이 자기 일관적인지**(매치 0건 아님, 등가류 개수 유지, 인용 존재)만
보지 **아키텍처 판단 자체가 맞는지**는 못 본다. 오늘 이미 별개 작업(`_unname_loop_indices` 정교화
시도, 무관한 함수)에서 스스로 확신했던 다단계 추론이 두 번 회귀로 되돌아간 전적이 있어서, 이번
Falcon-H1 작업 중 가장 추론 단계가 깊었던 자리만 짚어 검증을 요청한다.

## 소스 (전부 `develop/sources/modeling_falcon_h1.py`)

`torch_forward`의 SSD naive 구현, 674~878행. 관련 발췌:

```python
# 813행 근처
L = torch.exp(segment_sum(A))
G_intermediate = C[:, :, :, None, :, :] * B[:, :, None, :, :, :]  # shape: (b, c, l, s, h, n)
G = G_intermediate.sum(dim=-1)                                     # shape: (b, c, l, s, h)
M_intermediate = G[..., None] * L.permute(0, 2, 3, 4, 1)[..., None]
M = M_intermediate.sum(dim=-1)
Y_diag = (M[..., None] * hidden_states[:, :, None]).sum(dim=3)

# 828행 근처 (inter-chunk)
decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
B_decay = B * decay_states.permute(0, -2, -1, 1)[..., None]
states = (B_decay[..., None, :] * hidden_states[..., None]).sum(dim=2)
previous_states = ... torch.zeros_like(states[:, :1])
states = torch.cat([previous_states, states], dim=1)
decay_chunk = torch.exp(segment_sum(nn.functional.pad(A_cumsum[:, :, :, -1], (1, 0))))
decay_chunk = decay_chunk.transpose(1, 3)
new_states = (decay_chunk[..., None, None] * states[:, :, None, ...]).sum(dim=1)
states, ssm_state = new_states[:, :-1], new_states[:, -1]

state_decay_out = torch.exp(A_cumsum)
C_times_states = (C[..., None, :] * states[:, :, None, ...])
state_decay_out_permuted = state_decay_out.permute(0, 2, 3, 1)
Y_off = (C_times_states.sum(-1) * state_decay_out_permuted[..., None])
```

`B`/`C`는 청킹 후 shape `(b, c, l, h, n)` (n=`ssm_state_size`=`d_state`), `hidden_states`는
`(b, c, l, h, headdim)`, `A`/`A_cumsum`은 `(b, h, c, l)`. `c`=num_chunks(이 fleet 트레이스는
전부 seq_len < chunk_size라 `c`=1), `l`,`s`=청크 내부 위치(=`chunk_size`=`d_chunk`).

## 쓴 판단 원칙

`sum`/`permute`/`elementwise_mul`은 축 정체성을 **새로 만들 수 없다** — `sum`은 지정한 축만
없애고 나머지는 그대로, `permute`는 순서만 바꾸고, `elementwise_mul`은 브로드캐스트(`1`)만
허용하고 non-1인 쪽의 정체성이 그대로 출력에 남는다. 그래서 각 op의 **피연산자 중 이미
확실한(모호하지 않은) 쪽**을 근거로 출력 축을 역추적했다.

## 가장 확신이 낮은 두 자리 — 이것만 확인해달라

### 1) `elementwise_mul` (C_times_states, op_id 137 — `nth=9`)

트레이스: `[B,1,d_chunk,n_h_ssm,1,d_state] x [B,1,d_chunk,n_h_ssm,d_head_ssm,1] -> [B,1,d_state,n_h_ssm,d_head_ssm,d_chunk]` (렌더된 이름 기준, 출력 축 2와 축 5가 실제로는 **서로 뒤바뀌어** 렌더되고 있다고 판단함).

내가 넣은 교정: 축 2 `d_state`→`d_chunk`, 축 5 `d_chunk`→`d_state`.

이 op은 `C_times_states = (C[..., None, :] * states[:, :, None, ...])`라고 판단했다 —
`C`는 `(b,c,l,h,n)`에 `[..., None, :]`를 끼워 `(b,c,l,h,1,n)`, `states`는 `(b,c,h,headdim,n)`에
`[:,:,None,...]`를 끼워 `(b,c,1,h,headdim,n)`, 브로드캐스트하면 `(b,c,l,h,headdim,n)`이 된다.
이 매핑대로면 축2=`l`(=`d_chunk`), 축5=`n`(=`d_state`)이어야 하는데, 트레이스 입력 피연산자
자체의 라벨(위 두 shape)은 그렇게 안 읽힌다 — 첫 피연산자 축2가 `d_chunk`(브로드캐스트
아님)이고 둘째 피연산자 축2는 `n_h_ssm`이 아니라 그 앞에 뭔가 있어야 할 것 같은데 실측
shape가 `[B,1,d_chunk,n_h_ssm,d_head_ssm,1]`로 5개뿐이라 내가 그린 6D 모델과 축 개수부터
안 맞는다. **이 op이 정말 `C_times_states`가 맞는지, 아니면 다른 op(예: `Y_off`의 `sum(-1)` 전
단계, 또는 전혀 다른 중간 계산)인지부터 다시 봐달라.**

### 2) `elementwise_mul` (op_id 167 — `nth=11`)

트레이스: `[B,1,d_state,n_h_ssm,1,d_state] x [B,1,1,n_h_ssm,d_head_ssm,d_state] -> [B,1,d_state,n_h_ssm,d_head_ssm,d_chunk]` — 넣은 판정: 축2(`d_state`, 확인/유지), 축5(`d_chunk`→`d_state`, 교정).

이것도 위와 같은 `C_times_states` 패턴으로 판단했는데, 두 자리(`nth=9`, `nth=11`)가 왜
별개의 `nth`로 두 번 나오는지(같은 코드 줄이 두 번 실행되는 이유 -- 레이어가 달라서인지,
아니면 애초에 서로 다른 두 코드 줄인지)를 소스에서 확인 못 했다. **`nth=9`와 `nth=11`이
정말 같은 코드 줄(`C_times_states`)의 두 인스턴스인지, 아니면 하나는 `Y_off` 관련 다른
연산인지 확인해달라.**

## 실제로 넣은 규칙 전문

`rules/label_overrides.yaml`과 `rules/label_confirmed.yaml`에서 `tiiuae__Falcon-H1-7B-Instruct`
로 검색하면 오늘 추가한 항목(9개 override, 16개 confirm)이 전부 나온다. 각 항목에 `op_id`와
정확한 shape를 인용해뒀다.

## 질문

1. 위 두 자리(`nth=9`, `nth=11`)의 축 배정이 실제로 맞는가, 아니면 내가 엉뚱한 소스 줄에
   꿰맞춘 것인가?
2. 이 두 자리가 잘못됐다면, `spread: class`로 이미 함대(이 모델 92개 층 전부)에 적용됐으니
   되돌리는 방법도 같이 알려달라 — `from`/`to`를 맞바꾸면 되는지, 아니면 selector 자체가
   잘못된 자리를 가리키고 있어서 다시 짚어야 하는지.
3. 나머지 7개 자리(`sum nth=1`, `elementwise_mul nth=7`, `permute nth=2/3`, `constant_pad_nd
   nth=3/5`)는 "피연산자 중 브로드캐스트 아닌 쪽이 이긴다"는 훨씬 단순한 추론이라 상대적으로
   자신 있는데, 혹시 이것도 같이 훑어봐 줄 여유가 있다면 짚어달라 — 없으면 위 두 자리만
   봐줘도 충분하다.
