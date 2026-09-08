# 라운드 5 판정 — Codex CSV/JSONL 라벨 지적 8건

근거 소스: 설치된 transformers 5.14.1 + 각 모델의 `<phase>.csv` 실측. 판정일 2026-09-05.

**8건 전부 CONFIRMED, 오탐 0.** 다만 두 건은 범위를 좁혔고, 두 건은 교정이 아니라
규칙/도구로 옮겼다.

| # | 주장 | 판정 | 처리 |
|---|---|---|---|
| 1 | 공통 MoE `unsqueeze(-1)` 의 마지막 축이 `B` | CONFIRMED | **라벨러 수정** |
| 2 | DeepSeek-V3 gate `sum` 입력 `k` → `n_grp` | CONFIRMED | override 1 |
| 3 | Kimi-K3 transpose `n_h_kda`/`d_head_kda` → `n_h`/`d_v` | CONFIRMED | override 2 |
| 4 | Kimi-Linear `n_h*d_v` → `n_h_kda*d_head_kda` | CONFIRMED, **범위 축소** | override 102 |
| 5 | GLM-4.5-Air `E` → `k*T` | CONFIRMED | **T=24 probe** |
| 6 | Llama-4 `reshape(-1,1)` 의 마지막 축 | CONFIRMED, **범위 축소** | override 3 |
| 7 | Mamba `self.D[..., None]` 의 마지막 축이 `B` | CONFIRMED | **라벨러 수정** |
| 8 | Granite / Nemotron-Super `d_state` ↔ `n_h_ssm` | CONFIRMED | override 6 |

## #1 · #7 — 개별 교정이 아니라 라벨러 결함이었다

두 지적은 같은 결함의 두 얼굴이다. `symbolic_shape.dim()` 은 크기-1 축에 `B` 를 답하고,
`resolve_shape` 가 "앞에 이미 B 나 T 가 있으면 1 로 내린다"로 대부분을 잡는다. 그런데 앞 축이
`k*T` 나 `n_h_ssm` 처럼 **B·T 라는 토큰 자체는 아닌** 이름이면 그 검사를 그냥 지나간다.

소스(설치본):

    integrations/moe.py:394  sample_weights = top_k_weights.reshape(-1)   # (S,)
    integrations/moe.py:426  (expert_ids_g >= self.num_experts).unsqueeze(-1)
    integrations/moe.py:468  proj_out * sample_weights_g.unsqueeze(-1)    # (S, hidden_dim)
    modeling_zamba2.py:798 / granitemoehybrid:640,656 / nemotron_h:494,510 /
    falcon_h1:784,800        self.D[..., None]

### 불변식을 먼저 재 봤다

함대 전체를 재 보고 나서야 규칙을 정했다:

| 측정 | 결과 |
|---|---|
| `unsqueeze` 가 **축 0** 에 끼운 경우 | 1,045행, **전부 `B`** — 정당 (`position_ids.unsqueeze(0)`) |
| `unsqueeze` 가 **0번이 아닌 자리**에 끼운 경우 | 5,399행, **전부 `B`** — 다른 이름 0건 |
| 참고: "B 는 축 0 이어야 한다"는 더 넓은 규칙 | **307,958행**이 걸린다 → 무효 |

세 번째 줄이 중요하다. `[n_h, B, d_head]` 처럼 전치 뒤 배치가 1번 축에 오는 **정당한** 배치가
훨씬 많다. 재 보지 않고 넣었으면 대형 회귀였다. 판별의 근거는 **그 축이 이 op 에서 새로
생겼다**는 사실 하나뿐이다.

반영: `src/build_table.py` `_unsqueeze_inserts_singleton` (교정보다 앞, 모든 추론보다 뒤) +
`develop/verify_all.py` 의 `batch_excl` 검사 + `verify_selftest.py` 주입 케이스.
Codex 가 언급하지 않은 conv1d 가중치 형태 183행도 같은 결함이라 함께 잡힌다.

끼운 자리가 유일하지 않은 경우(`[128,1] -> [128,1,1]`, decode 의 `self.D[..., None]` 80행)도
**후보가 전부 0번이 아니면** 어느 쪽이 새 축이든 배치일 수 없으므로 함께 고친다.

## #4 — Codex 범위가 넓었다. `o_proj` 는 두 층 타입이 공유한다

`n_h*d_v` 가 붙은 자리는 103개다. 그중 **97개는 KDA 층 전용**이지만 `self_attn.o_proj` 의
6개는 MLA 층과 `module_key` 를 공유한다. 그리고 **MLA 층에서는 `n_h*d_v` 가 맞다**
(o_proj 의 입력 = num_heads × v_head_dim). 전부 바꿨으면 7개 층을 망가뜨렸을 것이다.

Kimi-Linear 는 `layer_types` 를 쓰지 않고 `linear_attn_config.full_attn_layers` 에 **1-based
인덱스 목록**으로 적는다. `src/label_overrides.py` 의 `_schedule()` 이 그 목록도 층 스케줄로
펴도록 배선하고, `layer_types: [linear_attention]` 으로 KDA 층에만 걸었다.

**이 102개 항목은 재생성할 수 없다** — 한 번 반영하면 트레이스에 `n_h*d_v` 가 남지 않아
생성기가 0건을 낸다. YAML 의 목록이 원본이다(주석에 적어 뒀다).

## #6 — 마지막 축의 리터럴 1 만. `E → E*B` 는 물렀다

`modeling_llama4.py:169-170` 의 `router_scores.transpose(0,1).reshape(-1,1)` 은 마지막 축이
명시적 1 이 맞다. 그러나 축 0 을 `E → E*B` 로도 고쳤더니 **등가류 충돌 24건**이 났다:
decode 는 B = 1 이라 `E*B == E` 이고, 같은 클래스 안에 `routed_out.reshape(E, -1, d_model)` 의
**진짜 `E` 축**이 들어 있다. 마지막 축만 고친다 — 축 0 의 `E` 는 B = 1 일 때 참이다.

## #8 — A_log 사슬은 한 등가류다

`A = -torch.exp(self.A_log.float())   # [num_heads]` (granitemoehybrid:588, nemotron_h:444).
`d_state = n_h_ssm = 128` 인 모델은 이 둘뿐이라는 것도 실측으로 확인했다.

처음에 `_to_copy`/`exp`/`neg`/`elementwise_add` 자리마다 항목을 두었더니 **충돌 108/280건**이
났다 — 그 사슬이 하나의 등가류이기 때문이다. 대표 하나에 `spread: class` 를 걸어 사슬 전체를
끌고 가는 것으로 바꿨다. **V4-Pro 에서 얻은 교훈이 그대로 재현됐다: 옮기는 단위는 자리가
아니라 등가류다.**

## #5 — probe 가 두 번째로 통했다

`E == k*T` 인 모델은 GLM-4.5-Air 하나뿐이라는 것을 함대 전체에서 확인했다(T=16, k=8).
`k*T` 는 **T 에서 유도되는 축**이므로 V4-Pro 에 쓴 multi-length probe 가 그대로 통한다.
T=24 면 `k*T = 192 != 128`. 클래스 765건 교정 → 대표 항목 17개.

Codex 가 짚은 예외(`histc` 의 출력과 `grouped_matmul` 의 weight/offsets 은 진짜 전문가 축)는
probe 에서 128 로 남았고, 도구의 **반증 요구** 관문이 자동으로 걸렀다 — 손으로 예외를 적지
않아도 됐다.

## 인용 하나를 정정한다

#3 에서 인용하신 `modeling_kimi_linear.py:330` 은 **설치된 transformers 5.14.1 에 없다**
(`models/kimi_k25` 만 있다). 다만 결론은 유효하다:
`transpose [B, n_h, T, d_v] -> [B, T, n_h_kda, d_head_kda]` 는 재배열 불변식만으로 판정되는
자체 모순이라 소스가 필요 없다. (기록: 증거는 **실행된 판본**이어야 한다.)

## 마지막 질문 — lineage 제안

주신 네 가지 중 **전치본을 stride/storage 로 판별**하는 것과 **`Cache.update` 인자 슬롯을
축 lineage 로 전파**하는 것은 남은 미해결 3건에 직접 맞는다. 트레이서 코어를 건드리는
변경이라 이번 라운드에는 넣지 않고 별도 작업으로 남긴다.
