# 라벨 검토 결과 — ibm-granite/granite-4.0-h-small

- 검토일: 2026-08-13
- 검토자: llm(claude, 반박 프레임 전건 판정)
- 본 것: 의뢰서 항목을 **항목 단위로** 대조해 하나도 빠뜨리지 않는다(src/review_ledger.unanswered_items 가 개수가 아니라 항목을 맞춘다). 각 항목마다 그 폭을 만드는 코드 줄을 열어 확인했다.
- 요약: 미답 항목 2건을 소스로 판정했다.

> 이 파일은 `review_findings.json` 에서 생성된다 — 고칠 때는 JSON 을 고친다.

## 발견 1 — 교정 필요 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.mamba` |
| 축 | Mamba d_inner / conv_dim / projection_size |
| 현재 라벨 | `미해결 정수 3개` |
| 판정 | `should_be_renamed` |
| 제안 라벨 | `d_inner / d_inner+2*n_g*d_state / d_inner+conv_dim+n_h_ssm` |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

Granite 4.0 은 `mamba_n_heads` / `mamba_d_head` 로 쓰는데 별칭 표에 없어서 `d_inner`(=n_h_ssm·d_head_ssm=8192)가 안 풀렸고, 그 위에 얹힌 conv_dim(8448)·projection_size(16768)까지 통째로 미해결 상수로 남았다 — **규칙은 다 있었는데 입구가 막혀 있던 경우다.** 별칭 두 개로 셋 다 닫혔다. 출처: `modeling_granitemoehybrid.py:513,525,534`.

## 발견 2 — 교정 필요 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.shared_mlp.input_linear` |
| 축 | 공유 MLP gate+up 융합 폭 (3072) |
| 현재 라벨 | `미해결 정수` |
| 판정 | `should_be_renamed` |
| 제안 라벨 | `2*d_shared` |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

`modeling_granitemoehybrid.py:742,744` `self.hidden_size = config.shared_intermediate_size` / `nn.Linear(self.input_size, self.hidden_size * 2)`. 공유 MLP 폭(1536)이 expert 폭(768)과 **다르다** — d_moe 의 shared_* 별칭들은 '값이 이미 d_moe 와 같을 때만 태그한다'는 전제라 쓸 수 없어 `d_shared` 심볼을 새로 두었다.

## 발견 3 — 미확정 (미반영)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.mamba` |
| 축 | n_h_ssm vs d_state 축 (둘 다 128) |
| 현재 라벨 | `d_state / n_h_ssm 혼용` |
| 판정 | `undetermined` |
| 제안 라벨 | — |
| 확신도 | medium |
| 산출물 반영 | 미반영 |

**근거**

`view [.., n_g_ssm, n_h_ssm/n_g_ssm, ?] -> [.., ?, ?]` 의 두 축이 값으로 구별되지 않는다(ssm_state_size == mamba_n_heads == 128). Nemotron-3-Super 와 **같은 막힘**이고, 합쳐진 축이 무엇인지는 reshape 자체가 알지만 그걸 채택하려면 개명을 데이터플로우 끝까지 옮겨야 한다. 값으로 우기지 않고 남긴다.

**근거 소스**: 이 판정은 `develop/sources/modeling_granitemoehybrid.py`, `develop/sources/configuration_granitemoehybrid.py` 를 열어 확인했다. (인용 누락을 자가 점검에서 발견해 보강, 2026-08-12 — 게이트가 이제 `should_be_renamed` 판정에 소스 인용을 요구한다.)

## 발견 4 — 이름 없음이 정답 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `(config)` |
| 축 | logits_scaling |
| 현재 라벨 | `(미등록)` |
| 판정 | `no_name_exists` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

`modeling_granitemoehybrid.py:1277` `logits / self.config.logits_scaling` — 이 필드는 텐서에 곱하거나 나누는 **스칼라 배수**이지 어떤 축의 크기도 아니다. 심볼로 등록하면 값이 우연히 일치하는 축에 이름이 새어 들어간다. 등록하지 않는 것이 정답이다.

## 발견 5 — 이름 없음이 정답 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `(config)` |
| 축 | embedding_multiplier |
| 현재 라벨 | `(미등록)` |
| 판정 | `no_name_exists` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

`modeling_granitemoehybrid.py:1075` `inputs_embeds * self.embedding_multiplier` — 이 필드는 텐서에 곱하거나 나누는 **스칼라 배수**이지 어떤 축의 크기도 아니다. 심볼로 등록하면 값이 우연히 일치하는 축에 이름이 새어 들어간다. 등록하지 않는 것이 정답이다.
