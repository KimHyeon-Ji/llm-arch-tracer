# 라벨 검토 결과 — nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16

- 검토일: 2026-08-11
- 검토자: llm(claude, 전수 점검 2회차 — 모듈-필드 소속)
- 본 것: 전수 점검 2회차 — 1층(모듈-필드 소속: 가중치 축의 이름이 그 모듈/부모가 실제로 읽는 config 필드에서 나왔는가)을 함대 전건 수행. 값을 보지 않는 검사라 값 충돌이 숨길 수 없다. 1회차의 A절·B절 판정은 유지. C절(모듈별 출력 shape)은 여전히 미수행.
- 요약: 의뢰서 2건 → 1건. L=108, d=8192 의 최상위 모델이 **새 규칙 0개**로 들어왔다 — '규칙은 모델마다 늘지 않는다'가 대규모에서도 성립함을 보여준다.

> 이 파일은 `review_findings.json` 에서 생성된다 — 고칠 때는 JSON 을 고친다.

## 발견 1 — 이름 없음이 정답 (미반영)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.mixer` |
| 축 | [..., 2, 2] 의 뒤 두 축 |
| 현재 라벨 | `n_kv` |
| 판정 | `no_name_exists` |
| 제안 라벨 | `정수 2` |
| 확신도 | high |
| 산출물 반영 | 미반영 |

**근거**

n_kv=2 인데 `[..., 2, 2]` 로 렌더된다. 이 축은 Mamba2 청크 간 재귀의 청크 경계 상태 수다 — `modeling_nemotron_h.py:320` `decay_chunk = torch.exp(segment_sum(F.pad(A_cumsum[:, :, :, -1], (1, 0)))).transpose(1, 3)` → `[B, n_h_ssm, n_chunks+1, n_chunks+1]`. 우리가 트레이스하는 seq_len 은 항상 d_chunk(256)보다 작아 n_chunks=1 이므로 축은 말 그대로 2 이고, n_kv 와 우연히 같다. Nemotron-3-Nano 에서 같은 자리가 `k`(=2)로 붙었던 것과 동일한 건이다. **일반형(ceil(T/d_chunk)+1)으로 등록하지 않는다**: 관측한 적 없는 regime 을 주장하게 되고, 게이트의 라벨 검사가 `/` 를 floor division 으로 다뤄 그 식이 성립하지도 않는다. n_h/n_kv 에 `group: attn` 을 달아 계열 밖 폴백은 막았지만, 이 자리는 스코프가 아니라 값 충돌이라 남는다 — 정수로 두는 것이 정답이다.

## 발견 2 — 맞음 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.mixer` |
| 축 | decode KV concat |
| 현재 라벨 | `T+1` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

Super-120B 와 동일 — `T+1` 규칙 스코프에 `mixer` 를 추가해 해소했다.

## 발견 3 — 맞음 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.mixer.shared_experts.{up,down}_proj` |
| 축 | 공유 전문가 FFN 폭 |
| 현재 라벨 | `d_moe (필드 미접지)` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

`modeling_nemotron_h.py:689` `self.shared_experts = NemotronHMLP(config=config, intermediate_size=config.moe_shared_expert_intermediate_size)`. config 는 `moe_intermediate_size` 와 `moe_shared_expert_intermediate_size` 를 따로 두고 이 체크포인트에서 둘의 값이 같다. 이름 `d_moe`(전문가 FFN 중간 폭)의 뜻은 맞지만 **별칭 표에 그 필드가 없어** 소속 검사가 접지 실패로 잡아냈다 — 값으로는 영원히 안 보였을 자리다. `rules/symbols.yaml` 의 d_moe 별칭에 추가했다.
