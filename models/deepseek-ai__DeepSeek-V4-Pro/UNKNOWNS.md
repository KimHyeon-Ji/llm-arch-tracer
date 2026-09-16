# 확실하지 않은 것 -- deepseek-ai__DeepSeek-V4-Pro

이 파일은 **이 산출물에서 검증되지 않은 부분 전부**를 모은 것이다. 여기 없는 것은 규칙이 결정했고 게이트가 확인했다.

## 1. 축 이름 판정

축 자리 **660,040개** 중 확정 **494,984개 (75.0%)**, 미확정 **165,056개 (25.0%)**.

| 등급 | 자리 | 무엇을 믿어도 되나 |
|---|---:|---|
| `scope_inferred` | 99,014 | scope 정규식만이 후보를 갈랐다. 근거는 있으나 아무도 검증 안 했다. 구체 크기는 맞다 |
| `heuristic` | 33,866 | **산술로 지어낸 이름.** 값은 맞지만 이름이 틀릴 수 있다 -- 이 목록에서 가장 먼저 봐야 하는 등급이다 |
| `open_tie` | 25,911 | 후보 둘 이상이 같은 값이라 트레이스만으로 못 갈랐다. 구체 크기·FLOPs·바이트는 맞고 **이름만** 미정이다 |
| `unresolved` | 6,265 | 이름 붙일 근거가 없어 정수로 뒀다. 주장을 안 하므로 틀릴 것도 없다 |

### 아직 안 푼 질문

같은 `(등급, 후보, 현재 라벨)` 은 질문 하나다 -- 답 하나가 축 수천 개를 확정시킨다. 답은 `rules/axis_evidence.yaml` 에 인용과 함께 적는다.

질문 합계 **20개**.

**prefill** -- 질문 11개

| 축 수 | 등급 | 후보 | 현재 라벨 |
|---:|---|---|---|
| 41,386 | `scope_inferred` | m_csa \| n_hc | `n_hc` |
| 16,714 | `heuristic` | — | `n_hc` |
| 8,418 | `open_tie` | c_I \| m_hca \| n_h \| w_local | `n_h` |
| 6,944 | `scope_inferred` | d_rope \| n_h_I | `d_rope` |
| 3,090 | `open_tie` | d_rope \| n_h_I | `n_h_I` |
| 2,820 | `scope_inferred` | m_csa \| n_hc | `m_csa` |
| 2,520 | `open_tie` | c_I \| m_hca \| n_h \| w_local | `c_I` |
| 549 | `scope_inferred` | d_g \| k_I | `d_g` |
| 465 | `open_tie` | c_I \| m_hca \| n_h \| w_local | `m_hca` |
| 183 | `heuristic` | — | `2*d_head` |
| 72 | `heuristic` | — | `T` |

**decode** -- 질문 9개

| 축 수 | 등급 | 후보 | 현재 라벨 |
|---:|---|---|---|
| 41,386 | `scope_inferred` | m_csa \| n_hc | `n_hc` |
| 16,714 | `heuristic` | — | `n_hc` |
| 8,418 | `open_tie` | c_I \| m_hca \| n_h \| w_local | `n_h` |
| 5,358 | `scope_inferred` | d_rope \| n_h_I | `d_rope` |
| 2,190 | `open_tie` | d_rope \| n_h_I | `n_h_I` |
| 810 | `open_tie` | c_I \| m_hca \| n_h \| w_local | `c_I` |
| 549 | `scope_inferred` | d_g \| k_I | `d_g` |
| 183 | `heuristic` | — | `2*d_head` |
| 22 | `scope_inferred` | c_I \| m_hca \| n_h \| w_local | `w_local` |

## 2. 자유 평가(③층) 상태

**`STALE`** -- ③ 자유 평가 이후 산출물이 바뀜 (기록 eb6a2cee0c798920 != 현재 6efb27c04e0d3c35, 검토일 2026-09-01)

즉 규칙이 못 잡는 종류의 오류는 **이 판에서 다시 확인되지 않았다.** 규칙 게이트가 통과했다는 것과는 별개의 이야기다.

## 3. 손 안 댄 검토 지적

없다 (기록된 지적 21건은 전부 처리됨).

### 알고 받아들인 한계

**4건.** 고쳐야 할 결함이 아니라 **요약 표의 범위**다 -- 해당 계산은 원시 trace(`full/`)에는 있고 major-op 표에서 빠진다. 표의 행 수로 연산량을 세면 과소평가된다.

| 모듈 | 무엇이 빠졌나 | 근거 |
|---|---|---|
| `model.layers.*.attn_hc / ffn_hc` | mHC Sinkhorn 정규화 | 외부 검토(Codex) 2026-09-16, develop/codex_four_models_review_2026-09-16.md. 표는 softmax → stream collapse 로 보이지만 실제 comb 은 softmax 뒤 epsilon 을 더하고 열 정규화 후 행·열 정규화를 번갈아 반복한다. hc_sinkhorn_iters=20 이면 sum/div 가 39회다. softmax 하나는 이 계산과 같지 |
| `self_attn.compressor / indexer` | 압축 가중합·Indexer head 가중합 | 외부 검토(Codex) 2026-09-16, develop/codex_four_models_review_2026-09-16.md. 압축 softmax 와 norm 사이에 확률×KV 곱과 window 축 sum 이 있다. 그래서 요약의 두 행 사이에서 rank 4 → rank 3 변화가 생략된 reduction 때문에 일어난다. Indexer scorer 도 head 가중치 곱 후 n_h_I 축 sum 이 빠졌 |
| `model.layers.*.mlp.gate` | hash routing vs 동적 routing | 외부 검토(Codex) 2026-09-16, develop/codex_four_models_review_2026-09-16.md. 첫 3층은 frozen tid2eid 테이블 기반 hash routing 이고 이후 층은 score correction bias 를 더한 top-k 다. 양쪽 다 sqrtsoftplus scoring, 선택 score gather, top-k weight 정규화, 2.5 scali |
| `self_attn` | PV 뒤 inverse RoPE | 외부 검토(Codex) 2026-09-16, develop/codex_four_models_review_2026-09-16.md. K=V 라 attention 출력에 inverse RoPE 를 적용하는 단계가 있는데 major-op 표의 PV→o_a 사이에 안 보인다. modeling_deepseek_v4.py:862-868. |

## 4. 의뢰서의 판단 필요 항목

판단 필요 **9건**. 값 충돌이나 관례로 고른 자리라 규칙이 결정하지 못했다 -- 위 1절의 접힌 질문과 같은 종류다. 전문은 `review_request.md` 에 있다.

* 6. 값이 겹쳐 **임의로** 고른 축
* 0. 규칙이 끝내지 못한 축 — **여기부터 답한다**
* A. 붙은 이름 전부 (50종)
* B. 이름 없이 남은 정수 전부 (5쌍)
* C. 모듈이 내는 출력 shape 전부 (107개 모듈 / 1403종)

## 5. 더 이상 안 맞는 소스 확인 기록

**27건.** `rules/label_confirmed.yaml` 이 소스를 보고 "이 이름이 맞다"고 적어 둔 자리인데, 그 앵커가 지금 트레이스에 안 맞는다. 대부분 발행 배치가 B=1 이 아니게 되면서 접힌 배치 축이 생겨(`n_h` -> `B*n_h`) 앵커가 낡은 것이다. **그 축들이 틀렸다는 뜻이 아니라, 지금 판에서 소스로 확인된 상태가 아니라는 뜻이다.**

| 모듈 | 확인한 이름 | 기대값 | 낡은 shape 앵커 |
|---|---|---:|---|
| `self_attn$` | `n_h` | 128 | `['n_h', 'B', 'w_local+T/m_hca']` |
| `self_attn$` | `n_h` | 128 | `['n_h', 'B', 'w_local+T/m_csa']` |
| `self_attn$` | `n_h` | 128 | `['n_h', 'T', 'd_head']` |
| `indexer$` | `d_rope` | 64 | `['B', 'n_h_I', 'T', 'd_rope']` |
| `self_attn$` | `n_h` | 128 | `['B', '1', 'n_h', 'd_head']` |
| `indexer$` | `d_rope` | 64 | `['B', 'n_h_I', '1', 'd_rope']` |
| `self_attn$` | `n_h` | 128 | `['n_h', 'T', 'T+T/m_hca']` |
| `self_attn$` | `n_h` | 128 | `['n_h', 'T', 'T+T/m_csa']` |
| `self_attn\.compressor$` | `2*m_csa` | 8 | `['B', 'd_head', '2*m_csa', 'T/m_csa']` |
| `indexer$` | `c_I` | 128 | `['B', 'd_head', '2*m_csa', 'c_I']` |
| `indexer$` | `c_I` | 128 | `['B', 'd_head', 'c_I']` |
| `scorer$` | `c_I` | 128 | `['B', 'c_I', 'd_head']` |
| `indexer$` | `c_I` | 128 | `['B', 'd_head', 'm_csa', 'c_I']` |
| `indexer$` | `c_I` | 128 | `['B', 'd_head', 'm_csa', 'c_I']` |
| `self_attn$` | `n_h` | 128 | `['B', 'n_h', '1', '1']` |
| `indexer$` | `c_I` | 128 | `['B', '1', 'd_head', 'c_I']` |
| `indexer$` | `n_h_I` | 64 | `['B', 'n_h_I', 'T', 'd_rope']` |
| `self_attn$` | `n_h` | 128 | `['n_h']` |
| `indexer$` | `n_h_I` | 64 | `['B', 'n_h_I', 'T', 'd_rope']` |
| `indexer$` | `d_rope` | 64 | `['B', 'n_h_I', 'T', 'd_rope']` |
| `indexer$` | `n_h_I` | 64 | `['B', 'n_h_I', '1', 'd_rope']` |
| `self_attn$` | `n_h` | 128 | `['n_h']` |
| `indexer$` | `n_h_I` | 64 | `['B', 'n_h_I', '1', 'd_rope']` |
| `indexer$` | `d_rope` | 64 | `['B', 'n_h_I', '1', 'd_rope']` |
| `model$` | `w_local` | 128 | `['B', '1', '1', 'w_local']` |
| `self_attn$` | `n_h` | 128 | `['n_h', 'B', 'd_head']` |
| `scorer$` | `c_I` | 128 | `['B', 'd_head', 'c_I']` |

## 6. 표의 숫자가 관측값이 아닌 자리

없다 -- 모델 동작을 대체한 remedy 가 없다.
