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

**`STALE`** -- ③ 자유 평가 이후 산출물이 바뀜 (기록 eb6a2cee0c798920 != 현재 02d96a095a360e7b, 검토일 2026-09-01)

즉 규칙이 못 잡는 종류의 오류는 **이 판에서 다시 확인되지 않았다.** 규칙 게이트가 통과했다는 것과는 별개의 이야기다.

## 3. 손 안 댄 검토 지적

없다 (기록된 지적 16건은 전부 처리됨).

## 4. 더 이상 안 맞는 소스 확인 기록

**19건.** `rules/label_confirmed.yaml` 이 소스를 보고 "이 이름이 맞다"고 적어 둔 자리인데, 그 앵커가 지금 트레이스에 안 맞는다. 대부분 발행 배치가 B=1 이 아니게 되면서 접힌 배치 축이 생겨(`n_h` -> `B*n_h`) 앵커가 낡은 것이다. **그 축들이 틀렸다는 뜻이 아니라, 지금 판에서 소스로 확인된 상태가 아니라는 뜻이다.**

| 모듈 | 확인한 이름 | 기대값 | 낡은 shape 앵커 |
|---|---|---:|---|
| `self_attn$` | `n_h` | 128 | `['n_h', 'B', 'w_local+T/m_hca']` |
| `self_attn$` | `n_h` | 128 | `['n_h', 'B', 'w_local+T/m_csa']` |
| `self_attn$` | `n_h` | 128 | `['n_h', 'T', 'd_head']` |
| `self_attn$` | `n_h` | 128 | `['B', '1', 'n_h', 'd_head']` |
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
| `self_attn$` | `n_h` | 128 | `['n_h']` |
| `self_attn$` | `n_h` | 128 | `['n_h']` |
| `model$` | `w_local` | 128 | `['B', '1', '1', 'w_local']` |
| `self_attn$` | `n_h` | 128 | `['n_h', 'B', 'd_head']` |
| `scorer$` | `c_I` | 128 | `['B', 'd_head', 'c_I']` |

## 5. 표의 숫자가 관측값이 아닌 자리

없다 -- 모델 동작을 대체한 remedy 가 없다.
