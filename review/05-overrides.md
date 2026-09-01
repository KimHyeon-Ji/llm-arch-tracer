# 05. 판정을 산출물에 반영하는 법 — 라벨 교정

`03-output.md` 는 판정을 **기록**하는 형식이다. 이 문서는 그 판정이 실제 표에 들어가는 경로다.

## 왜 별도 경로가 필요한가

④층(LLM이 소스와 대조)이 내는 판정 중 상당수는 **규칙으로는 도달할 수 없다.** 규칙은 축 하나를
값으로 결정하는데, 두 config 필드가 같은 숫자를 갖는 순간 값으로 결정할 게 없어진다.

소스를 읽으면 답은 나온다. 문제는 그 답을 어떻게 표에 넣느냐였다.

**규칙을 고쳐 다시 추론하게 만드는 방법은 두 번 시도해서 두 번 되돌렸다.** 매번 같은 이유로
실패한다 — 값 충돌은 국소적이지 않다. 한 op 의 이름을 바꾸면 이웃이 옛 이름을 유지하고,
데이터플로우 검사가 그 자리를 정확히 짚는다:

| 시도 | 결과 |
|---|---|
| `_carry_reshape_labels` (2026-08-05) | flow_ambig 2배 (405B 504 → 882) — 비활성으로 남김 |
| `_split_from_registered_sum` (2026-08-12) | DeepSeek MLA reshape 61 → 122, flow_ambig 0 → 122 — 되돌림 |

그래서 **추론을 다시 돌리지 않는다.** 렌더가 전부 끝난 뒤, 선언된 모듈 아래의 그 이름을 전부
바꾼다. 추론이 아니기 때문에 사슬이 어긋나지 않는다.

## 쓰는 법

`rules/label_overrides.yaml` 에 한 항목을 추가한다:

```yaml
- model: LiquidAI__LFM2-8B-A1B
  module: '^model\.pos_emb$'    # 모듈 경로 정규식 (레이어 인덱스는 `*` 로 접힌 형태)
  from: E
  to: d_head/2
  expect: 32                    # 그 축의 실제 크기
  layer_types: [linear_attention]   # (선택) 하이브리드 스택에서 블록 종류 한정
  source: >
    실측 `[1, 16, 32]` 이고 바로 옆 concat 이 `[1, 16, 64]`(=d_head) 다 — rotary 의
    inv_freq 절반 축이다. 전문가 수 E(=32)가 값이 같아 그 이름이 붙었다.
```

`develop/regen_summaries.py` 를 돌리면 반영된다. 재추적은 필요 없다.

## 무엇이 이걸 정직하게 유지하는가

교정은 **주장**이고, 여기서는 모든 주장이 값을 치러야 한다.

| 장치 | 무엇을 막는가 |
|---|---|
| `source` 필수 — 파일:줄 인용 | 근거 없는 교정. 인용이 없으면 교정이 아니다 |
| `expect` 필수 — 축의 실제 크기 | 엉뚱한 축에 이름을 붙이는 것. 크기가 다르면 **발화하지 않는다** |
| **게이트가 발화 0건을 FAIL 로 잡는다** | 낡은 주장. 재추적·규칙 개선으로 이미 맞게 나오는데 항목이 남아 있으면, 그때부터 표가 "교정됨"이라고 거짓말한다 |
| `layer_types` | 같은 모듈 이름이 레이어마다 다른 블록인 경우. Nemotron 의 `mixer` 는 Mamba 이기도 하고 attention 이기도 하다 |
| `full/label_overrides.json` | 모델마다 무엇이 몇 축에 적용됐는지 산출물에 남는다 |

## 값이 겹치면 — 등록 전에 외부 검토를 거친다 (자동 게이트, 2026-09-01)

두 심볼이 값으로 안 갈리는 건 흔하다(`d_head` vs `n_h` 등). 문제는 **op 모양이 값 분할을
여러 가지로 설명할 수 있는 자리**(`slice`/`split`/`narrow`/`chunk`/`concat`/`view`/`reshape`
/`transpose`/`permute`)에서, 서로 무관한 이름 2개 이상이 같은 값으로 겹칠 때다 — 사람도(그리고
LLM도) op 모양만 보고 그럴듯한 설명을 지어내기 쉽고, 실제로 이 저장소에서 그렇게 틀렸다
(DeepSeek-V4-Pro, 2026-09-01: 트레이스의 `slice`(op 1874)/`concat`(op 1887)이 정확히 값
64에서 `n_h_I`/`d_rope`/`c_I-d_rope` 세 이름과 겹치는 자리인데 `c_I/2`라는 네 번째 설명을
지어 등록했다가 외부 검토로 정정).

`develop/check_value_collisions.py`가 이런 자리를 **실제 트레이스에서** 자동으로 찾는다 —
`structure.yaml`의 plain 심볼뿐 아니라 `rules/derived_dims.yaml`의 유도식도 후보에 넣고
(`c_I-d_rope`처럼 plain 심볼이 아닌 이름도 잡는다), 그 값이 실제로 위험 op의 축에 나타날
때만 신호로 센다(그냥 "모델 어딘가에 값이 같은 심볼이 있다"는 흔해서 신호가 안 된다 —
2026-09-01 실측, 위험 op로 좁히기 전엔 45개 모델 전부가 걸렸다).

**`develop/verify_all.py`가 이제 이걸 자동으로 강제한다** — 프로즈를 읽고 기억해서 돌리는
게 아니라, rules/를 고친 사람이 누구든 게이트가 대신 확인한다:

- `develop/verify/collision_baseline.json` — 게이트 도입 시점(2026-09-01)의 전체 스냅샷.
  여기 있는 자리는 **아직 검토 전이라도 WARN**(막지 않음, 급한 순서는 아님)만 뜬다.
- `develop/verify/references.yaml`의 `value_collisions_reviewed:` — 실제로 외부 검토를
  거쳐 해소된 자리. 등재되면 WARN도 안 뜬다. `source`는 파일:줄 또는 URL 형식이 없으면
  게이트가 FAIL한다(근거 없는 "검토함"은 검토가 아니다).
- **베이스라인에도 검토 원장에도 없는, 이 시점 이후 새로 생긴 자리는 FAIL한다.** 새 유도식
  등록이나 `spread: class` 배치가 새 값 충돌을 만들면 바로 여기 걸린다.

```
.venv\Scripts\python.exe develop\check_value_collisions.py            # 함대 전체 (읽기 전용)
.venv\Scripts\python.exe develop\check_value_collisions.py --model X  # 모델 하나
.venv\Scripts\python.exe develop\check_value_collisions.py --dump-baseline develop\verify\collision_baseline.json
    # 검토를 마치고 베이스라인을 갱신할 때만 쓸 것 -- 검토 없이 그냥 다시 찍으면 게이트를
    # 무력화하는 것과 같다.
```

## 대량 등록 전 표본 재검증 — `rule_coverage.py --emit --verified`

"규칙이 안전장치(agree/differ/clash 0)를 통과했다" 는 것과 "그 이름이 실제로 맞다"는 다른
주장이다. `rule_coverage.py --emit`은 이제 `--verified "<표본 재검증 근거, 파일:줄 포함>"`을
**필수**로 요구한다(배치 크기와 무관 — 1건이라도 근거 없이 대량 전파하지 않는다). 이
텍스트는 낸 YAML의 헤더 주석과 `develop/verify/batch_verification_log.yaml`(append-only
감사 로그)에 남는다.

**이게 강제하는 건 "재검증했다는 주장이 존재하는가"뿐이다.** "정말 맞게 재검증했는가"는
여전히 사람 몫이고, 이건 정직한 한계다. 그래도 근거 없는 빈말("확인함")은 인용 형식 검사로
걸러진다.

## 여기서 표현할 수 없는 것

**텐서가 어디서 왔는지로만 구별되는 충돌.** MLA 의 `d_nope` 와 `d_v` 는 둘 다 128 이고, 둘 다
`self_attn` 안이고, 둘 다 `[B, n_h, T, ·]` 다. 모듈로도 값으로도 shape 으로도 안 갈린다 —
갈리는 건 `torch.split` 의 첫째 조각이냐 둘째 조각이냐뿐이다.

그런 건 **고치지 않고 `open` 으로 남긴다.** 소스 줄과 함께 요약 카드에 실려서 읽는 사람에게는
전달되고, 표에는 반영되지 않는다는 사실도 함께 실린다. 그걸 반영하려면 *권위 있는 개명을
데이터플로우 끝까지 옮기는* 기계장치가 필요하고, 위 표의 두 시도가 그게 아직 안전하지 않다는
증거다.

## 1회차 적용 (2026-08-12)

| 모델 | 교정 | 축 | 근거 |
|---|---|---|---|
| LFM2-8B-A1B | `E` → `d_head/2` | 3 | rotary inv_freq 절반 축, E(=32)와 값 충돌 |
| DeepSeek-V2-Lite | `E_shared` → `2` | 324 | `view_as_real` 의 실수부·허수부 쌍 |
| Nemotron-3-Nano | `k` → `2` | 882 | Mamba2 chunk-scan `n_chunks+1` |
| Nemotron-3-Super | `n_kv` → `2` | 1,800 | 〃 (attention 레이어는 `layer_types` 로 제외) |
| Nemotron-3-Ultra | `n_kv` → `2` | 2,160 | 〃 |

합계 **5,169축**. 전부 지어낸 이름이 정직한 정수로 바뀐 것이라 `bare` 가 올라간다 — 나빠진 게
아니라 정직해진 것이고, 그 정수들의 사유는 `develop/verify/references.yaml` 에 등재돼 있다.
