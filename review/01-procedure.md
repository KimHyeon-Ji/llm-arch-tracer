# 01. 검토 절차

## 1. 의뢰서를 읽는다

`models/<모델>/review_request.md` 가 시작점이다. 파이프라인이 규칙으로 결정 가능한 것을 전부
결정하고 **남은 것만** 거기 적어뒀다. 네 종류로 나뉜다.

| 절 | 무엇이 미결인가 | 어떻게 답이 나오는가 |
|---|---|---|
| 1. 별칭 접지 | 심볼이 값을 읽은 config 필드가 이 모델의 `configuration_*.py` 에 없다 | config 클래스를 열어 그 필드가 무엇인지, 우연히 맞은 건 아닌지 |
| 2. 정사각 축 | `[..., X, X]` 인데 그 이름의 config 필드에서 나온 정사각 reshape을 소스에서 못 찾았다 | modeling 소스에서 그 텐서를 만드는 줄을 찾아 두 축이 정말 같은 것인지 |
| 3. 미등록 필드 | 모듈 폭으로 쓰이는데 심볼 표에 없다 | 소스에서 무엇인지 확인 → `rules/symbols.yaml` 등록 |
| 4. 산술로 지은 이름 | 값이 맞아떨어져 붙인 이름 (예: RoPE 절반 차원) | 산술적 우연인지 실제 의미인지 소스에서 |

의뢰서가 비어 있으면 그렇게 기록한다. **없는 발견을 만들지 않는다.**

## 2. 소스를 연다 — 건너뛰면 이 절차의 의미가 없다

`develop/sources/modeling_<model_type>.py` 에 실제 코드가 이미 받아져 있다. 문제의 shape을
만드는 **코드 줄**을 찾아라.

> 예: `[B, T, n_hc, n_hc]` 의 뒤 두 축이 뭔가?
> → `DeepseekV4HyperConnection.forward` 의
> `comb_logits = comb_w.view(*comb_w.shape[:-1], hc, hc) * comb_scale + comb_b.view(hc, hc)`
> → `hc_mult` 개 병렬 잔차 스트림 사이의 혼합 행렬. 정사각이므로 같은 이름 두 번이 정상.

소스 우선순위는 `02-new-module-handling.md` 의 Tier 2 사다리를 따른다:
실행 중인 modeling 소스 → config 클래스 docstring → vLLM/SGLang 독립 구현 → model card → 논문.
**위에서 답이 나오면 아래는 생략한다.** 모델별 URL은 `models/<모델>/research_agenda.md` 하단에
이미 채워져 있다.

## 3. 판정한다 — 네 가지뿐이다

| 판정 | 뜻 |
|---|---|
| `current_label_correct` | 안건이 오탐이다. **왜 오탐인지**가 5절의 재료다 |
| `should_be_renamed` | 교정 대상. 제안 이름과 근거 줄을 남긴다 |
| `no_name_exists` | 루프 인덱스·피연산자 개수. **정수로 두는 것이 정답** |
| `undetermined` | 소스에 근거가 없다. 무엇을 봤고 뭐가 없었는지 적는다 |

## 4. 기록한다

`03-output.md` 형식으로 `models/<모델>/review_findings.md` 에 쓰고 원장에 남긴다.

## 5. 승격한다 — 여기가 진짜 성과다

문서에만 남기면 다음 모델에서 똑같이 반복된다.

| 알아낸 것 | 옮길 곳 |
|---|---|
| 이 값은 이 config 필드다 | `rules/symbols.yaml` 의 `aliases` |
| 이 값은 이렇게 계산된다 | `rules/derived_dims.yaml` (`expr` + `sym` + **출처 주석**) |
| 이 모듈은 이런 구조다 | `rules/structures/<범주>/<이름>.md` (C17이 등재를 확인) |
| **안건이 오탐이었다** | **탐지기를 고친다** (`src/research.py` 또는 `src/source_check.py`) |
| 새 불변식을 찾았다 | `develop/verify_all.py` 검사 + `develop/verify_selftest.py` 폴트 인젝션 |

> **판정이 "맞다"로 나오는 것은 성과가 없는 게 아니다.** 오탐을 남겨두면 판정 표가 신호를
> 잃는다. 실제로 `[B,T,n_hc,n_hc]` 건은 라벨이 옳았고, 진짜 발견은 **탐지기가 이미 해결된
> 질문을 다시 묻고 있다**는 것이었다. 정사각 행렬 예외를 넣자 함대의 코드-조사 대상이
> 52,687축 → 1,752축으로 줄었다.

## 6. 반드시 확인한다

`rules/` 나 `src/` 를 고쳤으면:

```bash
.venv/Scripts/python.exe develop/regen_summaries.py    # 재추적 없이 산출물 갱신
.venv/Scripts/python.exe develop/verify_all.py         # EXIT 0 이어야 한다
.venv/Scripts/python.exe develop/verify_selftest.py    # 검사가 살아있는지
```

**EXIT 0 전에는 "고쳤다"고 말하지 않는다.** 베이스라인이 개선만 있으면 검토 후
`--update-baseline`. 퇴행이 있으면 방향을 확인하고, 정당하면 근거를 코드 주석에 남긴 뒤 수용한다.

## 하지 말 것

- **재현 없이 반영하지 않는다.** LLM의 지적은 그 자체로 근거가 아니다
- **정수를 억지로 이름 붙이지 않는다.** 루프 인덱스에 이름을 붙이는 건 지어내는 것이다
- **한 번에 한 모델.** 묶으면 어떤 변경이 무엇을 고쳤는지 귀속되지 않는다
