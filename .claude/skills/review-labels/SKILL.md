---
name: review-labels
description: Run the layer-3 free-form review — read a model's research agenda, open the real modeling source with WebFetch, settle the unresolved axes with quoted evidence, and record the result. Use when the gate reports the review is stale or missing, when asked to verify labels against sources, or when working through models/*/research_agenda.md.
---

# ③ 자유 평가 — 소스를 열어 축 이름을 확정한다

규칙 게이트는 산출물이 자기 규칙과 일관되는지만 본다. 규칙 자체가 틀렸거나 값이 겹쳐
못 가리는 축은 잡지 못한다. 이 절차가 그 자리를 메운다: **모델의 실제 modeling 소스를 열어**
확정하고, 근거와 함께 기록하고, 일반화되는 것은 규칙으로 승격한다.

API 호출이 아니라 **이 세션의 WebFetch로 직접 읽는다.** 인증도 SDK도 필요 없다.

## 1. 무엇을 검토할지 고른다

```bash
.venv/Scripts/python.exe src/review_ledger.py
```

`만료`(산출물이 검토 이후 바뀜) 또는 `미수행` 모델이 대상이다. 그중에서 고를 때는
`models/<모델>/research_agenda.md` 맨 위의 **판정** 표를 본다 — `modeling 소스를 읽어야 함`
축이 가장 많은 모델이 효율이 가장 좋다. 사용자가 모델을 지정했으면 그것을 쓴다.

## 2. 안건을 읽는다

`models/<모델>/research_agenda.md`:

- **1절** 한 shape에 같은 이름 두 번 — 두 축의 크기가 우연히 같아 값으로 못 가리는 자리
- **2절** reshape 자기 유도와 라벨 불일치 — 같은 텐서에 설명이 둘
- **3절** 규칙 없이 산술로 지은 이름 — 등록만 하면 되는 것
- **5절** 이름 없는 정수 — **전부 이름 붙일 대상이 아니다** (루프 인덱스·피연산자 개수)

안건 하단의 **확인할 소스**에 이 모델의 실제 URL이 이미 채워져 있다(HF modeling/config 파일).

## 3. 소스를 연다 — 이 단계를 건너뛰면 이 절차의 의미가 없다

WebFetch로 modeling 파일을 읽고, **문제의 shape를 만드는 코드 줄을 찾는다.**
구조를 묻지 말고 그 줄을 찾아라. 예:

> `[B, T, n_hc, n_hc]`의 뒤 두 축이 뭔지 → `comb_w.view(*comb_w.shape[:-1], hc, hc)`
> → hc_mult개 병렬 스트림 사이의 혼합 행렬. 정사각이므로 같은 이름 두 번이 정상.

소스 순서는 `02-new-module-handling.md`의 Tier 2 사다리를 따른다(실행 중인 modeling 소스 →
config docstring → vLLM/SGLang 독립 구현 → model card → 논문). 위에서 답이 나오면 아래는 생략.

## 4. 판정한다 — 네 가지뿐이다

| 판정 | 뜻 |
|---|---|
| 현재 라벨이 맞다 | 안건이 오탐. **왜 오탐인지**가 다음 단계의 재료다 |
| 다른 이름이어야 한다 | 교정 대상. 제안 이름과 근거 줄을 남긴다 |
| 이름이 존재하지 않는다 | 루프 인덱스·피연산자 개수. **정수로 두는 것이 정답** |
| 판정 불가 | 소스에 근거가 없다. 무엇을 봤고 뭐가 없었는지 적는다 |

**근거 없는 판정은 판정이 아니다.** 클래스·메서드 이름과 코드 줄을 인용한다.

## 5. 기록한다

`models/<모델>/review_findings.md`에 쓰고 원장에 남긴다:

```python
import sys, datetime; sys.path.insert(0, "src")
import review_agent, review_ledger
res = {"status": "ok", "reason": "", "summary": "<한 줄 요약>", "findings": [{
    "module": "...", "axis": "...", "current_label": "...",
    "verdict": "current_label_correct|should_be_renamed|no_name_exists|undetermined",
    "proposed_label": "", "evidence": "<클래스.메서드 + 인용한 코드 줄>",
    "source_url": "...", "confidence": "high|medium|low"}]}
review_agent.write_report(model_dir, res)
review_ledger.record(model_name, model_dir, datetime.date.today().isoformat(),
                     len(res["findings"]), "수동 수행 (WebFetch, modeling 소스 직접 대조)",
                     reviewer="llm(claude-code)")
```

## 6. 승격한다 — 여기가 진짜 성과다

발견을 문서에만 남기면 다음 모델에서 똑같이 반복된다. 성격에 따라 옮긴다:

| 알아낸 것 | 옮길 곳 |
|---|---|
| 이 값은 이 config 필드다 | `rules/symbols.yaml` 의 `aliases` |
| 이 값은 이렇게 계산된다 | `rules/derived_dims.yaml` (`expr` + `sym` + **출처 주석**) |
| 이 모듈은 이런 구조다 | `rules/structures/<범주>/<이름>.md` (C17이 등재를 확인한다) |
| 안건이 오탐이었다 | **`src/research.py` 의 탐지기를 고친다** — 오탐을 남겨두면 판정 표가 신호를 잃는다 |
| 새로운 불변식을 찾았다 | `develop/verify_all.py` 에 검사 추가 + `develop/verify_selftest.py` 에 폴트 인젝션 |

## 7. 반드시 확인한다

`rules/` 나 `src/` 를 고쳤으면:

```bash
.venv/Scripts/python.exe develop/regen_summaries.py    # 재추적 없이 산출물 갱신
.venv/Scripts/python.exe develop/verify_all.py         # EXIT 0 이어야 한다
.venv/Scripts/python.exe develop/verify_selftest.py    # 검사가 살아있는지
```

**EXIT 0 전에는 "고쳤다"고 말하지 않는다.** 베이스라인이 개선만 있다고 하면 검토 후
`--update-baseline`. 퇴행이 있으면 방향을 확인하고, 정당하면 근거를 코드 주석에 남긴 뒤 수용한다.

## 하지 말 것

- **재현 없이 반영하지 않는다.** LLM 지적은 그 자체로 근거가 아니다 — 트레이스나 소스로 확인한다
- **정수를 억지로 이름 붙이지 않는다.** 루프 인덱스에 이름을 붙이는 건 지어내는 것이다
- **한 번에 한 모델.** 여러 모델을 묶으면 어떤 변경이 무엇을 고쳤는지 귀속되지 않는다
- 안건이 비어 있으면 그렇게 기록한다. 없는 발견을 만들지 않는다
