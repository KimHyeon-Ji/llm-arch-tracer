# 03. 결과를 어떻게 남기는가

두 가지를 한다. **JSON 하나를 쓰고, 원장에 한 줄 기록한다.** 읽히는 문서는 전부 거기서 생성된다.

## 1. `models/<모델>/review_findings.json`

이것이 유일한 원본이다. 여기서 두 가지가 자동으로 나온다:

- `models/<모델>/review_findings.md` — 사람이 읽는 기록
- `models/<모델>/model_summary.md` 의 **"③ 라벨 검토"** 절 — 표를 읽는 사람이 바로 만나는 주의사항

**규칙으로 못 박는 것도 여기 남기면 산출물에 반영된다.** 두 config 값이 겹쳐 구별이 불가능하거나,
융합 파라미터의 축 순서가 트레이스에 안 보이거나, 근거가 논문에만 있는 경우 —
`status: "open"` 으로 두면 요약 카드에 "지금 렌더 / 소스가 말하는 것 / 근거"가 나란히 실린다.

```json
{
  "model_id": "allenai/OLMoE-1B-7B-0924",
  "reviewed_on": "2026-08-09",
  "reviewer": "<누가 / 어떤 모델>",
  "angle": "<어떤 각도로 봤는지>",
  "summary": "<한 줄 요약>",
  "findings": [
    {
      "module": "model.layers.*.mlp.experts",
      "axis": "[E, d_model, d_model] 의 가운데 축 (2048)",
      "current_label": "d_model",
      "verdict": "should_be_renamed",
      "proposed_label": "2*d_moe",
      "confidence": "high",
      "evidence": "modeling_olmoe.py:297 `gate_up_proj = nn.Parameter(torch.empty(num_experts, 2 * intermediate_dim, hidden_dim))` — 2·1024=2048 이 hidden_size 와 겹쳐 d_model 로 붙었다.",
      "status": "open"
    }
  ]
}
```

| 필드 | 뜻 |
|---|---|
| `verdict` | `01-procedure.md` 의 판정 4종 중 하나 |
| `confidence` | `high` / `medium` / `low` |
| `evidence` | **클래스·메서드 이름과 인용한 코드 줄.** 근거 없는 판정은 판정이 아니다 |
| `status` | `fixed` = 산출물이 이미 맞게 렌더한다 / `open` = 아직 아니다 → 요약 카드에 실린다 |

발견이 없으면 `findings: []` 로 두고 `summary` 에 **"의뢰서가 비어 있었다"** 또는
**"의뢰서 N건 전부 오탐이었다"** 를 적는다. 빈 결과도 결과다.

md 를 직접 고치지 않는다 — 다음 재생성 때 JSON 에서 다시 만들어져 덮어쓴다.

```bash
python develop/regen_summaries.py     # JSON -> review_findings.md + 요약 카드 절
```

## 2. 원장 기록

```bash
python src/review_ledger.py --record <모델 폴더 이름> \
    --findings <건수> --notes "<어떤 각도로 봤는지>" --reviewer "<누가>"
```

이걸 해야 게이트의 `③ 자유 평가 수행 기록` 이 **미수행 → 최신**으로 바뀐다.
산출물이 나중에 바뀌면 자동으로 **만료**가 되어 다시 올라온다.

확인:

```bash
python src/review_ledger.py
```

## 규칙을 고쳤다면

`rules/` 나 `src/` 를 손댔으면 기록만으로 끝나지 않는다:

```bash
python develop/regen_summaries.py
python develop/verify_all.py       # EXIT 0
python develop/verify_selftest.py
```

**EXIT 0 전에는 고쳤다고 하지 않는다.** 고쳐서 산출물이 맞게 나오면 그 발견의 `status` 를
`fixed` 로 바꾼다 — 그래야 요약 카드의 주의사항이 실제로 남은 것만 가리킨다.
