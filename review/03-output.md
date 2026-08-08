# 03. 결과를 어떻게 남기는가

두 가지를 한다. **문서 하나를 쓰고, 원장에 한 줄 기록한다.**

## 1. `models/<모델>/review_findings.md`

발견마다 아래 블록 하나. 형식이 고정인 이유는 사람이 훑을 때와 다음 검토가 이전 판정을
다시 읽을 때가 같아야 하기 때문이다.

```markdown
# 라벨 검토 결과 — <모델 id>

- 검토일: 2026-08-08
- 검토자: <누가 / 어떤 모델>
- 본 것: <의뢰서 몇 절을, 어떤 소스로>
- 요약: <한 줄>

## 발견 1 — <한 줄 제목>

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.attn_hc` |
| 축 | `[B, T, n_hc, n_hc]` 의 뒤 두 축 |
| 현재 라벨 | `n_hc` (두 번) |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |

**근거**

`DeepseekV4HyperConnection.forward` (`develop/sources/modeling_deepseek_v4.py`):

​```python
comb_logits = comb_w.view(*comb_w.shape[:-1], hc, hc) * comb_scale + comb_b.view(hc, hc)
​```

`hc = self.hc_mult` 이고 `self.hc_mult = config.hc_mult` 이므로 두 축은 같은 값이며,
`hc_mult` 개 병렬 잔차 스트림 사이의 혼합 행렬이다. 정사각이 정상이다.

**승격 제안**: 없음 / `rules/symbols.yaml` 에 `X` 별칭 추가 / 탐지기 수정 (…)
```

발견이 없으면 발견 절 없이 위쪽 머리말만 쓰고 요약에 **"의뢰서가 비어 있었다"** 또는
**"의뢰서 N건 전부 오탐이었다"** 를 적는다. 빈 결과도 결과다.

## 2. 원장 기록

```bash
.venv/Scripts/python.exe src/review_ledger.py --record <모델 폴더 이름> \
    --findings <건수> --notes "<어떤 각도로 봤는지>" --reviewer "<누가>"
```

이걸 해야 게이트의 `③ 자유 평가 수행 기록` 이 **미수행 → 최신**으로 바뀐다.
산출물이 나중에 바뀌면 자동으로 **만료**가 되어 다시 올라온다.

확인:

```bash
.venv/Scripts/python.exe src/review_ledger.py
```

## 규칙을 고쳤다면

`rules/` 나 `src/` 를 손댔으면 기록만으로 끝나지 않는다:

```bash
.venv/Scripts/python.exe develop/regen_summaries.py
.venv/Scripts/python.exe develop/verify_all.py       # EXIT 0
.venv/Scripts/python.exe develop/verify_selftest.py
```

**EXIT 0 전에는 고쳤다고 하지 않는다.**
