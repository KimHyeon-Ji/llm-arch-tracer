# llm-arch-tracer 결과물 전용 브랜치

`main` 브랜치의 각 모델 폴더에서 `full/`(원본 트레이스, 용량 대부분 차지)을 뺀
결과물 파일만 담은 스냅샷입니다. 받는 쪽은 이 브랜치만 얕게 클론하면 됩니다:

```
git clone --branch results --single-branch --depth 1 <repo-url>
```

## 지금 포함된 모델 (14개)

판단 필요 0건 + 게이트 PASS + 외부 검토(③ 자유 평가) 완료 — "100% 확신 가능" 기준을
통과한 모델만 우선 올렸습니다. 나머지 모델은 검토가 끝나는 대로 추가됩니다.

각 모델 폴더의 파일:
- `structure.yaml` — 심볼(축 이름) 표
- `model_summary.md` — 사람이 읽는 요약
- `prefill.csv` / `decode.csv`, `.jsonl` — 연산 표 (레이어 패턴으로 축약된 버전;
  레이어별 원본 전개는 `main` 브랜치의 `full/` 안에 있습니다)
- `review_findings.md` / `.json`, `review_request.md` — 검토 기록

## 갱신

`main`에서 결과물이 바뀔 때마다 이 브랜치에 새 스냅샷 커밋 하나가 추가됩니다
(히스토리 추적용이 아니라 매번 "지금 상태"만 필요하면 `--depth 1`로 받으세요).
