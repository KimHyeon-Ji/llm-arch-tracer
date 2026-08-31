# llm-arch-tracer 결과물 전용 브랜치

`main` 브랜치의 각 모델 폴더에서 `full/`(원본 트레이스, 용량 대부분 차지)을 뺀
결과물 파일만 담은 스냅샷입니다. 받는 쪽은 이 브랜치만 얕게 클론하면 됩니다:

```
git clone --branch results --single-branch --depth 1 <repo-url>
```

## 지금 포함된 모델 (8개)

Qwen2.5-0.5B, Qwen3-30B-A3B, gemma-2-2b, gemma-3-270m, Llama-3.1-8B, Llama-3.1-70B,
Phi-4, falcon-7b.

기준: (1) 판단 필요 0건 + 게이트 PASS + 내부 외부 검토(③ 자유 평가) 완료, **그리고**
(2) 이 저장소의 내부 코드를 전혀 안 보고 공식 논문/HF config/공식 GitHub 구현체만으로
독립 재현한 외부 검증(Codex, 2026-08-31)에서도 일치 판정.

원래 14개 중 6개(SmolLM3-3B-Base, LFM2-8B-A1B, ERNIE-4.5-21B-A3B-PT,
Llama-4-Maverick-17B-128E, Mistral-Small-3.2-24B-Instruct-2506, GLM-4.5-Air)는 이
외부 검증에서 실제 문제(레이어 스케줄 오기재, 심볼 값 누락 등)가 발견돼 빠졌습니다 —
수정 검증 끝나는 대로 다시 추가됩니다.

각 모델 폴더의 파일:
- `structure.yaml` — 심볼(축 이름) 표
- `model_summary.md` — 사람이 읽는 요약
- `prefill.csv` / `decode.csv`, `.jsonl` — 연산 표 (레이어 패턴으로 축약된 버전;
  레이어별 원본 전개는 `main` 브랜치의 `full/` 안에 있습니다)
- `review_findings.md` / `.json`, `review_request.md` — 검토 기록

## 갱신

`main`에서 결과물이 바뀔 때마다 이 브랜치에 새 스냅샷 커밋 하나가 추가됩니다
(히스토리 추적용이 아니라 매번 "지금 상태"만 필요하면 `--depth 1`로 받으세요).
