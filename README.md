# llm-arch-tracer 결과물 전용 브랜치

`main` 브랜치의 각 모델 폴더에서 `full/`(원본 트레이스, 용량 대부분 차지)을 뺀
결과물 파일만 담은 스냅샷입니다. 받는 쪽은 이 브랜치만 얕게 클론하면 됩니다:

```
git clone --branch results --single-branch --depth 1 <repo-url>
```

## 지금 포함된 모델 (16개)

**내부 0건 + 외부 검증 완료 (14개)**: Qwen2.5-0.5B, Qwen3-30B-A3B, gemma-2-2b,
gemma-3-270m, Llama-3.1-8B, Llama-3.1-70B, Phi-4, falcon-7b,
Mistral-Small-3.2-24B-Instruct-2506, SmolLM3-3B-Base, LFM2-8B-A1B,
ERNIE-4.5-21B-A3B-PT, Llama-4-Maverick-17B-128E, GLM-4.5-Air.

**개별 검증 완료, 구조적으로 "판단 필요 0건"에는 못 닿음 (2개)**: gpt-oss-20b,
gpt-oss-120b — `head_dim == num_attention_heads`, `hidden_size ==
intermediate_size`가 이 체크포인트 자체의 실제 하이퍼파라미터 우연이라
review_request.md가 항상 4건을 보고한다(문서화 필요, 미해결 아님 — 각 모델
`review_findings.json` 참고).

기준: (1) 판단 필요 0건(또는 구조적 이유로 불가능함을 소스로 문서화) + 게이트 PASS
+ 내부 외부 검토(③ 자유 평가) 완료, **그리고 대부분** (2) 이 저장소의 내부 코드를
전혀 안 보고 공식 논문/HF config/공식 GitHub 구현체만으로 독립 재현한 외부 검증
(Codex, 2026-08-31)에서 일치 판정 또는 지적된 버그를 실제 config로 재확인·수정.

이 파이프라인은 멀티모달 체크포인트에서 **텍스트 백본만** 트레이스한다(설계 의도,
`src/provenance.py` 참고 — vision encoder 등은 범위 밖). Mistral-Small-3.2는 Codex가
이 스코프 자체를 지적했었지만 의도된 설계로 확인돼 포함시켰다.

**알려진 잔여 갭 (막지는 않지만 투명하게 남김)**: ERNIE-4.5/GLM-4.5-Air는 Codex가
지적한 MTP(multi-token prediction) 레이어가 아직 `structure.yaml`에 반영 안 됨 —
핵심 아키텍처 수치(레이어 수/폭/전문가 구성)는 전부 수정·검증됐지만 이 부분은
후속 작업으로 남아 있음.

각 모델 폴더의 파일:
- `structure.yaml` — 심볼(축 이름) 표
- `model_summary.md` — 사람이 읽는 요약
- `prefill.csv` / `decode.csv`, `.jsonl` — 연산 표 (레이어 패턴으로 축약된 버전;
  레이어별 원본 전개는 `main` 브랜치의 `full/` 안에 있습니다)
- `review_findings.md` / `.json`, `review_request.md` — 검토 기록

## 갱신

`main`에서 결과물이 바뀔 때마다 이 브랜치에 새 스냅샷 커밋 하나가 추가됩니다
(히스토리 추적용이 아니라 매번 "지금 상태"만 필요하면 `--depth 1`로 받으세요).
