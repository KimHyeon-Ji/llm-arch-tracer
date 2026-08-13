# models/

**검증 통과한 완성 산출물**을 모델별 폴더로 둔다 — `01-main.md §9` 체크리스트(C1~C16)를 통과한 추출 결과(`<phase>.csv/.jsonl`, `structure.yaml`, `model_summary.md`, `full/`)만 여기 온다. "여기 있는 건 다 믿을 수 있다"가 보장되는 곳이다.

작업 흐름 (`develop/` → `models/` 승격):
1. `develop/models/<name>.yaml` — 프로파일(레시피·재생성 소스)을 여기서 작성/보관.
2. `python src/run.py --profile develop/models/<name>.yaml --out develop/out/` — 출력은 `develop/out/`에서 검증·반복.
3. `report.md`가 FAIL 0(+ C13 PASS)이면 그 출력 폴더를 `develop/out/`에서 이 `models/`로 옮긴다.

4. `python develop/verify_all.py` — 단일 게이트. **EXIT 0** 이어야 한다.
5. `review/prompt.md` 를 LLM 에 넘겨 ③ 라벨 검토를 돌린다. 게이트가 **의뢰서 질문 수와 판정 수를
   대조**하므로, 배정된 일을 안 하면 통과하지 못한다.

여기 있는 결과가 보장하는 것:

- C1~C17 체크 FAIL 0, 하드 불변식 12종 0 (**prefill·decode 양쪽**)
- 산출물이 **현재** `rules/` + `src/` 로 생성됨 (`full/generated.json` 지문을 게이트가 대조)
- ③ 검토 수행 완료 — 미결로 남은 판정은 `model_summary.md` 의 「③ 라벨 검토」 절에
  **"지금 렌더 / 소스가 말하는 것 / 왜 못 고쳤는지"** 로 실려 있다

시행착오·검증 전 결과는 여기 두지 않는다 — 통과한 완성품만. 포맷/코드가 바뀌면 재추적 없이 `develop/regen_*.py`로 이 폴더의 결과물을 제자리 갱신한다(각 폴더의 `full/` jsonl에서).
