# Codex 검토 요청 — Kimi-K3 재트레이스 검증 계획 사전 확인

## 배경

방금 두 가지 실수를 했다:
1. `git filter-repo`가 히스토리뿐 아니라 작업 디렉터리와 백업 브랜치까지 같이 재작성한다는
   걸 몰라서, Kimi-K3의 `prefill.trace.raw.jsonl`(300MB)/`prefill.csv`(121MB)를 로컬에서도
   영구히 잃었다(git 안에서도, 백업 브랜치 안에서도 복구 불가 확인함 -- reflog 비어있고
   pack 용량 자체가 그 파일 하나보다 작음).
2. `develop/promote.py`가 `develop/out/<model>`을 `models/<model>`로 옮길 때 디렉터리를
   통째로 바꿔치기한다는 걸 몰라서, `src/run.py`가 만들지 않는 수기 파일(`review_findings.json`)이
   삭제될 뻔했다(커밋 전에 발견해서 복구함).

사용자가 "실수가 계속 나오니 중간중간에 Codex한테도 물어보면서 하라"고 명시적으로 요청했다.
그래서 이번엔 **실행 전에** 계획을 검토받는다.

## 지금 하려는 것

1. `src/run.py --profile develop/models/phase25-kimi-k3.yaml --out develop/out` 로 Kimi-K3
   진짜 재트레이스 (진행 중, 백그라운드, ~40분 소요 예상 -- 2.8T MoE라 정상).
2. 완료되면 `develop/out/moonshotai__Kimi-K3/full/report.md`에서 C-check FAIL이 없는지 확인.
3. 오늘 낮에 고친 세 가지(`reused_symbol` 버그, `E_shared` 스코프, `n_kv` 스코프)가 Kimi-K3에
   실제로 반영되는지 확인:
   - decode의 `block_sparse_moe.experts.*.act_fn` 값 4가 더 이상 `2*E_shared`가 아니라
     bare `4`(또는 다른 이름)로 나오는지
   - prefill의 `self_attn` 안 값 2가 더 이상 `E_shared`가 아니라 bare `2`로 나오는지
   - Kimi-K3 전용 KDA 심볼(`n_h_kda`/`d_head_kda`, `scope_strict: true`)이 오늘 수정으로
     영향을 안 받고 그대로인지 (이 둘은 scope 밖일 때 드롭되지 promote 안 되므로 원래
     `reused_symbol` 경로를 안 탔을 것 같은데, 확실히 하고 싶다)
4. `rules/label_confirmed.yaml`의 Kimi-K3 selector 중 `shape: ["B", "n_h_kda", "5", "E_shared"]`
   (14133번째 줄 근처)를 실제 새 렌더 결과에 맞게 갱신 -- **새 shape을 추측하지 않고, 재트레이스
   결과에서 직접 읽어서** 넣을 것.
5. `develop/promote.py Kimi-K3`로 승격 (단, 이번엔 승격 **전에** `models/moonshotai__Kimi-K3/`
   에 `src/run.py`가 안 만드는 파일(`review_findings.json`, `review_findings.md`)이 있는지
   먼저 확인하고, 승격 후 그 파일들이 살아있는지 재확인한다).
6. `develop/regen_summaries.py`로 함대 전체 재생성 (fingerprint 갱신).
7. `develop/verify_all.py`로 회귀 0건 확인, 커밋.

## 확인받고 싶은 것

1. **위 순서·범위에 빠진 단계나 위험한 가정이 있는가?** 특히 4번(`label_confirmed.yaml` selector
   갱신)에서 "새 shape을 재트레이스 결과에서 직접 읽는다"는 접근이 맞는지, 아니면 더 안전한
   방법(예: `develop/rule_coverage.py` 같은 것으로 먼저 측정)이 있는지.
2. **Kimi-K3 KDA 전용 심볼들(`n_h_kda`/`d_head_kda`, `scope_strict: true`)이 오늘의 세 가지
   수정 중 어느 것에라도 영향을 받을 수 있는가?** `scope_strict: true`인 심볼은 스코프 밖일 때
   `miss_syms`에도 안 들어가고 완전히 드롭된다고 이해하고 있는데(정상 경로), `reused_symbol`
   폴백을 고친 게 이 드롭 자체를 우회할 방법을 새로 만들지는 않았는지 코드로 확인해줄 수 있는가?
   (`src/symbolic_shape.py` 최근 diff는 `if avoid: for s, v in ordered_ctx: if s in avoid and ...`
   한 줄 추가뿐이다.)
3. **재트레이스 후 Kimi-K3에서 오늘 세 수정과 무관하게 새로 나타날 수 있는 문제가 있는가?**
   (예: decode에 새 `concat` op가 이미 있었던 것과 별개로, 오늘 수정이 op_id 배치나 다른 축에
   부작용을 낼 가능성)

이번엔 코드를 이미 고친 게 아니라 **다음 단계를 실행하기 직전**이라, 실행 전에 이 계획 자체를
검토받고 싶다. 실행은 이 검토와 별개로(트레이스 자체는 이미 백그라운드에서 돌고 있음) 계속
진행하지만, 4~6번 단계는 Codex 의견을 받은 뒤에 진행한다.
