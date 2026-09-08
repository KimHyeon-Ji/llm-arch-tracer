# 인수인계: Codex CSV/JSONL 라벨 판정 검증 (중단 지점)

작성 2026-09-03. 직전 세션이 2026-09-02 21:24 에 VS Code 확장 호스트 크래시로
죽으면서 중단됨. 산출물은 하나도 안 남았음. **이 파일이 재개 지점이다.**

## 사용자가 원하는 것 (원문 기준)

> "요약 카드 말고 `<phase>.csv`, `<phase>.jsonl` 이 레이어별로 모델 구조를 심볼로
> 정확히 잘 표현했는지 외부 소스 보고 객관적으로 꼼꼼하게 검토 받아서, 다 완성된
> 것만 results 브랜치에 올려야 해."

> "정확히 모델 아키텍처를 표현하는 정확한 `<phase>.csv` / `<phase>.jsonl` 을 만드는 게
> 가장 큰 일이자 이 프로젝트의 목표야."

- 대상은 **`model_summary.md` 가 아니라 `<phase>.csv` / `<phase>.jsonl` 의 실제 축 라벨**이다.
- Kimi-K3 는 **보류**(사용자 지시: "kimi k3는 멈추고 다른 모델들 검토 및 확정부터").
- 최종 목적지는 **results 브랜치 커밋/푸시**.

## 지금까지 진행된 것

1. 검토 요청 패킷 작성 완료: `develop/codex_review_request_csvjsonl_PREAMBLE.md`
   + `codex_review_request_csvjsonl_batch{A..J}.md`, 묶음 요청문은
   `develop/codex_ask_csvjsonl_batches.md`.
2. **Codex 답변 수령 완료** → `develop/codex_answer_csvjsonl_20260902.md`
   (28개 모델 검토, 23개에서 심볼 오류 주장, 배치 A~J 표 형식).
3. 그 주장들의 **검증은 0건**. 여기서 끊겼다.

## 다음에 할 일

`codex_answer_csvjsonl_20260902.md` 의 주장을 **배치 A부터 순서대로, 하나씩** 검증한다.
Codex 답변을 그대로 믿고 라벨을 바꾸지 않는다 — 오탐을 심는다. 각 주장마다:

1. 지목된 자리의 **실제 CSV/JSONL 행**을 연다 (`models/<model>/full/<phase>.csv|jsonl`).
2. **설치된 transformers 5.14.1 소스**와 대조한다 (GitHub main 아님 —
   메모리 `evidence-must-be-the-traced-version` 참조).
3. `CONFIRMED` / `REJECTED` / `UNCERTAIN` 판정 + 근거 인용을 남긴다.
4. CONFIRMED 만 `rules/label_overrides.yaml` 로 반영한다
   (메모리 `verdicts-reach-the-tables`: 근거 인용 필수).
5. `develop/verify_all.py` 를 돌려 exit 0 확인 (메모리 `verify-all-gate`).

## 반드시 지킬 실행 규칙 (사고 재발 방지)

- **백그라운드 서브에이전트를 3개 이상 동시에 띄우지 않는다.** 직전 세션이 정확히
  이걸로 두 번 죽었다. 병렬이 필요하면 최대 2개, 기본은 순차.
- **배치 하나 끝날 때마다 즉시 `develop/verdicts_csvjsonl_batch<X>.md` 에 append 한다.**
  세션 메모리에만 있는 판정은 크래시 한 번에 0이 된다. 이번 사고의 실제 피해 원인이다.
- 진행 상황도 이 파일 하단에 갱신한다.

## 진행 표

| 배치 | 상태 | 판정 파일 |
|---|---|---|
| A | **완료 — 8/8 CONFIRMED, 0 REJECTED** | `develop/verdicts_csvjsonl_batchA.md` |
| B | **완료 — 9/9 CONFIRMED, override 15개, 3개 모델 promote** | `develop/verdicts_csvjsonl_batchB.md` |
| C | **완료 — 9/11 CONFIRMED, 2 REJECTED. 3개 모델 promote** | `develop/verdicts_csvjsonl_batchC.md` |
| D | **완료 — 4/4 CONFIRMED. 3개 모델 promote** | `develop/verdicts_csvjsonl_batchD.md` |
| E | **완료 — 1 CONFIRMED / 2 REJECTED. Granite promote** | `develop/verdicts_csvjsonl_batchE.md` |
| F | **완료 — 4/4 CONFIRMED (F-3 은 Codex 절반 오류 정정). V4-Flash promote** | `develop/verdicts_csvjsonl_batchFGHIJ.md` |
| G | **완료 — F 와 동일. V4-Flash-0731 promote** | `develop/verdicts_csvjsonl_batchFGHIJ.md` |
| H | **완료 — RoPE 3/3 CONFIRMED, value 경로 3/3 REJECTED. Kimi 3종 promote** | `develop/verdicts_csvjsonl_batchFGHIJ.md` |
| I | **완료 — 4/4 CONFIRMED. Nemotron Super/Ultra + Falcon-H1 promote** | `develop/verdicts_csvjsonl_batchFGHIJ.md` |
| J | **완료 — 6/7 CONFIRMED, J-3 은 적용 불가로 open. GLM-5.2 promote** | `develop/verdicts_csvjsonl_batchFGHIJ.md` |

---

# 막힌 지점 — 2026-09-03 (배치 A 반영 시도)

배치 A 판정 8건을 `rules/label_overrides.yaml` 에 22개 엔트리로 적어 넣었다(YAML 유효,
`develop/verdicts_csvjsonl_batchA.md` 에 전건 근거). **그러나 적용이 안 된다.**

## 원인

`develop/regen_tables.py` 는 `full/<phase>.trace.raw.jsonl` 을 읽어 `build_table.write_outputs`
로 다시 내보낸다. 그런데 그 jsonl 의 shape 은 **이미 심볼**이다:

    trace.raw.jsonl : "input_shape": [["V", "d_model"], ["B", "T"]]
    shapes.concrete.jsonl : "input_shape": [[100352, 4096], [1, 16]]

`label_overrides.apply()` 의 매처는 `expect`(정수)와 **구체 값**을 대조해야만 발화한다
(`if str(s) == frm and isinstance(c, int) and c == want`). regen 경로에는 정수가 없으므로
**어떤 override 도 구조적으로 발화할 수 없다.** `build_table.py:1803` 근처의
`_dims_are_concrete(rows)` 주석이 이 경로를 이미 명시하고 있다 —
"on the regen-from-symbolic path (develop/regen_tables.py) the ints are already gone".

실측 확인: 3개 모델에 regen 을 돌린 결과 `label_overrides.json` 의 **모든** 엔트리가
`applied: 0` 이 됐다(OLMo-2 는 기존 14건 3008 발화 → 0). CSV/JSONL 은 HEAD 와 바이트
동일이라 데이터 훼손은 없었고, concrete sidecar 도 무사했다(2026-08-04 가드가 지켰다).
장부 파일은 `git checkout` 으로 전부 복구했다 — **현재 models/ 는 세션 시작 상태 그대로다.**

## 남은 것

`rules/label_overrides.yaml` 의 신규 22개 엔트리는 아직 발화하지 않는다. 게이트는
"발화 0건인 override" 를 FAIL 로 잡으므로, 적용 경로가 생기기 전까지 이 22건은 게이트를
붉게 만든다. (이 파일은 세션 시작 시점에 이미 이전 세션의 미커밋 변경 +1026/-122 을
담고 있었다 — `git checkout` 금지.)

## 다음 단계 후보

1. **`regen_tables.py` 에 concrete 재생 경로를 추가** — sidecar(정수)를 `rows` 로 쓰고,
   저장된 심볼 shape 을 op_id 로 돌려주는 "재생 resolver" 를 쓴다. sidecar 가 존재하는
   이유가 정확히 이것이다. 다만 `rows` 가 concrete 이 되면 `anchors` 패스도 함께 살아나므로
   (build_table.py:1803) override 외의 라벨까지 바뀔 수 있다 — 회귀 위험을 재봐야 한다.
2. **`src/run.py --profile ...` 로 3개 모델 재트레이스** — 확실하지만 무겁고, 재트레이스는
   T 값 등 다른 것도 바꿀 수 있다(메모리 `t-fixed-at-trace-time`).

1번이 옳은 방향으로 보인다. 재트레이스 없이 적용된 과거 커밋(예: 1245d780 은
`prefill.csv` + `prefill.trace.raw.jsonl` 만 바꾸고 sidecar 는 안 건드렸다)이 있으므로
그런 경로가 존재했거나 존재해야 한다.


---

# 확정된 작업 절차 (2026-09-03, 배치 A 에서 실증됨)

앞서 "regen 으로는 override 를 적용할 수 없다"고 적은 것은 맞지만, **정식 경로가 따로
있었다.** `regen_tables.py` 가 아니라 재트레이스다:

    .venv\Scripts\python.exe src
un.py --profile develop/models/<프로파일>.yaml --out develop/out
    .venv\Scripts\python.exe develop\promote.py <필터>

프로파일은 `develop/models/*.yaml` 에 47개 있다. 재트레이스는 concrete shape 을 다시
만들어내므로 override 매처(`expect` 정수 대조)가 정상 발화한다. 과거 커밋에서 sidecar 가
diff 에 안 뜬 이유는 **재트레이스가 같은 정수를 내놓아 파일이 바이트 동일**이었기 때문이다.

## 배치 하나의 한 사이클

1. `codex_answer_csvjsonl_20260902.md` 의 그 배치 주장을 실제 CSV 행 + 설치된 5.14.1
   소스로 대조 → `develop/verdicts_csvjsonl_batch<X>.md` 에 **즉시** 기록
2. CONFIRMED 만 `rules/label_overrides.yaml` 에 자리 한정자와 근거 인용을 달아 작성
3. 해당 모델 재트레이스 → `full/label_overrides.json` 에서 **미발화 0건** 확인
4. 라벨이 의도대로만 바뀌었는지 육안 확인(건드리면 안 되는 이웃 축이 그대로인지)
5. `report.md` C-FAIL 0 확인 → promote

## 전부 끝난 뒤

6. `develop/regen_summaries.py` 로 신선도 스탬프 정리
7. `develop/verify_all.py` exit 0
8. `develop/make_review_packet.py` 로 **수정된 CSV/JSONL 기준 패킷을 다시 생성** →
   Codex 재검토 요청 (사용자 지시 2026-09-03: "모델 수정해서 결과 나오면 codex 한테 다시
   검토해서 통과하면 results 브랜치에 올릴 것")
9. 통과분만 results 브랜치 (`develop/sync_results_branch.py`)

## 배치 A 실적

3개 모델 promote 완료. 미발화 override 0건, C-FAIL 0.

| 모델 | override 엔트리 | 발화 |
|---|---|---|
| allenai__OLMo-2-1124-7B-Instruct | 15 (신규 1) | 4,416 |
| MiniMaxAI__MiniMax-M2 | 5 (전부 신규) | 2,108 |
| Qwen__Qwen3-Next-80B-A3B-Instruct | 26 (신규 20) | 9,012 |


## 배치 B 실적 (2026-09-03)

| 모델 | override | 발화 | 미발화 | 전치 위반 | C-FAIL |
|---|---|---|---|---|---|
| Qwen__Qwen3.5-397B-A17B | 17 | 8,640 | 0 | 0 | 0 |
| Qwen__Qwen3.5-4B | 13 | 4,176 | 0 | 0 | 0 |
| allenai__OLMoE-1B-7B-0924 | 19 | 2,304 | 0 | 0 | 0 |

Codex 지적 9건 외에 재트레이스 후 잔존 확인으로 **B-6b** 를 추가로 잡았다: Qwen3.5-4B 의
bare `self_attn` view 가 k/v 평탄 축을 `n_h*d_rope` 로 들고 있었다(B-6 은 k_proj/v_proj
서브모듈만 덮었다). **재트레이스 뒤 잔존 심볼을 세어 보는 절차가 이런 누락을 잡는다 —
사이클에 넣을 것.**

## 배치 C 진행 중 확인된 것

- Qwen3.6-27B / Qwen3.6-35B-A3B 는 아키텍처가 `Qwen3_5*ForConditionalGeneration` 로
  **Qwen3.5 와 같은 modeling 소스**를 쓴다. 따라서 C-2/C-3/C-4/C-5 는 B-5/B-7/B-4/B-1 과
  동일한 수정이다(`rule-set-converges` 대로 규칙이 수렴한다).
- C-1(Qwen3.6-27B rotate_half)은 partial rotary 라 `d_head/2` 가 아니라 `d_rope/2` 다.
  CSV 행 자체가 자명하다: 입력 `[B,n_h,T,d_rope]` 를 slice 한 결과가 `n_h+2*n_kv` 일 수 없다.

## 부수 발견 (Codex 미지적, 별도 처리 필요)

`rotate_half`/head-count 식 전 모델 스캔 결과 5개 모델이 걸렸는데 **2개는 오탐**이다:
- Phi-4 240행 `(n_h+2*n_kv)*d_head` / falcon-7b 64행 `[B,T,n_h+2*n_kv,d_head]`
  → **정당한 fused QKV 폭**이다. 건드리지 말 것.
- Qwen3-Next 36행 / Qwen3.6-35B 30행: `[B, 2*n_h*d_head, n_h+2*n_kv]` -> `[B, ..., T]`
  → conv 패딩 시퀀스 축이 head-count 식으로 잘못 붙었다. **진짜 결함이나 Codex 미지적.**


---

# 2026-09-03 마감 상태 — Codex 재검토 대기

## 산출물

46개 모델 전부 재트레이스 + promote 완료. 전수 검사 결과:

| 지표 | 결과 |
|---|---|
| 미발화 override | 0 / 46 |
| 등가류 충돌(flow_ambig) | 0 / 46 (회귀 시 32모델 → 0) |
| param_incons | 0 (8 → 0) |
| C-FAIL | 0 / 46 |
| 신선도 스탬프 | 0 (42 → 0) |

## Codex 재검토 패킷 — 생성 완료

`develop/make_csvjsonl_batches.py` (이번에 새로 만든 조립 스크립트)로 배치 A~J 재생성.
**수정된 CSV/JSONL 기준**이며 구성은 1차 라운드와 동일하다(판정 파일과 대조하려면 그래야 한다).
요청문은 `develop/codex_ask_csvjsonl_batches.md` — 1차 라운드 대비 무엇이 바뀌었는지,
무엇을 반려했는지, 무엇을 못 고쳤는지를 적어 두었다.

반영 확인(패킷 안에 실제로 들어갔는지 grep 으로 검증):
`n_kv*d_head`(OLMo-2/Falcon-H1) · `d_head/2`(MiniMax) · `2*n_h_lin_v`(Qwen3-Next) ·
`2*d_moe`(OLMoE) · `d_rope/2`(Qwen3.6-27B) · `d_v`(V2-Lite) · `n_h_ssm`(Granite) ·
`c_I-d_rope`(V4 indexer) · `d_rope`(Kimi) · `c_I`(GLM-5.2) 전부 확인.
옛 라벨 `n_h*d_rope` · `d_chunk, d_model` 은 0건.

## 게이트에 남은 FAIL 208건의 성격

| 유형 | 규모 | 성격 |
|---|---|---|
| `의뢰서 항목 N건에 대응하는 판정이 없다` | 37개 모델 | **③ 자유 평가 미수행**. `review/prompt.md` 를 모델별로 돌려야 닫힌다. 라벨 결함이 아니라 프로세스 단계다 |
| 낡은 확인 기록 | 7건 / 5모델 | 재앵커 대상. 확인된 축은 그대로 맞고 **이웃 축 이름만** 바뀌었다. `develop/reanchor.py` 가 이 용도다. Kimi-K3 2건은 세션 전부터 존재 |
| Kimi-K3 근거 없는 확인 908건 · bare 퇴행 | 3 | 세션 전부터 존재 |
| GLM-4.5-Air `reshape_incons` 0→90 | 1 | override 를 넣지 않은 모델. 현재 규칙으로 재트레이스하며 드러난 잠재 결함 — **미조사** |

## 다음 단계

1. `develop/codex_ask_csvjsonl_batches.md` 를 Codex 에 전달 → 배치별 답변 수령
2. 받은 주장을 **또다시 하나씩 검증**(1차 라운드에서 7/48 이 오탐이었다)
3. 통과분만 results 브랜치 (`develop/sync_results_branch.py`)
4. 별건: GLM-4.5-Air reshape_incons 조사, ③ 자유 평가 37개 모델, reanchor 7건

## 이번 라운드에서 배운 것 (메모리에도 남김)

- 새 override 는 **재트레이스로만** 적용된다([[override-apply-needs-retrace]])
- 내가 쓴 override 는 `spread: class` 를 빠뜨려 16개 모델에 등가류 충돌을 만들었다.
  기존 466개 중 대부분이 이미 쓰고 있었다. **모듈 스코프 override 는 기본으로 spread 를 붙여라.**
- 모델 하나 고칠 때마다 게이트를 돌렸어야 했다. 전치 불변식·잔존 심볼 수로는 등가류 충돌이 안 잡힌다.
- 외부 검토의 계열 단위 지적은 모델별로 확인해야 한다([[verify-each-model-not-family]])


---

# 2차 라운드 (Codex 2026-09-03 재검토) — 진행 중

## 1차 수정 검증 결과: 11개 모델 "이상 없음" 통과

MiniMax-M2, xLSTM-7b, OLMo-2, OLMoE, tiny-deepseek-v3, DeepSeek-V2-Lite, gpt2-xl,
DeepSeek-V3, Llama-3.1-405B, Hunyuan-A13B, Falcon-H1. **잘못 고친 것은 없었다.**

## 새 진단 도구 — "이름 생성" 불변식

축을 **지우거나 그대로 두기만** 하는 op(select/squeeze/_to_copy/clone/sum/copy_/mean)는
출력 이름 다중집합이 입력의 부분집합이어야 한다. 아니면 없던 이름이 생긴 것이다.

전 모델 스캔: **10개 모델 1,154행 위반.** Codex 2차 지적의 대부분(A2/B2/B3/C1/C2/C4/
D3~D6/E1~E7/I1~I11)을 독립적으로 재현했고, **Codex 가 못 본 것도 나왔다**:
Kimi-K3 414행, DeepSeek-V3 116행(Codex 는 "이상 없음" 판정한 모델).

전치/permute 불변식과 함께 쓰면 소스 없이 산출물 내부만으로 결함을 짚을 수 있다.

## 배치 A~C 의 GDN 계열 — 완료 (5개 모델 promote)

`select` 가 살아남은 축의 이름을 바꾸던 자리. **세 번 실패하고 네 번째에 됐다:**

1. 출력 shape 앵커 → 정상 그룹 72건까지 뒤집었다(같은 출력 shape 을 공유).
2. spread 제거 → 여전히 뒤집힘.
3. 앵커 모드(from==to)+spread → **발화 0**. select 가 rank 를 바꿔 등가류가 끊긴다
   ([[class-spread-stops-at-rank-change]] 와 같은 현상).
4. **nth(모듈 내 서수)로 그 op 만 지목 + spread: class** → 성공.

교훈: **shape 이 그룹을 못 가르면 nth 를 쓴다.** decode 의 select 세 그룹은
입력끼리·출력끼리 shape 이 겹쳐서 shape 만으로는 원리적으로 못 가른다.

| 모델 | 미발화 | 등가류 | 전치 | 이름생성 | C-FAIL |
|---|---|---|---|---|---|
| Qwen3-Next-80B | 0 | 0 | 0 | 0 | 0 |
| Qwen3.5-4B | 0 | 0 | 0 | 0 | 0 |
| Qwen3.5-397B | 0 | 0 | 0 | 0 | 0 |
| Qwen3.6-27B | 0 | 0 | 0 | 0 | 0 |
| Qwen3.6-35B | 0 | 0 | 0 | 0 | 0 |

fused-Q decode 짝(A1/B1/C3)도 함께 반영. 1차에서 shape 에 prefill 의 'T' 를 박아
decode('1')를 놓친 것으로, override 전수 감사로 재현 확인했다.

## 남은 2차 지적

| 배치 | 건수 | 대상 |
|---|---|---|
| D | 7 | Zamba2 attention n_h→n_kv, Mamba head/feature 교차, tiny-Llama |
| E | 7 | Granite Mamba 전 구간 |
| F·G | 8 | V4-Flash 2종 RoPE 잔여 |
| H | 8 | Kimi 3종 k_rot, **Nemotron-Nano 신규 3건** |
| I | 11 | Nemotron Super/Ultra mixer 전 구간 |
| J | 5 | GLM-5.2 (**J1 query projection 은 신규**) |

추가로 이름생성 불변식이 잡은 **Kimi-K3 414행 / DeepSeek-V3 116행** 은 Codex 미지적이다.


## 2차 라운드 배치 D — 완료 (2개 모델 promote)

| 지적 | 판정 | 결과 |
|---|---|---|
| D1 Zamba2 value transpose `n_h`→`n_kv` | CONFIRMED | 반영 |
| D2 Zamba2 key rotate_half slice `n_h`→`n_kv` | CONFIRMED | 반영 |
| D3~D5 Zamba2 Mamba head/feature 교차 | CONFIRMED | 반영(select nth=0, sum nth=1) |
| D6 Zamba2 decode select | CONFIRMED 이나 **적용 실패 — open** | 아래 |
| D7 tiny-Llama key slice `n_h`→`n_kv` | CONFIRMED | 반영 |

**내 1차 판단의 정정**: Zamba2/tiny-Llama 의 전치 위반을 전부 "repeat_kv(n_rep=1) 경계
미추적" 으로 뭉뚱그려 남겼는데, **그 안에 고칠 수 있는 것이 섞여 있었다.** 진짜 경계는
attention matmul 로 들어가는 `transpose [B,n_kv,..] -> [B,n_h,d_head,T]` 하나뿐이다.
Codex 도 정확히 같은 구분을 했다("repeat_kv 이후 축은 n_h 로 해석 가능하므로 보고하지 않음").

| 모델 | 전치 | 이름생성 | 등가류 | 미발화 | C-FAIL |
|---|---|---|---|---|---|
| Zamba2-1.2B | 30 → **12** (기지 케이스만) | 18 → **6** | 0 | 0 | 0 |
| tiny-random-Llama | 4 → **4** (기지 케이스만) | 0 | 0 | 0 | 0 |

### D6 을 못 고친 이유 (open)

`select [B,1,d_head_ssm] -> [B,n_h_ssm]` 6행. 위반은 자명하나 두 번의 시도가 모두 악화:
- `spread: class` → 이름생성 6→32, 전치 12→50 (등가류가 입력까지 번져 뒤집음)
- spread 없이 출력만 → 이름생성 6→32, 등가류 0→38

같은 모듈의 기존 판정(shape `["B","n_h_ssm","n_h_ssm","d_state"]`, spread: class)과
상호작용하는 것으로 보인다. 억지로 밀지 않고 남겼다.

### 이 과정에서 낸 사고와 복구

제거 필터를 `(model, op_type, nth, from, to)` 로만 걸어 **이전 세션의 유효한 판정 2건까지
지웠다**(shape 을 안 봤다). 새 위반 38행으로 즉시 드러났고 `git diff` 의 삭제 블록에서
원문을 찾아 복원했다. **엔트리를 지울 때는 shape 까지 포함해 지목할 것.**


## 2차 라운드 배치 E·F·G — 완료 (3개 모델 promote)

| 배치 | 모델 | 지적 | 결과 |
|---|---|---|---|
| E | ibm-granite__granite-4.0-h-small | E1~E7 mamba head/state 축 교차 | 이름생성 108 → **0**, 전 지표 0 |
| F | deepseek-ai__DeepSeek-V4-Flash | F1~F4 RoPE 잔여 | 전 지표 0 |
| G | deepseek-ai__DeepSeek-V4-Flash-0731 | G1~G4 (F 와 동일) | 전 지표 0 |

### 내 1차 수정의 구멍 (F·G)

1차에서 `module: 'self_attn$'`(bare, 끝 앵커)와 **prefill shape** 만 덮어서
`compressor` / `indexer` / decode 짝을 놓쳤다. 전부 같은 자명한 결함이었다:

    view [.., d_rope/2, 2] -> [.., n_h]      (interleaved 쌍을 합치면 d_rope)

같은 파일의 다른 view 가 이미 `[.., d_rope/2, 2] -> [.., d_rope]` 로 정확히 렌더링하고
있었다 -- **한 모델 안에서 같은 연산이 두 이름으로 갈리면 그 자체가 증거다.**

### 이번 구간의 작업 규칙 (Zamba2 에서 얻음)

1. **spread 없이 먼저 넣고 측정** -> 이름생성이 0 이 되는지 확인
2. 등가류 충돌이 생기면 **그때 spread 추가** -> 다시 측정
3. 두 지표가 동시에 0 이 아니면 되돌린다 (Zamba2 D6 이 그 경우)

Granite 는 1->2 로 깨끗하게 닫혔고(108→0, 등가류 108→0), V4-Flash 는 slice 엔트리에만
spread 를 더해 닫혔다.


## 2차 라운드 배치 H — 완료 (4개 모델 promote)

| 지적 | 판정 | 근거 |
|---|---|---|
| H1~H5 Kimi 3종 k_rot `n_h`→`d_rope` | CONFIRMED | `split_with_sizes [B,T,c_kv+d_rope] -> [B,T,c_kv]*[B,T,n_h]` -- 입력이 c_kv+d_rope 이고 첫 조각이 c_kv 이므로 둘째는 d_rope. **자체 모순** |
| H6 Nemotron-Nano k/v_proj `n_g*d_state`→`n_kv*d_head` | CONFIRMED | `modeling_nemotron_h.py:873-874`. 어텐션 투영에 SSM 심볼이 붙었고, 바로 다음 행이 이미 `[B,T,n_kv,d_head]` 다 |
| H7 Nemotron-Nano repeat 축 `n_h/n_g_ssm`→`n_h/n_kv` | CONFIRMED | `modeling_nemotron_h.py:827` -- n_rep = n_h // n_kv |
| H8 Nemotron-Nano conv 길이축 `n_h/n_g_ssm`→`d_conv+1` | CONFIRMED | 같은 행이 `[.., d_conv]` 와 `[.., 1]` 을 concat 한다 |

Kimi-K2.7-Code 의 prefill 엔트리는 **불필요**했다 -- Codex 도 "prefill 은 이미 d_rope 로
올바르다" 고 적었고 미발화로 확인되어 제거했다.

**Nemotron-Nano 는 1차에서 "이상 없음" 이었던 모델이다.** 1차 수정으로 산출물이 바뀌면
2차에서 새 지적이 나올 수 있다는 실증.

주의: `n_g*d_state` 는 SSM 경로(`mixer.in_proj` / `conv1d` / `act`, 735행)에서는
**정당하다**. k_proj/v_proj 서브모듈로만 한정했다.

| 모델 | 미발화 | 등가류 | 전치 | 이름생성 | C-FAIL |
|---|---|---|---|---|---|
| Kimi-K2-Instruct / K2.6 / K2.7-Code | 0 | 0 | 0 | 0 | 0 |
| Nemotron-3-Nano-4B | 0 | 0 | 0 | 0 | 0 |


## 2차 라운드 배치 I·J — 완료 (3개 모델 promote)

| 배치 | 모델 | 결과 |
|---|---|---|
| I | Nemotron-3-Super-120B / Ultra-550B | 이름생성 120·96 → **0**, 전 지표 0 |
| J | zai-org__GLM-5.2 | 전 지표 0 |

### J1 은 4행 안의 자체 모순이었다 (Codex 신규 지적)

    q_b_proj  t/matmul  [n_h*d_v, c_q]
    self_attn view      [B,T,n_h*d_v] -> [B,T,n_h,d_v]
    self_attn split     [B,n_h,T,d_v] -> [B,n_h,T,d_nope] * [B,n_h,T,d_rope]

즉 `d_v = d_nope + d_rope` 인데 query 폭을 value 폭 이름으로 부르고 있었다.

**spread 를 붙이면 안 되는 자리였다**: 같은 `self_attn` 의 transpose nth=2(key)·nth=3
(attention 출력)은 `d_v` 가 맞는 자리라 등가류가 얽힌다. 실제로 spread 를 붙였더니
전치 위반이 0 → 156 으로 뛰었고(1차 J-3 과 같은 숫자), nth=0 만 짚어 해결했다.

J4/J5 는 indexer 의 **key 경로**만 남아 있었다 -- query 경로는 1차에서 이미 고쳐졌다.

---

# 2차 라운드 총괄

**배치 A~J 전부 완료. 20개 모델 promote.**

| 배치 | 모델 | 상태 |
|---|---|---|
| A·B·C | Qwen3-Next, Qwen3.5-4B/397B, Qwen3.6-27B/35B | 완료 |
| D | Zamba2, tiny-random-Llama | 완료 (D6 1건 open) |
| E | Granite-4.0-H-Small | 완료 |
| F·G | DeepSeek-V4-Flash, -0731 | 완료 |
| H | Kimi-K2-Instruct/K2.6/K2.7-Code, Nemotron-3-Nano | 완료 |
| I | Nemotron-3-Super, -Ultra | 완료 |
| J | GLM-5.2 | 완료 |

## 두 불변식의 함대 전체 추이

| 지표 | 1차 시작 | 2차 시작 | 지금 |
|---|---|---|---|
| 전치/permute 위반 | 1,072행 / 13모델 | — | **582행 / 4모델** |
| 이름생성 위반 | — | 1,154행 / 10모델 | **536행 / 3모델** |

남은 것의 정체:

| 모델 | 전치 | 이름생성 | 성격 |
|---|---|---|---|
| moonshotai__Kimi-K3 | 444 | 414 | **Codex 패킷에 포함된 적이 없다**(28개 대상 밖). 미검토 |
| deepseek-ai__DeepSeek-V4-Pro | 122 | 0 | 동일하게 패킷 밖. 미검토 |
| deepseek-ai__DeepSeek-V3 | 0 | 116 | Codex 는 두 라운드 다 "이상 없음" 판정. **내 불변식만 잡는다** |
| Zyphra__Zamba2-1.2B | 12 | 6 | 전치 12 = 기지 repeat_kv 경계 / 이름생성 6 = D6 open |
| tiny-random-Llama | 4 | 0 | 기지 repeat_kv 경계 |

## 이번 라운드에서 굳어진 작업 절차

1. **패킷 행번호로 실제 행을 열어 자체 모순부터 확인**한다. 소스는 그 다음이다.
2. shape 이 정상 그룹과 겹치면 **`nth`(모듈 내 서수)로 그 op 만** 짚는다.
3. **spread 없이 먼저** 넣고 이름생성이 0 이 되는지 본다.
4. 등가류 충돌이 생기면 **그때 spread** 를 더한다.
5. 두 지표가 동시에 0 이 아니면 **되돌린다**(Zamba2 D6, GLM-5.2 J1 의 spread).
6. 엔트리를 지울 때는 **shape 까지 포함해 지목**한다(이전 세션 판정 2건을 날린 적 있다).


---

# 미검토 모델 정리 (2026-09-03, 2차 라운드 이후)

두 불변식이 잡은 잔여 위반의 **96%** 가 Codex 패킷(28개 대상)에 **한 번도 들어간 적이 없는**
두 모델에 있었다: moonshotai__Kimi-K3, deepseek-ai__DeepSeek-V4-Pro.

## DeepSeek-V4-Pro — 완료 (promote)

전치 위반 **122 → 0**. V4-Flash 의 F-4 와 완전히 같은 결함이었고, 1차에서 내가 엔트리를
V4-Flash 2종으로만 한정해 놓쳤다:

    transpose [T, T/m_hca, n_h*d_head/g_o] -> [g_o, T, n_h*d_head/g_o]

**엔트리를 그대로 복사했더니 미발화였다.** `g_o` 가 V4-Flash 는 8, **V4-Pro 는 16**
(T=2048, m_hca=128)이기 때문이다. `expect` 가드가 조용히 틀린 축에 이름을 붙이는 대신
미발화로 알려준 것 -- 이 가드가 제 역할을 했다.

**교훈: 같은 결함을 다른 모델에 복제할 때 `expect` 는 반드시 그 모델에서 다시 잰다.**

## Kimi-K3 — 진행 중

n_h = n_h_kda = 96, d_v = d_head_kda = 128 **완전 충돌**. KDA 경로의 conv 채널 축에
MLA 심볼이 붙었다(전치 위반 207행):

    transpose [B, n_h*d_v, T] -> [B, T, n_h_kda*d_head_kda]   (출력은 이미 맞다)

`n_h*d_v` 분포: q/k/v_conv1d 각 759, bare self_attn 489, q/k/v_proj 각 414, g_proj 351.
**bare self_attn 은 MLA 레이어와 섞여 있어** 거기서는 n_h*d_v 가 정당할 수 있다.
그래서 KDA 전용 서브모듈(q/k/v_conv1d)로만 좁혀 잡고 효과를 측정 중이다.

남은 이름생성 414행은 성격이 다르다 -- bare int 가 출력에서 심볼로 바뀌는 자리다:

    sum [B,n_h_kda,5,d_chunk,4]  -> [B,n_h_kda,5,d_conv]
    sum [B,n_h_kda,5,d_chunk,48] -> [B,n_h_kda,5,n_h_kda/2]

`48 -> n_h_kda/2` 처럼 **산술적으로만 맞는 이름**이 붙은 것으로 보인다
([[heuristic-fabricated-labels]] 계열). KDA 는 fla 참조 구현을 타므로 외부 눈이 필요하다.

## 3차 패킷 준비 — 배치 K 신설

`develop/make_csvjsonl_batches.py` 에 배치 K(Kimi-K3 + DeepSeek-V4-Pro)를 추가했다.
1·2차에서 이 두 모델이 빠져 있었던 것이 잔여 위반이 여기 몰린 이유다.


---

# 3차 라운드 패킷 생성 완료 (2026-09-03) — Codex 전달 대기

## 산출물 상태

46개 모델 전부 재트레이스 + promote. 전수 검사:

| 지표 | 결과 |
|---|---|
| 미발화 override | 0 / 46 |
| 등가류 충돌 | 0 / 46 |
| param_incons | 0 |
| 신선도 스탬프 | **0** (90 → 0) |
| C-FAIL | 0 / 46 |

게이트 FAIL 208건의 구성:

| 유형 | 규모 | 성격 |
|---|---|---|
| `의뢰서 항목 N건에 대응하는 판정이 없다` | 32개 모델 | ③ 자유 평가 미수행 — 프로세스 단계, 라벨 결함 아님 |
| 낡은 확인 기록 | 7건 / 5모델 | 재앵커 대상(`develop/reanchor.py`). Kimi-K3 2건은 세션 전부터 |
| Kimi-K3 근거 없는 확인 908건 · bare 퇴행 | 2 | 세션 전부터 |
| GLM-4.5-Air reshape_incons 0→90 | 1 | **재측정 시 0.** 게이트가 promote 직전 상태를 읽은 것으로 보인다 — 다음 라운드에 재확인 |

## 패킷 — `develop/codex_ask_csvjsonl_batches.md` 하나만 넘기면 된다

배치 A~K, **30개 모델**(기존 28 + Kimi-K3 + DeepSeek-V4-Pro). 총 3.4MB.
전부 2차 수정이 반영된 산출물이며, grep 으로 반영·소거를 검증했다.

3차 요청문에 새로 담은 것:
1. **배치 K 를 특히 봐달라** — 1·2차 패킷에 한 번도 없던 두 모델. Kimi-K3 는
   `n_h = n_h_kda = 96`, `d_v = d_head_kda = 128` 완전 충돌에 KDA 가 fla 참조 구현을 탄다.
2. **내 자체 검사 두 개가 이미 0** 이라는 사실(배치 K 제외). 1·2차엔 이걸 안 알려줘서
   Codex 와 내 검사가 같은 것을 중복해 잡았다. 이제 "같은 op 안에서 모순이 없는 오라벨"에
   집중할 수 있다.
3. 2차에서 **내가 정정한 것**(F3 조각 순서, D1/D2/D7 의 repeat_kv 구분)과
   **못 고친 3건의 사유**.

## Kimi-K3 잔여 — 배치 K 로 넘긴 이유

전치 444 → 237 (KDA conv 채널 207행 해소), 이름생성 414 는 그대로.
남은 것은 성격이 다르다 -- **bare 정수 ↔ 심볼**, **B ↔ 1** 전환이라 두 불변식으로는
어느 쪽이 옳은지 판정할 수 없다:

    permute [B,n_h_kda,5,d_chunk,d_head_kda] -> [B,n_chunk,d_chunk,n_h_kda,d_head_kda]
    permute [B,n_h,T,1,d_nope+d_rope]        -> [n_h,T,d_nope+d_rope,1,1]
    sum     [B,n_h_kda,5,d_chunk,48]         -> [B,n_h_kda,5,n_h_kda/2]

`48 = 96/2` 라 산술적으로는 맞지만 그 축이 정말 헤드 개수의 절반인지는 소스를 봐야 한다.

## 이번 라운드의 사고 두 건 (둘 다 즉시 복구)

1. **줄 번호 기반 편집으로 엉뚱한 엔트리 손상**: V4-Pro 의 `expect` 를 고치려다
   V4-Flash 의 다른 엔트리를 `8 → 16` 으로 바꿨다. 재트레이스 후 그 모델만 등가류 86건으로
   튀어 즉시 드러났고 `git show HEAD` 로 원값을 확인해 복구했다.
   **줄 번호로 YAML 을 편집하지 말 것.**
2. **재트레이스 후 규칙을 또 고쳐 45개 모델이 다시 낡음**: 신선도 스탬프가 90건 떴다.
   **전 모델 재트레이스는 규칙 편집을 완전히 멈춘 뒤 마지막에 한 번만.**


---

# 3차 라운드 (Codex 2026-09-04) — 처리 완료

## 결과 요약

**A~F, H~J 전부 "이상 없음".** 28개 모델의 2차 수정이 전면 통과했다.
Codex 가 F3(V4-Flash indexer)에 대해 "앞 조각은 c_I-d_rope, 뒤 조각은 d_rope 로 구분돼
있다" 고 명시 확인했는데, 이는 2차에서 **내가 Codex 의 지적을 정정한** 부분이다.

신규 지적 23건의 분포:

| 배치 | 건수 | 성격 |
|---|---|---|
| G | 2 | **내가 만든 과잉 치환** |
| K (Kimi-K3) | 9 | 첫 검토 |
| K (V4-Pro) | 12 | 첫 검토 |

## G — 같은 엔트리가 모델마다 다르게 퍼진다

2차에서 V4-Flash 두 모델에 **동일한 엔트리**를 넣었는데 결과가 갈렸다:

    F: prefill nth=20 -> c_I-d_rope, nth=21 -> d_rope    (정확)
    G: prefill nth=20 -> c_I-d_rope, nth=21 -> c_I-d_rope (틀림)

`spread: class` 가 0731 에서만 앞/뒤 두 slice 를 하나의 등가류로 묶었다.
**shape 이 같아도 모델마다 등가류 구성이 다르다 -- spread 는 모델별로 결과를 재야 한다.**
nth 로 뒤쪽 조각만 되돌려 F 와 동일하게 만들었다.

## K — 세 값 충돌

- `T/m_hca = g_o = n_hc*n_hc = 16`
- `n_h_I = d_rope = c_I-d_rope = 64`
- `d_head = T/m_csa = 512`
- (Kimi-K3) `n_h = n_h_kda = 96`, `d_v = d_head_kda = 128`

대부분 행 안에서 자체 모순으로 확인됐다. 예:

    split [(2+n_hc)*n_hc] -> [n_hc]*[n_hc]*[T/m_hca]   ; 뒤 view 가 [n_hc, n_hc] 로 되접는다
    concat [.., w_local-1, ..]*[.., 1, ..] -> [.., n_h, ..]
    concat 출력이 w_local+T/m_csa 인데 피연산자가 n_h, d_head
    split [B,n_h,1,d_nope+d_v] -> [.., d_nope]*[.., d_nope]

**K2 는 소스를 읽어야만 알 수 있었던 것**이다: `sum [.., d_chunk, 4] -> [.., d_conv]` 의
`4` 는 `fla/ops/kda/naive.py:134-136` 의 루프 prefix(`A[..., i, :i]`, i=4)이고 conv 커널
폭(d_conv=4)과는 값만 같다. 내 두 불변식은 "bare 정수 -> 심볼" 이라 판정하지 못한다.

## 게이트가 내 부작용을 잡았다 — `reshape_incons`

3차 수정이 **입력만 고치고 출력을 놓친** 자리를 정확히 짚었다:

| 모델 | 결과 |
|---|---|
| zai-org__GLM-4.5-Air | 90 -> **0** (`view [T,k] -> [E]` 는 k*T 다) |
| moonshotai__Kimi-K3 | 72 -> 24 -> **0** |
| deepseek-ai__DeepSeek-V4-Pro | **90, open** |

두 불변식(재배열·이름생성)으로는 `view` 의 자체 유도 불일치가 안 잡힌다. 게이트가
없었으면 "입력은 맞고 출력은 옛 이름" 상태로 패킷을 보낼 뻔했다.

## V4-Pro rotary-table view — open

    view [B, T/m_csa, d_rope/2, 2] -> [B, d_head, d_rope]

입력 축 1 은 K11 수정으로 T/m_csa 가 됐는데 출력이 d_head 로 남았다. 세 번 시도해 세 번
실패했다(입력 spread / 출력 명시 고정 / spread 제거+출력 고정 -> 등가류 0->180 악화).
K11~K15 는 소스가 명시하고 Codex 도 확인한 실제 교정이므로 유지하고, 이 한 자리만 남긴다.

## 수렴 곡선

| 라운드 | 지적 | 오탐 | 성격 |
|---|---|---|---|
| 1차 | 48 | 7 | 28개 모델 전반 |
| 2차 | 57 | 0 | 1차가 못 닿은 나머지 자리 |
| 3차 | 23 | 0 | **21건이 첫 검토 모델 2개**, 2건이 내 실수 |

**기존 28개 모델은 3차에서 지적 0건이다.**


---

# 4차 라운드 (Codex 2026-09-04) — 29/30 모델 통과

## 결과

**배치 A~J 전부 "이상 없음", Kimi-K3 도 "이상 없음".**
첫 검토를 받은 Kimi-K3 가 한 라운드 만에 통과했고(KDA conv/projection 폭, MLA d_v,
KDA 루프의 리터럴 4 가 모두 수정됨), G 의 `c_I-d_rope` 과잉 치환도 해소돼
"F 와 G 의 대표 트레이스가 동일하다" 는 확인을 받았다.

**남은 것은 deepseek-ai__DeepSeek-V4-Pro 하나뿐이다 (11건).**

## V4-Pro — d_head <-> T/m_csa 계열을 되돌렸다 (open)

> **2026-09-04 해결됨.** 이 절의 "open" 은 더 이상 유효하지 않다.
> 문서 맨 끝 "DeepSeek-V4-Pro 값 충돌, **해결**" 절을 보라 — T=1920 probe 로 갈랐다.

Codex 4차의 핵심 지적: *"CSA용 T/m_csa 치환이 HCA 및 KV feature 축으로 번져 원래 맞던
d_head 를 바꾼 과잉 수정"*. 실측으로 확인됐다:

    concat [B,1,1,d_head-d_rope] * [B,1,1,d_rope] -> [B,1,1,T/m_csa]    (d_head 여야 함)

이 모델은 **축 등가류가 CSA window 축과 HCA/KV feature 축을 하나로 묶는다.** 그래서
무엇을 고쳐도 반대쪽이 오염된다. 2026-09-04 에 네 가지를 시도해 모두 실패했다:

1. 입력만 spread -> 출력이 되돌아옴
2. 출력을 명시적으로 고정 -> 발화하는데도 최종 렌더에서 되돌아옴
3. spread 제거 + 출력 고정 -> 등가류 0 -> 180 악화
4. 대응 엔트리(T/m_csa -> d_head) 추가 -> **미발화**(선행 엔트리가 먼저 바꿔 from 이 안 맞음)

**결론: 이 계열 9개 엔트리를 전부 되돌렸다.** 자명한 교정(K10 `n_hc*n_hc`,
K16 `w_local`)은 유지한다. 되돌린 뒤 V4-Pro 는 전 지표 0 이다
(reshape_incons 90 -> 0, 전치 0, 등가류 0, 미발화 0, C-FAIL 0).

근본 해결은 `src/axis_classes.py` 가 이 두 축을 분리하도록 고치는 것이고, 함대 전체에
영향이 가는 변경이라 별도 작업으로 남긴다.

## 수렴 곡선 (최종)

| 라운드 | 지적 | 오탐 | 성격 |
|---|---|---|---|
| 1차 | 48 | 7 | 28개 모델 전반 |
| 2차 | 57 | 0 | 1차가 못 닿은 나머지 자리 |
| 3차 | 23 | 0 | 21건이 첫 검토 모델 2개, 2건이 내 실수 |
| 4차 | 11 | 0 | **전부 V4-Pro 한 모델**, 그중 4건이 내 과잉 치환 |

**45개 모델이 외부 검토를 통과했고, V4-Pro 한 모델만 open 이다.**

---

## 2026-09-04 — DeepSeek-V4-Pro 값 충돌, **해결**

3차 라운드에서 마지막으로 남았던 항목. `d_head == T/m_csa == 512` 라 트레이스 정보만으로
CSA window 축과 KV feature 축을 가를 수 없어 "원리적 미해결"로 남길 뻔했으나, Codex 의
multi-length probe 제안으로 풀었다.

### 방법

같은 모델을 **T=1920** 으로 한 번 더 트레이스한다(`develop/models/probe-deepseek-v4-pro-T1920.yaml`,
`--out develop/probe_out`, **promote 하지 않는다**). 1920 은 `m_csa=4` 와 `m_hca=128` 의
공배수라 나머지 없이 구조가 유지되면서 세 축이 전부 갈린다: d_head=512, T/m_csa=480, T/m_hca=15.
두 실행의 op 수는 (module, op_type) 별로 완전히 동일했다(46,268개).

옮기는 도구는 `develop/probe_transfer.py` 로 남겼다. 결과: 클래스 1,770건 교정 → 대표 항목 52개
(`rules/label_overrides.yaml`, 전부 `layer_types` + `spread: class`).

### 네 번 틀리고 나서 얻은 규칙 (전부 probe_transfer.py 의 docstring 에 박아 뒀다)

1. **shape 으로 자리를 대응시키지 않는다.** 두 실행은 shape 이 달라지는 것이 목적이다.
   안정 키는 (module_key, layer_type, op_type, module-local nth, field, shape_index, axis).
2. **layer_type 을 키에 넣는다.** `module_key` 가 층 인덱스를 `.*` 로 접으므로 hca/csa 가
   교대하는 61층에서 csa 층에서 잰 값이 hca 층에 적용됐다 → 등가류 충돌 91건.
3. **옮기는 단위는 자리가 아니라 등가류다.** 자리 단위로 옮기면 한 등가류가 두 이름으로
   쪼개진다 → 충돌 90건 + reshape_incons 60건. 클래스마다 대표 하나 + `spread: class`.
4. **반증을 요구한다.** probe 가 base 이름을 실제로 깨뜨린 자리가 하나는 있어야 한다.
   없으면 probe 에서도 두 이름의 값이 같다는 뜻이고, 이름을 바꾸는 건 판별이 아니라 probe
   실행 라벨러의 잡음을 베끼는 것이다. 이 조건이 486 클래스를 걸렀다.

**base 는 probe 교정이 없는 트레이스여야 한다.** 처음에 이미 교정이 적용된 develop/out 을
base 로 써서 386개를 헛짚었다.

### 부수 효과 하나

기존 항목 하나(`indexer$` / elementwise_add / nth=2, `n_h_I -> d_rope`)가 발화 0 이 됐다.
앵커 `shape: [B, 1, d_head, n_h_I]` 의 축 2 가 `T/m_csa` 로 바뀌었기 때문. shape 앵커를 뺐다 --
module + op_type + nth + field + shape_index + axis + expect 만으로 이미 한 자리를 정확히
지목하므로 잉여였다. **교훈: shape 앵커는 다른 교정이 같은 shape 의 다른 축을 건드리면 낡는다.**

### 결과

`develop/out/deepseek-ai__DeepSeek-V4-Pro`: reshape_incons 0 / 전치 0 / 이름생성 0 /
미발화 0 / 등가류 0 / C-FAIL 0. 승격 완료.

표는 이제 소스 그대로 읽힌다:

    new_zeros  in [[B, T/m_csa, m_csa, 2*d_head]]  out [[B, T/m_csa, 2*m_csa, d_head]]
    concat     in [[B, 1, T/m_csa, d_head-d_rope], [B, 1, T/m_csa, d_rope]] -> [[B, 1, T/m_csa, d_head]]
    topk       in [[B, T, T/m_csa]]

(`modeling_deepseek_v4.py:646-657` — `n_windows = chunk_kv.shape[1] // self.compress_rate`,
`chunk_kv.view(batch, n_windows, ratio, -1)`, `new_zeros((batch, n_windows, 2*ratio, head_dim))`.)
hca 층은 `T/m_hca` / `d_head` 로 그대로 남았다.

### 남은 값 충돌 3건에 같은 방법이 쓰이는가

probe 는 **T 로부터 유도되는 축**만 가른다. 남은 3건은 성격이 다르다:

| 모델 | 묶이는 두 축 | 값 | probe 로 되나 |
|---|---|---|---|
| zai-org__GLM-5.2 | 캐시 key ↔ value | 256 | ✗ 둘 다 config 상수. seq_len 을 바꿔도 같이 안 움직인다 |
| Zyphra__Zamba2-1.2B | mamba head ↔ head feature | 64 | ✗ 같은 이유 |
| MiniMax-M2 / OLMoE / V4-Flash 2종 | experts gate-up 원본 ↔ 전치본 | d_model=2*d_moe | ✗ 같은 이유 |

이 셋을 풀려면 **config 를 바꾼 모델**(예: d_model != 2*d_moe 인 같은 계열)로 probe 해야 하고,
그건 다른 아키텍처를 트레이스하는 것이라 판정 승격 가드(계열을 넘지 않는다)에 걸린다.
그대로 미해결로 문서화한다.

### 다음

전 모델 재트레이스 → `develop/verify_all.py` → 배치 A~K 재조립 → Codex 4차 검토 → results 브랜치.

## 2026-09-05 — 전 모델 재트레이스 + 패킷 재조립

트레이스 **47/47 성공**, 승격 **47/47**(SKIP 0). 함대 전체 하드 불변식:

| 지표 | 값 |
|---|---|
| 등가류 충돌 | **0** |
| `reshape_incons` | **0** |
| 미발화 override | **0** |
| C-check FAIL | **0** |
| 전치/permute 위반 | 277 (지난 582) — Kimi-K3 261, Zamba2 12, tiny-Llama 4 |
| 이름생성 위반 | 467 (지난 536) — Kimi-K3 345, DeepSeek-V3 116, Zamba2 6 |

V4-Pro 의 122/0 은 **0/0 이 됐다.**

### 배치 구성을 고쳤다 — 17개 모델이 검토 밖이었다

`develop/make_csvjsonl_batches.py` 의 배치 A~K 는 **30개 모델뿐**이었다. 1차 패킷을 손으로
28개 고른 구성이 라운드마다 그대로 이어졌고, 그래서 아래 17개는 CSV/JSONL 외부 검토를
**한 번도** 받지 못했다. results 브랜치에 올리기 전에 전 모델이 한 번은 받아야 하므로 채웠다.

- L 소형 dense: SmolLM3-3B, Qwen2.5-0.5B, gemma-2-2b, gemma-3-270m
- M 표준 dense: Llama-3.1-8B, Llama-3.1-70B, Phi-4, falcon-7b
- N 중형 MoE/hybrid: Mistral-Small-3.2-24B, Qwen3-30B-A3B, ERNIE-4.5-21B-A3B, LFM2-8B-A1B
- O 대형 MoE: gpt-oss-20b, gpt-oss-120b, Llama-4-Maverick-17B-128E
- P 하이브리드: GLM-4.5-Air, Kimi-Linear-48B-A3B-Instruct

배치 K 는 911,650자로 다른 배치의 3.5배였다. 검토자가 끝까지 못 읽으면 하필 이번에 가장 크게
바꾼 모델이 검토를 못 받으므로 **K1(Kimi-K3) / K2(V4-Pro)** 로 쪼갰다.

**패킷 수록 47 / models/ 47 — 빠진 모델 없음**을 기계적으로 확인했다.

### 게이트 잔여 FAIL 223건의 정체 (라벨 결함 아님)

- 32개 모델의 ③ 자유 평가 미실행("의뢰서 항목에 대응하는 판정이 없다") — **절차** 미이행
- Kimi-Linear-48B 의 신규 값 충돌 7건 — `references.yaml` 등재 대기
- Qwen3.5-397B 의 낡은 확인 기록 1건 — reanchor 대상
- Kimi-K3 `bare` 퇴행 446,563 → 448,840 — 기존 미해결

### 다음

`develop/codex_ask_csvjsonl_batches.md` 하나만 넘기면 된다(A~J, K1·K2, L~P 총 17개 파일 참조).
통과하면 `develop/sync_results_branch.py` 로 results 브랜치.

---

## 2026-09-05/06 — 외부 검토 라운드 5 반영 완료

Codex 지적 8건 **전부 CONFIRMED, 오탐 0.** 판정 전문은 `develop/verdicts_csvjsonl_round5.md`.

### 함대 최종 상태 (47개 모델, 47/47 재트레이스 + 승격)

| 지표 | 라운드 5 전 | 지금 |
|---|---|---|
| 등가류 충돌 | 0 | **0** |
| 미발화 override | 0 | **0** |
| C-check FAIL | 0 | **0** |
| `unsqueeze` singleton `B` | 5,399 | **0** |
| 전치/permute 위반 | 277 | 277 (전부 Kimi-K3 261 + 기지 경계 16) |
| 이름생성 위반 | 467 | **351** (DeepSeek-V3 116 → 0) |
| reshape_incons | 622 | **474** |

### 이번 라운드에서 굳어진 것

1. **`unsqueeze` 가 새로 끼운 축은 0번이 아니면 배치가 아니다** — `_unsqueeze_inserts_singleton`
   + `batch_excl` 게이트 검사 + selftest 주입. 규칙을 정하기 전에 함대를 재 본 것이 핵심:
   축 0 삽입 1,045행은 전부 정당한 `B`, 0번 아닌 삽입 5,399행은 전부 잘못된 `B`.
   더 넓은 "B 는 축 0" 규칙은 307,958행이 걸려 무효였다.
2. **probe 가 두 번째로 통했다** — GLM-4.5-Air 의 `E == k*T`(둘 다 128)를 T=24 로 갈랐다.
   `develop/probe_transfer.py` 가 모델 인자를 받도록 일반화됐다.
3. **`_schedule()` 이 `linear_attn_config.full_attn_layers` 도 읽는다** — Kimi-Linear 처럼
   `layer_types` 대신 1-based 목록으로 스케줄을 적는 모델에 `layer_types:` 선택자를 쓸 수 있다.

### 물린 것 두 가지 (둘 다 실측이 근거)

- **Llama-4 의 `E -> E*B`**: decode 는 B=1 이라 `E*B == E` 이고, 같은 등가류에 진짜 `E` 축이
  있어 충돌 24건. 마지막 축의 리터럴 1 만 남겼다.
- **교정 뒤 전치 불변식 재적용**: Kimi-K3 전치 261→168 이지만 충돌 0→117, Zamba2 는 전치
  12→0 이지만 충돌 0→12 이고 **정당한 `repeat_kv(n_rep=1)` 경계를 밀어버린 퇴행**이었다.
  좁혀도(판정이 쓴 입력 축만) 나빴다. 시도와 수치를 `build_table.py` 에 남겼다.

### 이번 라운드에 다시 밟은 실수

- **Windows CR**: 셸에 넘길 목록을 python `print` 로 만들면 줄 끝에 CR 이 붙어
  `develop/models/<name>\r.yaml` 을 열려다 죽는다(17건 × 2회). 목록은 python 이
  `newline='\n'` 으로 **직접 파일에 써야** 한다(`develop/.../pick_stale.py` 패턴).
- **생성기를 두 번 돌리면 사라지는 항목**: Kimi-Linear 의 102개는 한 번 반영하면 트레이스에
  `n_h*d_v` 가 남지 않아 재생성이 0건을 낸다. YAML 의 목록이 원본이고 주석에 적어 뒀다.
- **일괄 치환 금지**: `shape` 앵커의 `"B"` 를 정규식으로 한꺼번에 `"1"` 로 바꿨다가, 정상
  발화하던 6건까지 깨뜨렸다. `label_overrides.json` 의 id 에 트레이스 시점 상태가 남아 있어
  복구했다. **발화 0 인 것만 고쳐야 한다.**

### 남은 미해결 4건 — 전부 같은 뿌리

축 등가류가 값이 겹치는 두 축을 하나로 묶는다. `src/axis_classes.py` 를 고치지 않는 한
후처리로는 한쪽을 고치면 다른 쪽이 깨진다.

| 모델 | 묶이는 두 축 | 값 |
|---|---|---|
| Kimi-K3 | MLA `n_h`/`d_v` ↔ KDA `n_h_kda`/`d_head_kda` | 96 / 128 |
| GLM-5.2 | 캐시 key ↔ value | 256 |
| Zamba2-1.2B | mamba head ↔ head feature | 64 |
| Granite / Nemotron-Super | B/C `repeat_interleave` 의 head ↔ state | 128 |

Codex 가 제안한 lineage 보존(stride/storage 로 전치본 판별, `Cache.update` 인자 슬롯 전파)이
여기에 맞는다. 트레이서 코어 변경이라 별도 작업으로 남긴다.

### 게이트 잔여 (최상위 FAIL 52건, 라벨 결함 아님)

- 33개 모델의 ③ 자유 평가 미실행 — **절차** 미이행
- 낡은 확인 기록 7건 — reanchor 대상
- 새 값 충돌 7건 — `references.yaml` 등재 대기
- Kimi-K3 / Kimi-Linear 의 membership 검사 62/57건 — `KimiLinearDecoderLayer` 가
  설치된 transformers 에 없어(remote code) 모듈이 읽는 config 필드를 알 수 없다.
  **검사기의 사각지대이지 라벨 결함이 아니다.**

### 다음

`develop/codex_ask_csvjsonl_batches.md` 하나만 넘기면 된다(A~J, K1·K2, L~P = 17개 파일).
통과하면 `develop/sync_results_branch.py` 로 results 브랜치.

---

## 2026-09-06 — 외부 검토 라운드 6 반영

지적 15건 전부 CONFIRMED. **핵심은 개별 교정이 아니라 Codex 가 제안한 두 op-local 규칙이었다.**

| 규칙 | 적용 가능 | 위반 | 반영 |
|---|---:|---:|---|
| conv1d 길이 공식 | 1,145 | 156 | `_conv1d_length_axis` |
| split 항-순서 보존 | 2,476 | 14 | `_split_keeps_term_order` |
| expand 비방송축 이름 보존 | 90,835 | 653 | **게이트 검사만**(방향이 양쪽이라 자동 교정 불가) |

**손으로 쓴 판정 39건이 규칙으로 승격됐다.** 7개 모델의 `split_with_sizes` 교정이 전부 발화 0
이 됐고, 78자리가 이미 목표 이름임을 확인한 뒤 지웠다(860 → 821). `rule-set-converges` 의
실증 사례가 하나 더 생겼다.

### 이번 라운드에 배운 것 두 가지

1. **규칙은 패스가 실제로 도는 지점의 상태로 재야 한다.** 위반을 *발행된* 라벨로 세고
   (conv1d 156 + split 14) 패스를 넣었더니 파이프라인 중간 상태가 달라 훨씬 많이 발화해
   함대 충돌 0 → 1,331 을 만들었다.
2. **op-local 규칙도 옮기는 단위는 등가류다.** 한쪽 끝만 고치면 클래스가 두 이름으로
   쪼개진다. `_spread_slots_to_class` 로 규칙이 고친 자리만 그 축 전체에 퍼뜨려 0 으로
   되돌렸다(`_unify_axis_classes` 재호출은 그 사이 결정을 되돌리므로 쓰지 않았다).

### 되돌린 것 (전부 실측이 근거)

- **Granite / Nemotron-Super `d_state ↔ n_h_ssm`**: 등가류 충돌 108/160.
- **Qwen3-Next conv_dim**: 전치 위반 72. **축 등가류에 전치 간선이 없어** spread 가 멈춘다
  (`transpose-edge-rejected` 참고). A1/A3(`key_dim == d_model == 2048`)은 반영했다.

### 최종 상태 (47/47)

등가류 0 / 미발화 0 / C-FAIL 0 / unsqueeze singleton 0.
전치 277 · 이름생성 351 · reshape_incons 474 · **expand 653(신규 가시화)**.

### 다음 작업은 하나로 좁혔다

남은 결함이 전부 **축 등가류가 값이 겹치는 두 축을 묶는 것** 하나로 수렴한다.
`src/axis_classes.py` 에 provenance(producer op / output slot / config-field origin /
module family)를 보존하는 것이 정공법이고, Codex 도 세 라운드 연속 같은 제안을 했다.

---

## 2026-09-07 — 라운드 7: expand 653건 중 339건 해결

외부 검토가 **expand 불변식 653건을 전부 실제 오류로 확인**했다(오탐 0). 방향까지 소스로
지정해 줘서 반영했고, **한 방향 교정과 맞바꿈 교정이 갈렸다.**

| 묶음 | 형태 | 결과 |
|---|---|---|
| Kimi-K2 x3 + GLM-5.2 (339) | `k_pe` feature = `d_rope` — 한 방향 | `spread: class` → **expand 0, 등가류 0** |
| Granite / Nemotron-Super (296) | head ↔ state 맞바꿈 | 되돌림 |
| Zamba2 (18) | head ↔ head_dim 맞바꿈 | 되돌림 |

### 이번에 확실해진 규칙

**맞바꿈은 `spread` 로 표현할 수 없다.** `spread: class` 는 한 클래스에 이름 하나를 쓰는
연산이다. 같은 등가류 안의 두 축을 서로 바꿔야 하는 교정은 자리 단위로 넣을 수밖에 없고,
그러면 클래스가 쪼개져 충돌이 난다(실측: 등가류 0 → 887). **결함 653을 887로 바꾸는
거래이므로 하지 않는다.** 한 방향 교정만 지금 도구로 풀린다.

### 최종 상태 (47/47)

등가류 0 / 미발화 0 / C-FAIL 0 / unsqueeze singleton 0.
전치 277 · 이름생성 351 · reshape_incons 474 · **expand 653 → 314**.

### 다음 작업 = provenance 보존 (외부 검토와 합의됨)

남은 결함이 전부 여기로 수렴한다. 필요한 origin 정보(외부 검토 지정):

- 생성 위치와 operand/axis 번호
- `split` 의 출력 슬롯 번호
- `concat` 구성 항의 순서
- `Cache.update` 의 key/value 인자 위치
- MLA / KDA / SSM / linear-attention 같은 **branch namespace**
- transpose/view/reshape/expand 를 통과한 축 계보

**`expand` 검사는 자동 교정 규칙이 아니라 acceptance gate 로 유지한다.** 같은 크기 축을
어느 이름으로 통일할지는 upstream origin 이 정해야 한다. Qwen3-Next A2 의 전치 단절도
별도 예외가 아니라 같은 provenance graph 단절이다.

**검증 척도가 이미 준비돼 있다**: 되돌린 314건은 정답이 소스로 확정돼 있으므로,
provenance 가 제대로 들어갔다면 이것들이 저절로 풀려야 한다.

### 남은 미해결 목록 (전부 같은 뿌리)

| 모델 | 묶이는 두 축 | 값 | 드러나는 지표 |
|---|---|---|---|
| Kimi-K3 | MLA `n_h`/`d_v` ↔ KDA `n_h_kda`/`d_head_kda` | 96 / 128 | 전치 261, 이름생성 345 |
| Granite / Nemotron-Super | dt·A·B/C 의 head ↔ state | 128 | expand 216 / 80 |
| Zamba2-1.2B | mamba head ↔ head feature | 64 | 전치 12, 이름생성 6, expand 18 |
| Qwen3-Next | conv_dim ↔ `2*n_h*d_head` | 8192 | (A2, 전치 간선 부재) |
| GLM-5.2 | 캐시 key ↔ value | 256 | (J3~J7, 미반영) |
| experts gate-up 4종 | 원본 ↔ 전치본 | `d_model = 2*d_moe` | param_incons |
