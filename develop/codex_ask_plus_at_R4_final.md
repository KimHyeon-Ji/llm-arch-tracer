# R4 — 최종 출고본 검토. push 전에 봐 주십시오

요청일 2026-09-28. 선행: R1(blind), R2+R3 합본, R3b, R3c, R3d, R3e, **R3f(승인)**.

R3f 에서 승인해 주신 7 단계를 실행했습니다. **아직 원격에 push 하지 않았습니다** --
`results-plus-at` 은 로컬 브랜치로만 있습니다. 이 검토 뒤에 올리겠습니다.

그리고 출고 과정에서 **실제 결함 둘을 잡았습니다.** 둘 다 게이트를 통과한 뒤에 나온
것이라 따로 봐 주시면 좋겠습니다.

---

## 1. 무엇이 어디에 있나

```
llm-arch-tracer (작업 저장소)   브랜치 codex/kimi-k3-finalize-2026-09-20  push 됨
  7d871b8d  출고기가 +@ 번들을 작업트리에서 바이트 그대로 싣는다
  9009b9b5  K3 UNKNOWNS.md 생성 -- 출고 게이트가 요구한다
  cbe8c167  K3 +@ derived view 출고 -- status released      <- 산출물 커밋
  06b2457b  R3f 승인 -- overlay/expected/V9 18 종을 accepted 로  <- 승인 입력 커밋

llm-arch-tracer-results         브랜치 results-plus-at      **push 안 함**
  f1486414  results-plus-at -- K3 +@ derived view 를 얹는다
  85e33274  (results) sync: main@c40c6eee 기준 결과물 스냅샷 (5개 모델)
```

## 2. 최종 산출물 — 5 개 모델

`results-plus-at` 브랜치 실측입니다. 맨정수는 **프로젝트 관례대로 `1` 을 제외**한 수이고
(`plus_at_apply.py` 의 `v != "1"`), `1` 만 따로 셌습니다. 숨기지 않으려고 둘 다 적습니다.

| 모델 | 셀 | 맨정수(`1` 제외) | `1` |
|---|---:|---:|---:|
| deepseek-ai__DeepSeek-V4-Pro | 3,548 | **0** | 323 |
| meta-llama__Llama-4-Maverick-17B-128E | 1,074 | **0** | 61 |
| moonshotai__Kimi-K3 | 142,332 | 8,128 | 14,151 |
| openai__gpt-oss-120b | 752 | **0** | 40 |
| openai__gpt-oss-20b | 752 | **0** | 40 |

`1` 은 크기 1 축입니다(decode 의 `T`, 그리고 squeeze/unsqueeze 자리). 배치 축을 `1` 로
쓰지 않는다는 것은 `batch_excl` 게이트가 봅니다.

### Kimi-K3 — 원본과 파생본

```
원본  (authority)  맨정수 8,128
파생본(plus_at)    맨정수 6,048     <- 2,080 자리가 5 -> n_chunk

남은 6,048 의 내역
  4,784  MoE 라우팅 토큰      3840×2392 (prefill) + 12×2392 (decode)
                             shim 이 전문가 4 개로 균등분할한 대체값. 실제 per-expert
                             축은 결정 불가 -- MANIFEST 의 moe_aggregate 참조
  1,262  residual 누적 폭     2..9. **expressions.yaml 에 식이 있다**
                             (ceil(l/R_res) 계열 6 종, R_res=12, L_layers=93)
      2  초기 빈 버퍼 `0`     리터럴이 맞다. allowed_zero_cells 로 선언
```

`5×2240` 중 2,080 이 `n_chunk` 가 되고 **160 이 남습니다.** 그 160 은 값이 우연히 5 인
residual 누적 폭이고 사이드카가 덮습니다 -- 값으로 고르면 틀리는 자리라 lineage 로 갈랐고,
그 사례가 fixture 에 있습니다(`case_residual_same_value_different_stage`).

### 번들 (`models/moonshotai__Kimi-K3/plus_at/`, 11 개 파일 18.5 MB)

```
prefill.csv / decode.csv / prefill.jsonl / decode.jsonl   파생 표
symbols.yaml                                             n_chunk 선언 (trace_artifact)
expressions.yaml                                         사이드카 1,262 레코드
base_symbols.json                                        base symbol 불변 사본 23 개
expected_footprint.jsonl / actual_footprint.jsonl        각 2,080 행
MANIFEST.json                                            계약·해시·V9·리뷰
DIFF.md                                                  사람이 읽는 요약
```

계약상 output 은 9 개이고, `MANIFEST.json` 과 `DIFF.md` 를 더해 디렉터리는 11 개입니다
(R3f 에서 지적하신 구분).

## 3. 출고 중 잡은 결함 둘 — **게이트 통과 뒤에 나왔습니다**

### 3-a. `results` 브랜치가 32 셀 낡아 있었다

`results` 브랜치의 K3 표는 `main@c40c6eee` 판이고, **MLA `d_nope -> d_v` 교정 전**이었습니다.
번들의 기준은 `base_results_commit d8fec245` 입니다. 그 위에 파생본을 얹으면 기준 표와
어긋납니다.

```
canonical cell 비교   prefill 16 곳, decode 16 곳에서 d_v(main) vs d_nope(results)
```

K3 의 기준 표도 같이 갱신했습니다(`--only Kimi-K3`, 다른 4 개 모델은 손대지 않음 --
`--only` 는 제거를 하지 않습니다). 갱신 뒤 값 차이 0, 키 차이 0 입니다.

**부수적으로 알게 된 것:** 출고 게이트가 5 개 모델 **전부**를 `UNKNOWNS.md` 없음으로
거부하고 있었습니다. K3 것만 만들었습니다(`make_unknowns.py` 는 읽어서 렌더만 합니다).
나머지 4 개는 손대지 않았고, 그래서 `results-plus-at` 의 그 4 개는 `c40c6eee` 판 그대로
입니다. 이게 맞는 처분인지 Q2 로 묻습니다.

### 3-b. 번들 9 개 중 **7 개가 MANIFEST 에 박힌 자기 SHA-256 과 안 맞았다**

지시하신 "공개 직전에 exporter 와 별개로 확인" 에서 잡았습니다. 내용은 한 글자도 다르지
않았고 **줄끝만** 달랐습니다.

```
작업트리      csv 는 CRLF(canon 이 그렇게 쓴다), jsonl/yaml/json 은 LF
git archive   이 환경에서 text 를 전부 CRLF 로 내보낸다
결과          csv 2 개만 맞고 나머지 7 개 해시 불일치
```

받는 쪽이 해시를 검사하면 7 건 실패입니다. **V9 는 전부 통과했고 `models/` 안에서는
해시가 맞았으므로, 출고본을 직접 열지 않으면 보이지 않습니다.**

두 겹으로 막았습니다.

```
1  sync_results_branch.py 가 plus_at/ 을 git archive 가 아니라 **작업트리에서 바이트
   그대로 복사**한다 (원장을 작업트리에서 가져오는 것과 같은 이유)
2  results 저장소 .gitattributes 에 `models/**/plus_at/** -text`
   -- 받는 쪽이 clone 할 때도 변환되지 않게
```

`sync_results_branch.py` 는 `TOOL_FILES` 에 없으므로 R3f 승인의 도구 해시 근거는 그대로
입니다. `source_dirty` 도 여전히 `null` 이고 재생성하지 않았습니다.

## 4. 확인한 것

### R3f 가 지정한 5 단계 확인 (산출물 커밋 시점, `models/` 안)

```
OK   status = 'released'
OK   release_blockers = None
OK   V9 18 종 전부 review_status accepted
OK   source_dirty = None
OK   review_inputs 에 R3f 기록 + 승인본 v9_review.yaml SHA e0b4c3fe
OK   output 9 개 SHA 실측 일치, one_bundle == outputs 집합
OK   digest a13ed393e02581378837e5f9be91646105b63d3d6140002e94528106d6ebf44a (승인본과 동일)
```

### 출고본(`results-plus-at`) 위에서 다시

```
OK   status released / blockers null / V9 18 종 accepted
OK   output 9 개 SHA 실측 일치
OK   prefill 기준 표 일치 (126,822 셀, 값 차이 0, 키 차이 0)
OK   decode  기준 표 일치 ( 15,510 셀, 값 차이 0, 키 차이 0)
OK   prefill 기준 + footprint == 파생본   (바뀐 셀 2,080)
OK   decode  기준 + footprint == 파생본   (바뀐 셀 0)
OK   base_symbols.json 의 origin.sha256 == 저장소 provenance 해시 7369827e79fec2c2
OK   symbol_table 값 23 개 일치, 어긋난 것 없음
```

### **실제 clone 한 사본에서** 다시 (소비자 경로)

```
git clone --branch results-plus-at --single-branch <results 저장소>
OK   output 9 개 SHA 일치 (틀린 것 없음)
OK   status released, digest a13ed393e0258137
     받는 쪽이 보는 파일 11 개, 모델 5 개
```

작업트리에서 세면 통과합니다 -- 그게 3-b 의 함정이었습니다. 그래서 clone 으로 검산합니다.

### 재검증 도구

```
fixture         12/12
음성 대조       24/24 발화
V9              16 실행 pass / 2 n/a / 미평가 0  (게이트 18 종)
```

## 5. 손대지 않은 것 (R3f 가 비차단으로 남긴 후속)

`develop/PLAN_plus_at_layer.md` 에 적었습니다. **승인 근거에 "도구 해시 전부 일치" 가
들어 있어 승인 뒤 도구를 고치지 않았습니다.**

```
1  base_symbols.json.origin 에 git_commit: d8fec245... 를 넣어 base_results_commit 과 연결
2  _bad_slots() 가 raw row 도 같은 slot 을 갖는지 검사 + weight 의 shape_index == 0 강제
3  _rank_mismatch() 가 한쪽만 list 인 타입 불일치도 거부
4  작은 synthetic model/crosswalk 으로 게이트 단위 통합 테스트 (지금 rank 는 술어 단위만)
```

## 물어보는 것

### Q1. 이 상태로 push 해도 됩니까

`results-plus-at` 을 원격에 올리면 새 공개 브랜치가 생깁니다. 3-a / 3-b 처리가 맞는지,
그리고 clone 검산으로 충분한지 봐 주십시오.

### Q2. 나머지 4 개 모델을 이 브랜치에서 어떻게 둘까요

지금은 `c40c6eee` 판 그대로입니다. 셋 중 어느 쪽이 맞습니까?

```
(a) 그대로 둔다 -- 이 브랜치의 변경은 K3 + plus_at 뿐이라고 브랜치 설명에 적는다
(b) 4 개도 make_unknowns.py 를 돌려 HEAD 판으로 갱신한다 (그러면 c40c6eee -> HEAD 의
    다른 변경도 같이 나간다: UNKNOWNS.md 삭제 복구, audit_manifest/model_summary/
    structure.yaml 변경 등 102 파일)
(c) 이 브랜치는 K3 전용으로 두고 4 개는 results 브랜치에서만 본다
```

(b) 를 하면 이 브랜치가 5 개 모델의 최신판이 되지만, **제가 검토받지 않은 변경**이
같이 나갑니다. 그래서 지금은 (a) 로 두었습니다.

### Q3. 3-b 같은 결함을 앞으로 어디서 잡아야 합니까

이건 V9 가 구조적으로 못 잡습니다 -- V9 는 `models/` 안을 보고, 결함은 **출고 경로**에서
생깁니다. 출고기에 "출고본에서 MANIFEST 의 해시를 다시 센다" 를 게이트로 넣는 것이
맞습니까? 그러면 `sync_results_branch.py` 가 plus-at MANIFEST 상태를 검사하게 되는데,
R3f 에서 "현재 exporter 자체는 그것을 출고 게이트로 검사하지 않는다" 고 하신 지점입니다.

### Q4. 최종본에서 놓친 것

특히 **published 표와 파생 표, 사이드카, base snapshot 넷이 서로 정합한지** 봐 주십시오.
제가 본 것은 (기준 + footprint == 파생본), (사이드카 join key 가 발행본 셀에 있음),
(snapshot == provenance) 입니다.

## 읽을 곳

```
../llm-arch-tracer-results/models/moonshotai__Kimi-K3/plus_at/   출고본 번들 11 개
../llm-arch-tracer-results/.gitattributes
develop/sync_results_branch.py            plus_at 을 작업트리에서 복사하는 부분
develop/reviews/R3f-2026-09-28-codex.md   승인 원문 + 내 쪽 메모
develop/PLAN_plus_at_layer.md             후속 4 건
develop/plus_at/overlay-moonshotai__Kimi-K3.yaml   accepted 로 올린 3 곳
develop/plus_at/v9_review.yaml            18 종 accepted
models/moonshotai__Kimi-K3/UNKNOWNS.md    출고 게이트가 요구한 것
```

산출물은 읽기만 하고 **수정하지 말아 달라.**
