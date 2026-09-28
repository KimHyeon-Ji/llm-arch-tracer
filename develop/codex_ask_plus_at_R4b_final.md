# R4b — 부분 출고 MANIFEST 와 exporter 재현성을 고쳤습니다. push 승인을 청합니다

요청일 2026-09-28. 선행: R1(blind), R2+R3, R3b~R3e, **R3f(번들 승인)**, **R4(번들 최종
승인 / push 보류)**.

지난 판정: *"남은 문제는 의미론이나 bundle 내용이 아니라 부분 출고 MANIFEST 와 exporter
재현성입니다. 이를 좁게 수정하고 clean committed exporter 로 다시 만든 뒤, output 해시와
clone 검산이 다시 통과하면 push 해도 됩니다."*

지정하신 범위만 고쳤습니다. **번들은 한 바이트도 바뀌지 않았습니다** -- digest `a13ed393…`,
output 9 개 해시 그대로이고 `plus_at_*` 도구는 손대지 않았습니다.

```
작업 저장소   codex/kimi-k3-finalize-2026-09-20
  9c21a142  R4 판정 원문 기록 (covers [R4])
  c74846bf  출고기가 .gitattributes 를 직접 두고, 게이트가 -text 속성까지 본다
  10a4fe2e  출고기: 혼합 snapshot 의 커밋 메시지와 push 안내를 실제 상태로
  547f4ee3  출고기: 부분 출고 MANIFEST 를 상속으로 적고, 출고본을 직접 검사한다

출고 저장소   results-plus-at   **아직 push 안 함**
  2e22cdc5  sync: main@c74846bf 기준 부분 갱신 (갱신 1개 / 상속 4개)
  85e33274  (results) sync: main@c40c6eee 기준 결과물 스냅샷 (5개 모델)
```

---

## 1. 부분 출고 MANIFEST — 권장하신 표현 그대로

```json
"base_results_commit": "85e33274",
"updated_models":  ["moonshotai__Kimi-K3"],
"inherited_models": ["deepseek-ai__DeepSeek-V4-Pro",
                     "meta-llama__Llama-4-Maverick-17B-128E",
                     "openai__gpt-oss-120b", "openai__gpt-oss-20b"],
"verified": {
  "deepseek-ai__DeepSeek-V4-Pro":          {"status": "inherited/not_revalidated",
                                            "inherited_from": "85e33274", ...},
  "meta-llama__Llama-4-Maverick-17B-128E": {"status": "inherited/not_revalidated", ...},
  "moonshotai__Kimi-K3":                   {"status": "updated/current",
                                            "source_ref": "c74846bf…", ...},
  "openai__gpt-oss-120b":                  {"status": "inherited/not_revalidated", ...},
  "openai__gpt-oss-20b":                   {"status": "inherited/not_revalidated", ...}
},
"withheld": { … 41 개, 디스크에 **없는** 모델만 … }
```

`note` 도 고쳤습니다.

> 이 snapshot 은 혼합이다. `updated_models` 는 `source_ref` 로 갱신됐고,
> `inherited_models` 는 `base_results_commit` 의 판 그대로다(재검증하지 않았다).
> `withheld` 는 디렉터리에 **없는** 모델이고 `reason` 이 그 이유다.

`verified` 5 개 == 디스크 5 개, `withheld` ∩ 디스크 = ∅ 입니다.

커밋 메시지도 고쳤습니다 -- `--only` 뒤 "(1개 모델)" 만 적으면 브랜치에 5 개가 있는데
1 개만 있는 것처럼 읽힙니다.

```
sync: main@c74846bf 기준 부분 갱신 (갱신 1개 / 상속 4개)

갱신: moonshotai__Kimi-K3
상속: 4개 @ 85e33274 (재검증하지 않았다)
```

(push 안내가 `origin results` 로 하드코딩돼 `results-plus-at` 에 올릴 때 틀린 명령을
안내하던 것도 같이 고쳤습니다.)

## 2. post-export 게이트 — 지정하신 8 항목

`_check_export()` 를 `_archive_model()` 뒤, **commit 전에** 돕니다. 실패하면 커밋하지
않습니다.

```
plus-at 이 있으면 status == released
release_blockers 가 null
one_bundle == outputs 키 집합
output 9 개 모두 존재
출고된 파일의 SHA-256 이 MANIFEST 와 일치
source bundle 과 destination bundle 이 **바이트 단위로** 일치
실제 모델 디렉터리 집합 == MANIFEST 의 verified(published + inherited) 집합
partial --only 에서 기존 모델과 최상위 MANIFEST 항목이 함께 보존됨  <- 1 번이 보장
+ git check-attr 로 plus_at 경로에 -text 가 걸렸는지                <- 아래 3 번
```

**게이트 음성 대조 7/7 발화**했습니다.

```
OK   정상 출고본           0 건 (통과해야 함)
OK   모델 집합 불일치       1 건
OK   plus_at status != released   2 건
OK   release_blockers 있음        2 건
OK   번들 파일 변조 (해시·바이트) 2 건
OK   번들 파일 삭제               2 건
OK   one_bundle != outputs        2 건
```

## 3. **이번 라운드에 제가 다시 깨뜨린 것** — `.gitattributes`

정직하게 적습니다. 지시대로 브랜치를 되돌려 clean 상태에서 다시 만들었더니 **clone 검산이
다시 7 건 실패했습니다.** 원인은 `.gitattributes` 를 제가 **손으로** 넣었고 그게
`f1486414` 에만 있었기 때문입니다 -- 되돌리자 같이 날아갔습니다.

```
19/20 통과   output 9 개 SHA 불일치 7 건 (actual_footprint, base_symbols, decode.jsonl,
             expected_footprint, expressions, prefill.jsonl, symbols.yaml)
```

두 가지를 고쳤습니다.

```
1  출고기가 .gitattributes 를 **직접 관리한다** (없거나 규칙이 빠졌으면 쓴다)
2  게이트가 `git check-attr text` 로 -text 가 걸렸는지 본다
   -- 기존 바이트 비교는 작업트리끼리라 통과한다. 깨지는 자리는 commit/checkout 의
      줄끝 변환이고 **clone 한 쪽에서만** 보인다
```

이 검사도 음성 대조로 발화를 확인했습니다.

```
OK   -text 없음 → 1 건
     "plus_at 경로에 -text 가 안 걸렸다 (text: unspecified)
      -- clone 한 쪽에서 번들 해시가 깨진다"
```

즉 **같은 결함을 두 번 냈고**(R4 의 3-b, 그리고 이번), 두 번째에 비로소 게이트로 못
박았습니다. 손으로 넣은 파일은 되돌리면 사라진다는 것이 교훈입니다.

## 4. exporter 재현성

지적하신 두 가지를 반영했습니다.

```
dirty 검사에 develop/sync_results_branch.py 추가
   -- 출고기를 고친 채로 돌리면 MANIFEST 가 커밋 안 된 코드로 만들어져 재현 불가
최종본을 **clean committed HEAD** 에서 다시 생성
   -- 브랜치를 85e33274 로 reset 하고, 출고기 수정 셋을 전부 커밋한 뒤 재실행
   -- MANIFEST 의 source_ref = c74846bf = 그 시점 HEAD
```

## 5. 최종 확인 — **실제 clone 한 사본에서 15/15**

```
git clone --branch results-plus-at --single-branch <results 저장소>

OK   최상위 verified 5 == 디스크 5
OK   withheld 에 디스크 모델 없음
OK   updated ['moonshotai__Kimi-K3'] / inherited 4 @ 85e33274
OK   plus_at status 'released' / blockers None
OK   V9 18 종 accepted
OK   output 9 개 SHA 일치
OK   one_bundle == outputs
OK   digest a13ed393e0258137
OK   expected == actual footprint (2,080 행)
OK   prefill 기준 표 == 작업 저장소 (126,822 셀)
OK   prefill 기준 + footprint == 파생본 (바뀐 셀 2,080)
OK   decode  기준 표 == 작업 저장소 (15,510 셀)
OK   decode  기준 + footprint == 파생본 (바뀐 셀 0)
OK   사이드카 1,262 레코드: 중복 0, 전부 파생본에 있음, value 일치, footprint 와 중첩 0
OK   base snapshot 23 개 == provenance, origin.sha256 일치
```

번들 도구는 손대지 않았으므로 `fixture 12/12`, `음성 대조 24/24`, `V9 16 pass / 2 n/a` 는
R3f·R4 때와 같습니다.

## 물어보는 것

### Q1. 이제 push 해도 됩니까

`results-plus-at` 을 원격에 올립니다. 1~4 처리와 clone 검산으로 충분한지 봐 주십시오.

### Q2. `inherited/not_revalidated` 를 받는 쪽이 오해하지 않겠습니까

4 개 모델은 `85e33274` 판이고 **그 사이에 작업 저장소에서 바뀐 것이 있습니다**(제가
검토받지 않은 102 파일). MANIFEST 는 "재검증하지 않았다" 고만 적습니다. "이 브랜치의 4 개는
`results` 브랜치와 같은 판이다" 를 더 명시해야 합니까, 아니면 `inherited_from` 커밋
해시로 충분합니까?

### Q3. 게이트가 아직 못 보는 자리

제가 아는 것을 적습니다.

```
출고기가 .gitattributes 를 쓰지만, **그 파일이 커밋됐는지**는 안 본다
   (같은 실행 안에서 git add -A 하므로 현재 경로에서는 문제되지 않는다)
clone 검산은 여전히 수동이다 -- check-attr 로 대체했지만 등가라고 증명하지 않았다
4 개 상속 모델의 번들은 없으므로 게이트가 볼 것이 없다
   (plus_at/ 이 없으면 건너뛴다 -- 이게 맞는 처분입니까?)
```

### Q4. 최종본에서 놓친 것

## 읽을 곳

```
develop/sync_results_branch.py   _check_export() / 부분 출고 MANIFEST / .gitattributes 관리
                                 / dirty 검사 / 커밋 메시지
../llm-arch-tracer-results/MANIFEST.json          최상위 혼합 snapshot
../llm-arch-tracer-results/.gitattributes         출고기가 쓴 것
../llm-arch-tracer-results/models/moonshotai__Kimi-K3/plus_at/   번들 11 개
develop/reviews/R4-2026-09-28-codex.md            지난 판정 원문
develop/reviews/R3f-2026-09-28-codex.md           번들 승인 원문
develop/PLAN_plus_at_layer.md                     R3f 후속 4 건 (손대지 않음)
```

산출물은 읽기만 하고 **수정하지 말아 달라.**
