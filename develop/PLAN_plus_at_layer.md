# +@ 층 계획 v3 — 트레이서 산출물 위의 derived view

작성 2026-09-27. v2 를 외부 검토(A/B 승인, 차단 결함 2종 추가, 계열 C 규명) 반영해 고쳤다.
**1차 출시 범위는 계열 A + B 뿐이다.** 계열 C 는 critical path 에서 뺀다.

## 0. 성격 규정

> `plus_at/` 은 **고정된 트레이서 산출물에 고정된 overlay 를 결정론적으로 적용해 만드는
> 비권위적 derived view** 다.

```
권위 있는 출처   models/<m>/{prefill,decode}.{csv,jsonl}   트레이서 결과. 불변.
derived view     plus_at/...                               원본 + overlay 로 언제든 재생성
```

`plus_at/` 을 손으로 고치지 않는다. 고칠 것은 overlay 를 고치고 다시 생성한다.
브랜치는 **`results-plus-at`** 로 분리하고 기준 `results` commit 을 manifest 에 박는다.

## 1. 남은 것 — 발행본 실측

```
                      prefill   decode        1차 출시
MoE 라우팅 토큰          2392     2392         O  (계열 B)
KDA 청크 축              2080        0         O  (계열 A)
residual 누적 2..9        631      631         X  (계열 C -- 뒤로)
빈 residual 0               1        1         X  (계열 C)
합                       5104     3024   = 8128
```

네 모델(Llama-4, gpt-oss-20b/120b, V4-Pro)은 맨정수 0. **K3 만 남았다.**

1차 출시가 덮는 것: **6,864 자리 / 8,128 (84.4%)**. 남는 1,264 자리는 residual 누적이고
**B 만 스윕하면 값이 변하지 않는다** -- 그래서 critical path 에서 뺄 수 있다.

발행본이 원시 원장과 다른 두 지점(이 차이가 +@ 층을 가능하게 한다):

```
causal prefix 는 발행본에 없다 -- n_h_kda + 맨정수 2개 이상인 자리 0 건
                                (원시 원장엔 [B,n_h_kda,5,5] 1518 + [...,d_chunk,5] 828)
전문가는 분리돼 있다 -- module_path 가 experts.0 ~ experts.3
```

## 2. 계열 A — KDA 청크 축 (2,080 자리, prefill)

```
[B, n_h_kda, 5, d_chunk, d_head_kda]  축 2 = 5   ->   n_chunk
```

`n_chunk = T / d_chunk = 320/64 = 5`. 유효 조건 `T % d_chunk == 0`.

```
근거  fla/ops/kda/naive.py:108-109   NT = T//BT, BT = chunk_size (기본 64)
      src/build_table.py:1250        _MERGE_IDENTITY_EQUIVALENTS = {"T": {"d_chunk*n_chunk"}}
```

**소속 (외부 검토 Q4): 둘 다 한다.**

```
+@ 에서 n_chunk 로 교정해 쓸 수 있게 한다
트레이서의 review_findings.json 에는 status: open 을 **유지**한다
  workaround: plus_at  +  exact footprint 를 함께 기록
트레이서가 나중에 고치면 overlay 가 before: "5" 불일치로 **자동 실패**하고 superseded 처리
```

+@ 적용은 트레이서 결함의 해결이 아니라 **공개된 workaround** 다.

negative control (외부 검토 지정):

```
must_not_match  발행본 residual 계열의 5 (160 자리)
must_not_match  op_type 이 exp 가 아닌 다른 5
```

## 3. 계열 B — MoE 라우팅 토큰 축 (4,784 자리)

shim 은 몫/나머지다(`src/kda_shim.py:641-647`: `per = max(1, n // traced)`,
`end = n if i == traced-1 else min(n, start+per)`). 마지막 전문가가 나머지를 받는다.
전문가가 `module_path` 로 분리돼 있으므로 정확히 쓴다.

```
N_route  = B*T*k (prefill) / B*k (decode)      아키텍처 유도량
C_trace  = expert_cap = 4                      kind: trace_control, origin: adaptation
q        = floor(N_route / C_trace)

experts.0 .. experts.(C_trace-2)   n_trace_regular = q
experts.(C_trace-1)                n_trace_last    = N_route - (C_trace-1)*q
상위 concat                         shape_index 0..2 -> regular,  3 -> last

valid_when  N_route >= C_trace,  C_trace == 4
```

이 트레이스는 15,360/4 = 3,840 으로 딱 나눠져 네 값이 같지만 **표는 그렇게 적지 않는다.**

caveat 보강:

```
총 expert projection FLOPs 는 보존된다 (C_trace 행의 합 = N_route).
전문가별 실제 분포·active expert 수·weight traffic·cache·latency 는 보존되지 않는다.
N_route % C_trace != 0 이면 마지막 전문가 행의 크기가 다르다.
```

## 4. 계열 C — residual 누적 (1,264 자리) — **1차에서 뺀다, 식은 규명됐다**

v2 는 "식이 발행본을 설명하지 못한다" 고 적었다. **그 판단이 틀렸다.** 내 검증이
concat 의 **입력 축과 출력 축을 한 종류로 세었다.** 외부 검토의 교정을 stage 로 나눠
다시 검증했다 -- `S=12, L=93, l` 0-based:

```
R_before(l) = ceil(l/S)                  진입 시 저장된 residual 수
R_after(l)  = floor(l/S)+1               (l % S == 0 에서 저장한 뒤)

진입 전 _apply_attn_res   buffer 입력 폭 = R_before(l),  concat 출력·후속 = R_before(l)+1
                          R_before == 0 이면 호출 생략
l % S == 0 저장           concat 입력 = [R_before(l), 1],  저장 후 R_after(l)
attention 후              buffer 입력 폭 = R_after(l),    concat 출력·후속 = R_after(l)+1
전 층 종료 후             R_final = ceil(L/S) = 8,        최종 결합 폭 = 9
```

검증 결과:

```
residual 정수 자리 1264
  layers 로 펼쳐 검사   826 자리   불일치 0
  층 정보 없는 자리     438 자리   전부 폭 8 또는 9 = R_final / R_final+1 (model root)
합계 1264 전부 설명, 불일치 0
```

폭 2 가 l=15 에 나타난 것은 다른 결합이 아니라 `R_before(15) = ceil(15/12) = 2`,
즉 **concat 의 입력 buffer 축**이다.

**주의**: 접힌 행은 `layer_idx` 대표값이 아니라 `layers` 의 **모든 층**에서 식이 같은 값을
내는지 검사해야 한다(`repeat: 3 / layers: "15,19,23" / layer_idx: 15`).

처리: **1차에서는 손대지 않는다.** B 만 스윕하면 이 값은 변하지 않는다. 2차에서 셀별 stage
를 규명해 `verified` 로 `expressions.yaml` 에 싣는다. **불완전한 식을 unverified 로 싣지
않는다.**

## 5. 산출물 구조

```
models/<m>/
  prefill.csv  decode.csv  prefill.jsonl  decode.jsonl      트레이서 원본 (불변)
  plus_at/
    overlay.yaml                판정 = 항목 하나. 근거·exemplar·검증상태·검토이력
    expected_footprint.jsonl    **적용 전** 생성·독립검토·accepted 된 변경 허용 목록  [입력]
    ---- 아래는 생성물 ----
    prefill.csv  decode.csv  prefill.jsonl  decode.jsonl     derived view
    actual_footprint.jsonl      실제 적용 결과                                      [출력]
    symbols.yaml                심볼 범례. kind 로 architecture / trace_control 구분
    MANIFEST.json               입력·도구·출력·source 해시 전부
    DIFF.md                     사람이 보는 요약
    (2차) expressions.yaml      계열 C
```

### expected / actual footprint 분리 (차단 결함 1 교정)

v2 는 `footprint.jsonl` 하나였다. 적용기가 그것을 만들고 자기 결과와 비교하면 **검사가
순환한다.** 그래서 나눈다.

```
expected_footprint.jsonl   overlay 입력의 일부. 적용 전에 만들고 독립 검토를 받는다.
actual_footprint.jsonl     derived output 의 일부. 적용 결과에서 나온다.

검사   actual_cell_set == expected_cell_set
       digest(actual)  == digest(expected)
```

canonical cell key:

```json
{"phase":"prefill","op_id":55246,"field":"input_shape","shape_index":0,"axis":2,
 "before":"5","after":"n_chunk","module_path":"model.layers.3.self_attn","op_type":"exp",
 "sub_id":"k3-kda-nchunk"}
```

**공통 버그 차단**: expected 를 만든 selector 와 적용기가 같은 구현을 공유하면 같은 버그가
양쪽에 난다. 그래서 `develop/plus_at_refmatch.py` 를 **독립 reference matcher** 로 따로 쓰고,
`develop/fixtures/plus_at/` 에 golden fixture 를 둔다. 세 결과가 모두 같아야 통과한다.

### MANIFEST.json — 변조 방지

해시를 overlay 안에만 적으면 YAML 과 해시를 함께 고치면 끝이다. 분리하고 git commit 으로 고정한다.

```json
{"schema_version": 1,
 "base_results_commit": "<results 브랜치 commit>",
 "inputs":  {"models/<m>/prefill.csv":"<sha256>", "...jsonl":"...",
             "models/<m>/full/provenance.json":"<sha256>"},
 "overlay": {"overlay.yaml":"<sha256>", "expected_footprint.jsonl":"<sha256>"},
 "tool":    {"path":"develop/plus_at_apply.py","sha256":"...","git_commit":"...",
             "refmatch_sha256":"..."},
 "outputs": {"plus_at/prefill.csv":"<sha256>", "actual_footprint.jsonl":"...", "...":"..."},
 "sources": [{"file":"fla/ops/kda/naive.py","sha256":"...","lines":"108-109"},
             {"file":"src/kda_shim.py","sha256":"...","lines":"641-647"}],
 "v9": {"<gate>": "rerun|inherited_unchanged|not_applicable", "...": "..."}}
```

source SHA-256 은 **1차 필수**다 -- 계산 비용이 거의 없고 재현성의 핵심이다.

### 생성 규약

```
결정론  같은 입력 + 같은 overlay -> 같은 출력 해시
멱등    재적용 시 출력 해시가 동일. 다르면 실패
원자    임시 디렉터리에서 전부 만들고 검증 통과 시에만 교체
수동편집 금지  plus_at/ 의 해시가 MANIFEST 와 다르면 도구가 거부
```

## 6. 검증

### V1 은 **canonical cell 비교**다 (차단 결함 2 교정)

v2 는 "바이트 동일" 이라고 썼다. csv/jsonl 을 다시 serialize 하면 줄바꿈·공백·인용이
바뀔 수 있어 그 정의로는 통과할 수 없다.

```
V1  비대상 canonical cell 전부에 대해  original_value == derived_value
    (파일 바이트 비교가 아니다)
    파일 단위 결정론은 출력 SHA-256 + 멱등 검사로 따로 본다
```

### V2 ~ V8

```
V2 집합 동일성  actual_cell_set == expected_cell_set  그리고 digest 일치
V3 sanity       한 shape 에 같은 맨정수가 둘 이상이면 **전체 적용 실패**
                (v1 의 "자동 제외 후 계속" 은 폐기)
V4 형태 보존    행·열·열이름·op_id 순서 동일, shape 의 축 개수 동일
V5 canonical    csv 와 jsonl 을 **같은 파서로 shape 배열화해** 비교 (문자열 비교 금지)
V6 점 대입      심볼에 symbols.yaml 값을 대입하면 원본 구체값이 복원된다
                n_chunk=5 -> 5,  q = floor(15360/4) -> 3840,
                n_trace_last = 15360 - 3*3840 -> 3840
                **생성기와 evaluator 를 다른 구현으로** 쓴다(또는 golden fixture)
V7 잔여 보고    바꾸지 않고 남긴 맨정수를 전수 목록화 (계열 C 1,264 자리 포함) -- 숨기지 않는다
V8 의미 맥락    바뀐 셀의 (module_path, op_type) 집합 == overlay 선언 집합
                + must_match / must_not_match exemplar 가 각각 맞고 틀려야 한다
                + axis/field/shape_index/module 을 하나씩 변형한 mutation test 가 **실패**해야 한다
```

### V9 — 게이트 승계 표 (외부 검토 지정 형식)

게이트마다 표를 만든다: `gate / derived table 적용 가능 / 재실행 / 원본 결과 승계 / 사유`.
label 변경에 민감한 다음 항목은 **1차부터 재실행**한다.

```
schema, shape rank, token type
미등록·미정의 symbol 금지
expression dependency cycle 금지
대입 결과가 정수이며 음수가 아님
`0` 은 선언된 초기 residual 두 자리에서만 허용
배치·시퀀스·head 축 일관성
n_h / n_kv / n_h_kda scope 배타성
MoE token 축과 concat 입력의 quotient/remainder 일관성
prefill / decode 간 공통 구조 일관성
layers 파싱 후 repeat == expanded layer count
caveat · unmapped · params · depends_on 보존
op_id 유일성, dependency 존재, DAG 비순환
```

V1/V4 때문에 절대 바뀌지 않는 검사는 원본 PASS 를 승계해도 되지만 MANIFEST 에
`inherited_unchanged` 로 기록한다.

### 검증 범위 표기 (항목마다)

```yaml
verification:
  point_verified: true
  verified_assignments: {B: 3, T: 320, k: 16, d_chunk: 64, C_trace: 4}
  semantic_evidence_verified: true
  sweep_verified: false
  validity_domain: ["T % d_chunk == 0", "N_route >= C_trace"]
  unverified_dimensions: [T, k, d_chunk, C_trace]
```

V6 은 필요하지만 충분하지 않다 -- 틀린 식도 한 지점에서 우연히 맞을 수 있다. 메타모픽
(`T=384` 재트레이스)은 `sweep_verified: false` 로 두고 절차만 적는다.

## 7. 검토 — 독립성을 설계로 확보

```
R1 도출        현재 라벨과 후보를 **가리고** site context + source 만 준다.
               "이 축의 의미와 식을 직접 도출하라."  후보를 주지 않는다.
               불가피하면 cannot determine 을 포함하고 순서를 섞는다.
               **검토자가 selector 결과를 보지 않고 expected footprint 를 독립 도출한다.**
R2 반증        falsifies + risk + validity_domain + negative control
R3 산출물      표본이 아니라 **전수** -- expected footprint 전량과 V1~V9 결과를 확인시킨다
```

```
기록  모델명·버전·prompt sha256·source sha256
      원문 답변은 develop/reviews/<round>-<date>-<model>.md 로 보존하고 해시 기록
      overlay.yaml 에는 사람이 옮긴 요약 + 원문 파일 참조만
      판정이 바뀌면 지우지 않고 superseded_by 로 잇는다
상태  proposed -> reviewed -> accepted.  accepted 만 적용한다
```

## 8. 1차 / 2차 분리와 시간

### 1차 (A + B, 6,864 자리)

```
derived view 경계 + results-plus-at 브랜치
MANIFEST (base commit, 입력·provenance·overlay·tool·출력·source sha256)
expected / actual footprint 분리 + 집합 동일성 + digest
canonical V1 / V4 / V5
계열 B quotient/remainder + validity domain
atomic · deterministic · idempotent
label 에 민감한 V9 재실행
계열 A 의 tracer finding 을 status: open 으로 유지 (workaround 기록)
A/B 에 대한 최소 1회 독립 의미·footprint 검토
```

시간:

```
1  plus_at_apply.py + refmatch + V1~V9            60분
2  overlay.yaml (A, B) + expected_footprint 생성   35분
3  MANIFEST + 결정론·멱등·원자 확인                20분
4  V9 승계 표 작성                                 20분
5  DIFF.md + R1~R3 프롬프트                        25분
   1차 합계                                       약 2시간 40분, 재트레이스 0
```

### 2차 (뒤로)

```
계열 C 셀별 stage mapping + expressions.yaml (식은 이미 검증됨, 매핑만 남음)
T=384 메타모픽 재트레이스
R2/R3 추가 모델·사람 검토
polished REVIEW.md / 프롬프트 패키지
별도 aggregate roofline view (N_route 집계)
label 무관하며 V1/V4 로 불변 증명된 게이트의 완전 재실행
```

## 9. 트레이서 쪽 — 이번에 끝내고 더 손대지 않는다

```
완료  V4-Pro / Llama-4 / gpt-oss-20b / gpt-oss-120b  재트레이스 + promote, 지문 OK
완료  K3 의 d_nope -> d_v 32 자리 (발행본 csv 반영)
남음  K3 지문 STALE -- 규칙 델타가 V4-Pro 범위여서 K3 라벨에 영향 없음을
      frozen trace 에 두 규칙판을 적용해 동일 출력으로 증명 (재트레이스 없이)
남음  V4-Pro ANCHOR MODE 교정 3건이 applied 0 으로 잡힌다 -- from == to 는 0 이 정상.
      게이트 판정 기준 쪽 문제로 기록
남음  baseline.json 의 K3 bare 갱신 + 사유 (새 교정 2건 빼도 430,249 동일 -- 기준선이 낡음)
남음  results-plus-at 브랜치 생성, base results commit 기록
```

## 10. 열린 결정

```
D1  V9 승계 표에서 "not_applicable" 로 뺄 게이트를 내가 정해도 되는지
D2  1차에 A 와 B 를 함께 낼지, A 만 먼저 낼지 (A 는 2,080 자리 / 1 패턴이라 더 단순)
D3  plus_at/ 을 results-plus-at 에 둘 때 원본도 함께 둘지(대조 편의) 참조만 둘지
```

---

## R3f 승인 뒤 후속 (비차단) — 2026-09-28

외부 검토(R3f)가 **승인과 함께** 남긴 hardening 넷. 이번 출고에서는 **손대지 않았다** --
승인 근거에 "도구 해시 전부 일치" 가 들어 있어, 승인 뒤 도구를 고치면 그 근거가 깨진다.

1. `base_symbols.json.origin` 에 `git_commit: d8fec245…` 를 넣어 MANIFEST 의
   `base_results_commit` 과 연결 관계를 명시한다.
2. `_bad_slots()` 는 **concrete row 의 범위만** 본다. raw row 도 같은 slot 을 실제로
   갖는지 검사하고, weight 는 `shape_index == 0` 을 강제한다.
3. `_rank_mismatch()` 는 양쪽이 모두 list 일 때만 rank 를 비교한다. **한쪽만 list 인 타입
   불일치**도 거부한다.
4. **작은 synthetic model/crosswalk 으로 게이트 단위 통합 테스트**를 짠다. 지금 rank 검사는
   술어 단위 대조만 있다(657 MB 사이드카 복제를 피하려고). synthetic 이면 전체 사이드카
   없이 `g_reshape_derivation()` 이 실제로 `not_evaluated` 를 반환하는 호출 경로까지 본다.

문서 구분도 하나: **"bundle 9 개" 는 계약상 output 9 개**이고, 실제 `plus_at/` 디렉터리는
`MANIFEST.json` 과 `DIFF.md` 를 포함해 **11 개 파일**이다.

그리고 R3f 승인의 범위를 잊지 말 것 -- **이 입력과 footprint digest `a13ed393…` 에
한정**이다. 다른 모델이나 다른 입력으로 이 도구를 돌리면 승인은 승계되지 않는다.

## R4b 승인 뒤 후속 (비차단) — 2026-09-28

`results-plus-at` push 완료(`2e22cdc5`). 외부 검토(R4b)가 승인과 함께 남긴 것.

1. `_check_export()` 는 commit 전 **작업트리**를 본다. `git add -A` 뒤 **index blob 도 다시
   해시**하면 수동 clone 검산과 거의 같은 경로를 자동으로 덮는다.
2. **destination 이 실행 시작 시 dirty 하면** 상속 모델의 기존 변경을 함께 커밋할 수 있다.
   시작 시 destination clean 을 요구하거나, inherited 모델의 tree OID 가 실행 전 HEAD 와
   같은지 commit 전에 확인한다.
3. 기계 계약의 커밋 해시를 **40 자리 전체**로 (`base_results_commit`, `inherited_from` 이
   지금 8 자리다).
4. 브랜치 설명에 "K3 만 갱신, 나머지 4 개는 `results@85e33274` 와 바이트 동일" 한 줄.

그리고 R3f 후속 4 건(위 절)은 여전히 남아 있다.
