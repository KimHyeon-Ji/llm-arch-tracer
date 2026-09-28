# R3c — 차단 5 건을 고쳤습니다. 공개 승인을 청합니다

요청일 2026-09-28. 선행: R1(blind), R2+R3 합본, R3b.
지난 판정: *"reshape crosswalk 교정, axis fail-closed, DIFF 및 bundle contract 수정,
residual 분류/fixture 보강, sidecar phase 게이트 추가 후 R3c 결과가 깨끗하면 공개 승인할
수 있습니다."*

```
status               provisional
base_results_commit  d8fec245671e
tool_source_commit   109d1783
footprint digest     a13ed393e02581378837e5f9be91646105b63d3d6140002e94528106d6ebf44a
본표 바뀐 셀         2,080   (계열 A)
사이드카 셀          1,262   (계열 C, expressions.yaml)
V9                   14 종 실행 전부 pass / 2 종 n/a / **미평가 0**
fixture              12/12
입력 pin             15 개 (crosswalk·raw·concrete·ports·semantic 포함)
남은 차단            18 건 -- **전부 review_status**
```

---

## 차단 5 건 처리

### 1. residual 분류가 direct consumer 에 의존했다 -> 투명 op 추적 + cardinality 강화

```
TRANSPARENT_OPS = view, _unsafe_view, reshape, clone, contiguous, _to_copy, detach,
                  alias, squeeze, unsqueeze, expand, permute, transpose, slice, copy_
```

`_reaches_mix()` 가 이것들을 **건너뛰며** 혼합 체인에 닿는지 봅니다. 그룹 구성원 수집도
투명 op 을 통과시키되 구성원으로는 세지 않습니다.

cardinality 를 지적대로 강화했습니다.

```
층 0            pre 0, post 1
그 밖           pre 1, post 1
boundary_append **l % R_res == 0 인 층에만**            <- 새로 넣음
층 누락         `layers` 를 펼친 합집합이 0..max 를 덮어야 한다   <- 새로 넣음
final           정확히 1
norm 가중치가 없거나 둘 이상이면 ValueError
검사 순서       층별 검사를 전역 final 검사보다 **먼저** (구체적 진단이 먼저 나오게)
```

**adversarial fixture 4 종**을 추가했습니다(사람이 직접 작성). 뒤 셋은 **실패해야** 통과입니다.

```
투명 op(view)이 낀 mix 그룹도 mix 로 잡힌다 / 투명 op 은 구성원이 아니다
한 층에 mix 그룹이 셋이면 cardinality 가 실패시킨다
경계 아닌 층(R_res=4 에서 층 5)의 append 는 실패한다
norm 가중치가 없으면 추측하지 않고 실패한다
```

fixture 는 실제 모델과 다른 `R_res=4, L=9` 를 씁니다 -- 식이 값에 기대지 않는지 봅니다.

### 2. axis 게이트 fail-open -> fail-closed

결과를 셋으로 갈랐습니다.

```
pass      검사가 돌고 통과
FAIL      검사가 돌고 실패 -> 적용 중단
n/a       **overlay 가 그 대상을 선언하지 않았다** -> 막지 않는다
not_run   **필수 증거가 없어 평가하지 못했다** -> release 를 막는다   <- 새로 구분
```

crosswalk 없음 / crosswalk stale / 원시 사이드카 없음 은 모두 `not_evaluated` -> `not_run`
입니다. N/A 로 빠지지 않습니다.

### 3. reshape_derivation 의 op_id 번호 공간 -> crosswalk 경유 (가장 중요한 지적)

axis 게이트만 고치고 이쪽을 놓쳤습니다. 지적이 정확했습니다.

지금은 crosswalk 로 발행본 셀 -> raw 자리를 얻어, **바뀐 셀이 걸린 raw op 만** 골라 그 op 의
reshape 유도를 원본 라벨과 파생 라벨 각각으로 검사합니다.

```
prefill  건드린 raw op 4,485   이견 0 -> 0
decode   건드린 raw op     0   이견 0 -> 0   (계열 A 가 prefill 전용)
```

crosswalk 이 없거나 낡으면 `not_run` 입니다.

### 4. DIFF 설명 -> 값별 표 (앞 라운드에서 실제로 반영되지 않았던 것)

앞 라운드에 "고쳤다" 고 보고했지만 scratchpad 사본만 고쳤고 `develop/plus_at_diff.py` 의
하드코딩 문장은 그대로였습니다. 지금은 이렇게 나옵니다.

```
| 값     | 자리 | 왜 리터럴인가 |
| 12     | 2392 | MoE 라우팅 토큰 (decode) |
| 3840   | 2392 | MoE 라우팅 토큰 (prefill). shim 이 전문가 4 개로 균등분할한 대체값이고
                  실제 per-expert 축은 결정 불가 -- moe_aggregate 참조 |
| 2..9   |  각 160 등 | residual 누적 폭. 사이드카에 식이 있다 (expressions.yaml) |
| 0      |    2 | 초기 빈 residual 버퍼. 리터럴이 맞다 |
```

### 5. bundle 계약 -> 현재 산출물에서 생성

철회된 심볼 이름을 하드코딩하지 않습니다. 지적하신 항목을 전부 넣었습니다.

```
one_bundle              현재 출력 파일 목록 (expressions.yaml 포함)
inseparable             표는 symbols.yaml 과 분리 불가
architecture_symbols    []            <- n_chunk 가 trace_artifact 이므로 비었다
non_architecture_symbols[n_chunk]
sidecar                 expressions.yaml 도 같은 bundle. 없는 소비자는 숫자 표만 쓸 수 있고
                        residual recurrence 의미는 복원할 수 없다
sidecar_join_key        [phase, op_id, field, shape_index, axis]
canonical_cell_rule     op_id 는 **발행본 표의 번호**다. 원시 원장과 다른 번호 공간이므로
                        원시와 잇는 데는 crosswalk 이 필요하다
reject_unknown_symbol   symbols.yaml 에 없는 심볼을 만나면 거부
phase_consistency       prefill/decode 사이드카 레코드 수와 식 분포가 같아야 한다
caveat_stays            MoE 행의 caveat 은 그대로. 총 FLOPs 만 보존
```

### 추가 지적 반영

```
crosswalk 선택을 정렬해 **결정론**으로 (glob 첫 결과 의존 제거)
crosswalk **중복 키 거부** (조용히 마지막 것을 쓰면 어느 판을 읽었는지 모른다)
**빈 raw_sites 거부** -- 그 셀은 stale 로 보고 not_run
MANIFEST 입력에 crosswalk·trace.raw·shapes.concrete·ports·semantic 해시 pin (총 15 개)
```

### Q3 반영 -- sidecar phase 게이트 승격

```
pass  rerun  sidecar_phase_consistency  phase 별 631 레코드, 식 분포 동일
                                        {c_post 288, c_pre 276, b_after 28, b_in 26,
                                         b_final 1, c_final 12}
```

### Q2 반영 -- 검토 기록

복제하지 않았습니다. `covers` 필드를 게이트가 읽습니다.

```
R1-2026-09-27-codex-blind.md    covers [R1]
R2-2026-09-28-codex-merged.md   covers [R2, R3]   request_scope: R2+R3 명시
R3-2026-09-27-codex.md          (1 차. covers 없어 파일명 fallback)
R3b-2026-09-28-codex.md         covers [R3b]
```

`review_state()` 가 frontmatter 의 `covers` 를 읽고, 없으면 파일명 접두사로 fallback 합니다.

## V9 전체 (14 실행 / 2 n/a / 미평가 0)

```
pass  rerun                axis_class_consistency       건드린 class 69, 이름 충돌 0 (crosswalk 경유)
pass  rerun                reshape_derivation           raw op 4,485, 이견 0 -> 0 (crosswalk 경유)
pass  rerun                sidecar_phase_consistency    phase 별 631, 식 분포 동일
pass  rerun                schema_shape_rank_token_type
pass  rerun                symbol_declared              선언 1 개 + 허용 함수 ceil
pass  rerun                expression_no_cycle
pass  rerun                substitution_nonneg_integer
pass  rerun                zero_axis_only_initial_residual   `0` 축 2 자리 전부 선언된 자리
pass  rerun                batch_seq_head_axis_consistency
pass  rerun                head_scope_exclusive
pass  rerun                layers_repeat_consistency
pass  rerun                row_metadata_preserved
pass  rerun                op_id_dag
pass  inherited_unchanged  port_coverage                사이드카 해시 + cell 키 불변
n/a   not_applicable       moe_quotient_remainder_consistency  (MoE 철회 -- 지적대로 타당)
n/a   not_applicable       prefill_decode_structure     (본표 판정이 prefill 전용.
                                                        bundle 쪽은 sidecar 게이트가 본다)
```

## 남은 차단 18 건 — 전부 `review_status`

```
k3-kda-nchunk: status 'proposed'
k3-kda-nchunk: semantic_evidence_verified 가 참이 아니다
V9 게이트 16 종: review_status 'proposed'
```

R2 기록 차단은 `covers` 인식으로 사라졌습니다. **기계적 조건은 닫혔습니다.**

## 물어보는 것

### Q1. 공개 승인하십니까

승인하시면 다음을 올립니다.

```
overlay 의 k3-kda-nchunk status         proposed -> accepted
verification.semantic_evidence_verified false -> true  (R1 blind 도출 + R2/R3b 반증을 근거로)
develop/plus_at/v9_review.yaml 의 16 게이트  proposed -> accepted
MANIFEST status                         provisional -> released
develop/reviews/ 에 이 판정 원문 보존 (covers [R3c])
results-plus-at 브랜치 생성 -- base results commit d8fec245 기록
```

### Q2. `architecture_symbols` 가 빈 것이 맞습니까

`n_chunk` 를 `trace_artifact` 로 정정했으므로 활성 아키텍처 심볼이 **하나도 없습니다**.
표의 다른 심볼(`B`, `T`, `d_model`, …)은 트레이서가 낸 것이고 +@ 가 선언한 것이 아닙니다.
bundle 계약이 "+@ 가 새로 넣은 심볼" 만 나열하는 것이 맞습니까, 아니면 트레이서 심볼표까지
포함해 소비자에게 전체 목록을 줘야 합니까?

### Q3. `not_run` 과 `n/a` 의 경계가 이제 맞습니까

```
n/a       overlay 가 그 대상을 선언하지 않았다 (MoE 철회, 단일 phase)
not_run   필수 증거가 없어 평가하지 못했다 (crosswalk 없음/stale, 사이드카 없음)
FAIL      "선언했는데 검사 대상이 0 건" 은 여전히 FAIL
```

### Q4. 놓친 것

특히 **op_id 번호 공간 혼용**이 또 남아 있는지 봐 주십시오. 두 게이트는 crosswalk 로
고쳤지만 다른 자리에 같은 착오가 있을 수 있습니다. `port_coverage` 의 `inherited_unchanged`
근거(사이드카 해시 + cell 키 불변)도 그 관점에서 봐 주시면 좋겠습니다 -- cell 키는 발행본
공간이고 포트는 원시 공간입니다.

## 읽을 곳

```
develop/plus_at_resid.py          _reaches_mix(), TRANSPARENT_OPS, cardinality
develop/plus_at_v9.py             _crosswalk(), g_axis_class_consistency(),
                                  g_reshape_derivation(), g_sidecar_phase_consistency(), run()
develop/plus_at_apply.py          review_state()의 covers, bundle_contract, 입력 pin
develop/plus_at_diff.py           값별 리터럴 설명
develop/fixtures/plus_at/cases.yaml   residual.adversarial 4 종
develop/test_plus_at.py           12/12
develop/plus_at/overlay-moonshotai__Kimi-K3.yaml
develop/plus_at/expected-moonshotai__Kimi-K3.jsonl   2,080 줄
develop/plus_at/v9_review.yaml
develop/reviews/                  R1 / R2(covers R2,R3) / R3 / R3b
models/moonshotai__Kimi-K3/plus_at/   MANIFEST·DIFF·expressions·footprint·symbols
```

산출물은 읽기만 하고 **수정하지 말아 달라.**
