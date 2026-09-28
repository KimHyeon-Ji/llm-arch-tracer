---
round: R2
covers: [R2, R3]
request_scope: R2+R3
prompt: develop/codex_ask_plus_at_R2R3.md
date: 2026-09-28
reviewer: codex
note: >
  R2(반증)와 R3(전수·publish)를 한 요청으로 보냈고 한 답을 받았다. 같은 원문을 두 파일로
  복제하지 말라는 지침(R3b Q2)에 따라 **기록은 이 하나**이고 `covers` 로 범위를 밝힌다.
  release gate 는 파일명이 아니라 이 `covers` 를 읽는다.
---

# R2+R3 합본 원문 기록 — 외부 검토(Codex), 2026-09-28

**판정: R2/R3 합본은 괜찮지만 정식 publish 는 불승인.** footprint 자체는 강하다.

## 전수 확인 결과 (검토자가 독립 실행)

- 6,864 셀 footprint 가 실제 원본→파생 diff 와 정확히 일치. 중복 키·footprint 밖 변경 0
- `experts.0~2 -> n_trace_regular`, `experts.3 -> n_trace_last` 경계 오류 0
- prefill/decode 각 23 개 concat 이 `regular ×3 + last ×1`, 출력은 `B*k*T` / `B*k`
- MoE 변경 행의 caveat 보존. 입력·출력·근거 해시가 MANIFEST 와 일치
- fixture 6/6, dry-run 통과

## publish 차단 (원문 요지)

1. **expected footprint 가 독립 승인 입력이 아니다.** 적용기가 실행 시 refmatch 로 다시
   만들어 자기 결과와 비교한다. 문서의 `--expected` 인자가 parser 에 없다. digest 일치는
   "두 구현이 일치했다" 는 증거이지 "사전 승인 목록을 지켰다" 는 증거가 아니다.
2. **"V9 12 종 재실행 통과" 가 코드와 맞지 않는다.** 실제로는 metadata 보존·dependency·
   layers/repeat 정도다. `expression_no_cycle`(명시값을 먼저 환경에 넣어 순환도 통과),
   `zero_axis_only_initial_residual`(위치 미검증), `batch_seq_head_axis_consistency`,
   `head_scope_exclusive`, `moe_quotient_remainder_consistency`,
   `prefill_decode_structure`, `op_id_dag`(dict 변환으로 중복 검출 불가)는 이름만 있다.
3. **release gate 가 검토 상태를 강제하지 않는다.** substitution status 만 본다.
   V9 `review_status` 는 무조건 `proposed` 로 기록되고 `semantic_evidence_verified: false`
   / `point_verified: null` 이어도 release 가능하다.
4. **R1/R2 완료 기록이 없다.** `develop/reviews/` 가 없다. 이 세션은 제안식을 봤으므로
   blind R1 을 대신할 수 없다.
5. **`not_applicable` 3 종을 현재 사유로 승인할 수 없다.** "발행본만으로 재구성 불가" 는
   `not_evaluated` 에 가깝다. `axis_class_consistency` 는 직접 관련되므로 sidecar/crosswalk
   로 실행하거나 미검증으로. `reshape_derivation` 은 재검사하거나 불변 승계를 증명.
   `port_coverage` 는 해시 + cell-key 불변을 증명하면 `inherited_unchanged` 가 정확.
6. **재현성 메타데이터 오류.** `tool.git_commit` 이 `86353fd1` 인데 그 커밋에 도구가 없다.
   `base_results_commit` 과 `tool_source_commit` 을 분리할 것.
7. **문서 불일치.** fixture 수(16 -> 실제 14+4), `DIFF.md` 없음, `symbols.yaml` 의 kind
   구분이 CSV/JSONL 단독 배포에서 전달되지 않으므로 소비자 계약 필요.

## Q 별 결론

- Q1 `not_applicable` 분류 **불승인** -- 실행/불변승계/미검증 중 하나로 재분류
- Q2 footprint 6,864 에서 이상 없음. regular/last 경계 정확
- Q3 산출물은 내부적으로 일관. 다만 검사기가 그 일관성을 강제하지 않음
- Q4 `symbols.yaml` 을 함께 소비할 때만 구분됨. bundle 계약 필요
- Q5 **현재 publish 불승인**

## 승인 조건 다섯

1. expected footprint 를 사전 승인된 불변 입력으로 분리, 적용기가 재생성·덮어쓰기 금지
2. 실제 V9 검사를 구현하고 release 시 모든 필수 `review_status=accepted` 강제
3. R1/R2 원문과 해시를 보존하고 overlay 검증 상태를 근거에 맞게 갱신
4. 세 N/A 를 재실행·승계·미검증으로 정정
5. tool commit, fixture 개수, DIFF/bundle 계약을 바로잡고 다시 R3 요청

> "라벨 치환 결과는 상당히 잘 만들어졌지만 release 절차가 아직 그 결과를 증명하지 못합니다."

산출물은 읽기만 하고 수정하지 않았다고 명시함.
