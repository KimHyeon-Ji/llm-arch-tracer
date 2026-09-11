# Llama-4-Maverick 재검토 요청 — 출고 가능 판정 부탁드립니다

지난 검토의 **출고 전 필수 7항목** 중 5개를 처리했습니다. 2개는 아직 안 했고 이유도 적었습니다.
지금 상태로 출고해도 되는지 판정해 주십시오.

## (1) 전역 "배치는 축 0 에만" 규칙 — 되돌리고 B=2 프로브로 검증한 뒤 좁게 걸었습니다

지적이 정확했습니다. 제 반례 검사가 **반증 불가능한 구조**였습니다 -- `symbolic_shape.py` 가
`if n == 1: return _r("runtime", "B")` 이라 `B` 는 크기 1 축에만 붙습니다. "값이 1 이 아닌 B 가
없다" 는 나올 수 없는 관찰이었습니다. 전역 규칙을 물렸습니다.

`src/inputs.py` 에 `batch` 인자를 열고(기본 1 = 발행 경로 무변화) `develop/batch_axis_probe.py`
로 같은 모델을 B=1 / B=2 로 트레이스했습니다.

```
decode op130   B=1 [40, 1, 128]  ->  B=2 [80, 1, 128]
    축 0 만 2배(= B*n_h), 축 1 은 1 그대로   -> 축 1 은 배치가 아니다

MoE            [E, B*T, d_moe] 의 축 1 은 배치를 따라 2배가 된다
    -> 축 1 이 배치 의존일 수 있다. 전역 규칙이면 이쪽을 `1` 로 오염시킨다
```

그래서 `rules/label_overrides.yaml` 에 **소스로 확인된 자리에만** 걸었습니다
(self_attn 의 batched_matmul, `B` -> `1`, 192축 발화).

프로브도 한 번 틀렸습니다 -- `run_once()` 가 정규화 전 행을 돌려줘 `op_type` 이 없는데 그걸로
정합성을 검사해 가드가 죽었고, 서로 다른 op 를 비교해 T 축이 배치 의존으로 나왔습니다.
`raw_op` + `module_path` 정렬로 고치고 안 맞는 행을 세도록 했습니다.

## (2) ctx 분리 — references 로 교차검증

`ctx` 가 config 값임을 심볼 정의에 명시하고, 공개값은 `references.yaml` 의
`context_length_public` 에 등록했습니다(config 262,144 / public 1,048,576, 출처 Meta 모델 카드).
맞추려고 `ctx` 를 고치지는 않았습니다.

## (3) w_local -> chunk_size — 심볼을 갈랐습니다

`w_local` 의 별칭에 `sliding_window` 와 `attention_chunk_size` 가 함께 있었습니다. 둘은 다른
기제라 갈랐습니다. Llama-4 만 후자를 쓰고 나머지 7개는 전부 전자라 영향이 없었습니다.
결과: `w_local: null`, `chunk_size: 8192`.

이 분리로 근거 항목 하나가 낡았는데 원장의 `evidence_unused` 가 `["d_moe|w_local|d_moe"]` 로
정확히 짚었습니다 -- 낡음 감지가 실제로 작동한 첫 사례입니다.

## (5) text-only 범위 — structure.yaml 의 scope 필드

```yaml
  traced: text-only forward — input_ids 만 넣는다. pixel_values / vision tower 없음
  batch: 1
  prefill_len: 16
  decode_cache_len: 16
  decode_query_len: 1
  table: major-op 요약이다. 전체 ATen op 목록이 아니다 (전체는 full/<phase>.trace.raw.jsonl).
```

## (7) 축 판정 사이드카 출고

`full/` 은 계속 버리되 `*.axis_resolution.jsonl` 만 건져 올립니다. 출고 파일 목록:

```
prefill.csv  decode.csv  prefill.jsonl  decode.jsonl  structure.yaml  model_summary.md
prefill.axis_resolution.jsonl  decode.axis_resolution.jsonl
review_findings.json/.md  review_request.md
```

사이드카 요약 줄:

```json
{"kind": "summary", "occurrences": {"confirmed": 36247, "unresolved": 384}, "questions": 0, "sites": 36631, "coverage_ok": true, "evidence_entries": 6, "evidence_unused": []}
```

## (4) block_type 구분 / (6) MoE add 분리 — 아직 안 했습니다

둘 다 표 생성 로직(레이어 접기 / major-op 융합)이라 다른 모델에도 파급이 갑니다.
지금 목표가 이 4개 모델을 하나씩 닫는 것이라 먼저 판정을 받고 싶습니다(Q2).

## 표에서 바뀐 것

지난 검토본 대비 `decode.csv` **12칸**, `prefill.csv` 무변화.

```
op5 input_shape:  [[n_h, B, d_head], [n_h, d_head, T+1]]  ->  [[n_h, 1, d_head], [n_h, d_head, T+1]]
op5 output_shape:  [[n_h, B, T+1]]  ->  [[n_h, 1, T+1]]
op7 input_shape:  [[n_h, B, T+1], [n_h, T+1, d_head]]  ->  [[n_h, 1, T+1], [n_h, T+1, d_head]]
op7 output_shape:  [[n_h, B, d_head]]  ->  [[n_h, 1, d_head]]
op21 input_shape:  [[n_h, B, d_head], [n_h, d_head, T+1]]  ->  [[n_h, 1, d_head], [n_h, d_head, T+1]]
op21 output_shape:  [[n_h, B, T+1]]  ->  [[n_h, 1, T+1]]
... 총 12칸
```

## 검토 기록 상태

```
fixed  decode bmm 축 1                   should_be_renamed
fixed  ctx                              should_be_renamed
fixed  w_local                          should_be_renamed
open   E_shared                         undetermined
open   block_type                       should_be_renamed
open   MoE 결합                           should_be_renamed
open   표에 없는 연산                         undetermined
fixed  산출물 범위                           undetermined
```

---

## 질문

**Q1 (출고 가능).** 위 5개를 처리한 지금 상태로 results 브랜치에 내보내도 됩니까?

**Q2 ((4)(6) 이 출고 필수입니까).** `review_findings.json` 에 `status: open` 으로 기록하고
`model_summary.md` 에 "major-op 요약이며 두 add 가 융합돼 있다" 를 명시하는 것으로 갈음해도
됩니까, 아니면 출고 전에 반드시 고쳐야 합니까?

**Q3 (출고 게이트).** 제안하신 `severity` / `affects_published_artifact` 필드는 아직 안
넣었습니다. 지금 `sync_results_branch.py` 는 "판단 필요 0건" 과 ledger 존재만 봅니다. 이
모델 하나를 내보내는 데에도 게이트를 먼저 고쳐야 합니까, 아니면 이번은 수동 판정으로
내보내고 게이트는 다음에 고쳐도 됩니까?

**Q4 (다른 13개).** `--only` 로 고친 모델만 내보낼 수 있게 했습니다. 나머지 13개는 아직 옛
판(배치 축 오류 포함)입니다. 이미 results 에 나가 있는 그것들을 **내리는 편**이 낫습니까,
아니면 두고 하나씩 갱신해도 됩니까?