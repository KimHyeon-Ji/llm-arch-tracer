# Llama-4-Maverick 최종 출고 판정 요청

지난 재검토에서 주신 5건을 전부 고치고, 요구하신 **최소 출고 게이트**도 넣었습니다.
지금 게이트를 통과하는 것은 Llama-4 하나입니다. 내보내도 되는지 판정해 주십시오.

## 1. 지적 5건 — 전부 사실이었고 고쳤습니다

### (1) B=2 프로브가 검증기가 아니었다

네 결함 모두 맞았습니다.

```
op_id 로 먼저 짝지어 op 구성이 다르면 엉뚱한 op 를 비교
   -> (module_path, raw_op, 등장순서) 로 변경
      미정합 5,000여 -> 72개(0.10%),  비교 대상 2,615 -> 36,271 자리
str(sh[ax]) == "B" 가 항상 0 (행이 심볼화 전이라 전부 정수)  -> 죽은 검사 제거
미정합 수천에도 exit 0                                      -> 비율 1% 초과면 실패
빈 축이 0 == 0*2 로 배치 의존                               -> u <= 0 제외
```

고친 뒤 결론은 같습니다 -- 축 0 이 아닌 진짜 배치 축은 **전부 MoE 의 `[E, B*T, d]`** 입니다.
MoE 는 배치가 바뀌면 전문가별 토큰 수가 달라져 op 구성이 조금 달라지므로 미정합 0 을
요구할 수 없습니다. 그래서 비율로 봅니다(1%). 이 임계값이 타당한지 봐 주십시오.

### (2) w_local / chunk_size 그룹 분리

`group` 을 `sliding_window` / `chunked_attention` 으로 나누고 둘 다 `group_key` 로 두었습니다.
"두 기제는 개념이 같다" 던 옛 주석도 지웠습니다. 결과:

```
| w_local | —  _(해당 없음: 이 모델은 `sliding_window` 계열 구조를 쓰지 않음)_ |
| chunk_size | 8192 |
```

### (3) ctx 를 공개본에 직접

```
structure.yaml  context: {"config_max_position_embeddings": 262144, "public_context_length": 1048576, "public_source": "https://github.com/meta-llama/llama-models/blob/main/models/llama4/MODEL_CARD.md", "note": "둘이 다르면 그 자체가 사실이다. config 값을 공개값에 맞추지 않는다."}
```

model_summary.md 의 Context 줄에도 두 값이 함께 나옵니다.

### (4) 낡은 검토 기록 갱신 / (5) 없는 full/trace 안내 수정

첫 항목이 아직 전역 규칙과 30칸을 말하고 있던 것을 scoped override 와 12칸으로 고쳤고,
"전체 trace 는 출고본에 없으며 main 브랜치 해당 revision 에서 확인" 으로 바꿨습니다.

## 2. accepted_limit 를 공개본에 명시했습니다

`structure.yaml` 의 `scope.known_limits` -- 모델 사실에서 유도하며 하드코딩이 아닙니다.

```
- 연속된 elementwise 연산이 한 행으로 융합된다. 예를 들어 MoE 는 `shared_out += routed_sum` 과 `residual + combined` 가 **실제로는 두 번의 add** 인데 표에는 한 행으로 나온다.
- `block_type` 은 attention 종류와 위치 인코딩을 구분하지 않는다. 이 모델의 층은 2종(chunked_attention, full_attention)이고 실제 구성은 `symbols.layer_sched` 를 봐야 한다. 같은 `attn+MoE` 로 보이는 두 블록이 서로 다른 attention 일 수 있다.
- `E_shared = 1` 는 이 모델에서 **텐서 축이 아니다** -- shared expert 모듈 수라는 구조 사실이고, 표의 어떤 축 이름도 아니다 (`num_shared_experts` 같은 config 필드가 있는 것도 아니다).
- 표에 보이지 않는 연산: RoPE. major-op 선별에서 빠진 것이지 실행되지 않은 것이 아니다.
```

## 3. 최소 출고 게이트

`release_blockers()` 가 검사하는 것:

```
검토 기록의 모든 지적이 fixed 또는 accepted_limit 인가
축 판정 사이드카 2개가 있는가            (예전에는 없으면 조용히 건너뛰었습니다)
coverage_ok 가 참 / 미해결 질문 0 / 낡은 근거 없음
워킹트리가 깨끗한가                      (게이트는 워킹트리, 출고는 커밋을 읽으므로)
```

통과 못 하면 내보내지 않고 `MANIFEST.json` 의 `withheld` 에 이유를 남깁니다. 통과분은
`verified` 에 생성 시각·라벨 입력 지문·ports 스키마 판을 적습니다.

현재 결과:

```
PASS   meta-llama__Llama-4-Maverick-17B-128E
BLOCK  나머지 13개 -- prefill.axis_resolution.jsonl 없음 (원장 도입 전 판)
BLOCK  Qwen2.5-0.5B -- 미처리 지적 1건 (가중치 축 이름)
```

지적하신 Q4(13개 격리)가 게이트로 자동 해결됐습니다. 따로 내리지 않아도 안 나갑니다.

게이트를 만들면서 제 검사가 오탐을 하나 냈습니다 -- `낡은 근거 1건 ["|0"]` 로 Llama-4 를
막았는데, 그 근거는 prefill 의 빈 캐시 concat 용이고 decode 에는 그 자리가 없습니다.
phase 별로 보던 것을 **교집합**으로 고쳤습니다(두 phase 모두에서 안 쓰여야 낡은 것).

## 4. 현재 Llama-4 상태

```json
{"kind": "summary", "occurrences": {"confirmed": 36247, "unresolved": 384}, "questions": 0, "sites": 36631, "coverage_ok": true, "evidence_entries": 6, "evidence_unused": []}
{"kind": "summary", "occurrences": {"confirmed": 36151, "unresolved": 288}, "questions": 0, "sites": 36439, "coverage_ok": true, "evidence_entries": 6, "evidence_unused": ["|0"]}
```

```
검토 기록: {'fixed': 4, 'accepted_limit': 4}
  fixed           decode bmm 축 1                   
  fixed           ctx                              
  fixed           w_local                          
  accepted_limit  E_shared                         structure.yaml#scope.known_limits
  accepted_limit  block_type                       structure.yaml#scope / model_summary.md
  accepted_limit  MoE 결합                           structure.yaml#scope / model_summary.md
  accepted_limit  표에 없는 연산                         structure.yaml#scope.table
  fixed           산출물 범위                           
```

출고 파일:

```
prefill.csv  decode.csv  prefill.jsonl  decode.jsonl  structure.yaml  model_summary.md
prefill.axis_resolution.jsonl  decode.axis_resolution.jsonl
review_findings.json  review_findings.md  review_request.md
```

## 5. model_summary.md 머리 (공개본)

```
# Model Summary -- meta-llama/Llama-4-Maverick-17B-128E

## 기본 정보

- revision: `10751cb97a4d7c90f7ed89196b98eb8220cfa1c2`
- capture backend: meta (meta/fake device, 실제 가중치 연산 없음)
- 트레이스 seq_len (T): 16
- attn_implementation: None
- 라이브러리: torch 2.13.0+cpu, transformers 5.14.1

## 요약 정보

| # | 항목 | 값 |
|---|---|---|
| 1 | SCALE | 400.71B total, 17.18B active (4.3% active)  _(active = 토큰 1개 forward가 실제로 거치는 파라미터. embedding과 lm_head 포함 — 벤더 발표치는 본체만 세는 경우가 있어 다를 수 있음)_ |
| 2 | Context (tokens) | 262,144  _(config max_position_embeddings; **공급자 공개값 1,048,576** (둘 다 사실이다 — 공개값에 맞추려고 config 값을 고치지 않는다))_ |
| 3 | DATE | 2025-04-02  _(HF repo 생성일 — 대략적 출시 시점, 정확한 발표일과 다를 수 있음)_ |
| 4 | DECODER TYPE | Sparse MoE |
| 5 | Attention | GQA |
| 6 | LAYER MIX | 36× chunked_attention, 12× full_attention  (attention: GQA)  (FFN: 24 dense + 24 MoE) |
| 7 | KV CACHE / TOKEN (BF16) | 192.0 KiB (High) |
| 8 | KEY DETAIL | GQA attention; Sparse MoE (E=128, top-1, +1 shared, sigmoid gating/aux-loss-free) |
```

---

## 질문

**Q1.** 지금 Llama-4 를 results 브랜치로 내보내도 됩니까? 남은 것이 있으면 짚어 주십시오.

**Q2.** 프로브의 미정합 임계 1% 가 타당합니까? MoE 라우팅 차이 때문에 0 은 불가능한데,
더 나은 기준이 있습니까(예: 자리 수 기준 대신 특정 op 종류만 보기)?

**Q3.** `known_limits` 의 문구가 충분합니까? 표를 처음 보는 사람이 "두 add 가 한 행" 과
"block_type 이 attention 종류를 구분 안 함" 을 이 문장으로 이해할 수 있겠습니까?

**Q4.** 게이트에 더 넣어야 할 검사가 있습니까? 지금은 ruleset digest 를 매니페스트에 적기만
하고 **대조하지는 않습니다**(비교 기준선이 없어서입니다). 어떻게 잡는 것이 맞습니까?