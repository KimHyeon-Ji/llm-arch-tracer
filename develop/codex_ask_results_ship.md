# results 브랜치로 보내도 됩니까 — 판단 부탁드립니다

지적하신 decode 의 가짜 `B` 축을 고쳤습니다. 그런데 고치고 보니 **혼자만의 문제가
아니었습니다.** 지금 내보내도 되는지 판단을 받고 싶습니다.

## 1. 고친 것

```
decode  [n_h, B, d_head] × [n_h, d_head, T+1]     구체 [40, 1, 128]
    ->  [n_h, 1, d_head] × [n_h, d_head, T+1]
```

옛 규칙은 "앞에서 이미 B 나 T 를 봤으면" 일 때만 껐습니다. 그래서 앞이 head 수인
`[n_h, B, d_head]` 를 못 잡았습니다. HF 레이아웃은 배치를 항상 맨 앞에 두므로, 축 0 이 B 가
아닌 텐서는 배치가 접혔거나(bmm 의 `[B*n_h,T,d]`) 없는 것이고, 그 뒤의 크기 1 축은 배치가
아닙니다. `symbolic_shape` 의 불변식을 **"배치는 맨 앞에만"** 으로 강화했습니다.

규칙으로 올리기 전에 함대를 셌습니다:

```
축 0 이 아닌 자리의 `B`   294,408개 / 55 모델·phase
구체값이 1 이 아닌 것      **0개**          -- 반례 없음
```

위험해 보이던 xLSTM(7,168개)도 `[T,B,d]` 레이아웃이 아니라 mLSTM 의 per-head bmm 이었습니다
(`[n_h, B, d_head]` = `[8, 1, 512]`, 그 1 은 청크 위치).

Llama-4 결과: decode.csv **30칸**이 `B` -> `1`, prefill 무변화, 게이트 FAIL 0.

## 2. 그런데 이게 Llama-4 만의 문제가 아닙니다

`develop/sync_results_branch.py` 가 내보내는 **14개 모델 전부**가 같은 결함을 갖고 있습니다
(발행된 csv 기준):

```
32 자리  Llama-4-Maverick        8 자리  SmolLM3-3B / ERNIE-4.5 / gemma-2-2b / gemma-3-270m / GLM-4.5-Air
 4 자리  LFM2-8B / Qwen2.5-0.5B / Qwen3-30B-A3B / Llama-3.1-70B / Llama-3.1-8B /
         Phi-4 / Mistral-Small-3.2 / falcon-7b
```

Llama-4 는 이미 그 14개 안에 있습니다. 즉 **지금 results 브랜치는 이 오류가 있는 판을
싣고 있고**, Llama-4 하나만 재트레이스해 올리면 나머지 13개는 틀린 채로 남습니다.

## 3. 나머지 지적은 "열린 항목" 으로 기록했습니다

`models/meta-llama__Llama-4-Maverick-17B-128E/review_findings.json` 에 8건(fixed 1 / open 7):

```
open  ctx          config 262,144 vs 공식 공개 1M -- 두 값 분리 필요
open  w_local      sliding window 가 아니라 고정 청크 -> chunk_size 가 맞음
open  E_shared     config 필드가 아니라 구조 사실 -> 메타데이터로
open  block_type   chunked/full 과 RoPE/NoPE 차이가 표에 안 드러남 (접기 자체는 옳음)
open  MoE 결합     shared+routed add 와 residual add 를 한 행으로 뭉갬
open  빠진 연산    마스크 생성, RoPE, temperature tuning, GQA repeat, topk/scatter, cache update
open  산출물 범위  text-only 임을 명시해야 함
```

`promote` 가 이 파일을 더 이상 지우지 않으므로 재트레이스해도 남습니다.

## 4. 그동안 바뀐 것 (산출물에 영향)

```
ports.jsonl          v1 [op,slot] -> v2 {"kind":"op",...} + typed external.  full/ 안이라 미출고
axis_resolution      새 사이드카(축 판정 등급/질문).                          full/ 안이라 미출고
배치 축 불변식       위 1번. **발행 csv/jsonl 이 바뀐다**
ties 회계            버려지는 렌더가 값 충돌을 부풀리던 것. 라벨 불변
```

---

## 질문

**Q1 (재트레이스 범위).** 배치 축 수정이 14개 전부에 걸립니다. 셋 중 무엇입니까?

  (a) results 대상 **14개만** 재트레이스 -> 승격 -> sync. 출고본은 일관되지만 `models/` 안에서
      14개는 새 규칙, 33개는 옛 규칙이 됩니다
  (b) **47개 전부** 재트레이스. 일관되지만 오래 걸리고, 지금 목표(모델별 결과)와는 무관한
      작업까지 포함됩니다
  (c) Llama-4 만 올리고 나머지는 다음에

저희는 (a) 로 기울었습니다. 다만 `models/` 가 두 판으로 갈리는 것이 나중에 비교 기준
(`develop/verify/*.json` 의 기준선들)을 흔들지 걱정됩니다.

**Q2 (열린 항목 7건을 안은 채 출고).** `review_request.md` 의 "판단 필요 0건" 이 출고 기준인데,
그건 **자동 규칙이 전부 결정했다**는 뜻이지 "검토에서 지적이 없다" 는 뜻이 아닙니다. 지금
Llama-4 는 판단 필요 0건이면서 열린 지적이 7건입니다. 이 상태로 내보내도 됩니까, 아니면
출고 기준에 **`status: open` 인 발견이 없을 것**을 추가해야 합니까? 후자면 지금 14개 중
상당수가 빠질 수 있습니다.

**Q3 (등급이 출고본에 안 실립니다).** 축 판정 사이드카(`axis_resolution.jsonl`)는 `full/`
안에 있고, results 브랜치는 `full/` 을 제외합니다. 즉 **받는 사람은 어떤 축이 확정이고 어떤
축이 미확정인지 알 수 없습니다.** 앞서 "사람이 계속 속는다" 고 지적하신 그 상태입니다.
출고본에 무엇을 실어야 합니까 -- 사이드카를 `full/` 밖으로 옮깁니까, 요약만 `model_summary.md`
에 넣습니까, 아니면 CSV 자체에 등급을 표시해야 합니까?

**Q4 (혼합 스키마).** 재트레이스한 모델은 `ports.jsonl` 이 v2, 나머지는 v1 이 됩니다. 읽는
쪽은 **한 파일 안에 두 판이 섞이면** 거부하도록 했고 모델 간 혼합은 허용합니다. 이 상태로
두어도 됩니까?

**Q5 (Llama-4 자체).** 위 수정 뒤의 Llama-4 산출물을 **지금 내보내도 되는 상태**로 보십니까?
열린 7건 중 출고 전에 반드시 고쳐야 할 것이 있습니까? 저희 판단으로는 `ctx` 와 `w_local`
이름이 사실관계 오류에 가까워 먼저 고칠 후보로 보이는데, 나머지(표현 누락)는 기록만 해도
된다고 봤습니다.
