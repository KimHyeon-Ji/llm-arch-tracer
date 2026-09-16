# 검토 요청 — 이 표가 그 모델의 구조를 정확히 말하고 있습니까

## 딱 하나만 물어봅니다

네 모델의 **`prefill.csv` / `decode.csv` / `prefill.jsonl` / `decode.jsonl`** 이
**그 모델의 실제 아키텍처를 정확히 표현하는가**, 그리고 **모든 축 라벨이 맞는가**.

요약 카드가 아니라 **표 안의 op 구성과 축 이름**이 대상입니다. 공식 자료와 소스 코드를
직접 열어서 대조해 주십시오.

| 모델 | 발행 (B, T) |
|---|---|
| `meta-llama/Llama-4-Maverick-17B-128E-Instruct` | B=3, T=17 |
| `openai/gpt-oss-20b` | B=3, T=264 |
| `openai/gpt-oss-120b` | B=3, T=264 |
| `deepseek-ai/DeepSeek-V4-Pro` | B=3, T=2176 |

**같이 드리는 `codex_ask_four_models_claims.md` 에 표가 주장하는 구조 전부가 적혀 있습니다**
— block_type 별 대표 1 반복의 op 순서와 shape 입니다. 클론 없이 그것만으로 대조 가능합니다.
전체 파일이 필요하시면:

```
git clone --branch results --single-branch --depth 1 https://github.com/KimHyeon-Ji/llm-arch-tracer
```

## 두 가지 층위로 봐 주십시오

### 층위 1 — 구조가 맞습니까 (op 구성)

부록의 op 순서가 그 모델의 forward 와 일치합니까. 구체적으로:

* **빠진 op 이 있습니까.** 소스에는 있는데 표에 없는 연산.
* **없는 op 이 있습니까.** 표에는 있는데 그 모델이 하지 않는 연산.
* **block_type 분할이 맞습니까.** 예를 들어 gpt-oss 는 `attn+MoE` 하나에
  `repeat=12, layers=0,2,4,...,22` 인데 그 블록 안에 `layers.0` 과 `layers.1` 이 **둘 다**
  들어 있습니다. 즉 반복 단위를 **레이어 쌍**으로 잡았다는 주장입니다. sliding window 와
  full attention 이 교대하는 구조를 이렇게 표현한 것이 맞습니까?
* **모델 고유 기제가 표에 보입니까.**
  - gpt-oss: sliding window(`w_local=128`), attention sink. 표의 softmax 가
    `[[B, n_h, T, T+1]]` 인데 이 `T+1` 이 sink 한 칸입니까? 그리고 **sliding window 가
    표 어디에 나타나 있습니까** — 안 보이면 그게 누락입니까, 아니면 마스크라 op 이 없는
    것이 맞습니까?
  - Llama-4: chunked attention(`chunk_size=8192`), iRoPE/NoPE 레이어 구분, shared expert.
  - V4-Pro: MLA, CSA/HCA 두 압축기, Lightning Indexer, grouped o_proj, mHC.
    T=2176 은 `m_csa=4` 와 `m_hca=128` 의 공배수라 **내림이 아무것도 안 깎습니다** —
    압축기가 쓰는 길이가 곧 `T` 입니다. 표가 그렇게 말하고 있는데 맞습니까?

### 층위 2 — 라벨이 맞습니까 (축 이름)

각 축에 붙은 이름이 그 축이 실제로 세는 것과 일치합니까.

* **폭인가 개수인가.** 헤드 수(`n_h`)가 와야 할 자리에 헤드 폭(`d_head`)이 오거나 그 반대.
* **접힌 축의 이름.** 이번 판부터 발행을 **B=3** 으로 잡았습니다(이전 B=1). B=1 에서는
  배치·decode 쿼리 길이·방송 싱글턴이 수치적으로 같고 `B*T == T`, `B*n_h == n_h` 라
  **접힌 배치 축이 라벨에서 통째로 사라져 있었습니다.** 그래서 `B*T`, `B*n_h`, `B*k*T` 가
  이번에 처음 보입니다. 이것들이 정말 그 접힘입니까?
  - gpt-oss `[B*n_h, T, T]` 의 선두가 `n_h=64`, `B=3` → 192 로 맞습니까
  - gpt-oss `[B*k*T, d_model]` 이 routed 토큰 수(`k=4`)로 맞습니까
  - V4-Pro `B*k*T` (`k=6`)
* **decode 의 축.** decode 는 쿼리 길이가 1 이라 `[B, 1, ...]` 과 캐시 포함 길이가 섞입니다.
  크기 1 인 축을 `B` 로 잘못 읽거나 캐시 길이를 `T` 로 잘못 읽은 자리가 있습니까?
* **가중치 축.** `input_shape` 안에 가중치도 들어 있고 어느 것이 가중치인지는 `weight_pos`
  가 말합니다. 가중치 축에 배치나 시퀀스 이름이 붙은 자리가 있으면 그건 무조건 오류입니다.

## 이미 모른다고 표시한 것은 다시 지적 안 하셔도 됩니다

모델마다 **`UNKNOWNS.md`** 가 함께 나갑니다. 우리가 **이미 모른다고 판단한 자리**가 전부
거기 있습니다. 그걸 푸는 것은 다음 단계로 미뤄 둔 작업입니다.

등급별 의미:

* `open_tie` — 후보 둘이 **같은 값**이라 못 갈랐습니다. 구체 크기·FLOPs·바이트는 맞고
  **이름만** 미정입니다. (예: gpt-oss 의 `d_model`·`d_moe` 가 **둘 다 2880**)
* `scope_inferred` — scope 정규식만이 갈랐고 아무도 검증 안 했습니다.
* `heuristic` — **산술로 지어낸 이름.** 값은 맞지만 이름이 틀릴 수 있습니다.
* `unresolved` — 근거가 없어 정수로 뒀습니다.

확정률: Llama-4 99.1% / gpt-oss-20b 78.2% / gpt-oss-120b 75.5% / V4-Pro 75.0%.

**다만 이 목록 중에서도, 소스를 보면 바로 갈리는 것이 있으면 말씀해 주십시오.** 특히
gpt-oss 의 `d_model` vs `d_moe`(둘 다 2880)는 어느 자리가 어느 것인지 소스로 정해지면
그대로 반영합니다. 값으로는 영원히 못 가릅니다.

**가장 큰 가치는 이 목록에 없는 오류입니다** — 우리가 모른다고도 표시하지 못한 자리.

## 지적하실 때 필요한 것

각 지적마다:

* 어느 모델 / 어느 phase / 어느 `op_id` (또는 `module_path` + `op_type`)
* 지금 라벨(또는 지금 op 구성) → 맞는 것
* **근거**: 파일:줄. 설치된 transformers 5.14.1 기준이 가장 좋고, 없으면 공식 리포지토리의
  커밋/URL. **근거 없는 판정은 반영할 수 없습니다** — 판정은 인용과 함께
  `rules/label_overrides.yaml` 에 들어가고, 아무 축에도 안 맞는 항목은 게이트가 잡습니다.

자리 지목이 정확할수록 그대로 반영됩니다. 자리 단위 원장이
`prefill.axis_resolution.jsonl` / `decode.axis_resolution.jsonl` 에
`(op_id, field, shape_index, axis)` 키로 들어 있습니다.
