# 검토 요청 — 확정 출고된 4개 모델의 CSV/JSONL

## 무엇을 봐 달라는 것인가

아래 네 모델의 **`prefill.csv` / `decode.csv` / `prefill.jsonl` / `decode.jsonl`** 이
해당 모델의 실제 아키텍처를 정확히 표현하는지, **공식 자료와 소스 코드를 직접 열어**
대조해 주십시오. 요약 카드(`model_summary.md`)가 아니라 **표 안의 축 라벨**이 대상입니다.

받는 곳:
```
git clone --branch results --single-branch --depth 1 https://github.com/KimHyeon-Ji/llm-arch-tracer
```
네 모델은 `models/<이름>/` 아래에 있습니다. 37MB입니다.

| 모델 | 발행 (B, T) | prefill 행 | decode 행 |
|---|---|---:|---:|
| `meta-llama__Llama-4-Maverick-17B-128E` | B=3, T=17 | 67 | 67 |
| `openai__gpt-oss-20b` | B=3, T=264 | 47 | 47 |
| `openai__gpt-oss-120b` | B=3, T=264 | 47 | 47 |
| `deepseek-ai__DeepSeek-V4-Pro` | B=3, T=2176 | 224 | 212 |

표는 **접혀 있습니다** — 구조가 같은 레이어는 한 행이고 `repeat` / `layers` 열이 몇 개
레이어를 대표하는지 적습니다. 행 수가 적은 것은 그 때문이지 op 이 빠진 것이 아닙니다.

## 표를 읽는 법

열은 다음과 같습니다.

```
op_id, block_type, repeat, layers, h1, h2, op_type,
input_shape, weight_shape, weight_pos, output_shape,
depends_on, layer_idx, block, sub_block, depth,
module_path, raw_op, params, phase, unmapped, caveat
```

예 (gpt-oss-20b prefill):
```
linear          ...self_attn.q_proj   in=[[n_h*d_head], [B*T, d_model], [d_model, n_h*d_head]]  out=[[B*T, n_h*d_head]]
batched_matmul  ...self_attn          in=[[B*n_h, T, d_head], [B*n_h, d_head, T]]               out=[[B*n_h, T, T]]
```

주의할 점 세 가지:

1. **`weight_pos`** — `input_shape` 안에 가중치도 함께 들어 있습니다. 어느 피연산자가
   가중치인지는 `weight_pos` 가 말합니다. FLOPs/바이트를 세려면 이걸 먼저 보십시오.
2. **`B` 는 배치입니다.** 이번 판부터 발행 트레이스를 **B=3** 으로 잡았습니다(이전에는
   B=1). B=1 에서는 배치·decode 쿼리 길이·방송 싱글턴이 수치적으로 같고 `B*T == T`,
   `B*n_h == n_h` 라서 **접힌 배치 축이 라벨에서 통째로 사라졌습니다.** 그래서 `B*T`,
   `B*n_h` 같은 라벨이 이번에 새로 보입니다. 이게 맞는지가 핵심 확인 대상입니다.
3. **`caveat` 열** — 모델 동작을 대체한 자리를 표시합니다. 이 네 모델에서는 전부 빈칸입니다
   (Kimi-K3 만 차 있고 그 모델은 이번 대상이 아닙니다).

## 먼저 읽어 주십시오 — 이미 알고 있는 미확정

모델마다 **`UNKNOWNS.md`** 가 함께 있습니다. 여기에 **우리가 이미 모른다고 판단한 것**이
전부 적혀 있습니다. 이번 검토에서 그 목록을 다시 지적하실 필요는 없습니다 — 그것을 푸는
것은 다음 단계로 미뤄 둔 작업입니다.

등급별로 의미가 다릅니다:

* `open_tie` — 후보 둘이 **같은 값**이라 트레이스만으로 못 갈랐습니다. 구체 크기·FLOPs·
  바이트는 맞고 **이름만** 미정입니다.
* `scope_inferred` — scope 정규식만이 후보를 갈랐고 아무도 검증하지 않았습니다.
* `heuristic` — **산술로 지어낸 이름입니다.** 값은 맞지만 이름이 틀릴 수 있습니다.
* `unresolved` — 근거가 없어 정수로 뒀습니다.

확정률: Llama-4 99.1% / gpt-oss-20b 78.2% / gpt-oss-120b 75.5% / V4-Pro 75.0%.

자리 단위 원장은 `prefill.axis_resolution.jsonl` / `decode.axis_resolution.jsonl` 에
`(op_id, field, shape_index, axis)` 키로 들어 있습니다.

## 물어보고 싶은 것

**Q1. 접힌 배치 축이 맞습니까.**
B=3 으로 잡으면서 `B*T`, `B*n_h`, `B*k*T` 같은 라벨이 생겼습니다. 각 모델에서
**행렬곱·배치행렬곱의 선두 축**이 실제로 그 접힘인지, 아니면 다른 것을 접은 것인지
소스로 확인해 주십시오. 특히:
  * gpt-oss: `[B*n_h, T, T]` 의 선두가 정말 `B*n_h` 인가 (`n_h=64`, `B=3` → 192)
  * Llama-4: expert 경로의 `B*E` / `E*B` 계열
  * V4-Pro: `B*k*T` (`k=6`) 가 routed 토큰 수로 맞는가

**Q2. decode 의 축이 맞습니까.**
decode 는 쿼리 길이가 1 입니다. 그래서 `[B, 1, ...]` 과 `[B, T+1, ...]`(캐시 포함 길이)이
섞여 나옵니다. 크기 1 인 축을 `B` 로 잘못 읽거나, 캐시 길이를 `T` 로 잘못 읽은 자리가
있는지 봐 주십시오.

**Q3. 각 모델 고유 구조가 제대로 표현됐습니까.**
  * **gpt-oss** — sliding window (`w_local=128`) 와 full attention 의 교대, MoE 라우팅
    (`E=32/128`, `k=4`), attention sink. `d_model` 과 `d_moe` 가 **둘 다 2880** 이라
    값으로는 못 가릅니다(우리도 이걸 `UNKNOWNS.md` 에 올려 뒀습니다) — 소스 기준으로
    어느 자리가 어느 것인지 말해 주시면 그대로 반영합니다.
  * **Llama-4-Maverick** — chunked attention (`chunk_size=8192`), iRoPE/NoPE 레이어,
    MoE (`E=128`) + shared expert, `n_h=40 / n_kv=8`.
  * **DeepSeek-V4-Pro** — MLA, CSA/HCA 두 압축기(`m_csa=4`, `m_hca=128`), Lightning
    Indexer (`c_I=128`, `index_topk`), grouped o_proj (`g_o=16`, `d_g=1024`), mHC.
    T=2176 은 두 압축률의 공배수라 **내림이 아무것도 안 깎습니다** — 압축기가 쓰는 길이가
    곧 `T` 입니다. 이게 표에 맞게 반영됐는지 봐 주십시오.

**Q4. 표가 아키텍처를 잘못 말하는 자리가 있습니까.**
위 목록에 없는, 우리가 **모른다고도 표시하지 못한** 오류를 찾아 주십시오. 그게 이 검토의
가장 큰 가치입니다. 각 지적마다:
  * 어느 모델 / 어느 phase / 어느 `op_id` (또는 `module_path` + `op_type`)
  * 지금 라벨 → 맞는 라벨
  * **근거**: 파일:줄 (설치된 transformers 5.14.1 기준이 가장 좋고, 없으면 공식 리포지토리
    커밋/URL). 근거 없는 판정은 반영할 수 없습니다.

## 반영 방법 (참고)

판정은 `rules/label_overrides.yaml` 에 인용과 함께 들어가고, 아무 축에도 안 맞는 항목은
게이트가 잡습니다. 그러니 **자리 지목이 정확할수록** 그대로 반영됩니다.
