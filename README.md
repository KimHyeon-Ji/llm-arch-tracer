# llm-arch-tracer — 결과물 브랜치

LLM 체크포인트를 실제로 한 번 forward 시켜 얻은 **연산 그래프 요약**입니다.
모델마다 prefill / decode 두 단계의 major-op 표를 csv 와 jsonl 두 형식으로 담습니다.

```
git clone --branch results --single-branch --depth 1 <repo-url>
```

## 지금 들어 있는 모델 (5개)

```
deepseek-ai__DeepSeek-V4-Pro
meta-llama__Llama-4-Maverick-17B-128E
moonshotai__Kimi-K3
openai__gpt-oss-120b
openai__gpt-oss-20b
```

`main` 브랜치에는 더 많은 모델이 있지만, **출고 기준을 통과한 것만** 여기 옮깁니다.
기준은 게이트 통과 + 축 판정 원장 존재 + 미확정을 `UNKNOWNS.md` 에 공개할 것입니다.
기준을 못 채우게 되면 내려갑니다 — 그래서 모델 수가 줄어들 수 있습니다.

## 파일

| 파일 | 내용 |
|---|---|
| `prefill.csv` / `decode.csv` | major-op 표 |
| `prefill.jsonl` / `decode.jsonl` | 같은 내용의 JSON Lines |
| `structure.yaml` | 심볼표(`d_model`, `n_h`, …)와 이 산출물의 계약·한계 |
| `UNKNOWNS.md` | **이 판에서 검증되지 않은 것 전부** |
| `prefill.axis_resolution.jsonl.gz` | 축 자리마다 어떤 근거로 그 이름이 됐는지 (gzip) |
| `model_summary.md` | 사람이 읽는 요약 |
| `review_findings.json` / `review_request.md` | 검토 기록과 미결 항목 |
| `audit_manifest.json` / `lowering_proof.json` | 출고 감사 결과 |

축 원장은 **gzip** 입니다 — Kimi-K3 의 것이 657 MB 라 GitHub 파일당 100 MB 한도를
넘습니다. `gunzip` 또는 `gzip.open()` 으로 읽으세요. 형식은 JSON Lines 이고 첫 줄이
요약, 그다음이 접힌 질문, 그 뒤가 자리별 기록입니다.

## 표 읽는 법

```
op_id, block_type, repeat, layers, h1..h4(또는 h5), op_type,
input_shape, weight_shape, weight_pos, output_shape,
depends_on, layer_idx, block, sub_block, depth, module_path, raw_op,
params, phase, unmapped, caveat
```

* **shape 은 심볼 식**입니다(`[[B*T, d_model], [d_model, c_q]]`). `structure.yaml` 의
  심볼표 값을 대입하면 실제 크기가 나옵니다.
* **`input_shape` 는 가중치 피연산자도 포함**합니다. `weight_pos` 가 그중 몇 번째인지
  가리킵니다 — FLOPs·바이트를 셀 때 `weight_shape` 를 따로 더하면 **중복**입니다.
* **층은 접혀 있습니다.** `repeat` 는 그 블록이 몇 층에 해당하는지, `layers` 는 실제
  층 번호입니다. 전체를 세려면 `layers` 를 펼치세요.
* **`depends_on` 은 표 안에서 닫혀 있습니다** — 표 밖을 가리키는 간선이 없으므로
  `op_id` 를 노드로 그대로 DAG 를 만들 수 있습니다.
* `structure.yaml` 의 `symbols_label_only` 가 비어 있지 않으면, 표에 나오는데 심볼표에
  정의가 없는 식별자입니다. 그 식과 이번 실행의 값이 거기 적혀 있습니다.

## 반드시 알고 쓰셔야 하는 것

**이 표는 major-op 요약이지 전체 ATen op 목록이 아닙니다.** norm·elementwise·mask
생성·RoPE·cache concat 같은 연산이 빠져 있습니다. compute 쪽은 matmul 이 지배하니
영향이 작지만, **memory-bound 쪽을 세면 과소평가**됩니다. 전체 트레이스는 `main`
브랜치의 해당 revision `full/` 에 있습니다.

**축 이름의 확신도는 자리마다 다릅니다.** 크기는 실제로 추적한 값이라 전부 맞지만,
이름은 등급이 나뉩니다 — `confirmed`(소스로 확인) / `scope_inferred`(모듈 이름
규칙만) / `heuristic`(**산술로 지어낸 이름**) / `unresolved`(이름 없이 정수).
**`heuristic` 은 이 체크포인트에서만 우연히 맞는 식일 수 있습니다** — 심볼에 다른
값을 대입해 재계산하는 용도라면 이 등급은 믿지 마세요. 등급은 축 원장에
`(op_id, field, shape_index, axis)` 키로 자리마다 적혀 있고, 모델별 집계는 각
`UNKNOWNS.md` 1절에 있습니다.

**MoE 는 모델마다 표현이 다릅니다.** 같은 "토큰을 전문가로 라우팅" 인데 셋으로
갈립니다 — 배치를 키우며 sweep 하면 서로 다르게 스케일합니다.

| 모델 | 표현 | 주의 |
|---|---|---|
| gpt-oss, V4-Pro | `grouped_matmul`, `[B*k*T, …]` | 라우팅된 (토큰, 슬롯) 쌍 |
| Llama-4 | `batched_matmul`, `[E, B*T, …]` | reference 구현이라 **모든 전문가가 모든 토큰**을 계산한다. top-k 기준으로 읽으면 E/k 배 과대평가 |
| Kimi-K3 | 전문가별 `matmul` | even-split shim — 896 중 **4개만** 추적. `caveat` 열이 찬 행이 그것 |

**dtype 이 기록돼 있지 않습니다.** 바이트·메모리 트래픽을 정확히 내려면 별도로
넣으셔야 합니다. gpt-oss 의 MXFP4 unpack/dequant 연산도 표에 없습니다 — config 로
만든 weight 없는 모델을 추적한 것이라 양자화 로딩 경로를 실행하지 않았습니다.

**`weight_pos = -1` 이 "가중치 없음"과 같은 뜻은 아닙니다.** V4-Pro 의 grouped
output bmm 이 그렇습니다 — `weight_pos` 는 -1 인데 입력 1 이 실제로는 가중치의 view
입니다. gpt-oss 의 `linear` 는 입력 0 이 bias, 1 이 activation, 2 가 weight 입니다.

**`block_type` 은 attention 종류를 구분하지 않습니다.** 같은 `attn+MoE` 로 보이는 두
블록이 sliding 과 full 일 수 있습니다. 실제 구성은 `layers` 열과 `structure.yaml` 의
`symbols.layer_sched` 를 맞춰 봐야 합니다. Kimi-K3 의 `attn` 은 KDA(linear attention)
이고, DeepSeek-V4-Pro 의 `MLA` 는 DeepSeek-V2/V3 식 latent-KV 확장이 아닙니다.

**추적 범위**: FakeTensor/meta 경로로 text-only forward 한 번입니다. 멀티모달
체크포인트에서도 텍스트 백본만 잡습니다(vision tower 는 설계상 범위 밖). 실제 GPU
kernel 의 op 구성(Triton 등)도 이 표의 범위가 아닙니다.

## 검증 상태

통과한 검사: 게이트 C1~C17 하드 불변식, 독립 배치 검증(발행 라벨을 다른 배치 크기의
실제 shape 과 대조), module-field membership, reshape 자체 유도 대조.

**아직 열려 있는 것**은 각 모델 `UNKNOWNS.md` 3·5절에 그대로 적혀 있습니다.
특히 **규칙이 못 잡는 종류의 오류를 사람/LLM 이 직접 보는 층(③ 자유 평가)이
5개 모델 모두 만료 또는 미수행**입니다. 2026-09-21 에 외부 검토(Codex)가 층 접기와
주요 projection/attention/MoE 라벨을 소스와 대조해 확인했지만, 수백만 축 전부를
재확정한 것은 아닙니다.

**수치는 여기 베껴 적지 않습니다** — 각 모델의 `UNKNOWNS.md` 를 보세요. 그쪽이
산출물과 함께 자동 생성되므로 낡지 않습니다. 이 README 가 한동안 "17개 모델,
외부 검증 완료" 라고 적혀 있던 것이 그 교훈입니다.

## 갱신

`main` 에서 결과물이 바뀔 때마다 이 브랜치에 새 스냅샷 커밋 하나가 추가됩니다.
커밋 메시지가 기준이 된 `main` 커밋을 가리킵니다. 히스토리가 필요 없으면
`--depth 1` 로 받으세요.
