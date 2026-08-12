# 라벨 검토 결과 — deepseek-ai/DeepSeek-V4-Pro

- 검토일: 2026-08-12
- 검토자: llm(claude, C절 전수 + 소스 대조)
- 본 것: **A·B·C절 전건 수행 완료.** C절은 (모듈, 라벨) 쌍 8,886건을 모집단으로 삼고, 심볼 자신의 scope 가 그 모듈을 덮지 않는 경우를 기계로 선별해(등록 유도식이 그 모듈 스코프로 설명하는 라벨은 제외) 20건을 전건 판정했다. 9건은 규칙 교정으로 닫았고 11건은 판정과 함께 남는다. 모집단·선별 기준은 review/04-full-inventory.md.
- 요약: 의뢰서 1건 — 같은 op 의 입력과 출력이 다르게 렌더되던 것을 찾아 교정 완료.

> 이 파일은 `review_findings.json` 에서 생성된다 — 고칠 때는 JSON 을 고친다.

## 발견 1 — 교정 필요 (미반영)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.self_attn.compressor.kv_norm` |
| 축 | [B, 512, 512] 의 축 순서 |
| 현재 라벨 | `[B, d_head, d_head]` |
| 판정 | `should_be_renamed` |
| 제안 라벨 | `[B, T/m_csa, d_head]` |
| 확신도 | high |
| 산출물 반영 | 미반영 |

**근거**

`modeling_deepseek_v4.py:382` `self.kv_norm = DeepseekV4RMSNorm(self.head_dim, ...)` — RMSNorm 은 마지막 축을 정규화하므로 마지막이 `d_head`(512)이고 가운데가 압축 KV 길이다. **부분 교정(2026-08-09)**: rank-1 norm 앵커를 그 모듈 전체로 확장해 본체 텐서는 `[B, T, d_head]` / `[B, T/m_hca, d_head]` 로 맞았다. **정정(2026-08-10)** — 그때 '교정 완료'라고 적었지만 사실이 아니었다. 새로 넣은 elementwise 라벨 일관성 검사가 같은 모듈에서 30행을 잡아냈다: `elementwise_mul([B, d_head, T/m_csa], [B, d_head, 1]) -> [B, d_head, d_head]` — T/m_csa 가 2048/4 = 512 로 d_head 와 같은 자리라 입력과 출력이 서로 다른 이름을 달고 있다. 값으로는 못 가리고, norm 앵커는 마지막 축만 고정하므로 가운데 축이 남는다. 게이트가 이제 이 30행을 매번 보고한다.

## 발견 2 — 교정 필요 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.self_attn.compressor.indexer.scorer.weights_proj` |
| 축 | 출력 폭 (index_n_heads) |
| 현재 라벨 | `n_h` |
| 판정 | `should_be_renamed` |
| 제안 라벨 | `n_h_I` |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

`modeling_deepseek_v4.py:444` `self.weights_proj = nn.Linear(config.hidden_size, config.index_n_heads, bias=False)`. index_n_heads == num_attention_heads 라 값으로는 구별이 불가능했다. 원인은 스코프 정규식이 경로 어디서든 매치한다는 것 — indexer 는 `self_attn` 안에 있어 바깥 모듈의 스코프(`attn|attention`)를 물려받았고, 전역 우선순위로 `n_h`(7)가 `n_h_I`(24)를 이겼다. `symbolic_shape._ctx_symbols` 를 고쳐 **더 안쪽에서 매치하는 스코프가 이기도록** 했다.

## 발견 3 — 교정 필요 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.self_attn.compressor.indexer.kv_proj / gate_proj` |
| 축 | 출력 폭 (2·index_head_dim) |
| 현재 라벨 | `d_head/2` |
| 판정 | `should_be_renamed` |
| 제안 라벨 | `2*c_I` |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

`modeling_deepseek_v4.py:489` `self.kv_proj = nn.Linear(config.hidden_size, 2 * self.head_dim, bias=False)` — `self.head_dim = config.index_head_dim` (:487). V4-Pro 는 d_head=512, index_head_dim=128 이라 d_head/2 = 2·c_I = 256 으로 값이 겹쳤다. `rules/derived_dims.yaml` 에 `2*c_I` 를 indexer 스코프로 등록했다.

## 발견 4 — 교정 필요 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.self_attn.compressor.indexer.q_b_proj` |
| 축 | 출력 폭 (index_n_heads·index_head_dim) |
| 현재 라벨 | `g_o*d_g` |
| 판정 | `should_be_renamed` |
| 제안 라벨 | `n_h_I*c_I` |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

`modeling_deepseek_v4.py:494` `self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)`. V4-Flash 는 g_o·d_g = 8·1024 = 8192 = 64·128 = n_h_I·c_I 로 값이 같다. indexer 규칙을 grouped-o 규칙보다 **앞에** 두어 파일 순서로 이기게 했다.

## 발견 5 — 교정 필요 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.self_attn.compressor.indexer.*` |
| 축 | 입력 폭 (hidden_size) |
| 현재 라벨 | `n_h*d_head/g_o, n_h*d_rope` |
| 판정 | `should_be_renamed` |
| 제안 라벨 | `d_model` |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

위 세 투영의 입력은 전부 `config.hidden_size` 다. V4-Flash 는 n_h·d_head/g_o = 64·512/8 = 4096 이고 n_h·d_rope = 64·64 = 4096 이라 둘 다 d_model 과 겹쳤고, 스코프 유도식이 스코프 없는 평범한 심볼보다 먼저 평가되므로 잔차 스트림을 가져갔다. 두 규칙에 `unless_equals: [d_model]` 을 달았다.

## 발견 6 — 교정 필요 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.self_attn.compressor.indexer` |
| 축 | 겹침 창 슬롯 수 (2·compress_rate) |
| 현재 라벨 | `8 (이름 없음)` |
| 판정 | `should_be_renamed` |
| 제안 라벨 | `2*m_csa` |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

`modeling_deepseek_v4.py:526` `new_kv = chunk_kv.new_zeros((batch, n_windows, 2 * ratio, self.head_dim))`, `ratio = self.compress_rate`. 창 하나가 앞 창과 겹치도록 자리를 두 배로 잡는 Ca/Cb 레이아웃이다. 규칙으로 등록해 정수를 없앴다.

## 발견 7 — 맞음 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model` |
| 축 | mHC 병렬 잔차 스트림 수 (4) |
| 현재 라벨 | `n_hc (스코프 밖 폴백)` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

`expand [B,T,1,d_model] -> [B,T,4,d_model]` 이 레이어 안이 아니라 **스택 루트**에서 일어난다(실측 `[1,2048,4,7168]`). n_hc 가 정확히 맞는 자리인데 스코프가 루트를 못 덮어 폴백으로 붙어 있었다. 스코프에 `^model$` 를 추가했다.

## 발견 8 — 교정 필요 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.self_attn.compressor.{kv,gate}_proj` |
| 축 | 압축기 투영 폭 (1024) |
| 현재 라벨 | `d_g` |
| 판정 | `should_be_renamed` |
| 제안 라벨 | `2*d_head` |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

소스 docstring 이 그대로 적어 놓았다(`modeling_deepseek_v4.py:587-594`): "`kv_proj` / `gate_proj` / `position_bias` project to `2 * head_dim`: each token contributes two independent compressed series Ca and Cb". V4-Pro 는 2·512 = 1024 = o_lora_rank 라 **grouped output projection 의 그룹당 중간 차원** 이름이 붙어 있었다 — 전혀 다른 모듈의 이름이다. `2*d_head` 를 compressor 스코프로 등록했다.

## 발견 9 — 교정 필요 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.self_attn.compressor.indexer` |
| 축 | Ca/Cb 창 축 (4) |
| 현재 라벨 | `n_hc` |
| 판정 | `should_be_renamed` |
| 제안 라벨 | `m_csa` |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

`DeepseekV4Indexer.__init__` 이 `self.compress_rate = config.compress_rates["compressed_sparse_attention"]` 를 직접 읽고 `chunk_kv.view(batch, n_windows, ratio, -1)` 로 그 축을 만든다(:485,521). m_csa = n_hc = 4 라 hyper-connection 스트림 수의 이름이 붙어 있었다.

m_csa 스코프가 indexer 를 배제하고 있던 것이 원인인데, 그 배제는 원래 **m_hca** 하나 때문이었다(m_hca=128 == c_I=128). 예전에 배제를 풀려다 되돌린 기록이 있는데(V4-Pro heur 2,131→3,331) **둘을 함께 열었던 것**이 문제였다. m_csa 만 열자 퇴행 0 / 개선 3 으로 통과했다.

## 발견 10 — 교정 필요 (미반영)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.self_attn` |
| 축 | grouped output projection 그룹 축 (16) |
| 현재 라벨 | `T/m_hca` |
| 판정 | `should_be_renamed` |
| 제안 라벨 | `g_o` |
| 확신도 | high |
| 산출물 반영 | 미반영 |

**근거**

`clone [B,T,T/m_hca,d_g] -> _unsafe_view -> [B,T,g_o*d_g]` (실측 `[1,2048,16,1024]` → `[1,2048,16384]`). 합쳐진 축이 `g_o*d_g` 이므로 셋째 축은 `g_o` 여야 하는데 g_o = T/m_hca = 16 이라 압축 엔트리 수의 이름이 붙었다. `d_g` 자체는 맞다.

고치려면 권위 있는 출력 라벨(`g_o*d_g`)의 인수를 입력 축으로 되밀어야 하고, 그 기계장치(`_split_from_authoritative`)가 이 op 에서는 발화하지 않는다. MLA 의 `d_v` 건과 **같은 막힘**이다 — 개명을 데이터플로우 끝까지 옮기는 문제.
