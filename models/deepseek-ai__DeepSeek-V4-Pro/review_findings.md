# 라벨 검토 결과 — deepseek-ai/DeepSeek-V4-Pro

- 검토일: 2026-08-11
- 검토자: llm(claude, 전수 점검 2회차 — 모듈-필드 소속)
- 본 것: 전수 점검 2회차 — 1층(모듈-필드 소속: 가중치 축의 이름이 그 모듈/부모가 실제로 읽는 config 필드에서 나왔는가)을 함대 전건 수행. 값을 보지 않는 검사라 값 충돌이 숨길 수 없다. 1회차의 A절·B절 판정은 유지. C절(모듈별 출력 shape)은 여전히 미수행.
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
