# 라벨 검토 결과 — deepseek-ai/DeepSeek-V4-Pro

- 검토일: 2026-08-13
- 검토자: llm(claude, 반박 프레임 전건 판정)
- 본 것: 의뢰서 항목을 **항목 단위로** 대조해 하나도 빠뜨리지 않는다(src/review_ledger.unanswered_items 가 개수가 아니라 항목을 맞춘다). 각 항목마다 그 폭을 만드는 코드 줄을 열어 확인했다.
- 요약: 2026-08-13 미답 2건 + 2026-08-31 재검토(102개 앵커 확정, c_I/2 발견) + 2026-09-01 외부 검토(Codex): RoPE θ/KV cache/hash_moe 요약문 버그 3건 수정, c_I/2 판정을 c_I-d_rope/d_rope로 정정, g_o는 이미 해결돼 있었음을 재확인.

> 이 파일은 `review_findings.json` 에서 생성된다 — 고칠 때는 JSON 을 고친다.

## 발견 1 — 교정 필요 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.self_attn.compressor.kv_norm` |
| 축 | [B, 512, 512] 의 축 순서 |
| 현재 라벨 | `[B, d_head, d_head]` |
| 판정 | `should_be_renamed` |
| 제안 라벨 | `[B, T/m_csa, d_head]` |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

`modeling_deepseek_v4.py:382` `self.kv_norm = DeepseekV4RMSNorm(self.head_dim, ...)` — RMSNorm 은 마지막 축을 정규화하므로 마지막이 `d_head`(512)이고 가운데가 압축 KV 길이다. **부분 교정(2026-08-09)**: rank-1 norm 앵커를 그 모듈 전체로 확장해 본체 텐서는 `[B, T, d_head]` / `[B, T/m_hca, d_head]` 로 맞았다. **정정(2026-08-10)** — 그때 '교정 완료'라고 적었지만 사실이 아니었다. 새로 넣은 elementwise 라벨 일관성 검사가 같은 모듈에서 30행을 잡아냈다: `elementwise_mul([B, d_head, T/m_csa], [B, d_head, 1]) -> [B, d_head, d_head]` — T/m_csa 가 2048/4 = 512 로 d_head 와 같은 자리라 입력과 출력이 서로 다른 이름을 달고 있다. 값으로는 못 가리고, norm 앵커는 마지막 축만 고정하므로 가운데 축이 남는다. 게이트가 이제 이 30행을 매번 보고한다.

**재확인(2026-08-31)**: 이 A44급 문제는 이미 별도 세션에서 override로 닫혔다(커밋 ff831732, "Close A44 (DeepSeek-V4-Pro CSA compressor kv_norm) with a plain override"). 이 finding의 status가 그 뒤로 갱신 안 된 낡은 기록이었다 -- review_request.md 0절에서 더 이상 나타나지 않음을 오늘 재확인했다.

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

`modeling_deepseek_v4.py:444` `self.weights_proj = nn.Linear(config.hidden_size, config.index_n_heads, bias=False)`. index_n_heads == num_attention_heads 라 값으로는 구별이 불가능했다. 원인은 스코프 정규식이 경로 어디서든 매치한다는 것 — indexer 는 `self_attn` 안에 있어 바깥 모듈의 스코프(`attn|attention`)를 물려받았고, 전역 우선순위로 `n_h`(priority 10)가 `n_h_I`(priority 24)를 이겼다 — 앞서 이 자리에 적혀 있던 `n_h`(7) 은 잘못된 숫자였다(`rules/symbols.yaml:57` 이 10, `:202` 가 24다. 외부 검토 지적, 2026-08-13 정정). `symbolic_shape._ctx_symbols` 를 고쳐 **더 안쪽에서 매치하는 스코프가 이기도록** 했다.

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

**근거 소스**: 이 판정은 `develop/sources/modeling_deepseek_v4.py`, `develop/sources/configuration_deepseek_v4.py` 를 열어 확인했다. (인용 누락을 자가 점검에서 발견해 보강, 2026-08-12 — 게이트가 이제 `should_be_renamed` 판정에 소스 인용을 요구한다.)

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

**근거 소스**: 이 판정은 `develop/sources/modeling_deepseek_v4.py`, `develop/sources/configuration_deepseek_v4.py` 를 열어 확인했다. (인용 누락을 자가 점검에서 발견해 보강, 2026-08-12 — 게이트가 이제 `should_be_renamed` 판정에 소스 인용을 요구한다.)

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

**근거 소스**: 이 판정은 `develop/sources/modeling_deepseek_v4.py`, `develop/sources/configuration_deepseek_v4.py` 를 열어 확인했다. (인용 누락을 자가 점검에서 발견해 보강, 2026-08-12 — 게이트가 이제 `should_be_renamed` 판정에 소스 인용을 요구한다.)

## 발견 10 — 맞음 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.self_attn` |
| 축 | grouped output projection 그룹 축 (16) |
| 현재 라벨 | `T/m_hca` |
| 판정 | `current_label_correct` |
| 제안 라벨 | `g_o` |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

`clone [B,T,T/m_hca,d_g] -> _unsafe_view -> [B,T,g_o*d_g]` (실측 `[1,2048,16,1024]` → `[1,2048,16384]`). 합쳐진 축이 `g_o*d_g` 이므로 셋째 축은 `g_o` 여야 하는데 g_o = T/m_hca = 16 이라 압축 엔트리 수의 이름이 붙었다. `d_g` 자체는 맞다.

고치려면 권위 있는 출력 라벨(`g_o*d_g`)의 인수를 입력 축으로 되밀어야 하고, 그 기계장치(`_split_from_authoritative`)가 이 op 에서는 발화하지 않는다. MLA 의 `d_v` 건과 **같은 막힘**이다 — 개명을 데이터플로우 끝까지 옮기는 문제.

**근거 소스**: 이 판정은 `develop/sources/modeling_deepseek_v4.py`, `develop/sources/configuration_deepseek_v4.py` 를 열어 확인했다. (인용 누락을 자가 점검에서 발견해 보강, 2026-08-12 — 게이트가 이제 `should_be_renamed` 판정에 소스 인용을 요구한다.)

**재확인(2026-09-01)**: 이 finding이 남아있던 사이 어딘가에서(오늘 다른 DeepSeek-V4-Pro 작업 중 스코프 조정 등) 이 자리가 실제로 이미 고쳐졌다. 현재 review_request.md에는 g_o가 전혀 안 나오고(unsettled 풀에 없음), 실측 트레이스 (op_id 2039-2045, self_attn.o_a_proj)도 `[g_o, d_g, ...]`/`[g_o, T, d_g]`로 정확히 렌더된다. Codex 외부 검토(2026-09-01)가 'DeepseekV4GroupedLinear.forward의 `self.weight.view(self.n_groups,-1,hidden_dim)`이 g_o를 곱에서 역산할 필요 없이 가중치 자체의 reshape으로 직접 낸다'고 지적한 게 계기 -- 실제로 그 op(2039)이 이미 g_o를 정확히 렌더하고 있었다. 이 finding의 'A45급, 손대지 않기로 함' 판정은 낡은 기록이었다.

## 발견 11 — 맞음 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.self_attn*` |
| 축 | n_h vs w_local, c_I vs n_h (128 / 64) |
| 현재 라벨 | `(값 동률)` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

head **개수**와 head **폭**이 같은 값이라 값으로는 못 가른다. 결정은 값이 아니라 `src/anchors.py` 가 한다 — `nn.Linear.weight == [out, in]` 으로 모듈이 선언한 폭을 읽고, 그 이름을 그 모듈의 모든 op 에 고정한다.

**반박 시도**: 실제로 틀리면 어떤 모습인가? head-개수 이름이 head-폭 축을 가져가면 한 shape 안에 `n_h` 와 `n_kv` 가 함께 나온다(2026-07-30 에 8개 모델 16,859축이 그랬다). 그걸 잡는 `head_excl` 불변식이 현재 함대 전체 · 양쪽 phase 에서 **0** 이다. 또한 `[..., 개수, 폭]` 순서 규약을 어기면 `matmul_compose` 가 걸리는데 그것도 **0** 이다. 틀렸다는 증거를 찾지 못했다.

**근거 소스**: 이 판정은 `develop/sources/modeling_deepseek_v4.py`, `develop/sources/configuration_deepseek_v4.py` 를 열어 확인했다. (인용 누락을 자가 점검에서 발견해 보강, 2026-08-12 — 게이트가 이제 `should_be_renamed` 판정에 소스 인용을 요구한다.)

## 발견 12 — 맞음 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.self_attn.q_b_norm` |
| 축 | head 축 128 |
| 현재 라벨 | `n_h` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

`modeling_deepseek_v4.py:807-809` — V4-Flash 와 같은 코드이고 V4-Pro 는 num_attention_heads=128 이다. transpose(1,2) 뒤 축 1 은 head 개수이며 렌더도 `[B, n_h, T, d_head]`(실측 `[1, 128, 2048, 512]`)다. `w_local`(=128)은 시퀀스 창 길이라 이 자리에 올 수 없다.

## 발견 13 — 맞음 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.self_attn.compressor.indexer.kv_norm` |
| 축 | indexer head 폭 128 |
| 현재 라벨 | `c_I` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

`modeling_deepseek_v4.py:493` `self.kv_norm = DeepseekV4RMSNorm(self.head_dim, ...)`, `:488 self.head_dim = config.index_head_dim`(=128). RMSNorm 은 마지막 축을 정규화하므로 이 모듈의 마지막 축이 곧 index_head_dim 이다(실측 `[1, 512, 128]`). `n_h`(=128)/`w_local`(=128)은 각각 head 개수·창 길이이며 정규화 폭이 될 수 없다.

## 발견 14 — 맞음 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.self_attn / .q_b_norm / .compressor / .compressor.indexer(.scorer(.weights_proj))(.kv_norm)` |
| 축 | n_h vs w_local(128), c_I vs n_h vs w_local(128), n_h_I vs d_rope(64), m_hca vs n_h vs w_local(128) -- 9개 타이 카테고리, 102개 앵커 |
| 현재 라벨 | `위치별로 대부분 이미 정확 (개수 자리는 n_h/n_h_I, 폭/윈도우 자리는 w_local/c_I/d_rope/m_hca)` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

num_attention_heads=128, sliding_window=128, index_head_dim=128 이 전부 이 체크포인트의 실제 하이퍼파라미터 우연(index_n_heads=64=qk_rope_head_dim도 별도로 우연 일치)이다. review_request.md 0절의 self_attn/indexer/scorer/q_b_norm 앵커 102개를 develop/canonical_axis_rules.yaml 스타일 위치 규칙 15개로 측정(develop/rule_coverage.py, 3라운드에 걸쳐 세부 조정) -- 최종 agree=94/differ=0(open 3건은 아래 별도 finding)/clash=0. modeling_deepseek_v4.py의 각 view/transpose/batched_matmul 시퀀스가 근거이며, 각 규칙에 소스 줄 인용을 달아 rules/label_confirmed.yaml에 등록했다(2026-08-31). 루트 스코프의 sliding-window causal mask(w_local), compressor의 position_bias(m_hca), indexer kv_norm 가중치(c_I) 3건은 위치 규칙 커버리지 밖이라 개별 확인 등록.

## 발견 15 — 교정 필요 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.self_attn.compressor.indexer` |
| 축 | q/k rotate_half 분할 축 (64) -- [B,1,d_head,n_h_I] 형태 3개 앵커, 90축 |
| 현재 라벨 | `n_h_I` |
| 판정 | `should_be_renamed` |
| 제안 라벨 | `c_I/2` |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

modeling_deepseek_v4.py:564-565 (q_b_proj -> view -> apply_rotary_pos_emb) 이 이 텐서가 태어나는 자리다. 위 finding에서 못 가른 마지막 1개 자리. 실측 op 그래프를 직접 추적(prefill op_id 1874/1886/1887): slice가 [B,1,d_head,c_I]를 정확히 반으로 잘라 [B,1,d_head,X]를 만들고 (X=c_I/2=64, rotate_half의 x1=x[...,:dim//2]), 병렬 _to_copy 브랜치가 나머지 반을 처리한 뒤 concat이 둘을 다시 c_I 폭으로 합친다. 이 64라는 값이 n_h_I(index_n_heads)와 d_rope(qk_rope_head_dim) 둘 다와 우연히 같아서 타이 우선순위가 n_h_I를 골랐지만, 실제 계보는 그 축 자신의 부모 c_I를 반으로 나눈 것뿐이다. rules/derived_dims.yaml에 'c_I // 2' 식을 등록하고 rules/label_overrides.yaml에 3개 앵커(slice+concat 2개, spread: class)를 등록했다. review_request.md 0절이 이걸로 완전히 비었다(11건 -> 0건, 6절의 영구 disclosure 10건만 남음).

## 발견 16 — 교정 필요 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.self_attn.compressor.indexer` |
| 축 | q/k partial RoPE 분할 (c_I=128 -> nope(64)+rope(64)) |
| 현재 라벨 | `c_I/2 (양쪽 다)` |
| 판정 | `should_be_renamed` |
| 제안 라벨 | `c_I-d_rope (nope 앞부분) / d_rope (rope 뒷부분)` |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

외부 검토(Codex, 2026-09-01)가 지적: 어제(2026-08-31) 제가 이 자리를 'c_I를 정확히 반으로 나눈 rotate_half 짝'으로 판정하고 c_I/2로 등록했는데, 실제로는 modeling_deepseek_v4.py:342-359 apply_rotary_pos_emb가 `nope = x[...,:-rope_dim], rope = x[...,-rope_dim:]`로 쪼개는 partial RoPE의 NoPE/RoPE 경계다. rope_dim=d_rope. nope 폭은 c_I-d_rope(=128-64=64)이지 c_I/2(=64)가 아니다 -- 이 체크포인트에서 c_I=2*d_rope라 값이 우연히 같았을 뿐(세 번째 우연: n_h_I=64=d_rope=64=c_I-d_rope=64). rotate_half 자체의 진짜 짝(x1=x[...,0::2], x2=x[...,1::2])은 d_rope 안에서 일어나고 d_rope/2=32-폭이라, 이 트레이스가 별도 이름 붙은 축으로 등장하지도 않는다.

**재확인**: op_id 1863(unsqueeze, 전체 c_I) -> 1874(slice, nope=c_I-d_rope) 및 1885(elementwise_add, rope*cos+rotate_half(rope)*sin, 둘 다 d_rope) -> 1886(dtype cast, d_rope) -> 1887(concat[nope,rotated], c_I로 복원)까지 op 그래프 전체를 다시 추적해 확인했다. rules/derived_dims.yaml의 'c_I/2' 식을 'c_I-d_rope'로 교체하고, rules/label_overrides.yaml의 3개 앵커를 8개로 늘려(1885의 두 입력+출력, 1886의 입력+출력 추가) 정확한 값을 등록했다.

## 발견 17 — table_omits_computation (미반영)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.attn_hc / ffn_hc` |
| 축 | mHC Sinkhorn 정규화 |
| 현재 라벨 | `` |
| 판정 | `table_omits_computation` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 미반영 |

**근거**

외부 검토(Codex) 2026-09-16, develop/codex_four_models_review_2026-09-16.md. 표는 softmax → stream collapse 로 보이지만 실제 comb 은 softmax 뒤 epsilon 을 더하고 열 정규화 후 행·열 정규화를 번갈아 반복한다. hc_sinkhorn_iters=20 이면 sum/div 가 39회다. softmax 하나는 이 계산과 같지 않다. modeling_deepseek_v4.py:940-947, 원시 prefill layer0 attn_hc op 157-159.

## 발견 18 — table_omits_computation (미반영)

| 항목 | 값 |
|---|---|
| 모듈 | `self_attn.compressor / indexer` |
| 축 | 압축 가중합·Indexer head 가중합 |
| 현재 라벨 | `` |
| 판정 | `table_omits_computation` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 미반영 |

**근거**

외부 검토(Codex) 2026-09-16, develop/codex_four_models_review_2026-09-16.md. 압축 softmax 와 norm 사이에 확률×KV 곱과 window 축 sum 이 있다. 그래서 요약의 두 행 사이에서 rank 4 → rank 3 변화가 생략된 reduction 때문에 일어난다. Indexer scorer 도 head 가중치 곱 후 n_h_I 축 sum 이 빠졌다. modeling_deepseek_v4.py:414-418,548-550,673-675,455-459.

## 발견 19 — table_omits_computation (미반영)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.mlp.gate` |
| 축 | hash routing vs 동적 routing |
| 현재 라벨 | `matmul` |
| 판정 | `table_omits_computation` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 미반영 |

**근거**

외부 검토(Codex) 2026-09-16, develop/codex_four_models_review_2026-09-16.md. 첫 3층은 frozen tid2eid 테이블 기반 hash routing 이고 이후 층은 score correction bias 를 더한 top-k 다. 양쪽 다 sqrtsoftplus scoring, 선택 score gather, top-k weight 정규화, 2.5 scaling 이 있는데 표는 matmul → experts 로만 보인다. 층별 그룹 분할 자체는 맞다. modeling_deepseek_v4.py:1044-1051,1054-1082,1088-1102.

## 발견 20 — corrected (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.2.self_attn.compressor.indexer` |
| 축 | Indexer 쿼리의 nope 폭 |
| 현재 라벨 | `d_rope` |
| 판정 | `corrected` |
| 제안 라벨 | `c_I-d_rope` |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

외부 검토(Codex) 2026-09-16, develop/codex_four_models_review_2026-09-16.md. prefill op 1927(slice 출력)·1940(cat 첫 입력), decode op 1537·1550. c_I=128, d_rope=64 라 두 조각이 똑같이 64 여서 수치로는 안 드러난다. 트레이스로 확인: 1928 만 회전 사슬(op 1929-1938)로 가고 1927 은 회전 없이 cat 첫 피연산자로 간다. rules/label_overrides.yaml 에 네 항목 등록, 각 30축 발화. modeling_deepseek_v4.py:354-359,497,554-565.

## 발견 21 — table_omits_computation (미반영)

| 항목 | 값 |
|---|---|
| 모듈 | `self_attn` |
| 축 | PV 뒤 inverse RoPE |
| 현재 라벨 | `` |
| 판정 | `table_omits_computation` |
| 제안 라벨 | — |
| 확신도 | medium |
| 산출물 반영 | 미반영 |

**근거**

외부 검토(Codex) 2026-09-16, develop/codex_four_models_review_2026-09-16.md. K=V 라 attention 출력에 inverse RoPE 를 적용하는 단계가 있는데 major-op 표의 PV→o_a 사이에 안 보인다. modeling_deepseek_v4.py:862-868.
