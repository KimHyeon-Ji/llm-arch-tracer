# 미반영 교정 판정 58건 — 설계 자문 요청

소스로 **이름이 틀렸다는 것까지 확정했지만 산출물에 반영하지 못한** 판정이 58건 있다.
전부 `models/*/review_findings.json` 에 `status: open`, `verdict: should_be_renamed` 로 실려 있고
각 항목에 소스 줄이 인용돼 있다. 모르는 것이 아니라 **알지만 안전하게 넣지 못한 것**이다.

이 문서는 그 43건이 무엇인지, 왜 막혔는지, 그리고 **외부 검토자에게 무엇을 묻고 싶은지**를
한 곳에 모은 것이다. 아래 §4 를 그대로 LLM 에 붙여넣으면 된다.

---

## 1. 43건의 정체 — 이름 하나가 아니라 한 가지 병

| 건수 | 교정 | 어디 |
|---|---|---|
| 11 | `d_head` → `d_rope` | MLA 의 q/k split 둘째 조각, `model.rotary_emb` 의 cos/sin (DeepSeek-V3 · V2-Lite · tiny-v3 · Kimi 3종) |
| 6 | `d_nope` → `d_v` | MLA value 경로 (split 둘째 조각 ~ o_proj 입력) |
| 5 | `d_moe` → `d_shared` | Qwen3-Next `mlp.shared_expert.*` |
| 4 | `d_head_lin_k` → `d_head_lin_v` | Qwen3-Next 계열 `linear_attn.norm` |
| 3 | `2*n_kv*d_head` → `key_dim` | Qwen3.5/3.6 `linear_attn.in_proj_qkvz` 조각 |
| 2 | `d_head_ssm`/`n_h_ssm` **순서 뒤바뀜** | Zamba2 Mamba2 chunk-scan |
| 2 | `d_model` → `2*d_moe` | OLMoE · V4-Flash `mlp.experts` 가운데 축 |
| 2 | 정수 `4` → `m_csa` | V4-Flash indexer |
| 2 | `n_h` → `d_rope` | V4-Flash 복소수 되접기 축 |
| 2 | `d_head`/`n_h` → `d_rope`, `n_h_I` → `d_rope/2` | GLM-5.2 DSA indexer |
| 5 | **Qwen 계열 `d_rope` → `d_chunk`** — RoPE 가 없는 gated delta rule 블록이 회전 차원으로 렌더됨 (아래 §2.5) |
| 4 | 개별 (V4-Pro kv_norm 축 순서, `T/m_hca`→`g_o`, Qwen2.5 q_proj 가중치 축, Qwen3-Next 청크 루프 인덱스) | — |

**공통점이 전부다.** 43건 모두 *값으로는 구별할 수 없는 축*이다. 두 config 필드가 이 seq_len 에서
같은 수라(`qk_rope_head_dim == head_dim == 64`, `moe_intermediate_size ==
shared_expert_intermediate_size == 512`, `n_mamba_heads == mamba_headdim == 64`) 규칙이 고를
근거가 없다. 소스를 읽으면 답이 나오고, 그 답을 표에 넣으려는 순간 막힌다.

## 2. 왜 막히는가 — 이름은 모듈이 아니라 **텐서**에 붙는다

교정을 반영하는 층은 `rules/label_overrides.yaml` + `src/label_overrides.py` 다. 이 층은
**한 모듈 경로 아래의 라벨을 치환**한다. 그런데 같은 텐서는 그 모듈의 출력에도, 소비자의
입력에도, 그 소비자의 출력에도 렌더된다. 모듈 하나만 고치면 나머지가 옛 이름으로 남고,
게이트의 데이터플로우 검사(`flow_ambig`)가 정확히 그것을 잡는다.

2026-08-13 에 실제로 측정한 값(11건에 그대로 기록돼 있다):

| 적용해본 교정 | 결과 |
|---|---|
| Qwen3-Next `d_head_lin_k`→`d_head_lin_v` | flow_ambig 108 → 324 |
| Qwen3.5-4B 동일 | 72 → 216 |
| Qwen3.6-27B 동일 | 144 → 432 |
| Qwen3.6-35B 동일 | 90 → 270 |
| DeepSeek-V3 `d_head`→`d_rope` (rotary_emb) | 0 → 244 |
| tiny-deepseek-v3 동일 | 0 → 24 |
| GLM-5.2 indexer | flow_ambig 438 → 480, matmul_compose 0 → 42 |
| DeepSeek-V4-Pro 압축기 축 뒤바뀜 | flow_ambig 366 → 696, reshape_incons 0 → 60 |

**중요**: 이 상승은 라벨이 나빠져서가 아니다. 불일치는 원래부터 있었고 **양쪽이 똑같이
틀린 이름을 쓰고 있어서 보이지 않았을 뿐**이다. 한쪽을 맞추자 드러났다. 그래서 되돌리는 것도
정답이 아니고, 그냥 넣는 것도 정답이 아니다.

Zamba2 2건은 성격이 조금 다르다: `n_h_ssm` 과 `d_head_ssm` 이 **둘 다 64** 라 이름 치환으로는
**맞바꿀 수가 없다**(먼저 도는 치환이 두 축을 하나로 뭉갠다).

## 2.5 새로 추가된 유형 — 지어낸 이름이 반대편에서 이긴다 (2026-08-13)

Qwen3-Next 계열 `linear_attn` 의 64 축은 세 상태를 **전부 측정**했다:

| 상태 | 결과 |
|---|---|
| (A) 현재 | `d_rope` 가 120,513축 차지. 게이트 통과하지만 **그 블록에는 RoPE 가 없다** — 증명 가능하게 틀린 이름 |
| (B) `d_rope` 스코프만 교정 | 빈자리를 휴리스틱이 채움. Qwen3.6-27B heur **4,128 → 84,000** (`4*n_h_lin_k`) |
| (C) 스코프 교정 + `d_chunk` 상수 등록 | 이름은 전부 소스 근거를 얻지만 flow_ambig 상승 (108→288, 135→360, 72→192, 90→240) |

(C) 가 옳다. 막는 것은 반대편 `batched_matmul` 이 들고 있는 **휴리스틱이 지어낸**
`2*n_h_lin_v`(=2·32=64)다 — 한 텐서에 두 이름.

**그래서 §4 에 질문 6번을 추가한다**: 등록된 심볼이 그 값을 설명하는 자리에서 휴리스틱이
이름을 짓지 못하게 하려면 어떻게 해야 하나? 지금은 휴리스틱이 등록 심볼과 대등하게 경쟁하고,
심지어 `_matmul_compose_enforce` 를 타고 반대편으로 퍼진다.

덧붙여, 이 건을 파다가 **주석이 코드에 없는 수정을 주장하는 사례**를 또 찾았다:
`rules/derived_dims.yaml` 의 `round(d_head * pr)` 규칙 주석은 "스코프에서 linear_attn 을 뺀다"고
적어 두었지만 정규식은 `attn|attention|rotary` 그대로였다. `src/anchors.py` 의 `rank1` 앵커와
같은 부류다(그쪽은 2026-08-13 에 살렸다). **주석은 검사를 받지 않는다**는 것이 여기서 두 번
확인됐다 — 질문 7: 이런 "죽은 주장"을 자동으로 잡는 방법이 있나?

## 3. 이미 있는 재료와, 두 번의 실패

| 있는 것 | 무엇을 하는가 |
|---|---|
| `src/anchors.py:propagate()` | 앵커가 정한 라벨을 데이터플로우 따라 **덮어쓰며** 전파. 이미 동작한다 |
| `src/build_table.py:_propagate_labels()` | 단조(monotone) — 정수 자리만 메우고 이름은 절대 안 덮는다 |
| `src/build_table.py:_gather_keeps_features()` | **2026-08-13 성공 사례.** dim-0 게더는 뒤 축 이름을 바꿀 수 없다는 *연산의 정의*로 라벨을 고치고, `depends_on` 으로 소비자까지 전파. gpt-oss 의 flow_ambig 72/48 → 0 |
| `src/label_overrides.py` | 모듈 스코프 치환. `axis`/`rank`/`layer_types` 선택자와 `expect`(실측 크기) 가드 |
| `develop/verify_all.py` | 단일 게이트. EXIT 0 + 퇴행 0 이 합격선 |
| `develop/verify_selftest.py` | 게이트 자체 검증 (결함 주입 27종) |

**실패 기록 두 건** (되풀이하지 말 것, 코드 주석에 남아 있다):

1. `_carry_reshape_labels` — reshape 양단 라벨을 맞추려 했다. 그 view 자체는 고쳐지지만
   소비자가 옛 이름을 유지해 flow_ambig 이 함대 전체에서 두 배(Llama-3.1-70B 160→400,
   405B 504→882). **호출하지 않는 상태로 남아 있다**(감사용으로만 유지).
2. MLA 라벨을 규칙 우선순위로 바꾸기 — reshape_incons 61→122, flow_ambig 0→122.

두 번 다 같은 이유로 실패했다: **한 지점만 고쳤다.**

## 4. 검토 LLM에게 붙여넣을 질문

> **이 파일 하나면 된다.** 저장소를 통째로 전달할 수 없는 상황을 위해, 판단에 필요한
> 43건 전문(부록 A)과 코드 원문(부록 B)을 아래에 그대로 붙여 두었다. 아래 질문만
> 복사하지 말고 **이 md 파일 전체를 통째로** 넘기면 된다.

---

당신은 `llm-arch-tracer` 의 **설계 문제 하나**에 대해 자문한다. 코드를 고치라는 것이 아니라,
**어떤 메커니즘이 옳은지** 판단해 달라는 것이다.

**배경**: 이 도구는 HuggingFace 모델을 meta device 에서 트레이스해 연산마다 텐서 shape 을
기록하고 각 축에 `d_model`, `n_h`, `d_head` 같은 이름을 붙인다. 규칙이 이름을 붙이고,
규칙 게이트(`develop/verify_all.py`)가 자기 일관성을 검사한다.

**문제**: 두 config 필드가 같은 값을 가지면 규칙에는 고를 근거가 없다. 소스를 읽으면
어느 쪽인지 확정되지만, 그 답을 표에 넣는 층(`src/label_overrides.py`)이 **모듈 단위 치환**
이라 같은 텐서를 렌더하는 이웃 모듈이 옛 이름으로 남는다. 그러면 데이터플로우 일관성 검사가
걸린다. 이렇게 막힌 판정이 **43건**이다.

**먼저 읽을 것 — 전부 이 파일 안에 있다**:
1. 위 §1~§3 — 43건이 무엇이고, 왜 막혔고, 무엇을 두 번 실패했는지
2. **부록 A** — 43건 전문 (모듈 · 축 · 현재→제안 · 근거 소스 · 막힌 측정치)
3. **부록 B** — 판단에 필요한 코드 원문. 주석에 측정치와 실패 기록이 들어 있으니 **주석까지 읽어라**
   - B1 `src/label_overrides.py` 전문 — 지금 막혀 있는 반영 층
   - B2 `_gather_keeps_features()` — 최근 성공한 방식 (일반화 후보)
   - B3 `_carry_reshape_labels()` — 같은 목표로 시도했다가 되돌린 것
   - B4 `_propagate_labels()` — 단조 전파 (이름을 안 덮는다)
   - B5 `anchors.propagate()` — 이름을 덮어쓰는 전파 (이미 동작 중)
   - B6 `_dataflow_label_check()` — 우리를 막는 `flow_ambig` 이 정확히 무엇을 세는지

**받았다면 추가로 볼 것(없어도 답할 수 있다)**: `models/<모델>/` 폴더의 `full/*.trace.raw.jsonl`
(라벨이 붙은 실제 행), `full/*.shapes.concrete.jsonl`(같은 행의 실측 정수), `structure.yaml`
(심볼 표), `review_findings.json`(판정 원본). 가장 유용한 것은 `deepseek-ai__DeepSeek-V4-Pro`,
`Zyphra__Zamba2-1.2B`, `Qwen__Qwen3-Next-80B-A3B-Instruct`, `openai__gpt-oss-120b` 넷이다 —
앞의 셋이 막힌 세 가지 유형이고, 마지막이 유일하게 풀린 사례다.

**답해 주었으면 하는 것**:

1. **`_gather_keeps_features` 방식(연산의 정의 + `depends_on` 전파)을 일반화하는 것이
   맞는 길인가?** 맞다면 어떤 연산 부류까지 확장 가능한가 — transpose/view/slice/concat/cat
   /elementwise 는 각각 "이름을 바꿀 수 없다"가 참인가, 예외는 무엇인가?
2. 아니라면 **대안**은? 예컨대 (a) 텐서 신원(tensor id)별로 이름을 하나만 두는 정규화 층,
   (b) 소스 판정을 *심볼 우선순위*가 아니라 *텐서 단위 고정점*으로 넣는 방법,
   (c) 전파 순서를 위상정렬로 강제하는 방법. 각각의 함정은?
3. **Zamba2 케이스**를 어떻게 표현하나 — `n_h_ssm` 과 `d_head_ssm` 이 둘 다 64 라 치환으로는
   맞바꿀 수 없다. 소스는 `num_heads` 가 항상 `head_dim` 앞이라고 말한다
   (`modeling_zamba2.py:832, :622, :524, :527`). 이 **순서 규칙**을 안전하게 표현하는 방법은?
4. **flow_ambig 상승을 어떻게 해석해야 하나?** 우리는 "원래 있던 불일치가 드러난 것"이라
   보고 있는데, 이 해석 자체가 틀렸을 가능성을 반박해 달라. 만약 맞다면 게이트의 baseline
   퇴행 검사를 어떻게 바꿔야 이런 "드러남"과 진짜 퇴행을 구별할 수 있나?
5. 58건 중 **먼저 손대야 할 순서**는? (한 메커니즘으로 몇 건이 풀리는지 기준)
6. **휴리스틱이 등록 심볼과 대등하게 경쟁하는 것**이 맞나? (§2.5) 등록된 이름이 있는 자리에서 지어낸 이름이 이기고, 심지어 matmul 합성을 타고 퍼진다.
7. **주석이 코드에 없는 수정을 주장하는 것**을 자동으로 잡는 방법이 있나? 이 저장소에서 두 번 나왔다(죽은 `rank1` 앵커, 적용 안 된 linear_attn 스코프 제외).

**지켜야 할 제약**:
- `develop/verify_all.py` 가 **EXIT 0, 퇴행 0** 이어야 한다. 이것이 합격선이다.
- 값으로 우기지 않는다. 근거 없는 이름은 붙이지 않고, 모르면 `undetermined` 로 남긴다.
- 새 검사를 추가하면 `develop/verify_selftest.py` 에 결함 주입을 함께 넣어 **그 검사가 살아
  있음을 증명**해야 한다. (실제로 이 프로젝트에서 검사 두 개가 조용히 죽어 있었다.)
- FLOPs · 메모리 대역폭 · latency 는 범위 밖이다.

**자세**: 확인하지 말고 **반박하라.** 위 §2 의 진단("모듈 단위 치환이 원인")이 틀렸다는 증거를
먼저 찾아라. 원인이 다른 곳에 있다면 그것을 말해 달라 — 우리가 두 번 실패한 것은 진단이
틀렸기 때문일 수도 있다.

---

# 부록 A — 43건 전문

저장소를 통째로 전달할 수 없을 때를 위해, `models/*/review_findings.json` 의 해당 항목을 여기에 펼쳐 둔다. 모델 폴더를 못 받았어도 이 표만으로 판단할 수 있다.

### A1. Qwen__Qwen2.5-0.5B

- **모듈**: `model.layers.*.self_attn.q_proj`
- **축**: 가중치 축 이름
- **지금 → 제안**: `weight_shape=[n_h*d_head, n_h*d_head] / 피연산자=[d_model, d_model]` → `[n_h*d_head, d_model]`  (확신 high)
- **근거**: `Qwen2Attention.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim)` — out=n_h·d_head, in=d_model 이다. 그런데 **한 파라미터가 두 이름으로 나온다**: weight_shape 열은 `[n_h*d_head, n_h*d_head]`, 같은 파라미터가 `t`/`linear` 의 피연산자로 나올 때는 `[d_model, d_model]`. 어느 쪽도 `[out, in]` 이 아니다. **원인 진단**: `build_table._canonical_weight_labels` 가 앵커 적용 **전에** 계산된다 — `_contraction_pin` 이 활성화의 마지막 축 라벨을 가져오는데, `self_attn` 스코프 안에서 896 은 `n_h*d_head`(스코프 있는 유도식)로 먼저 해석되어 d_model 을 이긴다. 그 값이 가중치의 in 축에 박히고 canon 으로 굳는다. 고치려면 canon 을 앵커의 선언 in/out 으로 계산해야 한다 — 이번 회차에서는 진단까지만 하고 남긴다.  **근거 소스**: 이 판정은 `develop/sources/modeling_qwen2.py`, `develop/sources/configuration_qwen2.py` 를 열어 확인했다. (인용 누락을 자가 점검에서 발견해 보강, 2026-08-12 — 

### A2. Qwen__Qwen3-Next-80B-A3B-Instruct

- **모듈**: `model.layers.*.linear_attn`
- **축**: 청크 루프 인덱스 축 (elementwise op 의 입력 쪽)
- **지금 → 제안**: `n_kv / d_conv_lin / 3*n_kv / n_h/n_kv / k (입력) vs 정수 (출력)` → `정수`  (확신 high)
- **근거**: `build_table._unname_loop_indices` 가 청크 스캔의 루프 인덱스에서 지어낸 이름을 떼어내는데, 그 결과가 **출력 쪽에만** 남아 있었다. 새 elementwise 검사가 1,008행을 잡았다: `elementwise_add([B, n_h_lin_v, 1, n_kv], ...) -> [B, n_h_lin_v, 1, 2]` — 실제값 [1,32,1,2] 로 같은 텐서인데 들어갈 때는 `n_kv`, 나올 때는 `2` 다. 탐지 키에서 shape_index 와 field 를 빼 범위를 넓혔지만(2026-08-10) 남는다: 제거 패스가 `_propagate_labels` **앞**에 있어야 하는데(뒤로 옮기면 이웃 op 들이 옛 이름을 유지해 데이터플로우 불일치가 43,000행으로 폭증한다), 그 전파가 monotone 이라 비운 정수를 이웃에서 다시 채운다. 제대로 고치려면 그 텐서를 만지는 **모든** op 을 함께 비워야 하고, 값 기반 일괄 제거는 안전하지 않다 — `linear_attn` 안에서 4 는 루프 계단이면서 동시에 진짜 `d_conv_lin` 이다. 값으로 우기지 않고 남긴다.  **근거 소스**: 이 판정은 `develop/sources/modeling_qwen3_next.py`, `develop/sources/configuration_qwen3_next.py` 를 열어 확인했다. (인용 누

### A3. Qwen__Qwen3-Next-80B-A3B-Instruct

- **모듈**: `model.layers.*.linear_attn`
- **축**: q/k 조각 폭 (2048 = key_dim = d_model)
- **지금 → 제안**: `d_model` → `n_k*d_k`  (확신 high)
- **근거**: `modeling_qwen3_next.py:520` `key_dim = head_k_dim * num_k_heads` = 16·128 = 2048 인데 이 모델은 hidden_size 도 2048 이다. `n_k*d_k` 규칙을 등록했더니 이번엔 **linear_attn 으로 들어오는 잔차 스트림**까지 그 이름을 가져가, 레이어 루트가 d_model 이라 부르는 바로 그 텐서를 한 칸 안에서 다르게 부르게 됐다(flow_ambig 0→72). `unless_equals: [d_model]` 로 물러나게 했다 — 조각 이름 하나를 잃더라도 모델에서 가장 근본적인 축을 지키는 쪽을 택했다. **남은 것**: 이 seq_len·이 체크포인트에서 두 값이 같은 한, 트레이스 안에 둘을 가를 증거가 없다. key_dim ≠ hidden_size 인 다른 체크포인트를 추적하면 규칙이 그대로 작동한다.  **재분류 (2026-08-13)**: 이 판정은 `undetermined` 였다. 잘못된 분류다 — 근거 문장이 "트레이스 안에 가를 증거가 없다"고 적고 있었는데, 그건 *트레이스만으로는* 못 가른다는 말이지 *알 수 없다*는 말이 아니다. **소스는 답을 갖고 있다**(위 인용). 막는 것은 지식이 아니라 표현 수단이다: 두 이름이 같은 값이라 `label_overrides` 의 이름 치환으로는 갈 수 없고, 필요한 것은 권위 있는 이름을

### A4. Qwen__Qwen3-Next-80B-A3B-Instruct

- **모듈**: `model.layers.*.linear_attn`
- **축**: matmul 수축 축 (128)
- **지금 → 제안**: `d_head_lin_k / d_head_lin_v 혼용` → `(소스가 가리키는 쪽 — 근거 참조)`  (확신 medium)
- **근거**: `linear_key_head_dim == linear_value_head_dim == 128` 이라 수축 축의 두 끝이 서로 다른 이름을 달고 있다(행렬곱 합성 불일치 108건). 둘 다 소스에 있는 진짜 이름이고 이 체크포인트에서 값이 같을 뿐이라 **어느 쪽이 틀렸다고 말할 수 없다**. 두 값이 다른 체크포인트를 추적하기 전에는 결론을 낼 근거가 없다.  **근거 소스**: 이 판정은 `develop/sources/modeling_qwen3_next.py`, `develop/sources/configuration_qwen3_next.py` 를 열어 확인했다. (인용 누락을 자가 점검에서 발견해 보강, 2026-08-12 — 게이트가 이제 `should_be_renamed` 판정에 소스 인용을 요구한다.)  **재분류 (2026-08-13)**: 이 판정은 `undetermined` 였다. 잘못된 분류다 — 근거 문장이 "트레이스 안에 가를 증거가 없다"고 적고 있었는데, 그건 *트레이스만으로는* 못 가른다는 말이지 *알 수 없다*는 말이 아니다. **소스는 답을 갖고 있다**(위 인용). 막는 것은 지식이 아니라 표현 수단이다: 두 이름이 같은 값이라 `label_overrides` 의 이름 치환으로는 갈 수 없고, 필요한 것은 권위 있는 이름을 데이터플로우 따라 끌고 가는 메커니즘이다. `review/06-open-renam

### A5. Qwen__Qwen3-Next-80B-A3B-Instruct

- **모듈**: `model.layers.*.linear_attn`
- **축**: d_head_lin_k vs d_head_lin_v (128)
- **지금 → 제안**: `(값 동률)` → `판정 불가`  (확신 high)
- **근거**: `linear_key_head_dim == linear_value_head_dim == 128` 이다. 소스는 둘을 구별하지만(`torch.split(mixed_qkv, [key_dim, key_dim, value_dim])` 뒤 각각 `head_k_dim`/`head_v_dim` 으로 reshape) **이 체크포인트에서는 값이 같아 트레이스 안에 가를 증거가 없다.** 모듈도 shape 도 같고 갈리는 건 split 의 몇 번째 조각이냐뿐이다.  값으로 우기지 않고 남긴다. 두 값이 다른 체크포인트를 추적하면 규칙이 그대로 작동한다.  **근거 소스**: 이 판정은 `develop/sources/modeling_qwen3_next.py`, `develop/sources/configuration_qwen3_next.py` 를 열어 확인했다. (인용 누락을 자가 점검에서 발견해 보강, 2026-08-12 — 게이트가 이제 `should_be_renamed` 판정에 소스 인용을 요구한다.)  **재분류 (2026-08-13)**: 이 판정은 `undetermined` 였다. 잘못된 분류다 — 근거 문장이 "트레이스 안에 가를 증거가 없다"고 적고 있었는데, 그건 *트레이스만으로는* 못 가른다는 말이지 *알 수 없다*는 말이 아니다. **소스는 답을 갖고 있다**(위 인용). 막는 것은 지식이 아니라 표현 수단이다: 두 이름이 같은 값이

### A6. Qwen__Qwen3-Next-80B-A3B-Instruct

- **모듈**: `model.layers.*.linear_attn.norm`
- **축**: 정규화 폭 128
- **지금 → 제안**: `d_head_lin_k` → `d_head_lin_v`  (확신 high)
- **근거**: `modeling_qwen3_next.py:552` `self.norm = Qwen3NextRMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)` 이고 `:519` `self.head_v_dim = config.linear_value_head_dim` 다. 이 norm 의 폭은 **value** head dim 이다. linear_key_head_dim 과 linear_value_head_dim 이 둘 다 128 이라 값으로는 구별할 수 없었다. 같은 행의 앞 축이 이미 `n_h_lin_v*T` 로 렌더되고 있어 한 텐서 안에서도 앞뒤가 어긋나 있었다(`[n_h_lin_v*T, d_head_lin_k]`, 실측 `[544, 128]`).  **
- **막힌 이유(측정)**: 아직 반영하지 않은 이유(측정)**: 이 이름을 `rules/label_overrides.yaml` 로 적용해 봤더니 게이트 퇴행 검사가 걸렸다 — flow_ambig 108 -> 324. override 층은 **한 모듈 안의** 이름만 바꾸므로, 같은 텐서를 렌더하는 이웃 모듈이 옛 이름으로 남아 데이터플로우 불일치가 드러난다. 이름이 틀렸다는 판정 자체는 위 소스로 확정이고, 필요한 것은 '권위 있는 이름을 데이터플로우 따라 끌고 가는' 별도 메커니즘이다.

### A7. Qwen__Qwen3-Next-80B-A3B-Instruct

- **모듈**: `model.layers.*.mlp.shared_expert.gate_proj`
- **축**: FFN 폭 512
- **지금 → 제안**: `d_moe` → `d_shared`  (확신 high)
- **근거**: `modeling_qwen3_next.py:783` `self.shared_expert = Qwen3NextMLP(config, intermediate_size=config.shared_expert_intermediate_size)` — 공유 전문가의 폭은 `shared_expert_intermediate_size` 이지 `moe_intermediate_size`(=`d_moe`)가 아니다. `configuration_qwen3_next.py:115-116` 에서 둘 다 512 라 값으로는 구별되지 않고, `:118 num_experts=512` 까지 같은 값이라 `E` 도 후보로 올라와 있었다. 셋 중 이 모듈이 실제로 읽는 필드는 하나뿐이다.  **
- **막힌 이유(측정)**: 아직 반영하지 않은 이유(측정)**: 이 이름을 `rules/label_overrides.yaml` 로 적용해 봤더니 게이트 퇴행 검사가 걸렸다 — flow_ambig 108 -> 324 (d_shared 별칭 등록과 함께 되돌림). override 층은 **한 모듈 안의** 이름만 바꾸므로, 같은 텐서를 렌더하는 이웃 모듈이 옛 이름으로 남아 데이터플로우 불일치가 드러난다. 이름이 틀렸다는 판정 자체는 위 소스로 확정이고, 필요한 것은 '권위 있는 이름을 데이

### A8. Qwen__Qwen3-Next-80B-A3B-Instruct

- **모듈**: `model.layers.*.mlp.shared_expert.up_proj`
- **축**: FFN 폭 512
- **지금 → 제안**: `d_moe` → `d_shared`  (확신 high)
- **근거**: `modeling_qwen3_next.py:783` `self.shared_expert = Qwen3NextMLP(config, intermediate_size=config.shared_expert_intermediate_size)` — 공유 전문가의 폭은 `shared_expert_intermediate_size` 이지 `moe_intermediate_size`(=`d_moe`)가 아니다. `configuration_qwen3_next.py:115-116` 에서 둘 다 512 라 값으로는 구별되지 않고, `:118 num_experts=512` 까지 같은 값이라 `E` 도 후보로 올라와 있었다. 셋 중 이 모듈이 실제로 읽는 필드는 하나뿐이다.

### A9. Qwen__Qwen3-Next-80B-A3B-Instruct

- **모듈**: `model.layers.*.mlp.shared_expert.down_proj`
- **축**: FFN 폭 512
- **지금 → 제안**: `d_moe` → `d_shared`  (확신 high)
- **근거**: `modeling_qwen3_next.py:783` `self.shared_expert = Qwen3NextMLP(config, intermediate_size=config.shared_expert_intermediate_size)` — 공유 전문가의 폭은 `shared_expert_intermediate_size` 이지 `moe_intermediate_size`(=`d_moe`)가 아니다. `configuration_qwen3_next.py:115-116` 에서 둘 다 512 라 값으로는 구별되지 않고, `:118 num_experts=512` 까지 같은 값이라 `E` 도 후보로 올라와 있었다. 셋 중 이 모듈이 실제로 읽는 필드는 하나뿐이다.

### A10. Qwen__Qwen3-Next-80B-A3B-Instruct

- **모듈**: `model.layers.*.mlp.shared_expert`
- **축**: FFN 폭 512
- **지금 → 제안**: `d_moe` → `d_shared`  (확신 high)
- **근거**: `modeling_qwen3_next.py:783` `self.shared_expert = Qwen3NextMLP(config, intermediate_size=config.shared_expert_intermediate_size)` — 공유 전문가의 폭은 `shared_expert_intermediate_size` 이지 `moe_intermediate_size`(=`d_moe`)가 아니다. `configuration_qwen3_next.py:115-116` 에서 둘 다 512 라 값으로는 구별되지 않고, `:118 num_experts=512` 까지 같은 값이라 `E` 도 후보로 올라와 있었다. 셋 중 이 모듈이 실제로 읽는 필드는 하나뿐이다.

### A11. Qwen__Qwen3-Next-80B-A3B-Instruct

- **모듈**: `model.layers.*.mlp.shared_expert.act_fn`
- **축**: FFN 폭 512
- **지금 → 제안**: `d_moe` → `d_shared`  (확신 high)
- **근거**: `modeling_qwen3_next.py:783` `self.shared_expert = Qwen3NextMLP(config, intermediate_size=config.shared_expert_intermediate_size)` — 공유 전문가의 폭은 `shared_expert_intermediate_size` 이지 `moe_intermediate_size`(=`d_moe`)가 아니다. `configuration_qwen3_next.py:115-116` 에서 둘 다 512 라 값으로는 구별되지 않고, `:118 num_experts=512` 까지 같은 값이라 `E` 도 후보로 올라와 있었다. 셋 중 이 모듈이 실제로 읽는 필드는 하나뿐이다.

### A12. Qwen__Qwen3-Next-80B-A3B-Instruct

- **모듈**: `model.layers.*.linear_attn`
- **축**: gated delta rule 청크 길이 64 (chunk_size)
- **지금 → 제안**: `d_rope` → `d_chunk`  (확신 high)
- **근거**: `modeling_qwen3_next.py:381` `def torch_chunk_gated_delta_rule(..., chunk_size=64)` — 청크 길이가 **config 필드가 아니라 커널 fallback 의 기본 인자**다. 같은 리터럴이 `modeling_qwen3_5.py` / `modeling_qwen3_5_moe.py` 에도 있다. 심볼 표는 config 를 읽으므로 코드에만 있는 상수는 구조적으로 유도할 수 없고, 그래서 이 폭이 이름을 못 받거나 엉뚱한 이름을 받는다.  **현재 라벨이 틀렸다는 증거**: 이 블록(`Qwen3NextGatedDeltaNet`)에는 **RoPE 가 아예 없다** — RoPE 는 같은 스택의 `self_attn` 레이어에만 있다. 그런데 `rules/derived_dims.yaml` 의 `round(d_head * pr)` 규칙이 scope `attn|attention|rotary` 로 걸려 있고 `attn` 은 `linear_attn` 안에서도 매치한다. partial_rotary_factor(0.25) x head_dim(256) = 64 이고 chunk_size 도 64 라, 청크 스캔의 `[chunk, chunk]` triu 마스크가 통째로 `d_rope` 로 렌더되고 있다 — Qwen3.6-27B 한 모델에서만 31,440축(2026-08-13 측정).  그 규칙의 주석
- **막힌 이유(측정)**: 아직 반영하지 않은 이유**: (C) 가 옳지만 게이트 퇴행 검사가 막는다. 필요한 것은 '등록된 심볼이 그 값을 설명하는 자리에서는 휴리스틱이 이름을 짓지 않는다'는 규칙, 또는 권위 있는 이름을 데이터플로우 따라 끌고 가는 메커니즘이다. 43건과 같은 병이며 `review/06-open-renames.md` 의 자문 대상이다. 한쪽만 고치는 수정은 하지 않는다.

### A13. Qwen__Qwen3.5-397B-A17B

- **모듈**: `model.layers.*.linear_attn`
- **축**: d_head_lin_k vs d_head_lin_v (128)
- **지금 → 제안**: `(값 동률)` → `판정 불가`  (확신 high)
- **근거**: `linear_key_head_dim == linear_value_head_dim == 128` 이다. 소스는 둘을 구별하지만(`torch.split(mixed_qkv, [key_dim, key_dim, value_dim])` 뒤 각각 `head_k_dim`/`head_v_dim` 으로 reshape) **이 체크포인트에서는 값이 같아 트레이스 안에 가를 증거가 없다.** 모듈도 shape 도 같고 갈리는 건 split 의 몇 번째 조각이냐뿐이다.  값으로 우기지 않고 남긴다. 두 값이 다른 체크포인트를 추적하면 규칙이 그대로 작동한다.  **근거 소스**: 이 판정은 `develop/sources/modeling_qwen3_5_moe.py`, `develop/sources/configuration_qwen3_5_moe.py` 를 열어 확인했다. (인용 누락을 자가 점검에서 발견해 보강, 2026-08-12 — 게이트가 이제 `should_be_renamed` 판정에 소스 인용을 요구한다.)  **재분류 (2026-08-13)**: 이 판정은 `undetermined` 였다. 잘못된 분류다 — 근거 문장이 "트레이스 안에 가를 증거가 없다"고 적고 있었는데, 그건 *트레이스만으로는* 못 가른다는 말이지 *알 수 없다*는 말이 아니다. **소스는 답을 갖고 있다**(위 인용). 막는 것은 지식이 아니라 표현 수단이다: 두 이름이 같은 

### A14. Qwen__Qwen3.5-397B-A17B

- **모듈**: `model.layers.*.linear_attn.norm`
- **축**: d_head_lin_k vs d_head_lin_v (128, norm 쪽)
- **지금 → 제안**: `(값 동률)` → `판정 불가`  (확신 high)
- **근거**: 위 `linear_attn` 건과 같은 충돌이 norm 모듈에도 나타난다. 원인·근거 동일하다.  **근거 소스**: 이 판정은 `develop/sources/modeling_qwen3_5_moe.py`, `develop/sources/configuration_qwen3_5_moe.py` 를 열어 확인했다. (인용 누락을 자가 점검에서 발견해 보강, 2026-08-12 — 게이트가 이제 `should_be_renamed` 판정에 소스 인용을 요구한다.)  **재분류 (2026-08-13)**: 이 판정은 `undetermined` 였다. 잘못된 분류다 — 근거 문장이 "트레이스 안에 가를 증거가 없다"고 적고 있었는데, 그건 *트레이스만으로는* 못 가른다는 말이지 *알 수 없다*는 말이 아니다. **소스는 답을 갖고 있다**(위 인용). 막는 것은 지식이 아니라 표현 수단이다: 두 이름이 같은 값이라 `label_overrides` 의 이름 치환으로는 갈 수 없고, 필요한 것은 권위 있는 이름을 데이터플로우 따라 끌고 가는 메커니즘이다. `review/06-open-renames.md` 의 같은 병이므로 그쪽으로 합친다. **모르는 것과 못 넣는 것은 다르게 적는다.**

### A15. Qwen__Qwen3.5-397B-A17B

- **모듈**: `model.layers.*.linear_attn`
- **축**: gated delta rule 청크 길이 64 (chunk_size)
- **지금 → 제안**: `d_rope` → `d_chunk`  (확신 high)
- **근거**: `modeling_qwen3_next.py:381` `def torch_chunk_gated_delta_rule(..., chunk_size=64)` — 청크 길이가 **config 필드가 아니라 커널 fallback 의 기본 인자**다. 같은 리터럴이 `modeling_qwen3_5.py` / `modeling_qwen3_5_moe.py` 에도 있다. 심볼 표는 config 를 읽으므로 코드에만 있는 상수는 구조적으로 유도할 수 없고, 그래서 이 폭이 이름을 못 받거나 엉뚱한 이름을 받는다.  **현재 라벨이 틀렸다는 증거**: 이 블록(`Qwen3NextGatedDeltaNet`)에는 **RoPE 가 아예 없다** — RoPE 는 같은 스택의 `self_attn` 레이어에만 있다. 그런데 `rules/derived_dims.yaml` 의 `round(d_head * pr)` 규칙이 scope `attn|attention|rotary` 로 걸려 있고 `attn` 은 `linear_attn` 안에서도 매치한다. partial_rotary_factor(0.25) x head_dim(256) = 64 이고 chunk_size 도 64 라, 청크 스캔의 `[chunk, chunk]` triu 마스크가 통째로 `d_rope` 로 렌더되고 있다 — Qwen3.6-27B 한 모델에서만 31,440축(2026-08-13 측정).  그 규칙의 주석
- **막힌 이유(측정)**: 아직 반영하지 않은 이유**: (C) 가 옳지만 게이트 퇴행 검사가 막는다. 필요한 것은 '등록된 심볼이 그 값을 설명하는 자리에서는 휴리스틱이 이름을 짓지 않는다'는 규칙, 또는 권위 있는 이름을 데이터플로우 따라 끌고 가는 메커니즘이다. 43건과 같은 병이며 `review/06-open-renames.md` 의 자문 대상이다. 한쪽만 고치는 수정은 하지 않는다.

### A16. Qwen__Qwen3.5-4B

- **모듈**: `model.layers.*.linear_attn`
- **축**: in_proj_qkvz 조각 폭 (27B 에서 2048)
- **지금 → 제안**: `2*n_kv*d_head` → `key_dim (= n_h_lin_k · d_head_lin_k)`  (확신 high)
- **근거**: `modeling_qwen3_5.py:520-521` `self.key_dim = self.head_k_dim * self.num_k_heads` / `self.value_dim = self.head_v_dim * self.num_v_heads`. `split_with_sizes` 가 [key, key, value] 로 쪼개는 것이 트레이스에 그대로 보인다(실측 [2048, 2048, 6144]). 어텐션 head 수와 무관한 축인데 2·n_kv·d_head 와 값이 같아 그쪽으로 붙었다 — 확인된 오라벨. `n_h_lin_k * d_head_lin_k` 로 등록해봤으나 Qwen3-Next 의 flow_ambig 가 0 -> 72 로 퇴행해 보류했다(2026-08-10): 새 이름이 붙은 축의 하류 소비자가 옛 이름을 그대로 들고 있어 한 텐서가 두 이름을 갖는다. 라벨이 아니라 전파 쪽 과제다.

### A17. Qwen__Qwen3.5-4B

- **모듈**: `model.layers.*.linear_attn`
- **축**: matmul 수축 축 (128)
- **지금 → 제안**: `d_head_lin_k / d_head_lin_v 혼용` → `(소스가 가리키는 쪽 — 근거 참조)`  (확신 medium)
- **근거**: `linear_key_head_dim == linear_value_head_dim == 128` 이라 수축 축의 두 끝이 서로 다른 이름을 달고 있다(행렬곱 합성 불일치 108건). 둘 다 소스에 있는 진짜 이름이고 이 체크포인트에서 값이 같을 뿐이라 **어느 쪽이 틀렸다고 말할 수 없다**. 두 값이 다른 체크포인트를 추적하기 전에는 결론을 낼 근거가 없다.  **근거 소스**: 이 판정은 `develop/sources/modeling_qwen3_5.py`, `develop/sources/configuration_qwen3_5.py` 를 열어 확인했다. (인용 누락을 자가 점검에서 발견해 보강, 2026-08-12 — 게이트가 이제 `should_be_renamed` 판정에 소스 인용을 요구한다.)  **재분류 (2026-08-13)**: 이 판정은 `undetermined` 였다. 잘못된 분류다 — 근거 문장이 "트레이스 안에 가를 증거가 없다"고 적고 있었는데, 그건 *트레이스만으로는* 못 가른다는 말이지 *알 수 없다*는 말이 아니다. **소스는 답을 갖고 있다**(위 인용). 막는 것은 지식이 아니라 표현 수단이다: 두 이름이 같은 값이라 `label_overrides` 의 이름 치환으로는 갈 수 없고, 필요한 것은 권위 있는 이름을 데이터플로우 따라 끌고 가는 메커니즘이다. `review/06-open-renames.md`

### A18. Qwen__Qwen3.5-4B

- **모듈**: `model.layers.*.linear_attn.norm`
- **축**: 정규화 폭 128
- **지금 → 제안**: `d_head_lin_k` → `d_head_lin_v`  (확신 high)
- **근거**: `modeling_qwen3_next.py:552` `self.norm = Qwen3NextRMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)` 이고 `:519` `self.head_v_dim = config.linear_value_head_dim` 다. 이 norm 의 폭은 **value** head dim 이다. linear_key_head_dim 과 linear_value_head_dim 이 둘 다 128 이라 값으로는 구별할 수 없었다. 같은 행의 앞 축이 이미 `n_h_lin_v*T` 로 렌더되고 있어 한 텐서 안에서도 앞뒤가 어긋나 있었다(`[n_h_lin_v*T, d_head_lin_k]`, 실측 `[544, 128]`).  **
- **막힌 이유(측정)**: 아직 반영하지 않은 이유(측정)**: 이 이름을 `rules/label_overrides.yaml` 로 적용해 봤더니 게이트 퇴행 검사가 걸렸다 — flow_ambig 72 -> 216. override 층은 **한 모듈 안의** 이름만 바꾸므로, 같은 텐서를 렌더하는 이웃 모듈이 옛 이름으로 남아 데이터플로우 불일치가 드러난다. 이름이 틀렸다는 판정 자체는 위 소스로 확정이고, 필요한 것은 '권위 있는 이름을 데이터플로우 따라 끌고 가는' 별도 메커니즘이다.

### A19. Qwen__Qwen3.5-4B

- **모듈**: `model.layers.*.linear_attn`
- **축**: gated delta rule 청크 길이 64 (chunk_size)
- **지금 → 제안**: `d_rope` → `d_chunk`  (확신 high)
- **근거**: `modeling_qwen3_next.py:381` `def torch_chunk_gated_delta_rule(..., chunk_size=64)` — 청크 길이가 **config 필드가 아니라 커널 fallback 의 기본 인자**다. 같은 리터럴이 `modeling_qwen3_5.py` / `modeling_qwen3_5_moe.py` 에도 있다. 심볼 표는 config 를 읽으므로 코드에만 있는 상수는 구조적으로 유도할 수 없고, 그래서 이 폭이 이름을 못 받거나 엉뚱한 이름을 받는다.  **현재 라벨이 틀렸다는 증거**: 이 블록(`Qwen3NextGatedDeltaNet`)에는 **RoPE 가 아예 없다** — RoPE 는 같은 스택의 `self_attn` 레이어에만 있다. 그런데 `rules/derived_dims.yaml` 의 `round(d_head * pr)` 규칙이 scope `attn|attention|rotary` 로 걸려 있고 `attn` 은 `linear_attn` 안에서도 매치한다. partial_rotary_factor(0.25) x head_dim(256) = 64 이고 chunk_size 도 64 라, 청크 스캔의 `[chunk, chunk]` triu 마스크가 통째로 `d_rope` 로 렌더되고 있다 — Qwen3.6-27B 한 모델에서만 31,440축(2026-08-13 측정).  그 규칙의 주석
- **막힌 이유(측정)**: 아직 반영하지 않은 이유**: (C) 가 옳지만 게이트 퇴행 검사가 막는다. 필요한 것은 '등록된 심볼이 그 값을 설명하는 자리에서는 휴리스틱이 이름을 짓지 않는다'는 규칙, 또는 권위 있는 이름을 데이터플로우 따라 끌고 가는 메커니즘이다. 43건과 같은 병이며 `review/06-open-renames.md` 의 자문 대상이다. 한쪽만 고치는 수정은 하지 않는다.

### A20. Qwen__Qwen3.6-27B

- **모듈**: `model.layers.*.linear_attn`
- **축**: in_proj_qkvz 조각 폭 (27B 에서 2048)
- **지금 → 제안**: `2*n_kv*d_head` → `key_dim (= n_h_lin_k · d_head_lin_k)`  (확신 high)
- **근거**: `modeling_qwen3_5.py:520-521` `self.key_dim = self.head_k_dim * self.num_k_heads` / `self.value_dim = self.head_v_dim * self.num_v_heads`. `split_with_sizes` 가 [key, key, value] 로 쪼개는 것이 트레이스에 그대로 보인다(실측 [2048, 2048, 6144]). 어텐션 head 수와 무관한 축인데 2·n_kv·d_head 와 값이 같아 그쪽으로 붙었다 — 확인된 오라벨. `n_h_lin_k * d_head_lin_k` 로 등록해봤으나 Qwen3-Next 의 flow_ambig 가 0 -> 72 로 퇴행해 보류했다(2026-08-10): 새 이름이 붙은 축의 하류 소비자가 옛 이름을 그대로 들고 있어 한 텐서가 두 이름을 갖는다. 라벨이 아니라 전파 쪽 과제다.

### A21. Qwen__Qwen3.6-27B

- **모듈**: `model.layers.*.linear_attn`
- **축**: matmul 수축 축 (128)
- **지금 → 제안**: `d_head_lin_k / d_head_lin_v 혼용` → `(소스가 가리키는 쪽 — 근거 참조)`  (확신 medium)
- **근거**: `linear_key_head_dim == linear_value_head_dim == 128` 이라 수축 축의 두 끝이 서로 다른 이름을 달고 있다(행렬곱 합성 불일치 108건). 둘 다 소스에 있는 진짜 이름이고 이 체크포인트에서 값이 같을 뿐이라 **어느 쪽이 틀렸다고 말할 수 없다**. 두 값이 다른 체크포인트를 추적하기 전에는 결론을 낼 근거가 없다.  **근거 소스**: 이 판정은 `develop/sources/modeling_qwen3_5.py`, `develop/sources/configuration_qwen3_5.py` 를 열어 확인했다. (인용 누락을 자가 점검에서 발견해 보강, 2026-08-12 — 게이트가 이제 `should_be_renamed` 판정에 소스 인용을 요구한다.)  **재분류 (2026-08-13)**: 이 판정은 `undetermined` 였다. 잘못된 분류다 — 근거 문장이 "트레이스 안에 가를 증거가 없다"고 적고 있었는데, 그건 *트레이스만으로는* 못 가른다는 말이지 *알 수 없다*는 말이 아니다. **소스는 답을 갖고 있다**(위 인용). 막는 것은 지식이 아니라 표현 수단이다: 두 이름이 같은 값이라 `label_overrides` 의 이름 치환으로는 갈 수 없고, 필요한 것은 권위 있는 이름을 데이터플로우 따라 끌고 가는 메커니즘이다. `review/06-open-renames.md`

### A22. Qwen__Qwen3.6-27B

- **모듈**: `model.layers.*.linear_attn.norm`
- **축**: 정규화 폭 128
- **지금 → 제안**: `d_head_lin_k` → `d_head_lin_v`  (확신 high)
- **근거**: `modeling_qwen3_next.py:552` `self.norm = Qwen3NextRMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)` 이고 `:519` `self.head_v_dim = config.linear_value_head_dim` 다. 이 norm 의 폭은 **value** head dim 이다. linear_key_head_dim 과 linear_value_head_dim 이 둘 다 128 이라 값으로는 구별할 수 없었다. 같은 행의 앞 축이 이미 `n_h_lin_v*T` 로 렌더되고 있어 한 텐서 안에서도 앞뒤가 어긋나 있었다(`[n_h_lin_v*T, d_head_lin_k]`, 실측 `[544, 128]`).  **
- **막힌 이유(측정)**: 아직 반영하지 않은 이유(측정)**: 이 이름을 `rules/label_overrides.yaml` 로 적용해 봤더니 게이트 퇴행 검사가 걸렸다 — flow_ambig 144 -> 432. override 층은 **한 모듈 안의** 이름만 바꾸므로, 같은 텐서를 렌더하는 이웃 모듈이 옛 이름으로 남아 데이터플로우 불일치가 드러난다. 이름이 틀렸다는 판정 자체는 위 소스로 확정이고, 필요한 것은 '권위 있는 이름을 데이터플로우 따라 끌고 가는' 별도 메커니즘이다.

### A23. Qwen__Qwen3.6-27B

- **모듈**: `model.layers.*.linear_attn`
- **축**: gated delta rule 청크 길이 64 (chunk_size)
- **지금 → 제안**: `d_rope` → `d_chunk`  (확신 high)
- **근거**: `modeling_qwen3_next.py:381` `def torch_chunk_gated_delta_rule(..., chunk_size=64)` — 청크 길이가 **config 필드가 아니라 커널 fallback 의 기본 인자**다. 같은 리터럴이 `modeling_qwen3_5.py` / `modeling_qwen3_5_moe.py` 에도 있다. 심볼 표는 config 를 읽으므로 코드에만 있는 상수는 구조적으로 유도할 수 없고, 그래서 이 폭이 이름을 못 받거나 엉뚱한 이름을 받는다.  **현재 라벨이 틀렸다는 증거**: 이 블록(`Qwen3NextGatedDeltaNet`)에는 **RoPE 가 아예 없다** — RoPE 는 같은 스택의 `self_attn` 레이어에만 있다. 그런데 `rules/derived_dims.yaml` 의 `round(d_head * pr)` 규칙이 scope `attn|attention|rotary` 로 걸려 있고 `attn` 은 `linear_attn` 안에서도 매치한다. partial_rotary_factor(0.25) x head_dim(256) = 64 이고 chunk_size 도 64 라, 청크 스캔의 `[chunk, chunk]` triu 마스크가 통째로 `d_rope` 로 렌더되고 있다 — Qwen3.6-27B 한 모델에서만 31,440축(2026-08-13 측정).  그 규칙의 주석
- **막힌 이유(측정)**: 아직 반영하지 않은 이유**: (C) 가 옳지만 게이트 퇴행 검사가 막는다. 필요한 것은 '등록된 심볼이 그 값을 설명하는 자리에서는 휴리스틱이 이름을 짓지 않는다'는 규칙, 또는 권위 있는 이름을 데이터플로우 따라 끌고 가는 메커니즘이다. 43건과 같은 병이며 `review/06-open-renames.md` 의 자문 대상이다. 한쪽만 고치는 수정은 하지 않는다.

### A24. Qwen__Qwen3.6-35B-A3B

- **모듈**: `model.layers.*.linear_attn`
- **축**: in_proj_qkvz 조각 폭 (27B 에서 2048)
- **지금 → 제안**: `2*n_kv*d_head` → `key_dim (= n_h_lin_k · d_head_lin_k)`  (확신 high)
- **근거**: `modeling_qwen3_5.py:520-521` `self.key_dim = self.head_k_dim * self.num_k_heads` / `self.value_dim = self.head_v_dim * self.num_v_heads`. `split_with_sizes` 가 [key, key, value] 로 쪼개는 것이 트레이스에 그대로 보인다(실측 [2048, 2048, 6144]). 어텐션 head 수와 무관한 축인데 2·n_kv·d_head 와 값이 같아 그쪽으로 붙었다 — 확인된 오라벨. `n_h_lin_k * d_head_lin_k` 로 등록해봤으나 Qwen3-Next 의 flow_ambig 가 0 -> 72 로 퇴행해 보류했다(2026-08-10): 새 이름이 붙은 축의 하류 소비자가 옛 이름을 그대로 들고 있어 한 텐서가 두 이름을 갖는다. 라벨이 아니라 전파 쪽 과제다.

### A25. Qwen__Qwen3.6-35B-A3B

- **모듈**: `model.layers.*.linear_attn`
- **축**: matmul 수축 축 (128)
- **지금 → 제안**: `d_head_lin_k / d_head_lin_v 혼용` → `(소스가 가리키는 쪽 — 근거 참조)`  (확신 medium)
- **근거**: `linear_key_head_dim == linear_value_head_dim == 128` 이라 수축 축의 두 끝이 서로 다른 이름을 달고 있다(행렬곱 합성 불일치 108건). 둘 다 소스에 있는 진짜 이름이고 이 체크포인트에서 값이 같을 뿐이라 **어느 쪽이 틀렸다고 말할 수 없다**. 두 값이 다른 체크포인트를 추적하기 전에는 결론을 낼 근거가 없다.  **근거 소스**: 이 판정은 `develop/sources/modeling_qwen3_5_moe.py`, `develop/sources/configuration_qwen3_5_moe.py` 를 열어 확인했다. (인용 누락을 자가 점검에서 발견해 보강, 2026-08-12 — 게이트가 이제 `should_be_renamed` 판정에 소스 인용을 요구한다.)  **재분류 (2026-08-13)**: 이 판정은 `undetermined` 였다. 잘못된 분류다 — 근거 문장이 "트레이스 안에 가를 증거가 없다"고 적고 있었는데, 그건 *트레이스만으로는* 못 가른다는 말이지 *알 수 없다*는 말이 아니다. **소스는 답을 갖고 있다**(위 인용). 막는 것은 지식이 아니라 표현 수단이다: 두 이름이 같은 값이라 `label_overrides` 의 이름 치환으로는 갈 수 없고, 필요한 것은 권위 있는 이름을 데이터플로우 따라 끌고 가는 메커니즘이다. `review/06-open-ren

### A26. Qwen__Qwen3.6-35B-A3B

- **모듈**: `model.layers.*.linear_attn.norm`
- **축**: 정규화 폭 128
- **지금 → 제안**: `d_head_lin_k` → `d_head_lin_v`  (확신 high)
- **근거**: `modeling_qwen3_next.py:552` `self.norm = Qwen3NextRMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)` 이고 `:519` `self.head_v_dim = config.linear_value_head_dim` 다. 이 norm 의 폭은 **value** head dim 이다. linear_key_head_dim 과 linear_value_head_dim 이 둘 다 128 이라 값으로는 구별할 수 없었다. 같은 행의 앞 축이 이미 `n_h_lin_v*T` 로 렌더되고 있어 한 텐서 안에서도 앞뒤가 어긋나 있었다(`[n_h_lin_v*T, d_head_lin_k]`, 실측 `[544, 128]`).  **
- **막힌 이유(측정)**: 아직 반영하지 않은 이유(측정)**: 이 이름을 `rules/label_overrides.yaml` 로 적용해 봤더니 게이트 퇴행 검사가 걸렸다 — flow_ambig 90 -> 270. override 층은 **한 모듈 안의** 이름만 바꾸므로, 같은 텐서를 렌더하는 이웃 모듈이 옛 이름으로 남아 데이터플로우 불일치가 드러난다. 이름이 틀렸다는 판정 자체는 위 소스로 확정이고, 필요한 것은 '권위 있는 이름을 데이터플로우 따라 끌고 가는' 별도 메커니즘이다.

### A27. Qwen__Qwen3.6-35B-A3B

- **모듈**: `model.layers.*.linear_attn`
- **축**: gated delta rule 청크 길이 64 (chunk_size)
- **지금 → 제안**: `d_rope` → `d_chunk`  (확신 high)
- **근거**: `modeling_qwen3_next.py:381` `def torch_chunk_gated_delta_rule(..., chunk_size=64)` — 청크 길이가 **config 필드가 아니라 커널 fallback 의 기본 인자**다. 같은 리터럴이 `modeling_qwen3_5.py` / `modeling_qwen3_5_moe.py` 에도 있다. 심볼 표는 config 를 읽으므로 코드에만 있는 상수는 구조적으로 유도할 수 없고, 그래서 이 폭이 이름을 못 받거나 엉뚱한 이름을 받는다.  **현재 라벨이 틀렸다는 증거**: 이 블록(`Qwen3NextGatedDeltaNet`)에는 **RoPE 가 아예 없다** — RoPE 는 같은 스택의 `self_attn` 레이어에만 있다. 그런데 `rules/derived_dims.yaml` 의 `round(d_head * pr)` 규칙이 scope `attn|attention|rotary` 로 걸려 있고 `attn` 은 `linear_attn` 안에서도 매치한다. partial_rotary_factor(0.25) x head_dim(256) = 64 이고 chunk_size 도 64 라, 청크 스캔의 `[chunk, chunk]` triu 마스크가 통째로 `d_rope` 로 렌더되고 있다 — Qwen3.6-27B 한 모델에서만 31,440축(2026-08-13 측정).  그 규칙의 주석
- **막힌 이유(측정)**: 아직 반영하지 않은 이유**: (C) 가 옳지만 게이트 퇴행 검사가 막는다. 필요한 것은 '등록된 심볼이 그 값을 설명하는 자리에서는 휴리스틱이 이름을 짓지 않는다'는 규칙, 또는 권위 있는 이름을 데이터플로우 따라 끌고 가는 메커니즘이다. 43건과 같은 병이며 `review/06-open-renames.md` 의 자문 대상이다. 한쪽만 고치는 수정은 하지 않는다.

### A28. Zyphra__Zamba2-1.2B

- **모듈**: `model.layers.*.mamba`
- **축**: num_heads vs head_dim (둘 다 64)
- **지금 → 제안**: `d_head_ssm / n_h_ssm (순서 뒤바뀜)` → `n_h_ssm 이 앞, d_head_ssm 이 뒤`  (확신 high)
- **근거**: `modeling_zamba2.py:832` `hidden_states.view(batch_size, seq_len, self.num_heads, self.head_dim)`, `:622` `output.reshape(batch_size, -1, num_heads, head_dim)`, `:524` `ssm_states.view(batch_size * num_heads, head_dim, state_size)`, `:527` `out.view(batch_size, num_heads, head_dim)` — Mamba2 레이아웃에서 **num_heads 가 항상 head_dim 앞**이다. 우리 표는 정확히 반대로 붙어 있다: `[B, d_chunk, d_head_ssm, n_h_ssm]`(실측 `[1, 256, 64, 64]`), `[B, 1, d_head_ssm, n_h_ssm, d_state]`(실측 `[1, 1, 64, 64, 128]`). 홀로 나오는 자리도 마찬가지다 — `[B, d_head_ssm, 1, d_chunk]`(실측 `[1, 64, 1, 256]`)는 `:581 A_cumsum` 의 `[batch, num_heads, n_chunks, chunk]` 이고, `[B, T, d_head_ssm]`(실측 `[1, 16, 64]`)는 dt 의 `[batch, seq_len, num_heads]` 다. `n_mamba_h
- **막힌 이유(측정)**: 아직 반영하지 않은 이유**: 두 이름이 같은 값이라 `label_overrides` 의 이름 치환으로는 **맞바꿀 수 없다**(먼저 도는 항목이 두 축을 하나로 뭉갠다). 축 위치로 지정하려면 관측된 shape 형태를 전수로 열거해야 하는데, 여기서 확인한 8개 형태가 전부라는 보장이 없다. 필요한 것은 '이 스코프에서 head 개수가 head 폭보다 앞선다'는 **순서 규칙**이며 그것은 별도 변경이다. 값 하나만 보고 대량 치환하는 쪽이 지금 상태보다 나쁘

### A29. Zyphra__Zamba2-1.2B

- **모듈**: `model.layers.*.mamba_decoder.mamba`
- **축**: num_heads vs head_dim (둘 다 64)
- **지금 → 제안**: `d_head_ssm / n_h_ssm (순서 뒤바뀜)` → `n_h_ssm 이 앞, d_head_ssm 이 뒤`  (확신 high)
- **근거**: `modeling_zamba2.py:832` `hidden_states.view(batch_size, seq_len, self.num_heads, self.head_dim)`, `:622` `output.reshape(batch_size, -1, num_heads, head_dim)`, `:524` `ssm_states.view(batch_size * num_heads, head_dim, state_size)`, `:527` `out.view(batch_size, num_heads, head_dim)` — Mamba2 레이아웃에서 **num_heads 가 항상 head_dim 앞**이다. 우리 표는 정확히 반대로 붙어 있다: `[B, d_chunk, d_head_ssm, n_h_ssm]`(실측 `[1, 256, 64, 64]`), `[B, 1, d_head_ssm, n_h_ssm, d_state]`(실측 `[1, 1, 64, 64, 128]`). 홀로 나오는 자리도 마찬가지다 — `[B, d_head_ssm, 1, d_chunk]`(실측 `[1, 64, 1, 256]`)는 `:581 A_cumsum` 의 `[batch, num_heads, n_chunks, chunk]` 이고, `[B, T, d_head_ssm]`(실측 `[1, 16, 64]`)는 dt 의 `[batch, seq_len, num_heads]` 다. `n_mamba_h
- **막힌 이유(측정)**: 아직 반영하지 않은 이유**: 두 이름이 같은 값이라 `label_overrides` 의 이름 치환으로는 **맞바꿀 수 없다**(먼저 도는 항목이 두 축을 하나로 뭉갠다). 축 위치로 지정하려면 관측된 shape 형태를 전수로 열거해야 하는데, 여기서 확인한 8개 형태가 전부라는 보장이 없다. 필요한 것은 '이 스코프에서 head 개수가 head 폭보다 앞선다'는 **순서 규칙**이며 그것은 별도 변경이다. 값 하나만 보고 대량 치환하는 쪽이 지금 상태보다 나쁘

### A30. allenai__OLMoE-1B-7B-0924

- **모듈**: `model.layers.*.mlp.experts`
- **축**: [E, d_model, d_model] 의 가운데 축 (2048)
- **지금 → 제안**: `d_model` → `2*d_moe`  (확신 high)
- **근거**: `modeling_olmoe.py:297` `self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))`. intermediate_size=1024 이므로 2·1024=2048 이고, hidden_size 도 2048 이라 값이 겹쳐 d_model 로 붙었다. **부분 교정됨(2026-08-09).** 값·스코프로는 못 가려서(스코프로 이기게 했더니 잔차 스트림까지 바뀌어 flow_ambig 32→64) 하류 증거를 쓰는 규칙을 새로 넣었다 — `build_table._merge_from_split`: 한 축이 n 등분되어 같은 이름 n 개가 되면 그 축은 n 배다. 바로 다음 op 이 `split [[k*T, d_model]] -> [[k*T, d_moe], [k*T, d_moe]]` 로 한 행 안에서 자기모순을 드러내고 있었다. 이제 grouped_matmul 출력이 `[k*T, 2*d_moe]`, down_proj 가중치가 `[E, d_model, d_moe]` 로 맞게 나온다. **남은 것**: 융합 가중치 자체의 가운데 축은 여전히 `d_model` 이다 — 뒤 두 축이 둘 다 2048 이라 순서를 가릴 증거가 트레이스 안에 없다(`_weight_out_from_output` 이 이 경우를 의

### A31. bzantium__tiny-deepseek-v3

- **모듈**: `model.layers.*.self_attn`
- **축**: value 경로 head 폭 (128) — split 둘째 조각부터 o_proj 입력까지
- **지금 → 제안**: `d_nope` → `d_v`  (확신 high)
- **근거**: 같은 split 의 **둘째** 조각이 `value_states` 이고 그 head 폭은 `v_head_dim` 이다(`modeling_deepseek_v3.py:419`). o_proj 가 `nn.Linear(num_heads * v_head_dim, hidden_size)` (:401-402)이므로 합쳐진 폭은 실제로 `n_h*d_v` 로 맞게 렌더된다 — 그래서 `view [B,T,n_h,d_nope] -> [B,T,n_h*d_v]` 한 행 안에서 두 설명이 어긋난다(모델당 61행, 총 195행).  **고치지 못했다. 시도한 것과 결과를 남긴다.** 등록된 `A+B` 의 피연산자 순서가 소스의 split 순서 그대로라는 점을 이용해 조각을 A·B 로 이름 붙이는 규칙을 넣어 봤다(`_split_from_registered_sum`). split 출력은 맞게 바뀌었지만 **그 아래 사슬 전체가 옛 이름을 유지**해서 reshape 불일치가 61 → 122 로, flow_ambig 가 0 → 122 로 늘었다. `_propagate_labels` 는 monotone 이라(빈 정수만 채운다) 이름을 덮어쓰지 않는다. 이건 이 저장소가 이미 두 번 측정한 실패 형태다 — `_carry_reshape_labels` 가 같은 이유로 비활성 상태다. 제대로 고치려면 **권위 있는 개명을 데이터플로우를 따라 끝까지 옮기는** 기계장치가 필요

### A32. bzantium__tiny-deepseek-v3

- **모듈**: `model.layers.*.self_attn`
- **축**: q/k split 둘째 조각 (64)
- **지금 → 제안**: `d_head` → `d_rope`  (확신 medium)
- **근거**: `split_with_sizes [B,n_h,T,d_nope+d_rope] -> [B,n_h,T,d_nope], [B,n_h,T,d_head]` — 둘째 조각은 RoPE 를 받는 부분이므로 `d_rope` 다. 이 모델들은 head_dim == qk_rope_head_dim == 64 라 값이 겹친다. 위와 **정확히 같은 원인·같은 막힘**이라 함께 남긴다.  **근거 소스**: 이 판정은 `develop/sources/modeling_deepseek_v3.py`, `develop/sources/configuration_deepseek_v3.py` 를 열어 확인했다. (인용 누락을 자가 점검에서 발견해 보강, 2026-08-12 — 게이트가 이제 `should_be_renamed` 판정에 소스 인용을 요구한다.)  **
- **막힌 이유(측정)**: 아직 반영하지 않은 이유(측정)**: 이 이름을 `rules/label_overrides.yaml` 로 적용해 봤더니 게이트 퇴행 검사가 걸렸다 — flow_ambig 0 -> 24. override 층은 **한 모듈 안의** 이름만 바꾸므로, 같은 텐서를 렌더하는 이웃 모듈이 옛 이름으로 남아 데이터플로우 불일치가 드러난다. 이름이 틀렸다는 판정 자체는 위 소스로 확정이고, 필요한 것은 '권위 있는 이름을 데이터플로우 따라 끌고 가는' 별도 메커니즘이다. 한쪽

### A33. bzantium__tiny-deepseek-v3

- **모듈**: `model.rotary_emb`
- **축**: cos/sin 폭 64
- **지금 → 제안**: `d_head` → `d_rope`  (확신 high)
- **근거**: `configuration_deepseek_v3.py:124` `self.head_dim = self.qk_rope_head_dim` — MLA 는 config.head_dim 을 **rope 슬라이스 폭**으로 설정한다. `modeling_deepseek_v3.py:88-92` `dim = getattr(config, "head_dim", ...)`, `inv_freq = 1.0 / base ** (arange(0, dim, 2) / dim)` 이므로 inv_freq 는 dim/2 이고 cos/sin 은 그 두 배다. 같은 모듈의 다른 축이 이미 `d_rope/2`(32)로 렌더되고 있어 64 를 `d_head` 라고 부르면 한 모듈 안에서 2x(d_rope/2) != d_head 가 된다. `d_rope` 가 그 자리의 이름이다.

### A34. deepseek-ai__DeepSeek-V2-Lite

- **모듈**: `model.layers.*.self_attn`
- **축**: value 경로 head 폭 (128) — split 둘째 조각부터 o_proj 입력까지
- **지금 → 제안**: `d_nope` → `d_v`  (확신 high)
- **근거**: 같은 split 의 **둘째** 조각이 `value_states` 이고 그 head 폭은 `v_head_dim` 이다(`modeling_deepseek_v3.py:419`). o_proj 가 `nn.Linear(num_heads * v_head_dim, hidden_size)` (:401-402)이므로 합쳐진 폭은 실제로 `n_h*d_v` 로 맞게 렌더된다 — 그래서 `view [B,T,n_h,d_nope] -> [B,T,n_h*d_v]` 한 행 안에서 두 설명이 어긋난다(모델당 61행, 총 195행).  **고치지 못했다. 시도한 것과 결과를 남긴다.** 등록된 `A+B` 의 피연산자 순서가 소스의 split 순서 그대로라는 점을 이용해 조각을 A·B 로 이름 붙이는 규칙을 넣어 봤다(`_split_from_registered_sum`). split 출력은 맞게 바뀌었지만 **그 아래 사슬 전체가 옛 이름을 유지**해서 reshape 불일치가 61 → 122 로, flow_ambig 가 0 → 122 로 늘었다. `_propagate_labels` 는 monotone 이라(빈 정수만 채운다) 이름을 덮어쓰지 않는다. 이건 이 저장소가 이미 두 번 측정한 실패 형태다 — `_carry_reshape_labels` 가 같은 이유로 비활성 상태다. 제대로 고치려면 **권위 있는 개명을 데이터플로우를 따라 끝까지 옮기는** 기계장치가 필요

### A35. deepseek-ai__DeepSeek-V2-Lite

- **모듈**: `model.layers.*.self_attn`
- **축**: q/k split 둘째 조각 (64)
- **지금 → 제안**: `d_head` → `d_rope`  (확신 medium)
- **근거**: `split_with_sizes [B,n_h,T,d_nope+d_rope] -> [B,n_h,T,d_nope], [B,n_h,T,d_head]` — 둘째 조각은 RoPE 를 받는 부분이므로 `d_rope` 다. 이 모델들은 head_dim == qk_rope_head_dim == 64 라 값이 겹친다. 위와 **정확히 같은 원인·같은 막힘**이라 함께 남긴다.  **근거 소스**: 이 판정은 `develop/sources/modeling_deepseek_v2.py`, `develop/sources/configuration_deepseek_v2.py` 를 열어 확인했다. (인용 누락을 자가 점검에서 발견해 보강, 2026-08-12 — 게이트가 이제 `should_be_renamed` 판정에 소스 인용을 요구한다.)

### A36. deepseek-ai__DeepSeek-V3

- **모듈**: `model.layers.*.self_attn`
- **축**: value 경로 head 폭 (128) — split 둘째 조각부터 o_proj 입력까지
- **지금 → 제안**: `d_nope` → `d_v`  (확신 high)
- **근거**: 같은 split 의 **둘째** 조각이 `value_states` 이고 그 head 폭은 `v_head_dim` 이다(`modeling_deepseek_v3.py:419`). o_proj 가 `nn.Linear(num_heads * v_head_dim, hidden_size)` (:401-402)이므로 합쳐진 폭은 실제로 `n_h*d_v` 로 맞게 렌더된다 — 그래서 `view [B,T,n_h,d_nope] -> [B,T,n_h*d_v]` 한 행 안에서 두 설명이 어긋난다(모델당 61행, 총 195행).  **고치지 못했다. 시도한 것과 결과를 남긴다.** 등록된 `A+B` 의 피연산자 순서가 소스의 split 순서 그대로라는 점을 이용해 조각을 A·B 로 이름 붙이는 규칙을 넣어 봤다(`_split_from_registered_sum`). split 출력은 맞게 바뀌었지만 **그 아래 사슬 전체가 옛 이름을 유지**해서 reshape 불일치가 61 → 122 로, flow_ambig 가 0 → 122 로 늘었다. `_propagate_labels` 는 monotone 이라(빈 정수만 채운다) 이름을 덮어쓰지 않는다. 이건 이 저장소가 이미 두 번 측정한 실패 형태다 — `_carry_reshape_labels` 가 같은 이유로 비활성 상태다. 제대로 고치려면 **권위 있는 개명을 데이터플로우를 따라 끝까지 옮기는** 기계장치가 필요

### A37. deepseek-ai__DeepSeek-V3

- **모듈**: `model.layers.*.self_attn`
- **축**: q/k split 둘째 조각 (64)
- **지금 → 제안**: `d_head` → `d_rope`  (확신 medium)
- **근거**: `split_with_sizes [B,n_h,T,d_nope+d_rope] -> [B,n_h,T,d_nope], [B,n_h,T,d_head]` — 둘째 조각은 RoPE 를 받는 부분이므로 `d_rope` 다. 이 모델들은 head_dim == qk_rope_head_dim == 64 라 값이 겹친다. 위와 **정확히 같은 원인·같은 막힘**이라 함께 남긴다.  **근거 소스**: 이 판정은 `develop/sources/modeling_deepseek_v3.py`, `develop/sources/configuration_deepseek_v3.py` 를 열어 확인했다. (인용 누락을 자가 점검에서 발견해 보강, 2026-08-12 — 게이트가 이제 `should_be_renamed` 판정에 소스 인용을 요구한다.)  **
- **막힌 이유(측정)**: 아직 반영하지 않은 이유(측정)**: 이 이름을 `rules/label_overrides.yaml` 로 적용해 봤더니 게이트 퇴행 검사가 걸렸다 — flow_ambig 0 -> 244. override 층은 **한 모듈 안의** 이름만 바꾸므로, 같은 텐서를 렌더하는 이웃 모듈이 옛 이름으로 남아 데이터플로우 불일치가 드러난다. 이름이 틀렸다는 판정 자체는 위 소스로 확정이고, 필요한 것은 '권위 있는 이름을 데이터플로우 따라 끌고 가는' 별도 메커니즘이다. 한

### A38. deepseek-ai__DeepSeek-V3

- **모듈**: `model.rotary_emb`
- **축**: cos/sin 폭 64
- **지금 → 제안**: `d_head` → `d_rope`  (확신 high)
- **근거**: `configuration_deepseek_v3.py:124` `self.head_dim = self.qk_rope_head_dim` — MLA 는 config.head_dim 을 **rope 슬라이스 폭**으로 설정한다. `modeling_deepseek_v3.py:88-92` `dim = getattr(config, "head_dim", ...)`, `inv_freq = 1.0 / base ** (arange(0, dim, 2) / dim)` 이므로 inv_freq 는 dim/2 이고 cos/sin 은 그 두 배다. 같은 모듈의 다른 축이 이미 `d_rope/2`(32)로 렌더되고 있어 64 를 `d_head` 라고 부르면 한 모듈 안에서 2x(d_rope/2) != d_head 가 된다. `d_rope` 가 그 자리의 이름이다.

### A39. deepseek-ai__DeepSeek-V4-Flash

- **모듈**: `model.layers.*.mlp.experts`
- **축**: [E, d_model, d_model] 의 가운데 축 (4096)
- **지금 → 제안**: `d_model` → `2*d_moe`  (확신 high)
- **근거**: `modeling_deepseek_v4.py:992` `self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))`. d_moe=2048 이라 2·2048=4096=d_model 로 겹친다. OLMoE 와 같은 경로로 부분 교정됐다 — 활성화 사슬과 down_proj 는 맞고, 융합 가중치의 가운데 축만 남았다(사유 동일).

### A40. deepseek-ai__DeepSeek-V4-Flash

- **모듈**: `model.layers.*.self_attn.compressor.indexer`
- **축**: [B, T/m_csa, 4, c_I] 의 셋째 축
- **지금 → 제안**: `4 (이름 없음)` → `m_csa`  (확신 medium)
- **근거**: indexer 안의 `[B, T/m_csa, 4, c_I]` 는 압축 엔트리마다 그것이 덮는 원본 토큰 m_csa 개다(m_csa=4). 그런데 `m_csa` 의 스코프가 `compressor(?!\.indexer)` 라 이름이 안 붙고 정수로 남는다. 그 배제는 원래 **m_hca(=128)가 c_I(=128)를 뺏는 것**을 막으려고 넣은 것이라, 값이 겹치지 않는 m_csa 까지 막을 이유가 없다. 배제를 `compressor` 로 여는 것을 시도했으나 되돌렸다 — V4-Pro 의 heur 가 2,131 -> 3,331 로 퇴행한다(indexer 안에서 4 가 다른 축까지 가져가고 128 자리에 T/m_hca 가 밀려든다). 심볼 하나만 스코프를 여는 문법이 없어 그대로 둔다. **이 건은 미결 4범주(별칭·정사각·미등록·휴리스틱) 어디에도 안 걸렸고, 의뢰서에 새로 넣은 전수 점검 B절(이름 없는 정수 x 같은 값의 심볼)이 처음 드러냈다.**  **근거 소스**: 이 판정은 `develop/sources/modeling_deepseek_v4.py`, `develop/sources/configuration_deepseek_v4.py` 를 열어 확인했다. (인용 누락을 자가 점검에서 발견해 보강, 2026-08-12 — 게이트가 이제 `should_be_renamed` 판정에 소스 인용을 요구한다.)

### A41. deepseek-ai__DeepSeek-V4-Flash

- **모듈**: `model.layers.*.self_attn`
- **축**: 복소수 되접기 축 (64)
- **지금 → 제안**: `n_h` → `d_rope`  (확신 high)
- **근거**: `view [B, T, d_rope/2, 2] -> [B, T, n_h]` — 뒤 두 축을 합치면 d_rope/2 × 2 = **d_rope**(64)다. RoPE 를 복소수 곱으로 구현할 때 실수부·허수부를 되접는 자리이고, attention head 수와는 아무 관계가 없다. n_h 도 64 라 값으로는 안 보인다.  **반박 프레임으로 찾았다** — '이 라벨이 맞나' 대신 '이 view 의 입력이 출력을 설명하는가'를 물었더니 바로 드러났다. 고치려면 view 의 병합 유도를 채택해야 하는데, `n_h` 는 같은 모듈의 `[B, T, n_h, d_head]` 에서는 옳은 이름이라 모듈 단위 교정으로는 표현할 수 없다(review/05-overrides.md 의 '표현할 수 없는 것'). `open` 으로 남긴다.  **근거 소스**: 이 판정은 `develop/sources/modeling_deepseek_v4.py`, `develop/sources/configuration_deepseek_v4.py` 를 열어 확인했다. (인용 누락을 자가 점검에서 발견해 보강, 2026-08-12 — 게이트가 이제 `should_be_renamed` 판정에 소스 인용을 요구한다.)

### A42. deepseek-ai__DeepSeek-V4-Flash-0731

- **모듈**: `model.layers.*.self_attn.compressor.indexer`
- **축**: [B, T/m_csa, 4, c_I] 의 셋째 축
- **지금 → 제안**: `4 (이름 없음)` → `m_csa`  (확신 medium)
- **근거**: indexer 안의 `[B, T/m_csa, 4, c_I]` 는 압축 엔트리마다 그것이 덮는 원본 토큰 m_csa 개다(m_csa=4). 그런데 `m_csa` 의 스코프가 `compressor(?!\.indexer)` 라 이름이 안 붙고 정수로 남는다. 그 배제는 원래 **m_hca(=128)가 c_I(=128)를 뺏는 것**을 막으려고 넣은 것이라, 값이 겹치지 않는 m_csa 까지 막을 이유가 없다. 배제를 `compressor` 로 여는 것을 시도했으나 되돌렸다 — V4-Pro 의 heur 가 2,131 -> 3,331 로 퇴행한다(indexer 안에서 4 가 다른 축까지 가져가고 128 자리에 T/m_hca 가 밀려든다). 심볼 하나만 스코프를 여는 문법이 없어 그대로 둔다. **이 건은 미결 4범주(별칭·정사각·미등록·휴리스틱) 어디에도 안 걸렸고, 의뢰서에 새로 넣은 전수 점검 B절(이름 없는 정수 x 같은 값의 심볼)이 처음 드러냈다.**  **근거 소스**: 이 판정은 `develop/sources/modeling_deepseek_v4.py`, `develop/sources/configuration_deepseek_v4.py` 를 열어 확인했다. (인용 누락을 자가 점검에서 발견해 보강, 2026-08-12 — 게이트가 이제 `should_be_renamed` 판정에 소스 인용을 요구한다.)

### A43. deepseek-ai__DeepSeek-V4-Flash-0731

- **모듈**: `model.layers.*.self_attn`
- **축**: 복소수 되접기 축 (64)
- **지금 → 제안**: `n_h` → `d_rope`  (확신 high)
- **근거**: `view [B, T, d_rope/2, 2] -> [B, T, n_h]` — 뒤 두 축을 합치면 d_rope/2 × 2 = **d_rope**(64)다. RoPE 를 복소수 곱으로 구현할 때 실수부·허수부를 되접는 자리이고, attention head 수와는 아무 관계가 없다. n_h 도 64 라 값으로는 안 보인다.  **반박 프레임으로 찾았다** — '이 라벨이 맞나' 대신 '이 view 의 입력이 출력을 설명하는가'를 물었더니 바로 드러났다. 고치려면 view 의 병합 유도를 채택해야 하는데, `n_h` 는 같은 모듈의 `[B, T, n_h, d_head]` 에서는 옳은 이름이라 모듈 단위 교정으로는 표현할 수 없다(review/05-overrides.md 의 '표현할 수 없는 것'). `open` 으로 남긴다.  **근거 소스**: 이 판정은 `develop/sources/modeling_deepseek_v4.py`, `develop/sources/configuration_deepseek_v4.py` 를 열어 확인했다. (인용 누락을 자가 점검에서 발견해 보강, 2026-08-12 — 게이트가 이제 `should_be_renamed` 판정에 소스 인용을 요구한다.)

### A44. deepseek-ai__DeepSeek-V4-Pro

- **모듈**: `model.layers.*.self_attn.compressor.kv_norm`
- **축**: [B, 512, 512] 의 축 순서
- **지금 → 제안**: `[B, d_head, d_head]` → `[B, T/m_csa, d_head]`  (확신 high)
- **근거**: `modeling_deepseek_v4.py:382` `self.kv_norm = DeepseekV4RMSNorm(self.head_dim, ...)` — RMSNorm 은 마지막 축을 정규화하므로 마지막이 `d_head`(512)이고 가운데가 압축 KV 길이다. **부분 교정(2026-08-09)**: rank-1 norm 앵커를 그 모듈 전체로 확장해 본체 텐서는 `[B, T, d_head]` / `[B, T/m_hca, d_head]` 로 맞았다. **정정(2026-08-10)** — 그때 '교정 완료'라고 적었지만 사실이 아니었다. 새로 넣은 elementwise 라벨 일관성 검사가 같은 모듈에서 30행을 잡아냈다: `elementwise_mul([B, d_head, T/m_csa], [B, d_head, 1]) -> [B, d_head, d_head]` — T/m_csa 가 2048/4 = 512 로 d_head 와 같은 자리라 입력과 출력이 서로 다른 이름을 달고 있다. 값으로는 못 가리고, norm 앵커는 마지막 축만 고정하므로 가운데 축이 남는다. 게이트가 이제 이 30행을 매번 보고한다.

### A45. deepseek-ai__DeepSeek-V4-Pro

- **모듈**: `model.layers.*.self_attn`
- **축**: grouped output projection 그룹 축 (16)
- **지금 → 제안**: `T/m_hca` → `g_o`  (확신 high)
- **근거**: `clone [B,T,T/m_hca,d_g] -> _unsafe_view -> [B,T,g_o*d_g]` (실측 `[1,2048,16,1024]` → `[1,2048,16384]`). 합쳐진 축이 `g_o*d_g` 이므로 셋째 축은 `g_o` 여야 하는데 g_o = T/m_hca = 16 이라 압축 엔트리 수의 이름이 붙었다. `d_g` 자체는 맞다.  고치려면 권위 있는 출력 라벨(`g_o*d_g`)의 인수를 입력 축으로 되밀어야 하고, 그 기계장치(`_split_from_authoritative`)가 이 op 에서는 발화하지 않는다. MLA 의 `d_v` 건과 **같은 막힘**이다 — 개명을 데이터플로우 끝까지 옮기는 문제.  **근거 소스**: 이 판정은 `develop/sources/modeling_deepseek_v4.py`, `develop/sources/configuration_deepseek_v4.py` 를 열어 확인했다. (인용 누락을 자가 점검에서 발견해 보강, 2026-08-12 — 게이트가 이제 `should_be_renamed` 판정에 소스 인용을 요구한다.)

### A46. ibm-granite__granite-4.0-h-small

- **모듈**: `model.layers.*.mamba`
- **축**: n_h_ssm vs d_state 축 (둘 다 128)
- **지금 → 제안**: `d_state / n_h_ssm 혼용` → `(소스가 가리키는 쪽 — 근거 참조)`  (확신 medium)
- **근거**: `view [.., n_g_ssm, n_h_ssm/n_g_ssm, ?] -> [.., ?, ?]` 의 두 축이 값으로 구별되지 않는다(ssm_state_size == mamba_n_heads == 128). Nemotron-3-Super 와 **같은 막힘**이고, 합쳐진 축이 무엇인지는 reshape 자체가 알지만 그걸 채택하려면 개명을 데이터플로우 끝까지 옮겨야 한다. 값으로 우기지 않고 남긴다.  **근거 소스**: 이 판정은 `develop/sources/modeling_granitemoehybrid.py`, `develop/sources/configuration_granitemoehybrid.py` 를 열어 확인했다. (인용 누락을 자가 점검에서 발견해 보강, 2026-08-12 — 게이트가 이제 `should_be_renamed` 판정에 소스 인용을 요구한다.)  **재분류 (2026-08-13)**: 이 판정은 `undetermined` 였다. 잘못된 분류다 — 근거 문장이 "트레이스 안에 가를 증거가 없다"고 적고 있었는데, 그건 *트레이스만으로는* 못 가른다는 말이지 *알 수 없다*는 말이 아니다. **소스는 답을 갖고 있다**(위 인용). 막는 것은 지식이 아니라 표현 수단이다: 두 이름이 같은 값이라 `label_overrides` 의 이름 치환으로는 갈 수 없고, 필요한 것은 권위 있는 이름을 데이터플로우 따라 끌

### A47. moonshotai__Kimi-K2-Instruct

- **모듈**: `model.layers.*.self_attn`
- **축**: value 경로 head 폭 (128) — split 둘째 조각부터 o_proj 입력까지
- **지금 → 제안**: `d_nope` → `d_v`  (확신 high)
- **근거**: 같은 split 의 **둘째** 조각이 `value_states` 이고 그 head 폭은 `v_head_dim` 이다(`modeling_deepseek_v3.py:419`). o_proj 가 `nn.Linear(num_heads * v_head_dim, hidden_size)` (:401-402)이므로 합쳐진 폭은 실제로 `n_h*d_v` 로 맞게 렌더된다 — 그래서 `view [B,T,n_h,d_nope] -> [B,T,n_h*d_v]` 한 행 안에서 두 설명이 어긋난다(모델당 61행, 총 195행).  **고치지 못했다. 시도한 것과 결과를 남긴다.** 등록된 `A+B` 의 피연산자 순서가 소스의 split 순서 그대로라는 점을 이용해 조각을 A·B 로 이름 붙이는 규칙을 넣어 봤다(`_split_from_registered_sum`). split 출력은 맞게 바뀌었지만 **그 아래 사슬 전체가 옛 이름을 유지**해서 reshape 불일치가 61 → 122 로, flow_ambig 가 0 → 122 로 늘었다. `_propagate_labels` 는 monotone 이라(빈 정수만 채운다) 이름을 덮어쓰지 않는다. 이건 이 저장소가 이미 두 번 측정한 실패 형태다 — `_carry_reshape_labels` 가 같은 이유로 비활성 상태다. 제대로 고치려면 **권위 있는 개명을 데이터플로우를 따라 끝까지 옮기는** 기계장치가 필요

### A48. moonshotai__Kimi-K2-Instruct

- **모듈**: `model.layers.*.self_attn`
- **축**: q/k split 둘째 조각 (64)
- **지금 → 제안**: `d_head` → `d_rope`  (확신 medium)
- **근거**: `split_with_sizes [B,n_h,T,d_nope+d_rope] -> [B,n_h,T,d_nope], [B,n_h,T,d_head]` — 둘째 조각은 RoPE 를 받는 부분이므로 `d_rope` 다. 이 모델들은 head_dim == qk_rope_head_dim == 64 라 값이 겹친다. 위와 **정확히 같은 원인·같은 막힘**이라 함께 남긴다.  **근거 소스**: 이 판정은 `develop/sources/modeling_deepseek_v3.py`, `develop/sources/configuration_deepseek_v3.py` 를 열어 확인했다. (인용 누락을 자가 점검에서 발견해 보강, 2026-08-12 — 게이트가 이제 `should_be_renamed` 판정에 소스 인용을 요구한다.)

### A49. moonshotai__Kimi-K2-Instruct

- **모듈**: `model.rotary_emb`
- **축**: cos/sin 폭 64
- **지금 → 제안**: `d_head` → `d_rope`  (확신 high)
- **근거**: `configuration_deepseek_v3.py:124` `self.head_dim = self.qk_rope_head_dim` — MLA 는 config.head_dim 을 **rope 슬라이스 폭**으로 설정한다. `modeling_deepseek_v3.py:88-92` `dim = getattr(config, "head_dim", ...)`, `inv_freq = 1.0 / base ** (arange(0, dim, 2) / dim)` 이므로 inv_freq 는 dim/2 이고 cos/sin 은 그 두 배다. 같은 모듈의 다른 축이 이미 `d_rope/2`(32)로 렌더되고 있어 64 를 `d_head` 라고 부르면 한 모듈 안에서 2x(d_rope/2) != d_head 가 된다. `d_rope` 가 그 자리의 이름이다.

### A50. moonshotai__Kimi-K2.6

- **모듈**: `model.layers.*.self_attn`
- **축**: value 경로 head 폭 (128) — split 둘째 조각부터 o_proj 입력까지
- **지금 → 제안**: `d_nope` → `d_v`  (확신 high)
- **근거**: 같은 split 의 **둘째** 조각이 `value_states` 이고 그 head 폭은 `v_head_dim` 이다(`modeling_deepseek_v3.py:419`). o_proj 가 `nn.Linear(num_heads * v_head_dim, hidden_size)` (:401-402)이므로 합쳐진 폭은 실제로 `n_h*d_v` 로 맞게 렌더된다 — 그래서 `view [B,T,n_h,d_nope] -> [B,T,n_h*d_v]` 한 행 안에서 두 설명이 어긋난다(모델당 61행, 총 195행).  **고치지 못했다. 시도한 것과 결과를 남긴다.** 등록된 `A+B` 의 피연산자 순서가 소스의 split 순서 그대로라는 점을 이용해 조각을 A·B 로 이름 붙이는 규칙을 넣어 봤다(`_split_from_registered_sum`). split 출력은 맞게 바뀌었지만 **그 아래 사슬 전체가 옛 이름을 유지**해서 reshape 불일치가 61 → 122 로, flow_ambig 가 0 → 122 로 늘었다. `_propagate_labels` 는 monotone 이라(빈 정수만 채운다) 이름을 덮어쓰지 않는다. 이건 이 저장소가 이미 두 번 측정한 실패 형태다 — `_carry_reshape_labels` 가 같은 이유로 비활성 상태다. 제대로 고치려면 **권위 있는 개명을 데이터플로우를 따라 끝까지 옮기는** 기계장치가 필요

### A51. moonshotai__Kimi-K2.6

- **모듈**: `model.layers.*.self_attn`
- **축**: q/k split 둘째 조각 (64)
- **지금 → 제안**: `d_head` → `d_rope`  (확신 medium)
- **근거**: `split_with_sizes [B,n_h,T,d_nope+d_rope] -> [B,n_h,T,d_nope], [B,n_h,T,d_head]` — 둘째 조각은 RoPE 를 받는 부분이므로 `d_rope` 다. 이 모델들은 head_dim == qk_rope_head_dim == 64 라 값이 겹친다. 위와 **정확히 같은 원인·같은 막힘**이라 함께 남긴다.  **근거 소스**: 이 판정은 `develop/sources/moonshotai__Kimi-K2.6__modeling_deepseek.py`, `develop/sources/moonshotai__Kimi-K2.6__configuration_deepseek.py` 를 열어 확인했다. (인용 누락을 자가 점검에서 발견해 보강, 2026-08-12 — 게이트가 이제 `should_be_renamed` 판정에 소스 인용을 요구한다.)

### A52. moonshotai__Kimi-K2.6

- **모듈**: `model.rotary_emb`
- **축**: cos/sin 폭 64
- **지금 → 제안**: `d_head` → `d_rope`  (확신 high)
- **근거**: `configuration_deepseek_v3.py:124` `self.head_dim = self.qk_rope_head_dim` — MLA 는 config.head_dim 을 **rope 슬라이스 폭**으로 설정한다. `modeling_deepseek_v3.py:88-92` `dim = getattr(config, "head_dim", ...)`, `inv_freq = 1.0 / base ** (arange(0, dim, 2) / dim)` 이므로 inv_freq 는 dim/2 이고 cos/sin 은 그 두 배다. 같은 모듈의 다른 축이 이미 `d_rope/2`(32)로 렌더되고 있어 64 를 `d_head` 라고 부르면 한 모듈 안에서 2x(d_rope/2) != d_head 가 된다. `d_rope` 가 그 자리의 이름이다.

### A53. moonshotai__Kimi-K2.7-Code

- **모듈**: `model.layers.*.self_attn`
- **축**: value 경로 head 폭 (128) — split 둘째 조각부터 o_proj 입력까지
- **지금 → 제안**: `d_nope` → `d_v`  (확신 high)
- **근거**: 같은 split 의 **둘째** 조각이 `value_states` 이고 그 head 폭은 `v_head_dim` 이다(`modeling_deepseek_v3.py:419`). o_proj 가 `nn.Linear(num_heads * v_head_dim, hidden_size)` (:401-402)이므로 합쳐진 폭은 실제로 `n_h*d_v` 로 맞게 렌더된다 — 그래서 `view [B,T,n_h,d_nope] -> [B,T,n_h*d_v]` 한 행 안에서 두 설명이 어긋난다(모델당 61행, 총 195행).  **고치지 못했다. 시도한 것과 결과를 남긴다.** 등록된 `A+B` 의 피연산자 순서가 소스의 split 순서 그대로라는 점을 이용해 조각을 A·B 로 이름 붙이는 규칙을 넣어 봤다(`_split_from_registered_sum`). split 출력은 맞게 바뀌었지만 **그 아래 사슬 전체가 옛 이름을 유지**해서 reshape 불일치가 61 → 122 로, flow_ambig 가 0 → 122 로 늘었다. `_propagate_labels` 는 monotone 이라(빈 정수만 채운다) 이름을 덮어쓰지 않는다. 이건 이 저장소가 이미 두 번 측정한 실패 형태다 — `_carry_reshape_labels` 가 같은 이유로 비활성 상태다. 제대로 고치려면 **권위 있는 개명을 데이터플로우를 따라 끝까지 옮기는** 기계장치가 필요

### A54. moonshotai__Kimi-K2.7-Code

- **모듈**: `model.layers.*.self_attn`
- **축**: q/k split 둘째 조각 (64)
- **지금 → 제안**: `d_head` → `d_rope`  (확신 medium)
- **근거**: `split_with_sizes [B,n_h,T,d_nope+d_rope] -> [B,n_h,T,d_nope], [B,n_h,T,d_head]` — 둘째 조각은 RoPE 를 받는 부분이므로 `d_rope` 다. 이 모델들은 head_dim == qk_rope_head_dim == 64 라 값이 겹친다. 위와 **정확히 같은 원인·같은 막힘**이라 함께 남긴다.  **근거 소스**: 이 판정은 `develop/sources/moonshotai__Kimi-K2.7-Code__modeling_deepseek.py`, `develop/sources/moonshotai__Kimi-K2.7-Code__configuration_deepseek.py` 를 열어 확인했다. (인용 누락을 자가 점검에서 발견해 보강, 2026-08-12 — 게이트가 이제 `should_be_renamed` 판정에 소스 인용을 요구한다.)

### A55. moonshotai__Kimi-K2.7-Code

- **모듈**: `model.rotary_emb`
- **축**: cos/sin 폭 64
- **지금 → 제안**: `d_head` → `d_rope`  (확신 high)
- **근거**: `configuration_deepseek_v3.py:124` `self.head_dim = self.qk_rope_head_dim` — MLA 는 config.head_dim 을 **rope 슬라이스 폭**으로 설정한다. `modeling_deepseek_v3.py:88-92` `dim = getattr(config, "head_dim", ...)`, `inv_freq = 1.0 / base ** (arange(0, dim, 2) / dim)` 이므로 inv_freq 는 dim/2 이고 cos/sin 은 그 두 배다. 같은 모듈의 다른 축이 이미 `d_rope/2`(32)로 렌더되고 있어 64 를 `d_head` 라고 부르면 한 모듈 안에서 2x(d_rope/2) != d_head 가 된다. `d_rope` 가 그 자리의 이름이다.

### A56. nvidia__NVIDIA-Nemotron-3-Super-120B-A12B-BF16

- **모듈**: `model.layers.*.mixer`
- **축**: n_h_ssm vs d_state 축 순서 (둘 다 128)
- **지금 → 제안**: `n_h_ssm / d_state 혼용` → `(소스가 가리키는 쪽 — 근거 참조)`  (확신 medium)
- **근거**: 남은 128건은 Mamba 내부의 진짜 값 충돌이다: n_h_ssm(128) == d_state(128) 이라 `view [B,T,n_g_ssm,n_h_ssm/n_g_ssm,d_state] -> [B,T,?,?]` 의 두 출력 축을 우선순위로만 가르면 순서가 뒤집힌다. 합쳐진 축이 무엇인지는 reshape 자체가 알고 있지만(파생 계산), 그걸 채택하려면 권위 있는 개명을 데이터플로우 끝까지 옮겨야 한다 — MLA `d_v` 건과 **같은 막힘**이다. 값으로 우기지 않고 남긴다.  **근거 소스**: 이 판정은 `develop/sources/modeling_nemotron_h.py`, `develop/sources/configuration_nemotron_h.py` 를 열어 확인했다. (인용 누락을 자가 점검에서 발견해 보강, 2026-08-12 — 게이트가 이제 `should_be_renamed` 판정에 소스 인용을 요구한다.)  **재분류 (2026-08-13)**: 이 판정은 `undetermined` 였다. 잘못된 분류다 — 근거 문장이 "트레이스 안에 가를 증거가 없다"고 적고 있었는데, 그건 *트레이스만으로는* 못 가른다는 말이지 *알 수 없다*는 말이 아니다. **소스는 답을 갖고 있다**(위 인용). 막는 것은 지식이 아니라 표현 수단이다: 두 이름이 같은 값이라 `label_overrides` 의 이름 치환으로는 갈

### A57. zai-org__GLM-5.2

- **모듈**: `model.layers.*.self_attn.indexer`
- **축**: rope 슬라이스 폭 64
- **지금 → 제안**: `d_head / n_h` → `d_rope`  (확신 high)
- **근거**: `modeling_glm_moe_dsa.py:225-229`: `q = q.view(B, S, self.n_heads, self.head_dim)` 뒤 `q_rot, q_pass = torch.split(q, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)`, `k = self.k_norm(self.wk(hidden_states)).unsqueeze(2)` 뒤 같은 split. indexer 의 head 폭은 `index_head_dim`=128 이고 이것이 64+64 로 쪼개진다 — 즉 이 모듈 안의 **모든 64 는 rope/nope 슬라이스**다. config.head_dim(=64)/num_attention_heads(=64)와 값이 같아 `d_head`/`n_h` 가 붙었지만, 트레이스가 `slice [B,T,1,n_h] -> [B,T,1,n_h_I]`(64→32)와 `concat [.., n_h_I]x2 -> [.., n_h]` 로 그 축을 반으로 쪼갰다 되붙이고 있다. head 개수는 반으로 쪼개지지 않는다.  **
- **막힌 이유(측정)**: 아직 반영하지 않은 이유(측정)**: 이 이름을 `rules/label_overrides.yaml` 로 적용해 봤더니 게이트 퇴행 검사가 걸렸다 — flow_ambig 438 -> 480, matmul_compose 0 -> 42. override 층은 **한 모듈 안의** 이름만 바꾸므로, 같은 텐서를 렌더하는 이웃 모듈이 옛 이름으로 남아 데이터플로우 불일치가 드러난다. 이름이 틀렸다는 판정 자체는 위 소스로 확정이고, 필요한 것은 '권위 있는 이름을 데이터

### A58. zai-org__GLM-5.2

- **모듈**: `model.layers.*.self_attn.indexer`
- **축**: interleaved rope 절반 32
- **지금 → 제안**: `n_h_I` → `d_rope/2`  (확신 high)
- **근거**: 위 항목과 같은 자리의 짝이다. `apply_rotary_pos_emb_interleave`(`modeling_glm_moe_dsa.py:232`)가 rope 슬라이스를 짝/홀로 갈라 32 를 만든다. `index_n_heads`(=32)와 값이 같아 head 개수 이름이 붙었으나, `[B, T, 1, ·]` 의 마지막 축은 feature 다 — 같은 행의 앞쪽에 head 축이 따로 있다.  **
- **막힌 이유(측정)**: 아직 반영하지 않은 이유(측정)**: 이 이름을 `rules/label_overrides.yaml` 로 적용해 봤더니 게이트 퇴행 검사가 걸렸다 — flow_ambig 438 -> 480, matmul_compose 0 -> 42. override 층은 **한 모듈 안의** 이름만 바꾸므로, 같은 텐서를 렌더하는 이웃 모듈이 옛 이름으로 남아 데이터플로우 불일치가 드러난다. 이름이 틀렸다는 판정 자체는 위 소스로 확정이고, 필요한 것은 '권위 있는 이름을 데이터


---

# 부록 B — 판단에 필요한 코드 원문

요약이 아니라 발췌다. 주석에 지금까지의 측정과 실패가 그대로 적혀 있으므로 **주석까지 읽어야 한다.**

## B1. `src/label_overrides.py` — 교정을 표에 넣는 층 (전문)

이 층이 지금 막혀 있는 지점이다. docstring 이 스스로 한계를 적어 두었다.

```python
"""Apply the ④-layer verdicts to the rendered tables.

WHY THIS EXISTS
---------------
Layers ①-③ decide a label from the rules. Layer ④ -- a reader with the source open -- sometimes
knows better, and until now that knowledge stopped at `review_findings.json`: the summary card said
"지금 렌더 / 소스가 말하는 것", while `full/*.csv` kept the wrong name. A judgement nobody can act
on is half a judgement.

The obvious fix -- change the rule so the labeller gets it right -- was tried twice and reverted
both times. It fails for the same reason each time: a value collision is not local. Renaming one
op leaves every neighbour on the old name, and the dataflow checks light up (DeepSeek MLA:
reshape_incons 61 -> 122, flow_ambig 0 -> 122). The rules decide per axis from a number; when two
config fields hold the same number there is no number to decide from.

So this does not re-derive anything. It REWRITES a specific label, everywhere it occurs under a
declared module, after all inference is done. That is safe precisely because it is not inference.

WHAT KEEPS IT HONEST
--------------------
An override is a claim, and every claim here has to pay for itself:

  * `source` is REQUIRED and must name the file and line it came from. No citation, no override.
  * `expect` is REQUIRED: the concrete size the axis must have. If the real tensor is not that
    size, the override does not fire -- it cannot silently paste a name onto the wrong axis.
  * The gate FAILS on an override that matched nothing. A claim that no longer applies (the model
    was re-traced, the rules improved, the label is already right) is a stale claim, and stale
    claims are how a table starts lying.
  * `layer_types` narrows to a block kind, for hybrid stacks where the same module name holds a
    Mamba block in one layer and attention in the next.
  * `axis` narrows to one axis POSITION (negative counts from the right) and `rank` to shapes of
    one length. Needed when the same number means two different things inside one shape and a
    blanket rename would destroy the half that is already right: DeepSeek-V4-Pro's CSA compressor
    builds `new_zeros((batch, n_windows, 2 * ratio, head_dim))` where n_windows == head_dim == 512,
    so `[B, ·, 2*m_csa, ·]` needs `T/m_csa` at axis 1 and `d_head` at axis -1 -- one rename applied
    to both would collapse them onto a single name. `rank` separates `[B, n_windows, head_dim]`
    from `[B, 1, n_windows, head_dim]`, where the window sits at a different index.

Scope, deliberately: this renames a label under a module. It does NOT follow a tensor along the
dataflow, so a collision that is only distinguishable by where the tensor came from (MLA's
`d_nope` vs `d_v`, both 128, both inside `self_attn`, both on `[B, n_h, T, ·]`) still cannot be
expressed. Those stay `open` with their source line, which is the truthful outcome.
"""
import os
import re

import yaml

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "rules", "label_overrides.yaml")
_CACHE = None


def load(path: str = _PATH) -> list:
    global _CACHE
    if _CACHE is None:
        if not os.path.exists(path):
            _CACHE = []
        else:
            with open(path, encoding="utf-8") as f:
                _CACHE = (yaml.safe_load(f) or {}).get("overrides") or []
    return _CACHE


def for_model(model_dir_name: str, path: str = _PATH) -> list:
    return [o for o in load(path) if o.get("model") == model_dir_name]


_LAYER_IDX = re.compile(r"\.(?:layers|h|blocks|block|layer)\.(\d+)(?:\.|$)")

# An override is hand-written, so it can be wrong in ways a rule cannot: it names an axis by
# POSITION, and a position that holds a sequence axis in one tensor holds a static width in
# another. Writing `T/m_csa` at axis 1 of DeepSeek-V4-Pro's compressor was right for every
# activation and wrong for `position_bias`, an `nn.Parameter(compress_rate, head_dim)` whose
# axis 1 is head_dim -- 31 weight axes came out claiming a static parameter is sized by the
# runtime sequence length. The gate's weight_T invariant caught it, but a layer that can only
# be corrected after the fact is a layer that will ship the error once. Refuse it here instead:
# no override may put a T-derived name into a weight, whatever the entry says.
_MENTIONS_T = re.compile(r"\bT\b")


def _schedule(cfg):
    for f in ("layers_block_type", "layer_types"):
        v = getattr(cfg, f, None)
        if isinstance(v, (list, tuple)) and v:
            return [str(x) for x in v]
    return None


def apply(rows: list, ordered: list, model_dir_name: str, cfg=None, path: str = _PATH) -> list:
    """Rewrite labels in `ordered` per the declared overrides. Returns one report dict each.

    `rows` carries the CONCRETE shapes and `ordered` the rendered ones, index-aligned -- the same
    pairing every other pass in build_table uses. The concrete side is what `expect` is checked
    against, so an override can never fire on an axis of the wrong size.
    """
    ovs = for_model(model_dir_name, path)
    if not ovs:
        return []
    sched = _schedule(cfg) if cfg is not None else None
    prepared = []
    for o in ovs:
        prepared.append({
            "spec": o,
            "rx": re.compile(o["module"]),
            "kinds": set(o.get("layer_types") or ()),
            "n": 0,
            "vetoed": 0,
        })

    from anchors import module_key
    for row, out in zip(rows, ordered):
        mk = module_key(row.get("module_path")) or "(root)"
        kind = None
        if sched:
            m = _LAYER_IDX.search(row.get("module_path") or "")
            if m and 0 <= int(m.group(1)) < len(sched):
                kind = sched[int(m.group(1))]
        for p in prepared:
            if not p["rx"].search(mk):
                continue
            if p["kinds"] and kind not in p["kinds"]:
                continue
            frm, to, want = str(p["spec"]["from"]), str(p["spec"]["to"]), p["spec"]["expect"]
            ax, rank = p["spec"].get("axis"), p["spec"].get("rank")
            for fld in ("input_shape", "output_shape", "weight_shape"):
                cvals, svals = row.get(fld), out.get(fld)
                if cvals is None or svals is None:
                    continue
                if fld == "weight_shape" and _MENTIONS_T.search(to):
                    p["vetoed"] += 1
                    continue
                pairs = ([(cvals, svals)] if fld == "weight_shape"
                         else list(zip(cvals, svals)))
                # The same parameter appears twice per op: once as `weight_shape` and once as the
                # operand at `weight_pos` inside `input_shape`. Vetoing only the former left
                # DeepSeek-V4-Pro's `position_bias` operand renamed and its weight not, which is
                # the "one tensor, two names" defect the gate calls weight_operand -- 31 rows.
                if _MENTIONS_T.search(to) and isinstance(row.get("weight_pos"), int):
                    wp = row["weight_pos"]
                    if fld == "input_shape" and 0 <= wp < len(pairs):
                        p["vetoed"] += 1
                        pairs = [x for i, x in enumerate(pairs) if i != wp]
                for cv, sv in pairs:
                    if not isinstance(cv, list) or not isinstance(sv, list) or len(cv) != len(sv):
                        continue
                    if rank is not None and len(sv) not in (
                            rank if isinstance(rank, (list, tuple)) else (rank,)):
                        continue
                    want_i = None if ax is None else (ax if ax >= 0 else len(sv) + ax)
                    for i, (c, s) in enumerate(zip(cv, sv)):
                        if want_i is not None and i != want_i:
                            continue
                        if str(s) == frm and isinstance(c, int) and c == want:
                            sv[i] = to
                            p["n"] += 1
    return [{"from": p["spec"]["from"], "to": p["spec"]["to"], "module": p["spec"]["module"],
             "expect": p["spec"]["expect"], "source": p["spec"].get("source", ""),
             "applied": p["n"], "vetoed": p["vetoed"]} for p in prepared]
```

## B2. `src/build_table.py:_gather_keeps_features()` — 2026-08-13 성공 사례

**연산의 정의**로 라벨을 고치고 `depends_on` 으로 소비자까지 전파한다. gpt-oss 의 flow_ambig 72/48 → 0. 일반화 후보 1번.

```python
def _gather_keeps_features(rows: list[dict], ordered: list[dict]) -> int:
    """A gather along dim 0 selects ROWS. It cannot rename the trailing axes.

    `index(x[T, d_model], idx[k*T]) -> [k*T, d_model]` is a definition: the op picks rows out of
    `x`, so every axis after the first is the same axis it was, whatever its width collides with.
    gpt-oss has d_model == d_moe == 2880 and the routed-token gather came out
    `[T, d_model], [k*T] -> [k*T, d_moe]` -- the expert intermediate width pasted onto the residual
    stream at the moment it enters the experts, and then carried down the whole gate/up chain. No
    value can catch that (the numbers are equal) and no rule was looking, because the axis is not
    declared by any module: it is produced by an indexing op. Found by outside review, 2026-08-13.

    Deliberately narrow: dim-0 gather only (same rank in and out, identical trailing concrete
    widths, index operand of rank 1). A gather on a later axis, or one that changes rank, says
    nothing about which axis survived and is left alone.
    """
    changed = 0
    fixed = {}          # op_id -> the corrected rendered output shape
    for row, out in zip(rows, ordered):
        if row.get("op_type") != "index":
            continue
        ins, outs = row.get("input_shape") or [], row.get("output_shape") or []
        louts = out.get("output_shape") or []
        lins = out.get("input_shape") or []
        if len(ins) < 2 or len(outs) != 1 or len(louts) != 1 or len(lins) != len(ins):
            continue
        src, idx, dst = ins[0], ins[1], outs[0]
        if not (isinstance(src, list) and isinstance(idx, list) and isinstance(dst, list)):
            continue
        if len(idx) != 1 or len(src) != len(dst) or len(src) < 2:
            continue
        if list(src[1:]) != list(dst[1:]):      # trailing widths must be untouched
            continue
        lsrc, ldst = lins[0], louts[0]
        if not isinstance(lsrc, list) or not isinstance(ldst, list) or len(lsrc) != len(ldst):
            continue
        hit = False
        for i in range(1, len(ldst)):
            if ldst[i] != lsrc[i]:
                ldst[i] = lsrc[i]
                changed += 1
                hit = True
        if hit and row.get("op_id") is not None:
            fixed[row["op_id"]] = (list(dst), list(ldst))
    # Carry it to the consumers of that exact tensor. _propagate_labels is monotone -- it fills
    # integers and never overwrites a name -- so without this the gather's output read `d_model`
    # while the very next op read `d_moe` on the same tensor (gpt-oss: 72 axes on 120b, 48 on 20b,
    # all a `masked_fill_` applying the routing mask to the gathered hidden states). Renaming one
    # end and leaving the other is the defect, not the fix.
    if fixed:
        for row, out in zip(rows, ordered):
            deps = [d for d in (row.get("depends_on") or []) if d in fixed]
            if not deps:
                continue
            for oid in deps:
                cshape, lshape = fixed[oid]
                for cv, sv in zip(row.get("input_shape") or [], out.get("input_shape") or []):
                    if not isinstance(cv, list) or not isinstance(sv, list):
                        continue
                    if list(cv) != cshape or len(sv) != len(lshape):
                        continue
                    for i in range(1, len(sv)):
                        if sv[i] != lshape[i]:
                            sv[i] = lshape[i]
                            changed += 1
    return changed
```

## B3. `src/build_table.py:_carry_reshape_labels()` — 실패 사례 (호출되지 않음)

같은 목표를 reshape 에 대해 시도했다가 되돌린 것. **왜 실패했는지가 핵심이다.**

```python
def _carry_reshape_labels(rows: list[dict], ordered: list[dict]) -> int:
    """A view cannot change what an axis MEANS -- carry the input's label onto the output.

    Only the 1:1 axes are carried (an input axis that passes through a reshape untouched), never
    the merged ones: a merge product is derivable but the existing label may be a better-attested
    spelling of the same thing. Forward direction, because the input label came from the producer,
    which sits closer to the tensor's origin (the residual stream, an nn.Linear's declared width).

    What this fixes: wherever `d_model == n_h*d_head` -- Qwen2.5-0.5B (896 = 14*64), SmolLM3-3B
    (2048 = 16*128), and every other model with that coincidence -- the flatten in front of q/k/v_proj
    re-labelled the RESIDUAL STREAM as the packed head layout, because inside a `self_attn` scope
    the head formula outranks the plain symbol. The input side had it right all along.
    """
    n = 0
    for row, out in zip(rows, ordered):
        if row.get("op_type") not in _VIEW_FAMILY:
            continue
        cin_all, sin_all = row.get("input_shape") or [], out.get("input_shape") or []
        if not cin_all or not sin_all or not isinstance(cin_all[0], list):
            continue
        cin, sin = cin_all[0], sin_all[0]
        if not isinstance(sin, list) or len(sin) != len(cin):
            continue
        for sout, cout in zip(out.get("output_shape") or [], row.get("output_shape") or []):
            if not isinstance(cout, list) or not isinstance(sout, list):
                continue
            for idx, lab in derive_from_reshape(cin, sin, cout, merges=False).items():
                if idx < len(sout) and str(sout[idx]) != lab and not lab.isdigit():
                    sout[idx] = lab
                    n += 1
    return n
```

## B4. `src/build_table.py:_propagate_labels()` — 단조 전파 (이름을 안 덮는다)

```python
def _propagate_labels(rows: list[dict], ordered: list[dict]) -> None:
    """Carry a resolved axis label along the dataflow, in place.

    A tensor that flows from op A's output into op B's input is ONE tensor, so it must read the
    same in both places. Each op is otherwise labelled in isolation, which lets one side name an
    axis while the other leaves a bare integer: Nemotron-3-Nano names every block `mixer`, so the
    `n_kv*d_head` rule (scoped to attention-ish module names) fired inside `mixer.k_proj` but not
    in the parent `mixer`, and the same 1024-wide tensor came out `n_kv*d_head` then `1024`.

    Deliberately MONOTONE: only a bare integer is ever replaced, and only by the producer's label
    for the identical concrete tensor. It can add information but never overwrite a considered
    choice, so it cannot introduce the kind of regression a re-prioritisation can. Genuine
    two-concepts-one-value ambiguities (gpt-oss d_model/d_ff/d_moe all 2880) are left alone --
    those are documented, not guessable. Found by the dataflow audit, 2026-07-30."""
    by_id = {r.get("op_id"): (r, o) for r, o in zip(rows, ordered)}
    _IDENTITY = ("clone", "_to_copy", "contiguous", "detach", "alias", "copy_")
    for _pass in range(3):        # a fill-in can enable the next one; converges quickly
        changed = False
        for row, out in zip(rows, ordered):
            # An op that only copies cannot rename an axis, so a bare integer on one side takes
            # the other side's name. Same monotone rule as the cross-op fill below, applied
            # WITHIN the op -- xLSTM's backbone copies an axis already named `d_model*qk_f/n_h`
            # into a buffer and the output kept a bare 256.
            if row.get("op_type") in _IDENTITY:
                for ci, si in zip(row.get("input_shape") or [], out.get("input_shape") or []):
                    for co, so in zip(row.get("output_shape") or [], out.get("output_shape") or []):
                        if not (isinstance(ci, list) and isinstance(co, list) and ci == co):
                            continue
                        if not (isinstance(si, list) and isinstance(so, list)
                                and len(si) == len(so)):
                            continue
                        for i, (a, b) in enumerate(zip(si, so)):
                            if str(b).isdigit() and not str(a).isdigit():
                                so[i] = a
                                changed = True
                            elif str(a).isdigit() and not str(b).isdigit():
                                si[i] = b
                                changed = True
            for dep in (row.get("depends_on") or []):
                prod = by_id.get(dep)
                if not prod:
                    continue
                p_row, p_out = prod
                for bi_c, bi_s in zip(row.get("input_shape") or [],
                                      out.get("input_shape") or []):
                    if not isinstance(bi_c, list) or not bi_c:
                        continue
                    for ao_c, ao_s in zip(p_row.get("output_shape") or [],
                                          p_out.get("output_shape") or []):
                        if not isinstance(ao_c, list) or ao_c != bi_c:
                            continue      # not the same tensor
                        # Bidirectional: whichever side is still a bare integer takes the other
                        # side's name. Either end can be the unresolved one -- a scope can match
                        # the child module but not the parent, or the reverse.
                        for i, (mine, theirs) in enumerate(zip(bi_s, ao_s)):
                            if str(mine).isdigit() and not str(theirs).isdigit():
                                bi_s[i] = theirs
                                changed = True
                            elif str(theirs).isdigit() and not str(mine).isdigit():
                                ao_s[i] = mine
                                changed = True
        if not changed:
            break


# A reshape carries its own evidence: `view [T,k] -> [T*k]` says that output axis IS the product
# of those two input axes, with no value matching involved. Deriving it independently and comparing
# is the cheapest cross-check we have on intermediate-tensor labels -- the ones with no module to
# ask. Measured across 26 models: 97.6% of derivable axes AGREE with the labels already there,
# which is strong evidence for both; the disagreements are where to look. The first two inspected
# were real (DeepSeek-V4-Flash's indexer projections, where d_model=4096 collides with
# n_h*d_head/g_o=4096 and the scoped formula won over the residual stream -- DeepSeek-V4-Pro runs
# the same module at d_model=7168 with no collision and says d_model).
#
# Used as an AUDITOR, not a labeller: adopting the derived names outright would have changed only
# 1,024 axes (0.4%), so it earns its place by finding errors, not by filling gaps.
_VIEW_FAMILY = {"view", "reshape", "_unsafe_view", "flatten", "unflatten",
                "clone", "_to_copy", "contiguous", "detach", "alias", "squeeze", "unsqueeze",
                # `copy_` too: xLSTM's backbone copies an axis already named `d_model*qk_f/n_h`
                # into a buffer and the name was dropped, leaving a bare 256.
                "copy_"}
```

## B5. `src/anchors.py:propagate()` — 이름을 **덮어쓰는** 전파 (이미 동작 중)

앵커가 정한 라벨에만 쓰인다. 소스 판정을 여기에 태울 수 있는지가 질문 2와 직결된다.

```python
def propagate(rows, ordered, authoritative: dict, passes: int = 4) -> int:
    """Spread anchor-decided labels along the dataflow, overwriting named labels.

    relabel() fixes the tensor where the module declares it, but the SAME tensor is also
    rendered on the producing op's output and on every consumer's input, and those were left
    reading the old name. That is not cosmetic: the gate's dataflow check counts one tensor
    with two names, and correcting only one side turned Zamba2 from 0 to 36 such mismatches.

    Distinct from build_table._propagate_labels, which is deliberately monotone (bare integers
    only) so it can never overwrite a considered choice. Here the overwrite is the point --
    an anchor IS the considered choice, and it outranks whatever value matching produced. The
    label only travels between shapes that are concretely IDENTICAL, so it stays a statement
    about one tensor.
    """
    by_id = {r.get("op_id"): (r, o) for r, o in zip(rows, ordered)}
    total = 0
    for _p in range(passes):
        moved = 0
        for row, out in zip(rows, ordered):
            oid = row.get("op_id")
            for dep in (row.get("depends_on") or []):
                pair = by_id.get(dep)
                if not pair:
                    continue
                p_row, p_out = pair
                for bi, (bc, bs) in enumerate(zip(row.get("input_shape") or [],
                                                  out.get("input_shape") or [])):
                    if not isinstance(bc, list) or not bc:
                        continue
                    for ai, (ac, as_) in enumerate(zip(p_row.get("output_shape") or [],
                                                       p_out.get("output_shape") or [])):
                        if not isinstance(ac, list) or list(ac) != list(bc):
                            continue
                        for axis in range(min(len(bs), len(as_))):
                            mine = ("input_shape", bi, axis) in authoritative.get(oid, set())
                            theirs = ("output_shape", ai, axis) in authoritative.get(dep, set())
                            if mine and not theirs and as_[axis] != bs[axis]:
                                as_[axis] = bs[axis]
                                authoritative.setdefault(dep, set()).add(
                                    ("output_shape", ai, axis))
                                moved += 1
                            elif theirs and not mine and bs[axis] != as_[axis]:
                                bs[axis] = as_[axis]
                                authoritative.setdefault(oid, set()).add(
                                    ("input_shape", bi, axis))
                                moved += 1
        total += moved
        if not moved:
            break
    return total


# An unnamed dimension leaking into a tag expression means rules/symbols.yaml has no name for
# that width, so the tag is WORSE than the resolver's rendering there: DeepSeek-V3's q_b_proj
# comes out `192*n_h` (192 is the unregistered qk_head_dim) and Zamba2's mamba in_proj carries a
# bare 8192. Small coefficients are fine and common (`2*d_moe`, `2*n_h*d_head`); what disqualifies
# a tag is a literal standing in for a DIMENSION -- large, or added as its own term.
_BIG_LITERAL = re.compile(r"(?<![\w.])(?:1[6-9]|[2-9]\d|\d{3,})(?![\w.])")
# A literal added as its OWN term (`...+8192`, `(n_h+2)`) is an unnamed dimension. A literal that
# is followed by `*` is a coefficient of the next term (`...+2*n_h_lin_v*d_head_lin_v`) and is
# perfectly normal, so it must not disqualify the expression.
_ADDED_LITERAL = re.compile(r"[+\-]\s*\d+(?!\s*[*\w.])|(?<![\w.*])\d+(?=\s*[+\-])")
```

## B6. `develop/verify_all.py:_dataflow_label_check()` — flow_ambig 이 세는 것

우리를 막고 있는 그 검사다. 무엇을 불일치로 세는지 정확히 보고 질문 4에 답해 달라.

```python
def _dataflow_label_check(d, phase="prefill"):
    """(provably-wrong mismatches, benign-ambiguous mismatches) along the dependency graph.

    FIRST-PRINCIPLES CHECK, and the only one here that can catch a bug class nobody anticipated:
    a tensor flowing from op A's output into op B's input is ONE tensor, so it must read the same
    in both places. No domain knowledge required -- it needs only the graph and the concrete
    shapes, so it does not depend on us having seen the failure mode before. It is what would
    have caught DeepSeek-V4's `g_o` vs `T/m_hca` automatically instead of by eye.

    Two outcomes are separated because only one is a defect:
      wrong  -- one side is a bare integer while the other has a name (information dropped), or
                the same product is spelled two ways (`E*T` vs `T*E`). Always fixable; FAILs.
      ambig  -- both sides carry a real name and the model makes them numerically equal
                (gpt2-xl: d_model == n_h*d_head by construction; gpt-oss: d_model == d_ff ==
                d_moe == 2880). Both names are TRUE for that tensor, so forcing one would destroy
                information rather than add it. Tracked as a number so a sudden jump is visible.
    """
    raw = os.path.join(d, "full", f"{phase}.trace.raw.jsonl")
    conc = os.path.join(d, "full", f"{phase}.shapes.concrete.jsonl")
    if not (os.path.exists(raw) and os.path.exists(conc)):
        return 0, 0
    sym = {json.loads(l)["op_id"]: json.loads(l) for l in open(raw, encoding="utf-8")}
    con = {json.loads(l)["op_id"]: json.loads(l) for l in open(conc, encoding="utf-8")}
    wrong = ambig = 0
    for oid, r in sym.items():
        cr = con.get(oid)
        if not cr:
            continue
        for dep in (r.get("depends_on") or []):
            ar, ac = sym.get(dep), con.get(dep)
            if not ar or not ac:
                continue
            for bi_s, bi_c in zip(r.get("input_shape") or [], cr.get("input_shape") or []):
                if not isinstance(bi_c, list) or not bi_c:
                    continue
                for ao_s, ao_c in zip(ar.get("output_shape") or [], ac.get("output_shape") or []):
                    if not isinstance(ao_c, list) or ao_c != bi_c:
                        continue
                    for ls, lo in zip(bi_s, ao_s):
                        ls, lo = str(ls), str(lo)
                        if ls == lo:
                            continue
                        if ls.isdigit() or lo.isdigit():
                            wrong += 1                       # information dropped on one side
                        elif sorted(re.split(r"[*+]", ls)) == sorted(re.split(r"[*+]", lo)):
                            wrong += 1                       # same expression, two spellings
                        else:
                            ambig += 1                       # two true names, equal by construction
    return wrong, ambig
```

