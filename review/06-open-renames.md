# 미반영 교정 판정 43건 — 설계 자문 요청

소스로 **이름이 틀렸다는 것까지 확정했지만 산출물에 반영하지 못한** 판정이 43건 있다.
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

> 아래를 그대로 복사해 사용한다. 저장소 파일을 읽을 수 있는 환경이어야 한다.

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

**먼저 읽을 것**:
1. `review/06-open-renames.md` — 이 문서 (43건 목록 · 측정치 · 실패 기록)
2. `src/label_overrides.py` — 반영 층의 docstring 에 한계가 적혀 있다
3. `src/anchors.py` 의 `propagate()` — 이미 있는 데이터플로우 전파
4. `src/build_table.py` 의 `_gather_keeps_features()` — 최근 성공한 방식
5. `src/build_table.py` 의 `_carry_reshape_labels` docstring — 실패한 방식과 측정치
6. `models/deepseek-ai__DeepSeek-V3/review_findings.json`, `models/Zyphra__Zamba2-1.2B/review_findings.json` — 대표 사례 2개

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
5. 43건 중 **먼저 손대야 할 순서**는? (한 메커니즘으로 몇 건이 풀리는지 기준)

**지켜야 할 제약**:
- `develop/verify_all.py` 가 **EXIT 0, 퇴행 0** 이어야 한다. 이것이 합격선이다.
- 값으로 우기지 않는다. 근거 없는 이름은 붙이지 않고, 모르면 `undetermined` 로 남긴다.
- 새 검사를 추가하면 `develop/verify_selftest.py` 에 결함 주입을 함께 넣어 **그 검사가 살아
  있음을 증명**해야 한다. (실제로 이 프로젝트에서 검사 두 개가 조용히 죽어 있었다.)
- FLOPs · 메모리 대역폭 · latency 는 범위 밖이다.

**자세**: 확인하지 말고 **반박하라.** 위 §2 의 진단("모듈 단위 치환이 원인")이 틀렸다는 증거를
먼저 찾아라. 원인이 다른 곳에 있다면 그것을 말해 달라 — 우리가 두 번 실패한 것은 진단이
틀렸기 때문일 수도 있다.
