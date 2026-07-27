# 블라인드 검증 #2 — tiiuae/falcon-7b : Phase 0-A 가설

**작성 시점: modeling 코드 열기 전, 트레이스 실행 전.** 검증 #1은
`zamba2-phase0a-hypotheses.md`. 아래 예측은 이후 결과와 대조되며 **틀린 것도 수정하지 않는다.**

## 출처 (A단계)
- [The Falcon Series of Open Language Models (arXiv:2311.16867)](https://arxiv.org/pdf/2311.16867)
- [tiiuae/falcon-7b 모델카드](https://huggingface.co/tiiuae/falcon-7b), falcon-40b 카드

## 문서에서 파악한 구조
1. **병렬 attention + MLP.** attention과 MLP가 직렬이 아니라 **병렬**로 놓이고,
   **블록당 LayerNorm 하나**만 쓴다. GPT-J(Wang & Komatsuzaki, 2021)에서 도입된 방식.
   텐서 병렬 시 all_reduce를 레이어당 2회 → 1회로 줄이는 게 동기.
2. **선형 계층 bias 제거.**
3. **Multi-query attention** — KV head를 극단적으로 줄여 KV 캐시를 크게 절감.

## 오염 고지
대상 선정을 위해 config **값**을 조회했다(`falcon-rw-1b`는 `parallel_attn=False`·ALiBi라
병렬 잔차 테스트가 안 되고, `falcon-7b`가 `parallel_attn=True`). 따라서 이번엔 지난번보다
오염이 크다 — config에서 온 예측은 `[config]`로 표기한다. modeling 코드와 트레이스는 미열람.

---

## 예측 (falsifiable)

### H1 — `n_kv` 심볼이 **거짓말**한다 ★핵심 예측
`[config]` falcon-7b는 `num_kv_heads=71`(= `num_attention_heads`)인데 동시에
`multi_query=True`다. 즉 **필드 값과 실제 동작이 다르다** — 실제 KV head는 1개다.

우리 `rules/symbols.yaml`의 `n_kv` 별칭은 `[num_key_value_heads]`뿐이라 Falcon의
`num_kv_heads`를 **못 잡는다**. 그러면 `resolve_symbols`의 결정적 폴백
("GQA 필드 없으면 MHA이므로 n_kv = n_h")이 발동해 `n_kv = 71`이 된다.

⇒ **예측: C7이 "MHA (kv_heads == heads)"로 PASS를 내지만 실제 모델은 MQA다.**
이건 조용히 틀린 결과이므로 지금까지 만든 어떤 체크로도 안 잡힐 것이다.
조치는 별칭 추가만으로는 부족하고 `multi_query` 플래그를 함께 봐야 한다.

### H2 — 병렬 잔차라 레이어당 잔차 add가 **직렬 2회가 아니다**
표준 블록은 `h = h + attn(ln1(h))` → `h = h + mlp(ln2(h))`로 **순차 2회**다.
Falcon은 `h = h + attn(ln(h)) + mlp(ln(h))`로 **같은 h에서 갈라져 한 번에 합쳐진다**.
- 레이어 구조 표에 `input_layernorm`이 **1개만** 보이고 `post_attention_layernorm`이 없을 것
- C5(잔차 불변식)는 마지막 축 기준으로 보므로 **PASS할 것**으로 예상하나,
  "38/38층 잔차" 같은 카운트가 어떻게 잡히는지는 확인 필요
- 의존성 그래프에서 attn 입력과 mlp 입력이 **같은 노드**를 가리킬 것 (fork)

### H3 — `d_head`는 config 필드가 없고 `d_model / n_h`로 유도
`[config]` `hidden_size`=4544, `num_attention_heads`=71 ⇒ `d_head`=64.
71은 소수라 흔한 2의 거듭제곱 충돌이 없어, 심볼 충돌 관점에서는 오히려 깨끗할 것이다.

### H4 — bias 없음
`[config]` `bias=False`. 선형 계층 파라미터가 weight만 존재 → C10 커버리지에 bias 없음.

### H5 — C17이 WARN을 낸다
`falcon`이 `rules/structures/` 어디에도 없고, 병렬 잔차 구조 문서도 없다.
(현재 `residual/`에는 standard와 mHC뿐.) WARN이 안 나오면 게이트가 고장난 것이다.

### H6 — 새로 등록해야 할 구조
- `rules/structures/residual/parallel.md` (병렬 attention+MLP, 단일 LN) — 신규
- `n_kv` 별칭에 `num_kv_heads` 추가 + `multi_query` 처리
- ALiBi는 이 모델엔 없지만(`alibi=False`) `falcon-rw-1b`엔 있으므로 후속 과제

### H7 — 트레이싱 자체는 성공
표준 attention + MLP라 특수 커널 의존이 없다. Tier 0/1 조치 없이 통과할 것.

---

## 채점 기준
| 항목 | 기준 |
|---|---|
| A단계 적중률 | H1~H7 중 몇 개가 확인되는가 |
| **조용한 오류 탐지** | H1(MQA를 MHA로 오판)이 실제로 일어나는가. 일어난다면 **기존 체크가 못 잡는 결함**을 블라인드 테스트가 찾아낸 것 |
| 절차 유효성 | C17이 WARN을 냈는가 |
| 완결성 | 등록 후 `verify_all.py` FAIL 0 |

---
---

# 채점 결과 (B·C·D 완료 후 추가 기록, 위 가설은 무수정)

## H1 — ✅ **적중. 그리고 이번 검증의 최대 수확.**
예측대로 **조용히 틀린 결과**가 나왔다:
- `n_kv = 71` (실제 1), `Attention: MHA` (실제 MQA)
- `C7 PASS  MHA (kv_heads == heads, not GQA)` — **틀린 내용으로 PASS**
- **KV cache 카드 568.0 KiB (Very high)** — 실제는 **8.0 KiB (Very low)**, **71배 과대평가**

C1~C17 어느 체크도 못 잡았다. 트레이스 자체는 진실을 갖고 있었다 — fused QKV 가중치 폭이
`4672 = (71+2)·64`라 KV head가 1개임이 드러나 있었는데, 우리가 config 필드를 그대로 믿었다.

**근본 원인 2곳**:
1. `n_kv` 별칭에 Falcon의 `num_kv_heads`가 없어 폴백("GQA 필드 없으면 MHA")이 발동
2. `validate.c7_gqa`가 `resolve_symbols`를 안 쓰고 **config에서 독립적으로 다시 유도** —
   같은 버그가 두 곳에 존재

**조치**: 별칭 추가 + `multi_query` 규칙을 공식 소스 그대로 반영
(`num_kv_heads if (new_decoder_architecture or not multi_query) else 1`),
그리고 C7이 심볼 테이블(단일 출처)을 읽도록 수정. 재트레이싱 후
`C7 PASS  MQA (71 query heads : 1 kv head)`, KV 카드 `8.0 KiB (Very low)`.

## H2 — ✅ 적중
`layer 0-31: input_layernorm, mlp, self_attention` — **LayerNorm이 하나뿐**이고
`post_attention_layernorm`이 없다. C5는 예측대로 PASS(`32/32 layers`).

## H3 — ✅ 적중  `d_head` = 4544/71 = 64 (config 필드 없이 폴백으로 유도)
## H4 — ✅ 적중  bias 없음, C10 195개 파라미터 전부 커버
## H5 — ✅ 적중  C17 WARN (미해결 상수 `4672` + `falcon`이 구조 라이브러리에 없음)
## H6 — ✅ 적중  `residual/parallel.md` 신규 작성, `n_kv` 별칭 추가, fused QKV 파생 규칙 추가
## H7 — ✅ 적중  Tier 0/1 조치 없이 트레이싱 성공

## 최종
| | |
|---|---|
| A단계 적중 | **7/7** (검증 #1 Zamba2는 4/6) |
| 발견한 결함 | **1건 — 조용히 틀린 KV cache(71배) + 잘못된 attention 계열 판정** |
| 등록물 | `residual/parallel.md`, `n_kv` 별칭+multi_query 규칙, fused QKV 파생 규칙, C7 단일출처화 |
| 최종 검증 | `verify_all.py` FAIL 0 / 퇴행 0 |

## 남은 과제 (사람 확인용으로 남김)
- **ALiBi 미문서화**: `falcon-rw-1b`이 ALiBi를 쓰는데 `position_encoding/`에 문서가 없다.
  이번 대상(falcon-7b)은 RoPE라 범위 밖이라 손대지 않았다.
- **C7의 근본 강화**: 지금은 "config 의미론을 정확히 아는 것"에 의존한다. 더 견고하게 하려면
  fused QKV 폭 `(n_h+2·n_kv)·d_head`를 **트레이스에서 역산해 교차검증**하는 게 맞다.
  이번엔 새 버그를 만들 위험을 피해 단일출처화까지만 했다.
