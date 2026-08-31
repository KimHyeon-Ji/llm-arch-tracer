# Codex 검토 요청 — 2026-08-30 세션 전체 규칙 추가분 통짜 점검

## 배경

오늘 세션에서 review_ledger가 "미수행"/"만료"로 보고한 모델들을 하나씩 재검토했다.
Falcon-H1의 `d_chunk`/`d_state` 다단계 역추론(가장 위험도 높다고 판단한 부분)은 이미
별도로 검토받아 실제 버그 2건을 찾아 고쳤다(`2aae51d5` 커밋). 이번엔 **그때 안 본 나머지
전부**를 한 번에 봐 달라 — 각각은 Falcon-H1보다 훨씬 단순한 추론(대부분 "이 체크포인트는
GQA가 아니라 MHA라 두 번째 후보 심볼 자체가 불가능하다")이지만, 물량이 많고
(confirm 항목 총 135개, override 9개, derived_dims 2개) 외부 확인은 한 번도 안 거쳤다.

## 확인할 범위

```
git log --oneline 013ad058..20626ece
git diff 013ad058..20626ece -- rules/derived_dims.yaml rules/label_confirmed.yaml rules/label_overrides.yaml
```

Falcon-H1 관련(`d16ec386`, `2aae51d5`)은 이미 검토받았으니 **제외**하고, 그 뒤 커밋들만 봐주면 된다:

- `71f3cd77` gpt2-xl — n_h/n_kv confirm 20건
- `337c2565` Llama-4-Maverick — `derived_dims.yaml`에 `expr: "E * T"` (scope: `feed_forward|experts`)
- `b37244d1` OLMoE — n_h/n_kv confirm 28건
- `fc53bd1d`, `bf76dcd0`, `5f278d62` — xLSTM-7b/MiniMax-M2/Hunyuan-A13B (규칙 변경 없음, 순수 재확인이라 스킵해도 됨)
- `b15ad5d6` LFM2-8B — `derived_dims.yaml`에 `expr: "3 * d_model"` (scope: `conv`)
- `b7d78d0b` Nemotron-3-Nano — d_head/d_state, n_g_ssm/n_kv confirm 59건 (가장 크고 가장 복잡한 배치)
- `20626ece` OLMo-2 — n_h/n_kv confirm 28건

## 각 배치의 핵심 주장 (검증해달라는 것)

### 1. gpt2-xl / OLMoE / OLMo-2 — "이 체크포인트는 MHA라 n_kv가 후보 자체가 안 된다"

세 모델 다 같은 패턴으로 주장했다: `num_key_value_heads`가 `num_attention_heads`와
같거나(OLMoE/OLMo-2는 `if num_key_value_heads is None: = num_attention_heads`),
GPT-2는 애초에 `num_key_value_heads` 필드가 아예 없다(GQA 이전 아키텍처). 그래서 값이
겹치는 `n_h`/`n_kv` tie에서 실제 렌더는 전부 `n_h`이고(트레이스에 `n_kv` 문자열이 전혀
안 나타남을 확인함), 그걸 confirm으로 등록했다.

**의심되는 지점**: OLMo-2의 원래 2026-08-12 판정은 "Q 쪽은 n_h, KV 쪽은 n_kv로 모듈
단위로 갈린다"고 적혀 있었는데, 오늘 재확인하며 "실제로는 self_attn 전체가 n_h로만
렌더된다"고 **정정**했다. 이 정정이 맞는지, 아니면 내가 뭔가를 놓쳐서 실제로는 K/V
쪽에 `n_kv`가 나와야 하는데 못 찾은 건지 확인해달라.

### 2. Llama-4-Maverick — `E * T` (Llama4TextMoe.forward, modeling_llama4.py:169)

```python
routed_in = hidden_states.repeat(router_scores.shape[1], 1)
```

`router_scores.shape[1] == num_local_experts`이므로 `[T,d_model]`을 전문가 수만큼
반복한 `[E*T, d_model]`이 나온다고 보고, `expr: "E * T", scope: "feed_forward|experts"`를
등록했다. dropless MoE(모든 전문가가 모든 토큰을 계산)라는 전제가 맞는지, scope가 다른
모델의 `feed_forward`/`experts` 모듈에 있는 무관한 값과 우연히 곱해져 오탐을 낼 위험은
없는지 봐달라.

### 3. LFM2-8B — `3 * d_model` (modeling_lfm2_moe.py:406)

```python
self.in_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=self.bias)
```

short-conv 블록의 B/C/x 세 갈래 융합 투영이라고 보고 `expr: "3 * d_model", scope: "conv"`를
등록했다. 이것도 scope가 너무 넓어서 다른 모델의 `conv` 모듈(Mamba류의 `d_inner+2*n_g*d_state`
같은 값)과 우연히 3배 관계가 맞아 오탐할 가능성이 있는지 봐달라.

### 4. Nemotron-3-Nano — d_head/d_state(128), n_g_ssm/n_kv(8) confirm 59건 (가장 위험도 높다고 봄)

이 모델은 Mamba2 mixer와 full_attention이 둘 다 `mixer`라는 같은 이름을 쓰는 하이브리드
스택이다. `configuration_nemotron_h.py:97-111`에서 `num_key_value_heads=8`과
`n_groups=8`(mamba_n_groups)이 서로 다른 필드, `head_dim=128`과 `ssm_state_size=128`도
서로 다른 필드임을 확인했고, 59개 앵커 전부 이미 주변 축(Mamba 문맥엔 `n_h_ssm`/`d_chunk`/
`d_head_ssm`, attention 문맥엔 `n_h`/`T`/GQA-expand)만으로 명확히 갈려 있어서 교정 없이
전부 confirm으로 등록했다.

**확인해달라는 것**: 59개 앵커 중 실제로 Mamba↔attention이 뒤바뀐 자리가 섞여 있는지 —
특히 `[B, n_kv, n_h/n_g_ssm, T, d_head]`처럼 `n_kv`와 `n_g_ssm`이 **같은 shape 안에
동시에 나타나는 게 아니라 각각 다른 행에서만 나타나는지**(같은 행에 두 심볼이 같이
있으면 위험 신호), 그리고 `d_head`(attention)와 `d_state`(Mamba)가 실제로 값이 다른
체크포인트가 있는지(Nemotron-3-Super/Ultra도 128로 같은지, 다르면 이 확인이 Nano에만
유효하고 다른 크기엔 안 맞을 수 있음).

## 이미 게이트로 확인한 것 (다시 안 물어봐도 됨)

- `develop/verify_all.py` FAIL 0 / 퇴행 0 (매 커밋마다 확인, baseline 2회 갱신은 진짜
  heur 개선이 감지된 경우만)
- `develop/verify_selftest.py` 66/66 (죽은 검사 없음)
- 각 confirm/override 항목은 스키마상 `source`에 파일:줄 인용이 있어야 하고, 매치 0건이면
  dead verdict로 게이트가 FAIL시킨다 — 전부 통과함

## 질문

1. 위 4개 배치 중 실제로 문제 있는 자리가 있는가?
2. 있다면 정확히 어느 파일의 어느 항목(`model`/`shape`/`nth` 조합)인지 짚어달라 —
   `rules/label_confirmed.yaml`/`label_overrides.yaml`에서 해당 모델명으로 검색하면 된다.
3. `E*T`/`3*d_model` derived_dims 두 항목의 scope가 너무 넓어서 다른 모델에 오탐을 낼
   가능성이 있는지 — 있다면 스코프를 어떻게 좁히면 좋을지도 제안해달라.
