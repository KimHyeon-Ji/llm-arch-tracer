# Codex 검토 요청 — gpt-oss / Llama-4-Maverick 최근 수정 3건 점검

## 배경

세션에서 Llama-4-Maverick과 gpt-oss-20b/120b를 "100% 확신 가능" 상태로 만들려고 작업했다.
그중 두 모델은 **외부 검토(Codex, 별도 요청)가 이미 원인을 짚어준** 버그를 고친 것이고,
gpt-oss의 self_attn 타이는 이번 세션에서 스스로 찾아 고친 것이다. 커밋하고 결과 브랜치에
올리기 전에, 이 수정들이 실제로 맞는지 다시 한번 확인받고 싶다.

## 확인할 범위

```
git show 57881fdc --stat
git show 57881fdc -- rules/label_overrides.yaml rules/label_confirmed.yaml src/summarize.py rules/symbols.yaml rules/derived_dims.yaml
```

세 개 모델, 세 종류의 수정이다.

## 1. gpt-oss-20b/120b — `down_proj` 가중치 미스라벨

`GptOssExperts.down_proj` 가중치의 가운데 축이 `d_model`로 렌더되고 있었는데, 실제로는
`d_moe`여야 한다는 게 이번 수정의 주장이다.

```python
# modeling_gpt_oss.py:77-82
self.intermediate_size = config.intermediate_size
self.hidden_size = config.hidden_size
self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, 2 * self.intermediate_size))
self.down_proj = nn.Parameter(torch.empty((self.num_experts, self.intermediate_size, self.hidden_size)))
```

이 체크포인트는 `intermediate_size == hidden_size`(둘 다 2880)라 값으로는 두 축을 못 가리는데,
선언 순서(`down_proj`의 가운데 축 = `intermediate_size`)로는 명확하다고 판단해 고쳤다.
`gate_up_proj`의 가운데 축(`hidden_size`)은 처음부터 `2*intermediate_size != hidden_size`라
값이 안 겹쳐서 원래도 맞았다는 게 대조 근거다.

**확인해줄 것**: 이 선언 순서 해석이 맞는가? `nn.Parameter`가 `(E, mid, out)` 순서로
쌓였다는 것과, 실제 forward 코드(`grouped_matmul` 호출부)가 이 축 순서를 그대로 쓰는지도
봐줬으면 한다.

## 2. gpt-oss-20b/120b — self_attn `n_h` vs `d_head` 위치 규칙

`head_dim == num_attention_heads`(둘 다 64)인 체크포인트라 값으로는 못 가리는 self_attn
축 114개(양쪽 phase 합산 7,536+ 축)를, "표준 view+transpose 레이아웃에서 축 **위치**는
값과 무관하게 항상 같은 뜻"이라는 근거로 확정했다.

```python
# modeling_gpt_oss.py:320-324
hidden_shape = (*input_shape, -1, self.head_dim)
query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
```

주장: `.view(hidden_shape).transpose(1,2)` 뒤 4D 텐서는 축1=head 개수, 축3=head 폭이고,
`transpose(1,2)`는 축1/2만 바꾸므로 축3(마지막)은 절대 안 건드린다 — 이 위치 규약이
`head_dim`이 `num_attention_heads`와 값이 같아지는 것과 무관하게 항상 성립한다.

**확인해줄 것**: 이 위치 규약이 앞서 코드(sinks, attention 커널 호출, GQA-expand 등)에서도
일관되게 유지되는지, 값이 우연히 같아지는 지점에서 실제로 위치가 뒤바뀌는 예외가 있는지
(특히 `apply_rotary_pos_emb`나 attention sink 처리부처럼 축 순서를 바꿀 수 있는 연산).

## 3. SmolLM3 / Llama-4-Maverick — 요약문 생성 코드의 NoPE·FFN 스케줄 표시 버그

이건 첫 번째 Codex 검토가 지적했던 것을 고친 것이다 — **라벨/축 데이터 자체는 원래도
맞았고**, `model_summary.md`를 만드는 `src/summarize.py`가 그 정보를 안 쓰고 있었다.

근본 원인 두 가지:
```python
# 이전 (버그): 레이어마다 값이 다른 필드를 "불일치"로 보고 지움
def _first_attr(cfg, aliases, default=None):
    for a in aliases:
        if hasattr(cfg, a):
            return _per_layer_scalar(getattr(cfg, a))  # no_rope_layers=[1,1,1,0,...] -> None!
    return default
```
`no_rope_layers`는 원래 레이어마다 값이 다른(0/1 섞인) **스케줄**인데, `_per_layer_scalar`가
"레이어마다 다르면 불일치니까 None"으로 처리하는 함수라 스케줄 자체가 지워졌다. 고친 방식은
`no_rope_layers`만 `_first_attr`를 안 거치고 `getattr`로 raw 값을 직접 읽게 한 것.

```python
# 이전 (버그): 이 두 필드명만 인식
fk = dd.get("first_k_dense_replace") or dd.get("n_dense_layers")
```
LFM2는 `num_dense_layers`, ERNIE는 `moe_layer_start_index`, Llama-4는 `moe_layers`(prefix가
아니라 홀수 레이어만 MoE인 리스트)를 쓰는데 이 중 아무것도 인식을 못 해서 전부 dense 레이어가
0개인 것처럼("L× MoE") 나오고 있었다. 세 필드명을 추가하고 `moe_layers`는 리스트 길이로
따로 처리하게 고쳤다.

**확인해줄 것**:
1. `_per_layer_scalar`를 완전히 우회하는 대신 raw `getattr`를 쓴 게 안전한 선택인지 — 다른
   필드(예: `layer_types`처럼 이미 다르게 처리되는 필드)와 충돌하거나, `no_rope_layers`가
   실제로는 없는데 `getattr`가 조용히 `None`이 아닌 다른 값을 주는 경우가 있을 수 있는지.
2. `moe_layers`(명시적 리스트)와 `fk`(prefix 카운트) 중 `moe_layers`를 우선하도록 했는데,
   두 필드가 동시에 존재하면서 서로 다른 것을 가리키는 체크포인트가 있을 수 있는지.
3. Llama-4-Maverick 실측: `no_rope_layers`가 4번째마다 0(48개 중 12개)이고 이게 정확히
   `full_attention`(iRoPE의 non-rotary 레이어)과 일치한다고 확인했다 — 이 대응이 우연이
   아니라 아키텍처상 항상 성립하는지도 봐줬으면 한다.

## 답변 형식

세 항목 각각: **동의 / 이견 있음**, 이견이면 정확히 어느 부분이 왜 틀렸는지, 가능하면
`modeling_gpt_oss.py`/`modeling_llama4.py`의 줄 번호로.
