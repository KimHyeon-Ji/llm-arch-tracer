# Codex 검토 요청 — DeepSeek-V4-Pro (결과 브랜치 17개 중 유일하게 미검토)

## 배경

`results` 브랜치(GitHub의 결과물 전용 브랜치)에 17개 모델이 올라가 있는데, 그중
DeepSeek-V4-Pro만 아직 외부 검토를 안 받았습니다(나머지 16개는 이미 별도로 검토받아
반영·수정 완료). 이 모델은 MLA + Lightning Indexer + hyper-connection(mHC) + MTP가 겹친
가장 복잡한 아키텍처라 특히 검토가 필요합니다.

DeepSeek-V4-Pro는 실제 발표 논문이 없는(또는 이 환경에 없는) 체크포인트라, "공식 소스"는
설치된 `transformers==5.14.1`의 `modeling_deepseek_v4.py` / `configuration_deepseek_v4.py`와
체크포인트 자체의 `config.json`입니다. **이 저장소의 추출 파이프라인 코드(`src/`, `rules/`)는
보지 말고**, 이 두 가지(transformers 소스 + config)만으로 아래 주장들을 독립적으로
재현·검증해주세요.

## 1부 — 기본 아키텍처 수치 검증

`model_summary.md`가 주장하는 값들입니다:

- 레이어 수 L=61, d_model=7168, hidden_size 등
- Attention: MQA — query head 128개, KV head 1개, d_head=512 (query/key/value 모두 단일 헤드
  폭 512), sliding window 128 (전 레이어 적용) + 블록 압축 분기 2종류(HCA m=128 / CSA m=4,
  31층 HCA + 30층 CSA)
- 위치 인코딩: RoPE, θ=10000, 부분 회전(d_rope=64, d_head=512 중 64만 회전)
- MoE: routed experts E=384, top-6, shared experts 1개(E_shared=1), expert intermediate
  d_moe=3072, SwiGLU
- KV cache: 압축 레이어당 `d_head/m` (HCA는 512/128=4, CSA는 512/4=128) elems/token,
  K==V 단일 텐서 — 계산 결과 7.74 KiB/token
- context = 1,048,576 (config max_position_embeddings)
- q_lora_rank(c_q)=1536, o_lora_rank/o_groups로 만들어지는 grouped output projection 그룹 수
  g_o=16
- Lightning Indexer: index_n_heads=64, index_head_dim=128, index_topk 존재
- Multi-token prediction(MTP) 레이어가 config에 있는지, structure.yaml에 반영이 안 돼 있는지

**확인해줄 것**: 이 수치들이 transformers 5.14.1의 `modeling_deepseek_v4.py`/
`configuration_deepseek_v4.py`, 그리고 이 체크포인트의 실제 `config.json`과 일치하는가?
(1부 이전 라운드에서 실제 오픈 모델 14개를 이 방식으로 검토했을 때 6개에서 진짜 버그가
나왔습니다 — 레이어 스케줄 오기재, 심볼 값 누락 등. 같은 수준으로 꼼꼼히 봐주세요.)

## 2부 — 이번에 새로 고친 것 검증

값이 우연히 겹치는 자리가 많은 모델입니다:
- `num_attention_heads`(128) == `sliding_window`(128) == `index_head_dim`(128)
- `index_n_heads`(64) == `qk_rope_head_dim`(64)

이 우연 때문에 축 이름이 값만으로는 안 갈리는 자리 102곳을, "표준 view/transpose 레이아웃에서
축 **위치**는 값과 무관하게 항상 같은 뜻"이라는 근거로 확정했습니다. 예:

```python
# modeling_deepseek_v4.py:811,817 (self_attn 클래스)
hidden_shape = (*input_shape, -1, self.head_dim)
q = self.q_b_proj(q_residual).view(*hidden_shape).transpose(1, 2)
```
→ transpose 뒤 4D 텐서는 축1=head 개수(n_h), 축3=head 폭(d_head)이 항상 성립 —
`num_attention_heads`가 `sliding_window`와 값이 같아져도 무관.

**추가로 실측 트레이스를 직접 추적해서 새 유도값 하나를 찾았습니다**: Lightning Indexer의
q/k rotate_half 분할 축이 `n_h_I`(index_n_heads=64)도 `d_rope`(qk_rope_head_dim=64)도 아니라
`c_I/2`(index_head_dim=128의 정확히 절반)였습니다. 근거:

```python
# modeling_deepseek_v4.py:497 self.head_dim = config.index_head_dim (c_I=128)
# 실측 op 그래프 (prefill op_id 1874/1886/1887):
# slice [B,1,d_head,c_I] -> [B,1,d_head,X]   (X = c_I의 정확히 앞쪽 절반)
# 병렬 branch가 나머지 절반 처리
# concat([X_half1, X_half2]) -> [B,1,d_head,c_I]   (rotate_half 재조합)
```

**확인해줄 것**:
1. `hidden_shape=(*input_shape,-1,head_dim); .view().transpose(1,2)` 관례가 self_attn 클래스의
   q/k/v/attention-output/sinks 전체에서 일관되게 유지되는지 — rope 적용부나 sliding-window
   마스킹부에서 축 순서가 바뀌는 예외가 있는지.
2. `c_I/2` 판정 — indexer가 정말 `index_head_dim`(128) 전체를 rotate_half 방식으로 회전시키는지,
   아니면 부분 회전(`qk_rope_head_dim`만)인지. 만약 indexer가 실제로는 outer MLA와 같은
   `qk_rope_head_dim`(64)만 부분 회전시키고 나머지 64는 그냥 통과시키는 구조라면, 저희가
   `c_I/2`로 판정한 게 틀렸을 수 있습니다 — 이 경우 진짜 정답은 `d_rope`일 수 있습니다.

## 3부 — 의도적으로 미해결로 남긴 것

`self_attn.o_a_proj`의 grouped output projection에서 그룹 축(`g_o`=16)이 곱해진 결과
(`g_o*d_g`)에서만 나타나고, 그 값 자체를 독립적으로 확인할 방법이 트레이스 안에 없습니다
(값 역산 문제 — 이 저장소가 과거 이런 유형에서 3번 회귀를 낸 이력이 있어 손대지 않기로
결정했습니다). 이게 정말 트레이스만으로는 못 푸는 문제가 맞는지, 혹시 소스 어딘가에
`g_o`를 직접 읽을 수 있는 다른 경로가 있는지만 가볍게 봐주시면 좋겠습니다(안 보이면
그대로 두셔도 됩니다).

## 답변 형식

1부는 항목별로 **일치/불일치**, 불일치면 정확한 값과 근거(파일:줄).
2부는 각 판정에 **동의/이견**, 이견이면 정확히 어디가 왜 틀렸는지.
3부는 짧게 답 있음/없음만.
