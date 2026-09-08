# 검토 요청 — Kimi-K3 값 충돌 4건 (외부 소스만으로 판단)

이 프로젝트(`llm-arch-tracer`)는 HuggingFace LLM을 meta device에서 트레이스해 텐서 축마다
`n_h`, `d_head` 같은 아키텍처 심볼 이름을 붙인다. **이 요청에는 파이프라인 코드를 전혀
포함하지 않았다** — 아래 트레이스 사실(실측 shape·op 시퀀스)과 공개 소스만으로 판단해 달라.

## 배경 — Kimi-K3 (moonshotai/Kimi-Linear, `kimi_linear`/`kimi_k3` model_type)

93개 레이어의 하이브리드 아키텍처: 69개는 **KDA**(Kimi Delta Attention, linear attention,
`fla` 라이브러리의 `naive_chunk_kda` 참조 구현으로 도는 커널), 24개는 **MLA**(Multi-head
Latent Attention). **두 레이어 유형이 파이썬 클래스는 다르지만(`KimiDeltaAttention` vs
`KimiMLAAttention`) 모듈 속성 이름은 둘 다 그냥 `self_attn`이다** — 그래서 트레이스만 보면
어느 레이어가 어느 유형인지 모듈 경로로는 안 갈리고, `layer_idx`로만 갈린다.

`linear_attn_config`(KDA 설정): `num_heads=96`, `head_dim=128`, `short_conv_kernel_size=4`.
아래 레이어 0은 KDA 레이어로 이미 확인된 것이다(`kda_layers` config 목록에 포함).

## 값 충돌 4건 — 실측 트레이스

각 항목은 `prefill.trace.raw.jsonl`에서 그대로 뽑은 실제 shape이다(`B`=배치, `T`=시퀀스
길이, 나머지는 아직 확정 못한 후보 이름들 중 하나여야 하는 자리).

### 1. 값 96 — `n_h` / `n_h_kda` / `n_kv` 후보

```
op 113 (layer 0, self_attn, view): [B, T, n_h*d_v] -> [B, T, ?96, d_head_kda]
op 125 (layer 0, self_attn, view): [B, T, n_h*d_v] -> [B, T, ?96, d_head_kda]
op 130 (layer 0, self_attn, view): [B, T, n_h*d_v] -> [B, T, ?96, d_head_kda]
```

`?96` 위치가 지금 3개의 무관한 후보(`n_h`=MLA의 query head 수, `n_h_kda`=KDA의 head 수,
`n_kv`=MLA의 kv head 수)와 값이 같다. layer 0이 KDA라는 게 이미 확정이니 상식적으로는
`n_h_kda`가 맞아 보이지만, **입력 쪽 `n_h*d_v`라는 표기 자체가 이미 의심스럽다** —
KDA 레이어에 `n_h`(MLA 개념)와 `d_v`(MLA value 폭)가 섞인 이름이 붙어 있다는 것 자체가
값 충돌로 잘못 붙은 이름일 가능성이 있다.

### 2. 값 128 — `d_head_kda` / `d_nope` / `d_v` 후보

```
op 109 (layer 0, self_attn.f_b_proj, view): [B, T, d_head_kda] -> [T, ?128]
op 113 (layer 0, self_attn, view, 위와 같은 op): 출력 마지막 축 = ?128
```

`?128`이 `d_head_kda`(KDA head 폭, config 확정값 128), `d_nope`/`d_v`(MLA의 non-rope/value
폭)와 겹친다.

### 3. 값 6144 — `2*d_moe` / `E_shared*d_moe` 후보

```
op 38937 (layer 1, block_sparse_moe.shared_experts, concat):
  [B, T, E_shared*d_moe], [B, T, E_shared*d_moe] -> [B, T, ?6144]
op 38940 (layer 1, block_sparse_moe.shared_experts.act_fn, slice):
  [B, T, ?6144] -> [B, T, E_shared*d_moe]
```

MoE 공유 전문가(shared expert) 블록의 gate/up 두 프로젝션을 concat한 폭이다. `2*d_moe`
(공유 전문가가 1개뿐이라 `E_shared=2`인 경우와 값이 같음) vs `E_shared*d_moe`(공유 전문가
개수 × 폭, 이게 더 원래 의도에 가까워 보이지만 검증 필요) 중 어느 쪽이 실제 소스 구조를
반영하는지 확인 필요.

### 4. 값 64 — `d_chunk` / `d_rope` 후보

```
op 162 (layer 0, self_attn, view): [B, T, n_h_kda, d_head_kda] -> [B, 5, ?64, n_h_kda, d_head_kda]
op 164 (layer 0, self_attn, permute): [B, 5, ?64, n_h_kda, d_head_kda] -> [B, n_h_kda, 5, ?64, d_head_kda]
```

layer 0은 KDA(RoPE 없음, `naive_chunk_kda`의 청크 스캔만 있음)인데 `?64` 자리가 `d_chunk`
(청크 스캔 크기)와 `d_rope`(RoPE 절반 차원, MLA 레이어에만 있어야 할 개념)로 동시에
설명된다. `5`라는 축은 아마 청크 개수(`T/chunk_size` 근사)로 보인다.

## 확인해 줬으면 하는 것

각 항목에 대해:
1. **정확한 이름은 무엇인가** — 위 후보 중 하나, 또는 후보에 없는 다른 이름이라면 그것.
2. **근거** — 어디서 확인했는지(모델 저장소의 remote code, `fla` 라이브러리의
   `naive_chunk_kda`/`naive_recurrent_kda` 구현, Kimi-Linear 논문/블로그 등). 실행 코드
   파일명과 줄 번호(또는 함수명)를 정확히 인용해 달라.
3. **1번 항목의 `n_h*d_v`라는 입력 표기 자체가 의심스러운 이유**도 판단에 참고해서, 이게
   "값이 우연히 같아서 잘못 붙은 이름"인지 "실제로 그런 의도인지"도 언급해 주면 좋겠다.

## 참고 — 소스를 어디서 찾을 수 있는지

- 모델 저장소: `moonshotai/Kimi-Linear-48B-A3B-Instruct` (또는 관련 체크포인트)의 remote
  code (`modeling_kimi_linear.py` 류 파일).
- `fla`(flash-linear-attention) 라이브러리의 `fla/ops/kda/naive.py` —
  `naive_chunk_kda`/`naive_recurrent_kda`가 실제 청크 스캔의 축 순서를 정의한다.
- Kimi-Linear 기술 리포트/블로그(공개돼 있다면).

이 요청서 하나로 판단 가능하다 — 저장소 전체나 우리 파이프라인 코드는 필요 없다.
