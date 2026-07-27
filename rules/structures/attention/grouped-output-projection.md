# Grouped Low-Rank Output Projection (블록대각 그룹 o_proj)

## 정의
attention 출력 `o_proj`를 `[d_model, n_h*d_head]` 단일 dense 행렬로 두지 않고,
head를 `g`개 그룹으로 쪼개 **그룹마다 독립적인 저랭크 투영**을 건 뒤 합치는 방식
(DeepSeek-V4 paper §2.3.1 "Grouped Output Projection").

```
attn_out [B, T, n_h*d_head]
  → reshape [B, T, g, n_h*d_head/g]
  → o_a_proj: 블록대각 그룹 linear, 그룹당 (n_h*d_head/g) → d_g      # bmm
  → [B, T, g, d_g] → flatten → [B, T, g*d_g]
  → o_b_proj: [d_model, g*d_g] 단일 linear
  → [B, T, d_model]
```

동기: head가 많고 `d_head`가 큰 모델은 `n_h*d_head`가 매우 커진다(V4-Flash 32768,
V4-Pro **65536**). `65536 → 7168` 직접 투영은 레이어당 470M 파라미터로 토큰당 비용을
지배해버린다. `g`개 블록으로 쪼개고 `d_g < n_h*d_head/g`로 저랭크를 걸면
`g·(n_h*d_head/g)·d_g + g·d_g·d_model`로 줄어든다 — V4-Pro 기준 67M+117M ≈ 184M로 약 2.5배 절감.

블록대각이라 **그룹 간 정보가 `o_a_proj` 단계에서는 섞이지 않는다**. 섞임은 뒤따르는
`o_b_proj` 하나가 전담한다.

MoE의 grouped-GEMM과 혼동 주의: 저쪽은 **expert 축**으로 쪼갠 것이고 토큰마다 다른 그룹이
선택되지만, 여기는 **head 축** 고정 분할이라 모든 토큰이 전 그룹을 통과한다.

## 관련 심볼 (rules/symbols.yaml)
`n_h`, `d_head`, `d_model`. V4 고유 필드는 아직 별칭 없음(SKIP):
`o_groups`(g), `o_lora_rank`(d_g).

## 트레이스에서 식별하는 방법 (DeepSeek-V4-Pro 실측)
`self_attn` 아래에 **`o_proj` 대신 `o_a_proj` + `o_b_proj` 쌍**이 있고, `o_a_proj`가
`linear`가 아니라 **`aten.bmm`(batched_matmul)** 으로 잡힌다:

| op_id | 모듈 | op | shape |
|---|---|---|---|
| 524 | `o_a_proj` | `view` | weight `[16384, 4096]` → `[16, 1024, 4096]` |
| 525 | `o_a_proj` | `transpose` | → `[16, 4096, 1024]` |
| 526–527 | `o_a_proj` | `view`/`transpose` | 입력 `[B,T,16,4096]` → `[16, B*T, 4096]` |
| **528** | `o_a_proj` | **`batched_matmul`** | `[16,T,4096] × [16,4096,1024]` → `[16,T,1024]` |
| 529–530 | `o_a_proj` | `transpose`/`view` | → `[B,T,16,1024]` |
| 533 | `o_b_proj` | `t` | weight `[7168, 16384]` |

즉 `g`=16, `n_h*d_head/g` = 65536/16 = **4096**, `d_g`=1024, `g*d_g` = **16384**.
`weight.view(g, -1, in)` → `transpose(1,2)` → `bmm` → `transpose` 시퀀스가 블록대각 그룹 linear의
지문이다.

> **심볼 렌더링 주의 (알려진 표기 결함).** T=2048 실행에서 상수 4096이 `2*T`로 렌더된다
> (op 524~528의 `[16384, 2*T]`, `[16, 2*d_head, 2*T]`). 4096은 `n_h*d_head/g`라
> **seq_len과 무관한 config 고정 차원**인데 시퀀스 의존처럼 보이게 만든다. 마찬가지로
> `d_g`=1024가 `2*d_head`로 렌더된다(값은 맞지만 유래가 아님). `01-main.md` §6의
> "심볼로 못 되면 정수로 남긴다(P1)" 원칙에 어긋나는 케이스 —
> `resolve_seq_len`의 충돌 회피가 **파생 표현식**(`2*T`)까지는 못 막는다.
> `g*d_g`=16384는 정수로 남아 있어 이 문제를 피했다.

## 확인된 모델 (계속 추가)
- **`deepseek-ai/DeepSeek-V4-Pro`** (예약 최종테스트, 2026-07-23): `n_h`=128, `d_head`=512,
  `g`=16, `d_g`=1024, `d_model`=7168 ⇒ 그룹당 4096→1024, 합쳐 16384→7168.
  revision `b5968e9190ef611bbf34a7229255be88a0e937c1`.
- **`deepseek-ai/DeepSeek-V4-Flash`** (예약 최종테스트, 2026-07-23): `n_h`=64, `g`=8,
  `d_g`=1024, `d_model`=4096 ⇒ 그룹당 4096→1024, 합쳐 8192→4096.

두 모델 모두 [csa.md](csa.md)/[hca.md](hca.md) 레이어와 무관하게 **모든 attention 레이어**에
적용된다.

## 참고 소스
- transformers 5.14.1 `models/deepseek_v4/modular_deepseek_v4.py`
  — `DeepseekV4GroupedLinear`(docstring에 V4-Flash/Pro 수치 명시), `DeepseekV4Attention.__init__`.
  **트레이스로 직접 관측**(1차 소스)
- `configuration_deepseek_v4.py` docstring — `o_groups`, `o_lora_rank`
- DeepSeek-V4 technical report §2.3.1 "Grouped Output Projection" — 교차검증용
