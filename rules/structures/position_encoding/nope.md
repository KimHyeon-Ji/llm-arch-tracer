# NoPE — 일부 레이어에서 위치 인코딩 생략

## 정의
대부분 레이어는 RoPE를 쓰되, **주기적으로 일부 레이어에서 위치 인코딩을 아예 생략**(No
Positional Encoding)하는 스케줄. 긴 문맥 일반화에 도움이 된다고 알려져 있다. 레이어별로
RoPE 적용/미적용이 갈리므로 **이종(heterogeneous) 스케줄**의 한 형태.

## 관련 심볼 / config
`no_rope_layers`(레이어별 0/1 리스트, 1=RoPE 적용·0=생략), `no_rope_layer_interval`(주기).
RoPE 자체 파라미터는 `rope_parameters`/`rope_theta`. (표준 RoPE는 [rope.md](rope.md).)

## 트레이스에서 식별하는 방법
- RoPE 적용 레이어: self_attn 안에 `cos`/`sin`/`neg`(rotate_half)/`elementwise_mul` 시퀀스가
  q·k에 걸린다.
- NoPE 레이어: 그 RoPE op 시퀀스가 **없다** → op-타입 시퀀스가 달라져 **C2 클러스터링이 두
  클러스터로 분리**한다(RoPE층 vs NoPE층).
- **C2 검증**: config의 per-layer 스케줄 리스트(`no_rope_layers` 등, 길이=num_layers)를
  layer_types와 조합해 레이어별 시그니처를 만들고 distinct 개수를 기대 클러스터 수로 쓴다.
  SmolLM3는 layer_types 균일 + `no_rope_layers` 2값 → 기대 2, 트레이스 2 → C2 PASS.
  (NoPE는 layer_types에 안 실리므로, layer_types만 보던 옛 C2는 오탐 FAIL였다 — Phase 12에서 수정.)

## 확인된 모델
- **`HuggingFaceTB/SmolLM3-3B-Base`** (Phase 12): 36 layers, GQA 16:4, RoPE(θ=5e6) + **NoPE
  every 4th layer**(`no_rope_layers=[1,1,1,0,...]`, interval 4 → 9개 NoPE 층). C2 PASS(2==2),
  C7 GQA 16:4, C13 repro. revision `d78a42f79198603e614095753484a04c10c2b940`.
- **`meta-llama/Llama-4-Maverick-17B-128E`** (예약 최종테스트, 2026-07-27): 48 layers,
  GQA 40:8, `d_head`=128, `no_rope_layers` 길이 48 = `[1,1,1,0]` 반복 → **4층마다 1층 NoPE**
  (SmolLM3와 같은 주기). MoE 인터리브(`interleave_moe_layer_step`=2)와 **직교**하게 걸려서,
  major view가 3블록으로 나뉜다: 짝수층 = attn+dense FFN, 홀수층 = attn+MoE인데 그 홀수층이
  다시 RoPE/NoPE로 갈린다. C2는 트레이스 3클러스터 > config 스케줄 시그니처 2로 **WARN**인데,
  이는 결함이 아니라 NoPE 세분화를 C2 기대값 계산이 아직 못 반영한 것이다.
  Llama-4는 여기에 더해 **chunked attention**(`attention_chunk_size`=8192)과
  **attention temperature tuning**(`attn_temperature_tuning`, `floor_scale`=8192)을 쓴다.

## 참고 소스
- transformers `models/smollm3`, `models/llama4` 구현 — 트레이스로 직접 관측(NoPE 층에 RoPE op 부재)
- config: `no_rope_layers`, `no_rope_layer_interval`; Llama-4는 `text_config` 아래에 있다
  (`provenance.snapshot`이 `cfg.get_text_config()`로 내려감)
- Raschka's LLM Architecture Gallery(NoPE 계보; 교차검증용)
