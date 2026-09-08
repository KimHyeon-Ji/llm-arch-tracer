# 배치 E 판정 — Codex CSV/JSONL 라벨 주장 검증

대상: deepseek-ai__DeepSeek-V3, ibm-granite__granite-4.0-h-small,
meta-llama__Llama-3.1-405B, tencent__Hunyuan-A13B-Instruct
근거 소스: 설치된 transformers 5.14.1. 판정일 2026-09-03.

| # | 모델 | 주장 | 판정 |
|---|---|---|---|
| E-1 | DeepSeek-V3 | MLA value 경로 `d_nope`→`d_v` | **REJECTED** |
| E-2 | DeepSeek-V3 | q/k RoPE 및 rotary-table `d_head`→`d_rope` | **REJECTED** |
| E-3 | Granite-4.0-H-Small | mamba scan/chunk permute `d_state`·`n_h_ssm` 혼용 | **CONFIRMED** |
| E-4 | Llama-3.1-405B | 이상 없음 | — |
| E-5 | Hunyuan-A13B | 이상 없음 | — |

## E-1 REJECTED — DeepSeek-V3 의 value 경로는 이미 정확하다

layer 3 self_attn 을 op 단위로 따라간 결과:

    op 465 split_with_sizes [B,n_h,T,d_nope+d_v] -> [B,n_h,T,d_nope], [B,n_h,T,d_v]
    op 517 view             [B,n_h,T,d_v] -> [n_h,T,d_v]
    op 518 batched_matmul   [n_h,T,T] x [n_h,T,d_v] -> [n_h,T,d_v]
    op 521 transpose        [B,n_h,T,d_v] -> [B,T,n_h,d_v]      (입출력 일치)
    op 523 view             [B,T,n_h,d_v] -> [B,T,n_h*d_v]

전 모델 전치 불변식 전수 검사에서도 DeepSeek-V3 는 **위반 0건**이었다.

## E-2 REJECTED — `d_head` 라벨이 존재하지 않는다

prefill 전체 스캔 결과 `d_head` 는 **0건**이다(d_rope 2,882 / d_nope 1,525 / d_v 1,159).
이미 `d_rope` 로 렌더링돼 있다.

## Codex 의 DeepSeek MLA 주장은 3개 중 1개만 맞았다

같은 문구의 주장이 세 모델에 반복됐다:

| 모델 | 판정 | 실제 |
|---|---|---|
| bzantium__tiny-deepseek-v3 (C-8/C-9) | REJECT | 이미 정확 |
| deepseek-ai__DeepSeek-V3 (E-1/E-2) | REJECT | 이미 정확 |
| deepseek-ai__DeepSeek-V2-Lite (D-2/D-3) | **CONFIRM** | 전치 위반 54행 + concat 자체 모순 |

**계열 단위로 일괄 적용했다면 정확한 라벨 2개 모델분을 망가뜨렸을 것이다.**

## E-3 CONFIRMED — Granite mamba permute

`n_h_ssm = d_state = 128` 충돌. permute/transpose 불변식 위반 **180행**:

    permute   [B,n_h_ssm,1,d_chunk]          -> [B,1,d_chunk,d_state]            x72
    permute   [B,1,d_chunk,d_state]          -> [B,n_h_ssm,1,d_chunk]            x36
    permute   [B,n_h_ssm,1,d_chunk,d_chunk]  -> [B,1,d_chunk,d_chunk,d_state]    x36
    transpose [B,n_h_ssm,2,2]                -> [B,2,2,d_state]                  x36

소스가 어느 쪽이 맞는지 명시한다 -- `modeling_granitemoehybrid.py:665`:

    # [bsz, -1, chunk_size, num_heads] -> [bsz, num_heads, -1, chunk_size]
    A = A.permute(0, 3, 1, 2)

네 자리 모두 헤드 축이며 `num_heads`(=`n_h_ssm`)다. `d_state` 쪽이 틀렸다.
(`L.permute(0,2,3,4,1)`, `decay_states.permute(0,-2,-1,1)`, `decay_chunk.transpose(1,3)`
도 각각 대응한다.)

---

**집계: CONFIRMED 1 / REJECTED 2 / 3건 (이상 없음 2모델 제외).**

Granite 는 override 4개 작성 → 재트레이스 → promote 완료
(발화 6,552, 미발화 0, permute/transpose 위반 0, C-FAIL 0).
