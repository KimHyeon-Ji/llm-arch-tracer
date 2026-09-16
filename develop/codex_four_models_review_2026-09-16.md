# 네 모델의 op 구성·축 라벨 검토 답변

검토일: 2026-09-16. 대상: `codex_ask_four_models_final.md`, `codex_ask_four_models_claims.md`, 네 모델의 실제 prefill/decode CSV·JSONL 및 full 원시 trace·concrete shape·axis_resolution.

판정: **주요 투영·attention·expert 연산의 제시된 shape는 대체로 맞다. 질문에 든 `B*T`, `B*n_h`, `B*k*T`도 맞다. 그러나 표만으로 실제 forward의 모든 구조를 정확히 표현한다고 판정할 수는 없다.** 부록은 실제 표의 서로 다른 블록들을 버렸고, 실제 요약 표에도 모델 고유 계산이 생략된다. 축의 수치 일치와 축 이름의 의미 일치는 별도다.

소스 기준은 설치된 `transformers==5.14.1`이다. 아래 세 modeling 파일을 공개 `v5.14.1` raw 파일과 다운로드 후 비교했고, 세 파일 모두 내용이 정확히 같았다. 소스 줄 번호는 **로컬 파일 기준**이다.

- `.venv/Lib/site-packages/transformers/models/llama4/modeling_llama4.py`
- `.venv/Lib/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py`
- `.venv/Lib/site-packages/transformers/models/deepseek_v4/modeling_deepseek_v4.py`

공개 소스: [Llama-4](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/llama4/modeling_llama4.py), [gpt-oss](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gpt_oss/modeling_gpt_oss.py), [DeepSeek-V4](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/deepseek_v4/modeling_deepseek_v4.py).

## 1. 부록이 구조 전부라는 주장은 틀림 — 실제 결과 파일과 구분해야 함

`claims.md:3-5`의 `block_type 별 대표 1 반복`은 서로 다른 attention/routing 블록을 같은 문자열로 묶어 버린다. 아래 op_id는 **요약 JSONL의 ID**이고 prefill/decode가 같지 않은 곳은 따로 적었다.

| 모델 | 부록에 빠진 실제 표의 블록 | 실제 위치 |
|---|---|---|
| Llama-4 | full attention + NoPE + MoE, layers `3,7,...,47`, repeat 12 | 양 phase op 41–64 |
| gpt-oss-20b/120b | full attention + MoE, 홀수 layers, repeat 12/18 | 양 phase op 23–44 |
| V4-Pro | CSA + hash MoE, layer 2, repeat 1 | prefill 51–108 / decode 49–102 |
| V4-Pro | HCA + 동적 MoE, layers `3,5,...,59`, repeat 29 | prefill 109–158 / decode 103–150 |
| V4-Pro | CSA + 동적 MoE, layers `4,6,...,60`, repeat 29 | prefill 159–216 / decode 151–204 |

따라서 부록만 보면 Llama-4는 36층, gpt-oss는 12/18층, V4-Pro는 2층만 표현한다. **실제 CSV·JSONL에서 해당 층들이 사라졌다는 뜻은 아니다.** 실제 표의 반복 수를 합치면 각각 48, 24/36, 61층이다.

교정: `block_type` 하나로 대표를 선택하지 말고 서로 다른 layer 집합과 attention/position/routing 구성을 가진 템플릿을 모두 유지해야 한다.

근거: Llama-4 `configuration_llama4.py:179-200`, `modeling_llama4.py:337-385,418-422`; gpt-oss `modeling_gpt_oss.py:288,308,485-496`; V4 `configuration_deepseek_v4.py:266-282`, `modeling_deepseek_v4.py:797-798,1088-1089` 및 위 실제 JSONL 행.

### gpt-oss의 반복 단위

레이어 쌍을 하나의 superblock으로 표현하는 방식 자체는 가능하다. 하지만 **현재 JSONL은 짝수 레이어 템플릿과 홀수 레이어 템플릿에 각각 자신의 `layers`를 붙인다.** 예컨대 op 1은 `0,2,...,22`, op 23은 `1,3,...,23`이다. 동일한 `block_type` 문자열만으로 “현재 반복 단위가 레이어 쌍”이라고 해석하면 안 된다.

### V4-Pro의 마지막 norm 순서도 부록에서 변함

부록 `claims.md:258-272`는 norm끼리 먼저 모으면서 `model.norm`을 `hc_head`보다 앞에 놓는다. 실제 순서는 `hc_head.input_norm → hc_head linear/sigmoid/weighted sum → model.norm → lm_head`다. 실제 prefill op 217–223 / decode 205–211의 순서는 맞다. 근거: `modeling_deepseek_v4.py:967-971,1336`.

## 2. gpt-oss-20b/120b — expert bias와 clamp가 실제 요약 표에서 빠짐

대상: 양 phase `model.layers.{0,1}.mlp.experts`; 요약 op 13/35 및 19/41의 grouped_matmul 주변.

현재 구성은 gate/up grouped GEMM 다음에 바로 GLU 계산, down grouped GEMM 다음에 바로 routing weight 곱셈이다. 실제 계산에는 다음이 추가된다.

- gate/up GEMM 결과에 expert별 bias를 더함. 원래 parameter shape는 `[E, 2*d_moe]`, 선택 후 `[S, 2*d_moe]`.
- GLU의 gate와 up에 서로 다른 clamp를 적용함. gate는 상한 7, up은 [-7, 7].
- down GEMM 결과에 expert별 bias를 더함. 원래 parameter shape는 `[E, d_model]`, 선택 후 `[S, d_model]`.

여기서 `S=B*k*T`(prefill), `S=B*k`(decode)다. 요약 op 17/39의 add는 **`up + 1`**이며 이 두 bias add를 대신하지 않는다. grouped_matmul 행의 `raw_op`도 순수 `_grouped_mm`이며, 해당 bias parameter가 `params`에 포함되지 않는다.

근거: `modeling_gpt_oss.py:80-92,113-116`; `.venv/Lib/site-packages/transformers/integrations/moe.py:369-378,435-468`.

원시 trace에서 직접 확인되는 위치:

| 모델/phase | gate/up bias add | gate/up clamp | down bias add |
|---|---:|---|---:|
| 20b prefill | 195 | 198,199 | 207 |
| 20b decode | 181 | 184,185 | 193 |
| 120b prefill | 201 | 204,205 | 213 |
| 120b decode | 181 | 184,185 | 193 |

교정: 별도 행을 유지하거나, bias 및 clamp까지 포함하는 복합 op라는 것을 입력·parameter·계산 의미에 명시해야 한다. 구조 완전성을 요구할 때의 누락이며, 현재 `structure.yaml`이 밝힌 major-op 선택이라는 범위에서는 요약 생략으로 분류할 수 있다. **새 축 오명명으로 분류하는 지적은 아니다.**

## 3. V4-Pro — mHC에 필요한 Sinkhorn 계산이 표에 없음

대상: 양 phase 모든 `model.layers.*.attn_hc`, `ffn_hc`의 softmax 이후. 대표 요약 op는 prefill attn 5 / ffn 31, decode attn 5 / ffn 29.

현재 표는 `softmax → stream collapse`를 보여 준다. 실제 `comb`은 softmax 결과에 epsilon을 더하고 열 정규화를 한 뒤, 나머지 반복에서 행·열 정규화를 번갈아 수행한다. `hc_sinkhorn_iters=20`이면 sum/div 정규화가 39회다. softmax 하나는 이 계산과 같지 않다.

교정: Sinkhorn projection을 복합 op로 넣거나, softmax 이후의 reduction/div 반복을 표현해야 mHC의 manifold 제약을 표로 재현할 수 있다.

근거: `modeling_deepseek_v4.py:940-947`; 원시 prefill layer 0 attn_hc op 157–159에서 첫 열 sum/epsilon/div가 확인된다. 로컬 및 공개 checkpoint config의 `hc_sinkhorn_iters=20`과도 일치한다.

## 4. V4-Pro — 압축기의 가중합과 Indexer의 head 가중합이 표에서 빠짐

압축 softmax와 norm 사이에는 softmax 확률과 KV를 곱한 뒤 window 축을 sum하는 계산이 있다.

| 대상 | 현재 요약 위치 | 빠진 계산의 shape |
|---|---|---|
| HCA compressor | prefill op 17 → 18 (layer 0 대표) | `[B,T/m_hca,m_hca,d_head]`에서 가중 곱·axis 2 sum → `[B,T/m_hca,d_head]` |
| CSA compressor | prefill op 67 → 68 (layer 2 대표) | `[B,T/m_csa,2*m_csa,d_head]`에서 가중 곱·axis 2 sum → `[B,T/m_csa,d_head]` |
| Indexer compressor | prefill op 71 → 72 (layer 2 대표) | `[B,T/m_csa,2*m_csa,c_I]`에서 가중 곱·axis 2 sum → `[B,T/m_csa,c_I]` |
| Indexer scorer | layer 2 prefill op 74–76 / decode 68–70 | scores와 head weights를 곱하고 `n_h_I`축 sum → `[B,Q,T/m_csa]` |

교정: weighted reduction을 명시해야 한다. 특히 compressor norm은 rank 4 입력을 rank 3으로 바꾸는 연산이 아니므로, 현재 두 행 사이의 rank 변화는 생략된 reduction에 의해 일어난다.

근거: `modeling_deepseek_v4.py:414-418,548-550,673-675,455-459`. Indexer scorer의 원시 prefill op 1963–1964 / decode op 1571–1572는 head weight 곱·sum이다.

**decode에서 compressor softmax/norm이 없는 것은 이 발행 조건에서는 정상이다.** Prefill 2176 이후 한 토큰만 추가하면 새 m=4/128 window가 아직 완성되지 않는다. projection은 하지만 buffer에 남기고 기존 압축 KV를 재사용한다. 근거: 같은 파일 `237-239,412-427,644-685`.

## 5. V4-Pro — hash routing과 동적 routing의 의미가 실제 표에서 사라짐

대상: 양 phase `model.layers.*.mlp.gate`. 대표 요약 gate op는 prefill 35,93,143,201 / decode 33,87,135,189.

현재 표는 이 모듈을 `matmul → experts`로만 보여 준다. 실제로 첫 3층은 frozen `tid2eid` 테이블에서 token ID별 expert를 선택하는 **hash routing**이다. 이후 층은 score correction bias를 더한 점수로 top-k를 선택한다. 양쪽 모두 sqrtsoftplus scoring, 선택 score gather, top-k weight 정규화, 2.5 scaling이 있다.

교정: 최소한 `hash lookup`과 `dynamic top-k`를 구분하고 routing weight를 만드는 계산을 나타내야 한다. 표의 층별 그룹 자체는 올바르게 나뉘어 있지만 `MLA+MoE` 문자열과 gate matmul만으로는 이 차이를 알 수 없다.

근거: `modeling_deepseek_v4.py:1044-1051,1054-1082,1088-1102`; `configuration_deepseek_v4.py:278-282`. 원시 layer 0 lookup은 prefill op 717 / decode 580이다. `scoring_func`, scaling 값은 로컬 및 [공개 checkpoint config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/config.json)에 명시되어 있다.

## 6. 소스로 바로 갈리는 축 이름 — V4 Indexer의 NoPE 부분

다음은 **요약 CSV·JSONL에 없는 slice/cat의 원시 축**이다. 요약의 GEMM 축이 틀렸다는 지적과 구분해야 한다. UNKNOWNS의 `scope_inferred`에 포함된 질문을 소스로 해결하는 항목이다.

Indexer head는 폭 `c_I=128`이고 마지막 `d_rope=64`만 회전한다. leading 부분은 **`c_I-d_rope`**이며 `d_rope`가 아니다. 값이 둘 다 64라 현재 수치 대조로는 드러나지 않는다.

원장 field는 실제로 `input_shape/output_shape` 대신 `i/o`를 사용한다.

| 모델/phase | 원시 op_id, field, shape_index, axis | 현재 → 교정 |
|---|---|---|
| V4 prefill | `(1927,o,0,3)` | `d_rope → c_I-d_rope` |
| V4 prefill | `(1940,i,0,3)` | `d_rope → c_I-d_rope` |
| V4 decode | `(1537,o,0,3)` | `d_rope → c_I-d_rope` |
| V4 decode | `(1550,i,0,3)` | `d_rope → c_I-d_rope` |

module_path는 모두 `model.layers.2.self_attn.compressor.indexer`다. trailing slice와 cat의 두 번째 입력은 `d_rope`가 맞다. 다른 CSA층에도 같은 역할 구분을 적용할 수 있으나, 이 보고서의 위치 앵커는 위 네 자리를 직접 확인한 것이다.

근거: `modeling_deepseek_v4.py:354-359,497,554-565` 및 `DeepseekV4RotaryEmbedding`의 부분 회전 폭 계산 `134-148`. 회전한 64폭의 interleaved pair 절반은 `d_rope/2=32`다. 현재 원시 trace의 해당 32폭 라벨은 이미 `d_rope/2`로 맞게 되어 있다. 이전 의뢰서의 `c_I/2` 주장을 현재 결과물의 오류로 다시 지적하지 않는다.

## 7. 질문에 든 주요 축·고유 기제 판정

### gpt-oss-20b/120b

- Prefill self_attn bmm의 선두는 **`B*n_h=192`**. query/key 마지막 feature 축은 `d_head`; softmax axis 1은 `n_h`다. 크기가 64로 같아도 역할은 소스의 view/transpose와 matmul 수축축으로 결정된다. 요약 op 5–7 및 27–29의 라벨에 동의한다. 근거: `modeling_gpt_oss.py:261-278,319-327`.
- Prefill softmax의 **`T+1`은 KV 264칸 + sink 1칸**이다. Sink 확률은 PV GEMM 전에 버려지므로 PV의 수축축은 여전히 `T`다. 근거: 같은 파일 `267-277`.
- Sliding window는 prefill의 dense eager GEMM 폭을 128로 줄이지 않는다. mask가 범위를 제한한다. 따라서 짝수층 prefill `T×T`는 정상이며, 별도 “sliding” op가 반드시 필요한 것은 아니다. **마스크 종류의 메타데이터가 없으면 shape만으로는 full/local을 구분할 수 없다.** 근거: 같은 파일 `264-265,308,485-496`.
- Decode 짝수층 요약 op 5–7은 KV `w_local=128`, softmax `w_local+n_sink=129`; 홀수층 op 27–29는 KV `T+1=265`, softmax `(T+1)+n_sink=266`이다. 현재 표는 이 차이를 실제로 표현한다. 쿼리 축은 1이며 B가 아니다.
- Prefill expert 선두 **`B*k*T=3168`**, decode **`B*k=12`**는 unique token 수가 아니라 token–expert 선택 쌍 수다. `E`축은 모든 expert weight의 저장 축이며 모든 expert에 GEMM을 수행한다는 뜻이 아니다. offsets가 expert별 처리 구간을 정한다. 근거: `integrations/moe.py:390-409,429-448,478-484`.
- 소스로 `d_model`/`d_moe`를 확정할 수 있다. gate_up weight `[E,d_model,2*d_moe]`, down weight `[E,d_moe,d_model]`; GLU 중간은 `d_moe`, down 이후 및 down bias는 `d_model`이다. **현재 요약 op 13/19/35/41의 weight 라벨은 맞다.** 근거: `modeling_gpt_oss.py:77-92,113-116`. 이름이 같은 값이라는 이유로 이 자리를 미정으로 유지할 필요는 없다.

### Llama-4-Maverick

- Dense 층은 24개 짝수층, MoE는 24개 홀수층; 그중 layer `3 mod 4`인 12개가 full/NoPE다. 실제 표의 두 MoE 집합에 동의한다. 근거: `configuration_llama4.py:179-200`, `modeling_llama4.py:418-422` 및 로컬 checkpoint의 interleave step=2.
- T=17과 decode의 전체 길이 18은 chunk_size=8192 안에 있다. 따라서 chunked/full 양쪽의 eager attention shape가 같은 것은 정상이다. chunk 차이는 mask에 있다. 근거: `modeling_llama4.py:560-573`.
- Shared expert가 있다. 이 구현은 routed 입력을 `[E,B*T,d_model]`로 확장하고 top-1 이외 expert 입력을 0으로 만들어 전체 expert에 bmm한다. **표의 `B*E*T`, `[E,B*T,...]`는 설치된 구현에 맞다. 이를 `B*k*T`로 바꾸면 현재 trace를 잘못 표현한다.** 이는 sparse 최적화 구현의 실제 연산량과 구분해야 한다. 근거: 같은 파일 `79-83,147-173`.
- RoPE가 major-op 표에서 빠지는 것은 structure.yaml이 이미 명시한 선택 범위다. NoPE층의 temperature scaling도 표의 attention shape만으로 알 수 없다. 근거: 같은 파일 `368-385`.

### V4-Pro

- Prefill의 usable source 길이는 **2176 전체**가 맞다. 압축 출력 수는 HCA **17**, CSA/Indexer **544**다. source 길이가 T라는 말과 압축 출력 길이가 T라는 말은 다르다. `T/m`은 **이 발행점**에서는 정확하지만 일반 길이에서는 floor와 buffer state를 반영해야 한다. 근거: `modeling_deepseek_v4.py:237-239,407-418,639-675`.
- Prefill core attention KV 폭은 HCA `T+T/m_hca=2193`, CSA `T+T/m_csa=2720`; sink 포함 softmax는 각각 2194/2721이다. Decode KV 폭은 145/672, sink 포함 146/673이다. 현재 요약 라벨에 동의한다. 근거: 같은 파일 `204-216,733-743,824-844`.
- Core attention bmm 선두는 **`B*n_h=384`**. Indexer scorer bmm 선두는 **`B*T=6528`**이고, head 축은 `n_h_I=64`, feature 축은 `c_I=128`이다. 둘은 다른 layout이므로 self_attn 하위 전체에 하나의 4D head 위치 규칙을 적용하면 안 된다. 근거: 같은 파일 `455-459,564-565,811-821`.
- Routed expert 선두는 prefill **`B*k*T=39168`**, decode **`B*k=18`**이다. weight 저장 layout은 gate_up `[E,2*d_moe,d_model]`, down `[E,d_model,d_moe]`; grouped GEMM의 effective 입력 weight는 transpose되어 각각 `[E,d_model,2*d_moe]`, `[E,d_moe,d_model]`다. **현재 weight_shape와 GEMM input_shape의 순서 차이는 정상이다.** 근거: 같은 파일 `1001-1002`; `integrations/moe.py:369-374`.
- Grouped o_a의 effective weight `[g_o,n_h*d_head/g_o,d_g]`, 저장 weight `[g_o*d_g,n_h*d_head/g_o]`와 o_b의 `[d_model,g_o*d_g]`에 동의한다. B/T는 weight 축이 아니다. 근거: `modeling_deepseek_v4.py:322-332,792-795,870-872`.
- Indexer `k_I=1024`보다 이 발행점의 압축 블록 수 544가 작아 선택 폭은 544다. 모든 query가 544개를 유효하게 볼 수 있는 것은 아니며 causality mask가 적용된다. 근거: 같은 파일 `568-584,693-702`.
- K=V이므로 attention 출력에 inverse RoPE를 적용하는 단계도 있다. major-op 표의 PV→o_a 사이에는 보이지 않는다. 근거: 같은 파일 `862-868`.

## 8. 검증 범위와 최종 답

네 모델의 실제 요약 파일은 prefill 385행, decode 373행이다. 원시 trace의 `input_shape`, `output_shape`, `weight_shape` 축 라벨을 structure.yaml의 발행 심볼로 수치 평가하고 concrete shape와 비교했으며, **수치 불일치는 0건**이었다. 위 주요 연산의 축 역할은 소스와 별도로 대조했다. 숫자가 같다는 검사만으로 모든 원시 축 이름을 소스 검증했다고 주장하지 않는다.

검토 대상 Llama 디렉터리와 structure.yaml의 실제 model_id는 `meta-llama/Llama-4-Maverick-17B-128E`이고, 의뢰서는 `...-Instruct`라고 쓴다. 본 소스·config 대조는 실제 capture된 모델 기준이다. 이름 불일치가 두 모델의 구조 차이를 증명하는 것은 아니지만, 결과물의 모델 식별은 일치시켜야 한다.

**층위 1:** 주요 계산의 요약으로는 상당 부분 맞다. 실제 forward의 모든 구조를 표 자체가 표현한다는 강한 주장에는 동의하지 않는다. 특히 부록의 템플릿 손실, V4의 mHC/Sinkhorn 및 routing, gpt-oss expert bias/clamp는 구별해서 반영해야 한다.

**층위 2:** 의뢰서에서 직접 질문한 접힌 배치·헤드·routed 축과 주요 weight 축은 현재 표가 맞다. gpt-oss의 동일값 타이는 source role로 확정 가능하다. V4 Indexer의 leading NoPE 폭은 위 원장 네 자리에서 `c_I-d_rope`로 교정할 수 있다. 요약 표의 주요 축에서 새 B 소실이나 weight의 B/T 오명명은 발견하지 않았다. **모든 축의 의미가 맞다는 일괄 인증은 하지 않는다.**

이 문서는 검토 답변이다. 추출 코드, rules, 기존 모델 결과물은 수정하지 않았다.
