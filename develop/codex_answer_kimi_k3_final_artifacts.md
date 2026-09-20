검토일: 2026-09-20. 요청: `develop/codex_ask_kimi_k3_final_artifacts.md`.

**93층 구성과 두 대표 행의 shape는 맞다. 그러나 현재 산출물을 구조·계산·의존관계까지 올바른 최종본이라고 승인하기는 어렵다.** 공개된 미확정 라벨과 별개로, decode KDA forget gate의 수식이 다르고 KDA 결과의 의존 간선이 빠져 있다. `structure.yaml`의 설명에도 확정적으로 틀린 항목이 있다.

산출물과 구현은 수정하지 않았다. 이 답변과 `develop/kimi_k3_final_artifacts_evidence.json`만 추가했다. 이미 공개한 23개 질문, 미확정 비율, MoE expert cap, 인용 부채, 배치 전환 수치 시험의 한계는 새로운 결함으로 세지 않았다. 기존 게이트와 전체 retrace도 다시 실행하지 않았다.

검토 대상은 요청한 **results 작업 디렉터리의 파일**이다. 코드 HEAD는 `67ea697a05696c32f70f3b8e64601de585583f70`, results HEAD는 `69c9219f327110fd07fec5f13c71c8382de2414c`, 모델 revision은 `f831ab66814297da540d832a5235f8e904f29d06`이었다. 파일별 SHA-256과 전수 집계는 증거 JSON에 남겼다. 네 CSV/JSONL과 `structure.yaml`은 현재 main 작업 디렉터리의 해당 파일과 바이트 단위로 같았다. `UNKNOWNS.md`는 다르므로 아래 공개 문구 판단은 **results에 실린 파일**을 기준으로 했다.

아래 경로와 줄 번호는 다음 기준이다. JSONL은 `op_id + 1`번째 줄, CSV는 헤더 때문에 `op_id + 2`번째 줄이다.

| 약칭 | 소스 / 산출물 |
|---|---|
| R | 저장소 루트 기준 `../llm-arch-tracer-results/models/moonshotai__Kimi-K3/` |
| M | `C:/Users/99ktx/.cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/f831ab66814297da540d832a5235f8e904f29d06/modeling_kimi_linear.py` |
| C | 같은 snapshot의 `config.json` |
| CC | 같은 snapshot의 `configuration_kimi_k3.py` |
| N | `.venv/Lib/site-packages/fla/ops/kda/naive.py` |
| F | `.venv/Lib/site-packages/fla/ops/kda/fused_recurrent.py` |
| G | `.venv/Lib/site-packages/fla/ops/kda/gate.py` |

**1. [P1] decode의 KDA forget gate는 요청된 lower-bound 수식을 표현하지 않는다.**

대상: `R/decode.jsonl`, `R/decode.csv`, layer 0의 `op_id=12..18`; 특히 `op_id=17`의 decay `exp`로 들어가는 값. 다른 68개 KDA층도 같은 경로다. 이 행들은 MoE 대체 경고와 무관하고 `caveat`도 비어 있다.

prefill에는 `op_id=16`에 `[B,T,n_h_kda,d_head_kda]`의 forget-gate sigmoid가 있다. decode의 `op_id=16`은 `[B,1,n_h_kda]`의 **beta sigmoid**이며, 그 forget-gate sigmoid에 대응하는 것이 아니다. 단순히 T를 줄인 차이가 아니다.

| 구분 | forget gate의 log-decay |
|---|---|
| 모델이 요청한 것 | `-5 * sigmoid(exp(A_log) * (g + dt_bias))` |
| 현재 decode 추적 경로 | `-exp(A_log) * softplus(g + dt_bias)` |

근거:

- C:93에 `gate_lower_bound=-5.0`이 있고, M:629–644의 decode `fused_recurrent_kda` 호출은 `use_gate_in_kernel=True`, `lower_bound=self.gate_lower_bound`를 넘긴다.
- F:29는 `lower_bound is not None`만으로 `USE_LOWER_BOUND`를 켠다. F:160–170의 실제 분기도 위 sigmoid 수식이다. **이 recurrent API는 `safe_gate=True`를 요구하지 않는다.**
- 반면 `src/kda_shim.py:265–283`은 `safe_gate=False`를 기본값으로 삼고 두 조건을 함께 요구한다. M의 recurrent 호출에는 `safe_gate`가 없으므로 plain gate를 고른다. 같은 wrapper가 recurrent에 연결된 근거는 `src/kda_shim.py:481–485`다.
- 현재 원시 trace에서도 확인했다. `models/moonshotai__Kimi-K3/full/decode.trace.raw.jsonl`의 layer 0 `op_id=151`은 `aten.softplus.default`, `op_id=154`는 beta sigmoid다. prefill 원시 `op_id=155`에는 forget-gate sigmoid가 있다. 즉 요약 표에서 sigmoid 한 행만 선별 누락한 경우가 아니다.

설치된 G의 두 torch reference 함수를 불러 실제 wrapper에 모델과 같은 kwargs를 넣어 확인했다. `g=A_log=dt_bias=0`, `lower_bound=-5`일 때 현재 decode가 다음 recurrence에 전달하는 값은 **-0.6931471824645996**, 기준은 **-2.5**였다. `safe_gate=True`를 주는 prefill 경로는 -2.5였다.

재현은 GPU나 모델 전체 실행이 필요 없다:

```python
import os
import torch
from src.kda_shim import _extract_func, _wrap_kda

path = os.path.join("ops", "kda", "gate.py")
plain = _extract_func(path, "naive_kda_gate")
bounded = _extract_func(path, "naive_kda_lowerbound_gate")
call = _wrap_kda(lambda **kw: kw["g"], plain, bounded)
z = torch.zeros(1, 1, 1, 1)
a = torch.zeros(1)
actual = call(q=z, k=z, v=z, g=z, beta=torch.zeros(1, 1, 1),
              A_log=a, dt_bias=a, use_gate_in_kernel=True, lower_bound=-5.)
expected = bounded(z, a, dt_bias=a, lower_bound=-5.)
print(actual.item(), expected.item())  # -0.6931471824645996, -2.5
```

이는 GPU 커널의 op 구성을 표에 싣지 않았다는 한계와 다르다. CPU reference로 대체하더라도 호출자가 요청한 gate 수식은 보존해야 한다. 기존 provenance의 “decode가 safe_gate를 안 넘기므로 plain gate가 모델 자체의 동작”이라는 설명도 F의 API와 맞지 않는다. 행 이름만 고쳐 닫을 문제는 아니다.

**2. [P1] KDA 출력으로 가는 `depends_on`이 끊겨 있다.**

첫 층 안에서 확인한 것이므로 반복 층 접기를 어떻게 해석하느냐와 관계없다.

| 파일 | 행 | 현재 의존관계 | 빠진 의미 |
|---|---:|---|---|
| `R/prefill.jsonl`, `.csv` | 825, `self_attn.o_norm` | `[11,824]` | KDA가 계산한 출력 `o`로부터의 의존 경로 |
| `R/decode.jsonl`, `.csv` | 20, `self_attn.o_norm` | `[11,19]` | 같은 층 `op_id=18`의 attention 결과로부터의 의존 경로 |

prefill 825의 전체 선행 노드를 따라가면 `[0,2,5,10,11,824,825]`뿐이다. decode 20은 `[0,2,5,10,11,19,20]`뿐이다. Q/K projection, forget gate, beta, attention contraction이 모두 이 출력의 선행 노드에서 빠진다. 현재 표만 그래프로 읽으면 V projection과 output gate만으로 attention 출력이 만들어지는 셈이다.

실제 소스는 M:651–659에서 `o = self.o_norm(o, g)`를 실행한다. 그 `o`는 N:158–166의 chunk attention 결과 또는 N:59–66의 recurrent 결과다. 특히 N:63의 `o[:, i] = einsum(...)` 결과가 decode 정규화에 도달해야 한다. **shape가 같은 `zeros_like(v)`의 초기 생성 경로만으로는 결과의 생산자를 설명할 수 없다.**

두 phase의 모든 KDA `o_norm` 대표 행을 검사했다. phase당 **16개 대표 행**, `repeat`를 적용하면 **69개 층** 모두 같은 층의 attention `batched_matmul`이 선행 노드에 하나도 없었다. 필요한 수정은 결과 텐서의 slice/in-place 쓰기를 거친 의존 경로를 보존하는 것이다. 단순히 특정 op_id를 수작업으로 붙이라는 뜻은 아니다.

**3. [P2] `structure.yaml`의 숫자 해설이 표의 올바른 라벨을 다시 잘못 설명한다.**

이것은 공개된 bare/heuristic 축의 개수를 다시 지적하는 것이 아니다. 별도 산출물인 `literal_dims.expr`가 확정형 설명으로 틀린 의미를 붙이는 문제다.

| 위치 | 현재 설명 → 올바른 설명 | 구체 근거 |
|---|---|---|
| `R/structure.yaml:27` | `d_head=74` → 활성 attention head 폭으로는 부적합. null/비적용 또는 사용하지 않는 config 기본값임을 명시 | CC:63–65의 `7168//96` 기본값이다. MLA Q/K 폭은 M:357의 192, V 폭은 128, KDA 폭은 M:485의 128이다. 두 최종 표에서 독립 토큰 `d_head`를 사용하는 shape 행은 0개다. 단일 전역 `d_head`를 192로 바꾸는 것도 V/KDA 폭을 설명하지 못한다. |
| `:510–538`, 값 5 | 모두 `d_conv+1` → prefill 청크 수와 decode conv-cache 길이를 구분 | prefill `op_id=18`의 `[B,n_h_kda,5,d_chunk,d_head_kda]`에서 5는 N:108–116의 `NT=T/64=5`. conv 캐시 축이 아니다. |
| `:539–550`, 값 10/32/37 | RoPE 통과분 / rotate_half 폭 → KDA 청크 내부 루프의 길이는 정수로 유지 | N:134–135의 prefix slice. 이 모델 경로에는 실제 RoPE 회전도 없다. bare로 물러난 행에 옛 이름을 범례에서 재부여하면 안 된다. |
| `:559–562`, 값 288 | `n_h+2*n_kv`, fused QKV head 수 → KDA bmm에서는 `B*n_h_kda` | prefill `op_id=819` 또는 decode `op_id=18`의 첫 축은 `3*96=288`. N:61–63, :160–163의 head별 contraction이다. 이 모듈은 Q/K/V도 M:498–502에서 별도 projection으로 만든다. |
| `:589–600`, 값 6144 | `n_h*d_rope` → shared 폭은 `E_shared*d_moe`, routed gate/up concat은 `2*d_moe` | prefill `op_id=1740`과 `1693`, decode `130`과 `83`. M:798–801 및 :263–265. 두 용도가 같은 숫자라는 이유로 attention 폭이 되지 않는다. |
| `:601–616`, 값 12288 | 모든 scope에 `n_h*d_v` → MLA는 그 이름이 맞지만 KDA는 `n_h_kda*d_head_kda`, shared concat은 `2*E_shared*d_moe` | prefill `op_id=3`, `1742`, MLA `1770`; decode `3`, `132`, `160`. M:495–502, :400–401, :798–801 및 :294–298. |

`192=d_nope+d_rope`, `576=c_kv+d_rope`, `18432=n_h*(d_nope+d_rope)`, `24576=n_h*(d_nope+d_v)`, dense gate/up의 `67584=2*d_ff` 설명은 각 해당 모듈에서 맞다. 3,840을 shim 토큰 수로 설명한 것도 맞다.

**4. [P2] `structure.yaml`의 세 `known_limits` 문구는 이 산출물과 맞지 않는다.**

- **:14–15, 공유 전문가:** `num_shared_experts` 필드가 없다는 문장은 틀렸다. C:191에 2가 명시돼 있다. M:798–801은 두 개의 별도 shared 모듈을 생성하는 대신, `intermediate_size=3072*2=6144`인 하나의 `shared_experts` MLP를 만든다. 독립 축 `E_shared`가 없는 것은 맞지만 `E_shared*d_moe`라는 실제 축 식에는 사용된다. 각 phase에서 그 심볼을 포함하는 shape 행은 253개다.
- **:16, RoPE:** “실행됐지만 major-op 선별에서 빠짐”이 아니다. C:173은 `mla_use_nope=true`, M:396은 이를 assert하고 :403은 `rotary_emb=None`으로 둔다. M:423–440은 64차원 조각을 분리·확장·재결합할 뿐 회전을 적용하지 않는다. `d_rope=64`는 이 경로에서 config 필드 이름을 따른 폭이며, 실제 회전이 실행됐다는 뜻이 아니다.
- **:12–13, 두 add가 한 행이라는 예:** 이 판에서는 반례가 표에 있다. prefill `1845`는 `block_sparse_moe`의 shared+routed add, `1846`은 층 잔차 add다. decode도 `235`, `236`으로 분리돼 있다. elementwise를 요약할 수 있다는 일반론과 이 모델의 실제 두 행을 구분해야 한다. 소스는 M:836–837, :1044다.

**5. 공개 문구도 일부 정정이 필요하다.**

결함 개수나 기존 한계를 다시 문제 삼는 것이 아니라, 이번에 전달된 공개 파일의 설명을 확인한 결과다.

- `R/UNKNOWNS.md:63–73`은 7건 전부를 “지적됐고 아직 안 고쳤다”고 설명한다. 함께 실린 `review_findings.json`의 앞 세 건은 `verdict=current_label_correct`, `status=current`다. **현재 라벨이 맞다는 기록을 미수정 결함으로 세면 안 된다.** 과거 open 상태를 최신 수정 여부와 자동으로 동일시해서도 안 된다.
- `:77–81`은 B=1↔B=3의 lowering 서명 불일치를 “major-op 표에 계산이 빠지는 한계”라고 설명한다. 서로 다른 배치의 ATen 분해가 달라진 것과 한 표에서 연산을 생략한 것은 다른 사실이다. 이 설명은 :123 이후의 배치 전환 대조와 구분해야 한다.
- 의뢰서가 `UNKNOWNS.md`에 공개했다고 적은 **736건**과 **446,563→451,876**은 이 results 파일에는 없다. 해당 부채를 새 문제로 세지는 않지만, 이 파일에 이미 공개했다는 설명은 현재 파일과 맞지 않는다.
- `:125`가 가리키는 `full/batch_transition_proof.json`도 results 디렉터리에는 없다. 실제 발행 위치는 같은 디렉터리의 `batch_transition_proof.json`이다. 수치 시험의 기존 한계와 별도로 근거 경로는 바로잡아야 한다.

**Q1에 대한 답: 층 접기 자체는 맞다. 실행 순서는 `layers`로 복원해야 한다.**

두 phase 모두 그룹별 `layers`를 펼치면 0–92가 누락·중복 없이 정확히 한 번씩 나온다. 모든 그룹에서 `repeat == len(expanded_layers)`다. KDA 69층, MLA 24층이 고정 config와 정확히 맞고 첫 층만 dense FFN이다.

마지막 그룹 `layers="87,91-92"`, `repeat=3`도 맞다. C:67–92의 full-attention 목록에 92와 93이 명시돼 있고, KDA 목록과 상호 배타적으로 전체 93층을 덮는다. CC:152–156 및 M:883–890의 **실제 층 선택 기준은 `(layer_idx+1) in kda_layers`**이며 이것으로도 같은 결과다. 따라서 마지막 두 MLA층은 스케줄을 잘못 펼친 결과가 아니다. 저자가 마지막 층을 그렇게 선택한 동기까지 config만으로 추정할 필요는 없다.

`attn+MoE`와 `MLA+MoE`의 차이는 attention 종류이지 MoE 종류가 아니다. 전자를 `KDA+MoE`, 첫 층을 `KDA+FFN`으로 쓰거나 명시적인 범례를 두면 해석이 분명해진다. 현재 `layers` 정보로는 구분 가능하다.

예를 들어 `1-2,4-6,8-10` 다음에 `3,7,11` 그룹이 나온다고 해서 KDA 8층 다음 MLA 3층을 연속 실행하는 스택이 아니다. 각 대표 블록을 `layers` 위치에 배치해야 한다. `module_path`, 파라미터 이름과 `layer_idx`는 대표 층의 값이다.

의뢰서의 행 수 요약에는 마지막 attention-residual 집계 **7행**이 빠져 있었다. 실제로 prefill `15189–15195`, decode `2309–2315`에 `block_type="-"`로 존재한다. 따라서 13,590+846+752+3에 7을 더해야 15,198이다. 이 계산은 표에서 누락된 것이 아니다. M:987–997, :1028–1033의 층별 attention residual도 표에 보인다. 다만 앞의 KDA 의존 간선 문제 때문에 전체 계산 그래프가 온전하다고까지 승인할 수는 없다.

**Q2에 대한 답: KDA↔MLA 스코프 누출은 최종 표에서 추가로 발견하지 못했다. 범례는 별도 문제다.**

17,516개 JSONL 행의 input/output/weight shape를 전부 검사했다. KDA의 `self_attn`에 독립 심볼 `n_h`, `n_kv`, `d_rope`, `d_nope`, `d_v`가 나온 경우는 0개다. MLA에 `n_h_kda`, `d_head_kda`, `d_chunk`가 나온 경우도 0개다. `d_rope`는 MLA projection 폭 식에만 있고, `d_chunk`는 prefill KDA에만 있다.

`n_kv`는 두 최종 표의 shape에 **0회** 나온다. `structure.yaml`의 96은 C:188의 값과 맞지만, 이 표에 `n_kv`가 잘못 붙은 축은 없다. M:384–388의 KV 복원 폭은 실제로 `num_heads*(d_nope+d_v)`이므로 현행 `n_h` 식이 맞다. 이 구현은 M:442–444에서 **복원된** key/value를 cache에 넘긴다. MLA라는 이름만 보고 이 CPU 경로가 512차원 latent만 cache한다고 읽으면 안 된다.

범위를 지키기 위해 다음은 신규 결함으로 세지 않았다. prefill `1769`, decode `159`의 A×V bmm에는 여전히 마지막 폭 `d_nope`가 남아 있다. 소스 M:432–458상 정답은 `d_v`다. 그러나 발행 sidecar를 원시 op `55245`/`1676`과 대조하니 해당 input/output 축이 모두 **공개된 `open_tie`**였다. 따라서 “공개 밖의 새 오라벨”이라고 보고하지 않는다. 이는 전체 라벨이 모두 맞다고 승인했다는 뜻도 아니다.

값 충돌이 64/96/128 세 묶음에서만 생긴다는 의뢰서의 전제는 넓혀야 한다. `literal_dims`의 5/288/6144/12288처럼 **런타임 곱과 합성 차원**도 같은 값을 가질 수 있다.

**Q3에 대한 답: 두 대표 행은 맞다.**

| 대표 행 | 확인 결과 |
|---|---|
| prefill `1761`, decode `151`, MLA `q_a_proj` | `c_q=1536`은 C:199 및 M:364–368의 Q LoRA rank다. 실제 matmul 피연산자는 `[d_model,c_q]`, 저장 weight는 `[c_q,d_model]`, `weight_pos=1`이다. 올바른 전치 표기다. 같은 weight를 두 번 집계하지 않으면 된다. |
| prefill `19`, KDA bmm | `[B*n_h_kda*n_chunk,d_chunk,d_head_kda] @ [B*n_h_kda*n_chunk,d_head_kda,1]`이 맞다. N:108–130의 head별·chunk별 contraction으로 `3*96*(320/64)=1440`이다. |

다만 `n_chunk`는 현재 `structure.yaml.symbols`에 정의돼 있지 않다. 소비자가 이 표만 평가할 수 있도록 `n_chunk=T/d_chunk`, 이 prefill에서는 5라는 런타임 정의를 제공하는 편이 맞다. 아키텍처 상수로 5를 고정하라는 뜻은 아니다.

또한 `weight_pos` 규약에는 예외가 있다. phase당 104개 RMSNorm 행은 `weight_pos=-1`이며 weight 피연산자가 요약된 `input_shape`에 없다. 예: 양쪽 `op_id=2`. -1을 Python의 마지막 입력 인덱스로 해석하면 activation을 weight로 잘못 뺀다. 대표 MLA 행은 맞지만 “항상 input_shape에 weight가 들어 있다”는 설명은 너무 넓다.

**Q4에 대한 답: 주요 projection 경로는 존재한다. decode의 계산과 의존관계는 위 두 문제가 남는다.**

| 확인 대상 | 최종 표에서 확인한 것 |
|---|---|
| MLA KV 압축·복원 | prefill `1764→1765→1766`, decode `154→155→156`: `kv_a_proj_with_mqa`, `kv_a_layernorm`, `kv_b_proj`가 모두 있다. 폭 576→512→24,576도 소스와 맞다. split/view 같은 개별 ATen 행까지 모두 싣는 표는 아니다. |
| KDA conv | 첫 층 두 phase의 `6,8,10`: Q/K/V 세 depthwise conv가 있다. prefill 출력 길이 `T+d_conv-1=323`, decode 창 `d_conv=4`에서 출력 1도 해당 reference 경로와 맞다. |
| KDA forget gate·scan | prefill `12–17`의 projections/gate 및 이후 chunk contractions가 있다. decode는 recurrent 경로가 보여야 하며 chunk scan 행이 없는 것은 정상이다. 다만 lower-bound gate가 다른 수식으로 대체된 점은 1번 문제다. |
| MoE router·shared | prefill `1688/1689`의 `gate`와 `1740/1741/1750`의 `shared_experts`가 별도 경로로 나타난다. routed latent down/up/norm도 있다. router `topk`, scatter/argsort 등의 모든 세부 op가 최종 major 표에 있는 것은 아니다. |
| MLA decode 길이 | prefill score `1767`: `[B*n_h,T,T]`; decode `157`: `[B*n_h,1,T+1]`. cache 320 + 현재 query 1이 맞다. decode에서 `T`를 일괄 1로 치환하면 안 된다. 이 표의 `T+1`에서 T는 기존 cache 길이 320이다. |
| 마지막 residual 집계 | 앞서 적은 마지막 7행이 존재한다. 일반 norm/head 외의 계산이 사라진 것은 아니다. |

CSV와 JSONL의 24개 필드도 두 phase 전 행을 파싱해 대조했고, 내용 불일치는 **0건**이었다. `caveat`가 채워진 행은 요청서대로 각 1,472개다. 이는 알려진 MoE 대체 범위를 확인한 것이며, 실제 라우팅 토큰 수까지 검증했다는 뜻이 아니다.

**Q5에 대한 답: 대부분의 config 수치는 맞지만, 74와 일부 null의 의미를 정리해야 한다.**

`L,d_model,n_h,n_kv,d_ff,V,ctx,E,E_shared,k,k_grp,d_moe,d_moe_lat,c_kv,c_q,d_nope,d_v,d_rope,n_h_kda,d_head_kda,n_attn_res_block,d_conv`는 고정 config와 일치한다. `d_chunk=64`는 raw config의 필드가 아니라 N:78의 torch reference 기본 chunk 길이다.

| 항목 | 판단 |
|---|---|
| `d_head=74` | config 클래스의 fallback 값을 옮긴 것이지만 이 모델의 유효 head 폭은 아니다. 3번의 설명대로 비적용 처리해야 한다. |
| `n_grp=null` | 구조 값은 **1**로 알 수 있다. C:184의 `num_expert_group`, CC:101, M:683. M:724의 grouped-routing 분기는 이 값에서는 실행되지 않으므로 실제 그룹 축이 생긴다는 뜻은 아니다. |
| `d_shared=null` | 독립 config 필드가 없다는 것은 맞다. 그러나 M:798–801로 shared MLP 폭 **6144=E_shared*d_moe**를 확정할 수 있다. derived 항목으로 표현 가능하다. 현재 행들의 `E_shared*d_moe` 자체는 맞다. |
| `layer_sched=null` | 직접 별칭이 없을 뿐 스케줄은 안다. C의 두 층 목록과 CC:152–156에서 93개 항목을 명확히 유도할 수 있다. 최종 표의 `layers`는 이를 이미 보존한다. |
| `chunk_size=null` | 이 저장소 심볼은 `attention_chunk_size`, 즉 고정 attention window를 뜻한다(`rules/symbols.yaml:185–198`). KDA scan 길이 `d_chunk`와 다르므로 여기에 64를 넣으면 안 된다. |
| `w_local,n_sink,m_csa,m_hca,g_o,d_g` 등 | 이번 고정 text 경로에서 임의의 숫자로 채울 소스 근거를 발견하지 못했다. null을 유지하는 것이 맞다. |

이번 결과를 이어받을 때 먼저 처리할 대상은 **decode gate 수식과 KDA 출력 의존 경로**다. 그 뒤 심볼 표의 범례·공개 문구를 실제 산출물에 맞춰 정정하면 된다. 알려진 미확정 축을 억지로 확정하거나 전체 label/shape가 틀렸다고 되돌릴 근거는 이번 검토에서 나오지 않았다.
