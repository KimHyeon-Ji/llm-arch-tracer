# 네 모델 round 2 검토 답변

검토일: 2026-09-18. 대상은 `codex_ask_four_models_round2.md`, 현재 `codex_ask_four_models_claims.md`, 네 모델의 prefill/decode CSV·JSONL 및 full trace·concrete shape·axis_resolution이다.

**판정: 현재 major-op 표의 레이어 분할, 추가 add_ 행, 주요 투영·attention·expert 축은 설치된 forward와 맞는다. 이번 검토에서 최종 표의 새 op/축 결함은 확인하지 못했다. 그러나 최종 라벨과 축 판정 원장이 5,658자리에서 불일치하고, 수정한 resolver에도 산술 오류를 만드는 경로가 남아 있다. 따라서 현재 확정률·질문 묶음을 그대로 의미 검증의 근거로 쓰면 안 된다.**

## 검토한 판과 검증 범위

`results` HEAD는 `13ec885c59d7b20b2e62183ad220c6b05cb4b395`. 공개 commit의 네 모델 × 두 phase × CSV/JSONL = 16파일을 다운로드해 로컬 파일과 비교했다. checkout의 CRLF/LF만 정규화하면 모두 일치했다. 요청서의 “지난 검토 이후” 표에 남은 70/52행 대신 실제 현재판 **69/51/51/224행**, decode **69/51/51/212행**을 검토했다.

설치된 `transformers 5.14.1`의 세 modeling 파일과 `integrations/moe.py`를 공개 v5.14.1 소스와 byte 비교했고 모두 일치했다. 이후 파일:줄은 로컬 설치 파일 기준이다.

- [Llama-4 소스](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/llama4/modeling_llama4.py), [공식 checkpoint](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E)
- [gpt-oss 소스](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/gpt_oss/modeling_gpt_oss.py), [20b checkpoint](https://huggingface.co/openai/gpt-oss-20b), [120b checkpoint](https://huggingface.co/openai/gpt-oss-120b)
- [V4 소스](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/deepseek_v4/modeling_deepseek_v4.py), [공식 checkpoint](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
- [MoE integration 소스](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/integrations/moe.py)

**전건 수행한 것은 발행 B=3 산술 검사와 export 일치 검사다.** 전체 모델을 새 B=4에서 재트레이스하지 않았다. 요청서의 독립 배치 PASS를 이번 실행의 결과로 주장하지 않는다. 의미 검토는 모든 요약 템플릿의 연산 역할과 그 근거 소스, 관련 원시 anchors를 대조한 것이다. 모든 raw 축을 각각 소스로 증명한 결과는 아니다.

| 모델 | 발행 B=3에서 평가한 축, 양 phase·weight 포함 | 산술 불일치 | 평가 불가 | weight에 B/T |
|---|---:|---:|---:|---:|
| Llama-4-Maverick | 76,730 | 0 | 0 | 0 |
| gpt-oss-20b | 47,906 | 0 | 0 | 0 |
| gpt-oss-120b | 71,426 | 0 | 0 | 0 |
| DeepSeek-V4-Pro | 669,758 | 0 | 0 | 0 |
| 합계 | **865,820** | **0** | **0** | **0** |

CSV/JSONL 요약 778행의 주요 metadata와 shape·parameter·depends_on export도 모두 일치했다. claims 부록의 778행도 op_type과 input/output/weight shape를 최종 JSONL에 전건 대조해 불일치 0이었다. raw/concrete의 op ID 대응 및 tensor 개수·rank 불일치도 0이었다. 이 수치 일치로 심볼 이름의 의미까지 증명되지는 않는다.

## Q1 — op 구성과 레이어 분할

현재 부록은 이전에 빠뜨린 layer 집합을 포함한다. 반복 수와 레이어 집합도 config와 맞는다.

| 모델 | 실제 반복 블록 |
|---|---|
| Llama-4 | 짝수 dense FFN 24층, `1 mod 4` RoPE/chunked MoE 12층, `3 mod 4` NoPE/full MoE 12층; 합계 48층 |
| gpt-oss-20b | 짝수 sliding 12층 + 홀수 full 12층; 합계 24층 |
| gpt-oss-120b | 짝수 sliding 18층 + 홀수 full 18층; 합계 36층 |
| V4-Pro | HCA/hash layers 0–1, CSA/hash layer 2, HCA/dynamic 홀수 3–59, CSA/dynamic 짝수 4–60; 합계 61층 |

Llama-4의 use_rope와 dense/MoE 선택은 `modeling_llama4.py:337,367–375,418–422`; gpt-oss의 layer_type과 mask 선택은 `modeling_gpt_oss.py:288,308,485–496`; V4의 attention/routing 선택은 `modeling_deepseek_v4.py:797–798,1088–1089`와 각 provenance config의 layer schedule에 따른다.

추가된 add_는 제자리에 있다. 아래 첫 번호는 **요약 JSONL ID**다.

| 대상, 양 phase | 요약 op | 의미와 소스 |
|---|---|---|
| Llama-4 MoE 두 템플릿 | 40,65 | shared expert 출력에 routed expert의 expert축 sum을 더한다. `modeling_llama4.py:168–173`. expert bias가 아니다. |
| gpt-oss 두 모델, 두 템플릿 | 14,38 | gate/up GEMM 뒤 expert별 bias, 폭 `2*d_moe`. `integrations/moe.py:369–378,435–444`; parameter 정의 `modeling_gpt_oss.py:80–81`. |
| 같은 대상 | 21,45 | down GEMM 뒤 expert별 bias, 폭 `d_model`. `integrations/moe.py:455–468`; parameter 정의 `modeling_gpt_oss.py:82–83`. |

gpt-oss layer 0의 raw add_는 20b prefill 195/207, 120b prefill 201/213, 양 모델 decode 181/193이다. Llama-4 layer 1은 raw prefill 330 / decode 288이다. gpt-oss의 `up+1`(요약 18/42)은 이 두 bias add와 별개의 계산이다.

이미 공개한 clamp·Sinkhorn·weighted reduction·routing·inverse RoPE 등의 생략을 새 결함으로 다시 집계하지 않았다. 그 범위의 major-op 표로는 현재 구성을 수용할 수 있다. V4의 마지막 실제 JSONL 순서도 `hc_head.input_norm → hc_head → model.norm → lm_head`로 맞는다. 부록의 그룹별 출력은 전역 실행 순서를 나타내는 형식은 아니다.

## Q2 — 주요 축 라벨

### Llama-4

양 phase 요약 self_attn의 `B*n_h`, projection의 `B*T`/`B`는 맞다. MoE의 `B*E*T`는 이 구현이 **각 expert에 모든 token을 복제**하므로 맞다. `experts` bmm 입력은 `[E,B*T,d_model]`, 중간은 `d_moe`, down 출력은 `d_model`이다. shared expert도 d_moe=8192를 쓰고 dense FFN만 d_ff=16384를 쓴다. 근거: `modeling_llama4.py:58–83,89–103,168–173,422`.

### gpt-oss-20b/120b

양 phase 요약 13/37 및 20/44의 expert 선두 `B*k*T`/`B*k`는 token–expert 선택 쌍 수다. 저장 weight의 E는 expert bank 개수이며, grouped_mm의 offsets가 expert별 구간을 정한다. 근거: `integrations/moe.py:390–409,429–468`.

`d_model=d_moe=2880`이어도 의미는 소스로 갈린다. expert 입력과 down 출력/bias는 d_model, gate/up의 절반과 GLU 중간은 d_moe다. 현재 최종 요약 라벨은 그 구분에 동의한다. weight는 gate/up `[E,d_model,2*d_moe]`, down `[E,d_moe,d_model]`; source parameter가 transposed storage를 쓰므로 일반 nn.Linear와 같은 out/in 순서를 강제하면 안 된다. `modeling_gpt_oss.py:77–92,113–116`, `integrations/moe.py:365–374`.

Prefill sink softmax의 T+1은 token T + sink 1이고, PV 수축축은 sink를 버려 T다. decode 짝수층은 KV w_local=128 및 softmax w_local+n_sink, 홀수층은 KV T+1 및 softmax (T+1)+n_sink다. 쿼리 축은 1이다. source `modeling_gpt_oss.py:261–277,319–327`.

### V4-Pro

q의 head 폭 d_head=512, Indexer 폭 c_I=128, Indexer head 개수 n_h_I=64는 각 클래스가 읽는 config 필드로 갈린다. output group 축은 g_o=16, 그룹당 output 폭은 **d_g=1024**다. d_g와 k_I가 값이 같아도 o_a_proj는 Indexer top-k 개수가 아니다. group weight 저장 `[g_o*d_g,n_h*d_head/g_o]`와 bmm operand `[g_o,n_h*d_head/g_o,d_g]`는 모두 맞다. `modeling_deepseek_v4.py:324–332,496–503,564`.

mHC의 n_hc=4는 stream 개수이며, `[n_hc,n_hc]`의 두 축은 stream 결합 행렬의 입력/출력 축이다. CSA의 m_csa=4는 token window 크기다. HCA의 m_hca=128도 window 길이이며 n_h나 c_I가 아니다. `modeling_deepseek_v4.py:387–418,534–550,934–950`.

Indexer NoPE 교정은 최종 raw에 반영됐다: prefill `(1927,o,0,3)`, `(1940,i,0,3)`와 decode `(1537,o,0,3)`, `(1550,i,0,3)` 모두 c_I-d_rope다. 근거: `modeling_deepseek_v4.py:354–359,497,564–565`. **아래 Q4처럼 원장에는 이전 이름이 남아 있다. 최종 라벨이 다시 틀렸다는 뜻은 아니다.**

## Q3 — 네 수정의 방향과 남은 결함

네 방향은 해당 발견 사례에 적절하다. 접힌 배치 우선순위 조정은 KDA의 fused-QKV 오명을 막고, 이미 batch가 있는 텐서의 추가 B*X 추측 차단은 loop slice를 막는다. 크기 3을 singleton 1로 바꾸는 것도 금지해야 한다. n_*/E/k의 자동 반 나누기 제외도 안전한 축소다. 실제 head partition에서 n_h/2가 필요한 모델은 소스에 근거한 명시적 규칙으로 표현하면 된다.

하지만 **현재 resolver에 남은 반례 세 개**를 재현했다. 네 발행 표에 이 오류가 있다는 주장이 아니라, “규칙 자체가 완전히 고쳐졌다”는 주장을 제한하는 사례다. 실제 V4-Pro config와 B=3, T=2176을 사용했다.

| resolver 입력 | 현재 결과 | 맞는 것 | 코드 근거 |
|---|---|---|---|
| static weight `[3]`, is_weight=True | `[1]` | `[3]` | `symbolic_shape.py:407–408`에서 B로 읽은 뒤 **845–850의 최종 pass가 무조건 1로 변경**한다. mHC의 실제 `.scale` parameter도 길이 3이지만 발행 raw는 별도 경로에서 3으로 보존되어 있다. parameter 정의 `modeling_deepseek_v4.py:919–922`. |
| folded NoPE `[1344]` | `B*d_head-d_rope` → **1472** | `B*(d_head-d_rope)` → **1344** | `symbolic_shape.py:627–628`에서 합/차 식에 B*를 문자열로 붙이며 괄호를 보존하지 않는다. 의미 판별과 별개로 수치 자체가 틀린다. |
| transpose된 activation `[T,B,d_head]=[2176,3,512]` | `[T,3,d_head]` | `[T,B,d_head]` | `symbolic_shape.py:802`의 no_batch는 seen_batch뿐 아니라 seen_seq도 사용한다. `[B,T,d].transpose(0,1)`은 유효한 내부 layout이다. sequence보다 뒤라는 이유만으로 batch를 차단하면 안 된다. |

권장 수정은 weight에서 B를 만들기 전에 차단하고, 마지막 정리도 실제 크기를 확인하며, 합/차 식의 곱셈에는 AST 또는 괄호를 사용하는 것이다. batch 차수 제약은 유지하되 **축 순서를 batch의 증거로 사용하지 않는 것**이 필요하다. 이와 별도로 B로 나누어떨어진다는 사실만으로 folded batch가 증명되지는 않는다. 독립 배치와 dataflow 근거가 뒷받침해야 한다.

## Q4 — 새로 확인한 발행 메타데이터 결함

### 최종 라벨과 axis_resolution 원장의 불일치: 5,658자리

원장 site.label을 같은 `(op_id,field,shape_index,axis)`의 최종 `full/<phase>.trace.raw.jsonl`과 전건 비교했다.

| 모델 | prefill | decode | 합계 |
|---|---:|---:|---:|
| Llama-4 | 0 | 0 | 0 |
| gpt-oss-20b | 720 | 720 | 1,440 |
| gpt-oss-120b | 1,080 | 1,080 | 2,160 |
| V4-Pro | 1,327 | 731 | 2,058 |
| 합계 | **3,127** | **2,531** | **5,658** |

구체적인 **현재 원장 → 최종 발행 라벨과 소스상 맞는 것**:

| 모델 / phase / raw anchor | 원장 → 맞는 것 | 파일:줄 |
|---|---|---|
| 20b prefill `(139,o,0,2)` | n_h → d_head | `modeling_gpt_oss.py:319–327,263`의 query/key feature 축. 최종 raw는 d_head로 이미 맞다. |
| 20b prefill `(184,i,0,1)`, `(184,o,0,1)` | d_moe → d_model | `integrations/moe.py:400`의 selected_hidden_states는 residual 입력 width. |
| 20b prefill `(207,i,0,1)`, `(207,i,1,1)`, `(207,o,0,1)` | d_moe → d_model | down bias add. `modeling_gpt_oss.py:82–83`, `integrations/moe.py:455–468`. decode 동일 역할 raw 193. |
| V4 prefill `(533,i,0,2)`, `(533,o,0,2)` | 16 → g_o | `modeling_deepseek_v4.py:329–332`의 group 축. |
| V4 prefill `(533,i,0,3)`, `(533,o,0,3)`, `(534,i,0,3)` | 2*d_head → d_g | 같은 source의 per-group output 폭. 현재 raw는 d_g로 이미 맞다. |
| V4 prefill `(1871,o,0,2)` | n_h_I → d_rope | cos pair를 full RoPE 폭으로 확장. `modeling_deepseek_v4.py:352–357`. head 개수가 아니다. |
| V4 prefill `(1927,o,0,3)`와 decode `(1537,o,0,3)` | d_rope → c_I-d_rope | 이번에 고친 leading NoPE slice. `modeling_deepseek_v4.py:357,497,564–565`. |

이는 UNKNOWNS의 기존 미확정 목록을 다시 나열하는 지적이 아니다. **그 목록이 최종 발행 label을 기준으로 묶인 질문이라는 전제가 틀린 것**이다. 예컨대 20b 원장의 n_h 묶음에는 phase당 192개의 실제 d_head가 섞이고, d_moe 묶음에는 528개의 실제 d_model이 섞인다. “d_model|d_moe → d_moe가 맞다”를 그 묶음 전체에 등록하면 반대로 잘못 확정한다.

원인은 기록 시점과 갱신 누락으로 설명된다. `build_table.py:1898–1903`에서 최초 resolver 판정을 기록하고, 뒤의 propagation/unification과 교정이 최종 이름을 바꾼다. 원장은 `2346`에서 쓰는데 `label_overrides.apply`는 **2359에서 그 뒤에 실행**된다. `Ledger.overwrite`는 있지만 이러한 최종 변경 전체를 원장에 반영하지 않는다. `axis_ledger.py`의 coverage_ok는 등급별 count 합계이지, 발행 label과의 대조가 아니다.

권장 교정:

1. 원장은 모든 교정·parameter resync·label 전파 이후 최종 라벨과 동기화한다.
2. 최초 판정과 최종 판정을 분리하고, 각 변경의 이유/근거를 chain으로 보존한다. 최종 문자열만 바꾸어 confirmed로 승격하지 않는다.
3. gate에 site anchor 유효성, `site.label == published label`, 최종 질문 count 재계산을 넣는다.
4. 그 결과로 UNKNOWNS와 확정률을 다시 생성한 뒤, 의미 근거를 묶음 단위로 적용한다.

소스로 바로 갈리는 d_model/d_moe, n_h/d_head, n_hc/m_csa, d_g/k_I 등의 역할은 Q2에 적었다. **현재 원장의 질문 키를 그대로 사용해 blanket correct 판정을 추가하지 않는 것**이 우선이다.

## 재현 파일

```powershell
.venv\Scripts\python.exe develop/review_four_models_round2.py
```

`develop/four_models_round2_evidence.json`에 phase별 산술/export 결과, 템플릿, add_ 행, 원장 불일치 count·대표 anchors, resolver 반례를 기록한다. 모델 다운로드·재트레이스·후보/규칙 변경은 하지 않는다. 공식 소스와 results commit의 네트워크 비교는 이번 검토에서 별도로 수행했으며 이 재현 스크립트의 기능에는 포함하지 않았다.
