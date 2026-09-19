# Kimi-K3 게이트 정리 — round 4 답변

검토일: 2026-09-19. 대상: `develop/codex_ask_kimi_k3_gate_cleanup.md`, 현재 워크트리의 검사 코드,
직전 승격본과 재트레이스 중인 후보. 기존 규칙·후보·승격본은 수정하지 않았다. 재트레이스가
진행 중이므로 아래 산출물 수치는 검토 시점의 스냅샷이다.

## 판정

**지금은 K3를 출고하지 않는다.** 함대 전체 `verify_all`을 0으로 만들 필요는 없지만, 새
K3 판의 하드 회귀를 닫고 게이트 자체의 두 오류를 고쳐야 한다. 첫째,
`lowering_proof.py`의 전량 재실행은 유용한 경험적 증거이지만 모든 배치의 국소 동치
**증명**은 아니다. 둘째, `membership` 62건은 K3에서 실행한 remote code가 아닌
Transformers 계열의 캐시된 동명 소스 파일을 읽어 발생한 거짓 양성이다. 현재의
`approved: true`는 이
두 사실을 반영하지 않는다.

## Q0 — `lowering_proof.py`의 주장 범위

새 검증기는 이전의 shape 다중집합 검사보다 확실히 진전했다. ports로 경계를 만들고, 5개
합성 시험에서 진짜 전치는 통과시키며 네 가지 위조를 거절한다. 저장된 proof artifact도
`semantic_topology_change` 372,948건과 `layout_lowering_verified` 207건, 합계
373,155건을 모두 coverage로 세고 있다. `--whole-module --per-template 100000` 실행에서는
각 template의 instance를 전부 한 번씩 재실행했다.
여기서 `--whole-module`은 같은 `module_path`에 기록된 op를 한 단위로 묶는 옵션이다.
하위 projection 모듈까지 재귀적으로 합쳐 전체 attention 함수를 검증한다는 뜻은 아니다.

그러나 **coverage 100%와 함수 동치 100%는 다르다.** `compare_components`는 새 trace의
무작위 float64 경계 입력을 한 번 만들고, 옛 trace와 새 trace에서 오직 **`b=0` 출력**을
비교한다. 다음 반례를 실제 합성 ports/raw trace로 넣어 확인했다.

```text
old: y = x
new: y[0] = x[0]; y[1:] = 0
batch = 3, 입력 shape [B, 2], 출력 shape [B, 2]
현재 compare_components 판정: True (통과)
```

즉 현재 다섯 시험 중 `batch_mix`는 *첫 배치가 다른 배치에 의존하는* 오류를 잡지만,
첫 배치만 옳고 나머지가 틀린 오류는 잡지 못한다. 우선 모든 배치 조각 `b=0,1,2`를 같은
경계 입력의 해당 old 조각과 비교하고, 위 반례를 회귀 시험에 추가해야 한다. folded batch
축의 `B*X` 해제도 각 `b`에 대해 확인해야 한다.

그 뒤에도 한두 개의 난수 대조는 수치 **시험**이지 모든 입력에 대한 대수적 증명이 아니다.
특히 `template_of`는 op 이름의 다중집합만 지문으로 쓰므로 DAG 간선·`scalar_args`가 같은
template인지 말하지 않는다. 이번 실행은 모든 instance를 재실행했기 때문에 template
표본 누락 문제는 없지만, 통과 이름은 우선 `lowering_replay_consistent`처럼 두는 편이
정확하다. `contraction_lowering_verified`로 승격하려면 최소한 각 연결 성분의 경계 tensor
identity, layout 원소 인덱스 대응, bmm 수축 축과 출력 복원을 정규화해 비교해야 한다.

질문한 두 한계도 실제로 중요하다.

- **소비자 없는 출력:** 실제로 버려지는 임시값은 결과 동치에서 빼도 된다. 하지만
  `boundary()`는 소비자 목록이 비어 있으면 무조건 버린다. 모델 반환값, 외부에 노출된 cache,
  in-place 변경을 이 규칙만으로 “관측 불가능”하다고 단정할 수 없다. 모듈 반환·상태·alias
  경계를 명시적으로 포함하거나, 해당 범위에 그런 출력이 없음을 확인해야 한다.
- **dtype 복원:** 소비 op에서 bool/long으로 맞추고 팩토리 부동소수점을 float64로 통일하면
  양쪽을 *같은 재해석 아래* 비교하는 시험은 된다. 원 trace의 bf16/float32, 정수 index의
  실제 범위, 반올림·overflow·mask 동작의 동치는 보장하지 않는다. `seed_dtype`는 long index를
  모두 0으로 채워 일부 경로를 약하게만 검사한다. 적어도 trace에 dtype을 기록하고
  원 dtype으로 재실행하거나, dtype과 무관한 layout/contraction 인덱스 증명으로 좁혀야 한다.

`UNKNOWNS.md`에 이 한계를 적은 것은 좋지만, 현재의 “미증명 0”과
`audit_manifest.approved: true`가 위 한계를 무력화한 것처럼 읽히면 안 된다. 정확한 문구는
“373,155개 미매칭 레코드가 255개 whole-module 실행에 포함됐고, 생성한 float64 입력의
첫 배치 조각에서 불일치가 없었다. 전 배치·전 입력 동치는 미검증”이다. 수치적 coverage와
증명된 coverage를 별도 필드로 기록해야 한다.

## Q1 — MLA `g_proj`

**진단이 맞다.** 고정 K3 소스의 `KimiMLAAttention`은
`mla_use_output_gate=True`이면 `g_proj = Linear(hidden_size, num_heads * v_head_dim)`을
만든다. forward에서는 `g_proj(hidden_states).sigmoid()`를 attention 출력에 곱한 후
`o_proj`에 넣는다. 따라서 MLA `g_proj`의 출력 폭은 `n_h*d_v = 96*128 = 12,288`이다.
KDA `g_proj`도 같은 수치 12,288을 갖지만 소스상 별개의 head 개수/폭이다.
`g_proj$`의 KDA 이름 교정에 `layer_types: [linear_attention]`을 붙이는 것이 옳다.

`_schedule`이 `text_config`를 보고, `_spread`가 등가류의 각 자리에 층 종류를 다시 검사하도록
한 수정도 필요한 방향이다. 이 경로가 실제 K3 93층에 대해 69개 KDA/24개 MLA를 반환하는지,
그리고 class spread footprint가 반대 층의 op_id를 포함하지 않는지 재트레이스 후 확인해야
한다. [고정 K3 모델 소스](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/modeling_kimi_linear.py)

## Q2 — `d_nope -> d_v` 앵커

**현재 capture에서는 `layer_types + input_shape + op_type:view`가 `nth`보다 낫다.**
승격본의 layer 3을 직접 보면 prefill의 문제 view는 op 55256 하나다. query의 최초 view
입력은 `[B,T,n_h*(d_nope+d_rope)]`이고 그 출력 폭은 `d_nope+d_rope=192`이다.
`q_pass` 자체는 transpose 이후 `[B,n_h,T,d_nope]` 경로에 있다. 따라서
`[B,T,n_h,d_nope]` 입력 view를 q 경로와 혼동할 근거는 현재 trace에 없다. 고정 소스에서
attention의 value 입력 `value_states`는 `v_head_dim`이고, attention 출력이
`o_proj` 직전에 reshape되는 경로도 `v_head_dim`이다.

다만 **shape selector만으로 의미가 영구히 보장되지는 않는다.** 재트레이스 후 prefill과
decode의 24개 MLA 층에서 앵커가 각각 정확히 하나씩 발화하는지 보고,
`verdict_footprint.json`의 class spread 대상이 attention 출력에서 `o_proj`로 이어지는
동일 텐서의 축인지 ports의 producer/consumer로 확인해야 한다. 실패하거나 둘 이상
맞으면 fail-closed로 두고 input producer/consumer anchor를 추가한다. 서수를 11 또는 10으로
고정하는 방식은 trace 변화에 더 취약하다.

## Q3 — `2*E_shared*d_moe`

**제시한 12,288 자리의 이름은 맞다.** 고정 소스에서
`KimiSparseMoeBlock`은 공유 MLP의 intermediate size를
`num_shared_experts * moe_intermediate_size`로 만든다. `KimiMLP`는 그 폭의 gate와 up
출력을 `cat`으로 붙인다. 값은 `2 * 2 * 3,072 = 12,288`이다. `2*d_shared`라는 새
심볼은 `d_shared`를 따로 정의하지 않는 한 오히려 근거를 흐린다.

그러나 이 규칙은 **C17의 새 미해결 상수 3,840을 해결하지 않는다.** 검토 시점의 후보와
승격본 `full/report.md` 모두 C17에서 `[3840]`을 보고한다. 이것은 별도의 MoE even-split
shim 토큰 수 `B*k*T/4`이지 아키텍처 폭 12,288이 아니다. `4*d_moe` 72축 교정과 C17
3,840 한 건을 별도 작업으로 추적해야 한다. 후자에 `d_moe`식 이름을 붙여 숨기면 안 된다.

## Q4 — `label_no_name.expect`

**지금은 구체값 3,840을 유지한다.** 현재 verdict는 `expect`뿐 아니라 `shape`와
`expected_classes`, adaptation marker를 함께 고정한 snapshot-specific 판정이다. B가
바뀌면 죽어서 사람이 새 trace와 shim 산술을 확인하게 하는 것이 본래의 fail-closed 성격과
맞는다. B=3, T=320, `k=16`, expert cap=4인 이번 판에서는 3,840이 정확하다.

향후 여러 B를 한 verdict로 지원하려면 `expect_expr`만 도입하지 말고 `shape` selector도
같은 평가 좌표계에서 유도하고, cap은 실제 provenance의 `expert_cap`에서 읽어야 한다.
그때도 계산된 정수와 실제 trace 값, adaptation marker, 등가류 범위를 대조해야 하며
`B*k*T/cap`을 아키텍처 이름으로 발행해서는 안 된다.

## Q5 — 낡은 `label_confirmed` 186건

**(a)의 무검증 일괄 교체와 (c)의 영구 방치는 모두 피한다.** 매치 0인 항목이 현재 축을
검토 목록에서 빼주지 않는다는 진단은 맞다. 그러나 활성 `label_confirmed.yaml`에
“확인했다”는 규칙 186개를 죽은 상태로 두고 `verify_all`의 FAIL만 공개하는 것은 기록과
게이트의 의미를 약하게 만든다.

권장 처리는 **(b)를 기본으로 한 선별 재앵커링**이다. 옛 source 인용과 selector는 이력으로
보존하고, 현재 trace에 대응을 확인할 수 있는 항목만 새 B=3 anchor로 재등록한다. 대응을
못 찾거나 옛 shape에 이미 거둬들인 지어낸 이름(`2*d_conv` 등)이 포함된 항목은 활성
규칙에서 제거해 그 축을 다시 검토 대상으로 돌린다. 186개를 손으로 전부 다시 판단할
필요는 없고, 동일 source site·동일 ports 경계별로 묶어 후보를 생성한 뒤 각 그룹의
대표와 적용 건수를 검증하면 된다. 그때까지는 “현재 186건의 확인 기록이 무효”라고
공개하되, 출고 게이트의 FAIL을 예외 승인으로 바꾸지는 않는다.

## Q6 — `membership` 62건

**이 62건은 현재 검사기의 소스 선택 오류다.** `source_check.run`은
`model_type=kimi_linear`로 `fetch()`를 호출한다. 이 환경에 해당 설치 파일이 없어
`develop/sources/modeling_kimi_linear.py`에 캐시된 Transformers 소스를 읽는다. 그 파일은
현재 K3 trace가 실행한 고정 revision의
remote `modeling_kimi_linear.py`와 다르며, 실제 K3의 `KimiMLAAttention` 및
`KimiSparseMoeBlock` 클래스를 포함하지 않는다. 그래서 `module_classes.json`에는 그
클래스가 있는데 정적 read map에는 없어, 모든 62건의 owner가 `model / (root)`로 떨어졌다.
현재 `review_request.md`도 검사 근거로 잘못 선택한 `develop/sources/modeling_kimi_linear.py`를
안내하고 있다.

고정 revision의 실제 remote source와 현재 `module_classes.json`, 현재 가중치 라벨에
대해 `membership_gaps`를 **읽기 전용으로 다시 계산한 결과 0건**이었다. 대표적으로
MLA의 `c_q`/`c_kv`는 각각 실제 source가 읽는 `q_lora_rank`/`kv_lora_rank`이고,
MoE `w1`·`w3`의 `[d_moe,d_moe_lat]`, `w2`의 `[d_moe_lat,d_moe]`는 각각
3,072의 expert FFN 폭과 3,584의 routed latent 폭이다. 같은 가중치에 두 이름이 있는 것은
서로 다른 두 축이므로 오류가 아니다.

수정할 것은 라벨이 아니라 **소스 선택**이다. 모델이 실제 import한 클래스의 module path
또는 고정 revision의 `auto_map`에서 `modeling_kimi_linear.py`와 대응 config를 선택하고,
그 파일 해시를 provenance/검사 결과에 기록해야 한다. repository에서 “가장 짧은 modeling
파일”을 고르는 fallback도 K3 wrapper를 골라 text tower를 놓칠 수 있으므로 충분하지 않다.
이 소스 선택으로 membership뿐 아니라 `review_request`와 alias/source 확인 결과도 다시
생성해야 한다. [고정 K3 모델 소스](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/modeling_kimi_linear.py)

## Q7 — 출고 시점

선택지로는 **(b)에 가깝다.** 단, §6은 라벨 62개를 고치는 작업이 아니라 검사기가 올바른
고정 소스를 읽게 하는 작업이다. 함대 46개 모델의 과거 FAIL을 모두 0으로 만들 필요는 없다.
K3 출고에는 다음을 요구한다.

1. 재트레이스가 끝난 한 snapshot에서 prefill/decode reshape 불일치 0, 새 `4*d_moe`
   오라벨 0, no-name verdict 2건 살아 있음, C17의 3,840 shim 상수 처리 상태를 확인한다.
2. `lowering_proof`가 모든 배치 조각을 검사하도록 고치고 위 반례를 거절하게 한다. 수치
   시험을 계산 동치 “증명”으로 승격하려면 인덱스/수축 축 검증도 추가한다. 검사 범위와
   결과를 manifest 및 `UNKNOWNS.md`에 같은 용어로 쓴다.
3. `source_check`가 고정 K3 remote source를 사용하도록 고친 뒤 membership과 의뢰서를
   다시 생성한다. 낡은 확인 기록은 재앵커링 또는 활성 규칙에서 퇴역시킨다.
4. 독립 B=4 라벨 검사와 K3별 하드 불변식·기준선 대비 신규 FAIL이 없고, 기존의 미확정
   사항은 실제 수치로 `UNKNOWNS.md`에 공개한다. 함대 전체 FAIL 수는 별도 관리한다.
5. 그 동일한 snapshot으로 audit를 재실행한다. 현재 승격본 manifest의
   `candidate_hash=d646ecb23029c6a7`와 검토 시점 승격본 public hash
   `cee6ca73095b95d2`가 다르다. `release_blockers('moonshotai__Kimi-K3')`도 현재는
   `UNKNOWNS.md`의 열린 지적 7건 수가 맞지 않아 출고를 막는다. 이 상태의
   `approved: true`를 최종 승인으로 사용하지 않는다.

audit 구현에도 간단한 coverage 안전장치가 더 필요하다. 현재 proof JSON은
`records_total`과 `records_paired`를 따로 내지만, audit의 `unproven` 계산은
`records_total - records_paired`를 실패로 세지 않는다. 이번 저장된 결과에서는 둘이
일치했으나, gate는 항상 그 동등성 및 proof/diff 프로세스의 exit code를 검사해야 한다.

게이트의 계약도 일치시켜야 한다. `transition_release.audit`는 `full/report.md`의 C 검사
FAIL, 독립 배치 검사, diff, lowering 재실행을 보고 승인하지만, `verify_all`의 membership,
낡은 확인 기록, reshape 기준선 회귀까지 승인에 반영하지 않는다. 반면
`sync_results_branch.py`의 문서는 “최근 `verify_all` FAIL 0”을 전제로 쓰고 실제
`release_blockers`는 공개 문구 일부만 검사한다. 출고 전 K3별 하드 검사 결과와
`release_blockers` 결과를 manifest에 묶거나 같은 snapshot에서 모두 실행해야 한다.
proof artifact는 현재 `full/` 안에만 있고 results 스냅샷의 `CARRY_FROM_FULL` 목록에는
포함되지 않는다. 공개판에서 verified 주장을 할 계획이면 작은 proof summary와 digest도
함께 실어야 독자가 근거를 추적할 수 있다.
