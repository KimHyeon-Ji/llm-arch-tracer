# Kimi-K3 게이트 정리 — round 4

요청일: 2026-09-19. 선행: `codex_answer_kimi_k3_ship_decision.md` (round 3).
고정 revision `f831ab66814297da540d832a5235f8e904f29d06`.

## 0. round 3 결과 보고 (먼저 이게 맞는지 봐 달라)

round 3 이 요구한 `contraction_lowering_verified` 를 구현했다. `develop/lowering_proof.py`.

- ports 의 producer/consumer 로 DAG 를 세우고, 모듈의 **실제 바깥 경계**를 찾는다.
- 기록된 `scalar_args` 로 **그 구간을 다시 실행**해 float64 로 대조한다.
  배치 축은 발행 라벨을 가설로 써서 `b=0` 조각을 뽑는다 -- 라벨이 틀리면 대조가 깨진다.
- 결과: prefill 371,751 + decode 1,404 레코드, **미증명 0**. instance 전수(69/69, 24/24 …).
- 감사가 이 증명으로 `semantic_topology_change` 372,948 + `layout_lowering_verified` 207 을
  해소했고, `semantic_change` 936 은 `develop/verify/references.yaml` 의
  `transition_reviewed` 에 소스를 인용해 등재했다. `audit_manifest.json` 은 `approved: true`.

**검증기가 거절할 수 있는지**는 `develop/test_lowering_proof.py` 가 증명한다(5/5):
진짜 전치 bmm 통과 / `cat([a,a])` · shape 같은 딴 텐서 · 배치 섞기 · 전치 복원 누락은 거절.

만드는 동안 내 검증기의 결함 6 개를 잡았다. 전부 실제 데이터가 잡아준 것이다:
경계 입력을 위치로 짝지음, 출력 앵커 충돌, sink 앵커 비대칭, 제자리 연산의 생산자 시점,
`c > oid` 필터가 만든 가짜 경계, 새/옛 앵커 규칙 비대칭.

**Q0. 이 증명의 주장 범위가 과한가?** 특히 (a) 아무 op 도 소비하지 않는 중간 값을 경계에서
뺀 것, (b) 트레이스에 dtype 이 없어(`torch.bool` 이 직렬화되지 않는다) 마스크·색인을 소비
지점에서 맞추고 팩토리 op 의 부동소수점 결과를 float64 로 통일한 것 -- 이 둘이 증명을
약화시키는가. `UNKNOWNS.md` 에 둘 다 적어 뒀다.

---

## 1. 배경: 출고 직전에 게이트가 잡은 것

승격 후 `develop/verify_all.py` 를 돌려 **승격 전 기준선과 대조**했다(워크트리로 HEAD 를
떠서 같은 검사를 돌렸다). 함대 전체 FAIL 178 -> 185.

내가 **새로 만든** 회귀:

```
reshape_incons  0 -> 24      (하드 불변식. 기준선에서 0 이었다)
미해결 유도 상수 0 -> 1
4*d_moe  72축                (지어낸 이름)
label_no_name 죽은 판정 2건
확인 기록 낡음 5 -> 186
```

원래 있던 것(기준선에도 FAIL): membership 62축, 근거 없는 확인 기록 908, 의뢰서 18건.

**이번 세션에 나는 원인 진단을 두 번 틀렸다.** 그래서 결론이 아니라 **근거를 봐 달라.**

- 1차: "`spread: class` 가 층 경계를 넘는다" -> 방향은 맞았지만 `self_attn$` 7 건에 층
  타입을 달았더니 발화 수가 552/414/138 로 **완전히 동일**했다. 걸릴 자리가 없었다.
- 2차: "`g_proj`/`f_b_proj` 는 KDA 전용 모듈이라 안전하다" -> 세어 보니 **MLA 층에도
  `g_proj` 가 96 행 있다**(`f_b_proj`·`q_conv1d` 는 정말 KDA 전용).

## 2. reshape_incons 24 건 — 고친 내용과 남은 의문

24 건 전부 같은 자리다. MLA 층(0-based 3,7,…,91,92 — 24 개)의 `o_proj` 직전 reshape:

```
op 55256 (layer 3, view)
  라벨 in  = [B, T, n_h, d_nope]          구체 [3, 320, 96, 128]
  라벨 out = [B, T, n_h_kda*d_head_kda]   구체 [3, 320, 12288]
  reshape 자체 유도: out 축 2 는 n_h*d_nope 여야 한다
```

문제 축의 **등가류 구성원을 전부 나열**했더니 전부 MLA 층 안이었다:

```
6  MLA  ...self_attn    (문제의 자리)
5  MLA  ...g_proj       (여기가 앵커)
5  MLA  ...o_proj
```

resolver 에 직접 물으니 두 층 유형 모두 `n_h*d_v` 라고 답한다(resolver 는 층 타입을 모른다).
즉 `n_h_kda*d_head_kda` 는 **층 타입 없는 `g_proj$` 교정**이 MLA `g_proj` 에 앵커를 잡고
`spread: class` 로 끌고 간 결과다.

고친 것:

1. `g_proj$` 교정에 `layer_types: [linear_attention]`.
2. `d_nope -> d_v` 교정의 앵커를 `nth: 11` -> `layer_types: [full_attention]` 로 교체
   (서수가 10 으로 밀려 죽어 있었다). MLA 층 안에서 입력 `[B,T,n_h,d_nope]` 인 view 는
   층당 하나뿐임을 전수로 확인했다(24 층 × 1).
3. 이걸 가능하게 하려고 `src/label_overrides.py` 를 둘 고쳤다:
   - `_schedule` 이 `text_config` 까지 본다. K3 는 `KimiK3ForConditionalGeneration` 래퍼라
     `linear_attn_config.full_attn_layers` 가 `text_config` 안에 있어서 `layer_types`
     선택자가 **조용히 죽어 있었다**.
   - `_spread` 가 퍼뜨리는 자리마다 층 타입을 확인한다(전에는 앵커만 봤다).

결과: prefill 24 건 해소. 그런데 **decode 에서 24 건이 새로 나왔다** -- 내가 decode 쪽
`d_nope -> d_v` 짝을 "resolver 가 이미 d_v 로 낸다" 고 보고 지운 것이 틀렸다. 지우고 나니
decode MLA 48 축이 `d_nope` 로 돌아갔다. 같은 앵커 방식으로 되살렸고 재트레이스 중이다.

**Q1. `g_proj` 는 MLA 층에서 무엇인가.** 나는 출력 게이트이고 폭이 `n_h*d_v`(=12288)라고
보고 KDA 이름을 막았다. 막으면 그 자리에 resolver 의 `n_h*d_v` 가 남는다. 소스로 확인해 달라.

**Q2. `d_nope -> d_v` 선택자가 안전한가.** round 3 에서 `value_states` 가 `d_v` 이고
`o_proj` 의 `num_heads*v_head_dim` 으로 이어진다고 확인해 줬다. 그런데 q 경로의 `q_pass` 도
`[B,T,n_h,d_nope]` 형태일 수 있다. `layer_types` + 입력 shape + `op_type: view` 만으로 그
둘을 가를 수 있는가, 아니면 서수를 유지해야 하는가?

## 3. `4*d_moe` 72축 — 고친 내용

`heur_multiple` 이 12288 을 `4*d_moe`(4×3072)로 지어냈다. 트레이스가 출처를 직접 보여준다:

```
op 36620 _unsafe_view  ...shared_experts.gate_proj  out=[B, T, E_shared*d_moe]  (6144)
op 36626 _unsafe_view  ...shared_experts.up_proj    out=[B, T, E_shared*d_moe]  (6144)
op 36627 concat        ...shared_experts            out=[B, T, 12288]
op 36630 slice         ...shared_experts.act_fn     pos=[2, 0, 6144]
op 36635 slice         ...shared_experts.act_fn     pos=[2, 6144, ...]
```

gate+up 융합이다. `rules/derived_dims.yaml` 에 이미 같은 관례가 둘 있다
(`2 * d_moe` = 라우팅 전문가 gate+up, `2 * d_ff` = dense FFN gate+up). 그 옆에
`2 * E_shared * d_moe`(scope `shared_expert`, `unless_equals: [d_model]`)를 등록했다.
`num_shared_experts = 2`, `moe_intermediate_size = 3072`, `hidden_size = 7168`.

**Q3. 이 이름이 맞는가.** `rules/symbols.yaml:124-127` 에 **2026-08-26 당신의 검토로
K3 의 MoE 캡 셔플 부산물(값 4)이 `2*E_shared` 로 오라벨됐던 전례**가 적혀 있다. 이번 것은
값(12288)도 자리(shared_experts gate/up concat)도 다르지만, 같은 함정인지 확인해 달라.
`2*d_shared` 같은 다른 이름이 더 맞는가?

## 4. `label_no_name` 죽은 판정 2 건 — 설계 질문

MoE even-split shim 의 부산물 축을 "이름 없음" 으로 판정해 둔 항목이다. 근거는
`1280 = k(16)*T(320)/4`. B=3 발행에서 `3840 = B*k*T/4` 가 되어 `expect: 1280` 이 죽었다.
`expect: 3840` 으로 갱신했다.

**Q4. `expect` 가 구체값이라 배치를 탄다.** 배치를 바꿀 때마다 이런 판정이 죽는다.
`expect` 를 식(`B*k*T/traced`)으로 받게 바꾸는 편이 나은가, 아니면 구체값을 유지하고
게이트가 잡아 주는 지금 방식이 나은가? 이 판정의 목적은 "shim 산술 부산물에 이름을 붙이지
않는다" 이므로, 값이 배치를 타는 것 자체가 그 성질의 일부이기도 하다.

## 5. 낡은 확인 기록 186 건 — 판단을 물음

`rules/label_confirmed.yaml` 은 "소스를 봤고 이 이름이 맞다" 는 기록이고, 그 축을 검토
목록에서 영구히 뺀다. 186 건이 **아무것도 매치하지 않게** 됐다. 앵커가 전부 B=1 shape 다:

```
99  d_head_kda  shape=('n_h_kda', 'B', 'd_head_kda')
64  d_head_kda  shape=('n_h_kda*n_chunk', 'B', 'd_head_kda')
 5  n_h_kda     shape=('n_h_kda', 'd_chunk', 'd_head_kda')
 1  n_h_kda     shape=('B','n_h_kda','5','2*d_conv')      <- 거둬들인 지어낸 이름
```

B=1 einsum lowering 의 형태(batch 축이 가운데 끼는)와, 이번에 정수로 물러난 이름들이다.
사실 자체(KDA head 폭 = `d_head_kda`)는 여전히 참이고 **앵커만 낡았다**.

**Q5. 어떻게 처리해야 하는가.**

  (a) 앵커를 B=3 형태로 일괄 갱신 -- 186 건을 기계적으로 고치는 것이라 검증 없이 근거
      기록을 건드리게 된다.
  (b) 삭제 -- 그 축들이 검토 목록으로 돌아온다(보수적이지만 이미 확인한 일을 다시 시킨다).
  (c) 그대로 두고 공개 -- 낡은 항목은 아무것도 매치하지 않으므로 **"잘못된 확인이 축을
      검토에서 빼 준다" 는 위험은 이미 없다**. 게이트 FAIL 은 위생 신호로 남는다.

나는 (c) 로 기울어 있다. 하지만 (b) 가 더 정직한가? 판단 근거를 달라.

## 6. membership 62축 — 검사기 한계인가 진짜 오류인가

기준선에도 있던 FAIL 이다. 가중치 축이 "그 모듈도 부모도 읽지 않는 config 필드" 의 이름을
달고 있다는 지적:

```
d_moe      in block_sparse_moe.experts.*.w1/w2/w3              각 1472축
d_moe_lat  in block_sparse_moe.experts.*.w1/w2/w3              각 1472축
d_moe_lat  in block_sparse_moe.routed_expert_(up|down)_proj    각 368축
d_moe      in block_sparse_moe.shared_experts.(gate|up|down)_proj  각 368축
c_kv       in self_attn.kv_(a_proj_with_mqa|b_proj)            각 96축
c_q        in self_attn.q_(a|b)_proj                           각 96축
```

`routed_expert_hidden_size = 3584`, `kv_lora_rank = 512`, `q_lora_rank = 1536` 은 config 에
실재한다. 검사기는 "그 모듈 클래스의 `__init__` 이 그 필드를 읽는가" 를 본다.

**Q6. 이게 진짜 라벨 오류인가, 아니면 remote-code 모델에서 검사기가 필드 읽기를 못
따라가는 것인가.** `c_q`/`c_kv` 는 MLA 의 표준 LoRA rank 라 이름 자체는 맞아 보인다.
`d_moe` 와 `d_moe_lat` 이 **같은 가중치**(`experts.*.w1/w2/w3`)에 동시에 지적되는 것도
이상하다 -- 3072 와 3584 는 다른 값인데 같은 파라미터의 서로 다른 축일 것이다.

## 7. 출고 기준

`verify_all` 은 **이번 작업 전에도 함대 전체 178 FAIL** 이었고, 이미 출고한 네 모델도
자기 몫의 FAIL 을 갖고 있다(예: DeepSeek-V4-Pro 는 낡은 확인 기록 27 건, 미발화 교정 3 건).
출고 게이트는 `develop/sync_results_branch.py` 의 `release_blockers` 이고, 그건 "모름이
없는가" 가 아니라 **"모름이 공개됐는가"** 를 본다(round 2 이후 그렇게 바꿨다).

**Q7. K3 를 언제 내야 하는가.**

  (a) 내가 새로 만든 회귀(§2, §3, §4)만 닫고, 기존 FAIL 은 `UNKNOWNS.md` 에 공개하고 출고.
  (b) §5, §6 까지 닫고 출고.
  (c) `verify_all` 전체를 0 으로 만들고 출고 -- 함대 46 개 모델의 낡은 산출물 재생성까지
      포함되므로 이번 출고와 무관한 작업이 대부분이다.

나는 (a) 로 기울어 있다. 네 모델이 같은 조건으로 나갔고 K3 만 다른 기준을 적용할 이유를
못 찾았다. 다만 §5/§6 이 "공개" 만으로 충분한 종류인지 판단이 필요하다.

## 8. 재현

```powershell
.venv\Scripts\python.exe develop/lowering_proof.py moonshotai__Kimi-K3 --whole-module --per-template 100000
.venv\Scripts\python.exe develop/test_lowering_proof.py
.venv\Scripts\python.exe develop/transition_release.py audit moonshotai__Kimi-K3 develop/models/phase25-kimi-k3.yaml
.venv\Scripts\python.exe develop/verify_all.py
```

검토 대상: `develop/out/moonshotai__Kimi-K3/` (재트레이스 중 -- §3/§4 수정과 decode 복원은
아직 반영 전이다), `models/moonshotai__Kimi-K3/` (직전 승격본),
그리고 `rules/derived_dims.yaml`·`rules/label_overrides.yaml`·`rules/label_no_name.yaml`·
`src/label_overrides.py` 의 이번 변경.
기존 규칙·후보 파일은 수정하지 말아 달라.
