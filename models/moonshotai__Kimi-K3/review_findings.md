# 라벨 검토 결과 — moonshotai/Kimi-K3

- 검토일: 2026-09-02
- 검토자: llm(claude) + codex(외부, 2026-09-02, 파이프라인 코드 미접근)
- 본 것: review_ledger가 이 모델을 '미수행'으로 보고해서 처음부터 다시 봤다. review_request.md가 '증거로 있다'고 주장한 develop/sources/modeling_kimi_linear.py 가 실제로는 저장소에 없다는 것부터 발견(실제 백본은 HF 모듈 캐시의 modeling_kimi_linear.py, .cache/huggingface/modules/transformers_modules/... 에 있음) -- 그 파일을 직접 열어 7건 전부 재확인했고, 2026-08-25 판정과 독립적으로 같은 결론에 도달했다. 그 과정에서 2026-08-25 판정이 산출물에 실제로 반영된 적이 없다는 것(review_request.md에 그대로 다시 올라옴)과 review_ledger에 한 번도 기록되지 않았다는 것도 함께 발견됨. 2026-09-02: 값충돌 자동 게이트(develop/check_value_collisions.py)가 96/128/64/6144 자리를 새로 잡아냈고, Codex 외부 아키텍처 검토가 model_summary.md 카드 자체의 4개 오류(RoPE→NoPE, SwiGLU→SiTU-GLU, KV cache 93→24 레이어, LAYER MIX가 KDA 누락)를 추가로 찾음 -- 전부 트레이스/config로 직접 재검증 후 반영.
- 요약: 7건 중 4건(square 축, n_h_kda tie, d_head_kda tie, MoE 캡 1280)은 이미 맞게 렌더되고 있음을 원본 소스로 재확인. 나머지 2건(2*d_conv류 3개, n_h_kda/2 1개, 전부 KDA 청크 스캔의 루프 인덱스)은 2026-08-25에 이미 no_name_exists로 판정됐지만 한 번도 산출물에 반영되지 못했다 -- label_no_name.yaml로 닫으려 시도했으나 그 메커니즘이 stub_ambiguous 축만 인식한다는 것을 게이트 FAIL로 확인(8건 dead verdict)하고 되돌렸다. `_unname_loop_indices`(src/build_table.py)를 직접 고치는 것만이 실제 경로인데, 그 함수는 오늘 이미 두 번의 정교화 시도가 전부 함대 회귀로 되돌아간 이력이 있어(git log 참고) 이번에도 손대지 않았다. review/06-open-renames.md A62로 기록. [2026-09-02 추가] 값충돌 4건(96/128/64/6144) 전부 Codex 판정을 독립 검증 후 반영(96/128은 n_h*d_v→n_h_kda*d_head_kda 실제 버그 수정 ~390축, 64/6144는 이미 정답이었음 확인만). Codex 아키텍처 검토로 model_summary.md의 RoPE/활성함수/KV캐시/LAYER MIX 4건 추가 수정(src/summarize.py). d_head=74(값 10=d_head-d_rope, 37=d_head/2로 오표시)는 Codex가 KDA naive_chunk_kda의 청크 내부 루프 인덱스(fla/ops/kda/naive.py:101-125, i in range(1,BT))라고 특정 -- 기존 「2*d_conv류」와 같은 부류(A62)로 합류, 근본 수정은 여전히 _unname_loop_indices(두 번 회귀 이력)뿐이라 이번에도 보류.

> 이 파일은 `review_findings.json` 에서 생성된다 — 고칠 때는 JSON 을 고친다.

## 발견 1 — 맞음 (미반영)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.self_attn` |
| 축 | KDA S state 정사각 [B, n_h_kda, d_head_kda, d_head_kda] |
| 현재 라벨 | `d_head_kda` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 미반영 |

**근거**

A59(review/06-open-renames.md)에서 확정: `KimiDeltaAttention.__init__`(modeling_kimi_linear.py:478-541)은 `config.qk_nope_head_dim`/`v_head_dim` 을 전혀 읽지 않는다 -- 모듈-필드-소속 검사로 `d_nope`/`d_v`는 이 클래스 안에서 원리적으로 불가능한 후보다. 정사각 자체는 `fla/ops/kda/naive.py:141,607`의 in-place S 누적 버퍼(`new_zeros`/`add_`)이고 `rules/label_confirmed.yaml`에 소스 인용과 함께 이미 확인 기록으로 등록돼 있다(69 layer x 2 phase 매치). 의뢰서의 후보 생성 단계가 KDA 스코프를 안 보고 전역 심볼에서만 값을 찾아 `d_nope`/`d_v`를 후보로 잘못 만든 것이 A59의 결론이었고, 이번 재확인에서도 같다.

## 발견 2 — 맞음 (미반영)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.self_attn` |
| 축 | KDA 자신의 head 개수 (96) -- `n_h vs n_kv` 값 충돌 |
| 현재 라벨 | `n_h_kda` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 미반영 |

**근거**

A59와 동일한 근거: `KimiDeltaAttention`은 `config.num_attention_heads`/`num_key_value_heads`를 읽지 않고 `config.linear_attn_config['num_heads']`(=96)를 읽는다(:485-488, `self.num_k_heads = self.num_heads`이므로 애초에 K/V head 수가 갈라지는 구조가 아니라 `n_kv`라는 개념 자체가 없다). 값이 MLA 레이어의 `num_attention_heads`(=96)와 우연히 같을 뿐, 다른 근거로 같아진 것이다. `n_h_kda`는 `rules/symbols.yaml`에 `linear_attn_config.num_heads`로 등록돼 있고 `scope_strict: true`로 MLA 스코프와 완전히 분리돼 있다(A59 RESOLVED, 2026-08-19). `n_h`/`n_kv` 후보는 KDA 클래스 안에서 원리적으로 불가능하다.

## 발견 3 — 맞음 (미반영)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.self_attn` |
| 축 | KDA 자신의 head_dim (128) -- `d_nope vs d_v` 값 충돌 |
| 현재 라벨 | `d_head_kda` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 미반영 |

**근거**

A59와 동일한 근거: `KimiDeltaAttention`은 `config.qk_nope_head_dim`/`v_head_dim`을 읽지 않고 `config.linear_attn_config['head_dim']`(=128)을 읽는다(:485-488). `b_proj`(:529, `Linear(hidden_size, num_heads)`)·`o_norm`(:539, `FusedRMSNormGated(head_dim, ...)`)·`f_a_proj`/`f_b_proj`(:523-524)가 전부 이 KDA 전용 `head_dim`을 직접 선언한다. 값이 MLA의 `qk_nope_head_dim`(=128)과 우연히 같을 뿐이다. `d_head_kda`는 `scope_strict: true`로 등록돼 있다(A59 RESOLVED, 2026-08-19). `d_nope`/`d_v` 후보는 KDA 클래스 안에서 원리적으로 불가능하다.

## 발견 4 — 이름 없음이 정답 (미반영)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.block_sparse_moe.experts.*.act_fn` |
| 축 | MoE 캡 셔플이 접은 토큰 배치 폭 -- prefill 1280 / decode 4 |
| 현재 라벨 | `2*E_shared` |
| 판정 | `no_name_exists` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 미반영 |

**근거**

A60(review/06-open-renames.md)에서 확정: 이 값은 아키텍처 차원이 아니라 `src/kda_shim.py`의 `patch_moe_infer`(MoE 라우팅이 값 의존적이라 트레이스 불가능해서 넣은 shim)가 `KDA_SHIM_EXPERT_CAP=4`(기본값)로 토큰을 4명의 전문가에게 균등 분할한 부산물이다 -- `1280 = k(16)*T(320)/4`, `4 = k(16)*1(decode)/4`, 실측으로 정확히 일치. `E_shared`(공유 전문가 수, =2)와는 아무 관계가 없고 `2*E_shared`라는 이름은 값이 우연히 맞아떨어진 heur_multiple 오라벨이다. 이미 `provenance.adaptation_log`에 `moe_infer_even_split`으로 기록돼 있고 C10 예외로도 처리된 것과 같은 부류라 이름을 붙이지 않는다 -- 이름을 붙이면 shim의 부산물을 아키텍처로 오해하게 만든다. **재확인(2026-08-29)**: 실제 렌더가 지금도 정확히 bare `1280`/`4`임을 review_request.md/트레이스로 재확인. 같은 shim의 세 번째 관측 지점(`model.layers.*.block_sparse_moe`, experts 밖, `sorted_tokens[start:end]` 디스패치 자체, src/kda_shim.py:592-593)도 이미 올바르게 bare로 렌더되고 있음을 추가로 확인 -- 조치 불필요.

## 발견 5 — 이름 없음이 정답 (미반영)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.self_attn` |
| 축 | KDA 인트라-청크 순차 재귀 루프의 슬라이스 크기 -- 값 8/12/16 |
| 현재 라벨 | `2*d_conv / 3*d_conv / 4*d_conv` |
| 판정 | `no_name_exists` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 미반영 |

**근거**

develop/verify/references.yaml의 irreducible_literals(2026-08-25 등록, max_value:63 로 2026-08-27 정정)에서 이미 확정: `fla/ops/kda/naive.py:134` `for i in range(1, BT)`(BT=chunk_size=64)가 청크 내부를 한 칸씩 넓혀가며 순차 처리하는 반복 계단이지 아키텍처 상수가 아니다. `d_conv`(=4)와 값이 우연히 맞아떨어져(2*4=8, 3*4=12, 4*4=16) heur_multiple 이 그럴듯한 합성 이름을 지어냈을 뿐, 이 축들은 conv kernel 폭과 아무 관계가 없다. `2*E_shared`(A60)와 정확히 같은 부류의 오라벨 -- 반복 계단은 이미 문서화된 값 범위(1~63) 안이라 이름을 붙이지 않는 게 정답이다. **재확인(2026-08-29)**: 실측 트레이스로 정확한 op_id까지 추적 -- `naive.py:127-136`의 `select`(행 i를 뽑음) → `slice`(길이 i로 자름) 사슬이 그대로 `model.layers.0.self_attn`(op_id 2984→2993 등)에 남아 있고, 같은 사슬의 다른 i 값(예 i=7)은 이미 정상적으로 bare로 렌더된다 -- 이 4개 값만 어떤 후속 휴리스틱 단계가 골라서 조작형 이름을 다시 붙인 것으로 보인다. **아직 산출물에 반영 안 됨** -- `rules/label_no_name.yaml`로 닫으려 시도했으나(4개 selector 등록) `src/axis_classes.py:bad_stub_count`/`src/label_no_name.py:covered_keys`가 `stub_ambiguous`로 표시된 축만 인식한다는 것을 확인(이 축들은 ambiguous가 아니라서 매치 0건 -- dead verdict FAIL 8건, 즉시 되돌림). 실제로 렌더를 고치려면 `_unname_loop_indices`(src/build_table.py) 자체를 손봐야 하는데, 이 함수는 오늘 이미 두 번(다중 피연산자 predecessor 모호성 / unary-only 보호 예외) 시도가 전부 함대 회귀로 되돌아갔다 -- 세 번째 시도는 하지 않았다. review/06-open-renames.md A62.

## 발견 6 — 이름 없음이 정답 (미반영)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.self_attn` |
| 축 | KDA 인트라-청크 순차 재귀 루프의 슬라이스 크기 -- 값 48 |
| 현재 라벨 | `n_h_kda/2` |
| 판정 | `no_name_exists` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 미반영 |

**근거**

위 항목과 같은 반복 계단(값 48도 1~63 범위 안). `n_h_kda`(=96)의 절반과 우연히 같아 heur_half 가 지어낸 이름이다 -- KDA의 head 개수를 반으로 나눌 아키텍처적 이유가 없다. develop/verify/references.yaml의 irreducible_literals 항목이 이미 이 값 범위 전체를 문서화해 뒀다. **재확인(2026-08-29)**: 위 2*d_conv류 항목과 정확히 같은 상태 -- 아직 미반영, 같은 이유로 label_no_name.yaml 시도 후 되돌림. review/06-open-renames.md A62.

## 발견 7 — should_be_no_name (미반영)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.self_attn` |
| 축 | KDA naive_chunk_kda 청크 내부 루프 prefix 길이 (값 10, 37 -- d_head(74, 미사용 유령값)에서 파생된 d_head-d_rope/d_head-2 로 오표시됨) |
| 현재 라벨 | `d_head-d_rope / d_head/2` |
| 판정 | `should_be_no_name` |
| 제안 라벨 | `(이름 없음 -- 청크 내부 loop index i, 축이 아니라 구현 세부)` |
| 확신도 | high |
| 산출물 반영 | 미반영 |

**근거**

Codex 외부검토(2026-09-02, develop/codex_review_request_kimi_k3_followup.md): FLA naive_chunk_kda(fla/ops/kda/naive.py:101-125)의 청크 내부 causal loop "for i in range(1, BT): A[..., i, :i] = ..."가 만드는 prefix-length slice이지 d_head 관련 폭이 아니다. d_head=74 자체가 공식 raw config에 없는(라이브러리 기본값/파이프라인 파생) 유령 값이라는 것도 Codex가 확인 -- KDA의 실제 head_dim은 128(=d_head_kda)이지 74가 아니다. 트레이스 전체에서 d_head라는 이름이 실제 축에 단 한 번도 안 쓰인다(0건, 직접 재확인). rules/derived_dims.yaml의 d_head-d_rope/d_head//2 규칙이 값 우연 일치로 이 KDA 내부 축에 잘못 매치되고 있다. 「2*d_conv류」(기존 A62, model.layers.*.self_attn heuristic tail)와 같은 근본 원인(KDA 청크 스캔의 루프 인덱스가 이름 붙는 축으로 오인됨) -- 진짜 수정은 src/build_table.py의 _unname_loop_indices뿐인데 그 함수는 이미 2번의 정교화 시도가 함대 회귀로 되돌아간 이력이 있어(git log) 이번에도 보류.

## 발견 8 — 맞음 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.{3,7,11,...}.self_attn` |
| 축 | MLA 층의 KDA 심볼 |
| 현재 라벨 | `d_nope / d_v / n_h` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

전환 diff 의 semantic_change 936건. 옛 발행본(2026-09-09, B=1)이 MLA 층에 KDA 심볼(`d_head_kda`, `n_h_kda`)을 쓰고 있었다. 층 단위로 확인: 새 라벨 `d_nope`/`d_v` 는 layer 3,7,11,15,19,23...(block_type=MLA+MoE)에만 나오고, 옛 `d_head_kda` 는 layer 0,1,2,4,5,6(KDA 층)에 나온다. **MLA 층에서 KDA 심볼은 의미가 없으므로 새 판이 맞다** -- 스코프 과잉 매칭이 규칙 개선으로 잡힌 것이다.

## 발견 9 — table_omits_computation (미반영)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.self_attn` |
| 축 | B=1 판과 op 구간이 안 맞는 것 (372,948) |
| 현재 라벨 | `` |
| 판정 | `table_omits_computation` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 미반영 |

**근거**

옛 발행본은 B=1 로 잡혔다. torch 의 einsum 은 `sumproduct_pair` 에서 크기에 따라 축을 lro/lo/ro 로 나누는데, B=1 이면 batch 축이 ro 로 들어가 `swap_lo_ro` 가 피연산자를 교환하고 B>1 이면 lro 라 교환하지 않는다. 그래서 같은 contraction 이 서로 전치된 bmm 으로 내려가고, 앞뒤 permute/view 도 달라져 서명이 안 맞는다. 외부 검토(Codex 2026-09-18)가 설치된 torch 2.13.0+cpu 소스와 float64 수치 대조(B=1,2,3,4, rtol=atol=1e-12)로 **같은 계산의 다른 lowering** 임을 확인했다. 다만 **구간별 동치는 검증하지 않았다** -- 그것은 ports 의 DAG·scalar_args 로 원소 인덱스 대응을 합성해야 하고 다음 단계 작업이다. 대신 독립 배치 검증(B=3 발행 라벨을 B=4 실제 shape 과 대조)이 축 5,594,183개에서 미검증 0 / 어긋남 0 으로 통과했다.
