# 라벨 검토 결과 — moonshotai/Kimi-K3

- 검토일: 2026-08-25
- 검토자: llm(claude, 반박 프레임 전건 판정)
- 본 것: 의뢰서의 남은 4건 전부 A59/A60 이 이미 소스로 확정해 둔 판정과 정확히 같은 자리다 -- 새로 조사하지 않고, 그 두 판정을 이 항목들이 요구하는 형식(module/axis/current_label/verdict)으로 옮겨 적는다.
- 요약: 4건 전부 기존 판정으로 닫힌다. `d_head_kda`/`n_h vs n_kv`/`d_nope vs d_v` 는 A59(KDA는 MLA의 n_h/n_kv/d_nope/d_v 필드를 전혀 읽지 않는다)로 이미 확정. `2*E_shared` 는 A60(MoE-cap shim의 부산물 4, 아키텍처 상수 아님)으로 이미 확정. 새 코드 변경 없음.

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

A60(review/06-open-renames.md)에서 확정: 이 값은 아키텍처 차원이 아니라 `src/kda_shim.py`의 `patch_moe_infer`(MoE 라우팅이 값 의존적이라 트레이스 불가능해서 넣은 shim)가 `KDA_SHIM_EXPERT_CAP=4`(기본값)로 토큰을 4명의 전문가에게 균등 분할한 부산물이다 -- `1280 = k(16)*T(320)/4`, `4 = k(16)*1(decode)/4`, 실측으로 정확히 일치. `E_shared`(공유 전문가 수, =2)와는 아무 관계가 없고 `2*E_shared`라는 이름은 값이 우연히 맞아떨어진 heur_multiple 오라벨이다. 이미 `provenance.adaptation_log`에 `moe_infer_even_split`으로 기록돼 있고 C10 예외로도 처리된 것과 같은 부류라 이름을 붙이지 않는다 -- 이름을 붙이면 shim의 부산물을 아키텍처로 오해하게 만든다.
