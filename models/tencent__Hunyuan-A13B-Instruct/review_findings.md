# 라벨 검토 결과 — tencent/Hunyuan-A13B-Instruct

- 검토일: 2026-08-12
- 검토자: llm(claude, 행 단위 전건 — 검토자 방식)
- 본 것: 의뢰서에 **행 단위 전건 절**을 앞에 세우고 그 뷰로 다시 봤다. 접힌 표의 고유 행은 모델당 중앙값 62개(최대 136)뿐이라 전부 읽을 수 있다 — A/B/C 절보다 작으면서 한 행 안의 어긋남까지 보인다.
- 요약: 

> 이 파일은 `review_findings.json` 에서 생성된다 — 고칠 때는 JSON 을 고친다.

## 발견 1 — 교정 필요 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.mlp.experts` |
| 축 | 라우팅 슬롯 축 (T x top-k = 192) |
| 현재 라벨 | `3*E` |
| 판정 | `should_be_renamed` |
| 제안 라벨 | `k*T` |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

Hunyuan 은 `moe_topk` / `num_experts_per_tok` / `moe_intermediate_size` 를 **레이어별 리스트**로 적는다(`[8, 8, ...]`). 리스트는 int 가 아니라 `k` 가 아예 안 풀렸고, 그 자리를 값이 같은 `n_kv`(=8)가 가져갔다. 합쳐진 축 192 는 `3*E`(=3·64) 라는 산술 휴리스틱을 받았다 — 실측 `[24, 8] -> [192]` 이므로 `k*T` 다.

**교정**: `summarize._per_layer_scalar` — 레이어별 리스트의 원소가 전부 같으면 그 값으로 접는다. 원소가 다르면 접지 않는다(한 숫자로 말하면 거짓이 되므로 미해결로 두는 편이 정직하다).

같은 버그가 `src/run.py` 와 `develop/regen_summaries.py` 의 활성 파라미터 추정에도 있었고, 후자는 **모델별 try/except 안이라 ERROR 한 줄만 흘려보내고 그 모델을 조용히 건너뛰고 있었다** — Hunyuan 의 structure.yaml 이 한 세션 내내 낡은 채였다. regen 이 이제 실패 목록을 끝에서 다시 보고하고 종료코드 1 로 나간다.

## 발견 2 — 교정 필요 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.self_attn.q_proj` |
| 축 | q_proj 가중치 축 순서 (정사각이라 값으로 구별 불가) |
| 현재 라벨 | `[d_model, n_h*d_head] (= [in, out], 뒤집힘)` |
| 판정 | `should_be_renamed` |
| 제안 라벨 | `[n_h*d_head, d_model]` |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

행 뷰를 켜자마자 첫 화면에서 보였다:

    q_proj  weight_shape = ['d_model', 'n_h*d_head']    ← [in, out]
    k_proj  weight_shape = ['n_kv*d_head', 'd_model']   ← [out, in]

같은 블록의 같은 종류 가중치인데 축 순서가 반대다. `nn.Linear` 는 `[out, in]` 으로 저장하므로 k_proj 가 맞고 q_proj 가 뒤집혔다.

**원인은 하루 전 내가 넣은 `_weight_agrees_with_operand` 다.** 그 패스는 concrete shape 으로 '저장형이냐 전치형이냐'를 판정하는데, q_proj 는 **정사각**(70B 8192×8192, 405B 16384×16384)이라 두 해석이 모두 성립해 항등 매핑을 골랐고, 피연산자(전치형)의 라벨을 그대로 저장형에 복사해 순서를 뒤집었다. 값으로는 구별할 수 없다.

교정: 수축 op(`matmul`/`linear`/`mm`/`bmm`)은 **전치를 먹는다**는 op 의미로 방향을 정한다. q_proj 가 `['n_h*d_head', 'd_model']` 로 k_proj 와 일치한다.

**이 결함은 (모듈, 라벨) 뷰에서는 원리적으로 보이지 않는다** — 두 축 다 그 모듈의 정당한 이름이고, 문제는 순서뿐이기 때문이다. 검토자가 빨랐던 이유가 이거였다.
