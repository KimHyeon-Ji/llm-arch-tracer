# 라벨 검토 결과 — meta-llama/Llama-3.1-405B

- 검토일: 2026-08-12
- 검토자: llm(claude, 행 단위 전건 — 검토자 방식)
- 본 것: 의뢰서에 **행 단위 전건 절**을 앞에 세우고 그 뷰로 다시 봤다. 접힌 표의 고유 행은 모델당 중앙값 62개(최대 136)뿐이라 전부 읽을 수 있다 — A/B/C 절보다 작으면서 한 행 안의 어긋남까지 보인다.
- 요약: 의뢰서의 질문에 전건 답했다.

> 이 파일은 `review_findings.json` 에서 생성된다 — 고칠 때는 JSON 을 고친다.

## 발견 1 — 맞음 (반영됨)

| 항목 | 값 |
|---|---|
| 모듈 | `model.layers.*.self_attn` |
| 축 | 128 을 두고 두 후보 |
| 현재 라벨 | `d_head vs n_h 동률` |
| 판정 | `current_label_correct` |
| 제안 라벨 | — |
| 확신도 | high |
| 산출물 반영 | 반영됨 |

**근거**

num_attention_heads == head_dim == 128 이다. **이 충돌이 `src/anchors.py` 가 존재하는 이유** — 모듈이 선언한 폭(`nn.Linear.weight == [out, in]`)으로 결정하지 값으로 결정하지 않는다. 예전에 값으로 결정하던 시절 KV head-size 축 16,859개가 `n_h` 로 오라벨됐고, 그래서 `head_excl`(한 shape 에 n_h·n_kv 공존 금지) 불변식이 추가됐다. 현재 그 지표는 0 이다.

**주의**: 외부 검토가 지적한 이 모델의 진짜 문제는 이 동률이 아니라 q/k/v_proj 의 `weight_shape` 가 `input_shape` 의 같은 텐서와 다른 이름을 쓰던 것이었고, 그건 새 불변식 `weight_operand` 로 잡아 교정했다(함대 4,406건 → 0).

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
