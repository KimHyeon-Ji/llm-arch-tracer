# Codex 검토 요청 — Kimi-K3 reshape_incons 완전 해소 구현안

## 배경

앞선 답변에서 정리해 주신 대로 게이트 구멍 2개(freeform 전체-면제, phase-completeness)는
이미 고쳐서 커밋했다(`0a6972c2`). Kimi-K3의 `reshape_incons`(현재 255건)를 마저 닫는
남은 세 가지 — n_chunk 정식 등록, reshape 체크 동치 인정, 기존 `n_h*d_v` 계열 닫기 —
는 **구현하기 전에** 안전한 방법을 먼저 물어보라고 하셨다. 코드를 아직 안 고쳤다.

## 데이터 — 255건의 정확한 구성

```
decode:  69 × ('view', 현재='n_h*d_v', 역산='n_h_kda*d_head_kda')   -- KDA 레이어
         24 × ('view', 현재='n_h*d_v', 역산='n_h*d_nope')           -- MLA 레이어
prefill: 69 × ('view', 현재='n_h*d_v', 역산='n_h_kda*d_head_kda')   -- 같은 자리, prefill
         24 × ('view', 현재='n_h*d_v', 역산='n_h*d_nope')           -- 같은 자리, prefill
         69 × ('view', 현재='T',       역산='5*d_chunk')            -- 신규(어제 발견)
```

세 가지 유형이 있다: **(A)** MLA 레이어의 `n_h*d_v` vs `n_h*d_nope` (48건, 두 phase 합산),
**(B)** KDA 레이어의 `n_h*d_v` vs `n_h_kda*d_head_kda` (138건), **(C)** `T` vs `5*d_chunk`
(69건, prefill만).

## (A) MLA `n_h*d_v` vs `n_h*d_nope` — 기존 해결 전례가 있다

`review/06-open-renames.md` A47(Kimi-K2-Instruct)이 **정확히 같은 패턴**이다:
`split_with_sizes` 의 value 조각이 `d_nope`로 잘못 렌더되는데(값이 우연히 같음), o_proj
쪽(`n_h*d_v`)은 이미 맞다. A47은 데이터플로우를 따라가며 재추론하는 방식(`_split_from_
registered_sum`)을 시도했다가 `reshape_incons 61→122, flow_ambig 0→122`로 실패했고,
최종적으로 `rules/label_overrides.yaml`의 `spread: class` **직접 교정**(추론이 아니라
렌더 끝난 뒤 이름을 못 박는 방식)으로 해결했다(915축, 세 지표 전부 0).

Kimi-K3도 같은 전제(split 순서가 답을 이미 안다, MLA value 경로 head 폭 = d_v)가 맞는지,
같은 override 방식으로 닫아도 안전한지 확인받고 싶다.

## (B) KDA `n_h*d_v` vs `n_h_kda*d_head_kda` — 다른 원인일 수 있음

이건 A47과 다르다 — `d_v`/`d_nope`는 둘 다 MLA 전용 심볼인데, 이 축은 **KDA 레이어**의
o_proj 입력이다. `n_h_kda`/`d_head_kda`는 `scope_strict: true`로 self_attn 안에서만
쓰이게 등록해 뒀는데, 그 아래(o_proj로 들어가는 활성값 이름)에서는 `n_h`/`d_v`(MLA 심볼)로
렌더된다 — **KDA 레이어인데 MLA 심볼이 붙어 있다.** `scope_strict`가 위쪽(self_attn 안)만
막고 o_proj 직전 활성값까지는 못 미치는 건지, 아니면 다른 경로(reused_symbol 등)로 새는
건지 원인을 모르겠다.

## (C) `T` vs `5*d_chunk` — n_chunk 미등록 + reshape 체크 한계

`fla/ops/kda/naive.py`의 `naive_chunk_kda`가 `NT=T//BT=5`, `BT=d_chunk=64`로 청크를
나눴다가(`[B,H,NT,BT,...]`) 다시 시퀀스로 합치는(`[B,H,T,...]`) view다. `T`는 실제로
`avoid`에 없는 상태에서 정상적인 plain-symbol 경로로 선택된 것이라(어제 확인해 주신 대로)
`reused_symbol`을 안 건드리는 게 맞다는 데 동의한다.

이미 `rules/derived_dims.yaml:680`에 `n_h_kda * (T // d_chunk)`가 등록돼 있어서
`T // d_chunk`(=5) 계산 인프라는 있다. 다만 이걸 **단독 심볼**(`n_chunk` 같은)로 등록하려면
같은 모듈(self_attn) 안의 **인트라-청크 루프 리터럴 5**(순서상 다른 자리)와 안 겹치게
위치/등가류 근거로 갈라야 한다.

## 질문

1. **(A)를 A47과 같은 `label_overrides.yaml` `spread: class` 방식으로 닫아도 되는가?**
   Kimi-K3의 실제 split 코드(`modeling_kimi_linear.py`의 MLA 부분)를 봐서 A47과 같은
   구조(split 둘째 조각이 value, o_proj가 `n_h*d_v`로 맞게 이미 렌더됨)인지 확인해 줄 수
   있는가?
2. **(B)의 정확한 원인은 무엇인가?** `n_h_kda`/`d_head_kda`가 `scope_strict: true`인데도
   왜 o_proj 입력 축에서 MLA 심볼(`n_h`/`d_v`)로 새는지 코드 경로를 짚어달라. (A)와 같은
   override 방식이 안전한지, 아니면 스코프 전파 자체를 고쳐야 하는지.
3. **(C)의 `n_chunk`를 안전하게 등록하는 구체적 방법은?** 어떤 위치 신호(op_type, shape
   패턴, depends_on 등)로 "청크 재배열의 NT" 자리와 "인트라-청크 루프의 리터럴 5" 자리를
   구별해야 하는가? `derived_dims.yaml`에 `T // d_chunk` 단독 항목을 추가하는 것만으로
   되는가, 아니면 `canonical_axis_rules.yaml` 같은 위치 규칙이 따로 필요한가?
4. **reshape 체크(`build_table.reshape_disagreements`/`derive_from_reshape`) 자체를
   고쳐서 `n_chunk*d_chunk == T`류 등록된 관계를 동치로 인정하는 게 안전한가**, 아니면
   (A)(B)(C)를 전부 라벨 쪽에서 먼저 닫아서 애초에 이 함수가 그 축들을 안 건드리게 만드는
   게 더 안전한가? (전자는 checker 로직을 건드리는 거라 fleet 전체에 영향, 후자는 라벨
   교정이라 국소적)

이번에도 구현 초안은 이쪽에서 하지 않고, 방향과 구체적 코드 위치를 먼저 받고 싶다.
