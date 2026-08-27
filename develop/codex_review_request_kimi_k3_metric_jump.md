# Codex 검토 요청 — Kimi-K3 bare/reshape_incons 급증, 원인 분석 결과 확인

## 배경

Kimi-K3 재트레이스 → 승격 → selector 수정(둘 다 완료, dead_confirm=0 확인) → 함대 전체
재생성까지 Codex가 준 16단계 순서대로 진행했다. 마지막 `verify_all.py`에서 baseline 대비
Kimi-K3의 두 지표가 크게 늘었다:

```
FAIL  moonshotai__Kimi-K3: bare 퇴행 6541 -> 446701
FAIL  moonshotai__Kimi-K3: reshape_incons 퇴행 93 -> 255
```

바로 baseline을 덮지 않고 원인을 먼저 조사했다. 아래가 그 결과다 — **이게 맞는 해석인지,
그리고 baseline을 올려도 되는지 확인받고 싶다.**

## 1. `bare` 68배 증가 — 원인 확인됨, 안전하다고 판단

`structure.yaml`의 `label_provenance.heuristic_examples`를 보면 새로 생긴 bare/heur 항목이
전부 `model.layers.N.self_attn`에서 `2*d_conv`/`3*d_conv`/`4*d_conv`/`n_h_kda/2` 같은 이름이고,
레이어마다 반복된다(93개 레이어 × 여러 축).

`self_attn` 안의 bare 정수를 값별로 세어보면:
```
self_attn bare<=16: 692589   self_attn bare>16: 52854
large_vals 최빈값: 17,18,19,20,21,...,26 (전부 1173회씩)
```

이건 오늘 아침 이미 `develop/verify/references.yaml`의 `irreducible_literals`에 등록해 둔
**KDA 청크 스캔의 인트라-청크 순차 재귀 루프 경계**(`fla/ops/kda/naive.py:134`
`for i in range(1, BT)`, BT=64)와 정확히 일치하는 패턴이다 — 그 등록 항목은 `values:
any_small_odd_or_scan`(freeform, 값 상한 없음)으로 이미 이 모델 **전체**를 커버해 뒀다.
실제로 `develop/verify_all.py`의 `[EXTERNAL] 이름 없는 정수에 문서화된 사유가 있는가`
검사는 여전히 통과한다("80개 트레이스 전부 문서화된 값만 사용").

**내 해석**: `reused_symbol` 버그가 이 루프-사다리 축들 상당수에 **가짜로 자신 있어 보이는
이름**(스코프 밖 심볼과 값만 맞아 떨어진 것)을 붙이고 있었는데, 버그를 고치자 그 가짜 이름이
없어지고 정직한 bare/heur로 드러났을 뿐이다. 오늘 아침 문서화해 둔 바로 그 현상의 **전체
규모**가 이제야 정확히 보이는 것이지, 새로 생긴 문제가 아니라고 판단한다.

**질문**: 이 해석이 맞는가? 아니면 `heur<=16`/`heur>16` 값 분포가 "68배"라는 크기에 비해
너무 획일적이라(1173회씩 정확히 반복) 다른 걸 놓치고 있을 가능성이 있는가?

## 2. `reshape_incons` 93 -> 255 — 새 불일치 유형 발견

decode는 그대로 93건(기존과 똑같은 `n_h*d_v` vs `n_h_kda*d_head_kda`/`n_h*d_nope` 패턴).
prefill에서 **새로운** 유형이 69건(=KDA 레이어 수) 추가됐다:

```
('view', 'T', '5*d_chunk')   69건
```

즉 어떤 view의 출력 축이 지금 `T`(시퀀스 길이 심볼)로 렌더되는데, 같은 텐서를 입력 축에서
역산하면 `5*d_chunk`(=5*64=320, 이 트레이스의 T=320과 값이 우연히 같음)가 나온다.

**내 가설**: `reused_symbol` 수정 전에는 이 축이 `avoid`에 들어있지도 않은 다른 심볼로
잘못 재사용되고 있었을 가능성이 있고(값만 맞으면 `ordered_ctx` 아무 데서나 골랐으니까),
그 우연이 이 특정 불일치를 우연히 피하고 있었을 수 있다. 수정 후 `T`가 **진짜로** `avoid`에
있어서(같은 shape 튜플 안 다른 축에서 이미 정당하게 쓰임) 재사용되는 것이라면, 이건
DeepSeek MLA의 `n_h*d_v` 케이스와 같은 부류(두 관점 다 참, 어느 쪽도 틀리지 않음)일 수 있다.

다만 코드에는 이미 "T는 값 충돌 시 배제한다"는 선례가 있다(`src/symbolic_shape.py`의
`heur_multiple`/`heur_plus1` 근처 주석: "T is EXCLUDED... would otherwise render 2*T and
fabricate a sequence dependency on a fixed dim" — 이 정확히 같은 클래스의 문제를 이미 한 번
겪었다는 뜻으로 보인다). **`reused_symbol` 폴백에는 이 T 배제가 없다.**

**질문**:
1. `reused_symbol`이 `T`를 재사용 후보로 허용하는 게 맞는가, 아니면 heur_multiple/heur_plus1
   처럼 여기서도 `s != "T"` 가드를 추가해야 하는가? (T가 실제 시퀀스 축이 아니라 청크 개수의
   곱과 우연히 같은 값을 갖는 자리라 T로 부르면 다른 seq_len에서 거짓이 된다 -- 이 트레이스는
   T=320 고정이라 안 드러나지만, 03-labeling-roadmap.md 등에서 강조하는 "seq_len 불변성
   위반" 문제로 보인다.)
2. 이 69건이 정확히 `naive_chunk_kda`의 `[B,H,NT,BT,...]` 청크 재배열(`NT=T//BT=5`,
   `BT=d_chunk=64`) 축이라고 보는 게 맞는가? (이미 등록된 `n_h_kda*n_chunk` 유도식이나
   `T // d_chunk` 관련 derived_dims 규칙과 관련이 있을 것 같다.)

## 결정을 미룬 것

`reused_symbol`에 `s != "T"` 가드를 추가하는 게 맞다면 그건 오늘 낸 커밋(`d3a922ce`) 이후의
**네 번째** 수정이 된다. 이것도 fleet 전체에 영향을 주는 공유 코드라 바로 고치지 않고
답변부터 받는다. bare 증가(1번)는 이미 확신이 있어 baseline을 올려도 된다고 보지만,
reshape_incons 증가(2번)는 실제 버그일 가능성이 있어 baseline을 올리지 않고 대기 중이다.
