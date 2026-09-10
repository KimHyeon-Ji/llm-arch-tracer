# Llama-4 canary 결과 — 상한 10,176 이 확정값 9,456 으로 갈렸다 (2026-09-11)

외부 검토 지시: 공식 재트레이스 전에 **버리는 canary** 로 돌려 "질문 0" 의 정체를 가릴 것.
`develop/out` 에도 `models/` 에도 승격하지 않았다. 수치만 읽었다.

## 결과

```
prefill  자리 36,631   confirmed 30,855 / scope_inferred 4,656 / heuristic 736 / unresolved 384
decode   자리 36,439   confirmed 31,351 / scope_inferred 4,800 /                  unresolved 288
         open_tie 0        <- `ambiguous.json` 의 0 과 일치한다
```

**모르는 것이 질문 4개로 접힌다.**

```
3,216축  scope_inferred   E|d_head        -> d_head      (prefill, decode 각각)
  888축  scope_inferred   E|d_head        -> E           (decode 는 1,032)
  552축  scope_inferred   d_moe|w_local   -> d_moe
  736축  heuristic        reused_symbol   -> T           (같은 shape 에서 이미 쓴 이름 재사용)
  288축  unresolved       bare            -> 2
   96축  unresolved       bare            -> 0
```

## 앞서 낸 상한과의 대조

밖에서 재구성하던 도구는 10,176 을 냈다. 원장은 9,456(4,656+4,800)이다.

```
d_head   상한 6,432  =  원장 3,216 + 3,216   일치
E        상한 2,160  >  원장   888 + 1,032 = 1,920   (240 은 앵커가 확정으로 올렸다)
```

차이가 **정확히 앵커 분기**다. 외부 검토가 "앵커가 나중에 해결했을 수 있으니 상한이다" 라고
한 그대로였다.

## 판정

* Llama-4 의 "충돌 질문 0" 은 **탐지 맹점이 맞다.** 다만 라벨이 틀렸다는 뜻은 아니다 --
  `open_tie` 가 0 이므로 규칙이 못 고르는 자리는 없다.
* 실제 상태는 **9,456축이 scope 정규식만으로 갈렸고 아무도 검증하지 않았다** 는 것이다.
  모듈 분포는 그럴듯하다(`d_head` 는 `self_attn`, `E` 는 `feed_forward.*`).
* 그 밖에 **736축이 마지막 수단인 이름 재사용(`T`)** 이고 **384축은 이름이 아예 없다**.
  이 둘은 값 충돌이 아니어서 예전 지표에는 전혀 안 잡혔다.

## 다음

`E|d_head` 와 `d_moe|w_local` 두 family 의 scope 근거를 소스로 확인하면 9,456축이 한 번에
확정된다. `reused_symbol -> T` 736축은 따로 봐야 한다 -- 재사용은 근거가 아니다.
