# Llama-4 canary 결과 — scope 는 맞게 갈랐지만 아무도 검증하지 않습니다

지시하신 순서대로 1~5를 끝내고 커밋했습니다. 6단계(Llama-4 재트레이스) 전에 canary 판정
결과를 확인받고 싶습니다. **제 도구에 한계가 있어 지금 수치는 상한입니다.**

## 1. 끝낸 것

```
1 promote 의 증거 보존           커밋 2684ca17
2 codec 엄격성 + graph-input 이름  커밋 266bb004 에 포함
3 전용 단위 시험                 test_noderef 10/10, test_tracer_version 11/11
4 작은 meta/fake A/B 쓰기 검증    아래
5 source·tests 독립 커밋          2684ca17 / 918b25e8 / 266bb004
```

지적하신 네 구멍은 전부 사실이었고 고쳤습니다: 미래 판을 v2 로 읽던 것, `encode(None)` 이
null 을 내던 것, graph input 이름이 안 남던 것, 전용 시험이 없던 것.

### A/B 결과 (기준: "전부 byte-identical" 이 아님)

처음에 `models/` 와 비교해 CSV 가 다르다고 나왔는데 **비교 대상이 틀렸습니다** -- 발행본이
HEAD 코드보다 낡은 것이었습니다. HEAD 코드로 기준선을 직접 뽑아 다시 했습니다.

```
meta(tiny-llama) / fake(Hunyuan-A13B)  -> 다른 파일은 *.ports.jsonl 둘뿐
   CSV·JSONL·raw trace·axis_classes·unsettled·verdict_footprint 전부 바이트 동일
포트를 op 출처로 투영: v1 ≡ v2   (5,182 / 5,086 op 에서 불일치 0)
external 분류: parameter 451 / buffer 1 / graph_input 3(input_ids, position_ids) /
               unknown_external 65
제자리 연산: uid 가 입출력에 함께 나오는 op 160개, version 계약 통과
```

`unknown_external` 65 는 별도 게이트로 두라고 하신 것이라 아직 지표로만 두었습니다.

## 2. Llama-4 canary — 질문 0 의 정체

`develop/axis_decision_trace.py` 를 만들어 축을 전수로 훑고 등식을 세웠습니다.

```
=== meta-llama__Llama-4-Maverick-17B-128E
   값이 겹치는 축 35,858   (충돌 아닌 축 40,392)
     식/후보 밖                     25,682
     후보 여럿 · scope 만이 가른다    10,176     <- 질문으로 기록되지 않음
     후보 여럿 · scope 로도 안 갈림        0
     후보 여럿 · scope 에 아무도 안 맞음    0
```

등식이 맞습니다(25,682 + 10,176 = 35,858).

**그런데 그 10,176 을 모듈별로 보면 scope 가 맞게 갈랐습니다.**

```
  6,432축 -> d_head   self_attn
    720축 -> E        feed_forward.router
    720축 -> E        feed_forward.experts
    432축 -> E        feed_forward
     96축 -> E        feed_forward.experts.act_fn
```

즉 **Llama-4 의 라벨이 틀렸다는 증거는 아직 없습니다.** 문제는 다른 것입니다: 이 10,176건은
`rules/symbols.yaml` 의 정규식 두 개(`d_head`: `attn|attention|rotary|...`, `E`:
`expert|moe|router|...`)가 조용히 결정했고, **어디에도 질문으로 남지 않았습니다.** 모듈
경로가 하나만 어긋나도 경고 없이 뒤집힙니다 -- 이 저장소에서 가장 자주 난 라벨 버그가
정확히 그 형태입니다.

## 3. 제 도구의 한계 (먼저 밝힙니다)

`resolved_by_anchor` 분기가 **아직 없습니다.** 그래서 "scope 만이 가른다" 는
"scope 가 결정했다" 가 아니라 **"후보가 여럿인데 규칙에서 그 둘을 가르는 것이 scope 뿐이고
질문으로도 안 남았다"** 는 뜻입니다. 앵커가 정했는데 우연히 scope 와 일치했을 수도 있습니다.
지금 수치는 상한입니다.

세는 단위도 **occurrence** 입니다(prefill+decode, input/output/weight 전부). 그래서
`ambiguous.json` 의 `axes` 와 기준이 다릅니다.

```
모델              ambiguous.json    후보 여럿·scope 만이 가름    scope 로도 안 갈림
gpt-oss-120b            5,620                    7,398               13,608
gpt-oss-20b             3,748                    3,362                9,072
Llama-4                     0                   10,176                    0
DeepSeek-V4-Pro        27,282                  137,410               28,418
```

gpt-oss-120b 의 "scope 로도 안 갈림 13,608" 이 `ambiguous.json` 의 5,620 보다 큰 것이
설명되지 않습니다. 세는 단위 차이인지, `_pick()` 을 거치지 않고 앵커/전파로 이름이 붙은
축이 섞인 것인지 아직 모릅니다.

---

## 질문

**Q1 (scope 판정을 빈칸으로 둘 것인가).** 앞서 정한 빈칸 설계는 `_pick()` 동점만 덮습니다.
scope 로 갈린 축까지 덮으면 Llama-4 만 10,176축, V4-Pro 는 137,410축이 빈칸이 됩니다. 그런데
**분포를 보면 대체로 맞습니다.** 셋 중 무엇입니까?

  (a) scope 판정도 빈칸 -- 정직하지만 대부분 맞는 라벨을 지웁니다
  (b) 빈칸은 아니지만 **별도 증거 등급**으로 기록하고(`resolved_by_scope`) 게이트에서 세며,
      소스 확인 전까지 "확정" 으로 세지 않습니다
  (c) 지금처럼 둡니다

저는 (b) 로 기울었습니다. 다만 그러면 CSV 를 보는 사람은 여전히 확정 라벨과 구분 못 합니다 --
그건 앞서 지적하신 "사람이 계속 속는다" 와 같은 문제로 보입니다.

**Q2 (`resolved_by_anchor` 를 어디에 겁니까).** 축별 판정 경로를 남기려면 어느 지점을
계측해야 합니까? `_dim_core` 안에서 규칙 이름을 축마다 기록하는 것이 맞습니까, 아니면
`resolve_shape.stats` 처럼 집계만 하던 것을 축 단위로 바꾸는 편이 낫습니까? 후자는 메모리가
걱정됩니다(V4-Pro 는 축이 66만입니다).

**Q3 (세는 단위).** coverage 등식을 **occurrence** 로 셉니까, **등가류** 로 셉니까? 질문
개수와 맞추려면 등가류가 맞아 보이는데, 그러면 "몇 개의 축이 영향을 받는가" 를 잃습니다.
둘 다 내는 것이 맞습니까?

**Q4 (gpt-oss 의 13,608 vs 5,620).** 위 불일치를 어떻게 좁힙니까? 저는 `_pick()` 에 진입한
축과 앵커로 정해진 축을 갈라 세려 하는데, 그러려면 Q2 가 먼저 필요합니다.

**Q5 (scope 정규식의 근거).** `rules/symbols.yaml` 의 scope 정규식에는 `label_overrides` 의
`source:` 같은 근거 필드가 없습니다. 수천 축을 결정하는데 근거가 없습니다. **scope 에도
소스 인용을 요구**해야 합니까? 그러면 기존 정규식 전부를 소급해서 채워야 합니다.

**Q6 (다음 순서).** Q1~Q2 를 정한 뒤 Llama-4 재트레이스로 갑니까, 아니면 재트레이스는
빈칸 설계가 확정된 뒤로 미룹니까? 지금 재트레이스해도 `promote` 수정 덕에 증거는 보존되지만,
빈칸 설계가 바뀌면 어차피 다시 돌려야 합니다.
