# 접힌 배치 축 — 지적이 맞습니다. 고치는 방법을 묻습니다

`B*n_h` / `B*T` / `E*B` 누락 지적이 정확했습니다. 기계적으로 확인했고, 원인까지 찾았는데
**고치는 방법이 간단하지 않아** 판단을 받고 싶습니다.

## 1. 실측 — 지적대로입니다

배치 의존 축(B=2 프로브로 확인)인데 **라벨에 `B` 가 없는** 자리:

```
prefill  배치 의존 8,569 중 라벨에 B 없음 **2,884**
            722  matmul          `T`   축0     -> B*T 여야
            313  view            `T`   축0
            288  batched_matmul  `n_h` 축0     -> B*n_h 여야
            265  _unsafe_view    `T`   축0
             96  batched_matmul  `T`   축1
decode   배치 의존 10,105 중 없음 **720**
            288  batched_matmul  `n_h` 축0
            192  view            `n_h` 축0
            144  E               축0            -> E*B 여야
```

제 B=2 프로브가 이걸 못 잡은 이유도 지적하신 그대로입니다 -- **축의 위치**만 검사하고
**라벨이 `B` 를 담는지**는 안 봤습니다.

## 2. 원인 두 겹

### (가) reshape 도출이 크기 1 축을 통째로 건너뛴다

`build_table.derive_from_reshape` 의 주석: "Size-1 axes are skipped entirely: they are runtime
singletons already governed by the batch-axis invariant." B=1 이면 배치 축이 크기 1 이라
곱에서 빠집니다. `[1, n_h, T, d] -> [n_h*?, ...]` 가 `n_h` 로 찍히는 이유입니다.

### (나) 그 도출을 적용하는 패스가 **일부러 꺼져 있다**

`_apply_merge_derivation` 과 `_carry_reshape_labels` 둘 다 호출되지 않습니다. 코드 주석에
2026-08-06 측정이 남아 있습니다:

> reshape_incons 773 -> 8 로 내려가고 라벨도 맞지만, flow_ambig 가 2,015 -> 2,187 로 오른다.
> MoE 라우팅 영역에는 텐서 정체가 실제로 바뀌는 op(`sort`, `index`)가 있어 값 매칭이 그걸
> 따로 라벨링한다. 부분적으로 고쳐진 영역은 일관되게 틀린 영역보다 읽기 나쁘다.

즉 `derive_from_reshape` 는 지금 **검사기(`reshape_disagreements`)에서만** 쓰입니다.

## 3. 순진하게 고쳤더니 틀린 라벨이 나왔습니다

(가)를 고치고(크기 1 축이라도 라벨이 `B` 면 곱에 넣는다) (나)를 켜서 Llama-4 를 돌렸습니다.

```
좋아진 것   op2  [T, d_model]      -> [B*T, d_model]        <- 원하던 것
망가진 것   op0  [B, T, d_model]   -> [B, B*T, d_model]     <- B 가 중복
```

prefill 90칸이 바뀌었는데 그중 상당수가 후자입니다. **배치 축이 아직 살아 있는 텐서에까지
곱을 먹였습니다.** 게다가 `verify_all` 의 게이트가 FAIL 0 을 냈습니다 -- 기존 `batch_excl`
불변식은 `B` 가 **곱 안에** 들어간 경우를 안 봅니다.

그래서 되돌렸습니다. 지금 트리는 이 실험을 담고 있지 않습니다.

## 4. 지금 상태

```
verify_all       Llama-4 FAIL 0
B=2 프로브        미설명 밀림 0 / 축 0 밖 위반 0 -> PASS  (다만 위 누락은 못 잡는다)
출고 게이트       Llama-4 PASS
```

지적하신 나머지도 정리했습니다.

* decode 의 가짜 `B` -> `1` 은 `spread: class` 로 등가류 전체에 폈습니다(불일치 192 -> 0).
* `context` 의 `public_context_length: 1048576` 은 **base SKU 기준으로 틀렸다**는 지적을
  확인했습니다. `sku_types.py` 가 base 262144 / Instruct 1048576 으로 가릅니다. 아직 안
  고쳤습니다 -- 아래 Q3.

---

## 질문

**Q1 (어디까지 고쳐야 합니까).** 접힌 배치를 라벨에 담으려면 (가)+(나)를 둘 다 해야 하는데,
(나)는 2026-08-06 에 측정 후 거부된 변경입니다. 셋 중 무엇입니까?

  (a) (나)를 다시 켜되 **MoE 라우팅 영역을 제외**하고 적용한다(거부 사유가 그 영역이었다)
  (b) 접힌 배치만을 위한 **별도 좁은 패스**를 새로 만든다 -- reshape 이 배치 축을 없앨 때만
      출력 축에 `B*` 를 붙이고, 그 밖의 merge 도출은 계속 끈 채로 둔다
  (c) 이번 출고는 접힌 배치를 그대로 두고 `known_limits` 에 "이 표는 B=1 물리 shape 이며
      접힌 배치가 라벨에 드러나지 않는다" 를 명시한다

저는 (b) 로 기울었습니다. 거부된 변경을 되살리지 않으면서 지적하신 자리를 정확히 덮고,
"배치 축이 사라진 reshape" 이라는 조건이 구조로 판정 가능해 보입니다. 다만 그러면
`[B*n_h, T, d]` 같은 라벨이 새로 생기므로 **`batch_excl` 불변식을 곱 안까지 보도록** 함께
고쳐야 합니다(지금은 `[B, B*T, d]` 를 통과시킵니다).

**Q2 (게이트).** 제안하신 "B=1/B=2 각각에서 심볼식을 수치화해 실제 shape 와 전건 비교" 를
넣으려 합니다. 심볼식 평가기가 필요한데(`B*n_h`, `E*B*T`, `T+1` …), 기존에 그런 것이
있는지 못 찾았습니다. 새로 만드는 것이 맞습니까? 그 게이트가 있으면 이번 누락은 물론
`[B, B*T, d]` 같은 중복도 잡힙니다.

**Q3 (context).** base SKU 는 262,144 가 맞다는 지적을 받아들입니다. 두 가지 중 무엇입니까?

  (a) `public_context_length: 262144` 로 고치고 1M 은 빼기
  (b) `public_context_length: 262144` + `family_claim: 1048576 (Instruct SKU)` 로 둘 다 남기기

저는 (b) 가 나아 보입니다 -- 1M 이라는 숫자를 본 사람이 왜 다른지 알 수 있어야 합니다.

**Q4 (출고 패키지).** "네 파일만 공개하지 말고 `structure.yaml`, provenance, known-limits 를
같은 패키지에" 라고 하셨습니다. 지금 출고본에는 `structure.yaml` / `model_summary.md` /
`review_findings.*` / 축 판정 사이드카가 들어가고 **provenance 는 `full/` 안이라 빠집니다**.
`full/provenance.json` 도 건져 올리면 됩니까, 아니면 필요한 필드만 추려 별도 파일로
내보내는 편이 낫습니까?
