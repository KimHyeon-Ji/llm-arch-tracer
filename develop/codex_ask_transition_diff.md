# 전환 diff 분류기 — 남은 480 을 어떻게 봅니까

지시하신 절차 중 4번(diff 를 분류해 `semantic_change`/`unexplained` 만 사람이 본다)을
만들었습니다. `develop/transition_diff.py` 입니다.

## 결과 (Llama-4, B=1 -> B=3)

```
자리(양쪽에 있음)   75,410
같음               70,332
batch_expected      3,820     T -> B*T (722, mm), n_h -> B*n_h (576, bmm) …
singleton_fixed       370     `B` -> `1` (방송 싱글턴을 배치로 오인했던 자리)
alignment_drift       408
semantic_change       480     <- 사람이 볼 것
unexplained             0
새 판에만            1,320 / 옛 판에만 840
```

## 만들면서 밀림을 세 번 걸러냈습니다

전부 "짝짓기가 틀렸는데 의미 변화로 보이던" 것이었습니다.

```
module_key 로 레이어를 접음        서수가 전 레이어에 걸쳐 세어져 밀림
  -> 전체 module_path 로                       사람이 볼 것 5,078 -> 1,262
같은 서수라도 rank 가 다름
  -> rank 를 키에 넣음                                      -> 888
그룹의 라벨 다중집합이 그대로
  -> 라벨이 추가·삭제되지 않았으면 의미 변화가 아니다          -> 480
```

마지막 것이 `alignment_drift` 입니다: 한 `(module_path, raw_op, rank)` 그룹에서 옛 판과 새
판의 **라벨 다중집합이 같으면** 라벨이 추가되거나 사라지지 않은 것이므로 자리만 밀린 것으로
봅니다.

## 남은 480 의 모습

```
120  `T` -> `d_head`        aten.view.default
120  `d_head` -> `T`        aten.view.default
 96  `d_head` -> `T`        aten._unsafe_view.default
 48  `d_head` -> `n_h*d_head`  aten.view.default
 48  `n_h` -> `B`           aten.view.default
 24  `B` -> `d_model`       aten.view.default
 12  `n_h` -> `T`  /  12  `T` -> `n_h`
```

`T <-> d_head` 가 120씩 대칭으로 남아 있습니다. 다중집합 판별식이 그룹 단위라 **그룹 안에
다른 변화가 섞이면** 다중집합이 안 맞아 대칭쌍까지 `semantic_change` 로 넘어갑니다.

---

## 질문

**Q1 (대칭쌍).** `T <-> d_head` 같은 대칭쌍을 그룹 단위가 아니라 **자리 쌍 단위**로 상쇄해도
됩니까? 즉 같은 그룹에서 `X -> Y` 와 `Y -> X` 가 같은 수로 있으면 그만큼을
`alignment_drift` 로 빼는 방식입니다. 밀림을 더 걸러내지만, 진짜로 두 축이 맞바뀐 경우
(전치 오라벨)도 함께 숨길 것 같아 걱정됩니다.

**Q2 (짝을 못 지은 자리).** "새 판에만 1,320 / 옛 판에만 840" 이 있습니다. 배치가 바뀌며
op 구성이 달라진 결과(연속성 때문에 `view` 가 `clone`+`_unsafe_view` 로 갈림)로 보이는데,
이것도 분류해서 봐야 합니까? 지금은 개수만 셉니다.

**Q3 (`n_h -> B` 48건).** 새 판에서 `B` 가 된 자리입니다. 배치가 40 이 아니라 3 이므로
값으로는 맞을 텐데, attention head 축이 배치가 되는 것은 이상해 보입니다. 이런 것은
`semantic_change` 로 두고 개별 확인하는 것이 맞습니까, 아니면 별도 경보 범주가 필요합니까?

**Q4 (기준선 전환 절차).** 주신 7단계 중 1~5 는 이 도구로 되는데, 6번(독립 배치 검사)은
`check_batch_labels.py` 가, 7번(승격)은 `promote.py` 가 합니다. 이 셋을 하나의 전환
스크립트로 묶는 것이 맞습니까, 아니면 따로 두고 순서만 문서로 남기는 편이 낫습니까?
