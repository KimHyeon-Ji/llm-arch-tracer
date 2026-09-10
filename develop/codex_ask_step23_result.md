# 1~3단계 결과 보고 — 두 가지가 계획을 바꿉니다

지시하신 순서대로 1(314 manifest), 2(anti-union 분해), 3(semantic fixture)을 했습니다.
**#4/#5 구현은 시작하지 않았습니다.** 2번과 3번에서 계획의 전제를 바꾸는 실측이 나왔습니다.

---

## 1단계 — 314 oracle 을 구조 manifest 로 변환 (완료, 자기검사 통과)

`develop/build_314_manifest.py` / `develop/verify/oracle_314_manifest.json` (165 자리쌍).

각 항목은 "이 두 축 자리는 같은 등가류여야 한다" 는 주장이고 라벨을 안 봅니다. 자리 이름은

```
module_key | layer_sig | op_type | nth | i|o | shape_index | axis | rank
```

`layer_sig` 는 그 레이어의 `(module_key, op_type)` 집합 해시입니다. **config 를 안 읽고**
하이브리드 스택을 가릅니다.

> 처음엔 구체 shape 을 키에 넣었는데, 그러면 T(16/24)가 키에 박혀서 다른 `seq_len` 으로
> 재트레이스하면 자리를 잃고 검사가 조용히 다시 vacuous 해집니다. 지적하신 실패 형태 그대로라
> 레이어 지문으로 바꿨습니다. 두 방식이 같은 수치를 내서 교차 검증도 됐습니다.

```
mode = legacy       165건: 이어짐  80 / 끊김 85 / 자리없음 0
mode = provenance   165건: 이어짐 165 / 끊김  0 / 자리없음 0
```

자기검사(`verify-the-verifier`): ① 모드를 가른다(85 vs 0) ② 키를 10개 망가뜨리면
자리없음 10을 잡는다. 둘 다 PASS. `provenance_oracle.py` 에 배선했고 **자리없음 증가도
FAIL** 로 잡습니다(검사가 조용히 무력화되는 것을 막습니다).

참고로 기존 `positive_expand` 는 두 모드 모두 **0** 을 냅니다 — vacuous 하다는 지적이
출력으로 확인됐습니다. 참고용으로만 남겼습니다.

---

## 2단계 — anti-union 112 분해: **#4 로 풀리는 건 24건뿐입니다**

세 모델 모두 `batched_matmul` 의 앞쪽 배치 축 규칙(`src/axis_classes.py:270-283`)에서
합쳐집니다. 그런데 **원인이 셋 다 다르고, 셋 다 union 은 옳고 라벨이 틀렸습니다.**

| 모델 | 건수 | 원인 | #4 대상 |
|---|---|---|---|
| Zamba2-1.2B | 24 | `n_rep=1` 로 역할만 n_kv→n_h 로 바뀐 축 | **예** |
| Nemotron-3-Super | 40 | `view` 축 병합 오라벨 (`n_h_ssm=d_state=128` 값 충돌) | 아니오 |
| Kimi-K3 | 48 | 한 op 안에서 입력 `n_h` / 출력 `n_h_kda` (=96) | 아니오 |

```
Zamba2   op856 bmm  inp0 [n_h,  T,      d_head]  inp1 [n_kv, d_head, T]  out0 [n_h, T, T]
Nemotron op85  view inp0 [B, n_g_ssm, n_h_ssm/n_g_ssm, d_state] -> out0 [B, d_state, d_state]
                    (8*16 = 128 = n_h_ssm 인데 d_state 도 128 이라 d_state 로 찍혔다)
Kimi-K3  op58713 bmm inp0 [n_h, T, T]  inp1 [n_h, T, d_head_kda]  out0 [n_h_kda, T, d_head_kda]
```

Nemotron 은 `n_rep=16`, noop barrier 0. Kimi-K3 는 **semantic 이벤트가 0개**입니다.
barrier 로 건드릴 수 있는 모델이 아닙니다.

**질문 2-1**: 그렇다면 #4 의 성공 기준은 `112 -> 0` 이 아니라 **`112 -> 88`** 이 맞습니까?
지금 지표대로면 #4 가 끝나도 "실패" 로 보입니다.

**질문 2-2**: `anti_union` 이 **잘못 이음**과 **한 축에 두 이름**을 구분하지 않습니다.
`wrongly_joined` / `name_conflict` 로 쪼개려 합니다. 맞습니까? 쪼개면 88은 ④층
(`label_overrides`)이나 reshape 규칙의 일이 되는데, 그 88을 지금 손대는 것은 아직
이르다고 봅니다(#5 먼저).

---

## 3단계 — semantic fixture 5개 (`develop/test_semantic_fixtures.py`)

모델을 안 띄웁니다. 1·2·4 초록, **3·5 는 일부러 빨간불(XFAIL)** 로 두고 #5 가 끝나면
초록이 되게 했습니다. XFAIL 이 통과해버리면 러너가 FAIL 로 알립니다.

그런데 3번을 쓰면서 **지금 산출물에 있는 결함**을 찾았습니다.

### barrier 가 텐서/축 경계가 아니라 **전역 시간 구간**입니다

```python
# src/axis_classes.py:291
if not (any(b <= oid for b in bars) and any(b >= oid for b in bars)):
```

`min(bars) <= oid <= max(bars)` 인 전치를 **전부** 막습니다. barrier 가 층마다 흩어져 있으니
사이의 무관한 op 까지 통째로 막힙니다.

```
Zamba2-1.2B     barrier 12개, span [850, 5910] / 전체 op [0, 6419]
                전치 449개 중 355개 막힘(79%), 그중 330개가 attention 밖(mamba 275)
함대 합계       barrier 를 가진 모델 11개, 막힌 전치 5,134개, 그중 attention 밖 1,790개
                (OLMo-2 는 전부 attention 안이지만 161개 중 155개가 막힌다 -- 다른 층의
                 barrier 때문에 이 층의 전치가 막히는 것도 같은 결함이다)
```

이것이 `provenance` 모드의 `permute 끊김 13,900` 의 상당 부분입니다.

**질문 3-1**: 이건 #5(축별 barrier)의 일부입니까, 아니면 **그 전에 따로 고칠 결함**입니까?
지금 상태로는 `permute` sentinel 이 13,900 을 내는데, 그중 얼마가 이 결함이고 얼마가 진짜
계보 누락인지 구분이 안 됩니다. #5 설계를 확정하기 전에 barrier 를 `(tensor_uid, at_op_id)`
단위로 좁혀 sentinel 을 다시 재는 것이 순서로 맞습니까?

**질문 3-2**: 좁힌다면 경계를 무엇으로 답니까? no-op 이라 `in_tensor_id == out_tensor_id`
이므로 텐서 정체성만으로는 이전/이후를 못 가릅니다. `(tensor_uid, at_op_id)` 쌍으로 두고
"그 텐서를 소비하는 op 중 `op_id > at_op_id` 인 것만 n_h" 로 보는 것이 맞습니까?

---

## 지금 상태

```
mode = provenance
oracle manifest(165 자리쌍)  이어짐 165 / 끊김 0 / 자리없음 0
expand   이어짐  90,835 | 끊김      0 | 잘못 이음 0
permute  이어짐 409,775 | 끊김 13,900 | 잘못 이음 0   <- 상당 부분이 위 barrier 결함
split    이어짐  14,623 | 끊김      0 | 잘못 이음 0
anti-union 112  = Zamba2 24(#4 대상) + Nemotron 40 + Kimi-K3 48(라벨 결함)
```

`src/` 는 이번에 한 줄도 안 바꿨습니다. 전부 `develop/` 의 검증 도구입니다.
