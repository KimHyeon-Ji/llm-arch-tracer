# Kimi-K3 B=1 → B=3 전환 검토

검토일: 2026-09-18. 대상: `develop/codex_ask_kimi_k3_batch_topology.md`와 로컬에 저장된 후보 `develop/out/moonshotai__Kimi-K3/`.
기존 규칙·후보 파일은 수정하지 않았다. 아래 오류는 **저장된 산출물**의 오류이며, 현재 resolver를 다시 실행해도 그대로 발생한다고 단정하지 않는다.

**판정: 제시된 bmm 전치는 같은 contraction의 다른 lowering이다. 그러나 실제 라벨 오류도 확인되어, 376,428건 전체를 정상 lowering으로 승인하거나 독립 배치 검사 하나만으로 이 후보를 출고할 수는 없다.**

## Q1 — 전치 원인과 동치

설치된 `torch 2.13.0+cpu`, git `cf30153c4c131c8164ee7798e5022d810682e2cb`에서 요청서의 차이를 재현했다.
`einsum('... c d, ... d -> ... c', X, y)`의 결과는
`z[b,h,c] = sum_d X[b,h,c,d] * y[b,h,d]`이다.

| B | 실제 bmm |
|---|---|
| 1 | `[96,1,128] @ [96,128,64] -> [96,1,64]` |
| 3 | `[288,64,128] @ [288,128,1] -> [288,64,1]` |

ATen `sumproduct_pair`는 두 피연산자에서 크기가 1인지에 따라 축을 lro/lo/ro로 분류한다. B=1일 때 batch 축은 ro에 들어가고, chunk 축 lo보다 앞에 있으므로 `swap_lo_ro`가 피연산자를 교환한다. B>1이면 batch 축이 lro로 들어가며 교환하지 않는다. 뒤의 view/permute가 결과 축 순서를 복원한다. 이는 **같은 einsum 구현 내부의 크기 의존 분해 최적화**다. CPU/CUDA 커널 dispatch가 달라졌다는 설명은 정확하지 않다. [설치된 torch commit의 소스](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/Linear.cpp#L166-L274)

Float64 난수로 B=1,2,3,4 각각을 명시적 곱·합과 비교하고, 배치 실행을 독립 B=1 실행들의 연결과 비교했다. 모두 `rtol=atol=1e-12`로 통과했다. 이 검사는 해당 contraction을 지지하며 전체 모델의 동치를 증명하지 않는다. 부동소수점 결과의 bitwise 일치도 요구하지 않는다.

**320회 패턴의 정확한 위치:** 설치된 FLA `naive.py:155–158`의 청크별 `Aqk` 계산이다. NT=320/64=5, 내부 j 루프 64이므로 320회다. 요청서가 지목한 첫 `A` 계산(`127–130`)은 64회이고, 그때 bmm batch 축에는 NT까지 접혀 B=1에서 480, B=3에서 1440이다. 두 계산에 같은 einsum 식이 사용된다. [FLA 소스](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/naive.py)

## Q2 — 승인 조건

**입출력 shape 집합, 전치 허용, numel 보존, 파라미터 접근 일치만으로는 충분하지 않다.** 같은 shape의 다른 텐서를 사용하거나 slice 조각을 중복해도 이 조건을 만족할 수 있다.

현재 `develop/transition_diff.py:127`의 `classify_unmatched`는 실제 외부 경계 텐서를 비교하지 않는다. 남은 op들의 모든 input/output **shape 집합**을 합쳐 포함관계를 비교한다. 텐서 identity, 입력 순서·중복, slice 범위, permute 차원, 연결 관계를 검증하지 않는다. 호출부 또한 모듈 전체의 leftovers를 전달하므로 하나의 연결된 구간이라고 보장되지 않는다.

재현 스크립트의 반례:

```text
old: alias(x)                         # [4] -> [4]
new: a=x[:2]; cat([a,a])              # [4] -> [2] -> [4]
x=[0,1,2,3], new=[0,1,0,1]
현재 판정: sequence_partition_expected
```

토큰 손실·중복이 없다는 주석의 주장은 이 검사로 성립하지 않는다. 따라서 bmm를 reshape 허용 목록에 추가하는 수정은 피해야 한다.

권장하는 별도 분류는 `contraction_lowering_verified`다. 승인 대상은 지원하는 국소 패턴으로 제한한다.

1. ports의 `input_sources`, `op_id/output_slot`, tensor IDs로 연결된 DAG와 실제 경계 입력·출력을 찾는다. shape가 같은 것만으로 대응시키지 않는다.
2. `scalar_args`로 view/permute/broadcast의 원소 인덱스 대응을 합성한다. 양쪽이 같은 batch/head 대응, 같은 입력값, 같은 수축 축을 갖는지 확인한다.
3. 두 피연산자의 교환·전치와 출력 복원이 같은 contraction으로 귀결되는 경우만 승인한다. 모든 외부 출력과 상태 갱신, alias/in-place 효과를 보존해야 한다. 확인할 수 없는 stride/storage 관계는 보수적으로 미검증 처리한다.
4. proof에 경계와 op anchors를 기록한다. 난수 수치 비교를 보조 증거로 붙이고, 승인한 구간 수와 남은 미검증 차이를 따로 집계한다.

현재 ports는 schema 2이고 scalar_args와 producer 연결을 이미 포함한다. 이 자료를 우선 활용할 수 있다. 모듈 전체의 op 수·종류 일치는 원인 추적에 유용하지만 동치 증명은 아니다.

## Q3 — 출고 기준

**diff가 0이어야 한다는 기준은 버릴 수 있다. 미검증 diff를 독립 배치 PASS 하나로 승인하는 기준은 채택하면 안 된다.**

독립 배치 검사는 기존 발행 라벨을 다시 만들지 않고 B=4의 구체 shape와 비교해야 한다. 이는 라벨의 배치 의존성을 검증한다. B=3과 B=4는 모두 B>1 lowering 경로이므로, 이 검사 자체는 B=1 경로와의 계산 동치를 증명하지 않는다. 고정 길이 48을 `n_h_kda/2`로 잘못 이름 붙인 경우도 H가 고정이면 통과한다.

현재 `develop/check_batch_labels.py`의 PASS에는 다음 빈틈이 있다:

- `weight_shape`를 검사하지 않는다.
- 텐서 목록을 zip으로 비교해 입력/출력 개수 차이를 놓칠 수 있다.
- rank 불일치와 평가 불가능한 식을 실패/미검증으로 세지 않고 건너뛴다.
- probe가 반환한 phase만 순회하며, 발행 trace가 없는 phase도 건너뛴다.
- op 종류별 등장 순서는 dataflow 대응 자체를 보장하지 않는다.

짝을 못 지은 op 수는 이미 실패로 처리한다. 여기에 위 coverage 누락도 실패 또는 명시적 미검증으로 포함하고, 발행 B=3에서의 산술 일치부터 확인해야 한다. probe의 model revision을 후보의 resolved revision에 고정하고 T, config, torch/FLA 버전, shim 조건을 동일하게 해야 한다.

출고 조건은 **(a) 기본 산술 검사와 누락 없는 독립 배치 검증, (b) 지원 lowering 패턴의 국소 동치 검증, (c) 남은 의미 변화와 미검증 구간의 해소**로 권장한다. 증명된 lowering 차이는 공개 사항으로 기록할 수 있다.

저장된 `audit_manifest.json`은 2026-09-17 02:23:40 판으로 `independent_batch_check=fail`, `approved=false`다. diff 수도 요청서의 최신 376,428과 다르다. 이를 지금 실행 중인 검증의 결과로 해석하지 않았다. 이번 검토에서는 새 전체 모델 B=4 probe를 실행하지 않았다.

후보 provenance는 FakeTensor/CPU reference KDA와 MoE even-split 대체 경로(896개 중 expert cap 4)를 선언한다. 따라서 출고 설명에서도 이 트레이스의 구현 범위를 명시해야 한다. 실제 GPU Triton kernel의 op 구성이나 성능을 이 검증으로 승인할 수는 없다.

## Q4 — 실제 라벨 오류

layer 0 `self_attn`과 하위 모듈을 raw labels, concrete shapes, ports, 실행 소스로 대조했다. weight 축도 포함했다. 전 모델을 검사한 결과는 아니다.

| phase | op 수 | 평가한 축 | 평가 불가 | 산술 불일치 |
|---|---:|---:|---:|---:|
| prefill | 8,810 | 76,192 | 0 | 14 |
| decode | 136 | 1,017 | 0 | 6 |

별도로 prefill에서 값은 맞지만 의미가 틀린 `n_h_kda/2` 라벨 11곳도 찾았다.

| 문제 | 대표 anchor | 근거와 교정 |
|---|---|---|
| `B*d_head-d_rope` | prefill 3461, output[0], axis 3 | 식은 3×74−64=158인데 실제 길이는 30. ports slice `[3,0,30]`, FLA 삼각 루프 i=30. **30** 또는 scoped loop index로 교정. 연결된 clone/sum/add/copy까지 11축 오류. |
| `n_h_kda/2` | prefill 4001, output[0], axis 3 | 96/2=48로 수치는 맞지만 ports slice `[3,0,48]`, 삼각 루프 i=48. head 절반이 아니다. **48** 또는 scoped loop index로 교정하고 전파. 11축. |
| convolution cache `1` | prefill 71/85/99, output[0], axis 2 | q/k/v cache 실제 길이는 3. ports는 마지막 3개를 slice. 실행된 `src/kda_shim.py:204`의 `kernel_size-1`, kernel=4. **3** 또는 해당 convolution의 `d_conv-1`. |
| decode cache `1` | decode 61/76/91, input[0], axis 2; 72/87/102, output[0], axis 2 | 같은 cache 길이 3을 cat 입력과 slice 출력에서 1로 표기. decode query 길이 1과 구분해야 한다. |

`B*(d_head-d_rope)`로 괄호만 고쳐도 안 된다. B=3에서 우연히 30이지만 이 slice 길이는 batch와 무관하다.

KDA의 주요 역할은 소스와 맞는다:

- `B*n_h_kda`: 청크를 하나씩 처리하는 Aqk/state matmul의 folded batch. B=3이면 288.
- `B*n_h_kda*n_chunk`: 모든 청크를 함께 처리하는 첫 A, A@k/A@v의 folded batch. B=3이면 1440. B=1의 480을 후보에서도 batch 없는 480으로 취급하면 안 된다.
- `d_chunk=64`: 청크 안 token 축. `d_head_kda=128`: KDA key/value 폭. 일반 `d_head=74`를 이 폭으로 대체하면 안 된다.
- `n_chunk=T//d_chunk=5`는 이 함수의 chunk 축으로 scoped하게 사용한다. 숫자 5 또는 48 전체에 값 매칭으로 이름을 붙이지 않는다.
- key 폭 K와 value 폭 V는 이 모델에서 둘 다 128이다. 양쪽에 같은 이름이 보이는 것만으로 일반 K≠V 구현까지 검증됐다고 간주하지 않는다.

## 재현

```powershell
.venv\Scripts\python.exe develop/repro_kimi_k3_batch_topology.py --output develop/kimi_k3_batch_topology_evidence.json
```

스크립트는 모델을 다운로드하거나 재트레이스하지 않는다. bmm lowering 수치 검사, 현재 classifier의 반례, 저장된 layer-0 산술 검사와 오류 anchors를 기록한다. 산술 검사 통과와 의미 정확성은 별개이므로, `n_h_kda/2`처럼 수치가 맞는 오류는 소스 검토 항목으로 분리했다.
