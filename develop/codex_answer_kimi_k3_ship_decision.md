# Kimi-K3 출고 판정 — round 3

검토일: 2026-09-19  
대상: `develop/codex_ask_kimi_k3_ship_decision.md`, 현재 저장된 `develop/out/moonshotai__Kimi-K3/`,
전환 검사 코드와 Kimi-K3의 고정 모델 revision.

기존 규칙과 후보 산출물은 수정하지 않았다.

## 결론

**현재 Kimi-K3는 출고하지 않는 편이 맞다. `UNKNOWNS.md` 공개만으로는 부족하며,
최소 범위의 `contraction_lowering_verified`를 먼저 구현해야 한다. 이미 출고한 네 모델은
그대로 유지하고 K3만 별도 보류한다.**

이 결론은 diff가 크기 때문이 아니다. 현재 372,948은 계산 동치가 증명된 차이가 아니라,
모듈 단위로 모은 미매칭 op 레코드다. 현재 분류기는 연결된 구간을 만들지 않고, 실제 외부
경계 텐서·원소 인덱스 대응·수축 축을 검사하지 않는다. 더구나 요청서의 원인 설명과 달리
이 레코드에는 KDA뿐 아니라 MLA의 layout 차이도 포함된다. 공개는 미검증 상태를 정직하게
알릴 수는 있어도, 출고 조건인 국소 동치 검증을 대신하지 못한다.

검토 시점의 릴리스 상태에도 별도 차단 사유가 있다.

- `audit_manifest.json`은 `approved: false`이고, `needs_review`가 남아 있다.
- manifest의 `source_commit`은 `9f0dfd...`인데 현재 HEAD는 `1fe8dca...`다.
- manifest는 `semantic_change: 3,696`을 기록하지만 현재 전환 검사는 936을 출력한다.
- 요청서의 전환 표는 `같음 4,047,411`, `fabrication_withdrawn 2,760`이지만 현재 재실행 결과는
  각각 4,045,893과 4,278이다. 두 항목 사이에 1,518건이 이동했다.
- 후보 디렉터리에 `UNKNOWNS.md`도 아직 없다.

따라서 동치 검증을 마친 뒤 현재 commit과 산출물로 audit manifest와 공개 문서를 다시
생성해야 한다.

## Q1 — `semantic_change` 936

**회귀가 아니라 옛 라벨 오염의 교정이라는 판정은 맞다.** 다만 `layer_idx` 분포만으로
끝내지 말고 모델 소스의 layer dispatch와 MLA tensor split을 근거로 삼아야 한다. 이번
검토에서 그 대조까지 수행했다.

Kimi-K3 설정의 `is_kda_layer(layer_idx)`는 `(layer_idx + 1)`이 `kda_layers`에 있는지를
검사한다. `KimiDecoderLayer`는 그 결과가 참이면 `KimiDeltaAttention`, 아니고 MLA 설정이면
`KimiMLAAttention`을 만든다. 고정 config에서 MLA 층은 다음 24개다.

```text
0-based: 3,7,11,15,19,23,27,31,35,39,43,47,
         51,55,59,63,67,71,75,79,83,87,91,92
```

따라서 요청서의 `3,7,11,…,47의 24개`는 끝점이 잘못됐다. 47까지는 12개이고, 실제 24개는
91까지의 4간격 층과 마지막 92층이다. 옛 trace에서 이 24개 MLA 층에 KDA 이름이 등장하고
새 trace에서 MLA 이름으로 바뀐 것은 layer dispatch와 일치한다.

같은 값인 `d_nope=128`과 `d_v=128`의 경계도 소스로 구별된다.

- query의 첫 split은 `qk_nope_head_dim`, `qk_rope_head_dim` 순서다. 첫 출력 `q_pass`가
  `d_nope`다.
- `kv_b_proj` 뒤의 split은 `qk_nope_head_dim`, `v_head_dim` 순서다. `k_pass`가 `d_nope`,
  `value_states`가 `d_v`다.
- `value_states`는 attention의 value 입력과 `o_proj`의 `num_heads * v_head_dim` 경로로
  이어진다.
- `n_h`는 MLA의 `num_heads`이며, KDA의 `num_heads`와 값 96이 같아도 역할은 별개다.

즉 값과 층 분포만으로는 `d_nope`/`d_v`를 구별할 수 없지만, split 순서와 소비 경로를
포함하면 새 라벨의 경계가 소스와 맞는다. 936은 교정으로 승인해도 된다. 다만 3,720은
옛 trace에서 센 오염 축 수이고 936은 전환 검사의 집계 단위이므로 서로 같은 숫자일 필요는
없다.

관련 공식 소스:

- [Kimi-K3 configuration (resolved revision)](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/configuration_kimi_k3.py)
- [Kimi-K3 text model (resolved revision)](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/modeling_kimi_linear.py)

## Q2 — 372,948을 공개하고 출고할 수 있는가

**아니다. 이 상태에서는 공개를 조건으로도 출고하면 안 된다.**

현재 `transition_diff.py`는 먼저 `module_path`별로 op를 모은다. `_leftovers`는 그 모듈
전체에서 같은 signature를 상쇄하고 남은 레코드를 반환하며, 호출부는 그 전체 목록을 한 번에
`classify_unmatched`에 넘긴다. 따라서 출력의 117과 69는 연결된 “구간” 수가 아니라
미매칭 레코드가 존재한 **모듈 인스턴스 수**다. 서로 떨어진 계산들이 한 목록에 함께 들어갈
수 있다.

현재 잔여를 phase와 실제 layer type으로 다시 나누면 다음과 같다.

| phase / layer type | 모듈 수 | 모듈당 old + new 미매칭 | 레코드 수 | 특징 |
|---|---:|---:|---:|---|
| prefill KDA | 69 | 2,688 + 2,688 | 370,944 | bmm 384개와 permute/view 계열 |
| prefill MLA | 24 | 12 + 13 | 600 | permute/layout 계열 |
| decode KDA | 69 | 6 + 6 | 828 | permute/layout 계열 |
| decode MLA | 24 | 12 + 12 | 576 | permute/layout 계열 |
| **합계** | 186 module instances |  | **372,948** |  |

따라서 요청서의 `117 permute-only`는 KDA decode 69개와 MLA prefill/decode 48개의 합이다.
“전부 KDA self_attn”이라는 설명은 사실이 아니다. 두 attention 구현의 module attribute가
모두 `self_attn`이라 마지막 경로명만 보면 구별되지 않았던 것이다.

KDA prefill의 384개 bmm은 같은 einsum 식을 쓰는 두 위치에서 나온다. 첫 `A` 계산 64개와
`Aqk` 계산 320개다. 독립 float64 검사는 PyTorch가 그 einsum을 B=1과 B>1에서 서로 다른
bmm 방향으로 내려도 같은 값을 낸다는 **메커니즘 가설**을 강하게 지지한다. 그러나 현재
trace의 69개 모듈에서 실제 경계 입력, contraction 축, 결과 복원이 그 패턴과 일치하는지는
아직 검사하지 않았다.

B=4 독립 배치 검증도 이 간극을 메우지 않는다. B=3과 B=4는 모두 B>1 lowering 경로라서
새 라벨의 배치 일관성을 보일 뿐, B=1 trace와의 계산 동치를 보이지 않는다.

## Q3 — 필요한 최소 검증기

범용 tensor algebra 증명기를 만들 필요는 없다. 이번에 관측된 패턴만 fail-closed로 승인하는
작은 검증기면 충분하다. 다만 **DAG 경계 추적은 생략할 수 없다.** permutation 차원만 합성하면
shape가 같은 다른 입력을 사용하거나, 서로 무관한 permute들을 잘못 한 구간으로 묶어도
통과할 수 있다.

최소 구현은 다음 순서가 적절하다.

1. **ports schema 정규화**  
   옛 ports의 `[op_id, output_slot]` source와 새 ports의 source object를 하나의 내부 형식으로
   바꾼다. `input_sources`, input/output tensor IDs, output slot을 모두 보존한다.

2. **연결 성분과 실제 경계 구성**  
   모듈 전체 leftovers를 그대로 쓰지 않고 producer/consumer DAG로 연결 성분을 만든다.
   미매칭 노드 사이를 잇는 matched layout 노드가 있으면 필요한 범위까지 포함한다. 각 성분의
   외부 입력, 외부 출력, parameter/state/alias 효과를 찾는다.

3. **old/new 성분 대응**  
   module path와 shape만으로 짝짓지 않는다. 경계 producer/consumer, parameter identity,
   source/occurrence anchor와 출력 사용처로 대응시킨다. 모든 외부 출력이 한 번씩 대응해야 한다.

4. **layout canonicalization**  
   `view`, `_unsafe_view`, `reshape`, 안전한 `clone`은 원소 linear-index map으로, `permute`와
   `transpose`는 `scalar_args`의 차원 순열로 합성한다. numel, rank, stride/contiguity 전제와
   입력 tensor identity를 확인한다. 근거가 부족한 alias/storage 관계는 승인하지 않는다.

5. **contraction canonicalization**  
   bmm 양쪽을 경계 leaf tensor의 index 식으로 환원한다. batch/head 대응, 같은 두 입력,
   같은 reduction 축과 범위, 피연산자 swap+transpose, 출력 inverse permutation을 확인한다.
   in-place·상태 갱신·누락 출력이 없어야 한다.

6. **template fingerprint와 완전 coverage**  
   연결 성분을 정규화한 뒤 같은 proof template는 한 번 증명하고 모든 instance에 정확히
   적용한다. 현재 module-level fingerprint로는 아래 네 family가 예상되지만, 연결 성분을 만든
   뒤 실제 template 수를 확정해야 한다.

   - KDA prefill contraction/layout: 69 layers, 370,944 records
   - KDA decode layout: 69 layers, 828 records
   - MLA prefill layout: 24 layers, 600 records
   - MLA decode layout: 24 layers, 576 records

7. **증명 산출물과 fail-closed gate**  
   각 template의 old/new boundary anchors, canonical map, 적용 instance 수, 레코드 coverage를
   JSON으로 기록한다. 정확히 372,948건을 덮고 잔여가 0일 때만
   `contraction_lowering_verified` 또는 더 세분한 verified category로 이동한다. standalone
   float64 비교는 보조 증거로 붙인다.

117개 permute-only 모듈부터 처리하는 것은 좋은 구현 순서다. DAG 경계가 세워진 뒤에는 bmm
없이 순열 합성만으로 상당수를 해소할 수 있다. DAG 없이 `scalar_args`만 연속으로 합치는
축소판은 승인 근거로 충분하지 않다.

## Q4 — `UNKNOWNS.md` 문안

검증 전 상태를 내부 후보에 기록한다면 다음처럼 써야 한다. 이 문구는 정직한 보류 사유이며,
출고 예외 승인을 뜻하지 않는다.

> 옛 B=1 trace와 현재 B=3 trace의 전환 검사에는 미매칭 ATen op 레코드 372,948건이 남아
> 있다. 이 수는 독립적인 의미 변화 또는 연결된 구간의 수가 아니다. 현재 검사는
> `module_path`별 미매칭 레코드를 한꺼번에 분류하며, producer/consumer DAG로 연결 구간을
> 구성하지 않았다.
>
> 내역은 KDA prefill 370,944건, KDA decode 828건, MLA prefill 600건, MLA decode 576건이다.
> KDA prefill 차이는 batch 크기에 따라 같은 chunked einsum이 서로 다른 bmm 방향과 주변
> layout op로 lowering된다는 가설과 일치한다. 이 일반 메커니즘은 별도 float64 수치 검사로
> 확인했지만, 저장된 trace의 각 연결 성분에서 같은 경계 입력·수축 축·출력 원소 대응을
> 보존한다는 검증은 아직 수행하지 않았다. 나머지 2,004건은 KDA/MLA의 layout 차이다.
>
> 발행 B=3 라벨은 독립 B=4 검사에서 통과했다. 이 검사는 B>1 경로 안의 라벨 일관성을
> 확인하며 B=1 경로와의 계산 동치를 증명하지 않는다. trace 범위는 FakeTensor/CPU reference
> KDA와 MoE even-split 대체 경로이며 GPU Triton kernel의 op 구성은 검증 범위 밖이다.
> 이 후보는 trace-local lowering 동치 검증이 끝날 때까지 출고 승인 상태가 아니다.

현재 초안에서 고쳐야 할 핵심은 세 가지다.

- 372,948을 “각 구간”이라고 부르지 않는다. op 레코드 수다.
- 전부 KDA라고 쓰지 않는다. 2,004건은 KDA decode와 MLA prefill/decode layout 차이다.
- float64 검증이 저장된 trace 전체를 검증했다고 쓰지 않는다. 일반 lowering 메커니즘의 보조
  증거다.

검증기가 모두 통과한 뒤의 공개 문구는 다음처럼 바꿀 수 있다.

> 전환 검사에서 372,948개의 미매칭 op 레코드가 관측됐다. ports DAG와 scalar index map을
> 사용한 국소 검증기가 이를 연결 성분과 정규화된 lowering template로 분해했고, 모든 성분의
> 외부 입력·출력 원소 대응과 contraction 축 보존을 확인했다. 승인된 template, 적용 instance,
> 레코드 coverage와 proof artifact digest는 `[artifact]`에 기록했다. B=4 독립 라벨 검사도
> 통과했다. 검증 범위는 FakeTensor/CPU reference KDA와 명시된 MoE 대체 경로이며 GPU Triton
> kernel은 포함하지 않는다.

## Q5 — 네 모델 유지 또는 다섯 모델 동시 출고

**기존 네 모델은 유지하고 K3만 미룬다.** 네 모델의 출고를 K3와 다시 묶을 이유가 없고,
K3의 미검증 차이를 조건부 공개로 함께 넣으면 전체 산출물의 “verified” 의미가 약해진다.
K3는 위 verifier가 372,948건을 모두 덮고, 현재 commit으로 manifest와 공개 문서를 다시
만들고, `approved: true`가 된 뒤 다섯 번째 모델로 별도 출고하는 편이 신뢰도 면에서 낫다.

최종 출고 gate는 다음 네 조건이면 충분하다.

1. `semantic_change` 936의 MLA source/dataflow 근거를 audit에 기록한다.
2. 372,948건 전부를 DAG 기반 국소 proof가 덮고 미검증 잔여가 0이다.
3. 현재 snapshot에서 산술 검사, 독립 B=4 검사, `verify_all.py`가 통과한다.
4. 같은 snapshot의 manifest, summary, proof artifact, `UNKNOWNS.md`를 함께 생성하고
   manifest가 승인 상태다.
