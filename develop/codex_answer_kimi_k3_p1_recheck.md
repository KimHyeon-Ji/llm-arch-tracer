검토일: 2026-09-20. 대상: `develop/codex_ask_kimi_k3_p1_recheck.md`.
코드 HEAD: `d52f515561b1f304eca52f3767f2499eb118e9e8`. 산출물은 요청한 `models/moonshotai__Kimi-K3/`의 작업 트리 판이다. 모델 revision은 `f831ab66814297da540d832a5235f8e904f29d06`.

**K3 실제 호출에서 forget gate 수정은 확인했다. KDA 결과가 `o_norm`에 도달하는 것도 복구됐다. 다만 두 P1을 모두 닫는 데에는 동의하지 않는다.** 새 tracer가 전체 베이스의 출처를 slice 출력 포트로 바꾸는 회귀가 있고, `computation_corrected`는 현재 선언 범위 밖의 실패도 해소한다. Q2에서 요청한 나머지 인자 대조에서는 Q/K 정규화 수식 차이도 확인했다.

기존 규칙·구현·후보·모델 파일은 수정하지 않았다. 이번 범위 밖의 93층 구성, CSV/JSONL 동등성, 기존 P2 항목은 다시 검토하지 않았다. 재현 코드는 `develop/repro_kimi_k3_p1_recheck.py`, 관측값은 `develop/kimi_k3_p1_recheck_evidence.json`, 4,278축의 별도 대조는 `develop/kimi_k3_p1_slice_evidence.json`에 남겼다.

아래 FLA 파일 경로의 기준은 `.venv/Lib/site-packages/fla/`다. 모델 소스 M은 `C:/Users/99ktx/.cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/f831ab66814297da540d832a5235f8e904f29d06/modeling_kimi_linear.py`다.

| 질문 | 판단 |
|---|---|
| Q1 gate 조건·바인딩 | **현재 K3 prefill/decode 호출은 수정됨.** 다만 “chunk는 safe_gate=False이면 plain gate”라는 일반 규약 해석은 설치된 하위 구현과 다르다. |
| Q2 다른 recurrent 인자 | beta·scale·native-layout initial state 전달은 확인. **Q/K L2 정규화 수식은 아직 다름.** state-layout 호환성도 구분해서 설명해야 한다. |
| Q3 베이스 생산자 갱신 | 쓰기 의존성을 보존하려는 방향은 맞다. **작은 view 출력 포트를 전체 base 값의 출처로 덮어쓰는 방식은 틀림.** |
| Q4 다른 모델 영향 | meta와 FakeTensor 모두에서 잘못된 포트와 이전 쓰기 누락을 재현했다. 현재 방식의 일반적 안전성은 승인할 수 없다. |
| Q5 계산 수정 예외 | 별도 범주는 적절하다. **현재 범위 검사·개수 제한·실패 처리는 충분하지 않음.** |
| Q6 4,278축 정수화 | 값 자체를 64로 남기는 것은 보수적 표시지만, **이 축들은 소스로 d_chunk임을 알 수 있다.** 새 포트 오류도 같은 4,278곳에 있으므로 단순한 이름 철회로 종결하면 안 된다. |

**Q1 — K3의 gate 수정은 맞다. chunk 일반 규약에 대한 근거는 정정해야 한다.**

현재 wrapper에 설치된 `naive_kda_gate`와 `naive_kda_lowerbound_gate`를 연결해 다시 확인했다. `g=A_log=dt_bias=0`, `lower_bound=-5`에서:

| 호출 | 현재 wrapper 결과 |
|---|---:|
| recurrent, `gate_needs_safe=False`, safe_gate 생략 | -2.5 |
| chunk, `gate_needs_safe=True`, `safe_gate=True` | -2.5 |
| chunk, `gate_needs_safe=True`, `safe_gate=False` | -0.6931471824645996 |

K3는 실제로 앞의 두 경우를 사용하므로 원래 지적한 plain-gate 오류는 수정됐다. 새 decode 원시 trace의 layer 0 `self_attn` 하위에서 softplus는 0개, sigmoid는 3개였다. 각각 raw `op_id=149`의 forget gate, `152`의 beta, `270`의 `o_norm` output gate다.

그러나 **표의 세 번째 결과를 “chunk API도 그러므로 올바르다”고 확인해 줄 수는 없다.** `ops/kda/chunk.py:394–398`은 safe gate의 입력값 검증이다. 이것만 보고 실제 gate 수식 선택 조건을 판단하면 안 된다.

설치된 구현의 실제 전달 경로는 다음과 같다.

- `chunk.py:77–95`, `:416–435`: safe_gate와 lower_bound를 각각 다음 함수로 넘긴다.
- `chunk_fwd.py:45–55`: `use_gate_in_kernel`이면 `kda_gate_chunk_cumsum(..., lower_bound=lower_bound)`를 호출한다. 여기에는 safe_gate 검사가 없다.
- `gate.py:359–363`: `USE_LOWER_BOUND = lower_bound is not None`.
- `gate.py:417–422`: 이 조건에 따라 bounded sigmoid를 고른다.

재현 코드에서는 설치된 `chunk_kda_fwd` 함수 본문을 추출해 `safe_gate=False, lower_bound=-5`로 호출하고, Triton 실행 직전 gate 호출을 가로챘다. `lower_bound=-5`가 그대로 전달되고 gate 함수에는 safe_gate 인자가 전달되지 않았다. 실제 수식 분기는 위 kernel 소스로 확인했다. GPU kernel 자체를 실행한 시험은 아니다.

따라서 **바인딩 시점에 API 차이를 고정하는 설계 자체는 괜찮지만, 이 설치 판의 수식 선택을 가르는 축으로 `gate_needs_safe`를 사용하는 것은 정확하지 않다.** safe_gate의 범위 검증·chunk 구현 선택과 gate activation 선택을 분리해야 한다. K3의 현재 prefill 호출은 safe_gate=True여서 이 일반 경우의 불일치에는 걸리지 않는다. 이번 recurrent 수정으로 새로 생긴 오류라고 세는 것도 아니다.

**Q2 — [P1] Q/K 정규화가 아직 다른 계산이다. 이번 수정 이전부터 있던 잔여 문제다.**

`src/kda_shim.py:270–272`는 `F.normalize(q, dim=-1, p=2)`를 사용한다. 이것은 기본적으로 `q / max(norm(q), 1e-12)`다. 실제 recurrent kernel은 `ops/kda/fused_recurrent.py:150–156`에서 float32로 읽은 뒤 **`q / sqrt(sum(q*q) + 1e-6)`**를 쓴다. epsilon을 norm의 하한으로 놓는 것과 제곱합에 더하는 것은 다른 함수다. `F.normalize(..., eps=1e-6)`로 숫자만 바꿔도 해결되지 않는다.

K3처럼 head 폭 128, 모든 q/k 원소를 `1e-4`로 둔 CPU 재현 결과:

| 값 | 현재 wrapper | kernel 수식에 맞춘 전처리 |
|---|---:|---:|
| 정규화된 원소 | 0.0883883535861969 | 0.06622661650180817 |
| 같은 naive recurrent에 넣은 출력 원소 | 0.04419418424367905 | 0.024810772389173508 |

두 번째 행은 `v=1`, gate logits/A_log/dt_bias/beta logits=0, lower_bound=-5, initial_state=None, beta sigmoid 활성화라는 같은 조건에서 비교했다. recurrence는 양쪽 모두 설치된 `naive_recurrent_kda`를 사용했고 Q/K 전처리만 달리했다. kernel과의 bitwise 비교를 주장하는 시험이 아니라 **전처리 수식 자체의 차이**를 분리한 시험이다.

chunk 쪽도 `ops/kda/chunk.py:54–58`에서 `l2norm_fwd`를 쓰고, `modules/l2norm.py:43`, `:150`의 수식·기본 eps가 같은 `sqrt(sum(x*x)+1e-6)`이므로 함께 확인해야 한다. M:639가 decode에서 이 인자를 True로 넘긴다. 이 문제는 Q2가 요청한 “다른 인자에도 같은 종류의 어긋남이 있는가”에 해당한다.

다른 인자는 다음 범위까지 확인했다.

- `use_beta_sigmoid_in_kernel`: K3는 float beta와 기본 `allow_neg_eigval=False`를 사용한다. wrapper의 sigmoid는 recurrent `:187–190`의 해당 분기와 맞다.
- `scale`, `initial_state`, `output_final_state`: `src/kda_shim.py:296–297`에서 그대로 전달된다. `naive.py:48–66`은 기본 scale `K**-0.5`, 초기 state 누적, final-state 반환 여부를 실제로 처리한다.
- `transpose_state_layout=True`는 여전히 `**_kw`에서 버린다. 실제 recurrent는 `:441–449`에서 이를 `state_v_first`로 바꾸고, `:275–278`에서 `[B,H,V,K]` state를 반환한다. naive는 `naive.py:55`의 `[B,H,K,V]`를 쓴다. **prefill과 decode를 모두 naive로 수행하는 닫힌 경로는 이 자체로 값 오류라고 단정하지 않는다.** 하지만 GPU/API의 V-first state를 그대로 입력받아도 호환된다는 주장은 할 수 없다. K=V=128이라 shape가 같아도 비대칭 state의 원소 배치는 다르다. native CPU state 형식임을 명시하거나 입출력 전치를 처리해야 한다.
- 현재 K3의 unpacked trace에서 `cu_seqlens=None`인 경로는 이번 수정으로 새 문제를 발견하지 않았다. varlen·다른 모델의 추가 kwargs까지 지원한다고 검증한 것은 아니다.

**Q3·Q4 — [P1] 도달성은 복구됐지만 정확한 출력 포트 계약이 깨졌다.**

먼저 수정된 점은 확인했다. 양 phase 각각 KDA `o_norm` 대표 행 16개, repeat를 펼친 69개 층 모두에서 같은 층 attention bmm에 도달했다. major 표의 첫 층 기준 선행 노드 수는 자신을 포함해 prefill 54개, decode 21개였다. 요청서의 1,226/122와는 집계 대상이 다르므로 숫자를 직접 비교하지 않았다.

문제는 `src/tracer.py:200`, `:208`이다. view를 쓰는 op의 출력은 그 **view**인데, 동일한 `(op_id, output_slot)`을 전체 **base**의 값 출처에도 넣는다. 의존 관계가 있다는 사실과 같은 텐서 값이라는 사실을 혼동한다.

최소 예:

```python
x = torch.zeros(2, 3)
a = x[0]
a.copy_(torch.ones(3))
y = x.clone()
```

`y` 입력은 `[2,3]`인데 기록된 `input_sources`는 `[3]`을 반환한 `copy_`의 0번 출력이다. “전체 x가 그 쓰기의 영향을 받는다”는 depends_on 간선은 타당하지만, “전체 x가 그 slice 출력과 같은 값이다”라는 포트는 타당하지 않다. meta와 FakeTensor 양쪽에서 재현됐다.

현재 K3 파일에도 같은 오류가 있다:

| 파일 / op_id | 실제 기록 |
|---|---|
| `full/prefill.trace.raw.jsonl`, 2616 | `copy_`의 출력 `[B,n_h_kda,5,1]`, rank 4 |
| 같은 파일, 2633 | slice 입력 `[B,n_h_kda,5,d_chunk,64]`, rank 5 |
| `full/prefill.ports.jsonl`, 2633 | input_sources가 **2616의 출력 0번**을 가리킴 |

tensor ID도 다르다. 2633의 실제 입력은 base `1284`, 2616의 출력은 view `1297`이다. 정확한 value-port 연결로 해석하는 축 계보 소비자에게는 잘못된 정보다. 이것은 새 수정으로 생긴 회귀다.

또 “덮어써도 이전 생산자가 select 사슬에 항상 남는다”는 주장은 **뷰를 쓰기 직전에 새로 만드는 경우에만** 성립한다. 다음처럼 뷰를 미리 만들면 깨진다:

```python
x = torch.zeros(2, 3)
a, b = x[0], x[1]
p, q = torch.ones(3), torch.full((3,), 2.)
a.copy_(p)
b.copy_(q)
y = x.clone()
```

meta 재현에서 write op은 5와 6인데 최종 clone의 선행 노드는 `[0,2,4,6,7]`뿐이다. 첫 번째 write 5와 그 입력 p의 생산자 3이 사라진다. FakeTensor에서도 같은 누락이다. 마지막 b의 select는 첫 write 이전 x를 참조하기 때문이다. 이 누락 유형은 기존 구현에도 있었으며 **이번 보완이 아직 해결하지 못한 경우**다. `x.add_(1)` 뒤에 미리 만든 `a`를 읽는 방향도 누락된다.

질문의 세부 답은 다음과 같다.

- **(a) 덮어쓰기:** “새로운 storage/base version”을 세우는 것은 가능하다. 다만 이전 base version과 부분 쓰기를 합친 상태여야 한다. slice 출력 포트로 대체하면 안 된다. 미리 만들어 둔 alias를 읽을 때도 현재 version을 반영해야 한다.
- **(b) 부분 쓰기:** base 전체를 읽는 op이 그 쓰기에 의존하는 것은 맞다. 반면 서로 겹치지 않는 view 읽기까지 데이터 의존으로 연결하면 보수적 과연결이다. storage의 쓰기 순서 의존과 정확한 tensor 값의 출처를 분리하면 이 차이를 표현할 수 있다.
- **(c) 깊이 8:** 이번에 재현된 오류 원인은 깊이 제한이 아니다. 표준 view는 `_base`가 루트로 연결되는 경우가 많다. 그래도 한도에 도달했을 때 조용히 잘라 버리는 것은 완전성 보장이 아니다. 방문 집합으로 순환을 막거나, 잘린 경우 계보 미해결로 표시하는 편이 낫다. 깊이를 단순히 늘려도 앞의 두 오류는 해결되지 않는다.

권장 방향은 base/storage version과 mutation 관계를 별도로 기록하고, 필요한 경우 전체 base 값을 나타내는 명시적인 갱신 노드/포트를 만드는 것이다. `depends_on`에 쓰기 의존성을 추가하는 것과 `input_sources`를 동일 값 포트로 연결하는 일을 분리해야 한다. 다른 모델을 전량 재실행해야만 발견되는 문제가 아니라 위 작은 예제로도 계약 위반을 확인할 수 있다.

**Q5 — [P1] `computation_corrected`의 취지는 맞지만, 현재 구현은 범위를 고정하지 못한다.**

옛 계산이 틀렸으므로 새 계산과 수치가 다른 것을 별도 사유로 허용하는 것은 타당하다. 하지만 현 구현을 “모델·phase·실제 전체 scope·총 276건을 확인한다”고 설명하면 안 된다.

| 합성 증명 파일 / 감사 상황 | 실제 결과 |
|---|---|
| 276건짜리 서로 다른 template 두 개, 선언은 한 개의 276건 | `_corrections_cover`가 **552건 전부 해소** |
| 대표 4개 경로는 self_attn, 생략된 모듈에 mlp가 있음(`module_leaves`에 기록) | **276건 해소** |
| 실패 원인이 값 불일치가 아니라 지원하지 않는 op여서 재실행하지 못함 | **276건 해소** |
| 이전 proof JSON이 남은 상태에서 이번 proof 프로세스가 exit 2로 실패 | `audit()`가 **approved=true**, return 0 |

모두 원본 파일을 바꾸지 않고 임시 디렉터리에서 실제 `_corrections_cover`/`audit`를 호출해 재현했다. 마지막 경우만 외부 검증 프로세스 실행을 mock 처리했으며, 감사·예외 해소·manifest 생성 로직은 실제 함수를 사용했다.

원인:

- `develop/lowering_proof.py:588`은 modules를 **첫 4개만** 남긴다. `transition_release.py:127`은 그것만 검사하고 `module_leaves`조차 보지 않는다. 전체 module scope를 검사했다고 할 수 없다. `scope: self_attn` 자체도 KDA 69층과 MLA 24층을 구별하지 못한다.
- `transition_release.py:119–136`은 예상 레코드 수를 **template마다** 비교하고 같은 선언을 여러 번 재사용한다. 선언당 총량 제한이 아니다.
- 실패 이유·old/new op 구조·scalar_args·연결 구조를 검사하지 않는다. 동일 scope·개수의 다른 계산 변경이나 재실행 실패도 같은 예외에 들어갈 수 있다.
- `:215`는 예전의 `not rc_p` 조건을 제거했다. 정상적인 “예상된 불일치”와 프로세스 오류를 구별하지 않아 남은 JSON을 근거로 승인할 수 있다.

최소한 다음을 고정하는 편이 맞다.

1. 모델 config revision과 구현/FLA 판, **이전·새 trace의 digest**, phase, backend·batch·sequence 조건. 일회성 전환 승인이 이후 임의 변경에 재사용되지 않게 한다.
2. 69개 **전체 KDA module 경로** 및 대응 component/instance 집합. 대표 몇 개와 leaf 이름만으로 대체하지 않는다.
3. 예상 old→new 계산 변화의 지문. op 이름 다중집합뿐 아니라 scalar 값(-5), 입력·출력 경계, 내부 연결, dtype/shape 조건도 포함한다. 단순한 `276`은 부가 검산값이다.
4. 선언당 정확한 전체 합계 276과 69/69 적용 여부, 중복 적용 금지. 실패를 해소하지 않은 나머지 영역은 기존대로 검사한다.
5. 새 계산을 독립적인 올바른 식과 대조한 결과. “옛것과 다르다”는 관측만으로 “새것이 옳다”가 되지 않는다.
6. 새 proof의 생성 성공·입력 digest 확인과 오류 종류의 구조화. 예상된 value mismatch만 해당 예외에 넣고, 실행 오류·미지원 op·불완전 증거는 계속 보류한다.

현재 `audit_manifest.json`은 276건 모두 replay 불일치인데 `claim="lowering_replay_consistent"`로 끝난다. **이 명칭도 틀렸다.** `numerical_mismatch=276`, `reviewed_computation_correction=276`, `unreviewed=0`처럼 사실을 나눠 보존해야 한다. 전환을 허용하는 것과 동치였다고 주장하는 것은 구별해야 한다.

**Q6 — 4,278축의 의미는 d_chunk다. 공개된 충돌로만 처리하면 안 된다.**

이전 상태 보관 archive의 B=3 후보 원시 trace와 현재 prefill 원시 trace를 대조했다. op_id·module·raw_op가 같은 slice에서 `d_chunk→64`가 된 자리를 직접 셌으며 정확히 다음과 같다.

- **4,278 = 69층 × 62자리**, 전부 `input_shape[0][4]`.
- `aten.slice.Tensor`의 scalar 인자는 전부 `dim=4, start=0, end=2..63`.
- 첫 예: `model.layers.0.self_attn`, raw `op_id=2633`, 입력 `[B,n_h_kda,5,d_chunk,64]`, 출력 `[B,n_h_kda,5,d_chunk,2]`.
- 같은 자리가 전부 앞서 설명한 **rank 4 producer 출력 → rank 5 consumer 입력**의 잘못된 포트 연결을 갖고 있다.

소스 `ops/kda/naive.py:108–126`에서 `BT=chunk_size=64`, `g`는 `[B,H,NT,BT,K]`, `A=zeros(*g.shape[:-1],BT)`이므로 A는 **`[B,H,NT,BT,BT]`**다. `:134–135`의 반복문에서 `A[..., :, :i]`를 자를 때 마지막 **입력** 축은 전체 BT, 출력 축은 prefix 길이 i다. 이번에 바뀐 것은 바로 그 마지막 입력 BT 축이다. **d_rope와는 무관하다.**

따라서 입력 64의 정답은 d_chunk이고 출력의 2..63은 루프 길이로 정수 유지가 맞다. 64를 임시로 남기는 것이 수치 오류는 아니지만, “소스로 못 가르는 값 충돌이라 이름을 철회했다”거나 `fabrication_withdrawn`이라는 분류가 의미상 정확하다고 승인할 수는 없다. 이번 잘못된 포트 연결을 먼저 고친 뒤 축 계보를 다시 확인하는 것이 순서다. 라벨만 강제로 d_chunk로 덮어서 포트 문제를 숨겨서는 안 된다.

요청서의 합계에도 작은 산술 오류가 있다. `952,062 + 44,781 = 996,843`이며 정확히 997,000은 아니다. 반올림 수치라면 약 99.7만으로 표시하면 된다.

검증 범위: `develop/test_lowering_proof.py` 6/6 통과, 기존 `develop/test_tracer_version.py` 통과. 추가로 meta/FakeTensor alias 예제, 실제 torch reference의 gate·정규화 수치 대조, 설치된 chunk forwarding 경로, 임시 감사 예외 반례를 실행했다. **기존 테스트가 통과해도 위 반례들은 남는다.** 실제 후보에 대한 audit는 proof/manifest를 덮어쓰므로 이번 읽기 전용 검토에서는 실행하지 않았다.

재현 명령:

```powershell
.venv\Scripts\python.exe -X utf8 develop/repro_kimi_k3_p1_recheck.py
```

현 단계 판단은 **gate의 원래 오류는 해결, KDA 도달성도 개선, tracer 포트와 계산 수정 예외는 추가 수정 필요**다. Q/K 정규화는 이번 변경이 만든 회귀와 구분해서 남은 계산 불일치로 처리해야 한다.
