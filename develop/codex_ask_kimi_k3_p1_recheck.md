# Kimi-K3 — round 5 의 P1 두 건이 제대로 반영됐는지만 봐 달라

요청일: 2026-09-20. 선행: `codex_answer_kimi_k3_final_artifacts.md` (round 5).
고정 revision `f831ab66814297da540d832a5235f8e904f29d06`.

**이번 요청은 범위가 좁다.** 전체 재검토가 아니라 **P1 두 건이 옳게 고쳐졌는가**, 그리고
**그 수정이 만든 새 문제가 있는가** 만 봐 달라. round 5 에서 이미 확인해 준 것(93층 구성,
두 대표 행, KDA↔MLA 스코프 누출 0, CSV/JSONL 일치)은 다시 볼 필요 없다.

---

## 1. P1-1 — decode KDA forget gate

### 당신이 짚은 것

`fla` 의 두 API 는 조건이 다른데 `src/kda_shim.py` 가 한 wrapper 를 둘에 똑같이 붙이고
`safe_gate` 를 함께 요구해서, decode 가 plain gate 로 추적됐다.

### 내가 확인한 것

```
chunk.py:394           if safe_gate and use_gate_in_kernel:          -> safe_gate 필요
fused_recurrent.py:29  USE_LOWER_BOUND = lower_bound is not None     -> safe_gate 인자 없음
fused_recurrent.py:160-170   USE_LOWER_BOUND -> lower_bound * sigmoid(exp(A) * g)
                             아니면            -> -exp(A) * softplus(g)
modeling_kimi_linear.py:610,623-624   chunk_kda(..., safe_gate=..., lower_bound=...)
modeling_kimi_linear.py:629,642       fused_recurrent_kda(..., lower_bound=...)   (safe_gate 없음)
```

### 고친 방법

`_wrap_kda` 에 `gate_needs_safe` 를 두고, `fused_recurrent_kda` 바인딩에만 `False` 를 준다.
게이트 선택 조건이 `lower_bound is not None and (safe_gate or not gate_needs_safe)` 가 된다.

당신이 준 재현 스크립트 그대로:

```
기준 (naive_kda_lowerbound_gate)  : -2.5
decode  recurrent  (고친 뒤)      : -2.5      <- 전에는 -0.6931471824645996
prefill chunk  safe_gate=False    : -0.693    <- chunk 은 이게 맞다 (규약 유지)
prefill chunk  safe_gate=True     : -2.5      <- 모델이 실제로 넘기는 값
```

산출물에서도 decode layer 0 `self_attn` 의 `aten.softplus.default` 가 **0 건**, `sigmoid`
3 건이다.

**Q1. 이 조건이 맞는가.** 특히 (a) `chunk` 쪽 규약(`safe_gate` 필요)을 그대로 둔 것이
맞는지, (b) `gate_needs_safe` 를 바인딩 시점에 고정한 것이 두 API 를 가르는 옳은 방법인지.
더 나은 방법이 있으면 알려 달라.

**Q2. decode 의 나머지 recurrent 경로가 지금은 소스와 맞는가.** gate 만 보고 고쳤으니,
`fused_recurrent_kda` 로 들어가는 다른 인자(`use_qk_l2norm_in_kernel`,
`use_beta_sigmoid_in_kernel`, `initial_state` 등)의 처리도 같은 종류의 어긋남이 없는지
봐 달라.

## 2. P1-2 — KDA 출력의 `depends_on`

### 당신이 짚은 것

`o_norm` 의 선행 노드에 같은 층의 attention contraction 이 하나도 없었다. 표만 그래프로
읽으면 V projection 과 output gate 만으로 attention 출력이 만들어지는 셈이었다.

### 원인

`o[:, i] = einsum(...)` 은 `select` 가 만든 **뷰**에 `copy_` 하는데, 뷰와 베이스는 서로
다른 텐서 객체다. `src/tracer.py` 가 뷰의 `physical_producer` 만 갱신하고 베이스는
`zeros_like` 에 머물러서, 뒤에서 베이스를 읽는 op 이 그 쓰기를 못 봤다.

### 고친 방법

제자리 쓰기(`출력 객체가 입력 객체와 같다`)일 때 `_base` 사슬을 따라 올라가며 베이스의
`physical_producer` / `tensor_uid` / `version` / `logical_source` 를 함께 갱신한다
(사슬 깊이 8 로 제한). `_base` 가 FakeTensor·meta 양쪽에서 동작하는 것은 확인했다.
원래 간선은 사라지지 않는다 -- 이 op 이 뷰(=`select` 출력)에 의존하고 그 `select` 가
옛 생산자에 의존하므로 사슬로 남는다.

결과:

```
o_norm 선행 노드 (layer 0)   prefill  7 -> 1,226      decode  7 -> 122
                              bmm 도달  아니오 -> 예     아니오 -> 예
```

**Q3. 이 방법이 과하거나 모자라지 않은가.** 특히 (a) 베이스의 생산자를 **덮어쓰는** 것이
맞는지(누적이 아니라), (b) 일부만 쓴 slice 의 경우에도 베이스 전체가 그 op 에 의존한다고
보는 것이 맞는지, (c) `_base` 사슬을 8 로 끊은 것이 실제 경로를 자를 위험이 있는지.

**Q4. 이 수정으로 다른 모델의 `depends_on` 이 잘못 이어지지 않는가.** 제자리 연산을 쓰는
모든 모델에 영향이 간다. 과연결(over-connection)로 없던 의존이 생기는 사례가 있을지 봐 달라.

## 3. 이 수정들이 만든 변화

### 3-1. 계산이 달라졌으므로 구간 동치 대조가 실패한다 (의도된 것)

decode 276 레코드(69 개 KDA 층)에서 값이 다르다(최대 차이 ~0.27). 옛 판이 틀린 gate 를
계산했으니 당연하다. 이걸 "미증명" 으로 세면 고칠수록 게이트가 막히므로,
`develop/verify/references.yaml` 에 **`computation_corrected`** 를 새로 두고 등재했다.

`transition_reviewed`(같은 계산인데 이름만 교체)와 구분한다. 도피처가 되지 않게
모델·phase·모듈 scope 에 더해 **예상 레코드 수(276)** 까지 요구하고, 전부 맞을 때만 해소한다.
증명 산출물(`full/lowering_proof.json`)에 모듈 경로를 실어 scope 를 수가 아니라 실제로 검사한다.

**Q5. 이 장치가 적절한가.** 특히 "계산을 고쳤다" 는 선언이 남용되지 않도록 더 요구해야 할
것이 있는가. 예상 레코드 수 외에 무엇을 고정해야 하는가.

### 3-2. `d_chunk` -> `64`, 4,278 축

tracer 수정이 의존 그래프를 바꿔 축 등가류가 달라졌고, `aten.slice.Tensor` 4,278 축이
`d_chunk` 대신 정수 64 로 물러났다.

* `d_chunk` 는 여전히 **997,000 축**(952,062 scope_inferred + 44,781 heuristic)에 붙어 있다
* 그 자리는 이미 공개한 값 충돌이다: `64 = d_chunk | d_rope`
* 정수는 거짓을 주장하지 않는다

그래서 되돌리지 않고 공개했다.

**Q6. 이 판단이 맞는가.** 아니면 4,278 축이 `d_chunk` 여야 할 근거가 소스에 있는가?
(`fla/ops/kda/naive.py` 의 chunk 길이 64 가 그 slice 들의 축인지 확인해 달라.)

## 4. round 5 의 P2 는 이렇게 고쳤다 (확인만)

다시 볼 필요는 없지만, 같은 지적이 또 나오지 않게 무엇을 했는지 적는다.

| round 5 지적 | 조치 |
|---|---|
| 범례가 값만 보고 이름 부여 (`5 = d_conv+1` 등) | `_known_composites` 가 scope 를 함께 돌려주고, 그 scope 가 실제 모듈과 맞을 때만 이름을 붙인다 |
| `known_limits` 의 elementwise 융합 주장 | 이 모델에서 실제로 융합됐을 때만 적는다 (K3 는 두 add 가 따로 있어 이제 안 적힌다) |
| `E_shared` 가 축이 아니라는 주장 / config 필드 없다는 주장 | 합성 라벨 안의 사용도 세고(각 phase 253 행), 거짓 주장은 삭제 |
| RoPE 가 "실행됐지만 선별에서 빠졌다" | "선별에서 빠졌을 수도, 실행되지 않았을 수도 있다 -- 표만으로는 못 가른다" 로 |
| `current` 를 미수정으로 셈 | 미해결 3 건과 판정 완료 4 건을 분리해 출력 |
| 배치 전환을 `table_omits_computation` 으로 분류 | `different_lowering_verified` 로 정정하고, 이미 끝낸 373,155 레코드 대조 결과를 반영 |
| `batch_transition_proof.json` 경로 | results 스냅샷은 모델 디렉터리 루트에 싣는다(`CARRY_EVIDENCE`) |

`d_head=74` 와 `n_grp`/`d_shared`/`layer_sched`/`chunk_size` 의 null 처리에 대한 당신의
판단은 아직 반영하지 않았다 -- 심볼 표의 의미 정리는 별도 작업으로 남겼다.

## 5. 재현

```powershell
.venv\Scripts\python.exe develop/test_lowering_proof.py
.venv\Scripts\python.exe develop/transition_release.py audit moonshotai__Kimi-K3 develop/models/phase25-kimi-k3.yaml
```

검토 대상: `models/moonshotai__Kimi-K3/` 와 `src/kda_shim.py`, `src/tracer.py`,
`develop/verify/references.yaml` 의 이번 변경.
기존 규칙·후보 파일은 수정하지 말아 달라.
