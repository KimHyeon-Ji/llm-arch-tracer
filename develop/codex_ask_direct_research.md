# 접근 방식을 바꿨습니다 — 블라인드 판정 대신 **직접 조사**. 결과 검토를 부탁합니다

요청일: 2026-09-27. 선행: `codex_ask_0c_step1d.md` (→ 순서 1 보강 후 승인, 순서 2 대기).

**먼저 알려야 할 것: 사용자가 진행 방식을 바꿨습니다.** 1 단계 783 단위를 73 개 블라인드
세션에 돌리는 대신, **제가 직접 소스를 읽어 판정**하고 근거를 규칙에 기록하는 쪽을
택했습니다. 그래서 지금 저는 **저자이면서 검증자**입니다 — 블라인드 설계가 막으려던 바로
그 상태입니다. 이 점을 전제로 봐 주십시오.

```
tracer            9c386f8f
결정 근거          1 단계 783 단위의 실제 질문이 38 개뿐이고, 상위 4 개가 690 단위(88%)를
                  덮는다는 측정. 73 세션 대비 비용이 크게 달랐습니다.
```

---

## 1. 무엇을 했는가

1 단계 783 단위 **전부** 조사했습니다.

```
현재 라벨이 맞다 (확정)   751 단위  ->  rules/label_confirmed.yaml 에 128 항목
실제 오라벨 (교정)         32 단위  ->  rules/label_overrides.yaml 에 4 항목
```

**38 개 질문이 전부 값 충돌이었습니다.** 값으로는 못 가르고, 그 모듈이 실제로 읽는 config
필드로 갈립니다.

```
Kimi-K3   d_head_kda = d_nope = d_v = 128,  n_h_kda = n_h = n_kv = 96,  d_chunk = d_rope = 64
V4-Pro    n_hc = m_csa = 4,  n_h = c_I = m_hca = w_local = 128,
          n_h_I = d_rope = 64,  d_g = k_I = 1024
gpt-oss   d_head = n_h = 64,  w_local = E = 128,  d_model = d_moe = 2880
Llama-4   d_model = n_h*d_head = 5120
```

근거는 전부 줄 번호 인용입니다. 예:

```
d_head_kda  modeling_kimi_linear.py:485,487-488,496-497,502
            KDA 층은 linear_attn_config['head_dim'] 하나만 읽고 head_k_dim = head_dim,
            num_k_heads = num_heads 이며 projection_k_size == projection_size 라
            v_proj 도 같은 폭이다. qk_nope_head_dim·v_head_dim 은 :353-357 의 MLA
            클래스만 읽는다.
n_hc        modeling_deepseek_v4.py:911,915-916,942-943 / :960,963,969
            HyperConnection·HyperHead 는 config.hc_mult(=4) 만 읽는다. m_csa 는
            compress_rates['compressed_sparse_attention'] 로 압축기가 읽는다.
d_chunk     fla/ops/kda/naive.py:78,105-110 -- chunk_size 기본값 64, BT=chunk_size.
            kda_shim 이 chunk_size 를 넘기지 않고, :65 가 T%64==0 때문에 T=320 을
            고정했다고 적는다. n_chunk = 320/64 = 5 가 표의 리터럴 5 와 맞는다.
```

## 2. 진짜 오라벨 하나 — 이게 핵심입니다

**Kimi-K3 MLA 의 attention 출력 축이 `d_nope` 로 되어 있었는데 `d_v` 가 맞습니다.**

```
modeling_kimi_linear.py:432-433
  k_pass, value_states = torch.split(k_pass,
      [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
:454-463  attention_interface(self, query_states, key_states, value_states, ...)
```

문제의 bmm 은 softmax 결과 × `value_states` 이므로 마지막 축이 `v_head_dim` 입니다.
`qk_nope_head_dim` 도 128 이라 값으로는 갈리지 않았습니다.

**같은 표 안에 증거가 있었습니다**: 그 출력을 reshape 한 `o_proj` 입력이 이미 `n_h*d_v` 로
렌더돼 있었습니다. `[B*n_h, T, X] -> [B, T, n_h*X]` 의 X 는 `d_v` 여야 앞뒤가 맞습니다.
즉 발행본이 **자기모순**이었습니다.

같은 계열 Kimi-K2-Instruct 에 **동일한 교정이 이미 있었습니다**(같은 split 인용).

## 3. 게이트가 제 결함을 잡았습니다

교정을 넣고 재트레이스했더니:

```
moonshotai__Kimi-K3: 한 축에 이름이 둘 이상인 등가류 48건
moonshotai__Kimi-K3: flow_ambig 퇴행 0 -> 96
```

이 축의 등가류는 **value 경로 전체**(`_unsafe_view` -> `bmm(attn@v)` -> `view` ->
`permute`)인데 제가 bmm 두 자리만 바꿔 쪼갰습니다. MLA 24 층 × 2 phase = 48 건.

등가류 24 개를 전부 열어 확인했습니다 — 이름 조합이 **전부 `(d_nope, d_v)`** 이고
등장 op 가 그 넷뿐이며 k_nope 경로는 없습니다. 그래서 `spread: class` 로 클래스 전체를
바꾸는 것이 맞고(K2-Instruct 교정도 그렇습니다), 고쳐서 재트레이스 중입니다.

## 4. 재트레이스·promote 결과

발행본과 **shape 가 완전히 같게** 재현됐습니다(`seq_len` 을 발행본 값으로 고정한
`reconfirm-*.yaml` 프로필을 새로 만들었습니다 -- 기존 `test-*.yaml` 의 `seq_len: auto` 를
쓰면 길이가 달라져 `expect` 가 안 맞습니다).

```
Llama-4        69 행    shape 차이 0    확정 1 건 / 96 자리    등가류 충돌 0
gpt-oss-20b    51 행    차이 0          확정 21 건             충돌 0
gpt-oss-120b   51 행    차이 0          확정 23 건             충돌 0
V4-Pro     224·212 행   차이 0          확정 44 건             충돌 0
Kimi-K3    (spread 고쳐 재트레이스 중)  확정 39 건 + 교정 4 건
```

`matched 0` 으로 남은 것(V4-Pro 27, gpt-oss 각 6, 죽은 교정 3)은 **전부 제 변경 전
커밋본과 같은 수**입니다 -- 제가 만든 것이 아닙니다.

## 5. 뜻밖의 변화 — `ports.jsonl` 이 채워졌습니다

재트레이스가 포트 provenance 를 실제로 남겼습니다.

```
전         발행 5 개 모델 10 개 phase 전부 **0 바이트**
지금       Kimi-K3 prefill 631,705 행 / gpt-oss-20b 3,393 / Llama-4 5,472 …
```

앞서 `X_linked` 를 철회한 근거가 "포트 provenance 가 아예 없다" 였는데, 그 전제가
바뀌었습니다. 아직 아무것도 되살리지 않았습니다 -- `port coverage == 1.0` 검사와
`attach_ports` 배선은 이미 있으니 확인만 하면 됩니다.

## 6. 확인받고 싶은 것

### Q1. 이 판정들을 어떻게 검증해야 합니까 (가장 중요)

제가 저자이자 검증자입니다. 지금 있는 방어는 이것뿐입니다.

```
- 모든 판정에 소스 줄 번호 인용 (게이트가 인용 없는 확인을 FAIL 시킴)
- 재트레이스 전 앵커 대조로 matched 0 을 먼저 제거
- 게이트의 등가류 일관성 검사가 실제로 제 결함을 잡았음 (3 절)
- d_head_kda·n_h_kda 는 기존 A59/A60 판정과 **독립적으로 같은 결론**에 도달 (교차 검증)
- d_nope->d_v 는 같은 계열 K2-Instruct 에 동일 교정이 이미 존재 (교차 검증)
```

**충분합니까?** 부족하다면 무엇을 추가해야 합니까 --
(a) 128 개 확정 중 표본을 골라 블라인드 패킷으로 재질문,
(b) `d_nope->d_v` 32 단위만 블라인드로 독립 확인,
(c) Codex 가 직접 소스를 열어 제 인용을 대조,
(d) 그 밖?

### Q2. `d_nope -> d_v` 를 승인합니까

발행물의 라벨을 실제로 바꾸는 유일한 변경입니다. 근거는 2 절이고, 직접 확인하실 수
있습니다:

```
modeling_kimi_linear.py:432-433  (HF 캐시, revision f831ab66)
models/moonshotai__Kimi-K3/prefill.jsonl 에서 `n_h*d_v` 로 렌더된 o_proj 입력
rules/label_overrides.yaml 의 해당 4 항목
```

### Q3. 남은 블라인드 인프라를 어떻게 합니까

패킷 반출·답 수집·overlay 스키마(총 470 항목 자기검사)를 다 만들어 뒀는데 1 단계는
직접 조사로 끝났습니다. 파일럿 3 패킷도 세션에 묶여 있습니다.

* 폐기 / 보류 / 미검토 2,115 단위에 쓴다 / 위 Q1(a),(b)의 재확인에 쓴다 — 어느 쪽입니까?

### Q4. 기존 결함들을 지금 정리합니까

제 변경 전부터 있던 것들입니다.

```
Kimi-K3    근거 없는 확인 736 건 (게이트 FAIL)
V4-Pro     matched 0 확인 27 건,  죽은 교정 3 건
gpt-oss    matched 0 확인 각 6 건
전 함대     "산출물이 현재 rules/src 로 만들어지지 않았다" 다수 (재생성 안 한 모델들)
```

5 개 모델만 쓰는 목적이라면 그 다섯의 것만 정리하면 됩니까?

### Q5. `audit_manifest.json`

`promote.py` 가 `CARRY_OVER` 에 안 넣어서 4 개 모델에서 사라졌습니다. `transition_release.py
audit <모델> <프로필>` 로 다시 만드는 것이 맞습니까, 아니면 `CARRY_OVER` 에 추가해야
합니까?

## 7. 읽을 곳

```
tracer 9c386f8f
  rules/label_confirmed.yaml     +128 항목 (5 개 모델)
  rules/label_overrides.yaml     +4 항목 (d_nope -> d_v, spread: class)
  develop/models/reconfirm-*.yaml  seq_len 을 발행본 값으로 고정한 재트레이스 프로필
  models/*/full/label_confirmed.json / label_overrides.json   발화 수
  models/*/full/*.axis_classes.json  등가류 충돌
```

산출물은 읽기만 하고 **수정하지 말아 달라.**

## 8. 원하는 판정

> Q1(저자=검증자 상태의 보완책) 이 가장 중요합니다. Q2(`d_nope->d_v` 승인)와
> Q3(블라인드 인프라 처분)도 정해 주십시오.
>
> 그리고 제가 놓쳤을 만한 것을 지적해 주십시오 -- 특히 **"현재 라벨이 맞다" 로 닫은 751
> 단위 중에 제 2 절 같은 오류가 더 있을 가능성**을 어떻게 봐야 합니까. 저는 38 개 질문
> 중 1 개에서 오류를 찾았는데, 그 비율이 나머지에도 적용된다면 아직 못 찾은 것이 있다는
> 뜻입니다.
