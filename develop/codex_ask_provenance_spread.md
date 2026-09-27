# 회귀를 닫으려고 **기계장치를 하나 늘렸습니다.** 이게 정당한 변경인지 봐 주십시오

요청일: 2026-09-27. 선행: `codex_ask_direct_research.md` (→ Q1 전건 독립 검토 지시, Q2 `d_nope -> d_v`
의미 판정 승인, Q4 5 개 모델의 기존 죽은 규칙 정리 지시).

이번 라운드는 **두 가지**입니다.

```
(A) Q4 이행       출고 5 개 모델의 죽은 규칙 42 건 전수 처리
(B) 회귀 종결     reshape_incons 0 -> 24 를 닫으려고 src 에 옵트인 기계장치를 추가
```

**(B) 가 검토의 핵심입니다.** 사용자의 상시 지시는 "모델별 코드 변경 금지, 모델 지식은
`rules/` 로" 인데, 저는 `src/label_overrides.py` 를 고쳤습니다. 일반 기계장치이고 모델 지식은
규칙에 그대로 남지만, 판단은 받아야 한다고 봅니다.

---

## 1. (A) 죽은 규칙 42 건 -- **라벨이 틀린 것은 하나도 없었습니다**

전수 조사했습니다. 원인이 네 가지였고, 전부 앵커 문제였습니다.

```
배치 접힘   16 건   [n_h, T, T] -> [B*n_h, T, T]. 축 0 의 이름 자체가 B*n_h 가 됐으니
                    "이 축은 n_h" 라는 확정을 되살릴 자리가 없다          -> 은퇴
자리 소멸    3 건   그 shape 이 트레이스에 더는 없다                      -> 은퇴
중복         4 건   nth 를 떼자 선택자가 동일해졌다                        -> 은퇴
서수 이동 /
shape 노후 /
피연산자 번호 20 건  자리도 라벨도 그대로다                                -> 재앵커
죽은 교정     3 건   g_o -> g_o (from == to, 발화 불가) + 리졸버가 이미 그 이름을 낸다 2 건 -> 은퇴
```

은퇴는 삭제가 아니라 **주석 처리**입니다 -- 이 파일의 주석이 근거 기록이라서요.

**피연산자 번호 건은 실질적 발견입니다.** V4-Pro indexer 의
`concat i=[[B,n_h_I,T,c_I-d_rope], [B,n_h_I,T,d_rope]] -> [B,n_h_I,T,c_I]` 에서 `d_rope`
조각은 **둘째** 피연산자인데 확정 4 건이 `shape_index: 0`(= `c_I-d_rope`, nope 조각)을
가리키고 있었습니다. 값이 둘 다 64 라 안 갈렸습니다.

검증: 재앵커한 20 건이 재트레이스에서 전부 발화했습니다(죽은 확정 0).

## 2. (B) 회귀의 진짜 원인 -- `spread: class` 로는 못 닫습니다

`src/label_overrides.py:304` 가 `axis_classes.build(rows, conc)` 를 **mode 없이** 부릅니다.
`DEFAULT_MODE = "legacy"` 이므로 값 간선입니다. Kimi-K3 MLA 의 value 사슬은

```
split_with_sizes -> unsqueeze -> permute -> permute -> clone -> _unsafe_view -> bmm(attn, v)
```

이고 `d_nope`/`d_v` 가 둘 다 128 이라 값으로는 영원히 안 갈립니다. 실측:

```
legacy      클래스 3 자리(prefill) / 6 자리(decode)
provenance  클래스 10 자리 / 11 자리 -- 그 안에 split 의 **출력 1**(value_states, 이미 d_v)이
            앵커로 들어 있고, 출력 0(k_nope)은 안 들어온다
```

발화 수가 그대로 말합니다: 24 층 × 3 자리 = 72, 24 × 6 = 144. 앵커를 어디로 옮겨도
legacy 클래스가 3 자리면 3 자리만 바뀝니다.

## 3. 무엇을 넣었는가 -- `spread: provenance_class`

옵트인 값 하나입니다. 절제 세 가지를 걸었습니다.

```
(a) 기존 `spread: class` 동작은 **안 건드렸다.** 함대 30 개의 등가류가 한꺼번에 커지는
    변화는 따로 측정할 일이다.
(b) **폴백 없다.** 포트 기록이 한 행이라도 없으면 ValueError. 포트가 하나도 없으면 build 는
    조용히 단일 원소 클래스만 내놓고, 그건 "퍼뜨릴 곳이 없다" 와 구별되지 않는다.
(c) 알 수 없는 `spread` 값을 거부한다. 예전엔 오타가 조용히 무시돼 "퍼뜨리지 않는 교정" 으로
    통과했고, 발화 수는 앵커 한 자리만 세니 게이트도 넘어갔다.
```

**재트레이스 전에 실측 대조했습니다.** footprint 의 `changed` 를 `from` 으로 되돌려 교정 전
렌더를 복원하고, 같은 조건에서 두 모드를 돌렸습니다.

```
                 남은 d_nope (prefill / decode)                        발화 0 인 교정
class            {permute 48, clone 24, _unsafe_view 24} / {permute 48}   6 / 30
provenance_class {} / {}                                                   6 / 30
```

사슬이 닫히고 **다른 교정을 하나도 가리지 않습니다**(죽는 교정 수가 양쪽 같습니다).

## 4. 제가 잡은 제 결함 셋

```
1. 두 경로의 포트 표현이 다르다. 트레이스 시점의 행은 noderef.encode() 결과(dict)를 들고
   있고(build_table.write_ports 가 그대로 json 으로 쓴다), 발행본 재독 경로
   (axis_classes.attach_ports)는 decode_list 로 디코드해서 붙인다. lineage_edges 는 디코드된
   쪽만 읽는다. K3 재트레이스가 13 분 뒤 'dict' object has no attribute 'node' 로 죽었다.
   -> 정규화하고 그 사례를 자기검사에 넣었다(v2 인코딩에 raw null 은 없다 --
      미상 입력은 unknown_external 노드다).
2. 내 탐침이 조용히 0 을 냈다. reshape_disagreements 에 op_type 없는 concrete 레코드를
   넘겨 항상 [] 였다. 게이트는 심볼 행을 복사해 shape 만 덮는다(verify_all.py:426-428).
   발행본이 24 건인 것을 알고 있었기에 0/0 을 의심할 수 있었다.
3. replay 하네스가 footprint id 에 spread 가 들어가는 것을 놓쳐 flip 을 복원하지 못했고,
   "교정이 죽었다" 로 보였다. 8826 - 8754 = 72 가 정확히 그 발화 수라서 드러났다. 두 번 걸렸다.
```

## 5. 자기검사

새 파일 `develop/test_spread_provenance.py` -- 모델을 안 띄우는 fixture 8 사례.

```
포트가 없으면 거부한다 (조용한 폴백 금지)
트레이스 시점 인코딩을 받는다 (위 결함 1)
legacy 가 못 닿는 자리에 닿는다   -- legacy 가 이미 닿으면 **실패한다**(fixture 가 두 모드를
                                    가르지 못하면 이 시험은 무의미하다)
계보가 다른 텐서(k_nope)는 폭이 같아도 안 건드린다
value 쪽 자리는 전부 바뀐다
발화 수는 실제로 쓴 자리 수다 (앵커만 세지 않는다)
알 수 없는 spread 를 거부한다
`spread: class` 의 동작은 그대로다
```

기존 19 개 시험 파일도 돌렸습니다. `test_bundle_guards.py` 가 4 건 실패했는데 **제 변경 탓이
아니었습니다** -- `openai__gpt-oss-20b` 을 "포트가 빈 모델" 로 하드코딩해 뒀고, 이번 라운드
재트레이스로 그 모델 포트가 채워지면서 전제가 무너진 것입니다. 가드는 살아 있었습니다.
시험이 빈 포트 모델을 **직접 만들게** 고쳐 63/63 이 됐습니다.

## 6. 확인받고 싶은 것

### Q1. (B) 가 정당한 변경입니까

상시 지시는 "모델별 코드 변경 금지" 입니다. 제 판단은 "이건 모델별이 아니라 일반 기계장치이고,
모델 지식은 규칙에 그대로 남는다" 인데, 대안은 취약한 shape 앵커를 7 개 더 쌓는 것이었고
그건 라운드마다 앵커가 밀립니다([[shape-anchor-goes-stale]]).

**반대 의견이 있으면 듣고 되돌리겠습니다.**

### Q2. `spread: class` 를 provenance 로 **전환**해야 합니까

지금은 옵트인이라 규칙 두 줄만 새 경로를 씁니다. 함대 전체를 provenance 로 바꾸면 등가류가
커지면서 30 개 모델의 교정 도달 범위가 한꺼번에 달라집니다. 측정 없이는 안 한다는 판단인데,
측정을 해야 한다고 보십니까 -- 그렇다면 무엇을 세야 합니까(등가류 성장량 / 역할 혼재 /
판정 발화 자리 집합의 전후 차이)?

### Q3. 포트 커버리지를 어디까지 믿습니까

Kimi-K3 는 prefill 631,705/631,705, decode 33,199/33,199, `missing_port_records` 0 입니다.
이 수치 위에 `provenance_class` 를 세웠습니다. 앞서 `X_linked` 를 철회한 이유가
"포트가 아예 없다" 였는데 그 전제가 바뀌었습니다. **`X_linked` 되살리기는 아직 손대지
않았습니다** -- 음성 대조와 unit/crosswalk 재생성이 남았다고 봅니다. 순서가 맞습니까?

### Q4. Q1(전건 독립 검토)은 어떻게 합니까

지난 판정에서 "38 개 질문 전건을 독립적으로 검토하라" 고 하셨습니다. 블라인드 패킷 인프라는
그대로 있습니다(패킷 반출·답 수집·overlay 스키마, 자기검사 470 항목). 이번 라운드의
`d_nope -> d_v` 는 이제 **발행물의 라벨을 실제로 바꾸는 유일한 변경**이고 사슬 전체로
넓어졌으니, 최소한 이건 독립 확인이 필요하다고 봅니다. 어느 범위로 돌립니까?

## 7. 읽을 곳

```
rules/label_overrides.yaml      d_nope -> d_v 항목 (unsqueeze 앵커 2 건이 provenance_class)
                                은퇴 블록 3 건 (주석)
rules/label_confirmed.yaml      은퇴 23 / 재앵커 20
src/label_overrides.py          _SPREADS, _classes(), _decoded_ports(), _spread()
                                모듈 docstring 의 CLASS MODE 절
develop/test_spread_provenance.py   8 사례
develop/test_bundle_guards.py       빈 포트 fixture (63/63)
models/*/full/label_overrides.json  발화 수
models/*/full/label_confirmed.json  matched 수
models/moonshotai__Kimi-K3/full/*.ports.jsonl   포트 커버리지의 근거
```

산출물은 읽기만 하고 **수정하지 말아 달라.**

## 8. 원하는 판정

> Q1(기계장치 추가가 정당한가)이 가장 중요합니다. 되돌려야 한다면 그렇게 말해 주십시오.
> Q2(전환 여부와 측정 항목), Q3(포트 신뢰 범위와 X_linked 순서), Q4(독립 검토 범위)도
> 정해 주십시오.
>
> 그리고 제가 놓쳤을 만한 것을 지적해 주십시오 -- 특히 **`provenance_class` 가 조용히 너무
> 멀리 가는 경우**가 있는지. 저는 "다른 교정이 죽는 수가 양쪽 같다" 와 "k_nope 자리가 안
> 바뀐다" 두 가지로만 확인했습니다.
