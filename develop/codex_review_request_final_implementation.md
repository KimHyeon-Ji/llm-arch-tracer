# Codex 검토 요청 — 이번 작업 전체 최종 코드 리뷰

## 배경

지난 며칠간 Kimi-K3 게이트 FAIL 6건을 전부 닫았다(현재 `develop/verify_all.py` FAIL 0건).
그 과정에서 Codex가 **설계**를 준 것들을 이쪽에서 **코드로 구현**했는데, 구현된 실제 코드는
아직 검토받은 적이 없다. 커밋 순서대로: `d3a922ce`(E_shared/reused_symbol/n_kv 스코프),
`1ab10214`(재트레이스+승격), `0a6972c2`(게이트 구멍 2개), `70e33cfa`(reshape_incons
255→0), `aeb904ae`(stub_ambiguous, label_no_name.yaml).

**설계는 받았지만 실제 구현 코드는 처음 보여드리는 것**: `develop/verify_all.py`의 게이트
구멍 수정 2개, `rules/label_no_name.yaml` + `src/label_no_name.py` 전체.

## 1. freeform 상한 (`develop/verify_all.py:check_documented_literals`)

기존: `values: any_small_odd_or_scan` 인 항목이 있으면 그 **모델 전체**를 bare-정수 검사에서
건너뛰었다(Codex가 지적한 원래 버그). 고친 것: `max_value` 필드를 추가해서, 그 값 이하의
bare 정수만 봐주고 넘는 값은 계속 검사한다.

```python
for e in entries:
    vals, models = e.get("values"), e.get("models") or []
    if not isinstance(vals, list):
        mv = e.get("max_value", 0)
        for m in models:
            freeform_bound[m] = max(freeform_bound.get(m, 0), mv)
        continue
    ...
extra = sorted(n for n in seen - allowed if n > bound)
```

Qwen 5개 모델과 Kimi-K3 둘 다 `max_value: 63`(둘 다 `fla` 공용 `chunk_size=64` 청킹이라
동일)으로 설정했다. 반영 후 Qwen 5개 모델에서 실제로 `64`(=d_chunk 자신의 값이 그 자리에서
자기 심볼로 안 풀리는 자리, 아직 미조사)가 새로 드러나서 WARN으로 남겨뒀다.

**질문**: `max_value`가 없는 필드 기본값을 `0`으로 처리했는데(주석 참고: "documents nothing
rather than silently exempting everything"), 이게 맞는 실패 방향인가? 그리고 5개 Qwen
모델에서 새로 드러난 `64` WARN — d_chunk가 자기 자신의 값에서 안 풀리는 이유를 짚어줄 수
있는가, 아니면 이건 별도로 조사할 사안인가?

## 2. phase-completeness 추적 (`develop/verify_all.py:check_baseline`)

`m["phases_seen"]`를 raw trace 파일이 실제로 존재하는 phase만 채워서 baseline 스냅샷에
같이 저장하고, 비교 시 `phases_seen`이 다르면 그 모델의 지표 비교 자체를 건너뛴다:

```python
if o.get("phases_seen") is not None and o.get("phases_seen") != c["phases_seen"]:
    print(f"   PHASE 변경  {n}: {o.get('phases_seen')} -> {c['phases_seen']} "
          f"-- 지표 비교 건너뜀(비교 불가능, 검토 후 --update-baseline)")
    continue
```

**질문**: "phase가 바뀌면 비교를 건너뛴다"가 맞는 방향인가, 아니면 "phase가 줄었다"(파일이
사라짐, 진짜 문제일 수 있음)와 "phase가 늘었다"(파일이 새로 생김, Kimi-K3 케이스)를 다르게
다뤄야 하는가? 지금 코드는 둘 다 그냥 건너뛰기만 한다 — **phase가 줄어드는 쪽은 오히려
사고(파일 유실)의 신호일 수 있는데 그것마저 조용히 건너뛰는 게 안전한지** 확인받고 싶다.

## 3. `rules/label_no_name.yaml` + `src/label_no_name.py` (전체 신규)

파일 둘 다 아래에 원문 그대로 첨부한다. 핵심 함수:

```python
def selector_key(d):
    """module/shape/axis/field/shape_index/op_type/nth/expect 만 뽑아 튜플로 만든다."""
    ...

def covered_keys(model_dir_name, model_dir):
    """LIVE한 (phase, selector)만 돌려준다 -- dead/marker없음/등가류개수변경은 전부 제외."""
    ...

def issues(model_dir_name, model_dir):
    """dead/stale/marker없음인 항목을 [(rule_id, 이유)]로 돌려준다 -- verify_all.py가 FAIL."""
    ...
```

`src/axis_classes.py:bad_stub_count()`는 `covered_keys()`가 준 것만 빼고 센다.

**질문 1 — 매칭 정확성**: `selector_key()`는 `override_stub` 딕셔너리와 `label_no_name.yaml`
항목을 **정확히 같은 8개 필드**(module/shape/axis/field/shape_index/op_type/nth/expect)로만
비교한다. `label_overrides.py`의 실제 적용기(`apply()`, line 239 근처)가 매칭에 쓰는 필드와
이게 정확히 일치하는가, 아니면 내가 놓친 필드가 있어서 실제로는 다른 자리인데 같다고
오판할 여지가 있는가?

**질문 2 — `expected_classes`의 의미**: 지금은 phase별로 매치된 `classes` 값들의 집합이
`{368}`처럼 원소 하나여야 통과다(여러 phase에 걸쳐 다른 값이면 FAIL). Kimi-K3의 두 항목은
둘 다 `phase: prefill`만 지정했으니 이 문제가 안 생기지만, **만약 나중에 어떤 모델의
no_name 항목이 prefill/decode 둘 다에 걸리는 경우**(지금은 없음) 이 "집합 크기 1" 검사가
너무 엄격한가, 아니면 그것도 원래 다르면 안 되는 게 맞는가?

**질문 3 — 회귀 위험**: `src/axis_classes.py`에서 `import label_no_name as _lnn`을
`bad_stub_count()` 함수 안에서 지역 import로 넣었다(다른 함수들과 같은 스타일). 순환
import나 성능(매 모델마다 YAML을 다시 로드하는 건 아닌지 -- `_CACHE`가 모듈 전역이라 한 번
로드하면 그대로 재사용되는 게 맞는지) 문제가 있는가?

직접 열어봐 줬으면 하는 파일:
- `rules/label_no_name.yaml` (신규)
- `src/label_no_name.py` (신규)
- `src/axis_classes.py`의 `bad_stub_count()` (수정)
- `develop/verify_all.py`의 `check_documented_literals`/`check_baseline` (수정)
- 커밋 `aeb904ae`/`0a6972c2` 전체 diff (`git show aeb904ae`, `git show 0a6972c2`)로 봐도 됨

이번엔 순수 사후 검토다 -- 지적사항이 있으면 다음에 반영하고, 없으면 이 상태로 최종
확정한다.
