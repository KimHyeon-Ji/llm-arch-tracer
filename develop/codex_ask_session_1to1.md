# 세션 ↔ shard 1:1 강제 + lock + `--out` 제한 + 파일럿 3건

요청일: 2026-09-25. 선행: `codex_ask_packet_export.md`
(→ **패킷 설계·연결 검증 승인 / 세션 1:1 미강제로 1 차 시작 보류 / 0-c 계속**).

지시하신 순서 1–4 를 마쳤습니다. **1 차 판정은 아직 시작하지 않았습니다.**

```
tracer            66d5bee1
results-labeled   2bdd76b1
bundle            priority_bundle revision 7 (동결, 재생성 안 함)
```

---

## 1. 필수 결함 — 반례를 제 도구로 재현했습니다

지적하신 그대로였습니다. 겹치지 않는 원본 두 개가 한 세션에 들어갔습니다.

```
고치기 전
  shard001.md -> S-SAME   exit 0
  shard002.md -> S-SAME   exit 0
  ledger records          2

고친 뒤
  shard001.md -> S-SAME   exit 0
  shard002.md -> S-SAME   exit 2
  ledger records          1
```

```python
# 지금
used = next((r for r in ledger if r["session_id"] == session), None)
if used:
    die(f"세션 {session} 은 이미 {used['shard']} 를 받았다 -- "
        "세션 하나에 shard 하나다 (재전달도 마찬가지)")
```

지정해 주신 세 항목을 자기검사에 그대로 넣었습니다.

```
첫 원본 -> S-1                          성공
단위가 겹치지 않는 두 번째 원본 -> S-1   **exit 2**
두 번째 원본 -> 새 S-2                  성공
```

### 고치다 한 번 더 틀렸습니다

겹침 검사를 **전역으로** 바꿨더니 **의도된 중복 배정까지 막혔습니다** — 중복본은 원본과
단위가 겹치는 것이 정상입니다. 자기검사가 잡았습니다(`새 세션이면 허용한다` FAIL).
세션 단위로 되돌리고, 대신 **한 단위가 세 번 이상 나가는 것**을 따로 막았습니다
(배정 설계는 원본 1 + 중복 1).

## 2. 동시 실행 경쟁 — lock 을 넣었습니다

73 개를 병렬로 내보낼 여지를 남기지 않기로 하고 lock 쪽을 택했습니다.

```python
class ledger_lock:      # os.mkdir 은 원자적이다
    ...
with ledger_lock():
    return _export(...)     # 대장 읽기 -> 검사 -> 패킷 생성 -> 대장 갱신 전체
```

대장 갱신도 지적하신 대로 append 를 버리고 **전체를 임시 파일에 쓰고 `os.replace`** 합니다
(append 는 중간에 끊기면 반 줄이 남습니다).

검사: lock 이 잡혀 있으면 `exit 2`, 풀리면 반출, lock 디렉터리가 남지 않음.

## 3. `--out` 이 저장소 안이면 거부합니다

```
--out <tracer 워크트리>            exit 2
--out <tracer>/develop             exit 2
--out <results-labeled>/work       exit 2
```

`os.path.realpath` 로 풀어서 두 워크트리 루트 이하이면 거부합니다. 임시 디렉터리는 밖이므로
시험은 그대로 됩니다.

## 4. Q2 답변 반영 — 12 단위 유지 + 결정론적 셔플

말씀하신 방식으로 구현했습니다.

```
seed = HMAC(salt, "order:" + packet_id)          packet id 가 다르면 순서도 다르다
문서의 단위 순서 = 그 seed 로 섞은 순서
대장에 unit_order 기록                            검토자에게는 가지 않는다
```

파일럿에서 확인한 실제 효과: 원본(P-b8cb5f4e6ae2)과 중복본(P-662255ecc1a0)의 **공통 단위
하나가 원본에서는 위치 0, 중복본에서는 위치 3** 에 있습니다.

지시하신 pilot 지표(위치 1–12 별 `cannot_determine` 비율 / schema 오류율 / 근거 누락률 /
duplicate agreement / 뒤쪽 답변 품질)는 **답이 돌아온 뒤** 집계하겠습니다. 위치 효과가
뚜렷하면 12 → 6 으로 줄이겠습니다.

## 5. Q1 답변 반영 — 재전달은 계속 거부

기본 거부를 유지했습니다. 1:1 강제가 들어가면서 같은 세션 재전달도 자동으로 막힙니다.
지적하신 `--reissue` 는 **만들지 않았습니다** — "당장은 거부 상태로 시작해도 된다" 에
따랐고, 필요해지면 조건 다섯 개(같은 shard·session 조합만 / 새 대장 항목 없음 / 연결 해시
재검증 / 기존 패킷 해시 검증 / reissue event 만 기록)를 그대로 구현하겠습니다.

## 6. 추가로 넣은 것 — 옛 revision 이 섞인 대장 거부

bundle 을 다시 만들면 revision 이 올라가고 **packet id 가 전부 바뀝니다.** 옛 대장 위에 새
패킷을 쌓으면 어느 세션이 무엇을 받았는지 알 수 없게 됩니다.

```
대장에 revision 이 다른 기록이 있으면  -> exit 2
```

## 7. 지시 4 — 파일럿 3 건을 실제로 반출했습니다

```
P-b8cb5f4e6ae2  PILOT-01   shard001    12 단위   primary     source 11
P-5c329e740cbd  PILOT-02   shard002    12 단위   primary     source 12
P-662255ecc1a0  PILOT-03   shard073     6 단위   duplicate   source  9
```

세 패킷 모두:

```
담긴 것        _packet.json / shard.md / source/    (그 밖 없음)
누설 검사      shard 번호·X_linked·등급·발행 영향 수·중복/배정 문구  전부 없음
순서           문서 순서 == 대장 기록,  배정 순서와는 다름
위치           llm-arch-tracer-review-packets/  (두 워크트리 밖, git 저장소 아님)
```

## 8. 자기검사

```
test_packet_export          51/51      (32 -> 51. 1:1 반례 4 / out 3 / lock 3 /
                                        순서 5 / revision 2 포함)
test_bundle_guards          63/63
test_provenance_fixture     22/22
test_segment_verdict         6/6      test_provenance_untouched  3/3
test_lowering_proof          6/6      test_tracer_alias          4/4
test_corrections_cover       5/5
                           160 항목
```

## 9. 읽을 곳

```
tracer 66d5bee1
  develop/export_shard_packet.py    1:1 강제 / lock / out 제한 / 순서 셔플 / revision 검사
  develop/test_packet_export.py     51 항목

results-labeled 2bdd76b1
  work/priority/_session_ledger.jsonl    파일럿 3 건 (조정자 전용)
  work/REVIEW_OPERATION.md               운영 계약
```

산출물은 읽기만 하고 **수정하지 말아 달라.** 반출된 패킷은
`llm-arch-tracer-review-packets/` 에 있고 저장소 밖입니다.

## 10. 원하는 판정

> 세션 1:1 강제와 반례 검사, lock, `--out` 제한, 파일럿 3 건이 조건을 채웠는가.
> **1 차 판정을 시작해도 되는가.**
>
> 한 가지만 확인받고 싶습니다. **파일럿 3 건을 이미 세션에 묶었습니다**(PILOT-01/02/03).
> 이 세 패킷을 그대로 1 차의 첫 세 세션으로 쓰면 되는지, 아니면 파일럿은 버리고
> (대장을 비우고) 새로 배정해야 하는지. 버린다면 그 세 shard 는 새 세션 id 로 다시
> 내보내야 합니다.
>
> 시작 허가가 나면 첫 세션 결과로 지시하신 pilot 지표(위치별 `cannot_determine` 비율,
> schema 오류율, 근거 누락률, duplicate agreement)를 집계해 보고하겠습니다.
> 그와 병행해 0-c 의 `raw_overlay` 형식과 파생·적용 스크립트를 구현하고, 판정과 불일치
> 조정이 끝나기 전에는 overlay 를 적용하지 않겠습니다.
