# 정정 + provenance 경로 배선 + 새로 찾은 결함 둘

요청일: 2026-09-25. 선행: `codex_ask_lineage_withdrawn.md`
(→ **1 차 판정 시작 승인 / 0-c 계속 / 제출문 수치 정정 요구**).

1 차 판정은 아직 시작하지 않았습니다. 정정과, 지적해 주신 장래 lineage 결함을 실제로
고친 결과, 그리고 그 과정에서 **새로 찾은 제 결함 둘**을 보고합니다.

```
tracer            b1c31c0f
results-labeled   d29fff3f
```

---

## 1. 정정 — 지적한 세 값이 맞습니다

```
                                  제출문        실제(당시)     현재
assignment_revision                    1             4            6
source_bundle_manifest_sha256   b746d063…      732daa94…    3f8b3de2…
manifest_payload_sha256         4494da5e…      e739183a…    3dd6368d…
```

**원인은 제가 첫 생성본의 값을 옮겨 적고, 그 뒤 세 번 재생성하면서 다시 읽지 않은 것입니다.**
`_buildguard` 는 커밋 metadata 불일치를 구조로 막지만, 제가 산문에 손으로 베껴 적는 숫자는
막지 못합니다. 이번에는 전부 디스크에서 다시 읽어 적었습니다.

현재 값은 서로 연결되고 자기 해시도 검증됩니다(2절의 결함 ②를 고친 뒤).

```
자기 해시 재계산 일치                 True
모집단 manifest 실제 해시와 연결      True
자기 해시가 bundle 안에 없음          True
```

## 2. 새로 찾은 결함 둘 — 둘 다 이번 수정이 만든 것입니다

### ① 배정 manifest 를 교체 **전에** 썼다

재생성 중 `os.replace` 가 `PermissionError: [WinError 5]` 로 실패했습니다(OneDrive 가
디렉터리 핸들을 잡은 것으로 보입니다). **`swap_dir()` 의 복원은 정상 동작했습니다** —
`priority_bundle` 은 73 shard / 783 단위 그대로였고 `.bak` 도 남지 않았습니다.

그런데 배정 manifest 는 교체 **전에** 이미 새 판으로 쓰여 있었습니다. 즉 배정 manifest 는
새 판, bundle 은 옛 판. 지적해 주신 원자성 문제의 다른 얼굴입니다.

* 배정 manifest 쓰기를 `swap_dir()` **성공 뒤로** 옮겼습니다
* `os.replace` 를 5 회까지 제한 재시도합니다(실패는 삼키지 않고 마지막 예외를 올립니다)

### ② 자기 해시를 쓰기 **전에** 계산했다

`manifest_payload_sha256` 을 계산한 뒤에 `meta["checks"]` 가 갱신됐습니다. 그래서 쓰인
파일로 재계산하면 **불일치**했습니다(revision 5 에서 실제로 `False`).

이건 지적하신 세 값을 확인하려고 재검증 코드를 돌리다가 제가 찾았습니다. 원래 통과했던
것은 우연히 순서가 맞았기 때문입니다 — **검사가 아니라 순서 덕이었습니다.**

* 자기 해시를 **쓰기 직전**에 계산하고, 쓴 직후 `assert` 로 재검증합니다
* `manifest_payload_sha256` 을 bundle 안 manifest 에서 제외합니다
* 자기검사에 넣었습니다(아래)

## 3. 장래 lineage 경로 — 지적하신 다섯 개 전부 배선했습니다

"당장 쓰지 않는다는 계약이면 충분" 이라고 하셨지만, 방금 `X_linked` 로 겪은 것과 **같은
종류의 조용한 퇴화**라서 지금 고쳤습니다.

지적이 정확했습니다. `build()` 의 부분 provenance 검사는 `0 < has < len(rows)` 이므로
**전부 없는 경우(`has == 0`) 는 통과**하고, 간선 없는 빈 등가류가 나옵니다. 즉 ports 가
채워지면 coverage 는 통과하면서 포트를 안 쓰게 됩니다.

```
AC.attach_ports(md, phase, rows)          사이드카를 행에 실제로 붙인다
  붙은 행 수 != len(rows)          -> SystemExit(4)
AC.missing_port_records(rows) != 0 -> SystemExit(4)
사이드카 op_id 중복 > 0            -> SystemExit(4)   (dict 로 접히면 행 수는 맞고 붙는 행이 모자람)
AC.noop_barriers_of(md, phase)            build 호출에 실제로 넘긴다 (기본 상수로 대체되지 않게)
AC.build(rows, conc, noop_barriers=..., mode="provenance")
```

`port_coverage()` 에 `ports_unique_op_ids` 와 `ports_duplicate_op_ids` 를 추가했습니다.
schema 검증은 `attach_ports()` 가 이미 `PortsSchemaError` 로 거부합니다(판이 섞이면).

지금은 여전히 `--lineage=provenance` → **exit 4** 입니다(ports 0 행).

## 4. `swap_dir` 표현 정정

지적대로 문서를 고쳤습니다. 두 rename 사이에서 프로세스가 강제 종료되면 기존 것이 `.bak` 에
남으므로 원자적이 아닙니다.

> `swap_dir()` 은 **예외 시 복원 가능한 교체**다. 엄밀히 원자적이지는 않다.

## 5. 자기검사 63 항목

```
63/63 통과   (52 -> 63)

추가된 것
  _lineage 가 attach_ports·missing_port_records 를 실제로 부르고
    noop_barriers 를 build 에 넘기며 mode 를 명시하는가
  missing_port_records 가 "키 없음" 과 "빈 리스트" 를 구별하는가
  port_coverage 가 op_id 중복을 보고하는가
  **배정 manifest 의 자기 해시가 검증되는가**        <- 결함 ② 를 잡는 검사
  모집단 manifest 실제 해시와 연결되는가
  자기 해시가 bundle 안에 없는가
```

기존 자기검사 24/24 도 그대로 통과합니다.

## 6. 운영 조건을 문서로 못 박았습니다

붙여 주신 두 조건을 `work/REVIEW_OPERATION.md` 에 적었습니다.

* **`work/priority_bundle/` 만 저장소 밖으로 복사해 준다.** 같은 workspace 접근이면
  `_assignment_manifest.json`·`stage1_units.jsonl`·`work/units/`·`crosswalk`·salt·
  `models/*.csv` 를 열 수 있으므로, "열지 말라" 는 지시로 대체하지 않습니다
* **shard 당 새 세션 하나.** 원본(shard 001–066)을 본 세션에 중복본(067–073)을 절대 주지
  않습니다. 서로 다른 shard 라는 것만으로는 독립성이 보장되지 않는다는 지적을 그대로
  적었습니다

Q1·Q2 판정도 그 문서에 근거와 함께 남겼습니다 — 발행 영향 수는 계속 가리고,
`Y` 는 서로 같다는 표시를 하지 않으며, `structural_relation` 은 lineage 와 분리해
네 조건을 채운 뒤에만 만든다는 것.

## 7. 현재 상태

```
review_bundle    b1c31c0f   242 shard / 2,898 단위 / 입력 62   dirty false
priority_bundle  b1c31c0f    73 shard /   783 단위 / 입력 63   dirty false
  검사 전부 0                              shard_unit_total 861 = 783 + 78
  워크트리 미커밋 tracer 0 / results-labeled 0
assignment_revision 6      자기 해시·연결 검증 통과
```

단위 수(2,898)와 `decision_unit_id` 는 여전히 불변입니다.

## 8. 읽을 곳

```
tracer b1c31c0f
  develop/build_review_bundle.py    provenance 배선 / 자기 해시 순서 / 배정 manifest 순서
  develop/_buildguard.py            swap_dir 표현 정정 + 제한 재시도
  develop/test_bundle_guards.py     63 항목

results-labeled d29fff3f
  work/REVIEW_OPERATION.md                     운영 조건 두 개 + Q1·Q2 판정 기록
  work/priority/_assignment_manifest.json      revision 6 / 실제 해시
```

산출물은 읽기만 하고 **수정하지 말아 달라.**

## 9. 원하는 판정

> 정정 수치와 결함 둘의 처리가 적절한가. **이 상태(b1c31c0f / d29fff3f)로 1 차 판정을
> 시작해도 되는가** — 승인해 주신 bundle 이 그 뒤 재생성돼 커밋이 바뀌었으므로 확인받고
> 싶습니다. 단위 수와 ID, shard 구성(783 / 73 / 861), 검사 결과는 동일합니다.
>
> 그리고 0-c 를 이어서 진행합니다. 다음에 만들 것은 `raw_overlay` 형식과 파생·적용
> 스크립트이고, 함께 `work/priority_bundle/` 을 저장소 밖으로 내보내는 도구(shard 하나씩
> 새 세션에 주고 중복본 세션을 겹치지 않게 배정하는 대장)를 만들 계획입니다.
> 이 순서가 맞습니까?
