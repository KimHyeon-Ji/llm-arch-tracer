# shard 단위 격리 패킷 반출 도구 완료 — 1 차 시작 전 마지막 확인

요청일: 2026-09-25. 선행: `codex_ask_lineage_withdrawn_correction.md`
(→ **산출물 승인 / 1 차 세션은 per-shard 반출 도구 뒤로 / 0-c 계속**).

지시하신 순서 1–4 를 마쳤습니다. **1 차 판정은 아직 시작하지 않았습니다.**

```
tracer            e9a5cb3b
results-labeled   62d5f27a
bundle 동결 상태   priority_bundle revision 7 / built_from_commit f4a2c70b
                   review_bundle (모집단) built_from_commit b1c31c0f
```

---

## 1. 지시하신 두 필드 — 추가하고 fail-closed 로 배선했습니다

```
priority_bundle_manifest_sha256        97ae5829…   실제 bundle manifest 해시와 일치  True
priority_bundle_assignment_revision           7   bundle 이 주장하는 값과 일치      True
source_bundle_manifest_sha256         3f8b3de2…   모집단 manifest 해시와 일치      True
manifest_payload_sha256               cf7b17fb…   자기 해시 검증                   True
```

`priority_bundle_assignment_revision` 은 제 변수가 아니라 **쓰인 bundle manifest 를 다시
읽어** 기록합니다. 그래야 대조가 됩니다.

반출 도구는 패킷을 만들기 전에 이 셋을 모두 확인하고, 하나라도 어긋나면 **패킷을 만들지
않습니다**(`exit 2`). 지적하신 반대 방향 중단 구간(교체 성공 → 배정 manifest 쓰기 전 종료)을
완전한 원자성 없이 fail-closed 로 잡는 방식 그대로입니다.

## 2. 자기 해시를 **파일에서 다시 읽어** 검증합니다

지적이 맞았습니다. `json.dump()` 뒤에 메모리의 `meta` 를 다시 해시하는 것은 파일 검사가
아니었습니다. 지시하신 다섯 단계로 바꿨습니다.

```
1  _assignment_manifest.json.tmp 에 기록
2  닫은 뒤 다시 열어 JSON 파싱
3  **읽어 온 객체로** 자기 해시 재계산        -> assert
4  priority bundle 연결 해시·revision 확인   -> assert
5  os.replace(tmp, final)
```

## 3. provenance 는 **행동 검사**로 바꿨습니다

지적대로 `inspect.getsource()` 문자열 확인은 종단 간 증명이 아니었습니다.
`develop/test_provenance_fixture.py` 로 synthetic raw/ports/semantic fixture 를 만들어
실제로 통과시킵니다.

```
22/22 통과 — 포트가 붙어야만 계보가 생긴다

1) attach_ports 없이는 포트 간선이 하나도 생기지 않는다
     op1.o 와 op2.i 가 **이어지지 않는다**
     build 가 **예외를 내지 않는다** -- 그래서 조용한 퇴화다
2) attach_ports 를 부르면 실제로 이어진다
     op1.o[0]==op2.i[0], op1.o[1]==op2.i[1], 사슬이 op3.i 까지 닿는다
     축 0 과 축 1 은 섞이지 않는다 / 외부 입력은 간선을 내지 않는다
3) 부분 provenance -> PartialPortTrace
4) coverage 1.0 / 고유 op_id 3 / **중복 op_id 를 센다** (행 수만 보면 1 을 넘어 보인다)
5) semantic barrier 를 읽어 build 에 넘긴다
6) ports 판이 섞이면 PortsSchemaError
```

항목 1 이 지적하신 실패 모드를 그대로 재현합니다. 여전히 실제 모델에서는
`--lineage=provenance` → `exit 4` 이고 `lineage_mode: none` 입니다.

## 4. `develop/export_shard_packet.py` — 패킷 하나만 나갑니다

지시하신 계약 여섯 개를 모두 구현했습니다.

| 지시 | 구현 |
|---|---|
| coordinator 만 전체 보유 | 패킷은 저장소 **밖** `../llm-arch-tracer-review-packets/<pid>/` |
| shard 하나만 | `shard.md` + `source/` + `_packet.json` 뿐 |
| 그 shard 가 참조하는 source 만 | 목록 줄에서 뽑아 복사, 해시를 bundle manifest 와 대조 |
| 다른 shard·전역 번호·배정·운영 문서 제외 | 검사 후 거부 |
| opaque packet ID | `P-xxxxxxxxxxxx` = `HMAC(salt, "packet:<rev>:<shard>")` |
| 원본·중복본 세션 다름을 **강제 검사** | 세션 대장으로 거부(`exit 2`) |

전역 번호는 실제로 지웁니다: 머리글 `# shard 067 / 73 — 축 판정 (12 단위)` →
`# 패킷 P-… — 축 판정 (12 단위)`. 그 뒤 본문에 `shard\d` 가 남아 있으면 **패킷을 만들지
않습니다.**

거부 조건:

```
같은 세션이 그 단위를 이미 받음       -> exit 2   ("원본과 중복본은 다른 세션이어야 한다")
이미 다른 세션에 나간 shard           -> exit 2
같은 세션 재전달                       -> exit 2   (단위 겹침 검사에 먼저 걸린다. 4 절 주)
배정 manifest 가 이 bundle 것이 아님   -> exit 2
자기 해시 불일치                       -> exit 2
패킷 안에 금지된 이름                  -> exit 2 (임시 디렉터리 폐기)
```

세션 대장은 `work/priority/_session_ledger.jsonl` — **패킷 밖**이고 조정자만 봅니다.
단위별 `primary`/`duplicate` 역할도 여기에만 적습니다.

**재전달을 거부하는 것에 대해:** 같은 세션에 같은 shard 를 다시 내보내는 것도 막힙니다.
막지 않으려면 겹침 검사에 예외를 둬야 하는데, 재전달이 필요하면 이미 만들어 둔 패킷
디렉터리를 그대로 주면 되므로 거부가 안전한 쪽이라고 판단했습니다. **이 판단이 맞습니까?**

## 5. 지시 4 — 독립 패킷으로 실제 시험했습니다

`develop/test_packet_export.py`. 임시 디렉터리와 임시 대장으로 돌려 실제 대장·반출 위치는
건드리지 않습니다.

```
32/32 통과 — 패킷 하나만 나가고, 같은 단위가 한 세션에 두 번 가지 않는다

누설 (13)   전역 shard 번호 없음 / 머리글이 packet id / X_linked 없음 /
            등급 문자열 없음 / 발행 영향 수 없음 / 중복·배정 문구 없음 /
            금지 이름 7 종 파일 없음 / _packet.json 은 5 개 키만 /
            source 해시 64 자리 / 참조된 source 전부 존재
독립성 (4)  같은 세션에 중복본 거부 / 다른 세션 허용 /
            이미 나간 shard 거부 / 재전달 거부
대장 (4)    성공한 반출만 기록 / 거부는 미기록 / 원본·중복본 역할 기록 / 패킷 밖
연결 (2)    연결 해시 불일치 거부 / 자기 해시 불일치 거부
```

실측 패킷 크기: **348K** (bundle 1.7M). 패킷 하나에 source 11 파일.

## 6. 전체 자기검사

```
test_bundle_guards          63/63
test_provenance_fixture     22/22      <- 새로 만든 행동 검사
test_packet_export          32/32      <- 새로 만든 패킷 검사
test_segment_verdict         6/6
test_provenance_untouched    3/3
test_lowering_proof          6/6
test_tracer_alias            4/4
test_corrections_cover       5/5
                           141 항목
```

## 7. 운영 문서를 강화했습니다

`work/REVIEW_OPERATION.md` 의 "bundle 만 저장소 밖으로 복사한다" 가 부족하다는 지적을
반영했습니다. 지금은 **조정자만 전체를 갖고, 검토 세션에는 패킷 하나**라고 적었고, 세션
독립성은 문서가 아니라 도구가 강제한다는 것과 거부 조건을 그대로 실었습니다.

## 8. 읽을 곳

```
tracer e9a5cb3b
  develop/export_shard_packet.py      패킷 반출 + 세션 독립성 강제
  develop/test_packet_export.py       32 항목 (실제 패킷)
  develop/test_provenance_fixture.py  22 항목 (synthetic fixture)
  develop/build_review_bundle.py      연결 필드 + 파일 재읽기 검증

results-labeled 62d5f27a
  work/REVIEW_OPERATION.md                     강화된 계약
  work/priority/_assignment_manifest.json      revision 7 / 연결 해시
```

산출물은 읽기만 하고 **수정하지 말아 달라.**

## 9. 원하는 판정

> 지시하신 순서 1–4 가 끝났습니다. **1 차 판정을 시작해도 되는가.**
>
> 그리고 두 가지만 확인받고 싶습니다.
>
> **Q1.** 같은 세션 재전달을 거부하는 것(4 절)이 맞습니까, 아니면 허용해야 합니까?
>
> **Q2.** 패킷의 `shard.md` 는 여전히 12 단위를 한 문서에 담습니다. 한 세션이 12 단위를
> 순서대로 보므로 앞 단위의 판정이 뒤 단위에 영향을 줄 수 있습니다(순서 효과).
> 단위 1 개당 패킷 1 개로 쪼개면 783 세션이 되는데, 지금 규모에서 그게 맞습니까?
> 아니면 shard 안 단위 순서를 세션마다 다르게 섞는 것으로 충분합니까?
>
> 다음으로 0-c 의 `raw_overlay` 형식과 파생·적용 스크립트를 구현하고, 판정이 끝나기
> 전에는 실제 overlay 를 적용하지 않겠습니다.
