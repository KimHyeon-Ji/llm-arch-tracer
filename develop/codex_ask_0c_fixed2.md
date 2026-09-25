# 0-c 결함 넷 수정 — 함수 비교 · alias phase · fail-closed · 패킷 진본성

요청일: 2026-09-25. 선행: `codex_ask_0c_fixed.md`
(→ **다항식 수정 승인 / 0-c 전체 미승인. 차단 결함 넷**).

넷 다 재현하고 고쳤습니다. raw overlay 단계로 넘어가지 않았습니다.

```
tracer            bb93e6fa
results-labeled   749a1b3c   (파일럿 3 건 재반출, revision 7, packet id 동일)
```

---

## 1. 함수가 든 식의 false `different` — 다섯 개 전부 재현

```
                                              고치기 전            고친 뒤
round(n_h)                  vs n_h            different (틀림)  ->  same
min(n_h,n_h)                vs n_h            different (틀림)  ->  same
max(n_h,n_h)                vs n_h            different (틀림)  ->  same
min(n_h,n_kv)+max(n_h,n_kv) vs n_h+n_kv       different (틀림)  ->  cannot_determine
round(n_h,ndigits=1) vs round(n_h,ndigits=2)  same      (틀림)  ->  cannot_determine
```

**권장한 보수적 수정을 그대로 택했습니다.**

```
정규형이 정확히 같으면 same
확실히 건전한 항등식만 정규형에서 줄인다
    min(x,x) = x,  max(x,x) = x        (멱등)
    round(x) = x  (한 인자)             이 저장소의 축은 모두 정수다
keyword 인자는 **거부한다** (Undecidable) -- 지원하지 않는 것을 조용히 흘리지 않는다
그 밖에 호출이 끼고 정규형이 다르면 **cannot_determine**
```

`OPAQUE_DIVISION` 에 `call` 을 넣었습니다. 그래서 `max(T,d_chunk)` vs `min(T,d_chunk)` 도
이제 `different` 가 아니라 `cannot_determine` 입니다.

`min(a,b)` vs `a` 에 대한 판단을 받아들였습니다 — 수학적으로는 `different` 가 맞지만 구현이
`min`·`max` 항등식을 **완전히** 처리하지 못하는 동안은 통째로 내리는 것이 맞습니다.
`min+max` 같은 항등식은 검증한 뒤 따로 넣겠습니다(지금은 `cannot_determine`).

함수 원자에 **인자 개수**도 넣었습니다(`("call", fn, len(args), ...)`) — arity 가 다른
호출이 같은 원자가 되지 않게.

## 2. alias 의 phase 범위가 수집에 적용되지 않았다

지적한 그 줄이었습니다.

```python
if m not in uni:            # 모델만으로 캐시 -> prefill 전용 alias 가 decode 에서도 쓰였다
    uni[m] = U.universe(m)
```

```python
key = (u["model"], u.get("phase"))          # 고친 것
if key not in uni:
    uni[key] = U.universe(key[0], key[1])
```

### `load_aliases` 를 fail-closed 로

```
model=None                  ->  **아무 항목도 주지 않는다**  (전에는 통과시켰다)
phase 범위가 있는데 phase=None ->  그 항목을 쓰지 않는다
model: "*"                  ->  ValueError  (전역 alias 금지와 모순된다)
```

지적대로 `*` 를 **거부**하는 쪽을 택했습니다. 전역이 필요하면 모델을 모두 적어야 합니다.

검증:

```
prefill 전용 alias
  prefill 에서는 쓰인다              O
  decode 에서는 쓰이지 않는다        O
  phase 를 모르면 쓰이지 않는다      O
  model 을 모르면 아무것도 없다      O
```

## 3. 수집기를 완전히 fail-closed 로

지적한 다섯 경로 전부:

| 지적 | 반영 |
|---|---|
| 지정 패킷에 `answers.jsonl` 없어도 exit 0 | 오류로 기록 → exit 1 |
| 전체 점검은 어떤 오류든 exit 0 | **실제 오류는 언제나 exit 1.** "답이 아직 없음" 만 예외 |
| 중복 답 두 행이 모두 eligible | 중복을 각 행의 `errs` 에도 넣고, 패킷을 격리 |
| 일부 누락·깨진 JSON 이 있어도 나머지가 accepted | **패킷 단위 격리** |
| 레코드가 객체가 아니면 죽을 수 있음 | 1 패스에서 `isinstance(dict)` 를 먼저 본다 |

지정하신 두(실제로는 세) 패스로 바꿨습니다.

```
1 패스   줄을 모두 읽어 JSON·객체 여부·중복·누락을 센다
2 패스   레코드별 형식 검사를 **전부** 한 뒤  packet_ok = 문제 0
3 패스   행을 만든다.  eligible = packet_ok and uid in order
```

종료 규칙:

```
지정 실행 + 답 파일 없음              ->  1
지정 실행 + 어떤 오류든 있음          ->  1
전체 점검 + 답이 없는 패킷만          ->  0
전체 점검 + 실제 오류가 하나라도      ->  1
```

### 자기검사가 잡은 내 결함

패킷 격리를 처음에는 **패킷 수준 문제에만** 걸었습니다. 그래서 한 줄이 형식 오류여도
나머지 11 행이 `accepted` 에 남았습니다 — 지적한 것과 같은 결함이 다른 층에 남아 있었던
것입니다. `문제가 0 일 때만` 으로 바꿨습니다.

```
한 줄이 근거 한 종류      ->  accepted 0,  전부 packet_quarantined
한 단위를 빼먹음          ->  accepted 0
깨진 JSON 한 줄           ->  accepted 0
JSON 배열 한 줄           ->  검사기가 죽지 않고, 객체가 아니라고 잡고, 격리
중복 답                   ->  두 행 모두 거부
```

## 4. 패킷 진본성 — 전체 payload 해시로 결박

지적대로 `_packet.json` 과의 대조만으로는 둘을 함께 고치면 통과했고, `shard.md` 는 순서만
봐서 본문을 바꿔도 통과했습니다. 권장한 첫 번째 방식을 택했습니다.

```
반출 때   packet_payload_sha256 = H( _packet.json + shard.md + source/* 의 경로:해시 )
          -> 대장에 기록
수집 때   다시 계산해 대조. 다르면 그 패킷 전부 격리
```

검증:

```
source 와 _packet.json 을 **함께** 바꿔도 검출한다      O
shard.md **본문만** 바꿔도 검출한다                     O
되돌리면 문제 0                                          O
```

`_ingest.json` 의 `input_sha256` 에 `_packet.json`·`shard.md`·frozen source 전부를
넣었습니다(대장·배정·answers 에 더해).

### 여기서도 자기검사가 내 결함을 잡았습니다

처음에는 패킷 디렉터리 **전체**를 해시했습니다. 그러면 검토자가 `answers.jsonl` 을 넣는
순간 반드시 불일치합니다. **반출자가 쓴 파일만** 세도록 고쳤습니다.

## 5. 지정하신 회귀시험 전부 넣었습니다

```
round(x) == x                                          O  (same)
min(x,x) == x,  max(x,x) == x                          O  (same)
min(x,y)+max(x,y) == x+y                               O  (cannot_determine -- 아래 질문)
keyword 가 다른 호출을 same 으로 만들지 않음            O  (cannot_determine)
phase 전용 alias 가 다른 phase 에서 쓰이지 않음         O
지정 패킷 답 없음 -> exit 1                            O
전체 점검 중 형식 오류 -> exit 1                       O
중복·누락·깨진 JSON 이면 그 패킷 accepted 0            O
JSON 배열 입력이 검사기를 죽이지 않음                   O
source 와 _packet.json 을 함께 바꿔도 검출              O
shard.md 본문만 바꿔도 검출                             O
```

## 6. 자기검사 전체

```
test_expr_compare          77/77     (70 -> 77)
test_ingest_answers        80/80     (59 -> 80)
test_packet_export         58/58
test_bundle_guards         63/63
test_provenance_fixture    22/22
test_segment_verdict        6/6      test_provenance_untouched  3/3
test_lowering_proof         6/6      test_tracer_alias          4/4
test_corrections_cover      5/5
                          324 항목
```

발행 라벨 132 종은 여전히 **전부 정규화 성공(실패 0)** 입니다.

## 7. 읽을 곳

```
tracer bb93e6fa
  develop/expr_compare.py        호출 항등식·keyword 거부·call 불투명 / load_aliases fail-closed
  develop/ingest_answers.py      3 패스 격리 / 종료 규칙 / verify_packet
  develop/export_shard_packet.py packet_payload_sha256 (반출자가 쓴 파일만)
  develop/test_expr_compare.py   77 항목
  develop/test_ingest_answers.py 80 항목 (4-b 격리, 4-e 변조, 4-f phase)

실행해 볼 것
  .venv\Scripts\python.exe develop\expr_compare.py "min(n_h,n_h)" "n_h"          -> same
  .venv\Scripts\python.exe develop\expr_compare.py "round(n_h)" "n_h"            -> same
  .venv\Scripts\python.exe develop\expr_compare.py "max(T,d_chunk)" "min(T,d_chunk)"
      -> cannot_determine
```

산출물은 읽기만 하고 **수정하지 말아 달라.**

## 8. 원하는 판정

> 넷이 막혔는가. 특히 **`different` 를 낼 수 있는 경우가 이제 충분히 좁은가** —
> 지금은 "나눗셈·나머지·`ceil`·`roundup`·음수 지수·심볼 지수·**함수 호출**이 하나도 끼지
> 않은 순수 정수 다항식" 일 때만 `different` 를 냅니다.
>
> 그리고 하나 묻습니다. `min(x,y)+max(x,y) == x+y` 를 **구현할 가치가 있습니까?**
> 지금은 `cannot_determine` 입니다. 1 단계 783 단위의 현재 라벨 16 종에는 `min`·`max` 가
> **하나도 없어서**(전부 순수 곱·합) 실전 영향이 0 입니다. 검토자가 `min`·`max` 를
> 제안하는 경우에만 발동하는데, 그때는 사람 판정으로 올리는 것이 더 안전해 보입니다.
>
> 승인되면 Q4 순서 1(raw overlay candidate 및 adjudication 상태 스키마 확정)로 갑니다.
