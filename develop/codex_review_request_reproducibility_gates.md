# 검토 요청 — 재현성 게이트 3종 설계 (구현 전)

당신은 `llm-arch-tracer`(HuggingFace LLM을 meta device에서 트레이스하고 텐서 축에 `n_h`,
`d_head` 같은 아키텍처 심볼 이름을 붙이는 파이프라인)의 **설계 검토**를 요청받았다.
코드를 아직 짜지 않았다 — 아래 계획이 구현할 가치가 있는지, 빈틈은 없는지 판단해 달라.

## 배경

이 세션에서 당신(Codex)이 DeepSeek-V4-Pro의 `c_I/2` 오라벨을 잡았다 — 실제로는
`apply_rotary_pos_emb`의 `nope`(=`c_I-d_rope`)/`rope`(=`d_rope`) 분할이었는데, op 모양만
보고 "짝수/홀수 반분할"이라고 추측해 등록한 것이었다. 사용자가 "왜 매번 외부 검토가 더
잘 잡느냐, 재발을 어떻게 막느냐"고 물었고, 3가지 습관에 합의했다:

1. 값 충돌이 많은 자리(부분 RoPE, 압축 attention)는 트레이스보다 관련 함수 전문을 먼저 읽는다
2. 대량 배치 등록 전에 1-2개를 소스로 완전히 재검증하고 나서 나머지에 전파한다
3. "값이 3개 이상 우연히 겹치는" 자리는 규모와 무관하게 자동으로 외부 검토를 거친다

처음엔 이걸 `review/*.md` 문서에 프로즈로만 적었는데, 사용자가 정확히 지적했다:
**"이걸 메모리에만 올리면 안 되지 않냐, 이 프로젝트를 가진 누구라도(다른 LLM이든 사람이든)
실행하면 똑같은 결과가 나와야 하는 거 아니냐"**, 그리고 "md 문서는 읽는 주체마다 다르게
반영하는데, 습관을 md에 적어두면 결과가 static하게 항상 같을까?" — 답은 아니라는 것이고,
그래서 기계적으로 강제 가능한 절반을 실제 게이트 코드로 옮기는 게 이 계획이다.

## 이 저장소의 기존 설계 원칙 (참고)

`develop/verify_all.py`의 docstring 서두: *"공통 원인은 하나다: '고쳤다'는 주장에 자동
확인이 없었다. 그래서 주장 대신 종료 코드로 말하게 만든다."* — 이 계획은 이 원칙을
그대로 확장한 것이다. `verify_all.py`는 이미 STATIC(yaml 파싱)/FLEET(모델별 지표)/
EXTERNAL(공개 갤러리 수치 대조)/BASELINE(퇴행 검사) 4단계이고, `check_*()` 함수 하나마다
`fail()`/`warn()`을 호출하는 명확한 패턴이다(전역 `failures`/`warnings` 리스트, `main()`
끝에서 `FAIL 0건`이 아니면 `exit(1)`).

`rules/references.yaml`은 기존에도 "외부 공개 수치 대조" 톱레벨 키를 여럿 가진 파일이다
(`attention_layers`, `kv_cache_per_token`, `modeling_sourced_symbols` 등, 각각
`checked_on`/`source`/`values` 필드 패턴).

## 계획 (전문)

아래는 사용자 승인 전, 계획 모드에서 작성한 초안이다.

```
# 재현성 게이트 3종 — md 문서가 아니라 코드로 강제한다

## Context

이번 세션에 Codex 외부 검토가 DeepSeek-V4-Pro의 `c_I/2` 오라벨(실제로는 `c_I-d_rope`/
`d_rope` partial-RoPE 경계)을 잡아냈다. 사용자가 "왜 매번 Codex가 나보다 잘 잡아내느냐,
이걸 보완하려면 뭘 해야 하느냐"고 물었고, 3가지 습관에 합의했다:

1. 값 충돌이 많은 자리(부분 RoPE, 압축 attention)는 트레이스보다 관련 함수 전문을 먼저 읽는다
2. 대량 배치 등록 전에 1-2개를 소스로 완전히 재검증하고 나서 나머지에 전파한다
3. "값이 3개 이상 우연히 겹치는" 자리는 규모와 무관하게 자동으로 외부 검토를 거친다

처음엔 이걸 review/*.md 문서에 프로즈로 적어 넣었다. 그런데 사용자가 정확한 지점을
짚었다: **"이걸 메모리에만 올리면 안 되지 않냐, 이 프로젝트를 가진 누구라도 실행하면
똑같은 결과가 나와야 하는 거 아니냐"** — 그리고 이어서 "LLM마다 md 문서를 읽고 반영하는
게 다른데, 습관을 md에 명시해두면 결과가 static하게 항상 같을까?"

답은 **아니다**다. md 프로즈는 그걸 읽는 LLM/사람의 성실도에 의존하는 가이드일 뿐,
`develop/verify_all.py`처럼 누가 돌려도 종료 코드로 말하는 코드가 아니다. 이 저장소는
이미 정확히 이 원칙으로 돌아간다 — verify_all.py의 자체 docstring이 "'고쳤다'는 주장에
자동 확인이 없었다. 그래서 주장 대신 종료 코드로 말하게 만든다"고 적어 두고 있다. 3가지
습관 중 **기계적으로 강제 가능한 절반**을 실제로 그렇게 만드는 것이 이번 계획이다.

정직하게: 습관 ①("가설 세우기 전에 함수부터 읽는다")은 **추론 순서**에 관한 것이라 스크립트가
완전히 강제할 수 없다. 대신 그 결과물(실제 소스에 존재하는 함수 이름을 인용했는가)은 검증
가능하고, 그 절반을 습관 ③의 게이트에 묶어 넣는다. 습관 ②와 ③은 완전히 기계적으로
강제 가능하다.

## 설계

### 습관 ③ + ①(결과물 절반) — `check_value_collisions.py`를 `verify_all.py` 게이트로 승격

지금 `develop/check_value_collisions.py`(이미 작성·테스트됨, 미커밋)는 값 3개+ 충돌을
**보고만** 한다 — 누군가 그걸 기억하고 돌려야 효과가 있다. 대신 `develop/verify_all.py`
(이미 STATIC/FLEET/EXTERNAL/BASELINE 4단계, `check_*()` 함수 하나마다 `fail()`/`warn()`
호출하는 명확한 패턴)에 새 검사 `check_value_collisions()`를 추가해서, **rules/를 고친
사람이 누구든 이 검사를 건너뛸 수 없게** 만든다.

- `rules/references.yaml`(기존 "외부 공개 수치 대조" 파일, `attention_layers`/
  `modeling_sourced_symbols` 같은 톱레벨 키가 이미 이 패턴)에 새 키
  `value_collisions_reviewed:` 추가. 각 항목:
  ```yaml
  value_collisions_reviewed:
    - model: deepseek-ai__DeepSeek-V4-Pro
      value: 64
      symbols: [n_h_I, d_rope, c_I-d_rope]
      function_reviewed: apply_rotary_pos_emb
      source: "modeling_deepseek_v4.py:342-359 (nope/rope 분할); Codex 외부 검토 2026-09-01"
      reviewer: codex
      date: "2026-09-01"
  ```
  `function_reviewed`가 습관 ①의 기계적 절반이다 — "무슨 함수를 읽었다"고 **이름**을
  적어야 하고, 그 이름이 실제 캐시된 소스에 없으면 게이트가 FAIL한다(지어낸 함수 이름을
  인용할 수 없게). 오늘 이미 해결된 DeepSeek-V4-Pro의 두 충돌(값 64, 값 128)을 첫 항목으로
  백필한다.

- `verify_all.py`에 새 함수:
  ```python
  def check_value_collisions(refs):
      import check_value_collisions as _cvc
      reviewed = {(e["model"], e["value"], tuple(sorted(e["symbols"])))
                  for e in refs.get("value_collisions_reviewed") or []}
      for name in sorted(os.listdir(MODELS)):
          for value, symbols in _cvc.collisions(name):
              key = (name, value, tuple(sorted(symbols)))
              if key not in reviewed:
                  fail(f"{name}: 값 {value} 이 심볼 {symbols} 와 겹치는데 외부 검토 기록이 "
                       f"없다 (review/05-overrides.md, rules/references.yaml "
                       f"value_collisions_reviewed 에 등재할 것)")
      for e in refs.get("value_collisions_reviewed") or []:
          fn, src_dir = e.get("function_reviewed"), ...  # 모델의 model_type으로 캐시 소스 경로 결정
          if fn and not _function_exists_in_cached_source(name=e["model"], fn=fn):
              fail(f"{e['model']}: function_reviewed '{fn}' 이 캐시된 소스에 없다")
  ```
  `main()`의 `check_documented_literals(refs)` 옆에서 호출(refs는 이미 로드돼 있음).
  **지금 당장 Kimi-K3의 미검토 충돌 2건(값 96, 값 128)에서 FAIL이 뜨는 게 정상** — 검사가
  살아있다는 증거이자, Kimi-K3를 결과 브랜치에 올리기 전에 정확히 무엇을 해야 하는지
  게이트가 스스로 알려주는 것이다.

- `develop/verify_selftest.py`에 결함 주입 1개 추가(이 저장소 관례: "한 번도 실패하지
  않는 검사는 아무것도 증명하지 않는다") — 가짜 미검토 충돌을 주입해 FAIL이 뜨는지 확인.

### 습관 ② — `rule_coverage.py --emit`에 `--verified` 필수화

지금 `--emit`은 "지금 이름과 같다" 항목을 조건 없이 다 낸다(안전장치 통과 = 확인 기록으로
직행). `--emit`을 쓰려면 `--verified "무엇을 재검증했는지"`를 **필수**로 요구하도록 바꾼다
(`ap.error`로 거부, 값 없이는 `--emit` 자체가 안 돎). 이 텍스트는:
1. 낸 YAML의 헤더 주석에 그대로 박히고
2. `develop/verify/batch_verification_log.yaml`(신규, append-only)에
   `{date, spec, model, count, verified}`로 기록된다.

강제하는 건 "재검증했다는 주장이 존재하는가"뿐이고 "정말 맞게 재검증했는가"는 여전히
사람 몫이다 — 이건 정직하게 한계로 남긴다. 대신 이 로그가 남아서, 나중에 "이 배치를
누가 뭘 보고 확인했는지" 감사할 수 있다(review_findings.json/label_confirmed.yaml의
source 요구사항과 같은 설계 원칙).

### 문서 갱신 (프로즈는 게이트를 가리키는 역할로 축소)

- `review/05-overrides.md`의 "값이 3개 이상 겹치면" 절: 지금은 스크립트를 "돌려보라"는
  프로즈다. **"이제 verify_all.py가 자동으로 강제한다. 안 돌려도 걸린다"**로 갱신하고
  `value_collisions_reviewed` 스키마를 문서화. `--verified` 필수화도 "쓰는 법" 절에 추가.
- `review/02-evidence.md`의 "함수를 먼저 읽는다" 절: `function_reviewed` 필드가 이 습관의
  기계적 절반이라는 점, 그리고 **추론 순서 자체는 여전히 강제할 수 없다**는 한계를 명시.

## 순서

1. `rules/references.yaml`에 `value_collisions_reviewed:` 추가, DeepSeek-V4-Pro 2건 백필
2. `develop/verify_all.py`에 `check_value_collisions()` 추가 + `main()`에 배선
3. `develop/verify_selftest.py`에 결함 주입 케이스 추가
4. `develop/rule_coverage.py`에 `--verified` 필수 인자 + `develop/verify/batch_verification_log.yaml` 기록
5. `review/05-overrides.md`, `review/02-evidence.md` 갱신 (게이트를 가리키도록)
6. `develop/verify_all.py` 실행 — Kimi-K3의 미검토 충돌 2건에서 정확히 FAIL이 뜨는지,
   나머지 모델은 깨끗한지 확인 (이게 이 계획 자체의 검증)
7. `check_value_collisions.py` + 위 전부 커밋
8. (다음 단계, 이번 계획 밖) Kimi-K3의 값 96/128 충돌 2건을 Codex에 검토 요청 — 새 게이트의
   첫 실사용. 통과하면 `value_collisions_reviewed`에 등재하고 Kimi-K3를 결과 브랜치로.

## 검증

- `verify_all.py`가 STATIC 통과 후 FLEET에서 Kimi-K3 FAIL 2건(값 96/128), 다른 모델 FAIL 0건을
  내면 게이트가 제대로 배선된 것.
- `verify_selftest.py`가 새 결함 주입 케이스를 포함해 전부 OK.
- `rule_coverage.py --emit`을 `--verified` 없이 돌리면 즉시 에러로 거부되는지 확인.
- 최종적으로 `verify_all.py` 전체 재실행, 기존 FAIL/WARN 카운트에 새 항목 외 변화 없음 확인.

## 이번엔 다루지 않는 것

- A44/A45급 "권위 있는 이름 전파" 문제(DeepSeek-V4-Pro의 `T/m_csa`↔`d_head`, `g_o` 등,
  9개+ 모델에 남은 43건) — 별도 주제이고 과거 2번 실패 이력이 있어 별도 설계 세션이 필요.
  `review/06-open-renames.md`에 이미 상세 분석이 있다. 이번 계획이 끝난 뒤, 원하면 이어서
  계획한다.
- 습관 ①의 "추론 순서" 자체를 강제하는 메커니즘 — 근본적으로 코드가 못 하는 영역이라
  이번엔 시도하지 않는다.
```

## 구체적으로 검토해 줬으면 하는 것

1. **`check_value_collisions.py`의 탐지 범위 한계.** 이 스크립트는 `structure.yaml`의
   plain `symbols:` 딕셔너리만 보고, 값이 3개 이상 겹치는 자리를 찾는다. 그런데 오늘의
   실제 사고 원인은 `n_h_I`(64)와 `d_rope`(64) **단 2개**의 plain 심볼이 겹치는 자리였다
   (세 번째 `c_I-d_rope`는 파생 심볼이라 애초에 `structure.yaml`에 안 실린다). `min_value`
   임계값을 낮추거나 "심볼 개수 3개+"가 아니라 "2개+"로 낮추면 오탐(정상적인 `d_head`↔`n_h`
   같은 흔한 2-way 겹침)이 너무 많아질까? 더 나은 임계값·탐지 기준이 있는가?

2. **`value_collisions_reviewed` 원장의 `function_reviewed` 필드가 실제로 의미 있는
   신호인가?** 계획은 "그 함수 이름이 캐시된 소스에 존재하는가"만 검증한다 — 즉 실존하는
   함수 이름을 인용하면 통과하고, **정말 그 함수를 제대로 읽었는지**는 검증하지 않는다.
   이게 헛된 안전감(가짜 엄밀함)을 주는 장치인가, 아니면 "적어도 지어낸 이름은 못 쓴다"는
   최소한의 가치가 있는가? 더 나은 검증 방법이 있다면?

3. **FAIL vs WARN 선택.** 미검토 값-충돌은 FAIL(게이트 exit 1)로 설계했다 — 즉 이 저장소를
   쓰는 누구든 rules/를 고치고 `verify_all.py`를 돌리면 무조건 막힌다. 너무 엄격한가?
   (예: 이미 알려진, 위험이 낮다고 판단된 충돌까지 매번 막을 필요가 있는가?)

4. **`rule_coverage.py --emit`에 `--verified` 텍스트를 예외 없이 필수화하는 것**이
   과도한가? 지금 설계는 배치 크기·값-충돌 여부와 무관하게 항상 요구한다. 작은 배치
   (예: 확인 3-5건)까지 요구하는 게 실익이 있는지, 아니면 임계값을 두는 게 나은지?

5. **놓친 실패 모드.** 이 게이트들이 통과해도 여전히 뚫릴 수 있는 경로가 있는가? (예:
   `value_collisions_reviewed`에 거짓 항목을 그냥 손으로 적어 넣으면 게이트를 속일 수
   있다 — 이건 이 저장소의 다른 모든 "source 인용 필수" 장치와 같은 근본적 한계인데,
   더 나은 완화책이 있는가?)

6. **더 나은 대안이 있는가?** 습관 ①(추론 순서)을 "함수 이름 존재 검증"보다 더 잘
   기계적으로 근사하는 방법이 있는가?

## 지켜야 할 제약 (구현 시)

- `develop/verify_all.py` 는 EXIT 0 + 퇴행 0 이 합격선이다 (기존 관례).
- 새 검사를 추가하면 `develop/verify_selftest.py` 에 결함 주입을 함께 넣어야 한다
  (이 저장소에서 검사 두 개가 조용히 죽어 있었던 전례가 있다).
- 코드를 고치라는 요청이 아니다 — **설계가 옳은지**를 판단해 달라는 것이다.
