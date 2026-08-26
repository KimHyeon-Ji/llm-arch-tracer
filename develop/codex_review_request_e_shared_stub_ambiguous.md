# Codex 검토 요청 — E_shared 스코프 문제 + stub_ambiguous 게이트 예외 설계

## 배경

`llm-arch-tracer`는 HuggingFace LLM을 weight 없이(meta device) 실행해 텐서 shape을
`d_model`, `n_h` 같은 아키텍처 심볼로 이름 붙이는 도구다. 이름은 두 층에서 결정된다:

1. `rules/symbols.yaml` — config 필드 하나에 심볼 하나. `scope:` 정규식으로 이 심볼이 어느
   모듈 경로에서 유효한지 제한한다(예: `E_shared`는 shared-expert 개수 심볼).
2. `src/symbolic_shape.py`의 `_dim_core` — 값이 어느 심볼과 일치하는지 찾는 해석기.
   scope가 이 모듈에 **맞는** 심볼(`hit_syms`) > scope가 아예 없는 전역 심볼(`plain_syms`)
   > scope가 안 맞아 강등된 심볼(`miss_syms`, `group` 태그 있으면 강등도 안 되고 배제) 순으로
   우선순위를 매기고, 그래도 못 찾으면 최후 수단으로 "휴리스틱"(등록된 규칙이 아니라
   산술만 맞으면 지어내는 이름, 예: `2*d_moe`)을 쓴다.

이 프로젝트는 "산술적으로만 참인 이름"을 여러 번 사고로 겪었고(코드 주석에 전부 기록돼
있음), 그때마다 스코프를 좁히거나 새 가드를 넣어 고쳐왔다. 이번에 또 하나를 발견했다.

## 발견 1 — `E_shared` 스코프가 너무 넓어서 두 곳에서 잘못된 이름을 만든다

`rules/symbols.yaml:112`:
```yaml
E_shared: {meaning: "shared expert 수", aliases: [n_shared_experts, num_shared_experts], group: moe, dim: true, priority: 6, scope: "expert|moe|shared"}
```

`scope: "expert|moe|shared"`는 OR 정규식이라 "expert"나 "moe"라는 글자만 들어가면 매치된다
— 그런데 MoE 아키텍처의 거의 모든 모듈 경로에 "expert"나 "moe"가 들어간다(shared-expert와
아무 상관 없는 **routed** expert 경로도 마찬가지). Kimi-K3(moonshotai/Kimi-K3)에서 실제로
두 군데가 이 때문에 잘못 렌더된다:

**(a) MoE 캡 셔플 부산물이 `2*E_shared`로 둔갑** — `model.layers.*.block_sparse_moe.experts.*.act_fn`
(라우팅된 전문가, shared_experts 아님)의 값 4가 decode에서 `2*E_shared`로 렌더된다
(`E_shared`=2, 2*2=4로 우연히 일치). 실제로 이 값은 `src/kda_shim.py`의 MoE 캡 셔플
(`KDA_SHIM_EXPERT_CAP=4`)이 만드는 산술 부산물이지 아키텍처 차원이 아니다 — 이미
`review/06-open-renames.md` A60에서 "이름을 붙이면 안 된다"고 확정된 값인데, 휴리스틱이
스코프 밖 값으로 이름을 지어내 버렸다.

같은 축의 prefill 값(1280)은 바르게 bare(이름 없음)로 남는데, decode 값(4)만 `2*E_shared`로
잘못 렌더되는 비대칭도 있다 — 아마 1280은 `heur_multiple`의 배수 후보 {2,3,4}·값 범위에서
안 걸리고, 4는 2*E_shared(2*2)로 걸리는 값 크기 차이일 것으로 추정.

증거 (`models/moonshotai__Kimi-K3/full/prefill.csv`, `decode.csv`):
```
prefill: 38773,block_sparse_moe,experts,0,act_fn,slice,"[[1280, 2*d_moe]]",...,"[[1280, d_moe]]",...
decode:  749,block_sparse_moe,experts,0,act_fn,slice,"[[2*E_shared, 2*d_moe]]",...,"[[2*E_shared, d_moe]]",...
```

**(b) self_attn(KDA) 내부의 값 2가 그냥 `E_shared`로 렌더** — MoE와 전혀 무관한
`model.layers.0.self_attn`(KDA, KimiDeltaAttention) 안쪽 체인에서도 값 2가 `E_shared`로
나온다:
```
1302: model.layers.0.self_attn, slice, [[B, n_h_kda, 5, d_chunk]] -> [[B, n_h_kda, 5, E_shared]]
1311: model.layers.0.self_attn, elementwise_add, [[B, n_h_kda, 5, E_shared], [B, n_h_kda, 5, E_shared]] -> [[B, n_h_kda, 5, E_shared]]
```
이건 `heur_multiple`(배수 추측)이 아니라 값 2가 **그대로** `E_shared`(값도 2)와 일치해서
나온 것으로 보인다. `_ctx_symbols`(src/symbolic_shape.py:207-277)를 읽어보면 `E_shared`처럼
`scope`가 정의된 심볼은 스코프가 안 맞으면 `miss_syms`로 강등되고(그마저 `group: moe`
태그 때문에 최후 후보(`out_of_scope_symbol`, 3b 단계)에서도 배제되어야 정상인데), 실제
렌더는 `E_shared`가 붙었다. **`self_attn`이라는 경로 문자열엔 "expert"도 "moe"도 "shared"도
없는데 왜 `hit_syms`/`plain_syms` 어느 쪽으로도 안 걸려야 할 이 심볼이 나왔는지, 정확한
코드 경로를 못 짚었다** — `reused_symbol`(같은 shape 튜플 안에서 이미 쓰인 심볼 재사용,
symbolic_shape.py:436-439) 같은 다른 폴백을 통해 들어왔을 가능성이 있는데, 확인이 필요하다.

## 요청 1 — 정확한 원인 추적 + 안전한 수정안

1. `src/symbolic_shape.py`의 `_dim_core`/`_ctx_symbols`/`resolve_shape`를 따라가서 (b)가
   정확히 어느 분기를 타는지 짚어달라. `group` 태그가 있는 심볼이 `heur_ctx`
   (`hit_syms + plain_syms`, 440번째 줄 근처)나 `reused_symbol` 폴백에는 걸러지지 않고
   들어갈 수 있는 게 의도인지, 아니면 이것도 같은 부류의 새 구멍인지 판단해달라.
2. `E_shared`의 `scope: "expert|moe|shared"`를 좁히는 게 안전한지 봐달라 — 예를 들어
   `expert|moe` 를 빼고 `shared`만 남기거나(shared_experts/shared_mlp류 모듈은 이미
   "shared"를 포함), 아니면 라우팅된 experts 경로를 명시적으로 제외하는
   negative lookahead(`(?<!\.)experts\.\d`류, 정확한 패턴은 검토 바람)를 추가하는 방법이
   있다. **위험**: `E_shared`가 현재 실제로 렌더되는 곳은 fleet 전체에서 딱 2모델뿐로
   확인했다 — `deepseek-ai/DeepSeek-V2-Lite`(`mlp.shared_experts.gate_proj`, "shared" 포함,
   좁혀도 안전)와 Kimi-K3(둘 다 오탐). 근데 이건 CSV에 실제로 "E_shared"라는 글자가 찍힌
   것만 grep한 결과라 — `rules/derived_dims.yaml`에 등록된 합성식(`E_shared*d_moe`류)을 통해
   간접적으로 쓰이는 다른 모델(DeepSeek-V3, Qwen3-Next 등 shared-expert가 있는 MoE 계열)이
   있는지도 확인해달라. 그쪽은 `derived_dims.yaml`의 별도 매칭 경로라 이 스코프 변경의 영향을
   안 받을 수도 있지만, 정확히 확인이 필요하다.
3. 이전에 같은 부류 버그(`d_ff` 스코프가 너무 넓었던 것, 외부 검토 2026-07-29)가 있었다 —
   `git log`에서 그 커밋을 찾아 그때 어떤 식으로 좁혔는지, 같은 패턴을 여기도 쓸 수 있는지
   참고해달라.
4. 수정안을 낼 때 **fleet 전체 44개 모델 재검증(0 회귀) 없이는 반영하지 말 것** — 이
   프로젝트는 심볼 스코프를 건드려서 회귀를 낸 이력이 여러 번 있다(`review/06-open-renames.md`,
   특히 A44/A45 관련 절 참고). 코드 변경 초안만 제안하고, 실제 반영·검증·커밋은 이쪽에서
   `develop/rule_coverage.py`/`develop/regen_summaries.py`/`develop/verify_all.py` 순서로
   직접 하겠다.

## 발견 2 — `stub_ambiguous` 게이트 예외가 없다

`src/axis_classes.py`의 `bad_stub_count()`가 `full/*.unsettled.json`의 `stub_ambiguous`
플래그를 무조건 하드 FAIL로 취급한다. 이 플래그는 override 초안이 "한 레이어 안에서
등가류 여러 개를 동시에 잡는다"(같은 fingerprint를 가진 서로 다른 클래스가 여러 개 있어
`spread: class` override 하나로는 안전하게 못 지목한다는 뜻)일 때 붙는다.

Kimi-K3의 발견 1-(a) 값(MoE 캡 셔플 부산물, `experts.*` 라우팅 경로에 인스턴스가 여럿이라
매 레이어 안에서도 등가류가 여러 개 생김)이 정확히 이 경우다. **A60이 이미 "이 값은 이름을
붙이지 않는 게 정답"이라고 소스 근거와 함께 확정했는데도**(review/06-open-renames.md,
models/moonshotai__Kimi-K3/review_findings.json에 `verdict: no_name_exists`로 기록됨),
`bad_stub_count`는 이 판정을 전혀 모른다 — override를 만들 생각 자체가 없는 축인데도
"override를 안전하게 못 만든다"는 이유로 계속 FAIL이 뜬다.

C10(파라미터 커버리지 검사)에는 이미 같은 성격의 문제를 푼 전례가 있다 —
`src/run.py`의 `_expert_cap_gap`/`_attn_res_layer0_gap`이 `adaptation_log`/`config`에서
"이 파라미터들은 원리적으로 커버 안 되는 게 정상"인 집합을 계산해서 `c10_coverage`에
`expected_gap=` 으로 넘긴다. `bad_stub_count`에는 이런 예외 경로가 아예 없다.

## 요청 2 — 억제 범위를 좁게 vs 넓게, 어느 쪽이 맞는지

`review_findings.json`(모델별 ④층 수기 판정 파일, 이번 세션에 Kimi-K3용으로 처음 만듦)이
이미 `_covers()`류 매칭 로직(`src/review_ledger.py:229-240`, module 끝 두 마디 + 라벨 문자열
포함 여부로 판정)을 갖고 있다 — 의뢰서 항목이 판정에 커버됐는지 보는 데 이미 쓰인다.

두 가지 설계안을 고민 중인데 의견을 구한다:

**안 A (좁게)**: `bad_stub_count`(또는 그 바로 위 `write_unsettled` 호출부)에서, `stub_ambiguous`가
켜진 항목만 골라 `review_findings.json`에 `verdict: no_name_exists` 또는
`current_label_correct`인 판정이 커버하면 그 항목을 `bad_stub` 카운트에서 빼거나 `unsettled.json`
자체에서 제거. **장점**: 영향 범위가 정확히 이 실패 모드로 국한된다. **단점**: `unsettled.json`에
남아있는 다른 "규칙이 못 끝낸 축" 카운트(현재 8719건)에는 변화가 없다 — 그중에도
review_findings.json이 이미 답한 게 섞여 있을 텐데 그건 그대로 "④층 대기중"으로 계속 잡힌다.

**안 B (넓게)**: `write_unsettled()`가 `unsettled()` 결과를 쓰기 직전에, `review_findings.json`이
커버하는 항목은 (stub_ambiguous 여부와 무관하게) 전부 제거. **장점**: "이미 답변된 질문이
매 재생성마다 다시 질문으로 뜨는" 문제를 general하게 없앤다. **단점**: 이건 Kimi-K3 하나의
문제가 아니라 **모든 모델의 "규칙이 끝내지 못해 ④층으로 넘긴 축" 카운트 산정 방식을 바꾸는
일**이라 훨씬 넓은 파급이다 — 다른 모델의 review_findings.json에 있는 `status: fixed`류
판정이 실은 지금 라벨이 이미 바뀐 뒤라 더는 그 축을 안 가리키는데도 문자열 매칭만으로
잘못 커버 처리될 위험, 혹은 반대로 review_findings.json에 오래된 오기록이 있으면 진짜
새로운 문제를 가려버릴 위험도 있다.

어느 쪽이 이 프로젝트의 기존 설계 철학("판정은 있는데 산출물에 발화가 안 되면 게이트가
FAIL한다" — `src/label_overrides.py` docstring 참고)에 더 맞는지, 안 B를 하려면 어떤 안전장치
(예: review_findings.json의 verdict 종류별 화이트리스트, 혹은 재검증 스탬프)가 필요한지
의견을 달라. 코드 초안도 좋고, 그냥 설계 판단만 줘도 된다 — 실제 구현·검증·커밋은 이쪽에서
`develop/verify_all.py` 0 회귀 확인 후에 진행한다.

## 참고 파일
- `src/symbolic_shape.py` (심볼 해석기, `_dim_core`/`_ctx_symbols`)
- `rules/symbols.yaml` (E_shared 정의, 112번째 줄)
- `src/axis_classes.py` (`bad_stub_count`, `unsettled`, `write_unsettled`)
- `src/review_ledger.py` (`_covers`, `unanswered_items` — 이미 있는 매칭 로직 참고용)
- `src/validate.py` (`c10_coverage`의 `expected_gap` — 이미 검증된 예외 패턴)
- `review/06-open-renames.md` A60 (이 MoE 캡 셔플 값에 대한 기존 판정)
- `models/moonshotai__Kimi-K3/review_findings.json` (④층 판정 기록 예시)
