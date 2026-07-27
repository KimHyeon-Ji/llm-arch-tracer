# 구조 라이브러리

attention 방식·잔차 연결 방식 같은 구조적 패턴을 모델 단위가 아니라 **패턴 단위**로
문서화해 쌓아가는 곳. `01-main.md` §12 참고.

```
rules/structures/
  attention/         mha, gqa, mla, gated-deltanet, csa, hca,
                     grouped-output-projection
  moe/               dense, basic(shared expert 없는 단순 top-k), shared-routed
  normalization/     rmsnorm, qk-norm
  position_encoding/ rope, nope
  residual/          standard, mhc
  ssm/               mamba, xlstm            ← 미리 정해둔 6개 밖에서 늘어난 축
  auxiliary/         mtp
```

아직 안 채운 것(만나면 추가): expert-choice / hash 라우팅 MoE, logit soft-cap,
ALiBi, attention sink(gpt-oss·DeepSeek-V4 공통 — 현재
[attention/csa.md](attention/csa.md)에 서술만 되어 있음), 그리고 아래 "백필 남은 것" 참고.

attention 말고도 FFN/MoE, 정규화, 위치 인코딩, 잔차 연결, 보조 모듈 어디서든 새롭거나
중요한 구조가 나올 수 있어서 카테고리를 이렇게 나눴다. 새 축이 필요해지면
(예: 캐시 압축 방식, 병렬화 관련 구조 등) 카테고리를 추가한다 — 위 6개로 고정된 건
아니다.

## 새 항목을 추가/갱신하는 시점
`develop/`에서 새 모델을 검증하다가(각 Phase) 지금까지 없던 패턴을 만나면
(`02-new-module-handling.md` Tier 2/3으로 확인 후) 여기에 항목을 추가하거나, 이미 있는
패턴이면 "확인된 모델" 목록만 갱신한다.

**출발 시점에는 각 카테고리에서 가장 기본적인/이미 잘 알려진 패턴만 채워뒀다**(attention의
MHA·GQA, MoE의 dense·기본형 top-k, 정규화의 RMSNorm, 위치 인코딩의 표준 RoPE, 잔차
연결의 표준 add). 그보다 복잡하거나 특정 모델에 고유한 패턴(MLA, CSA, HCA, mHC,
MTP, shared+routed MoE, 압축형 위치 인코딩 등)은 의도적으로 비워뒀다 — develop/에서
새 모델을 처리하다가 실제로 만날 때(`02-new-module-handling.md` Tier 2/3으로 확인한
뒤) 그 시점에 채운다. 미리 답을 적어두면 시행착오 자체의 훈련 효과가 없어지고,
이 중 일부는 예약된 최종 테스트 모델과 같은 계열이라 미리 채우면 최종 테스트
의미도 옅어진다. 이 목록은 dev Phase 4~18과 DeepSeek-V4(예약)를 거치며 위 트리대로
채워졌다.

## 백필 이력 (2026-07-27 완료)
예약 최종테스트 7개 모델을 돌린 뒤(`models/` = 24개) **`02-new-module-handling.md`
Tier 3-3(구조 라이브러리 반영)이 한동안 DeepSeek-V4 계열에만 적용돼 있었다.** 지금은 전부 반영됨:

- **DeepSeek-V4 Pro/Flash** → `attention/csa.md`, `attention/hca.md`,
  `attention/grouped-output-projection.md`, `residual/mhc.md` 신규
- **gpt-oss-20b/120b** → `attention/attention-sink.md` 신규(V4와 공용), `moe/basic.md` 갱신
- **Llama-4-Maverick** → `position_encoding/nope.md`(NoPE 인터리브), `moe/basic.md`(top-1 라우팅)
- **Llama-3.1-70B/405B** → `attention/gqa.md`
- 공통 문서(`rope.md`, `rmsnorm.md`, `residual/standard.md`)의 "확인된 모델"을 24개 모델
  전수 기준으로 다시 씀

**이 누락이 다시 생기지 않도록** 검증 체크 **C17(모듈 온보딩)** 을 추가했다 — 유도 상수가
설명 안 되거나 모델이 이 라이브러리 어디에도 없으면 WARN이 뜬다. 절차는
`02-new-module-handling.md`의 「Phase 0 — 신규 모델 온보딩」 참고.

## 항목 형식
```markdown
# <이름> (<약자>)
## 정의
## 관련 심볼 (rules/symbols.yaml)
## 트레이스에서 식별하는 방법
## 확인된 모델 (계속 추가)
## 참고 소스
```

## 소스 확인 순서
`02-new-module-handling.md` Tier 2와 동일. Hugging Face(config docstring, model card)와
[Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/)를
나란히 확인하는 걸 기본으로 한다 — 후자는 여러 모델의 아키텍처가 이미 비교 정리되어
있어 계보 파악이 빠르지만, 2차 자료이므로 원 소스(모델 카드·논문·독립 구현)로
반드시 재확인한다.
