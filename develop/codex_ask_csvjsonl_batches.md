이 요청은 CSV/JSONL 축 라벨 정확도만 확인하는 **마지막 라운드** 검토입니다.
통과하면 이 산출물을 results 브랜치에 올립니다.

## 먼저 읽어주세요

develop/codex_review_request_csvjsonl_PREAMBLE.md

(PREAMBLE에 있는 "절대 하지 말아야 할 것" — 유도 상수 표는 증거로 쓰지 말라는 부분을
특히 지켜주세요.)

## 지난 라운드 이후 무엇이 바뀌었나

expand 653건의 방향을 소스대로 확정해 주신 것을 반영했습니다. **653 → 314** 로 줄었고,
등가류 충돌은 **0 을 유지**했습니다. 47개 모델 재트레이스했습니다.

### 두 갈래로 갈렸습니다 — 한 방향이면 풀리고, 맞바꿈이면 안 풀립니다

| 묶음 | 형태 | 결과 |
|---|---|---|
| **Kimi-K2 x3 + GLM-5.2 (339건)** | `k_pe` 의 feature 축이 `d_rope` — **한 방향** | `spread: class` 로 해결. **expand 0, 등가류 0** |
| **Granite / Nemotron-Super (296건)** | head ↔ state **맞바꿈** | 되돌림 |
| **Zamba2 (18건)** | head ↔ head_dim **맞바꿈** | 되돌림 |

H2/H4/H6/J2 는 지적하신 대로 **출력의 `n_h` 는 맞고 입력 마지막 축만** 틀렸습니다.
등가류 전체를 `d_rope` 로 끌고 가니 그 사슬이 전부 맞게 정렬됐습니다.

**맞바꿈은 왜 안 되는가**: 같은 등가류 안의 두 축을 서로 바꿔야 하는데, `spread` 는 클래스
하나에 이름 하나를 쓰는 연산이라 표현이 안 됩니다. 자리 단위로 넣으면 클래스가 쪼개져
**등가류 충돌 887건**이 났습니다(실측). 결함 653을 887로 바꾸는 거래라 하지 않았습니다.
**이 296+18건은 provenance 작업의 첫 검증 대상으로 남깁니다** -- 이미 정답이 소스로
확정돼 있으니, provenance 가 제대로 들어갔는지 재는 척도가 됩니다.

### A2 (Qwen3-Next conv_dim) 도 같은 이유로 남아 있습니다

`8192` 가 `2*key_dim+value_dim` 인 것은 맞습니다. 교정을 넣으면 **전치 위반 72** 가 납니다 --
저희 축 등가류에 **전치 간선이 없어서**(예전에 넣었다가 `repeat_kv(n_rep=1)` 경계 때문에
물린 설계) `spread` 가 전치에서 멈춥니다. 지적하신 대로 "별도 예외가 아니라 같은
provenance graph 단절" 이 맞습니다.

## 요청 — 이번엔 **모델이 47개**입니다 (배치 L~P 신설)

지난 라운드까지의 배치 A~K 는 **30개 모델뿐이었습니다.** 1차 패킷을 손으로 짤 때 28개를
골랐고 그 구성이 라운드마다 그대로 이어졌습니다. 그 결과 **17개 모델이 CSV/JSONL 외부 검토를
한 번도 받지 못했습니다.** results 브랜치에 올리기 전에 전 모델이 한 번은 검토를 받아야 하므로
배치 L~P 로 채웠습니다. 아키텍처가 비슷한 것끼리 묶었습니다.

배치 K 는 911,650자로 다른 배치의 3.5배라 **K1 / K2 로 쪼갰습니다** — 끝까지 못 읽으시면
하필 이번에 가장 크게 바꾼 모델이 검토를 못 받습니다.

아래 파일들을 순서대로 전부 읽고, 파일마다 PREAMBLE의 절차대로 모델별로 검토해 주세요.
답변은 배치 단위로 구분해 주세요.

**이미 검토받은 모델 (A~J, 30개 중 28개)**

develop/codex_review_request_csvjsonl_batchA.md
develop/codex_review_request_csvjsonl_batchB.md
develop/codex_review_request_csvjsonl_batchC.md
develop/codex_review_request_csvjsonl_batchD.md
develop/codex_review_request_csvjsonl_batchE.md
develop/codex_review_request_csvjsonl_batchF.md
develop/codex_review_request_csvjsonl_batchG.md
develop/codex_review_request_csvjsonl_batchH.md
develop/codex_review_request_csvjsonl_batchI.md
develop/codex_review_request_csvjsonl_batchJ.md

**지난 라운드에 처음 검토받은 2개 (이번엔 나눠서)**

develop/codex_review_request_csvjsonl_batchK1.md   (moonshotai__Kimi-K3)
develop/codex_review_request_csvjsonl_batchK2.md   (deepseek-ai__DeepSeek-V4-Pro)

**한 번도 검토받은 적 없는 17개 — 신설**

develop/codex_review_request_csvjsonl_batchL.md   소형 dense: SmolLM3-3B, Qwen2.5-0.5B, gemma-2-2b, gemma-3-270m
develop/codex_review_request_csvjsonl_batchM.md   표준 dense: Llama-3.1-8B, Llama-3.1-70B, Phi-4, falcon-7b
develop/codex_review_request_csvjsonl_batchN.md   중형 MoE/hybrid: Mistral-Small-3.2-24B, Qwen3-30B-A3B, ERNIE-4.5-21B-A3B, LFM2-8B-A1B
develop/codex_review_request_csvjsonl_batchO.md   대형 MoE: gpt-oss-20b, gpt-oss-120b, Llama-4-Maverick-17B-128E
develop/codex_review_request_csvjsonl_batchP.md   하이브리드: GLM-4.5-Air, Kimi-Linear-48B-A3B-Instruct

### 특히 봐주셨으면 하는 것

1. **배치 K2 (DeepSeek-V4-Pro)** — CSA window 교정이 hca 층이나 KV feature 축으로 번지지
   않았는지. 번졌다면 그게 이번 라운드 최대의 문제입니다.
2. **배치 L~P** — 미검토 17개. 표준 트랜스포머가 많아 저희 규칙이 이미 덮고 있을 것으로
   보지만, 그건 검증된 적 없는 기대입니다.
3. **지난 라운드에서 고친 자리가 정말 맞게 고쳐졌는지.** 잘못 고쳤다면 그다음으로 큰 문제입니다.
4. 값 충돌(두 config 필드가 우연히 같은 정수)이 남아 있는 자리 — 숫자로는 검증이 불가능한 곳입니다.

### 참고: 저희가 자체적으로 돌리는 검사

**전부 0 인 것:** 등가류 무모순 / 미발화 교정 / C-check FAIL / `unsqueeze` singleton.

**아직 0 이 아닌 것:**

| 모델 | 전치 | 이름생성 | reshape_incons | expand |
|---|---:|---:|---:|---:|
| moonshotai__Kimi-K3 | 261 | 345 | 0 | 0 |
| ibm-granite__granite-4.0-h-small | 0 | 0 | 36 | 216 |
| nvidia__Nemotron-3-Super-120B | 0 | 0 | 0 | 80 |
| Qwen__Qwen3.5-397B-A17B | 0 | 0 | 180 | 0 |
| Qwen__Qwen3-Next-80B-A3B | 0 | 0 | 144 | 0 |
| NX-AI__xLSTM-7b | 0 | 0 | 64 | 0 |
| nvidia__Nemotron-3-Ultra-550B | 0 | 0 | 24 | 0 |
| Zyphra__Zamba2-1.2B | 12 | 6 | 12 | 18 |
| moonshotai__Kimi-Linear-48B | 0 | 0 | 14 | 0 |
| tiny-random-Llama | 4 | 0 | 0 | 0 |
| **합계** | **277** | **351** | **474** | **314** |

지난 라운드 대비: expand **653 → 314**, 나머지는 동일합니다.
**Kimi-K2 x3 와 GLM-5.2 는 목록에서 완전히 빠졌습니다.**

### 요청

이번 라운드의 확인 대상은 **339건 교정 하나**입니다. 나머지는 전부 provenance 작업으로
넘겼고 그 목록은 위에 적었습니다. 교정이 반대편 축(`n_h` 쪽이나 value 사슬)으로 번지지
않았는지만 봐 주시면 됩니다.
