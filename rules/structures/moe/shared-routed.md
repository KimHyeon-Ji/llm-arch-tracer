# Shared + Routed MoE (+ dense-prefix schedule)

## 정의
기본형 top-k MoE([basic.md](basic.md))에 **shared expert**(모든 토큰이 공통으로 거치는
상시 활성 FFN)를 더한 구조. 출력 = routed top-k experts 합 + shared expert(들). DeepSeek
계열의 DeepSeekMoE가 대표. 흔히 **dense-prefix**(앞쪽 몇 개 레이어는 MoE가 아니라 일반
dense FFN)와 함께 온다 — config `first_k_dense_replace`(= 앞 몇 층을 dense로).

## 관련 심볼 (rules/symbols.yaml)
`E`(routed 수), `k`(top-k), **`E_shared`**(shared expert 수 = `n_shared_experts`),
`d_moe`(expert FFN intermediate). dense-prefix 층은 dense `d_ff`를 쓴다.

## 트레이스에서 식별하는 방법 (Phase 6 DeepSeek-V2-Lite 실측)
- routed 경로는 basic.md와 동일(router `linear`→`softmax`→`topk`→grouped/per-expert FFN).
  V2-Lite는 grouped(`_grouped_mm`, weight `[E, ...]`) SwiGLU, `E`=64, `k`=6.
- **shared 경로**: routed와 병렬로 별도 FFN(`...mlp.shared_experts.*`) 경로가 있어 모든
  토큰이 거침 → 그 출력이 routed 합과 `elementwise_add`로 합쳐짐. `E_shared`개(V2-Lite=2).
- **dense-prefix**: layer 0은 MoE가 아니라 일반 dense FFN(`mlp.gate_proj/up_proj/down_proj`)
  → op 시퀀스가 layer 1~26(MoE)과 달라 **C2 클러스터링이 2개 클러스터로 잡는다**(layer 0
  dense + layer 1-26 MoE). config에 `layer_types`가 없으면 C2는 WARN(값은 정확, 비교 대상
  필드 부재). 이 스케줄 근거는 `first_k_dense_replace`(V2-Lite=1).
- routed-token 수는 값 의존적(`k*T` 심볼) — C8이 심볼릭 처리(WARN 정상). C8은 router dim=E,
  top-k, expert weight를 트레이스에서 확인한다.

### 변형: sigmoid gating + aux-loss-free 로드밸런싱 (DeepSeek-V3, Phase 9 실측)
V2-Lite는 router가 softmax였지만, **V3는 sigmoid gating**을 쓴다. 라우터 점수에 학습된
**correction bias**(`e_score_correction_bias`)를 더한 뒤 top-k를 고르는 aux-loss-free 방식.
트레이스 라우터 시퀀스: `matmul`(router) → `sigmoid` → `elementwise_add`(bias) → `topk` →
`scatter_`/`gather`/`masked_fill`(라우팅) → `div_`(top-k 정규화). `sigmoid`는 optype_map에
매핑됨. C8은 이 경우에도 router dim=E/top-k/expert weight로 정상 검증(gating 함수 무관).

## 확인된 모델 (계속 추가)
- **`deepseek-ai/DeepSeek-V2-Lite`** (Phase 6): 27층, dense-prefix 1층 + MoE 26층, `E`=64,
  `E_shared`=2, `k`=6, `d_moe`=1408, **softmax gating**. MLA 결합([../attention/mla.md](../attention/mla.md)).
  총 15.71B / 활성 ~2.66B. revision `604d5664dddd88a0433dbae533b7fe9472482de0`.
- **`deepseek-ai/DeepSeek-V3`** (Phase 9): 61층, dense-prefix 3층 + MoE 58층, `E`=256, `k`=8,
  shared expert, **sigmoid gating + e_score_correction_bias(aux-loss-free)**. MLA 결합. C1 61==61,
  C5 61/61층, C8 E=256 top-8 grouped, C10 909 params 전부. revision `e815299b0bcbac849fa540c768ef21845365c9eb`.
- **`zai-org/GLM-4.5-Air`** (Phase 15): 46층, dense-prefix 1층(`first_k_dense_replace=1`) +
  MoE, `E`=128 routed + `E_shared`=1, `k`=8, GQA 96:8. DeepSeek와 다른 회사·설정으로 dense-prefix
  +shared MoE 일반화 재확인. MTP 선언(native 생략 → C15 WARN, [../auxiliary/mtp.md] 참고).
  C2 WARN(dense-prefix는 scalar `first_k_dense_replace`라 per-layer 리스트 없음), C8 트레이스 검증,
  C10 690 params 전부, C13 repro. GLM은 MLA 아님(GQA).

## 참고 소스
- transformers `models/deepseek_v2/modeling_deepseek_v2.py`(네이티브) — 트레이스로 직접 관측
- DeepSeek-V2 Technical Report (arXiv:2405.04434) — DeepSeekMoE(shared+routed) 설계 근거
- config: `n_shared_experts`, `num_experts_per_tok`, `n_routed_experts`, `first_k_dense_replace`

- **`baidu/ERNIE-4.5-21B-A3B-PT`** (Phase 34): 28 layers, `E`=64, `k`=6, `E_shared`=1,
  `d_moe`=1536. 새 벤더 계열, 새 규칙 0개.
- **`Qwen/Qwen3.5-397B-A17B`** (Phase 37): 60 layers, `E`=512, `k`=10, `E_shared`=1,
  Gated DeltaNet 혼합(`n_h_lin_k`=16). Qwen3.5/3.6 계열이 만든 규칙을 그대로 재사용 — 새 규칙 0개.
- **`ibm-granite/granite-4.0-h-small`** (Phase 33): 40 layers, `E`=72, `k`=10, `d_moe`=768,
  **공유 MLP 폭이 expert 폭과 다르다**(`d_shared`=1536 vs 768). 그래서 d_moe 의 shared_* 별칭
  ("값이 이미 d_moe 와 같을 때만 태그한다")을 쓸 수 없어 `d_shared` 심볼을 새로 두었다.
  출처: `modeling_granitemoehybrid.py:742,744`.
