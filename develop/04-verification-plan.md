# 검증/개선 계획 (개인용)

`../01-main.md` / `../02-new-module-handling.md`와 `../src/`의 코드는 아직 실제 파이썬
환경에서 실행 검증되지 않았다. 이 폴더(`develop/`) 안에서, 작고 확실한 모델부터
**한 번에 새 요소 하나만** 늘려가며 검증한다. 각 Phase는 이전 Phase의 체크리스트를
통과한 뒤에만 넘어간다.

여기서 검증을 마친 **출력 폴더**(`develop/out/<model>`)는 `../models/`로 승격하고(`promote.py`),
프로파일은 `develop/models/`에 남겨 재생성 소스로 보관한다. 과정에서 나온 새 규칙은 `../rules/`에
반영하되, 검증 과정 자체(회귀 테스트, 사람 검증 기록, 로그)는 이 폴더에 계속 쌓아 남겨둔다.

## 절대 사용하지 않는 모델 (최종 테스트용 예약)

아래 모델은 파이프라인이 어느 정도 완성된 뒤 **최종 테스트에만** 쓴다. 개발/검증
단계에서는 절대 로드하지 않는다.

- `deepseek-ai/DeepSeek-V4-Pro`
- `openai/gpt-oss-20b`
- `meta-llama/Llama-3.1-70B`, `meta-llama/Llama-3.1-405B`
- Llama 4 Maverick(400B) 계열
- `deepseek-ai/DeepSeek-V4-Flash` — Raschka gallery로 확인한 결과 V4-Pro와 완전히 같은 계열("MLA-style CSA/HCA with mHC")이라 정식으로 예약 목록에 포함
- `openai/gpt-oss-120b` — 같은 이유로 정식 포함("동일한 alternating-attention 레시피"로 확인됨)

**주의(권고)**: Llama 4 Scout 등 그 밖의 형제 모델도 아키텍처가 사실상 동일할 수 있어 아래 계획에는 포함하지 않았다. 새로 추가할 모델이 이 예약 목록과 같은 계열인지 애매하면, 투입 전에 먼저 확인할 것.

## 이 폴더의 위치

```
project-root/
  01-main.md
  02-new-module-handling.md
  src/            # <- develop/이 검증하는 대상 코드 (여기서 수정하지 않고, 여기서 발견한
                  #    버그를 여기에 고쳐 반영한다)
  models/         # <- 검증 통과한 완성 산출물(출력 폴더)만 승격
  rules/          # <- Tier 2/3에서 나온 규칙을 이동/누적
  develop/        # <- 지금 이 폴더. 프로파일·작업 출력·검증 기록의 집
    04-verification-plan.md   # 이 문서
    canary/suite.yaml
    escalations/
    logs/
    models/       # 프로파일(.yaml) — 초안 작성 + 통과분 보관(재생성 소스)
    out/          # run.py 출력 작업 공간. report.md FAIL 0이면 ../models/로 승격(promote.py)
```

## Phase별 계획

각 Phase 표에는 **이번에 검증하는 것(태그)**을 명시한다. 나중에 복잡한 모델에서
실패했을 때, 이 태그를 보고 "이미 검증된 요소"와 "처음 등장한 요소"를 구분하는 데
쓴다(`../02-new-module-handling.md`의 변수 격리 절차).

### Phase 0 — 환경 준비
- `pip install torch transformers --upgrade`. GPU 불필요(meta device 기반).
- `huggingface-cli login`으로 인증 + 모델 페이지에서 라이선스 동의가 필요한 **gated** 모델:
  **Gemma-2(Phase 5), Meta Llama(Phase 7), Gemma-3(Phase 13)** — config.json조차 인증 없이는
  401. (⚠️ 실측 정정: 원래 "Gemma는 인증 불필요"라고 적었으나 google/gemma-2-2b는 gated였다.)
- Qwen·OLMoE·DeepSeek·SmolLM3·OLMo-2·Nemotron·xLSTM 등은 인증 불필요(공개).
- 검증 이력: 환경(torch 2.13.0+cpu / transformers 5.14.1, py3.14)은 Phase 0에서 구축됨.
- `../rules/optype_map.yaml`, `../rules/error_remedies.yaml`은 이미 시드 규칙이 채워져
  있다 — 여기서부터 계속 늘려간다.

### Phase 1 — 트레이서 단위 테스트 (HF 모델 없이)
- 태그: `tracer-core`, `MLA-isolated`
- ①일반 검증: `../src/tracer.py`, `../src/scope.py`를 직접 만든 작은 `nn.Module`
  (Linear 2~3개 + attention 유사 연산 + norm)에 대해 meta device에서 돌려본다. op
  캡처, `depends_on` 순환 없음, view를 거친 weight_shape 귀속 유지를 확인.
- ②**MLA만 따로 격리 검증**: KV를 저차원으로 압축했다가 복원하는 down/up projection +
  decoupled RoPE 패턴을 흉내 낸 작은 블록을 별도로 만들어 트레이싱. Phase 6
  (DeepSeek-V2-Lite)에서 MLA+MoE를 동시에 처음 만나면 실패 원인이 둘 중 뭔지 구분이
  안 되므로, MLA 하나만 미리 검증해둔다.
- 여기서 실패하면 `../src/tracer.py` 자체 버그이므로 실제 모델을 붙이기 전에 여기서
  고친다.

### Phase 2 — 최소 실제 모델 sanity
| 항목 | 값 |
|---|---|
| 모델 | `hf-internal-testing/tiny-random-LlamaForCausalLM` |
| 태그 | `sanity` |
| 확인 | C1, C3, C4, C9, C11 |
| 특징 | transformers CI가 매일 재생성하는 공식 테스트용 모델이라 버전 호환 리스크가 가장 낮음 |

### Phase 3 — 실제 소형 dense + GQA
| 항목 | 값 |
|---|---|
| 모델 | `Qwen/Qwen2.5-0.5B` |
| 태그 | `GQA` |
| 스펙(확인됨) | 24 layers, hidden 896, attention heads 14, kv_heads 2 (GQA 7:1) |
| 확인 | C6, C7 |

### Phase 4 — 실제 소형 MoE (가장 단순한 구조)
| 항목 | 값 |
|---|---|
| 모델 | `allenai/OLMoE-1B-7B-0924` |
| 태그 | `MoE-simple` |
| 스펙(확인됨) | 16 layers, 64 experts, 토큰당 8개 활성화, shared expert 없음 |
| 확인 | C8 |

### Phase 5 — 이종 attention + 완전히 새로운 op
| 항목 | 값 |
|---|---|
| 모델 | `google/gemma-2-2b` |
| 태그 | `heterogeneous-attention`, `new-op(logit-softcap)` |
| 스펙(확인됨) | 26 layers, GQA, 로컬 슬라이딩 윈도우(4096)와 글로벌 attention(8192)이 한 층씩 교대, attention/최종 logit에 tanh 기반 soft-capping |
| 확인 | C2(클러스터링), C16(신규 op 표면화 — `rules/optype_map.yaml`에 이미 `tanh` 시드는 넣어뒀다) |
| 이유 | DeepSeek 계열과 무관한 "레이어마다 다른 attention" + "완전히 새로운 연산" 연습. 예약된 gpt-oss/Llama4도 레이어별 attention이 다르므로 미리 익숙해질 수 있다 |

### Phase 6 — MLA + MoE 조합
| 항목 | 값 |
|---|---|
| 모델 | `deepseek-ai/DeepSeek-V2-Lite` |
| 태그 | `MLA`(Phase 1 격리 검증 완료), `MoE-shared+routed`, `combo` |
| 스펙(확인됨) | 27 layers, hidden 2048, MLA(16 heads, head_dim 128, KV 압축 512, decoupled per-head dim 64), MoE(2 shared + 64 routed, top-6, intermediate 1408) |
| 확인 | C5, C6, C8 |
| 이유 | MLA·단순 MoE는 이미 개별 검증됐으므로, 실패하면 "조합 자체의 문제"로 의심 범위가 좁혀진다. `../02-new-module-handling.md`의 Tier 0~3을 여기서 처음 실전 적용한다 |

### Phase 7 — 실사이즈 dense + GQA로 확장
| 항목 | 값 |
|---|---|
| 모델 | `meta-llama/Llama-3.1-8B` |
| 태그 | `scale` |
| 스펙(확인됨) | 32 layers, hidden 4096, heads 32, kv_heads 8 (GQA 4:1) |
| 주의 | gated repo — Phase 0 인증이 안 되어 있으면 401/403. Tier 0(`../02-new-module-handling.md`)에서 바로 걸러낼 것 |

### Phase 8 — 복잡 아키텍처 축소판 (MTP 진입점 연습)
| 항목 | 값 |
|---|---|
| 모델 | `bzantium/tiny-deepseek-v3` |
| 태그 | `MTP-entrypoint` |
| 특징 | transformers 공식 `DeepseekV3Config` docstring이 예시로 지목하는 축소 테스트 모델. `num_mtp_layers` 설정으로 MTP 모듈이 config에 명시적으로 존재 |
| 확인 | C15 |

### Phase 9 — 복잡 아키텍처 실사이즈
| 항목 | 값 |
|---|---|
| 모델 | `deepseek-ai/DeepSeek-V3` |
| 태그 | `scale-complex` |
| 스펙(확인됨) | 671B total / 37B active, MLA + DeepSeekMoE(256 routed + shared, top-8), 별도 MTP 모듈 |
| 주의 | V2-Lite에 없던 sigmoid gating + expert bias(aux-loss-free 로드밸런싱)가 추가됨 — 새 op 1~2개 각오 |
| 성능 참고 | `TorchDispatchMode`는 op당 Python 인터셉트 비용이 있어, 61 layers 규모에서는 실제 연산이 없어도 순수 오버헤드로 시간이 걸릴 수 있다 |
| 비고 | DeepSeek-V4-Pro와 다른 세대 아키텍처라 최종 테스트 모델의 신선도를 해치지 않음 |

---

Phase 9까지가 최소 커버리지였다. 아래는 [Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/)(85개 모델, Hugging Face config와 나란히 확인됨)를 참고해 확장한 Phase다 — 목적은 산출물·구조 라이브러리를 더 다양한 실제 아키텍처로 쌓는 것. 각 Phase는 **지금까지 없던 축 하나**를 겨냥하도록 골랐다.

### Phase 10 — 진짜 MHA 기준선
| 항목 | 값 |
|---|---|
| 모델 | `openai-community/gpt2-xl` |
| 태그 | `MHA-baseline` |
| 스펙(확인됨) | 48 layers, 순수 MHA(GQA/MQA 아님), 학습된 절대 위치 임베딩(RoPE 아님) |
| 이유 | `rules/structures/attention/mha.md`가 아직 "확인된 모델" 없이 정의만 있었다 — 실제로 GQA가 전혀 아닌 모델로 채울 첫 기회. 2019년 모델이라 신선도 이슈 없음 |
| 확인 | C1, C6 |

### Phase 11 — MHA + QK-Norm + post-norm 변형
| 항목 | 값 |
|---|---|
| 모델 | `allenai/OLMo-2-1124-7B-Instruct` |
| 태그 | `MHA`, `QK-Norm`(신규), `post-norm-variant`(신규) |
| 스펙(확인됨) | 32 layers, MHA + QK-Norm, "residual 안쪽에 post-norm을 두는" 방식(보통의 pre-norm과 다름) |
| 이유 | 지금까지 없던 정규화 두 축(QK-Norm, norm 배치 순서)을 동시에 접하는 저비용 지점 |
| 확인 | C5(연결 불변식 — post-norm 배치가 C5에 어떻게 걸리는지 확인), C6 |

### Phase 12 — NoPE
| 항목 | 값 |
|---|---|
| 모델 | `HuggingFaceTB/SmolLM3-3B-Base` |
| 태그 | `NoPE`(신규) |
| 스펙(확인됨) | 36 layers, GQA, 4개 층마다 한 번씩 RoPE를 생략(NoPE) |
| 이유 | NoPE는 예약된 Llama 4와도 관련 있는 개념이지만, SmolLM3는 Llama 4와 무관한 별도 모델이라 예약 목록을 건드리지 않고 NoPE 자체를 먼저 익힐 수 있다 |
| 확인 | C2(레이어 클러스터링 — RoPE 있는 층/없는 층이 다른 클러스터로 잡히는지) |

### Phase 13 — 더 공격적인 로컬/글로벌 혼합
| 항목 | 값 |
|---|---|
| 모델 | `google/gemma-3-270m` |
| 태그 | `heterogeneous-attention`(Phase 5보다 비율 심화), `QK-Norm` |
| 스펙(확인됨) | 15 sliding + 3 global(5:1 비율, Gemma-2보다 로컬 쪽 비중 더 큼), multi-query attention(kv_heads=1), QK-Norm |
| 이유 | Gemma-2(Phase 5)와 같은 계열이지만 비율·MQA 극단치가 달라 C2 클러스터링을 다른 비율로 재확인 |
| 확인 | C2, C7(MQA는 GQA의 극단치) |

### Phase 14 — Shared expert 없는 MoE (2번째 변형)
| 항목 | 값 |
|---|---|
| 모델 | `Qwen/Qwen3-30B-A3B` |
| 태그 | `MoE-no-shared`(Phase 4 OLMoE와 다른 조합) |
| 스펙(확인됨) | 48 layers, GQA, shared expert 없는 MoE(OLMoE와 다르게 top-k 값이 다름 — 실행해서 직접 확인) |
| 이유 | OLMoE 하나만으로는 "shared expert 없는 MoE"가 일반화되는지 알 수 없다 — 다른 라우팅 설정값으로 재확인 |
| 확인 | C8 |

### Phase 15 — Dense-prefix + shared expert MoE
| 항목 | 값 |
|---|---|
| 모델 | `zai-org/GLM-4.5-Air` |
| 태그 | `MoE-shared-expert`, `dense-prefix-schedule`(신규) |
| 스펙(확인됨) | 46 layers, GQA, 앞쪽 일부 레이어는 dense FFN으로 시작한 뒤 나머지가 MoE로 전환(DeepSeek 계열과 유사한 "MoE 워밍업" 스케줄), shared expert 존재 |
| 이유 | Phase 6(DeepSeek-V2-Lite)의 dense-prefix+MoE 조합을 다른 회사·다른 레이어 수로 재확인 — 일반화 검증 |
| 확인 | C2, C8 |

### Phase 16 — 선형/게이트 attention 하이브리드 (완전히 새로운 메커니즘)
| 항목 | 값 |
|---|---|
| 모델 | `Qwen/Qwen3-Next-80B-A3B-Instruct` |
| 태그 | `hybrid-attention`(신규, DeltaNet), `gated-attention`(신규) |
| 스펙(확인됨) | Gated DeltaNet과 Gated Attention이 3:1로 섞인 구조(12 gated attention + 36 DeltaNet), MoE + shared expert + MTP |
| 이유 | MoE·MLA와 완전히 무관한, 지금까지 이 라이브러리에 전혀 없던 attention 메커니즘 자체(선형 attention 계열) — 이번 확장에서 가장 낯선 지점. `rules/structures/attention/`에 새 카테고리가 필요할 가능성 높음 |
| 주의 | 레이어 수·op 종류가 많아 Phase 5~15보다 트레이싱이 오래 걸릴 수 있음(§ Phase 9 성능 참고와 동일 이유) |
| 확인 | C2, C16(신규 op 다수 예상) |

### Phase 17 — 상태공간(Mamba) 하이브리드
| 항목 | 값 |
|---|---|
| 모델 | `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` |
| 태그 | `hybrid-attention`(Mamba, 신규) |
| 스펙(확인됨) | 42 layers 중 GQA 4 + Mamba-2 21 + FFN 17 (가장 작은 하이브리드 후보, 4B) |
| 이유 | attention도 MoE도 아닌 선택적 스캔(selective-scan) 계열 연산 — 트레이서가 완전히 낯선 op 계열을 얼마나 잘 잡아내는지 시험하는 지점. 크기가 작아 비용 부담 적음 |
| 확인 | C10(커버리지 — Mamba 블록의 모든 param이 표에 잡히는지), C16 |

### Phase 18 (스트레치, 선택) — attention이 아예 없는 경우
| 항목 | 값 |
|---|---|
| 모델 | `NX-AI/xLSTM-7b` |
| 태그 | `no-attention`(엣지 케이스) |
| 스펙(확인됨) | self-attention 자체가 없고 mLSTM(행렬 메모리 기반 순환) 블록만 32개 |
| 이유 | 이 파이프라인의 여러 가정(`self_attn`류 이름을 찾는 scope 파싱, KV cache 기반 decode 등)이 attention 없는 모델에서도 우아하게 실패하는지(또는 의미 있게 동작하는지) 확인하는 견고성 테스트. 필수는 아니고 여유가 될 때 |
| 확인 | 커버리지·스코프 파싱이 깨지는지 여부 자체가 결과 |

## 추가 후보 (아직 손대지 않음 — 나중을 위해 남겨둠)

Raschka 갤러리에는 이 계획에 안 쓴 최신/다양한 모델이 많다(Kimi K2 계열, GLM-5 계열, MiniMax M2/M3 계열, Mistral Large 3, Ling 2.5/2.6, Nemotron 3 Super/Ultra 등). 지금 다 끌어다 쓰지 않고 일부러 남겨뒀다 — 나중에 파이프라인을 더 검증하거나, 예약 목록 외에 "진짜 처음 보는 모델"이 필요할 때 쓸 수 있도록. 이 계획에 넣을 다음 후보를 고를 땐 "지금까지 다룬 축과 겹치지 않는 것"부터 고르는 게 효율적이다.

## 통과 기준 (모든 Phase 공통)
1. `../01-main.md` §9의 해당 체크리스트 항목이 FAIL 없이 통과.
2. **같은 프로파일로 두 번 실행해 결과(csv 해시)가 동일한지 확인**(C13을 실제로 실행해서 검증).
3. 통과한 모델은 예외 없이 `canary/suite.yaml`에 태그와 함께 추가하고, 프로파일을
   `../models/`로 옮긴다.

## 진행 원칙
- 실패를 만나면 원인을 (a) Tier 0(환경/접근) (b) `../src/` 자체 버그 (c) meta/fake
  트레이싱의 구조적 한계 (d) 해당 모델만의 새로운 요소 중 어디에 속하는지, 태그를
  참고해 좁혀서 분류한다.
- 한 Phase에서 발견한 개선사항은 즉시 `../rules/`에 반영한다.
- **새로운 구조적 패턴(태그: heterogeneous-attention, MLA, MoE-shared+routed,
  MTP-entrypoint 등)을 검증했다면, `../rules/structures/`에도 항목을 추가하거나
  "확인된 모델" 목록을 갱신한다** — 개별 모델 규칙은 `rules/optype_map.yaml` 등에,
  패턴 자체의 지식은 `rules/structures/`에 쌓는다(둘은 다른 층위).
- Phase 9(최소 커버리지)까지는 반드시 통과해야 하고, Phase 10 이후는 필요에 따라 순서를 조정해도 된다 — 다만 어느 Phase든 통과 없이는 예약된 최종 테스트 모델에 투입하지 않는다.
