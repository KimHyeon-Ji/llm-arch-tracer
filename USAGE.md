# 사용법

## 1. 입력 — 모델 프로파일 작성

`develop/models/<id>.yaml`에 아래 템플릿으로 하나 만든다(프로파일은 전부 여기 둔다 — `models/`는 검증 통과한 **출력 폴더**가 오는 곳이다). 대부분의 필드는 `auto`로 두면 `src/introspect.py`가 config에서 자동으로 안전한 값을 뽑는다 — 사람이 직접 채울 값은 `model_id` 하나뿐이다.

```yaml
# develop/models/<id>.yaml
model_id: <필수 — Hugging Face repo id, 예: "Qwen/Qwen2.5-0.5B">
revision: null              # 특정 commit에 고정하고 싶으면 hash 지정, null이면 최신을 resolve
phases: [prefill, decode]   # 보통 그대로 둔다

seq_len: auto                # src/introspect.py가 top-k/window/압축 파라미터를 보고 안전한 최소값을 계산
                             # (심볼 모호성 회피를 위해 config 차원값과 안 겹치는 값으로 자동 상향될 수 있음)
overrides:
  attn_implementation: auto  # 필요시 "sdpa" | "eager" 로 직접 지정 가능 (보통 auto로 충분)

# config_overrides: 로드 시 config 필드를 덮어쓴다(선택). config.json이 설치된 라이브러리가
# 거부하거나 meta에서 못 도는 옵션(예: triton 커널)을 지정할 때, 아키텍처를 바꾸지 않는 선에서
# native 값으로 강제하는 용도. 대부분 모델은 불필요. 예(xLSTM, Phase 18):
# config_overrides:
#   chunkwise_kernel: "chunkwise--native_autograd"
#   sequence_kernel: "native_sequence__native"
#   step_kernel: "native"

extra_entrypoints: auto       # MTP 등 메인 forward 밖 모듈을 자동 탐색. 자동 탐색이 놓치면 아래처럼 수동 지정:
# extra_entrypoints:
#   - module_path: "model.mtp_layers.0"
#     input_builder: "my_mtp_input_builder"   # src/에 직접 추가해야 하는 경우

structure_format: yaml         # structure.yaml 대신 json을 원하면 "json"

sources_file: null             # 02-new-module-handling.md Tier 2 리서치 결과 파일 경로(선택).
                                # 지정하면 model_summary.md의 "참고 소스" 표가 채워진다. 형식은 3번 참고.
```

이게 전부다. 게이트된 저장소(예: Meta Llama 계열)는 실행 전에 `huggingface-cli login`과 모델 페이지 라이선스 동의가 먼저 되어 있어야 한다(`02-new-module-handling.md` Tier 0).

## 2. 실행

```bash
python src/run.py --profile develop/models/<id>.yaml --out develop/out/
```

출력은 `develop/out/<model>/`에서 검증한다(§4의 `report.md`). `report.md`가 FAIL 0이면 완성품이니 `models/`로 승격한다:

```bash
python develop/promote.py <id>      # develop/out/<model> -> models/ (report.md FAIL 있으면 SKIP)
```

## 3. (선택) Tier 2 리서치 결과 제공하기

`model_summary.md`의 "참고 소스" 표는 코드가 자동으로 못 채운다 — `02-new-module-handling.md` Tier 2 절차(HF → vLLM/SGLang/TensorRT-LLM 독립 구현 → 공식 문서 → 논문 → 벤치마크 → 일반 검색 순)로 조사한 뒤, 아래 형식의 파일을 만들어 프로파일의 `sources_file`에 경로를 지정한다.

```yaml
# develop/escalations/<model>-sources.yaml (예시 위치, 어디에 둬도 무방)
- category: "독립 구현 — vLLM"
  ref: "vllm/model_executor/models/deepseek_v2.py"
  url: "https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/deepseek_v2.py"
  checked: "MLA KV 압축 차원 해석이 추출된 shape과 일치하는지 대조"
- category: "논문"
  ref: "DeepSeek-V2 Technical Report"
  url: "https://arxiv.org/abs/2405.04434"
  checked: "decoupled RoPE 차원(64) 근거 확인"
```

이 파일이 없으면 `model_summary.md`는 "참고 소스" 절을 빈 채로 두고 Tier 2가 필요하다고 표시한다 — 값을 지어내지 않는다(P1).

## 4. 출력

```
out/<model_id 슬래시를 __로 치환>/
  prefill.csv, decode.csv          # 주요 operator 표 — latency 관점, 반복 레이어 접힘 (01-main.md §6.1)
                                    # op_id | block_type | repeat | layers | h1 h2 h3(계층) |
                                    # op_type | input_shape | weight_shape | output_shape | ...
                                    # block_type=attn+FFN/MLA+MoE/SSM…, repeat=×반복수, layers=인덱스
  prefill.jsonl, decode.jsonl      # 주요 operator 표의 JSONL (csv와 동일 컬럼)
  structure.yaml                   # 모델 구조 요약 (공통 심볼, 01-main.md §10~11)
  model_summary.md                 # 요약 정보 카드 + 검증 로그 + 추출 방법 + 참고 소스 (01-main.md §11.2)
  full/
    prefill.csv, decode.csv        # 전체 operator 표 (모든 aten 프리미티브, 동일 컬럼)
    prefill.trace.raw.jsonl, decode.trace.raw.jsonl   # 각 행의 원시 aten 근거
    provenance.json                # revision hash, 버전, 적용된 조치 이력(Tier 0~3)
    report.md                      # 검증 체크리스트 결과 (01-main.md §9)
```
# 최상위엔 핵심 6개 파일(주요 op csv·jsonl ×2, structure.yaml, model_summary.md)만, 나머진 full/.
# 주요 표는 full/ 전체 표에서 파생(view/plumbing 제거, norm 롤업, 의존관계 그래프 축약; §6.1).
# shape/list 컬럼(input_shape, depends_on 등): CSV는 읽기용이라 심볼을 따옴표 없이 bare로 낸다
#   (예: [[V, d_model], [B, T]]). 프로그램으로 파싱할 땐 .jsonl을 쓴다 — 이쪽은 순수 JSON이다.
# 별도 .graph.json은 없다(depends_on이 각 행에 있어 그래프 복원 가능), report.json도 없다
# (같은 결과가 model_summary.md 검증 로그에도 있고 report.md는 줄 단위 파싱 가능).

가장 먼저 볼 파일:
- **"이 모델 구조가 뭐야?"** → `structure.yaml` 또는 `model_summary.md`
- **"latency에 영향 주는 주요 연산이 뭐야?"** → `<phase>.csv` (주요 operator 표)
- **"이 추출이 맞게 됐어?"** → `full/report.md`
- **"이 op은 정확히 뭘 하는 거야, 어디서 왔어?"** → `full/<phase>.csv` + `full/<phase>.trace.raw.jsonl`
