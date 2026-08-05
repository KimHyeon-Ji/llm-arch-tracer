"""자유 평가(free-form LLM review)용 리뷰 패킷 생성기.

왜 필요한가
-----------
`verify_all.py`는 **규칙 기반**이다. 규칙은 이미 겪은 오류만 인코딩하므로, 본 적 없는
라벨 오류는 원리적으로 못 잡는다. 실제로 이 프로젝트에서 발견된 심각한 오류는 전부
자유 평가(외부 LLM 리뷰, 사람의 적대적 감사)에서 나왔고 게이트는 하나도 못 잡았다.

그런데 "그냥 봐달라"고 던지면 2026-07-29 외부 리뷰처럼 된다 — **관찰은 대체로 옳았지만
근본 원인 진단은 절반이 틀렸고, 인용한 op_id는 재트레이싱으로 전부 어긋나 있었다.**
그래서 자유 평가를 재현 가능한 절차로 만든다:

  1. 리뷰어에게 **자기완결적 패킷**을 준다 (요약 + 심볼표 + 대표 트레이스 표본 + 이미
     알려진 한계). 저장소를 뒤지게 하지 않는다.
  2. 출력 형식을 강제해 **관찰과 가설을 분리**시키고, 모든 주장에 재현 가능한 근거를
     요구한다. op_id는 재트레이싱하면 바뀌므로 근거로 인정하지 않는다.
  3. 받은 주장은 **하나도 그대로 믿지 않고** 코드/트레이스로 재현한 뒤에만 반영한다.

사용법
------
    .venv\\Scripts\\python.exe develop\\make_review_packet.py <model-dir-name> [-o out.md]
    .venv\\Scripts\\python.exe develop\\make_review_packet.py --all -o develop/review/
"""
import argparse
import json
import os
import re
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
MODELS = os.path.join(PROJ, "models")

# 이미 조사해서 "자동 판별 불가"로 결론 난 것들. 패킷에 실어 보내 리뷰어가 같은 걸 다시
# 보고하느라 시간을 쓰지 않게 한다(외부 리뷰 때 이미 정정된 항목이 다시 올라왔다).
KNOWN_LIMITS = """\
- **값이 같은 서로 다른 개념은 자동 판별 불가.** 예: gpt-oss는 expert 블록 **안**에서
  `d_model`=`d_moe`=2880, Zamba2는 `n_h*d_head`=`2*d_model`=4096. module_path만으로는 어느
  개념인지 못 가른다. 이건 이미 `rules/structures/`에 명시해 뒀으므로 다시 보고하지 않아도 된다.
- **이미 고쳐서 다시 보고하지 않아도 되는 것**(2026-07-30~31):
  `post_attention_layernorm`의 잔차폭(→ `d_model`), MoE 라우터 입력폭(→ `d_model`),
  gpt-oss sliding 레이어의 KV 상한(→ `w_local`)과 attention sink 표기(→ `n_sink`),
  Llama-3.1-405B의 토큰축(재추적으로 T 충돌 해소), DeepSeek-V4 압축기의 블록 축(→ `m_csa`/`m_hca`),
  한 shape에 `n_h`와 `n_kv`가 동시에 나오던 문제(게이트가 자동 검사한다).
- **`d_model` ↔ `n_h*d_head`가 같은 텐서에 다르게 붙는 경우**는 표준 트랜스포머에서 두 값이
  정의상 같기 때문이며, 둘 다 참인 이름이다. 강제 통일하면 오히려 정보가 사라진다.
- **Qwen3-Next의 3,7,9,11… 같은 작은 정수**는 DeltaNet 청크 스캔의 언롤된 루프 경계다.
  아키텍처 상수가 아니므로 심볼을 붙이면 거짓이 된다.
"""

REVIEW_TASK = """\
## 리뷰 요청

당신은 이 산출물이 **실제 모델 아키텍처를 정확히 기술하는지** 판정해야 합니다.
규칙 체크리스트는 이미 전부 통과한 상태입니다. 그러니 규칙이 못 잡는 것을 찾아주세요.

### 반드시 대조할 것
1. 해당 모델의 **공식 HF modeling 코드**와 config 클래스
2. **논문 / 기술 리포트**, 벤더 공식 블로그
3. **vLLM · SGLang 등 독립 서빙 구현**의 같은 모델 코드·해설
4. 신뢰도 높은 아키텍처 정리 자료

### 특히 봐야 할 것
- 심볼 이름이 **그 위치에서 실제로 의미하는 것과 맞는가**
  (값이 맞아도 개념이 틀릴 수 있음 — 이게 지금까지 나온 오류의 거의 전부였다)
- attention 계열 판정(MHA/GQA/MQA/MLA/…)이 실제 구현과 맞는가.
  **config 필드를 그대로 믿지 말 것** — 필드가 있어도 실제 동작이 다를 수 있다
  (Falcon은 `num_kv_heads=71`이지만 `multi_query=True`라 실제 KV head는 1개였다)
- KV cache 계산의 **전제**가 맞는가 (어느 레이어가 캐시를 갖는지, K와 V가 별개인지)
- config에 없는데 코드에 하드코딩된 구조가 누락되지 않았는가
  (Llama-4는 shared expert 개수 필드가 없고 코드에 1개로 고정돼 있다)
- 이 아키텍처의 **핵심 특징 중 산출물에 아예 안 나타난 것**이 있는가
- **decode 표(5-2)를 반드시 보세요.** prefill에는 없는 축이 거기 있습니다 —
  sliding 레이어의 KV 상한, 캐시 길이, attention sink가 붙는 score 폭.
  실제로 이 표가 패킷에 없던 동안 gpt-oss의 sliding 컨텍스트 오라벨이 그대로 남아 있었습니다.
- **모듈 이름이 "무엇을 계산하는가"가 아니라 "블록 안 어디인가"를 뜻하는 곳**을 의심하세요.
  지금까지 나온 오류의 다수가 여기서 나왔습니다 — `post_attention_layernorm`은 attention이
  아니라 그 뒤의 잔차 정규화이고, `mlp.router`는 FFN 내부가 아니라 잔차를 읽는 라우터입니다.
- **같은 (모듈, op)인데 shape 표기가 갈리는 줄**을 찾으세요. 표본은 그 축을 일부러 접지
  않았습니다 — 라벨 오류는 정의상 거기서 드러납니다.

### 출력 형식 (반드시 지킬 것)

각 지적은 아래 표 형태로. **관찰과 가설을 반드시 분리**하세요.

| # | 관찰(사실) | 근거 | 내 가설(원인) | 확신도 | 검증 방법 |
|---|---|---|---|---|---|

- **관찰**: 패킷에서 직접 인용. "X라고 적혀 있다"
- **근거**: 공식 소스의 **파일명 + 함수/클래스명 + 인용문**, 또는 URL.
  `op_id`는 근거로 쓰지 마세요 — 재트레이싱하면 번호가 바뀝니다.
  대신 `module_path`와 shape 내용으로 지목하세요.
- **내 가설**: 왜 그렇게 됐다고 보는지. **틀려도 됩니다. 다만 관찰과 섞지 마세요.**
- **확신도**: 확실 / 아마도 / 추측
- **검증 방법**: 우리가 이 주장을 어떻게 확인하면 되는지 (구체적으로)

확실하지 않으면 "확실"이라고 쓰지 마세요. **틀린 지적보다 놓친 지적이 낫습니다** —
틀린 지적을 검증하는 비용이 더 큽니다.
"""


def _load(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return f.read()


def _sample_trace(model_dir, phase="prefill", max_per_group=6):
    """대표 트레이스 표본: (모듈 leaf, op_type)마다 **서로 다른 shape 조합을 전부**.

    46,000행을 통째로 주면 리뷰어가 못 읽는다. 레이어 번호는 정규화해 접는다 — 라벨 오류는
    '어떤 종류의 op'에서 나지 특정 레이어 번호에서 나지 않기 때문이다.

    조합당 **1행**만 뽑던 것을 shape가 다르면 다 뽑도록 바꿨다(2026-07-31). 1행 방식은
    같은 (모듈, op) 안에서 shape가 갈리는 경우를 통째로 가렸고, 그게 실제로 리뷰어가
    Llama-3.1-405B의 `[B,n_kv,n_h/n_kv,n_h]`(head 크기축이 head 개수 이름에 뺏긴 것)을
    못 본 이유였다 — 그 shape는 prefill 트레이스에 2,772번 있었는데 패킷에는 한 번도
    안 나왔다. 라벨 오류는 정의상 '같은 op인데 shape 표기가 다른' 곳에서 나므로,
    바로 그 축을 접어버리면 안 된다. max_per_group은 폭주 방지용 상한이다."""
    raw = os.path.join(model_dir, "full", f"{phase}.trace.raw.jsonl")
    if not os.path.exists(raw):
        return []
    seen, out = {}, []
    for line in open(raw, encoding="utf-8"):
        r = json.loads(line)
        mp = r.get("module_path") or ""
        leaf = re.sub(r"\.\d+\.", ".N.", mp)          # 레이어 번호 정규화
        key = (leaf, r.get("op_type"))
        sig = json.dumps([r.get("input_shape"), r.get("weight_shape"), r.get("output_shape")],
                         ensure_ascii=False)
        bucket = seen.setdefault(key, set())
        if sig in bucket or len(bucket) >= max_per_group:
            continue
        bucket.add(sig)
        out.append({
            "module": leaf,
            "op": r.get("op_type"),
            "in": r.get("input_shape"),
            "w": r.get("weight_shape"),
            "out": r.get("output_shape"),
        })
    return out


def build(name: str, model_dir: str | None = None) -> str:
    """`model_dir` overrides the models/ lookup so a packet can be built for a run that has not
    been promoted yet (src/run.py writes to develop/out/). Layer ③ is only useful if it happens
    on EVERY run -- see write_packet()."""
    d = model_dir or os.path.join(MODELS, name)
    prov = json.loads(_load(os.path.join(d, "full", "provenance.json"), "{}"))
    struct = yaml.safe_load(_load(os.path.join(d, "structure.yaml"), "") or "") or {}
    summary = _load(os.path.join(d, "model_summary.md"), "(없음)")
    report = _load(os.path.join(d, "full", "report.md"), "(없음)")
    sample = _sample_trace(d, "prefill")
    # decode도 반드시 넣는다. 예전엔 prefill만 표본해서, **decode에만 존재하는 축은 리뷰어에게
    # 아예 보이지 않았다** — sliding 레이어의 KV 상한(w_local), 캐시 길이(T+1), attention sink가
    # 붙는 score 폭이 전부 그렇다. gpt-oss의 sliding 컨텍스트가 `E*k`로 오라벨된 채 남아 있었던
    # 이유가 이것이다(자기점검 2026-07-31에 발견).
    sample_dec = _sample_trace(d, "decode")

    syms = struct.get("symbols", {})
    sym_lines = "\n".join(f"  {k:12s} = {v!r}" for k, v in syms.items())

    def _sh(v):
        """`[['B','n_h','T']]` 대신 `[B,n_h,T]`. 표본이 패킷의 대부분을 차지하므로 따옴표와
        중첩 대괄호를 걷어내면 절반 이하로 줄어든다 — 정보는 그대로다."""
        if v is None:
            return "-"
        if v and isinstance(v[0], list):
            return "*".join("[" + ",".join(str(x) for x in s) + "]" for s in v) or "-"
        return "[" + ",".join(str(x) for x in v) + "]"

    def _fmt(rows):
        out = []
        for s in rows:
            w = f" w={_sh(s['w'])}" if s["w"] else ""
            out.append(
                f"  {s['module']:50s} {str(s['op']):16s} {_sh(s['in'])} ->{w} {_sh(s['out'])}")
        return chr(10).join(out) or "  (없음)"

    trace_lines_pre, trace_lines_dec = _fmt(sample), _fmt(sample_dec)

    return f"""# 리뷰 패킷 — {prov.get('model_id', name)}

> 이 문서는 **자기완결적**입니다. 판단에 필요한 것은 전부 아래에 있습니다.
> revision `{prov.get('revision_resolved', '?')}` / 트레이스 seq_len(T) = {prov.get('seq_len_used', '?')}
> 라이브러리: torch {prov.get('torch_version', '?')}, transformers {prov.get('transformers_version', '?')}

## 1. 이 산출물이 무엇인가

Hugging Face의 **공식 config + modeling 코드를 meta device에서 실제로 forward 실행**하고,
그 실행을 PyTorch dispatch(ATen) 레벨에서 가로채 op·shape·의존관계를 기록한 것입니다.
가중치는 없지만(shape 계산에 불필요) 연산 그래프는 실제로 실행된 것이며, 값을 지어내지
않습니다. shape은 아키텍처 심볼(`B, T, d_model, n_h, …`)로 렌더됩니다.

**따라서 트레이스 자체(어떤 op이 어떤 크기로 도는가)는 관측값이고, 검토 대상은
"그 축에 붙은 이름이 맞는가"입니다.**

## 2. 심볼표 (이 모델에서 각 이름이 갖는 값)

```
{sym_lines}
```

## 3. 모델 요약 산출물

{summary}

## 4. 검증 체크리스트 결과

```
{report}
```

## 5. 대표 트레이스 표본

(모듈×op 조합마다 **서로 다른 shape는 전부**. 레이어 번호는 `.N.`으로 정규화.
같은 op인데 shape 표기가 갈리는 곳이 곧 라벨 오류가 사는 곳이므로 그 축은 접지 않습니다.)

### 5-1. prefill

```
{trace_lines_pre}
```

### 5-2. decode

**여기만 존재하는 축이 있습니다** — sliding 레이어의 KV 상한(`w_local`), 캐시 길이(`T+1`),
attention sink가 붙는 score 폭. prefill에는 나타나지 않으므로 위 표만 보면 놓칩니다.

```
{trace_lines_dec}
```

## 6. 이미 알려진 한계 — 다시 보고하지 않아도 됨

{KNOWN_LIMITS}

---

{REVIEW_TASK}
"""


def write_packet(model_dir: str, out_path: str | None = None) -> str:
    """Generate the layer-③ review packet for one model directory and write it INSIDE that
    directory (`full/review.md`). Returns the path.

    Called automatically at the end of src/run.py and develop/regen_summaries.py. Layer ③ is
    the only one that has ever caught an unanticipated label bug -- the rule gate encodes
    failures we already met, so by construction it cannot find a new class. Leaving the packet
    as a manual step meant it went stale: the packets in develop/review/ were from 2026-07-31
    and did not cover the anchoring change at all. Generating it is cheap and offline; the
    reading still has to be done by a reviewer with source access, but the artifact is now
    always present and always current.
    """
    name = os.path.basename(os.path.normpath(model_dir))
    text = build(name, model_dir=model_dir)
    path = out_path or os.path.join(model_dir, "full", "review.md")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", help="models/ 아래 폴더명")
    ap.add_argument("--all", action="store_true", help="전체 모델")
    ap.add_argument("-o", "--out", help="출력 파일(단일) 또는 디렉터리(--all)")
    a = ap.parse_args()

    names = (sorted(n for n in os.listdir(MODELS) if os.path.isdir(os.path.join(MODELS, n)))
             if a.all else [a.model])
    if not names or names == [None]:
        ap.error("모델명을 주거나 --all 을 쓰세요")

    for n in names:
        text = build(n)
        if a.all:
            outdir = a.out or os.path.join(HERE, "review")
            os.makedirs(outdir, exist_ok=True)
            p = os.path.join(outdir, f"{n}.review.md")
        else:
            p = a.out or os.path.join(HERE, "review", f"{n}.review.md")
            os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"wrote {os.path.relpath(p, PROJ)}  ({len(text):,} chars)")


if __name__ == "__main__":
    sys.exit(main())
