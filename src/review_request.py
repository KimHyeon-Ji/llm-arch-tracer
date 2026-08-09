"""Write the hand-off the label review starts from: `models/<model>/review_request.md`.

The pipeline is deterministic Python and stops at the point where a judgement is required. It
does not call an LLM -- doing that from inside `src/` would need an API key and would tie the
tool to one vendor. Instead it finishes by assembling everything a reviewer needs and naming
what is left undecided, so the review can be run afterwards by any LLM (or by a person) with
no preparation: see `review/`.

What goes in the request is only what the rules could NOT settle. A model whose axes are all
grounded gets a request that says so -- an empty agenda is a result, not a gap.
"""
import os
import re

import source_check


def _fmt_axis(a: dict) -> str:
    where = a.get("module") or a.get("op") or ""
    return f"| `{a.get('shape', '')}` | {a.get('why', '')} | {where} |"


def collect(structure: dict, sc_res: dict, fields: dict) -> dict:
    """The open questions, grouped by what kind of answer each needs."""
    q = {"unregistered": [], "square": [], "alias": [], "heuristic": []}
    q["unregistered"] = list(structure.get("unregistered_fields") or [])
    q["square"] = list(sc_res.get("square_unconfirmed") or [])
    q["alias"] = [f"{s} ← {f}" for s, f in (sc_res.get("alias_gaps") or [])]
    # `heuristic` is a count; the axes themselves are in heuristic_examples, which is what a
    # reviewer can actually open a source file against. Fold the layer index out first: the same
    # module in 12 layers is ONE question about that module, and listing it 12 times both inflates
    # the open count and buries the other kinds of question under it.
    folded = {}
    for ex in ((structure.get("label_provenance") or {}).get("heuristic_examples") or []):
        mod = re.sub(r"\.\d+\.", ".*.", str(ex.get("module") or ""))
        key = (str(ex.get("label")), mod, str(ex.get("rule")))
        agg = folded.setdefault(key, {"axes": 0, "layers": 0})
        agg["axes"] += int(ex.get("axes") or 0)
        agg["layers"] += 1
    for (label, mod, rule), agg in folded.items():
        where = f"{mod} (레이어 {agg['layers']}개)" if agg["layers"] > 1 else mod
        q["heuristic"].append(f"`{label}` in `{where}` — {rule}, {agg['axes']}축")
    return q


def build(model_dir: str, model_id: str, model_type: str, structure: dict,
          sc_res: dict, fields: dict) -> str:
    q = collect(structure, sc_res, fields)
    mt = model_type or "(미확인)"
    open_n = sum(len(v) for v in q.values())

    L = [f"# 검토 의뢰서 — {model_id}", "",
         "파이썬 파이프라인이 규칙으로 결정할 수 있는 것을 전부 결정하고, **판단이 필요한 것만** "
         "여기 남겼다. 절차와 출력 형식은 `review/` 에 있다.", "",
         f"- transformers 모듈: `{mt}`",
         f"- 판단 필요: **{open_n}건**", ""]

    L += ["## 증거 — 이미 받아둔 실제 소스", ""]
    for kind, ok in (("modeling", sc_res.get("modeling_ok")), ("configuration", sc_res.get("config_ok"))):
        path = f"develop/sources/{kind}_{mt}.py"   # forward slashes: this file is a deliverable
        L.append(f"- `{path}` — {'있음, 이 파일을 열어서 판정한다' if ok else '**없음** (네트워크 불가 또는 transformers 본체에 없는 아키텍처)'}")
    L += ["", f"- 온라인 원본: https://github.com/huggingface/transformers/tree/main/src/transformers/models/{mt}",
          "", "그 밖의 재료: `full/review.md`(리뷰 패킷 — shape 별 실제 행 표본), "
          "`structure.yaml`(이 모델의 심볼 표), `full/<phase>.csv`(전체 operator 표).", ""]

    L += ["## 판단이 필요한 것", ""]
    if not open_n:
        L += ["없다. 이 모델의 축은 전부 등록된 규칙이 이름을 냈고, 소스 대조도 어긋난 곳이 없다.",
              "", "그래도 검토를 돌린다면 `full/review.md` 의 표본을 보고 규칙 자체가 틀리지 "
              "않았는지를 본다 — 그것이 규칙 게이트가 구조적으로 못 보는 부분이다.", ""]
    else:
        if q["alias"]:
            L += ["### 1. 이 config 필드가 정말 이 뜻인가", "",
                  "값은 로드된 config 에 있지만 이 모델의 config 클래스가 선언한 필드가 아니다 "
                  "(체크포인트 `config.json` 에서 온 값). 클래스가 뜻을 보증하지 않으므로 "
                  "modeling 소스에서 이 필드가 실제로 어떻게 쓰이는지 확인해야 한다.", ""]
            L += [f"- `{x}`" for x in q["alias"]] + [""]
        if q["square"]:
            L += ["### 2. 이 정사각 축이 정말 같은 이름 두 번인가", "",
                  "`[..., X, X]` 로 렌더됐는데, 그 이름이 읽은 config 필드에서 나온 정사각 "
                  "reshape 을 modeling 소스에서 찾지 못했다. 두 축 크기가 우연히 같은 것일 수 있다.", ""]
            L += [f"- `{x}`" for x in q["square"]] + [""]
        if q["unregistered"]:
            L += ["### 3. 이름 붙일 근거가 없는 config 필드", "",
                  "모듈 폭으로 쓰이는데 심볼 표에 등록돼 있지 않다. 소스에서 무엇인지 확인하고 "
                  "`rules/symbols.yaml` 에 등록하면 다음 실행부터 자동으로 잡힌다.", ""]
            L += [f"- `{x}`" for x in q["unregistered"][:40]] + [""]
        if q["heuristic"]:
            L += ["### 4. 규칙 없이 산술로 지은 이름", "",
                  "값이 맞아떨어져서 붙인 이름이다. 산술적으로 참이어도 틀린 이름일 수 있으므로 "
                  "(예: RoPE 절반 차원) 소스에서 확인이 필요하다.", ""]
            L += [f"- {x}" for x in q["heuristic"][:40]] + [""]

    L += ["## 기계적으로 이미 확인된 것 — 다시 묻지 말 것", ""]
    gaps_n = len(sc_res.get("alias_gaps") or [])
    conf = sc_res.get("square_confirmed") or []
    L += [f"- **심볼이 읽은 config 필드**: {'전부 이 모델의 config 클래스(또는 상속/프로퍼티/getattr 기본값)에 존재한다' if not gaps_n else f'{gaps_n}건이 클래스 선언 밖 — 위 1절 참고'}",
          f"- **정사각 축**: " + (", ".join(f"`{lab}` ← 소스의 `{ident}` ← `{fld}`" for lab, ident, fld in conf)
                                  if conf else "소스에서 정사각 생성/reshape 과 대응이 확인된 축 없음"),
          f"- **모듈이 읽는 config 속성**: `__init__` 에서 config 를 읽는 클래스 {sc_res.get('module_reads', 0)}개를 소스에서 확인했다. "
          "그 목록이 각 모듈의 폭이 가질 수 있는 이름의 전부다.", ""]

    L += ["## 이 의뢰서를 처리하는 법", "",
          "`review/prompt.md` 를 LLM 에 넘기고 이 모델을 지정한다. 판정 4종과 근거 요건, "
          "결과를 어디에 어떤 형식으로 쓰는지는 전부 `review/` 안에 있다.", ""]

    path = os.path.join(model_dir, "review_request.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    return path
