"""Turn the axes the labeller could not settle into a research agenda the reader can act on.

The pieces were all here already and none of them met: `rules/structures/` is where a finding gets
written down, C17 asserts that writing happened, `label_provenance` says which axes were guessed,
and `02-new-module-handling.md` ranks the sources to consult. What was missing is the join --
"THIS module has THIS ambiguity, here is the question, here is which source answers it, here is
the file the answer goes in". Without it, deciding that research is needed was a judgement someone
made by hand each time, which does not survive the next new model.

Everything here is derived from published artifacts (the trace, the concrete sidecar,
structure.yaml), so it runs in regeneration with no re-trace.
"""
import collections
import json
import os

# The Tier 2 ladder from 02-new-module-handling.md, in the same order. Kept here so the generated
# agenda cites the project's own policy rather than an ad-hoc list -- if the doc changes, this
# should change with it.
_SOURCES = [
    ("실행 중인 modeling 소스의 주석·변수명·docstring", "{modeling_url}"),
    ("같은 저장소 config 클래스의 docstring", "{config_url}"),
    ("독립 서빙 구현 (vLLM / SGLang / TensorRT-LLM)", "https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/models"),
    ("저장소 README / 공식 model card", "https://huggingface.co/{model_id}"),
    ("model card 가 링크한 논문·기술 리포트", ""),
    ("아키텍처 갤러리 (2차 자료 — 원 소스로 재확인)", "https://sebastianraschka.com/llm-architecture-gallery/"),
]

_HF_BLOB = ("https://github.com/huggingface/transformers/blob/main/src/transformers/models/"
            "{mt}/modeling_{mt}.py")
_HF_CFG = ("https://github.com/huggingface/transformers/blob/main/src/transformers/models/"
           "{mt}/configuration_{mt}.py")


def _shapes(row, field):
    for sh in (row.get(field) or []):
        yield sh if isinstance(sh, list) else [sh]


def _collect(model_dir: str) -> dict:
    """{finding kind: [{module, detail, axes}]} from this model's own published trace."""
    raw = os.path.join(model_dir, "full", "prefill.trace.raw.jsonl")
    con = os.path.join(model_dir, "full", "prefill.shapes.concrete.jsonl")
    if not (os.path.exists(raw) and os.path.exists(con)):
        return {}
    conc = {}
    for line in open(con, encoding="utf-8"):
        r = json.loads(line)
        conc[r["op_id"]] = r

    import anchors as _anchors
    import build_table as _bt

    dup, bare, mismatch = collections.Counter(), collections.Counter(), collections.Counter()
    for line in open(raw, encoding="utf-8"):
        row = json.loads(line)
        key = _anchors.module_key(row.get("module_path")) or "(모듈 밖)"
        c = conc.get(row.get("op_id"))

        for field in ("input_shape", "output_shape"):
            for sym, cshape in zip(_shapes(row, field),
                                   _shapes(c, field) if c else []):
                labels = [str(x) for x in sym]
                counts = collections.Counter(x for x in labels if x not in ("B", "1"))
                # A SQUARE MATRIX legitimately names its axis twice: the trailing pair of an
                # attention score matrix `[B,n_h,T,T]`, DeepSeek-V4's mHC stream-mixing matrix
                # `[B,T,n_hc,n_hc]` (verified in modeling_deepseek_v4.py --
                # `comb_w.view(*comb_w.shape[:-1], hc, hc)`, one axis per source/destination
                # residual stream), and the within-chunk matrices of the SSM scans. Flagging
                # those sent 28,496 already-documented DeepSeek axes to "needs code research",
                # which is how a triage report loses its meaning. Only a repeat that is NOT the
                # trailing pair is evidence of a collision.
                square = len(labels) >= 2 and labels[-1] == labels[-2]
                for name, n in counts.items():
                    if n <= 1 or name == "T":
                        continue
                    if square and name == labels[-1] and n == 2:
                        continue
                    dup[(key, name, tuple(labels), tuple(cshape or ()))] += 1
                for lab, cv in zip(labels, cshape or []):
                    if lab.isdigit() and int(lab) > 1:
                        bare[(key, lab)] += 1

        if c:
            probe = dict(row)
            probe["input_shape"] = c.get("input_shape") or []
            probe["output_shape"] = c.get("output_shape") or []
            for _i, cur, derived in _bt.reshape_disagreements(probe, row):
                mismatch[(key, cur, derived)] += 1

    return {"dup": dup, "bare": bare, "mismatch": mismatch}


def build(model_dir: str, model_id: str, model_type: str | None,
          structure: dict | None = None) -> str | None:
    """Write `research_agenda.md` and return its path, or None when nothing is open."""
    found = _collect(model_dir)
    if not found:
        return None
    dup, bare, mismatch = found["dup"], found["bare"], found["mismatch"]
    lp = (structure or {}).get("label_provenance") or {}
    heur = [e for e in (lp.get("heuristic_examples") or []) if e.get("axes", 0) >= 8]
    unreg = (structure or {}).get("unregistered_fields") or []
    if not (dup or mismatch or heur or unreg or bare):
        return None

    mt = (model_type or "").strip()
    subs = {
        "modeling_url": _HF_BLOB.format(mt=mt) if mt else "(model_type 미확인)",
        "config_url": _HF_CFG.format(mt=mt) if mt else "(model_type 미확인)",
        "model_id": model_id,
    }

    L = [f"# 조사 안건 — {model_id}", ""]
    L.append("라벨러가 **혼자 결정하지 못한 축**만 모았다. 각 항목은 `02-new-module-handling.md` "
             "Tier 2 절차로 확인한 뒤 근거와 함께 등록하면 다음 실행부터 자동으로 잡힌다. "
             "비어 있으면 조사할 것이 없다는 뜻이다.")
    L.append("")
    L.append("> 값이 같아서 못 가리는 것과 규칙이 없어서 못 붙이는 것은 다르다. 앞의 것은 "
             "**코드를 읽어야** 풀리고, 뒤의 것은 **등록만** 하면 된다. 아래에서 구분해 둔다.")
    L.append("")

    # Triage first. Almost every model has SOMETHING open, so a bare list is not a signal --
    # what a reader needs is whether this model requires reading its modeling source at all.
    # A rank-2 shape whose two labels are equal is the SQUARE PROJECTION case: a weight of a
    # module where in_features == out_features, so only axis order distinguishes them and the
    # cause is already characterised (01-main.md §10.2). It is a known tool gap, not a question
    # about this architecture, so it must not push a plain Llama into "read the source".
    known = sum(n for (_m, _name, labels, _c), n in dup.items()
                if len(labels) == 2 and len(set(labels)) == 1)
    need_code = sum(dup.values()) - known + sum(mismatch.values())
    need_reg = sum(e.get("axes", 0) for e in heur) + sum(u.get("modules", 0) for u in unreg)
    verdict = ("코드 조사 필요" if need_code else
               ("등록만 하면 됨" if need_reg else "조사 불필요 — 정수로 남은 축만 있음"))
    L += ["## 판정", "",
          f"**{verdict}**", "",
          "| 성격 | 조치 | 해당 축 |",
          "|---|---|---:|",
          f"| 값이 겹쳐 어느 쪽인지 미결 | modeling 소스를 읽어야 함 | {need_code:,} |",
          f"| 규칙이 없어 이름을 못 붙임 | 확인 후 규칙 등록 | {need_reg:,} |",
          f"| 이름이 존재하지 않음 | 정수로 두는 것이 정직 | {sum(bare.values()):,} |",
          f"| 정사각 투영 (알려진 패턴) | 조사 불필요 — 축 순서만의 문제 | {known:,} |",
          ""]

    if dup:
        L += ["## 1. 한 shape 에 같은 이름이 두 번 — 어느 한쪽은 다른 이름이다", ""]
        L.append("텐서의 두 축이 같은 이름을 받았다. 두 축의 크기가 우연히 같아 값으로는 못 가린다. "
                 "**어느 축이 무엇인지는 모델 코드를 읽어야 안다.**")
        L.append("")
        L.append("| 모듈 | 중복된 이름 | 렌더된 shape | 실제 크기 | 축 수 |")
        L.append("|---|---|---|---|---:|")
        for (mod, name, labels, cshape), n in dup.most_common(10):
            L.append(f"| `{mod}` | `{name}` | `[{', '.join(labels)}]` | "
                     f"`[{', '.join(str(x) for x in cshape)}]` | {n} |")
        L.append("")

    if mismatch:
        L += ["## 2. reshape 자기 유도와 라벨이 불일치 — 같은 텐서에 설명이 둘", ""]
        L.append("reshape 는 자기 입력 축에서 출력 축을 유도할 수 있다. 그 유도와 붙어 있는 이름이 "
                 "다르면 둘 중 하나가 틀렸다. **검사는 어느 쪽이 틀렸는지는 말해주지 않는다.**")
        L.append("")
        L.append("| 모듈 | 현재 라벨 | 유도된 이름 | 축 수 |")
        L.append("|---|---|---|---:|")
        for (mod, cur, derived), n in mismatch.most_common(10):
            L.append(f"| `{mod}` | `{cur}` | `{derived}` | {n} |")
        L.append("")

    if heur:
        L += ["## 3. 규칙 없이 산술로 지은 이름 — 등록하면 해결된다", ""]
        L.append("값은 맞지만 근거가 규칙이 아니라 산술이다. 이번 트레이스의 seq_len 에서만 참일 수 "
                 "있으므로, 확인 후 `rules/derived_dims.yaml` 에 **식과 출처**를 등록한다.")
        L.append("")
        L.append("| 모듈 | 붙은 이름 | 방식 | 축 수 |")
        L.append("|---|---|---|---:|")
        for e in heur[:10]:
            L.append(f"| `{e['module'] or '(모듈 밖)'}` | `{e['label']}` | {e['rule']} | {e['axes']} |")
        L.append("")

    if unreg:
        L += ["## 4. 미등록 config 필드 — 이름을 붙일 근거 자체가 없다", ""]
        L.append("이 아키텍처가 실제로 쓰는 config 필드인데 `rules/symbols.yaml` 이 모른다. "
                 "역할을 확인해 `aliases` 또는 `derived_dims.yaml` 에 등록한다.")
        L.append("")
        L.append("| 필드 | 값 | 쓰는 모듈 수 |")
        L.append("|---|---:|---:|")
        for u in unreg[:10]:
            L.append(f"| `{u['field']}` | {u['value']} | {u['modules']} |")
        L.append("")

    if bare:
        top = bare.most_common(6)
        L += ["## 5. 설명 없는 정수 (상위)", ""]
        L.append("이름이 붙지 않아 정수로 남은 축이다. 루프 인덱스나 데이터 의존 크기라면 "
                 "**정수로 두는 것이 정직하다** — 전부 이름을 붙일 대상은 아니다.")
        L.append("")
        L.append("| 모듈 | 값 | 축 수 |")
        L.append("|---|---:|---:|")
        for (mod, val), n in top:
            L.append(f"| `{mod}` | {val} | {n} |")
        L.append("")

    L += ["## 확인할 소스 (신뢰도 순 — 위에서 답이 나오면 아래는 생략)", ""]
    for i, (label, url) in enumerate(_SOURCES, 1):
        u = url.format(**subs) if url else ""
        L.append(f"{i}. {label}" + (f" — {u}" if u else ""))
    L.append("")
    L += ["## 답을 적는 곳", "",
          "| 알아낸 것 | 적는 파일 |",
          "|---|---|",
          "| 이 값은 이 config 필드다 | `rules/symbols.yaml` 의 `aliases` |",
          "| 이 값은 이런 식으로 계산된다 | `rules/derived_dims.yaml` (`expr` + `sym` + 출처 주석) |",
          "| 이 모듈은 이런 구조다 | `rules/structures/<범주>/<이름>.md` (C17 이 등재를 확인한다) |",
          "| 둘 중 어느 쪽인지 사람이 정해야 한다 | `02-new-module-handling.md` Tier 3 |",
          ""]
    path = os.path.join(model_dir, "research_agenda.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    return path
