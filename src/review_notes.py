"""Carry the label review's verdicts into the published outputs.

The review is where a source is actually read, and some of what it establishes cannot be turned
into a rule: two config values coincide, the axis order of a fused `nn.Parameter` is not visible
in the trace, a width is only explained in a paper. Leaving those verdicts in a side document
means the model folder still shows the unexplained label with nothing next to it.

So the review writes ONE structured file, `models/<model>/review_findings.json`, and everything
else is generated from it:

  * `review_findings.md`  -- the readable record (rendered here, so the format never drifts)
  * `model_summary.md`    -- a "확인된 것과 남은 것" section, so a reader meets the caveat next to
                             the table it applies to rather than in a file they may never open

That keeps the procedure tool-neutral -- any LLM or person writes plain JSON -- while making the
finding travel with the artifact it describes.

Schema (every field a string unless noted):
    {"model_id": ..., "reviewed_on": "YYYY-MM-DD", "reviewer": ..., "angle": ..., "summary": ...,
     "findings": [{"module":..., "axis":..., "current_label":..., "verdict":...,
                   "proposed_label":..., "confidence": "high|medium|low", "evidence":...,
                   "status": "fixed|open"}]}

`verdict` is one of the four in review/01-procedure.md. `status` says whether the pipeline now
renders the corrected label -- an `open` finding is exactly what the summary must surface.
"""
import json
import os

VERDICTS = {
    "current_label_correct": "맞음",
    "should_be_renamed": "교정 필요",
    "no_name_exists": "이름 없음이 정답",
    "undetermined": "미확정",
}
PATH = "review_findings.json"


def load(model_dir: str) -> dict | None:
    p = os.path.join(model_dir, PATH)
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (ValueError, OSError):
        return None


def caveats(model_dir: str) -> list:
    """Findings a reader of the tables still needs to know -- anything not settled in the output.

    A `fixed` finding needs no warning: the table already says the right thing. What must travel
    is the residue -- a label the source contradicts but the rules cannot correct, and anything
    left undetermined.
    """
    data = load(model_dir) or {}
    out = []
    for f in data.get("findings") or []:
        if f.get("status") == "fixed":
            continue
        if f.get("verdict") in ("should_be_renamed", "undetermined"):
            out.append(f)
    return out


def render_md(model_dir: str, model_name: str) -> str | None:
    """(Re)write review_findings.md from the JSON so the two can never disagree."""
    data = load(model_dir)
    if not data:
        return None
    L = [f"# 라벨 검토 결과 — {data.get('model_id') or model_name}", "",
         f"- 검토일: {data.get('reviewed_on', '')}",
         f"- 검토자: {data.get('reviewer', '')}",
         f"- 본 것: {data.get('angle', '')}",
         f"- 요약: {data.get('summary', '')}", "",
         "> 이 파일은 `review_findings.json` 에서 생성된다 — 고칠 때는 JSON 을 고친다.", ""]
    finds = data.get("findings") or []
    if not finds:
        L += ["발견 없음. 빈 결과도 결과다 — 이 모델은 이번 각도에서 고칠 것이 없었다.", ""]
    for i, f in enumerate(finds, 1):
        state = "반영됨" if f.get("status") == "fixed" else "미반영"
        L += [f"## 발견 {i} — {VERDICTS.get(f.get('verdict'), f.get('verdict'))} ({state})", "",
              "| 항목 | 값 |", "|---|---|",
              f"| 모듈 | `{f.get('module', '')}` |",
              f"| 축 | {f.get('axis', '')} |",
              f"| 현재 라벨 | `{f.get('current_label', '')}` |",
              f"| 판정 | `{f.get('verdict', '')}` |",
              f"| 제안 라벨 | {('`' + f['proposed_label'] + '`') if f.get('proposed_label') else '—'} |",
              f"| 확신도 | {f.get('confidence', '')} |",
              f"| 산출물 반영 | {state} |", "",
              "**근거**", "", f.get("evidence", ""), ""]
    path = os.path.join(model_dir, "review_findings.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    return path


def summary_section(model_dir: str) -> list:
    """Markdown lines for model_summary.md, or [] when there is nothing to say."""
    data = load(model_dir)
    if not data:
        return ["## ③ 라벨 검토", "",
                "**아직 수행되지 않았다.** `review/prompt.md` 를 LLM 에 넘기면 이 자리에 결과가 "
                "들어온다 — 규칙 게이트가 구조적으로 못 보는 것(규칙 자체의 오류, 값이 겹쳐 "
                "구별 불가능한 축)이 여기서만 걸러진다.", ""]
    finds = data.get("findings") or []
    counts = {}
    for f in finds:
        counts[f.get("verdict")] = counts.get(f.get("verdict"), 0) + 1
    L = ["## ③ 라벨 검토 — 소스와 대조한 결과", "",
         f"{data.get('reviewed_on', '')} · {data.get('reviewer', '')}", "",
         data.get("summary", ""), ""]
    if counts:
        L += ["| 판정 | 건수 |", "|---|---|"]
        L += [f"| {VERDICTS.get(k, k)} | {v} |" for k, v in sorted(counts.items())]
        L.append("")
    cav = caveats(model_dir)
    if cav:
        L += ["### 이 표를 읽을 때 유의할 것", "",
              "소스를 열어 확인했지만 **산출물에 아직 반영되지 않은** 항목이다. 값이 겹쳐 규칙으로는 "
              "가릴 수 없거나, 근거를 더 찾아야 하는 것들이다.", "",
              "| 모듈 | 축 | 지금 렌더 | 소스가 말하는 것 | 근거 |", "|---|---|---|---|---|"]
        for f in cav:
            ev = " ".join((f.get("evidence") or "").split())
            if len(ev) > 200:
                ev = ev[:200] + " …"
            L.append(f"| `{f.get('module', '')}` | {f.get('axis', '')} | "
                     f"`{f.get('current_label', '')}` | "
                     f"{('`' + f['proposed_label'] + '`') if f.get('proposed_label') else '미확정'} | {ev} |")
        L.append("")
    L += ["전문은 `review_findings.md`(원본 `review_findings.json`), 대조에 쓴 실제 소스는 "
          "`develop/sources/` 에 있다.", ""]
    return L
