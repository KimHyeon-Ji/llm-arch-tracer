"""Run the layer-3 free-form review automatically: Claude reads the packet AND the real sources.

`review_ledger.py` records whether the review happened; this module is what makes it happen
without a human in the loop. It hands Claude three things -- the review packet (what we produced),
the research agenda (which axes are unresolved and which URLs answer them), and the web_fetch tool
(so it can open the modeling source itself) -- and asks for findings in a fixed schema.

Why the model fetches rather than us: the agenda already names the authoritative URL for each gap
(HF modeling/config file, built from `model_type`), and `web_fetch` only retrieves URLs already
present in the conversation. So the source list stays the project's Tier 2 ladder, not the model's
browsing whim.

NOT THE PRIMARY PATH. The review normally runs through the `review-labels` skill
(`.claude/skills/review-labels/SKILL.md`), where whatever Claude Code session is open reads the
sources with its own WebFetch -- no SDK, no API key, no unattended-agent design to maintain.
This module exists for the unattended case only: a scheduled or CI run with nobody at the
keyboard. Both paths write the same `review_findings.md` and record to the same ledger, so a
reader cannot tell which produced a finding, and neither can drift from the other.

DEGRADES VISIBLY. With no SDK and no credentials this returns a `skipped` result that the caller
reports as "manual review required" -- it must never look like a review that found nothing. That
distinction is the whole point of the ledger.
"""
import json
import os

MODEL = "claude-opus-5"

# One finding per unresolved axis. `verdict` is what the sources actually said; `evidence` must
# quote or cite them, because a finding without a source is exactly the guess this layer exists
# to replace.
_SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "module": {"type": "string"},
                    "axis": {"type": "string",
                             "description": "The rendered shape and which axis is in question"},
                    "current_label": {"type": "string"},
                    "verdict": {"type": "string",
                                "enum": ["current_label_correct", "should_be_renamed",
                                         "no_name_exists", "undetermined"]},
                    "proposed_label": {"type": "string",
                                       "description": "Empty unless verdict is should_be_renamed"},
                    "evidence": {"type": "string",
                                 "description": "Quote or cite the source that settles it"},
                    "source_url": {"type": "string"},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                },
                "required": ["module", "axis", "current_label", "verdict", "proposed_label",
                             "evidence", "source_url", "confidence"],
                "additionalProperties": False,
            },
        },
        "summary": {"type": "string"},
    },
    "required": ["findings", "summary"],
    "additionalProperties": False,
}

_SYSTEM = """You verify tensor-axis labels produced by a tracer against the model's real source.

The tracer runs a HuggingFace model on meta device, records every ATen op, and names each shape
axis from a rule set. Some axes it cannot settle -- usually because two config values are equal at
the traced sequence length, so the number alone cannot say which name is right.

Your job is to settle those, and only those, from the sources. Rules:

- Open the modeling source with web_fetch before answering. A verdict with no source is worthless
  here; that is the guess this step replaces.
- Quote the line that settles it. Name the class and method.
- `no_name_exists` is a real answer. Loop indices, unrolled scan offsets, and operand counts have
  no config field behind them, and inventing a name for one is worse than leaving the integer.
- `undetermined` is also a real answer. Say what you looked at and what was missing.
- Do not restate labels the tracer already got right unless the agenda asks about them."""


def available() -> tuple[bool, str]:
    """(can we run, why not). Never raises -- the caller reports the reason."""
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False, "anthropic SDK 미설치 (.venv\\Scripts\\pip install anthropic)"
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
            or os.path.isdir(os.path.expanduser("~/.config/anthropic"))):
        return False, "인증 없음 (ANTHROPIC_API_KEY 설정 또는 `ant auth login`)"
    return True, ""


def _read(path: str, limit: int) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8") as f:
        return f.read()[:limit]


def review(model_dir: str, model_id: str) -> dict:
    """{'status': 'ok'|'skipped'|'error', 'findings': [...], 'summary': str, 'reason': str}"""
    ok, why = available()
    if not ok:
        return {"status": "skipped", "reason": why, "findings": [], "summary": ""}

    agenda = _read(os.path.join(model_dir, "research_agenda.md"), 20000)
    packet = _read(os.path.join(model_dir, "full", "review.md"), 60000)
    if not agenda.strip():
        return {"status": "ok", "reason": "조사 안건 없음", "findings": [],
                "summary": "이 모델은 미해결 축이 없다."}

    import anthropic
    client = anthropic.Anthropic()
    prompt = (f"# 대상 모델\n{model_id}\n\n"
              f"# 조사 안건 (미해결 축과 확인할 소스)\n{agenda}\n\n"
              f"# 리뷰 패킷 (이 모델의 심볼표·요약·트레이스 표본)\n{packet}\n\n"
              "안건의 각 항목을 소스로 확인하고 findings 로 답하라.")
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=_SYSTEM,
            thinking={"type": "adaptive"},
            output_config={"effort": "high",
                           "format": {"type": "json_schema", "schema": _SCHEMA}},
            tools=[{"type": "web_search_20260209", "name": "web_search"},
                   {"type": "web_fetch_20260209", "name": "web_fetch"}],
            messages=[{"role": "user", "content": prompt}],
        )
        # A server-tool turn can hit its iteration cap; re-send to resume (no extra user text).
        rounds = 0
        while resp.stop_reason == "pause_turn" and rounds < 5:
            rounds += 1
            resp = client.messages.create(
                model=MODEL, max_tokens=16000, system=_SYSTEM,
                thinking={"type": "adaptive"},
                output_config={"effort": "high",
                               "format": {"type": "json_schema", "schema": _SCHEMA}},
                tools=[{"type": "web_search_20260209", "name": "web_search"},
                       {"type": "web_fetch_20260209", "name": "web_fetch"}],
                messages=[{"role": "user", "content": prompt},
                          {"role": "assistant", "content": resp.content}],
            )
        # Check the stop reason BEFORE reading content -- a refusal carries no answer block.
        if resp.stop_reason == "refusal":
            return {"status": "error", "reason": "모델이 요청을 거부함", "findings": [],
                    "summary": ""}
        text = next((b.text for b in resp.content if b.type == "text"), "")
        data = json.loads(text)
        return {"status": "ok", "reason": "", "findings": data.get("findings") or [],
                "summary": data.get("summary") or ""}
    except Exception as e:                       # noqa: BLE001 -- never lose a run over a review
        return {"status": "error", "reason": f"{type(e).__name__}: {str(e)[:160]}",
                "findings": [], "summary": ""}


def write_report(model_dir: str, result: dict) -> str:
    """Render the findings next to the agenda so a human can check them before anything is acted on."""
    L = ["# ③ 자유 평가 결과 (자동)", ""]
    if result.get("status") != "ok":
        L += [f"**수행되지 않음 — {result.get('reason')}**", "",
              "패킷과 안건은 생성돼 있으므로 사람이 직접 검토할 수 있다. "
              "자동 수행하려면 `anthropic` SDK 설치 + 인증 후 재생성하면 된다.", ""]
    else:
        L += [result.get("summary") or "", "",
              "> 아래는 LLM 이 소스를 대조해 낸 판단이다. **재현 전에는 반영하지 않는다** — "
              "근거 URL 을 직접 열어 확인한 뒤 규칙으로 승격한다.", ""]
        rows = result.get("findings") or []
        if rows:
            L += ["| 모듈 | 축 | 현재 라벨 | 판정 | 제안 | 확신 | 근거 |",
                  "|---|---|---|---|---|---|---|"]
            for f in rows:
                ev = (f.get("evidence") or "").replace("|", "/").replace("\n", " ")[:120]
                L.append(f"| `{f.get('module','')}` | `{f.get('axis','')}` | "
                         f"`{f.get('current_label','')}` | {f.get('verdict','')} | "
                         f"`{f.get('proposed_label','') or '-'}` | {f.get('confidence','')} | "
                         f"{ev} ({f.get('source_url','')}) |")
            L.append("")
        else:
            L += ["확인 결과 지적 사항 없음.", ""]
    path = os.path.join(model_dir, "review_findings.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    return path
