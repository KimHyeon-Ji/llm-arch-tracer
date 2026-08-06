"""Tier 0 / Tier 1 handling -- see 02-new-module-handling.md. Tier 0 rules out
environment/access failures before anything is blamed on the model. Tier 1 applies
known remedies from rules/error_remedies.yaml and retries a bounded number of times.
Tier 2 (official-source research) and Tier 3 (human escalation) are NOT automatable
here -- see escalate.py and 02-new-module-handling.md for those."""
import os
import re
import yaml

_DEFAULT_REMEDIES_PATH = os.path.join(
    os.path.dirname(__file__), "..", "rules", "error_remedies.yaml"
)


class Tier0Error(RuntimeError):
    """Environment/access problem, not a modeling problem. Fix and re-run; do not
    escalate through Tier 1/2/3 for these."""


def tier0_check(exc: Exception):
    msg = str(exc)
    if re.search(r"\b(401|403)\b|gated repo|access.*restricted", msg, re.I):
        raise Tier0Error(
            "Gated repo / auth failure. Run `huggingface-cli login` and accept the "
            "model's license on its Hub page, then retry."
        ) from exc
    if re.search(r"KeyError: '.*'", msg) and "model_type" not in msg:
        pass  # fall through, handled by support_gate() in provenance.py normally
    if re.search(r"revision.*not found|no file named", msg, re.I):
        raise Tier0Error("Revision/commit hash or file path looks wrong.") from exc


def load_remedies(path: str = _DEFAULT_REMEDIES_PATH) -> list[dict]:
    if not os.path.exists(path):
        return _BUILTIN_REMEDIES
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or []
    return data or _BUILTIN_REMEDIES


# Seed table -- 02-new-module-handling.md Tier 1. Extend rules/error_remedies.yaml
# instead of editing this in place once Tier 3 starts producing permanent answers.
_BUILTIN_REMEDIES = [
    {"pattern": r"NotImplementedError|could not run .*meta", "remedy": "meta_to_fake"},
    {"pattern": r"Boolean value of Tensor", "remedy": "attn_sdpa"},
    {"pattern": r"data-dependent", "remedy": "attn_eager"},
    {"pattern": r"(k out of range|topk|index.*out of range)", "remedy": "bump_seq_len"},
    {"pattern": r"Expected inputs of BF16|Expected .*bfloat16", "remedy": "use_bf16"},
    {"pattern": r"[Cc]ache", "remedy": "rebuild_cache"},
]


class AdaptationExhausted(RuntimeError):
    pass


def trace_adaptive(ctx, phase: str, max_retries: int = 6):
    """ctx must implement: run_once(phase), switch_backend(name), set_attn(name),
    bump_seq(factor), rebuild_cache(). See run.py for the concrete context."""
    remedies = load_remedies()
    applied = []
    for attempt in range(max_retries):
        try:
            rows = ctx.run_once(phase)
            return rows, applied
        except Exception as e:  # noqa: BLE001 -- intentionally broad, this is the retry loop
            try:
                tier0_check(e)
            except Exception:
                raise  # Tier 0 errors are not retried -- fix environment and re-run
            msg = f"{type(e).__name__}: {e}"
            match = next((r for r in remedies if re.search(r["pattern"], msg, re.I)), None)
            if match is None:
                raise AdaptationExhausted(
                    f"Unclassified failure at attempt {attempt}: {msg}\n"
                    f"Already tried: {applied}\n"
                    f"-> Tier 2/3 escalation (02-new-module-handling.md). NOTE: escalate.py "
                    f"is not wired, so no packet is written -- this message is the handoff."
                ) from e
            getattr(ctx, match["remedy"])()
            applied.append({"attempt": attempt, "error": msg[:200], "remedy": match["remedy"]})
    raise AdaptationExhausted(f"Retries exhausted. Applied: {applied}")
