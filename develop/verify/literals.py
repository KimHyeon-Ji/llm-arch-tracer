"""Shared reader for references.yaml's `irreducible_literals` -- the single source of truth for
"this bare integer is not an architecture dimension, and here is why" (see references.yaml's own
header comment).

Three independent places used to each decide "is this bare literal OK" on their own: C17
(src/validate.py:c17_module_onboarding, "no expr" = "unresolved"), model_summary.md's own "미해결
유도 상수" table (src/summarize.py:render_model_summary, the same "no expr" test, computed
independently), and structure.yaml's stored literal_dims (whatever the first two left behind). A
value documented here could still show up as "unresolved" in either, because neither looked at
this file -- found 2026-08-26 when C17 alone went PASS but "미해결 유도 상수 1개" (a separate,
model_summary-derived metric in develop/verify_all.py) kept failing for the same value.

annotate() is the fix: call it once, right after find_literal_dims() returns, and every
downstream consumer sees a filled-in `expr` for a documented value and treats it exactly like a
real derived formula -- which is correct, since both mean "this is not an unexplained mystery
number." develop/verify_all.py's check_documented_literals (a fourth, independent check: it scans
the raw trace for undocumented bare digits rather than reading literal_dims) is untouched here --
it already reads `irreducible_literals` directly and was not part of this gap."""
import os

import yaml

_REFS_PATH = os.path.join(os.path.dirname(__file__), "references.yaml")


def _load_refs() -> dict:
    if not os.path.exists(_REFS_PATH):
        return {}
    with open(_REFS_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def documented_values(model_dir_name: str, refs: dict | None = None) -> dict[int, str]:
    """{value: value_note} for `irreducible_literals` entries covering this model (universal +
    model-specific). Only covers list-style entries (`values: [n, ...]`) -- freeform entries
    (`values: any_small_odd_or_scan`) describe an open-ended CLASS of small loop-unroll values,
    not a specific number, and are outside what a single-value lookup can check against (C17 does
    not need them either: find_literal_dims only reports values >= 128 or with a known composite).

    `model_dir_name` is the directory-style id (`org__model`, e.g. `moonshotai__Kimi-K3`) --
    the same form the registry's own `models:` lists use, NOT the HF slash form."""
    refs = _load_refs() if refs is None else refs
    out: dict[int, str] = {}
    for e in refs.get("irreducible_literals") or []:
        vals, models = e.get("values"), e.get("models") or []
        if not isinstance(vals, list) or (model_dir_name not in models and "all" not in models):
            continue
        note = " ".join((e.get("value_note") or "").split())     # collapse the YAML block scalar
        for v in vals:
            out.setdefault(v, note)
    return out


def annotate(literal_dims: list[dict], model_dir_name: str, refs: dict | None = None) -> list[dict]:
    """Fill in `expr` for any literal whose value is documented as an intentionally-unnamed
    non-architectural constant, so it no longer reads as an unresolved mystery number to C17, the
    model_summary table, or anything else that just checks `L.get("expr")`. Values already
    carrying a real derived-dims `expr` are left untouched -- this only ever fills a gap, never
    overrides a genuine formula."""
    documented = documented_values(model_dir_name, refs)
    out = []
    for L in literal_dims or []:
        L = dict(L)
        if not L.get("expr") and L.get("value") in documented:
            note = documented[L["value"]]
            L["expr"] = f"(비-아키텍처 상수, 의도적으로 이름 없음 -- {note[:160]})" if note else \
                        "(비-아키텍처 상수, 의도적으로 이름 없음 -- references.yaml irreducible_literals 참고)"
        out.append(L)
    return out
