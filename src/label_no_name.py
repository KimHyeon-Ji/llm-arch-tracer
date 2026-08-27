"""`rules/label_no_name.yaml` -- verified "this axis intentionally has no name" verdicts.

See that file's header for why this exists instead of using review_findings.json as a safety
gate (src/review_ledger.py's `_covers()` is a loose text match, unsafe for suppressing a hard
FAIL -- a stale finding can keep "covering" an axis whose value has since changed for an
unrelated reason). Every entry here is checked LIVE every run: dead (matches nothing), stale
(matched equivalence-class count changed), or missing its required adaptation-log marker are all
treated as NOT covering anything, so `src/axis_classes.py:bad_stub_count()` keeps failing on them
exactly as if the entry did not exist.
"""
import json
import os

import yaml

_PATH = os.path.join(os.path.dirname(__file__), "..", "rules", "label_no_name.yaml")
_CACHE = None

_SELECTOR_KEYS = ("module", "shape", "axis", "field", "shape_index", "op_type", "nth", "expect")


def load(path: str | None = None) -> list:
    global _CACHE
    path = path or _PATH               # NOT a default arg -- _PATH is monkeypatched in tests
                                        # (verify_selftest.py); a default bakes in the value at
                                        # import time and silently ignores the patch.
    if _CACHE is None:
        if not os.path.exists(path):
            _CACHE = []
        else:
            with open(path, encoding="utf-8") as f:
                _CACHE = (yaml.safe_load(f) or {}).get("no_name") or []
    return _CACHE


def for_model(model_dir_name: str, path: str | None = None) -> list:
    return [o for o in load(path) if o.get("model") == model_dir_name]


def selector_key(d: dict | None):
    """Selector identity, ignoring bookkeeping fields (rule_id, verdict, source, ...) -- the same
    fields label_overrides.py/label_confirmed.yaml use to anchor a specific axis position."""
    if not d:
        return None
    try:
        return tuple((k, tuple(d[k]) if isinstance(d.get(k), list) else d.get(k))
                     for k in _SELECTOR_KEYS)
    except Exception:                                   # noqa: BLE001 -- malformed entry, no match
        return None


def _adaptation_markers(model_dir: str) -> set:
    prov = os.path.join(model_dir, "full", "provenance.json")
    if not os.path.exists(prov):
        return set()
    try:
        p = json.load(open(prov, encoding="utf-8"))
    except (ValueError, OSError):
        return set()
    return {e.get("remedy") for e in (p.get("adaptation_log") or []) if e.get("remedy")}


def _stub_items(model_dir: str, phase: str) -> list:
    p = os.path.join(model_dir, "full", f"{phase}.unsettled.json")
    if not os.path.exists(p):
        return []
    try:
        with open(p, encoding="utf-8") as f:
            items = (json.load(f) or {}).get("items") or []
    except (ValueError, OSError):
        return []
    return [it for it in items if it.get("stub_ambiguous")]


def _matches(model_dir_name: str, model_dir: str) -> dict:
    """{rule_id: [unsettled item, ...]} -- every no_name entry paired with the stub_ambiguous
    items whose (phase, override_stub selector) matches it exactly. An entry whose declared
    `phase` doesn't match where the item actually showed up is NOT a match -- phase drifting is
    itself a sign the verdict no longer describes the same thing."""
    entries = for_model(model_dir_name)
    by_key = {}
    for e in entries:
        by_key.setdefault((e.get("phase"), selector_key(e)), []).append(e["rule_id"])
    out = {e["rule_id"]: [] for e in entries}
    for phase in ("prefill", "decode"):
        for it in _stub_items(model_dir, phase):
            k = (phase, selector_key(it.get("override_stub")))
            for rid in by_key.get(k, []):
                out[rid].append((phase, it))
    return out


def covered_keys(model_dir_name: str, model_dir: str) -> set:
    """(phase, selector) pairs whose no_name verdict is LIVE right now -- matched at least once,
    required adaptation marker present, and matched equivalence-class count equals what was
    recorded when the verdict was written. Only these are safe to exclude from
    axis_classes.bad_stub_count(); anything else must keep failing exactly as if unregistered."""
    entries = for_model(model_dir_name)
    markers = _adaptation_markers(model_dir)
    m = _matches(model_dir_name, model_dir)
    live = set()
    for e in entries:
        hits = m.get(e["rule_id"]) or []
        if not hits:
            continue                                          # dead -- never live
        req = e.get("required_adaptation_marker")
        if req and req not in markers:
            continue                                          # marker gone -- never live
        classes = {it.get("classes") for _ph, it in hits}
        if len(classes) != 1 or next(iter(classes)) != e.get("expected_classes"):
            continue                                          # scope changed -- never live
        live.add((e.get("phase"), selector_key(e)))
    return live


def issues(model_dir_name: str, model_dir: str) -> list:
    """[(rule_id, problem)] for every no_name entry that is dead, stale, or missing its required
    adaptation marker -- develop/verify_all.py FAILs on any of these, the same way a dead
    label_overrides.yaml/label_confirmed.yaml entry does."""
    entries = for_model(model_dir_name)
    markers = _adaptation_markers(model_dir)
    m = _matches(model_dir_name, model_dir)
    out = []
    for e in entries:
        rid = e["rule_id"]
        hits = m.get(rid) or []
        if not hits:
            out.append((rid, "매치 0건 -- dead verdict"))
            continue
        req = e.get("required_adaptation_marker")
        if req and req not in markers:
            out.append((rid, f"필요한 adaptation marker '{req}' 가 provenance.adaptation_log 에 없음"))
            continue
        classes = {it.get("classes") for _ph, it in hits}
        if len(classes) != 1:
            out.append((rid, f"등가류 개수가 phase마다 다름: {sorted(classes)}"))
            continue
        actual = next(iter(classes))
        if actual != e.get("expected_classes"):
            out.append((rid, f"등가류 개수 변경 -- 기록 {e.get('expected_classes')}, 실측 {actual}"))
    return out
