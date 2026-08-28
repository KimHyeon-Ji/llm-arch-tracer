"""`rules/label_no_name.yaml` -- verified "this axis intentionally has no name" verdicts.

See that file's header for why this exists instead of using review_findings.json as a safety
gate (src/review_ledger.py's `_covers()` is a loose text match, unsafe for suppressing a hard
FAIL -- a stale finding can keep "covering" an axis whose value has since changed for an
unrelated reason). Every entry here is checked LIVE every run: dead (matches nothing), stale
(matched equivalence-class count changed), or missing its required adaptation-log marker are all
treated as NOT covering anything, so `src/axis_classes.py:bad_stub_count()` keeps failing on them
exactly as if the entry did not exist.

SCHEMA IS ENFORCED, NOT ASSUMED. Found 2026-08-27 (Codex review): the first version of this file
trusted every entry to already have `verdict: no_name_exists`, a real
`required_adaptation_marker`, etc., without checking. An entry with `verdict: NOT_SUPPORTED` or a
missing marker field still granted coverage -- the exact fail-open this mechanism exists to
prevent, just moved one layer down. `_validate()` runs at load time; an entry that fails it can
never appear in `for_model()`'s output (so it can never grant coverage) and is reported as its own
problem via `validation_problems()`, the same way a dead/stale entry is reported via `issues()`.
"""
import json
import os

import yaml

_PATH = os.path.join(os.path.dirname(__file__), "..", "rules", "label_no_name.yaml")
_CACHE = None                          # (valid_entries, [(rule_id, problem), ...])

_SELECTOR_KEYS = ("module", "shape", "axis", "field", "shape_index", "op_type", "nth", "expect")

# field -> required type(s). Every one of these must be present AND non-empty/non-None; a present
# but empty string/list does not count (an entry that "has" a blank source is still undocumented).
_REQUIRED_FIELDS = {
    "rule_id": str, "model": str, "phase": str, "module": str, "axis": int,
    "field": str, "shape_index": int, "op_type": str, "nth": int, "expect": (int, float),
    "verdict": str, "required_adaptation_marker": str, "expected_classes": int, "source": str,
}


def _validate(entries: list) -> tuple:
    """(valid_entries, [(rule_id, problem), ...]). A malformed entry is dropped from
    valid_entries entirely -- it must never reach selector_key()/covered_keys() and grant
    coverage on the strength of a selector alone while its verdict/marker/etc. are wrong or
    missing."""
    valid, problems = [], []
    seen_ids = set()
    for i, e in enumerate(entries):
        if not isinstance(e, dict):
            problems.append((f"(#{i})", "항목이 dict가 아님"))
            continue
        rid = e.get("rule_id") or f"(rule_id 없음, #{i})"
        missing = [k for k in _REQUIRED_FIELDS
                   if e.get(k) is None or e.get(k) == "" or e.get(k) == []]
        if missing:
            problems.append((rid, f"필수 필드 누락/빈값: {missing}"))
            continue
        bad_types = [k for k, t in _REQUIRED_FIELDS.items() if not isinstance(e[k], t)]
        if bad_types:
            problems.append((rid, f"필드 타입 오류: {bad_types}"))
            continue
        if e["verdict"] != "no_name_exists":
            problems.append((rid, f"verdict가 'no_name_exists'가 아님: {e['verdict']!r} "
                                   f"-- 이 파일은 그 판정 하나만 표현한다"))
            continue
        if not isinstance(e.get("shape"), list) or not e["shape"]:
            problems.append((rid, "shape가 비어있지 않은 리스트가 아님"))
            continue
        if e["phase"] not in ("prefill", "decode"):
            problems.append((rid, f"phase가 prefill/decode가 아님: {e['phase']!r}"))
            continue
        if rid in seen_ids:
            problems.append((rid, "rule_id 중복"))
            continue
        seen_ids.add(rid)
        valid.append(e)
    return valid, problems


def load(path: str | None = None) -> list:
    """Valid entries only -- see `validation_problems()` for what got dropped and why."""
    return _load_cached(path)[0]


def validation_problems(path: str | None = None) -> list:
    return _load_cached(path)[1]


def _load_cached(path: str | None = None) -> tuple:
    global _CACHE
    path = path or _PATH               # NOT a default arg -- _PATH is monkeypatched in tests
                                        # (verify_selftest.py); a default bakes in the value at
                                        # import time and silently ignores the patch.
    if _CACHE is None:
        if not os.path.exists(path):
            _CACHE = ([], [])
        else:
            with open(path, encoding="utf-8") as f:
                raw = (yaml.safe_load(f) or {}).get("no_name") or []
            _CACHE = _validate(raw)
    return _CACHE


def for_model(model_dir_name: str, path: str | None = None) -> list:
    return [o for o in load(path) if o.get("model") == model_dir_name]


def selector_key(d: dict | None):
    """Selector identity, ignoring bookkeeping fields (rule_id, verdict, source, ...) -- the same
    fields label_overrides.py/label_confirmed.yaml use to anchor a specific axis position.

    NOT identical to label_overrides.py's actual apply() matcher, which also reads `from`, an
    optional `rank`, and `layer_types` (Codex review, 2026-08-27). Safe today only because every
    entry here is hand-written to match this project's stub-generator invariants exactly
    (`from == shape[axis]`, `spread: class`, no `rank`/`layer_types` -- see
    src/axis_classes.py's unsettled()/write_unsettled()): the generator never emits a stub whose
    `from` differs from the axis value itself, so `from` is redundant with `expect`+`axis` here.
    If a future entry ever needs `rank`/`layer_types` to disambiguate, this key must grow to
    include them -- it will silently under-match (treat two different axes as the same one)
    otherwise."""
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
    """{rule_id: [(phase, unsettled item), ...]} -- every no_name entry paired with the
    stub_ambiguous items whose (phase, override_stub selector) matches it exactly. An entry whose
    declared `phase` doesn't match where the item actually showed up is NOT a match -- phase
    drifting is itself a sign the verdict no longer describes the same thing. Since the entry's
    own `phase` is part of the join key, every hit list is already phase-homogeneous by
    construction (all its entries share one phase) -- see covered_keys()/issues()."""
    entries = for_model(model_dir_name)
    by_key = {}
    for e in entries:
        by_key.setdefault((e["phase"], selector_key(e)), []).append(e["rule_id"])
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
    axis_classes.bad_stub_count(); anything else must keep failing exactly as if unregistered.
    Entries that failed schema validation never reach here at all (for_model() already dropped
    them), so a malformed entry can never grant coverage."""
    entries = for_model(model_dir_name)
    markers = _adaptation_markers(model_dir)
    m = _matches(model_dir_name, model_dir)
    live = set()
    for e in entries:
        hits = m.get(e["rule_id"]) or []
        if not hits:
            continue                                          # dead -- never live
        if e["required_adaptation_marker"] not in markers:
            continue                                          # marker gone -- never live
        classes = {it.get("classes") for _ph, it in hits}
        if len(classes) != 1 or next(iter(classes)) != e["expected_classes"]:
            continue                                          # scope changed -- never live
        live.add((e["phase"], selector_key(e)))
    return live


def issues(model_dir_name: str, model_dir: str) -> list:
    """[(rule_id, problem)] for every no_name entry that is dead, stale, or missing its required
    adaptation marker -- develop/verify_all.py FAILs on any of these, the same way a dead
    label_overrides.yaml/label_confirmed.yaml entry does. Schema problems are reported separately
    by `validation_problems()` (global, not per-model -- a malformed entry may not even have a
    valid `model` field to key this function by)."""
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
        req = e["required_adaptation_marker"]
        if req not in markers:
            out.append((rid, f"필요한 adaptation marker '{req}' 가 provenance.adaptation_log 에 없음"))
            continue
        classes = {it.get("classes") for _ph, it in hits}
        if len(classes) != 1:
            # phase는 매칭 키에 이미 들어있어 hits는 항상 단일 phase다 -- 그런데도 등가류
            # 개수가 갈린다면 같은 selector에 매치된 서로 다른 항목들의 값이 다르다는 뜻이다.
            out.append((rid, f"같은 selector에 매치된 항목들의 등가류 개수가 서로 다름: {sorted(classes)}"))
            continue
        actual = next(iter(classes))
        if actual != e["expected_classes"]:
            out.append((rid, f"등가류 개수 변경 -- 기록 {e['expected_classes']}, 실측 {actual}"))
    return out
