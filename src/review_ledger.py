"""Record whether the layer-3 free-form review actually happened, and against WHICH artifacts.

`full/review.md` is generated on every run, but nothing reads it and nothing recorded whether a
reviewer ever did. So "the review step is part of the pipeline" was true only of the packet: a
model that had never been reviewed looked exactly like one reviewed this morning, and a model
reviewed before a labelling change looked current. Both are the failure mode C17 exists to stop
-- a silently-unfinished deliverable that every automatic check passes.

This module fingerprints what a review depends on. If the fingerprint moved since the recorded
review, the review is STALE and says so; the gate surfaces it. It does not perform the review --
it makes not having performed it visible.
"""
import hashlib
import json
import os

import yaml

LEDGER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "develop", "verify", "review_ledger.yaml")

# What a reviewer actually looks at. structure.yaml carries the symbol table and label
# provenance; the raw traces carry every rendered label. A change in any of them can invalidate
# a finding, so all of them go into the fingerprint.
_INPUTS = ("full/prefill.trace.raw.jsonl", "full/decode.trace.raw.jsonl", "structure.yaml")


def fingerprint(model_dir: str) -> str | None:
    """Short hash of the artifacts a review is a statement about."""
    h = hashlib.sha256()
    seen = False
    for rel in _INPUTS:
        p = os.path.join(model_dir, rel)
        if not os.path.exists(p):
            continue
        seen = True
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()[:16] if seen else None


def load() -> dict:
    if not os.path.exists(LEDGER):
        return {}
    return yaml.safe_load(open(LEDGER, encoding="utf-8")) or {}


def status(model_dir: str, name: str, ledger: dict | None = None) -> tuple[str, str]:
    """('PASS'|'STALE'|'NONE', detail) for one model."""
    ledger = load() if ledger is None else ledger
    cur = fingerprint(model_dir)
    rec = (ledger.get("models") or {}).get(name)
    if not rec:
        return "NONE", "③ 자유 평가 기록 없음"
    was = rec.get("fingerprint")
    if cur and was and cur != was:
        return "STALE", (f"③ 자유 평가 이후 산출물이 바뀜 (기록 {was} != 현재 {cur}, "
                         f"검토일 {rec.get('reviewed_on')})")
    return "PASS", (f"{rec.get('reviewed_on')} 검토, 발견 {rec.get('findings', 0)}건")


def record(name: str, model_dir: str, reviewed_on: str, findings: int,
           notes: str = "", reviewer: str = "llm") -> None:
    """Write one model's review into the ledger. Called after a review is actually performed."""
    led = load()
    led.setdefault("note", "③ 자유 평가(README) 수행 기록. fingerprint 는 검토 대상 산출물의 해시이고, "
                           "지금 산출물과 다르면 그 검토는 만료다.")
    led.setdefault("models", {})[name] = {
        "reviewed_on": reviewed_on,
        "reviewer": reviewer,
        "fingerprint": fingerprint(model_dir),
        "findings": findings,
        "notes": notes,
    }
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    with open(LEDGER, "w", encoding="utf-8") as f:
        yaml.safe_dump(led, f, allow_unicode=True, sort_keys=False)


def summary(models_root: str) -> dict:
    """{'PASS': n, 'STALE': n, 'NONE': n} plus the per-model verdicts."""
    led = load()
    out, counts = {}, {"PASS": 0, "STALE": 0, "NONE": 0}
    for name in sorted(os.listdir(models_root)):
        d = os.path.join(models_root, name)
        if not os.path.isdir(d):
            continue
        st, detail = status(d, name, led)
        out[name] = (st, detail)
        counts[st] += 1
    return {"counts": counts, "models": out}


if __name__ == "__main__":  # quick status dump
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    s = summary(root)
    print(json.dumps(s["counts"], ensure_ascii=False))
    for n, (st, d) in s["models"].items():
        if st != "PASS":
            print(f"  {st:6s} {n}: {d}")
