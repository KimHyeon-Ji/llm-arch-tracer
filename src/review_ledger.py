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
#
# review_request.md is in here because it states WHAT needs judging. Without it a model reviewed
# from one angle counted as reviewed forever: the 24 models reviewed on matmul composition all
# read PASS while 149 newly-surfaced open items sat unjudged underneath. A changed request is a
# changed question, and a review is only an answer to the question it was asked.
_INPUTS = ("full/prefill.trace.raw.jsonl", "full/decode.trace.raw.jsonl", "structure.yaml",
           "review_request.md")


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


if __name__ == "__main__":
    import argparse
    import datetime

    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    ap = argparse.ArgumentParser(description="③ 라벨 검토 수행 기록 조회/기록")
    # A reviewer is whatever ran the review -- an LLM, an agent, a person. Recording has to be
    # one command, not a Python snippet, or the procedure only works for whoever wrote it.
    ap.add_argument("--record", metavar="MODEL", help="검토를 마친 모델 폴더 이름")
    ap.add_argument("--findings", type=int, default=0, help="발견 건수 (0도 결과다)")
    ap.add_argument("--notes", default="", help="어떤 각도로 봤는지 한 줄")
    ap.add_argument("--reviewer", default="llm", help="누가 검토했는지")
    a = ap.parse_args()

    if a.record:
        d = os.path.join(root, a.record)
        if not os.path.isdir(d):
            raise SystemExit(f"그런 모델 폴더가 없다: {d}")
        record(a.record, d, datetime.date.today().isoformat(), a.findings, a.notes, a.reviewer)
        print(f"기록됨: {a.record} (발견 {a.findings}건, {a.reviewer})")
    else:
        s = summary(root)
        print(json.dumps(s["counts"], ensure_ascii=False))
        for n, (st, dt) in s["models"].items():
            if st != "PASS":
                print(f"  {st:6s} {n}: {dt}")

# 의뢰서가 낸 질문 수와 판정 수를 대조한다.
#
# 왜: 외부 검토(2026-08-12)가 세 모델에서 같은 실패를 찾아냈다 — Llama-3.1-405B 는 미결 1건이
# 있는데 판정에는 "의뢰서가 비어 있었다"고 적혀 있었고, gpt-oss-20b 는 미결 2건을 무시한 채
# 엉뚱한 항목만 기록했고, Llama-4 는 유일한 질문에 답하지 않았다. 셋 다 원장에는 "최신"으로
# 찍혀 있었다. 원장이 **지문만** 보고 답변 여부는 보지 않았기 때문이다.
#
# 이제 답을 셀 수 있게 한다. 완전한 판정은 아니다(질문 하나에 판정 하나가 1:1로 대응하지는
# 않는다) — 하지만 "질문이 N개인데 판정이 0개"는 명백한 미수행이고, 그것만 잡아도 위 셋이 전부
# 걸린다.
def open_questions(model_dir: str) -> int:
    """의뢰서가 '판단 필요'로 센 건수. 파일이 없으면 -1(판정 불가)."""
    import re as _re
    p = os.path.join(model_dir, "review_request.md")
    if not os.path.exists(p):
        return -1
    m = _re.search(r"판단 필요: \*\*(\d+)건", open(p, encoding="utf-8").read())
    return int(m.group(1)) if m else -1


def unanswered(model_dir: str) -> int:
    """질문이 있는데 판정이 하나도 없으면 그 질문 수, 아니면 0."""
    import json as _json
    n = open_questions(model_dir)
    if n <= 0:
        return 0
    p = os.path.join(model_dir, "review_findings.json")
    if not os.path.exists(p):
        return n
    try:
        with open(p, encoding="utf-8") as f:
            finds = (_json.load(f) or {}).get("findings") or []
    except (ValueError, OSError):
        return n
    return 0 if finds else n
