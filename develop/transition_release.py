r"""전환 검사와 승격을 한 흐름으로 묶는다. **manifest 없이는 승격하지 않는다.**

왜 필요한가
-----------
전환 절차는 네 단계다 -- candidate 자체 게이트, 독립 배치 검증, 전환 diff 분류, 승격.
순서를 문서로만 적으면 한 단계를 빼먹거나, 검사 뒤 candidate 가 바뀐 채로 승격된다
(외부 검토 2026-09-13). 그래서 검사 결과를 `audit_manifest.json` 으로 남기고, 승격은 그
manifest 가 **지금 candidate 와 같은 해시**일 때만 한다.

로직은 기존 도구에 그대로 둔다. 이 파일은 순서와 증거만 책임진다.

    develop/check_batch_labels.py   새 라벨을 다른 배치로 재평가
    develop/transition_diff.py      옛 판 대비 변화를 원인별로 분류
    develop/lowering_proof.py       짝 못 지은 구간을 재실행해 수치로 대조
    develop/promote.py              파일 승격

실행:
    .venv\Scripts\python.exe develop\transition_release.py audit   <모델> <프로필>
    .venv\Scripts\python.exe develop\transition_release.py promote <모델>
"""
import argparse
import datetime
import hashlib
import io
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJ, "src"))
sys.path.insert(0, HERE)
if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import dim_expr as DE                      # noqa: E402

OUT = os.path.join(HERE, "out")
MODELS = os.path.join(PROJ, "models")
MANIFEST = "audit_manifest.json"

# 자동 승인 가능한 변화. 그 밖은 전부 사람이 본다.
# `synthetic_dispatch_scaling` 은 **보존 검사를 통과했을 때만** 여기 해당한다. 깨지면
# transition_diff 가 `synthetic_dispatch_unconserved` 를 따로 내고, 그 이름은 여기 없다.
AUTO_OK = {"같음", "batch_expected", "singleton_fixed", "layout_lowering_verified",
           "literal_resolved", "짝지은 op", "옛 판에만(op)", "새 판에만(op)",
           "자리(양쪽에 있음)", "synthetic_dispatch_scaling",
           # 옛 T 에서 내림이 아무것도 안 깎아 숨어 있던 분할이 드러난 것. 새 이름이 더
           # 정확하다. 축 단위는 "옛 좌표에서 값이 같았다", 구간 단위는 경계 동치로 검증한다.
           "sequence_partition_expected",
           # 인용을 달아 등록한 ④층 교정이 실제로 바꾼 자리. 반영이지 회귀가 아니다 --
           # 발화 0건인 교정은 애초에 이 집합에 안 들어간다.
           "verdict_applied",
           # 지어낸 이름을 거두고 맨 정수로 남긴 자리. 주장이 줄어든 변화라 회귀가 아니다.
           "fabrication_withdrawn",
           # 값이 같아 트레이스로는 못 가르는 이름 교체를, 소스를 인용해 받아들인 자리.
           # 등재는 develop/verify/references.yaml 의 `transition_reviewed` 이고
           # `source` 가 없는 항목은 읽히지 않는다.
           "transition_reviewed"}

# `classify_unmatched` 가 내는 범주 -- 짝 못 지은 **구간** 을 가리킨다. 이것들은
# `lowering_proof` 가 그 구간을 빠짐없이 덮고 전부 통과했을 때만 해소된다.
SEGMENT_CATS = {"semantic_topology_change", "unexplained_topology_change",
                "layout_lowering_verified", "sequence_partition_expected",
                "parameter_access_change"}


def _dir_hash(d: str) -> str:
    """디렉터리의 내용 해시. `full/` 밖 산출물만 본다 -- 승격되는 것이 그것이다."""
    h = hashlib.sha256()
    for name in sorted(os.listdir(d)):
        p = os.path.join(d, name)
        if os.path.isdir(p) or name == MANIFEST:
            continue
        h.update(name.encode())
        with open(p, "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:16]


def _run(cmd: list) -> tuple:
    r = subprocess.run([sys.executable] + cmd, cwd=PROJ, capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def audit(model: str, profile: str) -> int:
    cand = os.path.join(OUT, model)
    if not os.path.isdir(cand):
        print(f"candidate 가 없다: {cand}")
        return 1
    prov = DE.load_provenance(cand)
    report = os.path.join(cand, "full", "report.md")
    fails = [l.strip() for l in open(report, encoding="utf-8")
             if l.startswith("C") and " FAIL " in l] if os.path.exists(report) else ["<report 없음>"]

    print("1) 독립 배치 검증 …")
    rc_b, out_b = _run([os.path.join("develop", "check_batch_labels.py"), profile,
                        "--model-dir", cand])
    print("   " + ("PASS" if rc_b == 0 else "**FAIL**"))

    print("2) 전환 diff 분류 …")
    old = os.path.join(MODELS, model)
    if os.path.isdir(old):
        rc_d, out_d = _run([os.path.join("develop", "transition_diff.py"), model])
    else:
        rc_d, out_d = 0, "옛 판이 없다 -- 새 모델이므로 전환 diff 없음"
    print("   " + ("PASS" if rc_d == 0 else "**FAIL**"))

    cats = {}
    for line in out_d.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].replace(",", "").isdigit():
            cats[parts[0]] = int(parts[1].replace(",", ""))

    # 3) 짝 못 지은 구간의 **계산 동치**를 증명한다. diff 분류는 "어떤 op 이 남았나" 까지만
    #    말하고, 그 구간이 같은 계산인지는 말하지 못한다(외부 검토 2026-09-19). 증명이
    #    그 구간을 **빠짐없이** 덮었을 때만 해당 범주를 해소한다.
    proof = None
    if os.path.isdir(old):
        print("3) 구간 동치 증명 …")
        rc_p, out_p = _run([os.path.join("develop", "lowering_proof.py"), model,
                            "--whole-module", "--per-template", "100000",
                            "--json", os.path.join(cand, "full", "lowering_proof.json")])
        print("   " + ("PASS" if rc_p == 0 else "**FAIL**"))
        pj = os.path.join(cand, "full", "lowering_proof.json")
        if os.path.exists(pj):
            d = json.load(open(pj, encoding="utf-8"))
            tot = sum(v["records_total"] for v in d["phases"].values())
            bad = sum(v["records_failed_template"] + sum(v["uncovered"].values())
                      for v in d["phases"].values())
            proof = {"records": tot,
                     "paired": sum(v["records_paired"] for v in d["phases"].values()),
                     "unproven": bad,
                     "phases": {k: {"records": v["records_total"],
                                    "unproven": v["records_failed_template"]
                                    + sum(v["uncovered"].values()),
                                    "templates": len(v["templates"])}
                                for k, v in d["phases"].items()}}

    needs_review = {k: v for k, v in cats.items() if k not in AUTO_OK and v}
    # 재실행이 **덮지 못한 레코드**도 실패로 센다. `records_total != records_paired` 면
    # 짝을 못 지은 구간이 있다는 뜻인데, 예전 계산은 그걸 실패로 세지 않았다
    # (외부 검토 2026-09-19). 프로세스 exit code 도 함께 본다.
    if proof is not None:
        proof["uncovered_records"] = proof["records"] - proof.get("paired", proof["records"])
        proof["exit_code"] = rc_p
        if proof["uncovered_records"] or rc_p:
            proof["unproven"] += proof["uncovered_records"]
    if proof and proof["unproven"] == 0 and not rc_p:
        seg = {k: v for k, v in cats.items() if k in SEGMENT_CATS and v}
        if sum(seg.values()) == proof["records"]:
            # 증명이 이 구간 범주를 통째로 덮었다. 덮은 수가 정확히 같을 때만 해소한다 --
            # 하나라도 남으면 해소하지 않는다.
            for k in seg:
                needs_review.pop(k, None)
            proof["discharged"] = seg
            # **이름을 정확하게 쓴다.** 이것은 저장된 trace 를 난수 입력으로 다시 돌려
            # 모든 배치 조각에서 경계 출력이 일치함을 본 **수치 시험**이지, 모든 입력에
            # 대한 대수적 동치 증명이 아니다(외부 검토 2026-09-19).
            proof["claim"] = "lowering_replay_consistent"
        else:
            proof["discharged"] = None
            proof["note"] = (f"증명 {proof['records']:,} 건이 구간 범주 합 "
                             f"{sum(seg.values()):,} 건과 달라 해소하지 않는다")

    man = {
        "model": model,
        "audited_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "candidate_hash": _dir_hash(cand),
        "old_hash": _dir_hash(old) if os.path.isdir(old) else None,
        "source_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJ,
                                        capture_output=True, text=True).stdout.strip(),
        "capture_batch": prov.get("capture_batch"),
        "seq_len_used": prov.get("seq_len_used"),
        "label_inputs_digest": (json.load(open(os.path.join(cand, "full", "generated.json"),
                                              encoding="utf-8")) or {}).get("label_inputs")
        if os.path.exists(os.path.join(cand, "full", "generated.json")) else None,
        "gate_fails": fails,
        "independent_batch_check": "pass" if rc_b == 0 else "fail",
        "transition_categories": cats,
        "lowering_proof": proof,
        "needs_review": needs_review,
        "approved": (not fails) and rc_b == 0 and not needs_review,
    }
    with open(os.path.join(cand, MANIFEST), "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=1)

    print(f"\n게이트 FAIL {len(fails)} / 독립 배치 {man['independent_batch_check']}"
          f" / 사람이 볼 것 {sum(needs_review.values())}")
    for k, v in sorted(needs_review.items(), key=lambda kv: -kv[1]):
        print(f"   {k:34} {v:,}")
    print(f"\n{'승인' if man['approved'] else '**보류**'} -> {os.path.join(cand, MANIFEST)}")
    return 0 if man["approved"] else 1


def promote(model: str) -> int:
    cand = os.path.join(OUT, model)
    mp = os.path.join(cand, MANIFEST)
    if not os.path.exists(mp):
        print(f"{model}: audit_manifest 가 없다 -- 먼저 `audit` 를 돌려라")
        return 1
    man = json.load(open(mp, encoding="utf-8"))
    if not man.get("approved"):
        print(f"{model}: manifest 가 승인 상태가 아니다 (사람이 볼 것 "
              f"{sum((man.get('needs_review') or {}).values())})")
        return 1
    now = _dir_hash(cand)
    if now != man.get("candidate_hash"):
        # 검사 뒤 candidate 가 바뀌었다. 그 상태는 검사된 적이 없다.
        print(f"{model}: candidate 가 검사 뒤 바뀌었다 ({man.get('candidate_hash')} -> {now})"
              f" -- 다시 `audit` 를 돌려라")
        return 1
    rc, out = _run([os.path.join("develop", "promote.py"), model])
    print(out.strip())
    return rc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["audit", "promote"])
    ap.add_argument("model")
    ap.add_argument("profile", nargs="?")
    a = ap.parse_args()
    if a.action == "audit":
        if not a.profile:
            print("audit 에는 프로필 경로가 필요하다")
            return 1
        return audit(a.model, a.profile)
    return promote(a.model)


if __name__ == "__main__":
    sys.exit(main())
