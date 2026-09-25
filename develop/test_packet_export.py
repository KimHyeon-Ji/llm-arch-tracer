r"""반출 패킷의 **누설 방지와 세션 독립성**을 실제 패킷으로 확인한다.

외부 검토가 "독립 패킷 한두 개로 누출·중복 세션 방지를 시험하라" 고 했다(2026-09-25).
손으로 한 번 해 보는 것으로는 다음에 깨져도 모른다.

임시 디렉터리에 패킷을 만들고, 대장도 임시 파일로 돌린다. **실제 대장과 반출 위치는
건드리지 않는다.**

실행:
    .venv\Scripts\python.exe develop\test_packet_export.py
"""
import io
import json
import os
import re
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import export_shard_packet as X                                  # noqa: E402

OK, FAIL = [], []
NL = chr(10)


def check(name, cond):
    (OK if cond else FAIL).append(name)
    print(("  OK   " if cond else "  FAIL ") + name)


def run(*argv):
    """`main()` 을 인자로 부른다. 종료 코드를 돌려준다."""
    old = sys.argv
    sys.argv = ["export_shard_packet.py", *argv]
    try:
        return X.main() or 0
    except SystemExit as e:
        return e.code
    finally:
        sys.argv = old


tmp = tempfile.mkdtemp()
real_ledger, real_out = X.LEDGER, X.OUT_DEFAULT
try:
    X.LEDGER = os.path.join(tmp, "_ledger.jsonl")
    out = os.path.join(tmp, "packets")

    am, bman = X.load_assignment()
    check("배정 manifest 가 이 bundle 과 맞는다", True)   # load_assignment 가 통과함
    shards = [e["shard"] for e in am["manifest"]]
    dup = am.get("duplicate_assignment") or {}
    dup_shards = {s for v in dup.values() for s in v}
    prims = [s for s in shards if s not in dup_shards]
    primary, primary2 = prims[0], prims[1]
    # 그 원본과 단위가 겹치는 중복본 shard
    pu = set(next(e for e in am["manifest"] if e["shard"] == primary)["unit_ids"])
    counterpart = next((e["shard"] for e in am["manifest"]
                        if e["shard"] in dup_shards and pu & set(e["unit_ids"])), None)

    print("1) 패킷 하나를 실제로 만든다")
    check("반출 성공", run("--shard", primary, "--session", "S-1", "--out", out) == 0)
    pids = os.listdir(out)
    check("패킷 디렉터리 하나", len(pids) == 1)
    pd = os.path.join(out, pids[0])
    check("opaque packet id (shard 번호가 아니다)",
          re.fullmatch(r"P-[0-9a-f]{12}", pids[0]) is not None)
    check("담긴 것은 shard.md / source / _packet.json 뿐",
          sorted(os.listdir(pd)) == ["_packet.json", "shard.md", "source"])
    txt = io.open(os.path.join(pd, "shard.md"), encoding="utf-8").read()

    print("2) 누설 검사 — 패킷 안에 없어야 하는 것")
    check("**전역 shard 번호가 없다**", re.search(r"shard\s*\d", txt) is None)
    check("머리글이 packet id 로 바뀌었다", txt.startswith(f"# 패킷 {pids[0]}"))
    check("X_linked 가 없다", "X_linked" not in txt)
    check("등급 문자열이 없다",
          not re.search(r"heuristic|open_tie|scope_inferred|unresolved", txt))
    check("발행 영향 수가 없다", "발행 셀" not in txt and "raw 자리" not in txt)
    check("중복·배정 관련 문구가 없다",
          not re.search(r"중복|배정|revision", txt))
    for bad in ("units", "crosswalk", "assignment", "salt", "_manifest",
                "priority", "shard0"):
        check(f"금지 이름 `{bad}` 인 파일이 없다",
              not any(bad in n for _r, d, f in os.walk(pd) for n in list(d) + f))
    pj = json.load(io.open(os.path.join(pd, "_packet.json"), encoding="utf-8"))
    check("_packet.json 에 packet_id·단위 수·source 해시만",
          set(pj) == {"packet_id", "units", "sources", "answer_file", "note"})
    check("source 해시가 64 자리",
          all(len(s["sha256"]) == 64 for s in pj["sources"]))
    check("참조된 source 가 전부 들어 있다",
          all(os.path.exists(os.path.join(pd, s["path"])) for s in pj["sources"]))

    print("3) **세션 <-> shard 1:1** — 지정된 반례")
    # 외부 검토가 실행해 보인 반례: 단위가 겹치지 않는 원본 두 개가 한 세션에 들어갔다
    check("첫 원본을 S-1 에 반출 -> 성공 (위에서 했다)", True)
    check("단위가 겹치지 않는다",
          not (pu & set(next(e for e in am["manifest"]
                             if e["shard"] == primary2)["unit_ids"])))
    check("**겹치지 않는 두 번째 원본을 S-1 에 반출 -> 거부**",
          run("--shard", primary2, "--session", "S-1", "--out", out) == 2)
    check("두 번째 원본을 새 세션 S-2 에 반출 -> 성공",
          run("--shard", primary2, "--session", "S-2", "--out", out) == 0)

    print("3-b) 세션 독립성 — 도구가 거부한다")
    if counterpart:
        check("**같은 세션에 중복본을 주면 거부한다**",
              run("--shard", counterpart, "--session", "S-1", "--out", out) == 2)
        check("새 세션이면 허용한다",
              run("--shard", counterpart, "--session", "S-3", "--out", out) == 0)
    else:
        check("겹치는 중복본을 찾았다", False)
    check("이미 나간 shard 를 다른 세션에 주면 거부한다",
          run("--shard", primary, "--session", "S-4", "--out", out) == 2)
    # 같은 세션에 **재전달도 거부**한다 -- 단위 겹침 검사에 먼저 걸린다. 재전달이
    # 필요하면 이미 만들어 둔 패킷 디렉터리를 그대로 주면 되므로, 거부가 안전한 쪽이다.
    check("같은 세션 재전달도 거부한다 (fail-closed)",
          run("--shard", primary, "--session", "S-1", "--out", out) == 2)
    check("없는 shard 는 거부한다",
          run("--shard", "shard999.md", "--session", "S-9", "--out", out) == 2)

    print("3-c) 반출 위치가 저장소 안이면 거부한다")
    for inside in (PROJ, os.path.join(PROJ, "develop"),
                   os.path.join(PROJ, "..", "llm-arch-tracer-results-labeled",
                                "work")):
        check(f"거부: {os.path.basename(os.path.normpath(inside))}",
              run("--shard", "shard003.md", "--session", "S-X",
                  "--out", inside) == 2)

    print("3-d) 대장 lock")
    lockp = X.LEDGER + ".lock"
    os.makedirs(lockp)
    try:
        check("**lock 이 잡혀 있으면 반출하지 않는다**",
              run("--shard", "shard003.md", "--session", "S-L", "--out", out) == 2)
    finally:
        shutil.rmtree(lockp, ignore_errors=True)
    check("lock 이 풀리면 반출된다",
          run("--shard", "shard003.md", "--session", "S-L", "--out", out) == 0)
    check("lock 디렉터리가 남지 않는다", not os.path.exists(lockp))

    print("3-e) 단위 순서가 결정론적으로 섞인다")
    ledx = [json.loads(l) for l in io.open(X.LEDGER, encoding="utf-8") if l.strip()]
    rec = next(e for e in ledx if e["shard"] == primary)
    check("대장에 순서가 기록된다", len(rec["unit_order"]) == rec["units"])
    check("순서는 배정 단위 집합과 같다",
          set(rec["unit_order"]) == set(rec["unit_ids"]))
    doc = io.open(os.path.join(pd, "shard.md"), encoding="utf-8").read()
    in_doc = re.findall(r"^### (\S+)", doc.split("## 심볼 정의와 config 값")[0],
                        flags=re.M)
    check("**문서의 순서가 대장의 순서와 같다**", in_doc == rec["unit_order"])
    check("원본 순서(배정 순서)와는 다르다", in_doc != rec["unit_ids"])
    if counterpart:
        rc = next(e for e in ledx if e["shard"] == counterpart)
        shared = [u for u in rc["unit_order"] if u in rec["unit_order"]]
        rel = [u for u in rec["unit_order"] if u in shared]
        check("중복본은 원본과 다른 순서를 쓴다 (packet id 가 다르므로)",
              len(shared) < 2 or shared != rel)

    print("4) 대장에 역할이 기록된다 (검토자에게는 가지 않는다)")
    led = [json.loads(l) for l in io.open(X.LEDGER, encoding="utf-8") if l.strip()]
    check("성공한 반출만 기록됐다 (원본 2 + 중복본 1 + lock 뒤 1)", len(led) == 4)
    check("거부된 시도는 기록되지 않았다",
          {e["session_id"] for e in led} == {"S-1", "S-2", "S-3", "S-L"})
    check("모든 기록이 현재 revision", {e["assignment_revision"] for e in led}
          == {am["assignment_revision"]})
    check("세션 <-> shard 가 1:1 이다",
          len({e["session_id"] for e in led}) == len({e["shard"] for e in led})
          == len(led))
    roles = {r for e in led for r in e["roles"].values()}
    check("원본과 중복본이 모두 기록됐다", roles == {"primary", "duplicate"})
    check("대장이 패킷 밖에 있다",
          not os.path.exists(os.path.join(pd, os.path.basename(X.LEDGER))))

    print("4-b) 옛 revision 이 섞인 대장은 거부한다")
    keep = [json.loads(l) for l in io.open(X.LEDGER, encoding="utf-8") if l.strip()]
    poisoned = [dict(keep[0], assignment_revision=keep[0]["assignment_revision"] - 1)]
    X.write_ledger(poisoned + keep[1:])
    check("**revision 이 다른 기록이 있으면 반출하지 않는다**",
          run("--shard", "shard004.md", "--session", "S-R", "--out", out) == 2)
    X.write_ledger(keep)
    check("되돌리면 다시 반출된다",
          run("--shard", "shard004.md", "--session", "S-R", "--out", out) == 0)

    print("5) bundle 과 배정 manifest 가 어긋나면 거부한다")
    hold = X.ASSIGN
    try:
        broken = os.path.join(tmp, "broken.json")
        b = dict(am)
        b["priority_bundle_manifest_sha256"] = "0" * 64
        json.dump(b, io.open(broken, "w", encoding="utf-8", newline=NL),
                  ensure_ascii=False)
        X.ASSIGN = broken
        check("**연결 해시가 다르면 패킷을 만들지 않는다**",
              run("--shard", primary, "--session", "S-8", "--out", out) == 2)
        b2 = dict(am)
        b2["manifest_payload_sha256"] = "0" * 64
        json.dump(b2, io.open(broken, "w", encoding="utf-8", newline=NL),
                  ensure_ascii=False)
        check("자기 해시가 다르면 패킷을 만들지 않는다",
              run("--shard", primary, "--session", "S-8", "--out", out) == 2)
    finally:
        X.ASSIGN = hold
finally:
    X.LEDGER, X.OUT_DEFAULT = real_ledger, real_out
    shutil.rmtree(tmp, ignore_errors=True)

print(NL + f"{len(OK)}/{len(OK) + len(FAIL)} 통과 — "
      "패킷 하나만 나가고, 같은 단위가 한 세션에 두 번 가지 않는다")
if FAIL:
    print("실패: " + ", ".join(FAIL))
sys.exit(1 if FAIL else 0)
