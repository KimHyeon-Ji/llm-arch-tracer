r"""승격 게이트가 **실제로 막는가**. 죽은 게이트는 통과시키고 아무 말도 안 한다.

이 게이트가 있는 이유는 "순서를 문서로만 적으면 한 단계를 빼먹거나, 검사 뒤 candidate 가
바뀐 채로 승격된다" 는 것이다(외부 검토 2026-09-13). 그러니 그 두 가지를 실제로 주입해
막히는지 본다.

실행:  .venv\Scripts\python.exe develop\test_transition_release.py
"""
import io
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, HERE)
import transition_release as TR          # noqa: E402

MAN = {"model": "acme__thing", "candidate_hash": None, "approved": True,
       "needs_review": {}, "gate_fails": [], "independent_batch_check": "pass"}


def run():
    tmp = tempfile.mkdtemp()
    ok = True
    old_out, old_models = TR.OUT, TR.MODELS
    try:
        TR.OUT = os.path.join(tmp, "out")
        TR.MODELS = os.path.join(tmp, "models")
        cand = os.path.join(TR.OUT, "acme__thing")
        os.makedirs(os.path.join(cand, "full"))
        open(os.path.join(cand, "prefill.csv"), "w", encoding="utf-8").write("a\n")

        def check(name, cond, detail=""):
            nonlocal ok
            print(f"  {'PASS  ' if cond else '**FAIL**'} {name}" + (f"  {detail}" if detail else ""))
            ok = ok and cond

        # 1) manifest 가 없으면 막는다
        check("manifest 없으면 거부", TR.promote("acme__thing") != 0)

        # 2) 승인되지 않은 manifest 는 막는다
        m = dict(MAN, approved=False, needs_review={"semantic_change": 3})
        m["candidate_hash"] = TR._dir_hash(cand)
        json.dump(m, open(os.path.join(cand, TR.MANIFEST), "w", encoding="utf-8"))
        check("미승인 manifest 거부", TR.promote("acme__thing") != 0)

        # 3) 검사 뒤 candidate 가 바뀌면 막는다
        m = dict(MAN, approved=True)
        m["candidate_hash"] = TR._dir_hash(cand)
        json.dump(m, open(os.path.join(cand, TR.MANIFEST), "w", encoding="utf-8"))
        open(os.path.join(cand, "prefill.csv"), "a", encoding="utf-8").write("변조\n")
        check("검사 뒤 변조 거부", TR.promote("acme__thing") != 0,
              f"해시 {m['candidate_hash']} != {TR._dir_hash(cand)}")

        # 4) manifest 자신은 해시에 안 들어간다 -- 안 그러면 쓰는 순간 자기 해시가 달라진다
        h1 = TR._dir_hash(cand)
        json.dump({"x": 1}, open(os.path.join(cand, TR.MANIFEST), "w", encoding="utf-8"))
        check("manifest 는 해시에서 제외", h1 == TR._dir_hash(cand))
    finally:
        TR.OUT, TR.MODELS = old_out, old_models
        shutil.rmtree(tmp, ignore_errors=True)
    print("\n승격 게이트 " + ("PASS" if ok else "**FAIL**"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(run())
