r"""승격이 **트레이스가 만들지 않는 증거**를 지우지 않는가.

이 결함은 검사가 없어서 살아남았다. `promote()` 가 `shutil.rmtree(dest)` 로 모델 디렉터리를
통째로 갈아끼우는데, `review_findings.json` 은 ③ 자유 평가층이 쓰는 파일이라 새 산출물에
없다. 그래서 재트레이스마다 사라졌고, 47/47 모델에서 없어져 있었다(2026-09-10).

실행:  .venv\Scripts\python.exe develop\test_promote_preserves.py
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
import promote as P                                   # noqa: E402

REPORT = "C1 PASS ok\nC2 WARN meh\n"
FINDINGS = {"model_id": "test/model", "reviewed_on": "2026-09-10", "reviewer": "t",
            "angle": "a", "summary": "s",
            "findings": [{"module": "m", "axis": 1, "current_label": "n_h",
                          "verdict": "should_be_renamed", "proposed_label": "d_head",
                          "confidence": "high", "status": "open", "evidence": "src:12"}]}


def run():
    tmp = tempfile.mkdtemp()
    ok = True
    try:
        old_out, old_models = P.OUT, P.MODELS
        P.OUT = os.path.join(tmp, "out")
        P.MODELS = os.path.join(tmp, "models")
        name = "acme__thing"
        dest = os.path.join(P.MODELS, name)
        os.makedirs(os.path.join(dest, "full"))
        # 옛 산출물: ③층 증거 + 트레이스가 만드는 파일
        json.dump(FINDINGS, open(os.path.join(dest, "review_findings.json"), "w",
                                 encoding="utf-8"))
        open(os.path.join(dest, "review_findings.md"), "w", encoding="utf-8").write("옛 문서")
        open(os.path.join(dest, "prefill.csv"), "w", encoding="utf-8").write("old\n")
        # 새 산출물: 트레이스가 만든 것만 있다
        src = os.path.join(P.OUT, name)
        os.makedirs(os.path.join(src, "full"))
        open(os.path.join(src, "full", "report.md"), "w", encoding="utf-8").write(REPORT)
        open(os.path.join(src, "prefill.csv"), "w", encoding="utf-8").write("new\n")

        P.promote(src)

        got = os.path.join(dest, "review_findings.json")
        if not os.path.exists(got):
            print("  1. **FAIL** review_findings.json 이 승격에서 사라졌다")
            ok = False
        else:
            same = json.load(open(got, encoding="utf-8")) == FINDINGS
            print(f"  1. {'PASS  ' if same else '**FAIL** '}review_findings.json 승계"
                  f"{'' if same else ' -- 내용이 바뀌었다'}")
            ok = ok and same
        md = os.path.join(dest, "review_findings.md")
        if not os.path.exists(md):
            print("  2. **FAIL** review_findings.md 가 없다")
            ok = False
        else:
            body = open(md, encoding="utf-8").read()
            # JSON 에서 다시 그려야 한다 -- 옛 파일을 그대로 옮기면 둘이 어긋날 수 있다.
            good = "옛 문서" not in body and "d_head" in body
            print(f"  2. {'PASS  ' if good else '**FAIL** '}review_findings.md 를 JSON 에서 재생성")
            ok = ok and good
        new_csv = open(os.path.join(dest, "prefill.csv"), encoding="utf-8").read().strip()
        good = new_csv == "new"
        print(f"  3. {'PASS  ' if good else '**FAIL** '}트레이스 산출물은 새것으로 교체"
              f" (읽은 값 {new_csv!r})")
        ok = ok and good
    finally:
        P.OUT, P.MODELS = old_out, old_models
        shutil.rmtree(tmp, ignore_errors=True)
    print("\n승격 증거 보존 " + ("PASS" if ok else "**FAIL**"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(run())
