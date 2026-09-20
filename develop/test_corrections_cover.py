r"""`computation_corrected` 예외가 **넘겨선 안 되는 것을 넘기지 않는가.**

"계산을 고쳤다" 는 선언은 구간 동치 실패를 통째로 덮는다. 그래서 이 예외가 헐거우면
게이트 전체가 헐거워진다. 외부 검토(2026-09-20)가 네 가지 반례로 옛 구현을 뚫었다:

    1. 같은 276 선언이 서로 다른 template 두 개에 재사용돼 552 를 해소
    2. `modules` 가 대표 4 개로 잘려 있어, 생략된 자리에 `mlp` 가 섞여도 해소
    3. 실패 사유가 "재실행조차 못 했다" 인데도 해소
    4. 선언과 총량이 다른데(276 != 300) 해소

실행:
    .venv\Scripts\python.exe develop\test_corrections_cover.py
"""
import io
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJ, "src"))
sys.path.insert(0, HERE)
if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import transition_release as TR                                # noqa: E402

DECL = [{"model": "m", "phase": "decode", "scope": "self_attn",
         "expect_records": 276, "source": "x"}]


def _tmpl(records, mods, kinds=("value_mismatch",), failed=1):
    return {"n": 69, "records": records, "proved": 0, "failed": failed,
            "why": {"값 불일치(b=0)": 1}, "modules": list(mods),
            "module_leaves": sorted({m.rsplit(".", 1)[-1] for m in mods}),
            "why_kinds": list(kinds), "old_ops": {}, "new_ops": {}}


def _proof(templates):
    return {"model": "m", "phases": {"decode": {
        "records_total": sum(t["records"] for t in templates),
        "records_paired": sum(t["records"] for t in templates),
        "records_verified_template": 0,
        "records_failed_template": sum(t["records"] for t in templates),
        "uncovered": {}, "templates": templates}}}


def run(name, templates, want_done, want_left_nonzero):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "proof.json")
        json.dump(_proof(templates), io.open(p, "w", encoding="utf-8"), ensure_ascii=False)
        done, left, why = TR._corrections_cover(p, DECL)
    ok = (done == want_done) and (bool(left) == want_left_nonzero)
    print(f"{'OK ' if ok else '**실패**'} {name:38} 해소 {done:>4}  남은 {left}  {why[:60]}")
    return 0 if ok else 1


def main():
    A = ["model.layers.0.self_attn"]
    B = ["model.layers.1.self_attn"]
    fails = 0
    # 정상: 선언과 총량이 맞고 전부 self_attn, 사유가 값 불일치
    fails += run("정상 276 -> 해소", [_tmpl(276, A)], 276, False)
    # 1. 같은 선언 재사용 (276 + 276 = 552)
    fails += run("반례1 template 둘로 552", [_tmpl(276, A), _tmpl(276, B)], 0, True)
    # 2. scope 밖 모듈이 섞임
    fails += run("반례2 mlp 섞임", [_tmpl(276, A + ["model.layers.0.mlp"])], 0, True)
    # 3. 값 불일치가 아닌 실패
    fails += run("반례3 재실행 실패", [_tmpl(276, A, kinds=("replay_error",))], 0, True)
    # 4. 총량이 다름
    fails += run("반례4 총량 300", [_tmpl(300, A)], 0, True)
    print()
    if fails:
        print(f"**{fails}개 실패 -- 예외가 넘겨선 안 되는 것을 넘긴다**")
    else:
        print("5/5 — 선언한 범위·총량·사유가 전부 맞을 때만 해소한다")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
