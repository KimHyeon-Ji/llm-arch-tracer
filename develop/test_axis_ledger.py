r"""축 판정 원장의 계약. 등식이 성립하는가, 그리고 **검사가 살아 있는가**.

죽은 검사는 0 을 내고 통과시킨다. 그래서 각 등급마다 그 등급이 나와야 하는 상황을 만들어
실제로 그 등급이 나오는지 본다.

실행:  .venv\Scripts\python.exe develop\test_axis_ledger.py
"""
import io
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJ, "src"))
if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import axis_ledger as AL             # noqa: E402


def case_grades_are_reachable():
    """다섯 등급이 **전부** 실제로 나온다. 하나라도 안 나오면 그 등급은 장식이다."""
    g = AL.Ledger()
    g.record((1, "i", 0, 0), "d_model", "scoped_symbol")                     # confirmed
    g.record((1, "i", 0, 1), "d_head", "scoped_symbol",
             raw=("E", "d_head"), scoped=("d_head",))                        # scope_inferred
    g.record((1, "i", 0, 2), "d_rope/2", "heur_half")                        # heuristic
    g.record((1, "i", 0, 3), "n_h", "scoped_symbol",
             raw=("n_h", "n_kv"), scoped=("n_h", "n_kv"))                    # open_tie
    g.record((1, "i", 0, 4), "512", "bare")                                  # unresolved
    got = {g.grade(s) for s in g.label}
    want = {AL.CONFIRMED, AL.SCOPE_INFERRED, AL.HEURISTIC, AL.OPEN_TIE, AL.UNRESOLVED}
    assert got == want, f"{got}"


def case_coverage_equation():
    """자리 수 = 등급별 합. 어긋나면 어떤 자리가 어디에도 안 세어진 것이다."""
    g = AL.Ledger()
    for i in range(50):
        g.record((i, "o", 0, 0), "x", "scoped_symbol",
                 raw=("a", "b") if i % 3 == 0 else None,
                 scoped=("a",) if i % 3 == 0 else None)
    occ, _ = g.counts()
    assert sum(occ.values()) == len(g.label) == 50, (dict(occ), len(g.label))
    assert g.coverage_ok()


def case_heuristic_never_hides_in_confirmed():
    """값이 안 겹쳐도 **지어낸 이름**은 확정이 아니다.

    처음에 "충돌이 없으면 확정" 으로 짰다가 tiny-deepseek-v3 의 131축(지어낸 이름 119 +
    이름 없음 12)이 확정에 숨었다(2026-09-11 실측).
    """
    g = AL.Ledger()
    for kind in ("heur_half", "heur_multiple", "heur_product", "reused_symbol"):
        g.record((hash(kind) % 1000, "i", 0, 0), "made_up", kind)
    assert all(g.grade(s) == AL.HEURISTIC for s in g.label), \
        {g.s(g.reason[s]): g.grade(s) for s in g.label}
    g2 = AL.Ledger()
    g2.record((1, "i", 0, 0), "512", "bare")
    assert g2.grade((1, "i", 0, 0)) == AL.UNRESOLVED


def case_anchor_promotes_but_keeps_history():
    """앵커가 덮어쓰면 확정이 된다. 그래도 **값이 겹쳤다는 사실은 남는다**."""
    site = (7, "o", 0, 1)
    g = AL.Ledger()
    g.record(site, "n_h", "scoped_symbol", raw=("n_h", "n_kv"), scoped=("n_h", "n_kv"))
    assert g.grade(site) == AL.OPEN_TIE
    g.overwrite(site, "n_kv", AL.ANCHOR_DECLARED_WIDTH)
    assert g.grade(site) == AL.CONFIRMED
    assert site in g.raw_cands, "덮어쓰면서 충돌 기록이 사라졌다"
    chain = [(g.s(a), g.s(b)) for a, b in g.chain[site]]
    assert chain == [("scoped_symbol", "n_h"), (AL.ANCHOR_DECLARED_WIDTH, "n_kv")], chain


def case_questions_fold():
    """같은 `(후보, 라벨, 등급)` 은 **질문 하나**다. 수천 축이 한 답으로 채워진다."""
    g = AL.Ledger()
    for i in range(300):
        g.record((i, "i", 0, 0), "n_h", "scoped_symbol",
                 raw=("n_h", "n_kv"), scoped=("n_h", "n_kv"))
    g.record((999, "i", 0, 0), "E", "scoped_symbol", raw=("E", "k"), scoped=("E", "k"))
    _occ, qs = g.counts()
    assert len(qs) == 2, dict(qs)
    assert max(qs.values()) == 300, dict(qs)


def case_sidecar_roundtrip():
    """사이드카는 **확정이 아닌 자리만** 적는다. 요약과 질문이 함께 나온다."""
    tmp = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(tmp, "full"))
        g = AL.Ledger()
        g.record((1, "i", 0, 0), "d_model", "scoped_symbol")
        g.record((2, "o", 0, 1), "n_h", "scoped_symbol",
                 raw=("n_h", "n_kv"), scoped=("n_h", "n_kv"))
        path, occ, nq = g.write(tmp, "prefill")
        recs = [json.loads(l) for l in open(path, encoding="utf-8")]
        kinds = [r["kind"] for r in recs]
        assert kinds[0] == "summary" and recs[0]["coverage_ok"] is True, recs[0]
        assert nq == 1 and kinds.count("question") == 1, kinds
        sites = [r for r in recs if r["kind"] == "site"]
        assert len(sites) == 1 and sites[0]["op_id"] == 2, sites
        assert sites[0]["grade"] == AL.OPEN_TIE
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


CASES = [case_grades_are_reachable, case_coverage_equation,
         case_heuristic_never_hides_in_confirmed, case_anchor_promotes_but_keeps_history,
         case_questions_fold, case_sidecar_roundtrip]


def main():
    bad = []
    for fn in CASES:
        try:
            fn()
            print(f"  PASS      {fn.__name__:42} {(fn.__doc__ or '').splitlines()[0][:44]}")
        except AssertionError as e:
            bad.append(fn.__name__)
            print(f"  **FAIL**  {fn.__name__:42} {str(e).splitlines()[0][:60]}")
    print()
    print(f"{len(CASES) - len(bad)}/{len(CASES)} 통과" + (f"  실패: {bad}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
