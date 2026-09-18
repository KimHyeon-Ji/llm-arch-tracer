r"""**어느 축이 진짜 배치인가.** 같은 모델을 B=1 과 B=2 로 트레이스해 견준다.

왜 필요한가
-----------
`dim()` 은 크기 1 축이면 무조건 `B` 를 답한다(`symbolic_shape.py` 의
`if n == 1: return _r("runtime", "B")`). 그래서 **`B` 로 찍힌 축의 구체값은 항상 1** 이고,
"값이 1 이 아닌 B 가 있는가" 같은 검사는 반례를 찾을 수 없다 -- 2026-09-11 에 그것을 반례
검사라고 내세웠다가 외부 검토에 걸렸다.

`transpose`/`permute` 를 지나면 `[T,B,d]` 나 `[n_h,B,T,d]` 가 실재할 수 있고, B=1 이면 값으로
구분할 방법이 없다. **크기가 배치를 따라 변하는지**만이 판별식이다.

무엇이 통과 조건인가
--------------------
비율이 아니라 **의미**로 판정한다(외부 검토 2026-09-12). 축 0 이 아닌 자리에서 배치를 따라
변하는 축은 MoE 의 routed expert bmm 뿐이어야 한다 -- 거기서는 축 1 이 `B*T` 라 정당하다.
그 밖의 자리에서 나오면 레이아웃 가정이 틀린 것이므로 **실패**다.

미정합은 **양방향**으로 센다. B=1 에만 있는 op 과 B=2 에만 있는 op 을 둘 다 봐야 한다 --
한쪽만 순회하면 다른 쪽을 통째로 놓친다(그렇게 72개로 과소계상했다). 비율은 **op 단위끼리**
계산한다(행 수를 축 수로 나누면 안 된다). 비율은 회귀 경보일 뿐 주 판정이 아니다.

실행:
    .venv\Scripts\python.exe develop\batch_axis_probe.py develop\models\test-llama4-maverick.yaml
"""
import argparse
import collections
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJ, "src"))
if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import yaml                                     # noqa: E402
import adapt
import provenance                                # noqa: E402
import run as R                                  # noqa: E402

# 축 0 이 아닌 자리에서 배치를 따라 변해도 되는 **유일한** 형태. MoE 의 routed expert 는
# `[E, B*T, d]` 로 토큰을 모으므로 축 1 이 배치에 비례한다.
# routed expert 경로는 토큰을 `[E, B*T, d]` 로 모으므로 **축 1 이 배치에 비례한다.** 그
# 레이아웃은 bmm 하나가 아니라 view/split/mul/silu/sum/transpose 사슬 전체를 타고 흐른다
# (실측으로 확인했다). op 종류가 아니라 **경로와 축 번호**가 허용 근거다.
ALLOWED_OFFSET0 = {"module_re": re.compile(r"experts|expert_mlp|moe|feed_forward", re.I),
                   "axis": 1}

# 배치를 바꾸면 **메모리 연속성**이 달라져 reshape 경로가 갈린다 -- B=1 에서 `view` 로 되던
# 것이 B=2 에서는 `clone` + `_unsafe_view` 가 된다. op 개수가 달라지는 것은 그 때문이지
# 모델이 다르게 도는 것이 아니다(실측: prefill 972개 전부 self_attn, decode 120개 전부
# feed_forward, 전부 이 세 op 의 개수 차이였다). 그래서 이 계열의 밀림은 **설명된 것**으로
# 따로 세고, 그 밖의 밀림만 경보한다.
RESHAPE_OPS = {"aten.view.default", "aten._unsafe_view.default", "aten.clone.default",
               "aten.contiguous.default", "aten.reshape.default", "aten._reshape_alias.default"}
UNEXPLAINED_DRIFT_WARN = 0.01


def trace(profile, batch, seq_len=None, revision=None):
    """`seq_len` 은 **발행 트레이스가 쓴 T** 를 그대로 넘겨야 한다.

    안 넘기면 `RunContext` 가 `resolve_seq_len` 으로 다시 고르는데, 발행은
    `resolve_capture_sizes` 로 `(B, T)` 를 함께 골라서 **T 가 다를 수 있다**. 그러면 배치만
    다른 두 트레이스를 비교하려던 것이 T 까지 다른 비교가 되어, 라벨이 맞는데도 전부
    불일치로 나온다 -- V4-Pro 가 발행 2049 / 프로브 2048 로 그렇게 됐다(2026-09-14).
    """
    # **후보가 실제로 쓴 revision 에 고정한다.** 프로파일의 `revision` 은 보통 null(떠 있는
    # 참조)이라, 그대로 쓰면 그 사이 업스트림이 바뀐 다른 모델을 재 볼 수 있다. 그러면 배치만
    # 다른 두 트레이스를 비교하려던 것이 모델까지 다른 비교가 된다(외부 검토 2026-09-18).
    _rev = revision or profile.get("revision")
    cfg, _prov = provenance.snapshot(
        profile["model_id"], _rev,
        config_overrides=profile.get("config_overrides"))
    _sl = seq_len if isinstance(seq_len, int) else profile.get("seq_len")
    ctx = R.RunContext(cfg, profile["model_id"], _rev,
                       seq_len=_sl if isinstance(_sl, int) else None, batch=batch,
                       seq_len_multiple=profile.get("seq_len_multiple"))
    # **`run_once` 를 직접 부르면 안 된다.** 발행 트레이스는 `trace_adaptive` 를 거쳐
    # 재시도 remedy 를 적용한다(gpt-oss 의 MoE `grouped_mm` 은 bf16 이 아니면 죽는다).
    # 프로브가 그 경로를 건너뛰면 발행본은 되는데 검증만 실패한다(2026-09-14 실측).
    out = {}
    for phase in profile.get("phases", ["prefill", "decode"]):
        rows, _applied = adapt.trace_adaptive(ctx, phase)
        out[phase] = rows
    return out


def _groups(rows):
    """`(module_path, raw_op)` -> 그 순서대로의 행 목록."""
    out = collections.defaultdict(list)
    for r in rows:
        out[(r.get("module_path") or "", r.get("raw_op") or "")].append(r)
    return out


def _index(rows):
    """`((module_path, raw_op), 등장순서)` -> 행."""
    out = {}
    for k, rs in _groups(rows).items():
        for i, r in enumerate(rs):
            out[(k, i)] = r
    return out


def compare(a_rows, b_rows):
    """(자리별 판정, 통계). `a` 는 B=1, `b` 는 B=2 다."""
    ga, gb = _groups(a_rows), _groups(b_rows)
    # **개수가 다른 그룹은 짝짓지 않는다.** 순서로 맞추면 밀려서 서로 다른 op 를 비교하게
    # 되고, 그 결과가 "2배도 그대로도 아닌 축" 으로 둔갑한다. Llama-4 prefill 에서 864건이
    # 그렇게 나왔는데 실제로는 `[1,40,128,16]` 과 `[2,40,16,16]` 처럼 아예 다른 op 였다
    # (외부 검토 2026-09-12 가 양방향 계상을 요구한 끝에 드러났다).
    pairable = {k for k in set(ga) & set(gb) if len(ga[k]) == len(gb[k])}
    drifted = {k for k in set(ga) & set(gb) if len(ga[k]) != len(gb[k])}
    # 연속성 때문에 갈린 reshape 경로인가, 설명 안 되는 밀림인가
    explained = {k for k in drifted if k[1] in RESHAPE_OPS}
    stat = {"a_only": sum(len(ga[k]) for k in set(ga) - set(gb)),   # **양방향**으로 센다
            "b_only": sum(len(gb[k]) for k in set(gb) - set(ga)),
            "drift_reshape": sum(max(len(ga[k]), len(gb[k])) for k in explained),
            "drift_unexplained": sum(max(len(ga[k]), len(gb[k]))
                                     for k in drifted - explained),
            "matched_ops": sum(len(ga[k]) for k in pairable),
            "rank_mismatch": 0}
    ia, ib = _index(a_rows), _index(b_rows)
    verdict = {}
    for key in sorted(k for k in set(ia) & set(ib) if k[0] in pairable):
        r, o = ia[key], ib[key]
        for fld, tag in (("input_shape", "i"), ("output_shape", "o")):
            sa, sb = r.get(fld) or [], o.get(fld) or []
            if len(sa) != len(sb):
                stat["rank_mismatch"] += 1
                continue
            for si, (x, y) in enumerate(zip(sa, sb)):
                if not (isinstance(x, list) and isinstance(y, list)):
                    continue
                if len(x) != len(y):
                    stat["rank_mismatch"] += 1
                    continue
                for ax, (u, v) in enumerate(zip(x, y)):
                    if not (isinstance(u, int) and isinstance(v, int)) or u <= 0:
                        continue          # 빈 축은 0 == 0*2 로 오분류된다
                    site = (r["op_id"], tag, si, ax)
                    if v == u * 2:
                        verdict[site] = ("batch_dependent", r)
                    elif v == u:
                        verdict[site] = ("batch_invariant", r)
                    else:
                        verdict[site] = ("other", r)
    return verdict, stat


def _offset0_violations(verdict):
    """축 0 이 아닌 배치 의존 축 중 **허용 형태가 아닌 것**. 하나라도 있으면 실패다."""
    ok, bad = collections.Counter(), collections.Counter()
    for (oid, tag, si, ax), (g, r) in verdict.items():
        if g != "batch_dependent" or ax == 0:
            continue
        mp, raw = r.get("module_path") or "", r.get("raw_op") or ""
        fits = (ALLOWED_OFFSET0["module_re"].search(mp)
                and ax == ALLOWED_OFFSET0["axis"])
        (ok if fits else bad)[(mp.split(".")[-1][:24], raw, ax)] += 1
    return ok, bad


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("profile")
    ap.add_argument("--show", type=int, default=6)
    a = ap.parse_args()
    profile = yaml.safe_load(open(a.profile, encoding="utf-8"))

    print("B=1 트레이스 …")
    one = trace(profile, 1)
    print("B=2 트레이스 …")
    two = trace(profile, 2)

    fails, tot = [], collections.Counter()
    for phase in sorted(one):
        if phase not in two:
            continue
        verdict, stat = compare(one[phase], two[phase])
        kinds = collections.Counter(g for g, _ in verdict.values())
        tot.update(stat)
        tot.update(kinds)
        print(f"\n=== {phase}: 자리 {len(verdict):,}  {dict(kinds)}")
        print(f"   op 짝: 맞음 {stat['matched_ops']:,} / B1 에만 {stat['a_only']:,} / "
              f"B2 에만 {stat['b_only']:,} / 밀림(reshape 경로) {stat['drift_reshape']:,} / "
              f"**밀림(미설명) {stat['drift_unexplained']:,}** / "
              f"rank 불일치 {stat['rank_mismatch']:,}")

        ok, bad = _offset0_violations(verdict)
        print(f"   축 0 밖 배치 의존: 허용 {sum(ok.values()):,} / **위반 {sum(bad.values()):,}**")
        for k, n in ok.most_common(a.show):
            print(f"      허용  {n:6,}  {k[0]:24} {k[1]} 축{k[2]}")
        for k, n in bad.most_common(a.show):
            print(f"      **위반** {n:6,}  {k[0]:24} {k[1]} 축{k[2]}")
        if bad:
            fails.append(f"{phase}: 축 0 밖 배치 의존이 MoE routed bmm 이 아닌 자리에 "
                         f"{sum(bad.values()):,}개")
        if kinds.get("other"):
            fails.append(f"{phase}: 2배도 그대로도 아닌 축 {kinds['other']:,}개 "
                         f"-- 배치 외의 무언가가 함께 바뀌었다")
        if stat["rank_mismatch"]:
            fails.append(f"{phase}: rank 불일치 {stat['rank_mismatch']:,}건")

    # 비율은 **op 단위끼리**. 회귀 경보일 뿐 주 판정이 아니다.
    # 짝을 못 지은 op 은 **세 갈래** 다: B1 에만, B2 에만, 그리고 그룹 개수가
    # 어긋나 아예 짝짓지 않은 것. 셋 다 세야 한다.
    miss = (tot["a_only"] + tot["b_only"] + tot["drift_reshape"]
            + tot["drift_unexplained"])
    denom = tot["matched_ops"] + miss
    share = tot["drift_unexplained"] / max(denom, 1)
    print(chr(10) + f"짝 못 지은 op {miss:,} / {denom:,} "
          f"(reshape 경로 {tot['drift_reshape']:,} + 한쪽에만 "
          f"{tot['a_only'] + tot['b_only']:,} + **미설명 {tot['drift_unexplained']:,}**)")
    print(f"미설명 비율 {share:.2%}  (경보선 {UNEXPLAINED_DRIFT_WARN:.0%})")
    if share > UNEXPLAINED_DRIFT_WARN:
        fails.append(f"설명 안 되는 밀림 {tot['drift_unexplained']:,}개 ({share:.2%}) "
                     f"-- 두 트레이스가 다르게 돌았다")

    if fails:
        print("\n" + chr(10).join("**FAIL** " + f for f in fails))
        return 1
    print(chr(10) + "PASS -- 축 0 밖 배치 의존은 전부 routed expert 경로의 "
          + "`[E, B*T, d]` 축 1 이다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
