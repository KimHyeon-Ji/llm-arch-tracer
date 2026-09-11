r"""**어느 축이 진짜 배치인가.** 같은 모델을 B=1 과 B=2 로 트레이스해 견준다.

왜 필요한가
-----------
`dim()` 은 크기 1 축이면 무조건 `B` 를 답한다(`symbolic_shape.py` 의
`if n == 1: return _r("runtime", "B")`). 그래서 **`B` 로 찍힌 축의 구체값은 항상 1** 이고,
"값이 1 이 아닌 B 가 있는가" 같은 검사는 반례를 찾을 수 없다 -- 2026-09-11 에 그것을 반례
검사라고 내세웠다가 외부 검토에 걸렸다.

`transpose`/`permute` 를 지나면 `[T,B,d]` 나 `[n_h,B,T,d]` 가 실재할 수 있고, B=1 이면 값으로
구분할 방법이 없다. **크기가 배치를 따라 변하는지**만이 판별식이다.

이 프로브는 산출물을 만들지 않는다. 두 트레이스를 메모리에서 비교해 자리마다 판정한다.

    batch_dependent    B=2 에서 크기가 2배 -> 진짜 배치 축
    batch_invariant    그대로 1 -> 배치가 아니다. `B` 라면 틀린 이름이다
    shape_changed      rank 나 op 구성이 달라져 대조 불가

실행:
    .venv\Scripts\python.exe develop\batch_axis_probe.py develop\models\test-llama4-maverick.yaml
"""
import argparse
import collections
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJ, "src"))
if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import yaml                                  # noqa: E402
import provenance                            # noqa: E402
import run as R                               # noqa: E402


def trace(profile, batch):
    cfg, _prov = provenance.snapshot(
        profile["model_id"], profile.get("revision"),
        config_overrides=profile.get("config_overrides"))
    _sl = profile.get("seq_len")
    ctx = R.RunContext(cfg, profile["model_id"], profile.get("revision"),
                       seq_len=_sl if isinstance(_sl, int) else None, batch=batch)
    out = {}
    for phase in profile.get("phases", ["prefill", "decode"]):
        out[phase] = ctx.run_once(phase)
    return out


def compare(a_rows, b_rows):
    """자리 -> 판정. `a` 는 B=1, `b` 는 B=2 다.

    **`op_id` 로 짝짓지 않는다.** 두 트레이스의 op 개수가 다르면(MoE 라우팅은 배치에 따라
    전문가별 토큰 수가 달라진다) 같은 번호가 다른 op 를 가리킨다. `(module_path, raw_op)`
    별로 **등장 순서**를 세어 짝을 만든다.

    빈 축(크기 0)은 제외한다 -- `0 == 0 * 2` 가 참이라 배치 의존으로 오분류된다.
    """
    def index(rows):
        seen, out = collections.Counter(), {}
        for r in rows:
            k = (r.get("module_path") or "", r.get("raw_op") or "")
            out[(k, seen[k])] = r
            seen[k] += 1
        return out

    ia, ib = index(a_rows), index(b_rows)
    verdict, unmatched = {}, 0
    for key, r in ia.items():
        o = ib.get(key)
        if o is None:
            unmatched += 1
            continue
        for fld, tag in (("input_shape", "i"), ("output_shape", "o")):
            sa, sb = r.get(fld) or [], o.get(fld) or []
            if len(sa) != len(sb):
                unmatched += 1
                continue
            for si, (x, y) in enumerate(zip(sa, sb)):
                if not (isinstance(x, list) and isinstance(y, list)) or len(x) != len(y):
                    continue
                for ax, (u, v) in enumerate(zip(x, y)):
                    if not (isinstance(u, int) and isinstance(v, int)) or u <= 0:
                        continue          # 빈 축은 0 == 0*2 로 오분류된다
                    site = (r["op_id"], tag, si, ax)
                    if v == u * 2:
                        verdict[site] = "batch_dependent"
                    elif v == u:
                        verdict[site] = "batch_invariant"
                    else:
                        verdict[site] = "other"
    return verdict, unmatched


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("profile")
    ap.add_argument("--show", type=int, default=8)
    a = ap.parse_args()
    profile = yaml.safe_load(open(a.profile, encoding="utf-8"))

    print(f"B=1 트레이스 …")
    one = trace(profile, 1)
    print(f"B=2 트레이스 …")
    two = trace(profile, 2)

    bad_pairs, total_sites = 0, 0
    for phase in sorted(one):
        if phase not in two:
            continue
        v, unmatched = compare(one[phase], two[phase])
        bad_pairs += unmatched
        total_sites += len(v)
        c = collections.Counter(v.values())
        print(f"\n=== {phase}: 자리 {len(v):,}  {dict(c)}")
        # **배치를 따라 변하는 축이 축 0 이 아닌 자리** -- 여기가 핵심이다.
        by_id = {r.get("op_id"): r for r in one[phase]}
        off0 = collections.Counter()
        for (oid, tag, si, ax), g in v.items():
            if g == "batch_dependent" and ax > 0:
                r = by_id.get(oid) or {}
                fld = "input_shape" if tag == "i" else "output_shape"
                sh = (r.get(fld) or [])
                sh = sh[si] if si < len(sh) else None
                off0[(r.get("op_type"), tuple(str(z) for z in (sh or ())), ax)] += 1
        print(f"   **축 0 이 아닌 자리의 진짜 배치 축: {sum(off0.values()):,}개**")
        for k, n in off0.most_common(a.show):
            print(f"      {n:6,}  {str(k[0]):18} {list(k[1])} 축{k[2]}")

        # `B` 라벨 검사는 **여기서 못 한다.** `run_once()` 는 심볼화 **전** 행을 돌려주므로
        # shape 이 전부 정수다. 예전에 `str(sh[ax]) == "B"` 를 넣어 뒀는데 항상 0 이 나오는
        # 죽은 검사였다(외부 검토 2026-09-11이 짚었다). 라벨 대조는 산출물 쪽에서 한다.

    # **짝을 못 지은 행이 많으면 결론을 못 낸다.** MoE 는 배치가 바뀌면 전문가별 토큰 수가
    # 달라져 op 구성이 조금 달라지므로 0 을 요구할 수는 없다. 비율로 본다 -- 1% 를 넘으면
    # 비교 대상이 흔들린 것이라 판정을 내지 않는다.
    share = bad_pairs / max(total_sites + bad_pairs, 1)
    print(chr(10) + f"짝을 못 지은 행 {bad_pairs:,}개 ({share:.2%})")
    if share > 0.01:
        print("**1% 를 넘는다 -- 두 트레이스의 op 구성이 달라 판정을 신뢰할 수 없다.**")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
