r"""**같은 라벨을 B=1 과 B=2 로 평가해** 실제 shape 과 전건 비교한다.

왜 필요한가
-----------
B=1 로 트레이스하면 `T` 와 `B*T`, `n_h` 와 `B*n_h` 가 같은 값이다. 그래서 라벨이 접힌
배치를 빠뜨려도 **산술적으로 참**이고, 기존 `label_false` 검사는 통과시킨다. 실제로
Llama-4 의 배치 의존 축 3,604개가 라벨에 `B` 를 안 담고 있었다(외부 검토 2026-09-13).

B=2 로 한 번 더 트레이스해서, **B=1 이 만든 그 라벨**을 B=2 심볼표로 평가해 B=2 의 실제
shape 과 비교한다. 라벨을 다시 만들지 않는 것이 핵심이다 -- 다시 만들면 두 번 독립으로
오판할 수 있다.

    eval(label, B=1) == B=1 concrete      (기존 검사와 같다)
    eval(label, B=2) == B=2 concrete      (접힌 배치를 여기서 잡는다)

정적 방어선도 함께 본다: 텐서 하나의 배치 차수 합은 **1 을 넘을 수 없다**.
`[B, B*T, d]` 는 차수합 2 라 실패한다(기존 `batch_excl` 은 문자열 `== "B"` 만 봐서 통과시킨다).

실행:
    .venv\Scripts\python.exe develop\check_batch_labels.py develop\models\test-llama4-maverick.yaml
"""
import argparse
import collections
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJ, "src"))
sys.path.insert(0, HERE)
if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import yaml                                      # noqa: E402
import dim_expr as DE                            # noqa: E402
import batch_axis_probe as P                     # noqa: E402


def check(profile, model_dir, show=8):
    """발행된 **심볼 라벨**을 B=2 구체 shape 과 맞춰 본다.

    `run_once()` 가 돌려주는 행은 심볼화 **전**이라 전부 정수다. 그걸로 검사하면 라벨이
    아니라 숫자를 비교하게 된다(처음에 그렇게 짰다). 심볼은 발행된
    `full/<phase>.trace.raw.jsonl` 에 있으므로 그것과 B=2 트레이스를
    `(module_path, raw_op, 등장순서)` 로 짝짓는다.
    """
    prov = DE.load_provenance(model_dir)
    # **발행 배치와 다른 값으로 검증한다.** 같은 값이면 아무것도 대조하지 못한다.
    # 옛 산출물은 `capture_batch` 가 없으니 1 로 본다.
    b_pri = int(prov.get("capture_batch") or 1)
    b_probe = b_pri + 1
    ns2 = DE.namespace(prov, b_probe)

    print(f"발행 B={b_pri} / 검증 B={b_probe} -- 검증 트레이스 …")
    # **발행이 쓴 T 를 그대로 넘긴다.** 안 넘기면 프로브가 T 를 다시 골라 배치 말고 T 도
    # 달라진 비교가 된다.
    two = P.trace(profile, b_probe, seq_len=prov.get("seq_len_used"))

    fails, tot = [], collections.Counter()
    for phase in sorted(two):
        pub_path = os.path.join(model_dir, "full", f"{phase}.trace.raw.jsonl")
        if not os.path.exists(pub_path):
            print(f"   {phase}: 발행 트레이스가 없다 -- 건너뛴다")
            continue
        pub = [json.loads(l) for l in open(pub_path, encoding="utf-8")]
        gp, gb = P._groups(pub), P._groups(two[phase])
        pairable = {k for k in set(gp) & set(gb) if len(gp[k]) == len(gb[k])}
        skipped = sum(max(len(gp.get(k, [])), len(gb.get(k, [])))
                      for k in (set(gp) | set(gb)) - pairable)

        wrong, degree, seen = collections.Counter(), collections.Counter(), 0
        for k in pairable:
            for r, o in zip(gp[k], gb[k]):
                for fld in ("input_shape", "output_shape"):
                    sa, sb = r.get(fld) or [], o.get(fld) or []
                    for x, y in zip(sa, sb):
                        if not (isinstance(x, list) and isinstance(y, list)) or len(x) != len(y):
                            continue
                        d = DE.shape_batch_degree(x, DE.namespace(prov, b_pri))
                        if d is not None and d > 1:
                            degree[(r.get("raw_op"), tuple(str(z) for z in x), d)] += 1
                        for ax, (lab, cb) in enumerate(zip(x, y)):
                            if not isinstance(cb, int):
                                continue
                            v2 = DE.evaluate(lab, ns2)
                            if v2 is None:
                                continue
                            seen += 1
                            if v2 != cb:
                                wrong[(r.get("raw_op"), str(lab), ax, v2, cb)] += 1
        print(chr(10) + f"=== {phase}: 검사한 축 {seen:,}  (짝 못 지은 op {skipped:,})")
        print(f"   **검증 배치에서 라벨이 실제 shape 과 다름: {sum(wrong.values()):,}**")
        for kk, n in wrong.most_common(show):
            print(f"      {n:6,}  {str(kk[0]):26} `{kk[1]}` 축{kk[2]}  "
                  f"라벨값 {kk[3]} vs 실제 {kk[4]}")
        print(f"   배치 차수 합 > 1 인 shape: {sum(degree.values()):,}")
        for kk, n in degree.most_common(4):
            print(f"      {n:6,}  {str(kk[0]):26} {list(kk[1])} 차수 {kk[2]}")
        tot["wrong"] += sum(wrong.values())
        tot["degree"] += sum(degree.values())
        if wrong:
            fails.append(f"{phase}: 검증 배치에서 어긋난 라벨 {sum(wrong.values()):,}개")
        if degree:
            fails.append(f"{phase}: 배치 차수 합이 1 을 넘는 shape {sum(degree.values()):,}개")
        # **짝을 못 지은 op 는 검사되지 않은 op 다.** 출력만 하고 통과시키면 "검사했다" 가
        # 아니라 "안 본 것이 있다" 가 된다(외부 검토 2026-09-13).
        if skipped:
            fails.append(f"{phase}: 짝을 못 지은 op {skipped:,}개 -- 검증되지 않았다")

    if fails:
        print(chr(10) + chr(10).join("**FAIL** " + f for f in fails))
        return 1
    print(chr(10) + f"PASS -- 발행 라벨(B={b_pri})이 B={b_probe} 실제 shape 과도 맞는다.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("profile")
    ap.add_argument("--model-dir")
    a = ap.parse_args()
    profile = yaml.safe_load(open(a.profile, encoding="utf-8"))
    md = a.model_dir or os.path.join(PROJ, "models",
                                     profile["model_id"].replace("/", "__"))
    return check(profile, md)


if __name__ == "__main__":
    sys.exit(main())
