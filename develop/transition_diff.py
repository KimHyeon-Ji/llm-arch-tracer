r"""B=1 판과 비퇴화 배치 판의 **라벨 차이를 원인별로 가른다.**

왜 필요한가
-----------
47개를 전량 재트레이스하면 라벨이 대거 바뀐다. 기준선을 그대로 다시 박으면 **진짜 회귀가
배치 변화에 묻힌다**(외부 검토 2026-09-13). 그래서 차이를 분류해 사람이 볼 것만 남긴다.

    batch_expected   B=1 을 대입하면 옛 식과 값이 같고, 비배치 심볼 구성도 그대로다
                     예: `n_h` -> `B*n_h`, `T` -> `B*T`, `E*T` -> `B*E*T`
    literal_resolved 옛 판이 정수였는데 이름이 붙었다(배치가 드러나 가릴 수 있게 됐다)
    singleton_fixed  옛 판이 `B` 였는데 `1` 이 됐다 -- 방송 싱글턴을 배치로 오인했던 자리
    alignment_drift  그 그룹의 라벨 다중집합이 그대로다 -- 자리만 밀렸지 의미는 그대로
    semantic_change  비배치 심볼 구성이 바뀌었다. **사람이 봐야 한다**
    unexplained      위 어디에도 안 맞는다. **사람이 봐야 한다**

`op_id` 로 짝짓지 않는다 -- 배치가 바뀌면 op 구성이 조금 달라져 번호가 밀린다.
`(module_key, raw_op, 등장순서, field, shape_index, axis)` 로 맞춘다.

실행:
    .venv\Scripts\python.exe develop\transition_diff.py <모델이름> [--new develop/out]
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
if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import dim_expr as DE                     # noqa: E402
from anchors import module_key            # noqa: E402

_FIELDS = (("input_shape", "i"), ("output_shape", "o"), ("weight_shape", "w"))


def _axes(model_dir: str, phase: str):
    """{안정 키: 라벨}. 키에 `op_id` 를 쓰지 않는다."""
    p = os.path.join(model_dir, "full", f"{phase}.trace.raw.jsonl")
    if not os.path.exists(p):
        return {}
    seen, out = collections.Counter(), {}
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        # **레이어 번호를 접지 않는다.** `module_key` 로 접으면 서수가 전 레이어에 걸쳐
        # 세어져, 한 레이어에서 op 하나가 늘면 그 뒤 전부가 밀린다 -- `n_h -> T` 와
        # `T -> n_h` 가 비슷한 수로 함께 나오는 것이 그 증상이었다(2026-09-13).
        gk = (r.get("module_path") or "(root)", r.get("raw_op") or "")
        n = seen[gk]
        seen[gk] += 1
        for fld, tag in _FIELDS:
            v = r.get(fld)
            if v is None:
                continue
            shapes = [v] if fld == "weight_shape" else v
            for si, sh in enumerate(shapes):
                if not isinstance(sh, list):
                    continue
                for ax, lab in enumerate(sh):
                    # **rank 를 키에 넣는다.** 같은 서수라도 rank 가 다르면 다른 op 다.
                    # 안 넣었더니 4-D 와 3-D 가 짝지어져 `T -> d_head` 와 `d_head -> T` 가
                    # 192개씩 대칭으로 나왔다 -- 의미 변화가 아니라 밀림이다(2026-09-13).
                    # 안 맞는 자리는 "한쪽에만" 으로 떨어진다. 그게 정직하다.
                    out[(gk[0], gk[1], n, tag, si, len(sh), ax)] = str(lab)
    return out


def classify(old_lab, new_lab, old_ns, new_ns1):
    """이 차이는 무엇 때문인가."""
    if old_lab == new_lab:
        return None
    ov, nv1 = DE.evaluate(old_lab, old_ns), DE.evaluate(new_lab, new_ns1)
    old_syms = DE.free_symbols(old_lab) - {"B"}
    new_syms = DE.free_symbols(new_lab) - {"B"}

    if old_lab == "B" and new_lab == "1":
        return "singleton_fixed"
    if old_lab.lstrip("-").isdigit() and not new_lab.lstrip("-").isdigit():
        # 정수였던 것이 이름을 얻었다. B=1 대입값이 같아야 한다.
        return "literal_resolved" if (ov is not None and ov == nv1) else "unexplained"
    # **B=1 을 대입하면 같은 값이고, 비배치 심볼 구성도 그대로다.**
    if ov is not None and nv1 is not None and ov == nv1 and old_syms == new_syms:
        if DE.batch_degree(new_lab, new_ns1) != DE.batch_degree(old_lab, old_ns):
            return "batch_expected"
        return "unexplained"        # 값도 심볼도 같은데 문자열이 다르다 -- 표기 흔들림
    if old_syms != new_syms:
        return "semantic_change"
    return "unexplained"


def run(model, new_root, show=8):
    old_dir = os.path.join(PROJ, "models", model)
    new_dir = os.path.join(new_root, model)
    if not (os.path.isdir(old_dir) and os.path.isdir(new_dir)):
        print(f"{model}: 옛 판 또는 새 판이 없다")
        return 1
    old_prov, new_prov = DE.load_provenance(old_dir), DE.load_provenance(new_dir)
    old_b = int(old_prov.get("capture_batch") or 1)
    old_ns = DE.namespace(old_prov, old_b)
    new_ns1 = DE.namespace(new_prov, 1)       # 새 라벨을 B=1 에서 평가 -> 옛 판과 비교
    print(f"=== {model}   옛 B={old_b} -> 새 B={new_prov.get('capture_batch')}")

    tot = collections.Counter()
    samples = collections.defaultdict(collections.Counter)
    for phase in ("prefill", "decode"):
        a, b = _axes(old_dir, phase), _axes(new_dir, phase)
        common = set(a) & set(b)
        tot["자리(양쪽에 있음)"] += len(common)
        tot["옛 판에만"] += len(set(a) - set(b))
        tot["새 판에만"] += len(set(b) - set(a))
        # **한 (모듈, op, rank) 그룹의 라벨 다중집합이 그대로면 순서가 밀린 것이다.**
        # 라벨이 추가되거나 사라지지 않았는데 자리만 서로 바뀌었다면 의미 변화가 아니다 --
        # `d_head -> T` 와 `T -> d_head` 가 192개씩 대칭으로 나오는 것이 그 증상이다.
        # 배치를 바꾸면 연속성 때문에 op 가 끼어들어 서수가 밀린다(2026-09-13).
        grp_old, grp_new = collections.defaultdict(collections.Counter),             collections.defaultdict(collections.Counter)
        for k in common:
            g = (k[0], k[1], k[5])
            grp_old[g][a[k]] += 1
            grp_new[g][b[k]] += 1
        drifted = {g for g in grp_old if grp_old[g] == grp_new[g]}

        for k in common:
            c = classify(a[k], b[k], old_ns, new_ns1)
            if c is None:
                tot["같음"] += 1
                continue
            if c in ("semantic_change", "unexplained") and (k[0], k[1], k[5]) in drifted:
                c = "alignment_drift"
            tot[c] += 1
            samples[c][(a[k], b[k], k[1])] += 1

    for k, v in tot.most_common():
        print(f"   {k:22} {v:>9,}")
    review = tot["semantic_change"] + tot["unexplained"]
    for kind in ("semantic_change", "unexplained", "alignment_drift", "batch_expected",
                 "literal_resolved", "singleton_fixed"):
        if not samples[kind]:
            continue
        print(f"\n   --- {kind}")
        for (o, n, op), cnt in samples[kind].most_common(show):
            print(f"      {cnt:7,}  `{o}` -> `{n}`   {op}")
    print(f"\n사람이 봐야 하는 것: **{review:,}**")
    return 1 if review else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model")
    ap.add_argument("--new", default=os.path.join(HERE, "out"))
    ap.add_argument("--show", type=int, default=8)
    a = ap.parse_args()
    return run(a.model, a.new, a.show)


if __name__ == "__main__":
    sys.exit(main())
