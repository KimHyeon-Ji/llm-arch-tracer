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


def _ops(model_dir: str, phase: str, ns1):
    """모듈별 op 목록. 각 op 은 **B=1 로 정규화한 구체 shape** 을 서명으로 갖는다.

    서수로 짝지으면 안 된다 -- 배치를 바꾸면 연속성 때문에 `view` 가 `clone`+`_unsafe_view`
    로 갈려 op 이 끼어들고, 그 뒤가 전부 밀린다. 그 밀림이 `T -> d_head`(16 -> 128) 같은
    "있을 수 없는 재라벨" 로 나타났다(외부 검토 2026-09-13).

    새 판은 비퇴화 배치로 잡혔지만 **라벨을 B=1 로 평가하면** 옛 판과 같은 좌표가 된다.
    그 값으로 서명을 만들어 맞춘다.
    """
    p = os.path.join(model_dir, "full", f"{phase}.trace.raw.jsonl")
    if not os.path.exists(p):
        return {}
    per = collections.defaultdict(list)
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        shapes = {}
        for fld, tag in _FIELDS:
            v = r.get(fld)
            if v is None:
                continue
            group = [v] if fld == "weight_shape" else v
            got = []
            for sh in group:
                if not isinstance(sh, list):
                    got.append(None)
                    continue
                got.append(tuple(DE.evaluate(x, ns1) for x in sh))
            shapes[tag] = tuple(got)
        sig = (r.get("raw_op") or "", tuple(sorted(r.get("params") or [])),
               shapes.get("i"), shapes.get("o"), shapes.get("w"))
        per[r.get("module_path") or "(root)"].append((sig, r))
    return per


def _match(old_ops, new_ops):
    """(짝지은 쌍, 옛쪽 미짝, 새쪽 미짝). 서명이 같은 것끼리 등장 순서대로 맞춘다."""
    pairs, un_a, un_b = [], 0, 0
    for mod in set(old_ops) | set(new_ops):
        a, b = old_ops.get(mod, []), new_ops.get(mod, [])
        by_b = collections.defaultdict(collections.deque)
        for sig, r in b:
            by_b[sig].append(r)
        used = collections.Counter()
        for sig, ra in a:
            q = by_b.get(sig)
            if q:
                pairs.append((ra, q.popleft()))
                used[sig] += 1
            else:
                un_a += 1
        un_b += sum(len(q) for q in by_b.values())
    return pairs, un_a, un_b


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
    # 구조 심볼이 런타임 심볼(/)로, 또는 그 반대로 바뀌는 것은 정상적인 배치 전환이
    # 아니다. 정상 전환은 구조 심볼을 **보존**한다(). 별도 범주로 둔다.
    RT = {"B", "T", "1"}
    if (old_lab in RT) != (new_lab in RT):
        return "runtime_role_change"
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
        old_ops = _ops(old_dir, phase, old_ns)
        new_ops = _ops(new_dir, phase, new_ns1)      # 새 라벨을 B=1 로 평가해 정규화
        if not (old_ops and new_ops):
            continue
        pairs, un_a, un_b = _match(old_ops, new_ops)
        tot["짝지은 op"] += len(pairs)
        tot["옛 판에만(op)"] += un_a
        tot["새 판에만(op)"] += un_b
        for ra, rb in pairs:
            for fld, tag in _FIELDS:
                va, vb = ra.get(fld), rb.get(fld)
                if va is None or vb is None:
                    continue
                ga = [va] if fld == "weight_shape" else va
                gb = [vb] if fld == "weight_shape" else vb
                for sa, sb in zip(ga, gb):
                    if not (isinstance(sa, list) and isinstance(sb, list)) or len(sa) != len(sb):
                        continue
                    for lo, ln in zip(sa, sb):
                        lo, ln = str(lo), str(ln)
                        c = classify(lo, ln, old_ns, new_ns1)
                        if c is None:
                            tot["같음"] += 1
                            continue
                        tot[c] += 1
                        samples[c][(lo, ln, ra.get("raw_op"))] += 1

    for k, v in tot.most_common():
        print(f"   {k:22} {v:>9,}")
    # `alignment_suspected` 는 **자동 승인 범주가 아니다.** 그룹 다중집합이 같다는 것만으로는
    # 진짜 축 맞바뀜(전치 오라벨)과 구별되지 않는다(외부 검토 2026-09-13).
    review = (tot["semantic_change"] + tot["unexplained"]
              + tot["runtime_role_change"] + tot["alignment_suspected"])
    for kind in ("runtime_role_change", "semantic_change", "unexplained",
                 "alignment_suspected", "batch_expected", "literal_resolved",
                 "singleton_fixed"):
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
