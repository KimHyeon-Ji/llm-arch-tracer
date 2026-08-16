"""새 축-등가류 간선을 켜면 **무엇이 합쳐지는지** 잰다. 아무것도 쓰지 않는다.

WHY THIS EXISTS
---------------
간선을 하나 더 놓는 일은 지표 몇 개로 판정하면 안 된다. 실제로 그렇게 했다가 물렸다:
전치 간선은 충돌 0 / 퇴행 0 / 개선 3 이라는 깨끗한 지표를 내면서 **소스로 확인한 판정을
뒤집고 있었다**(`axis_classes` docstring, [[transpose-edge-rejected]]).

특히 `axis_conflict == 0` 은 아무것도 보장하지 않는다 -- 잘못 합쳐진 두 등가류의 이름이
우연히 같으면 조용히 통과한다. 그래서 **합쳐진 내용 자체**를 본다.

무엇을 보는가 (외부 검토가 정리한 순서)
--------------------------------------
1. **간선 단위 폴트 인젝션** (`--selftest`) -- 이을 자리는 잇고, 안 될 자리는 안 잇는가.
   이게 먼저다. 간선이 애초에 의도대로 동작하지 않으면 나머지 측정은 의미가 없다.
2. **자기합류 0** -- 한 등가류 안에 *같은 op 의 같은 텐서* 의 서로 다른 두 축이 들어가면
   즉시 중단. `[L, L]` 정사각처럼 이름이 같아도 축 신원은 다른 자리를 합치는 사고를 잡는다.
   외부 검토는 여기가 **가장 먼저 걸릴 것** 이라고 봤다: 국소 대응은 일대일이어도
   H == L == N == 128 인 기존 등가류와 새 간선이 만나 순환이 닫힐 수 있기 때문이다.
3. **성장량** -- 새 union 쌍 수, 병합된 기존 class 쌍과 그 크기 곱, 최대 class 크기 증가,
   병합을 일으킨 실제 op id, 증가량 상위 class. 최대 크기 하나만 보면 큰 두 class 가
   합쳐진 폭발을 설명하지 못한다.
4. 새로 연결된 op_type 분포 -- `unsqueeze`/`squeeze` 외의 것이 보이면 간선이 새는 것이다.

실행:
    .venv\\Scripts\\python.exe develop\\class_diff.py --selftest
    .venv\\Scripts\\python.exe develop\\class_diff.py --model nemotron3-super
    .venv\\Scripts\\python.exe develop\\class_diff.py            # 전 모델
"""
import argparse
import collections
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
MODELS = os.path.join(PROJ, "models")
sys.path.insert(0, os.path.join(PROJ, "src"))

import axis_classes as AC          # noqa: E402
import build_table as BT           # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def selftest() -> int:
    """간선이 이을 자리는 잇고, 안 될 자리는 안 잇는가."""
    P = AC.singleton_pairs
    cases = [
        # 크기-1 축은 **양쪽 다** 제외된다 -- 아래 기대값에 축 0(B=1)이 없는 것이 요점이다.
        ("중간 singleton 삽입", P([1, 8, 64, 32], [1, 8, 64, 1, 32]), [(1, 1), (2, 2), (3, 4)]),
        ("끝 singleton 삽입", P([1, 8, 64, 128], [1, 8, 64, 128, 1]), [(1, 1), (2, 2), (3, 3)]),
        ("앞 singleton 삽입", P([2, 8, 64], [1, 2, 8, 64]), [(0, 1), (1, 2), (2, 3)]),
        ("squeeze 역방향", P([1, 8, 1, 64], [1, 8, 64]), [(1, 1), (3, 2)]),
        ("같은 값 셋이 나란히", P([2, 128, 128, 128], [2, 128, 128, 1, 128]),
         [(0, 0), (1, 1), (2, 2), (3, 4)]),
        ("비-1 열 순서 다름 -> 거부", P([2, 8, 64], [2, 64, 8]), None),
        ("비-1 개수 다름 -> 거부", P([2, 8, 64], [2, 8]), None),
        ("비-1 값 다름 -> 거부", P([2, 8, 64], [2, 8, 32]), None),
        ("전부 singleton", P([1, 1], [1, 1, 1]), []),
    ]
    ok = True
    for name, got, want in cases:
        good = got == want
        ok = ok and good
        print(f"   {name:<26} {'OK' if good else 'FAIL got=' + str(got)}")

    # 간선이 **op_type 을 넘지 않는가**: 같은 shape 쌍이라도 transpose/view/expand 에서는
    # 새 union 이 0 이어야 한다. build() 를 직접 돌려 확인한다.
    def unions(op):
        """이 op 에서 새로 이어진 축 쌍 수. **고정된 슬롯 집합** 위에서 센다.

        union-find 의 `p` 만 세면 안 된다 -- 간선이 꺼져 union 이 하나도 없으면 `p` 가 비어
        class 수가 0 으로 나오고, 부호가 뒤집힌다(처음에 그렇게 틀렸다).
        """
        si, so = [2, 8, 64], [2, 8, 64, 1]
        rows = [{"op_id": 0, "op_type": op, "module_path": "m",
                 "input_shape": [si], "output_shape": [so], "depends_on": []}]
        conc = {0: {"op_id": 0, "input_shape": [si], "output_shape": [so]}}
        slots = ([(0, "i", 0, k) for k in range(len(si))]
                 + [(0, "o", 0, k) for k in range(len(so))])
        n = []
        for flag in (False, True):
            uf = AC.build(rows, conc, singleton_edge=flag)
            n.append(len({uf.find(s) for s in slots}))
        return n[0] - n[1]        # 줄어든 class 수 = 새로 이어진 만큼
    for op, want in (("unsqueeze", 3), ("squeeze", 3), ("transpose", 0), ("view", 0),
                     ("expand", 0), ("repeat_interleave", 0)):
        got = unions(op)
        good = got == want
        ok = ok and good
        print(f"   {'op=' + op:<26} {'OK' if good else f'FAIL {got} != {want}'}")
    print("\n" + "=" * 60)
    print("간선 정상" if ok else "간선이 틀렸다 -- 켜면 안 된다")
    return 0 if ok else 1


def _selfmerge(uf, rows) -> list:
    """한 등가류 안에 **같은 op 의 같은 텐서** 의 서로 다른 두 축이 있는가.

    있으면 그 등가류는 "같은 축"이라는 선언을 스스로 깬 것이다. 이름이 우연히 같으면
    `axis_conflict` 는 아무 말도 하지 않으므로 이 검사가 따로 필요하다.
    """
    seen = collections.defaultdict(set)
    bad = []
    for r in rows:
        oid = r.get("op_id")
        for fld, tag in (("input_shape", "i"), ("output_shape", "o")):
            for si, sh in enumerate(r.get(fld) or []):
                if not isinstance(sh, list):
                    continue
                for ax in range(len(sh)):
                    root = uf.find((oid, tag, si, ax))
                    key = (oid, tag, si)
                    if root in seen[key]:
                        bad.append((oid, tag, si, ax))
                    seen[key].add(root)
    return bad


def compare(model: str, phase: str) -> dict | None:
    d = os.path.join(MODELS, model)
    raw = os.path.join(d, "full", f"{phase}.trace.raw.jsonl")
    if not os.path.exists(raw):
        return None
    rows = [json.loads(l) for l in io.open(raw, encoding="utf-8")]
    conc = BT.load_concrete(d, phase) or {}
    off = AC.build(rows, conc, singleton_edge=False)
    on = AC.build(rows, conc, singleton_edge=True)

    slots = []
    for r in rows:
        oid = r.get("op_id")
        for fld, tag in (("input_shape", "i"), ("output_shape", "o")):
            for si, sh in enumerate(r.get(fld) or []):
                if isinstance(sh, list):
                    slots += [(oid, tag, si, ax) for ax in range(len(sh))]

    a = collections.defaultdict(list)
    for s in slots:
        a[off.find(s)].append(s)
    b = collections.defaultdict(list)
    for s in slots:
        b[on.find(s)].append(s)

    # off 의 어떤 class 들이 on 에서 하나로 합쳐졌나
    merged = collections.defaultdict(set)
    for root, mem in a.items():
        merged[on.find(mem[0])].add(root)
    groups = [(r, sorted(v)) for r, v in merged.items() if len(v) > 1]

    # 후보 쌍과 **실제로 새로 이어진** 쌍을 나눠 센다. 처음에는 후보만 세어 Zamba2 를
    # "새 union 2,117" 이라고 보고했는데, 그중 상당수는 다른 간선이 **이미 이어 둔** 자리였다
    # (실제 독립 병합은 1,407). 외부 검토가 짚었다 -- 성장량을 부풀려 보면 안전 판정이 흐려진다.
    ops = collections.Counter()
    pairs = new_pairs = 0
    for r in rows:
        if r.get("op_type") not in ("unsqueeze", "squeeze"):
            continue
        c = conc.get(r["op_id"]) or {}
        ins, outs = c.get("input_shape") or [], c.get("output_shape") or []
        if len(outs) == 1 and len(ins) >= 1:
            pr = AC.singleton_pairs(ins[0], outs[0])
            for x, y in (pr or []):
                pairs += 1
                if off.find((r["op_id"], "i", 0, x)) != off.find((r["op_id"], "o", 0, y)):
                    new_pairs += 1
                    ops[r.get("op_type")] += 1

    grow = []
    for root, srcs in groups:
        sizes = sorted((len(a[s]) for s in srcs), reverse=True)
        grow.append((len(b[root]), sizes[:4], root))
    grow.sort(reverse=True)

    return {"classes_off": len(a), "classes_on": len(b), "union_pairs": pairs,
            "new_pairs": new_pairs,
            "merged_groups": len(groups), "ops": dict(ops),
            "max_off": max((len(v) for v in a.values()), default=0),
            "max_on": max((len(v) for v in b.values()), default=0),
            "selfmerge_off": len(_selfmerge(off, rows)),
            "selfmerge_on": len(_selfmerge(on, rows)),
            "top_grow": grow[:4]}


def main():
    ap = argparse.ArgumentParser(description="새 등가류 간선의 영향을 잰다 (읽기 전용)")
    ap.add_argument("--model", default="")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    names = [m for m in sorted(os.listdir(MODELS))
             if os.path.isdir(os.path.join(MODELS, m))
             and (not a.model or a.model.lower() in m.lower())]
    # 필터가 아무것도 못 고르면 멈춘다. 이 필터는 **모델 폴더명**에 걸린다 --
    # `regen_summaries.py` 의 필터는 프로파일 **파일명**에 걸려서 둘이 다르다
    # (`nvidia__NVIDIA-Nemotron-3-Super-...` vs `phase27-nemotron3-super-120b`).
    # 조용히 0개를 처리하면 "자기합류 없음" 이 통과처럼 보인다.
    if a.model and not names:
        print(f"필터 '{a.model}' 에 맞는 모델 폴더가 없다(이 필터는 모델 폴더명에 걸린다).")
        return 2

    bad = 0
    for m in names:
        for phase in ("prefill", "decode"):
            r = compare(m, phase)
            if not r or not r["merged_groups"]:
                continue
            flag = ""
            if r["selfmerge_on"] > r["selfmerge_off"]:
                flag = f"   ← 자기합류 {r['selfmerge_off']} -> {r['selfmerge_on']} **중단**"
                bad += 1
            print(f"{m.split('__')[-1][:26]:<28}{phase:<9}"
                  f"class {r['classes_off']}->{r['classes_on']}  "
                  f"후보쌍 {r['union_pairs']} (실제 새 병합 {r['new_pairs']})  "
                  f"병합그룹 {r['merged_groups']}  "
                  f"최대 {r['max_off']}->{r['max_on']}  {r['ops']}{flag}")
            for size, srcs, _root in r["top_grow"]:
                print(f"      상위 성장: 새 크기 {size} <- 기존 {srcs}")
    print("\n" + "=" * 60)
    print("자기합류 없음 -- 다음 단계(역할 혼재 / touched 자리 집합)로" if not bad
          else f"자기합류가 생긴 자리 {bad}건 -- 여기서 멈춘다")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
