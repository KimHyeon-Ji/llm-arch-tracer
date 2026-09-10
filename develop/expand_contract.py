r"""`expand` 의 **비방송 축 계약**을 구조 manifest 로 못 박는다.

각 항목은 "이 두 축 자리는 같은 등가류여야 한다" 는 주장이고, 라벨이 어떻게 바뀌든 그대로다.

**이건 라운드 5~7 의 source-confirmed 314 가 아니다.** 그건 11개 selector / 314 occurrence 이고
`develop/oracle_314.py` 가 따로 검사한다. 이 파일은 대상 세 모델의 **모든** 비방송 expand 축을
모은 더 넓은 계약이다(165 자리 이름, 2천여 occurrence). 외부 검토(2026-09-10)가 둘을 섞어
불렀다고 짚어서 이름을 갈랐다.

자리 이름:
    module_key | layer_sig | op_type | nth | i|o | shape_index | axis | rank

`layer_sig` 는 그 레이어의 `(module_key, op_type)` 집합 해시다. **config 를 안 읽고**
하이브리드 스택을 가른다. 구체 shape 을 넣어도 갈리지만 쓰지 않는다 -- 키에 T 가 박혀서
다른 `seq_len` 으로 재트레이스하면 자리를 잃고, 검사가 조용히 무력화된다.

실행:
    .venv\Scripts\python.exe develop\expand_contract.py --build
    .venv\Scripts\python.exe develop\expand_contract.py --check --mode provenance
"""
import argparse
import collections
import hashlib
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJ, "src"))
# `__main__` 일 때만 감싼다. import 하는 쪽도 같은 짓을 하면 첫 래퍼가 GC 될 때
# 밑의 버퍼를 닫아버려 `I/O operation on closed file` 이 난다.
if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import axis_classes as ac          # noqa: E402
import build_table as BT           # noqa: E402
from anchors import module_key     # noqa: E402

MODELS = os.path.join(PROJ, "models")
OUT = os.path.join(HERE, "verify", "expand_contract_manifest.json")

# 314 의 출처. 라운드 5~7 에서 소스로 방향을 확정했고, 교정을 넣으면 등가류가 쪼개져
# 되돌렸던 자리들이다. **`expand` 의 비방송 축**이 그 공통 형태다.
TARGETS = {
    "ibm-granite__granite-4.0-h-small": 216,
    "nvidia__NVIDIA-Nemotron-3-Super-120B-A12B-BF16": 80,
    "Zyphra__Zamba2-1.2B": 18,
}


def _rows(model, phase):
    d = os.path.join(MODELS, model)
    raw = os.path.join(d, "full", f"{phase}.trace.raw.jsonl")
    if not os.path.exists(raw):
        return None
    rows = [json.loads(l) for l in open(raw, encoding="utf-8")]
    ac.attach_ports(d, phase, rows)
    return rows, (BT.load_concrete(d, phase) or {}), ac.noop_barriers_of(d, phase)


_LAYER_IDX = re.compile(r"\.(\d+)\.")


def layer_signatures(rows):
    """{레이어 번호: 그 레이어의 구성 지문}.

    하이브리드 스택을 가르는 **config 없는** 판별자다. `module_key` 가 레이어·전문가 번호를
    접으므로, 같은 타입의 레이어는 같은 `(module_key, op_type)` 집합을 내고 다른 타입은
    다른 집합을 낸다. 값도 라벨도 `seq_len` 도 안 본다.
    """
    per = collections.defaultdict(set)
    for r in rows:
        mp = r.get("module_path") or ""
        m = _LAYER_IDX.search(mp)
        if not m:
            continue
        per[int(m.group(1))].add((module_key(mp), r.get("op_type")))
    return {i: hashlib.sha1(
        "::".join(f"{a}|{b}" for a, b in sorted(v)).encode()).hexdigest()[:8]
        for i, v in per.items()}


def _stable_key(r, ordn, tag, si, ax, sig=None):
    """레이어 번호·op_id·라벨·`seq_len` 어디에도 기대지 않는 자리 이름.

    **레이어 지문과 rank 가 들어간다.** `module_key` 가 레이어 번호를 접으므로 하이브리드
    스택에서 같은 `(module_key, op_type, nth)` 가 **레이어 타입에 따라 다른 op** 가 된다.
    안 넣었더니 Nemotron 의 `mixer` 에서 방송 축(axis 3)이 비방송 축 항목에 섞여 거짓 끊김
    13건이 났다(2026-09-10 실측). "각 레이어에서 하나를 잡는다" 와 "각 레이어의 그것이 같은
    역할이다" 는 별개다(외부 검토가 미리 짚었다).

    구체 shape 을 넣어도 갈리지만 **쓰지 않는다** -- 키에 T(16/24)가 그대로 박혀서 다른
    `seq_len` 으로 재트레이스하면 자리를 잃고, oracle 이 조용히 다시 아무것도 검증하지
    못하는 상태가 된다. 그게 이 manifest 를 만든 이유다.
    """
    mp = r.get("module_path") or ""
    m = _LAYER_IDX.search(mp)
    lsig = (sig or {}).get(int(m.group(1)), "") if m else ""
    fld = "input_shape" if tag == "i" else "output_shape"
    sh = (r.get(fld) or [])
    rank = len(sh[si]) if si < len(sh) and isinstance(sh[si], list) else -1
    return "|".join(str(x) for x in (
        module_key(mp) or "(root)",
        lsig, r.get("op_type"), ordn.get(r.get("op_id")), tag, si, ax, f"r{rank}"))


def build():
    man = {"note": "314 oracle as STRUCTURAL site pairs. Each entry asserts the two axis "
                   "sites must be in the same equivalence class, regardless of label.",
           "source": "rounds 5-7 source-confirmed verdicts (expand non-broadcast axes)",
           "entries": []}
    for model, expect in TARGETS.items():
        d = os.path.join(MODELS, model)
        if not os.path.isdir(d):
            continue
        n = 0
        for phase in ("prefill", "decode"):
            got = _rows(model, phase)
            if not got:
                continue
            rows, conc, _ = got
            ordn = ac.op_ordinals(rows)
            sig = layer_signatures(rows)
            seen = set()
            for r in rows:
                if r.get("op_type") not in ("expand", "broadcast_to", "expand_as"):
                    continue
                c = conc.get(r.get("op_id")) or {}
                ci = (c.get("input_shape") or [None])[0]
                co = (c.get("output_shape") or [None])[0]
                if not (isinstance(ci, list) and isinstance(co, list) and len(ci) == len(co)):
                    continue
                for a in range(len(ci)):
                    if ci[a] != co[a] or ci[a] == 1:
                        continue
                    k = (_stable_key(r, ordn, "i", 0, a, sig),
                         _stable_key(r, ordn, "o", 0, a, sig))
                    if k in seen:
                        continue
                    seen.add(k)
                    man["entries"].append({
                        "model": model, "phase": phase,
                        "src": k[0], "dst": k[1], "value": ci[a],
                        "relation": "same",
                        "why": "expand non-broadcast axis"})
                    n += 1
        man.setdefault("counts", {})[model] = {"entries": n, "round57_axes": expect}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=1)
    print(f"manifest {len(man['entries'])}건 -> {os.path.relpath(OUT, PROJ)}")
    for m, c in man["counts"].items():
        print(f"   {c['entries']:5} 자리쌍  (라운드5~7 축 {c['round57_axes']})  {m}")
    return 0


def evaluate(mode):
    """{model: {"ok":n,"broken":n,"missing":n}}. oracle 이 이걸 불러 쓴다."""
    if not os.path.exists(OUT):
        return None
    man = json.load(open(OUT, encoding="utf-8"))
    by = collections.defaultdict(list)
    for e in man["entries"]:
        by[(e["model"], e["phase"])].append(e)
    per = collections.defaultdict(lambda: {"ok": 0, "broken": 0, "missing": 0})
    for (model, phase), entries in sorted(by.items()):
        got = _rows(model, phase)
        if not got:
            per[model]["missing"] += len(entries)
            continue
        rows, conc, bars = got
        ordn = ac.op_ordinals(rows)
        sig = layer_signatures(rows)
        uf = ac.build(rows, conc, noop_barriers=bars, mode=mode)
        idx = collections.defaultdict(list)
        for r in rows:
            for fld, tag in (("input_shape", "i"), ("output_shape", "o")):
                for si, sh in enumerate(r.get(fld) or []):
                    if not isinstance(sh, list):
                        continue
                    for ax in range(len(sh)):
                        idx[_stable_key(r, ordn, tag, si, ax, sig)].append(
                            (r["op_id"], tag, si, ax))
        for e in entries:
            a, b = idx.get(e["src"]) or [], idx.get(e["dst"]) or []
            if not a or not b:
                per[model]["missing"] += 1
            elif all(uf.find(x) == uf.find(y) for x, y in zip(a, b)):
                # 같은 안정 키가 여러 레이어에 있다 -- 전부 같은 클래스여야 한다.
                per[model]["ok"] += 1
            else:
                per[model]["broken"] += 1
    return {"entries": len(man["entries"]), "per_model": dict(per)}


def check(mode):
    ev = evaluate(mode)
    if ev is None:
        print("manifest 가 없다. --build 먼저.")
        return 1
    ok = sum(v["ok"] for v in ev["per_model"].values())
    bad = sum(v["broken"] for v in ev["per_model"].values())
    miss = sum(v["missing"] for v in ev["per_model"].values())
    print(f"mode = {mode}")
    print(f"manifest {ev['entries']}건: 이어짐 {ok} / **끊김 {bad}** / 자리없음 {miss}")
    for m, v in sorted(ev["per_model"].items(), key=lambda kv: -kv[1]["broken"]):
        if v["broken"] or v["missing"]:
            print(f"   끊김 {v['broken']:4} / 자리없음 {v['missing']:4}  {m[:46]}")
    return 1 if (bad or miss) else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--mode", default="provenance")
    a = ap.parse_args()
    if a.build:
        return build()
    return check(a.mode)


if __name__ == "__main__":
    sys.exit(main())
