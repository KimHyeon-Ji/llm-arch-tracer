"""등가류가 넓어져 앵커가 깨진 판정을, **조건을 전부 만족할 때만** 새 shape 으로 다시 짚는다.

WHY THIS EXISTS
---------------
`unsqueeze` 간선을 켜면 지표는 좋아지는데(퇴행 0 / 개선 8, 자기합류 0, 보호 경계 유지)
**기존 판정 61건이 발화를 멈춘다**(`changed` 64 -> 0). 등가류가 넓어지며 라벨이 바뀌고,
판정의 앵커 `shape` 가 더는 안 맞기 때문이다.

`shape` 를 앵커에서 빼는 것은 답이 아니다. 그건 판정 ID 의 일부이고, 같은 `nth` 가 레이어
유형마다 다른 코드 줄을 가리키는 반례(DeepSeek-V4-Flash-0731 의 `slice/nth10`)를 가르는
유일한 수단이다. 문제는 `shape` 가 **자리 선택자**와 **라벨 상태 사전조건** 을 동시에 맡고
있다는 것이고, 넓어진 등가류가 후자만 흔든다.

그래서 이 도구는 **자리는 그대로인데 주변 축 이름만 바뀐 경우** 에 한해 `shape` 만 갈아
끼운다. 조건(외부 검토가 정한 것) 중 하나라도 어기면 `ambiguous` / `target_changed` 로
남겨 사람에게 넘긴다 -- 자동으로 우기지 않는다.

    같은 (op_id, field, shape_index, axis) 자리일 것
    module / op_type / nth / layer_type 이 그대로일 것
    **전체 concrete shape** 이 그대로일 것
    대상 축 concrete 값이 `expect` 와 같을 것
    대상 축 이름이 여전히 override 의 `from` (확인은 `label`) 일 것
    달라진 것은 **대상 축이 아닌 주변 축의 이름뿐** 일 것
    레이어별 후보가 정확히 하나일 것

실행 (기본은 보고만 한다):
    .venv\\Scripts\\python.exe develop\\reanchor.py --off <OFF스냅샷> --on <ON스냅샷> --model <폴더명>
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

import build_table as BT           # noqa: E402
import label_overrides as LO       # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def _rows(path):
    return {json.loads(l)["op_id"]: json.loads(l) for l in io.open(path, encoding="utf-8")}


def _shape_at(row, field, si):
    fld = "input_shape" if field == "i" else "output_shape"
    shs = row.get(fld) or []
    return [str(x) for x in shs[si]] if si < len(shs) and isinstance(shs[si], list) else None


def plan(model: str, off_dir: str, on_dir: str) -> list:
    """판정별 (분류, 새 shape, 설명). 아무것도 쓰지 않는다."""
    out = []
    specs = {}
    for i, s in enumerate(LO.for_model(model)):
        specs[f"OV:{i}"] = ("override", s)
    for i, s in enumerate([c for c in LO.load_confirmed() if c.get("model") == model]):
        specs[f"CF:{i}"] = ("confirm", s)

    for phase in ("prefill", "decode"):
        fo = os.path.join(off_dir, f"{phase}.verdict_footprint.json")
        fn = os.path.join(on_dir, f"{phase}.verdict_footprint.json")
        ro = os.path.join(MODELS, model, "full", f"{phase}.trace.raw.jsonl")
        rn = os.path.join(on_dir, f"{phase}.trace.raw.jsonl")
        if not all(os.path.exists(x) for x in (fo, fn, ro, rn)):
            continue
        A = {v["id"]: v for v in json.load(io.open(fo, encoding="utf-8"))["verdicts"]}
        B = {v["id"]: v for v in json.load(io.open(fn, encoding="utf-8"))["verdicts"]}
        old_rows, new_rows = _rows(ro), _rows(rn)
        conc = BT.load_concrete(os.path.join(MODELS, model), phase) or {}

        for vid, a in A.items():
            b = B.get(vid)
            if b is None:
                continue
            lost = {tuple(x) for x in (a.get("changed") or [])} - \
                   {tuple(x) for x in (b.get("changed") or [])}
            if not lost:
                continue
            spec = next((s for k, (_kind, s) in specs.items()
                         if LO._report_id(s) == vid), None) if hasattr(LO, "_report_id") else None
            if spec is None:
                spec = next((s for _k, (_kind, s) in specs.items()
                             if json.dumps(s.get("shape") or [], ensure_ascii=False)
                             in vid and str(s.get("op_type")) in vid), None)
            if spec is None:
                out.append((phase, vid, "spec_not_found", None, "판정 원본을 못 찾음"))
                continue

            target = str(spec.get("from") or spec.get("label"))
            want_ax = spec.get("axis")
            cands = collections.Counter()
            reasons = collections.Counter()
            for oid, fld, si, ax in sorted({tuple(x) for x in (a.get("anchors") or [])}):
                ro_row, rn_row = old_rows.get(oid), new_rows.get(oid)
                cr = conc.get(oid) or {}
                if not (ro_row and rn_row):
                    reasons["행 없음"] += 1
                    continue
                if ro_row.get("op_type") != rn_row.get("op_type") or \
                        ro_row.get("module_path") != rn_row.get("module_path"):
                    reasons["구조 좌표 바뀜"] += 1
                    continue
                co = _shape_at(cr, fld, si)          # 구체 shape (라벨과 무관)
                so, sn = _shape_at(ro_row, fld, si), _shape_at(rn_row, fld, si)
                if so is None or sn is None or co is None or len(so) != len(sn):
                    reasons["shape 못 읽음"] += 1
                    continue
                if ax >= len(co) or str(co[ax]) != str(spec.get("expect")):
                    reasons["expect 불일치"] += 1
                    continue
                if sn[ax] != target:
                    # 대상 축 자체가 이미 바뀌었다 -- 다른 판정에 흡수됐을 수 있으니 사람에게.
                    reasons[f"대상 축이 {sn[ax]} 로 바뀜"] += 1
                    continue
                if [x for i, x in enumerate(so) if i != ax] == \
                        [x for i, x in enumerate(sn) if i != ax] and so[ax] == sn[ax]:
                    reasons["차이 없음"] += 1
                    continue
                cands[tuple(sn)] += 1
            if len(cands) == 1 and not reasons:
                new_shape = list(next(iter(cands)))
                out.append((phase, vid, "auto", new_shape,
                            f"{cands[tuple(new_shape)]}자리, 대상 축 {want_ax}={target} 유지"))
            elif len(cands) > 1:
                out.append((phase, vid, "ambiguous", None,
                            f"후보 shape {len(cands)}종: {[list(c) for c in cands][:2]}"))
            else:
                out.append((phase, vid, "target_changed", None, dict(reasons)))
    return out


def main():
    ap = argparse.ArgumentParser(description="깨진 판정 앵커를 조건부로 다시 짚는다 (읽기 전용)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--off", required=True, help="간선 OFF 시점 footprint 디렉터리")
    ap.add_argument("--on", required=True, help="간선 ON 시점 footprint/trace 디렉터리")
    a = ap.parse_args()
    res = plan(a.model, a.off, a.on)
    c = collections.Counter(k for _p, _v, k, _s, _w in res)
    for phase, vid, kind, shape, why in res:
        head = json.loads(vid)
        print(f"  [{kind:<14}] {phase:<8}{head[10]}/nth{head[11]} ax{head[5]} "
              f"{head[1]}->{head[2]}")
        print(f"       {'새 shape ' + str(shape) if shape else why}")
    print("\n" + "=" * 60)
    print(f"자동 가능 {c['auto']} / 모호 {c['ambiguous']} / 대상축 변경 {c['target_changed']}"
          f" / 원본 못 찾음 {c['spec_not_found']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
