"""제안된 **위치 규칙**이 실제로 몇 건을 닫는지, 적용하기 전에 잰다. 아무것도 쓰지 않는다.

WHY THIS EXISTS
---------------
남은 인계 1,179건(SSM·선형어텐션)은 접을 수 없는 서로 다른 축이다 -- Qwen3.5-397B 의 300개
항목이 13,500개 등가류를 덮는데 겹침이 0 이다(2026-08-16 실측). 하나씩 판정받는 방식으로는
끝나지 않는다.

그런데 앵커를 보면 **현재 이름이 이미 정준 위치에 있다**:

    [B, n_h_lin_v, 1, d_chunk]       축1 = n_h_lin_v,  축3 = d_chunk
    [B, n_h_lin_v, T, d_head_lin_k]  축3 = d_head_lin_k

청크 스캔의 축 순서는 모델마다 다시 발명되지 않으므로(SSD 는 `(b,c,l,s,h,n)`), 이건 **규칙**으로
박을 수 있는 종류다. 문제는 규칙을 심볼라이저에 넣는 순간 함대 전체가 흔들린다는 것이다 --
이 영역의 무단 변경이 과거 88k 축 회귀를 낸 기록이 `symbolic_shape.py` 에 남아 있다.

그래서 **먼저 잰다**: 규칙이 몇 건을 건드리는가, 그중 몇 건이 지금 이름과 **같은가**(확인으로
종결됨), 몇 건이 **다른가**(교정 -- 근거를 요구해야 한다). 다르다는 것 자체가 나쁜 게 아니라,
**얼마나 다른지 모르는 채로 적용하는 것**이 나쁘다.

규칙 파일 형식 (YAML)
--------------------
    rules:
      - name: gdn-head-after-batch
        scope: 'linear_attn|mamba|mixer|ssm'     # 모듈 경로에 걸리는 정규식
        shape: ['B', '*', '*', '*']              # 렌더된 앵커 shape 패턴, '*' 는 아무 라벨
        axis: 1                                  # 이 축에 대해
        symbol: n_h_lin_v                        # 이 이름이어야 한다
        source: 'modeling_qwen3_5.py:NNN ...'    # 근거 (필수)

`shape` 는 길이가 같아야 매칭된다. 끝에서 세고 싶으면 `'...'` 를 앞에 두면 된다:
`['...', 'd_chunk', '*']` 은 "마지막 두 축이 (아무거나, d_chunk 뒤)" 가 아니라 **뒤에서부터**
맞춘다.

실행:
    .venv\\Scripts\\python.exe develop\\rule_coverage.py 규칙.yaml
    .venv\\Scripts\\python.exe develop\\rule_coverage.py 규칙.yaml --model Qwen  # 일부만
    .venv\\Scripts\\python.exe develop\\rule_coverage.py --demo                  # 내장 예시
"""
import argparse
import collections
import io
import json
import os
import re
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
MODELS = os.path.join(PROJ, "models")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


# Codex 가 확정한 축 순서를 규칙 형식으로 옮긴 **예시**다. 값이 아니라 형식을 보여주려는 것이고,
# 실제 규칙은 소스를 읽은 쪽이 채운다. --demo 로 지금 함대에 대고 돌려 볼 수 있다.
DEMO = {"rules": [
    {"name": "gdn-head-after-batch", "scope": r"linear_attn",
     "shape": ["B", "*", "*", "*"], "axis": 1, "symbol": "n_h_lin_v",
     "source": "(예시) GDN 상태축 [B, h, k, v] -- B 다음이 head 축"},
    {"name": "gdn-value-width-last", "scope": r"linear_attn",
     "shape": ["B", "*", "T", "*"], "axis": 3, "symbol": "d_head_lin_k",
     "source": "(예시) 마지막 축이 head 폭"},
    {"name": "ssd-square-mask-is-chunk", "scope": r"mamba|mixer|ssm",
     "shape": ["*", "*"], "axis": 0, "symbol": "d_chunk",
     "source": "(예시) segment_sum 의 ones(chunk_size, chunk_size) 는 양 축 모두 chunk"},
]}


def _match(pat: list, shape: list) -> bool:
    """`pat` 이 `shape` 에 맞는가. '...' 는 앞쪽 아무 개수, '*' 는 축 하나 아무 라벨."""
    if pat and pat[0] == "...":
        tail = pat[1:]
        if len(tail) > len(shape):
            return False
        shape = shape[len(shape) - len(tail):]
        pat = tail
    if len(pat) != len(shape):
        return False
    return all(p == "*" or p == str(s) for p, s in zip(pat, shape))


def _mixed_layers(model: str) -> list:
    """이 모델의 층 유형이 여러 가지면 그 목록. 아니면 빈 리스트.

    스코프가 `mixer` 같은 **위치 이름**에 걸리면 층 유형을 넘나든다 -- Nemotron-H 계열은
    Mamba 층도 attention 층도 모듈을 똑같이 `mixer` 라고 부른다. 외부 검토 실측: `mixer` 로
    잡힌 461항목 중 **72개가 SSM 이 아니라 full-attention 축**이었다(2026-08-16).

    인계 항목의 모듈 경로는 `model.layers.*.mixer` 로 층 번호가 지워져 있어 **항목별로는
    층 유형을 알 수 없다.** 그래서 거르지 않고 경고만 한다 -- 못 거르는 것을 거른 척하는 것이
    이 도구가 하면 안 되는 일이다.
    """
    p = os.path.join(MODELS, model, "structure.yaml")
    if not os.path.exists(p):
        return []
    try:
        sched = ((yaml.safe_load(io.open(p, encoding="utf-8")) or {})
                 .get("symbols", {}) or {}).get("layer_sched")
    except (ValueError, OSError):
        return []
    kinds = sorted({str(x) for x in sched}) if isinstance(sched, list) else []
    return kinds if len(kinds) > 1 else []


def _items(model: str):
    for ph in ("prefill", "decode"):
        p = os.path.join(MODELS, model, "full", f"{ph}.unsettled.json")
        if not os.path.exists(p):
            continue
        try:
            for it in (json.load(io.open(p, encoding="utf-8")) or {}).get("items") or []:
                yield ph, it
        except (ValueError, OSError):
            pass


def evaluate(spec: dict, only: str = "") -> dict:
    """규칙이 인계 항목에 무엇을 하는가. 산출물은 건드리지 않는다."""
    rules = spec.get("rules") or []
    for r in rules:
        for k in ("scope", "shape", "axis", "symbol"):
            if k not in r:
                raise SystemExit(f"규칙 '{r.get('name')}' 에 '{k}' 가 없다")
        # 대상 축은 패턴에서 반드시 와일드카드여야 한다. 거기에 이름을 박아 두면 "그 축이 X 인
        # 자리에서 그 축은 X 다"를 확인하는 셈이라 아무것도 증명하지 못한다 -- 그리고 그
        # 순환이 **오판을 영구 종결시킨다**. 외부 검토(Codex)가 지적, 2026-08-16.
        pat, ax = r["shape"], r["axis"]
        off = len(pat) - 1 if pat and pat[0] == "..." else 0   # '...' 는 뒤에서부터 센다
        idx = ax if not off else None
        if off:                       # 꼬리 패턴이면 대상 축이 꼬리 안에 있을 때만 검사 가능
            continue
        if idx is not None and 0 <= idx < len(pat) and pat[idx] != "*":
            raise SystemExit(
                f"규칙 '{r.get('name')}': 대상 축 {ax} 가 패턴에 '{pat[idx]}' 로 박혀 있다. "
                f"현재 이름을 조건으로 현재 이름을 확인하는 순환이다 -- '*' 로 둘 것")
    scoped = [(r, re.compile(r["scope"])) for r in rules]

    res = {"agree": collections.Counter(), "differ": collections.Counter(),
           "clash": collections.Counter(), "offcand": collections.Counter(),
           "mixed": {}, "untouched": 0, "total": 0, "emit": [],
           "differ_ex": [], "clash_ex": [], "axes_agree": 0, "axes_differ": 0}
    for model in sorted(os.listdir(MODELS)):
        if not os.path.isdir(os.path.join(MODELS, model)):
            continue
        if only and only.lower() not in model.lower():
            continue
        for _ph, it in _items(model):
            res["total"] += 1
            shape = [str(x) for x in (it["override_stub"].get("shape") or [])]
            ax, cur = it["override_stub"].get("axis"), str(it["current_label"])
            hits = [r for r, rx in scoped
                    if rx.search(it["module"]) and r["axis"] == ax and _match(r["shape"], shape)
                    and (not r.get("op_type")
                         or r["op_type"] == it["override_stub"].get("op_type"))]
            if not hits:
                res["untouched"] += 1
                continue
            mk = _mixed_layers(model)
            if mk:
                res["mixed"].setdefault(model, mk)
            says = {r["symbol"] for r in hits}
            if len(says) > 1:
                # 두 규칙이 같은 축에 서로 다른 이름을 말한다 -- 규칙끼리의 모순이므로
                # 적용 전에 반드시 없애야 한다. 라벨이 맞고 틀리고와는 다른 층위의 문제다.
                res["clash"][model] += 1
                if len(res["clash_ex"]) < 8:
                    res["clash_ex"].append((model, shape, ax, sorted(says),
                                            [r["name"] for r in hits]))
                continue
            want = says.pop()
            if want == cur:
                res["agree"][model] += 1
                res["axes_agree"] += it.get("axes") or 0
                # 확인 기록으로 낼 재료. 항목마다 같은 근거를 복제하지 않고 **rule_id 와 공통
                # source** 를 남긴다 -- 규칙 하나가 여러 자리를 종결시키는 것이 요점이므로,
                # 어느 규칙이 종결시켰는지 되짚을 수 있어야 한다.
                res["emit"].append((model, it, hits[0]))
            else:
                res["differ"][model] += 1
                res["axes_differ"] += it.get("axes") or 0
                # 규칙이 말하는 이름이 그 축의 **후보에도 없다**면, 그 축의 폭이 그 심볼의 값과
                # 아예 다르다는 뜻이다 -- 라벨 판단 이전에 규칙이 과매칭이라는 신호다. 실제로
                # 내장 예시의 `shape: ['*','*']` 이 mamba 안의 모든 rank-2 텐서를 잡아
                # 정사각 마스크가 아닌 자리까지 건드렸다(2026-08-16).
                if want not in (it.get("candidates") or []):
                    res["offcand"][model] += 1
                if len(res["differ_ex"]) < 12:
                    res["differ_ex"].append((model, shape, ax, cur, want,
                                             sorted(it.get("candidates") or [])))
    return res


def selftest() -> int:
    """매처가 실제로 무는지 결함을 먹여 확인한다.

    이 도구는 **숫자를 내는 도구**이고, 숫자를 보고 규칙을 적용할지 말지 정한다. 매처가
    조용히 틀리면 "규칙이 138건을 확인으로 닫는다" 같은 문장이 근거 없이 나온다. 그래서
    `develop/verify_selftest.py` 와 같은 원칙을 여기에도 둔다 -- **한 번도 실패하지 않는
    검사는 아무것도 증명하지 않는다.**
    """
    cases = [
        ("정확일치", _match(["B", "n_h", "T"], ["B", "n_h", "T"]), True),
        ("와일드카드", _match(["B", "*", "T"], ["B", "n_h", "T"]), True),
        ("길이다름-거부", _match(["B", "*"], ["B", "n_h", "T"]), False),
        ("라벨다름-거부", _match(["B", "n_kv", "T"], ["B", "n_h", "T"]), False),
        ("꼬리매칭", _match(["...", "T"], ["B", "n_h", "T"]), True),
        ("꼬리매칭-라벨다름", _match(["...", "d_head"], ["B", "n_h", "T"]), False),
        ("꼬리가더김-거부", _match(["...", "A", "B", "C", "D"], ["B", "n_h", "T"]), False),
        ("빈패턴-거부", _match([], ["B"]), False),
    ]
    ok = True
    for name, got, want in cases:
        good = got is want
        ok = ok and good
        print(f"   {name:<20} {'OK' if good else 'FAIL — 매처가 틀렸다'}")

    # op_type 선택자가 실제로 좁히는가 (양성 대조 포함): 같은 shape·축이라도 op 가 다르면 안 문다
    it = {"module": "model.layers.0.mixer", "current_label": "d_state", "axes": 1,
          "candidates": ["d_state", "d_chunk"],
          "override_stub": {"shape": ["a", "b"], "axis": 0, "op_type": "ones"}}
    base = {"scope": "mixer", "shape": ["*", "*"], "axis": 0, "symbol": "d_chunk"}

    def run(rule):
        saved, out = globals()["_items"], []
        globals()["_items"] = lambda m: [("prefill", it)] if m == _probe else []
        try:
            return evaluate({"rules": [rule]}, only=_probe)
        finally:
            globals()["_items"] = saved

    # 순환 논증 거부: 대상 축에 이름이 박힌 규칙은 받아들이면 안 된다
    circ = False
    try:
        evaluate({"rules": [dict(base, shape=["d_state", "*"])]}, only="!!없는모델!!")
    except SystemExit:
        circ = True
    ok = ok and circ
    print(f"   {'순환논증-거부':<20} {'OK' if circ else 'FAIL — 순환 규칙을 통과시켰다'}")

    _probe = next((m for m in sorted(os.listdir(MODELS))
                   if os.path.isdir(os.path.join(MODELS, m))), None)
    if _probe:
        hit = run(dict(base, op_type="ones"))
        miss = run(dict(base, op_type="slice"))
        for name, got, want in (
                ("op_type-일치", sum(hit["differ"].values()) == 1, True),
                ("op_type-불일치-거부", miss["untouched"] == 1, True),
                ("과매칭신호", sum(hit["offcand"].values()) == 0, True)):
            good = got is want
            ok = ok and good
            print(f"   {name:<20} {'OK' if good else 'FAIL'}")
    print("\n" + "=" * 60)
    print("매처 정상" if ok else "매처가 틀렸다 — 이 도구가 낸 숫자를 믿으면 안 된다")
    return 0 if ok else 1


def _emit_confirmations(rows, path) -> int:
    """'지금 이름과 같다' 를 rules/label_confirmed.yaml 형식으로 낸다.

    **다른 항목은 절대 내지 않는다.** 규칙이 지금 이름을 뒤집는 자리는 근거를 개별로 확인해야
    하고, 규칙 하나로 대량 교정하는 것은 이 저장소가 계속 잡아온 실패다.
    """
    out = ["# develop/rule_coverage.py --emit 가 낸 것. 각 항목은 rule_id 로 어느 규칙이",
           "# 종결시켰는지 되짚을 수 있다. 사람이 손댄 곳은 없다.",
           "confirmed:"]
    for model, it, rule in rows:
        st = it["override_stub"]
        src = " ".join(str(rule.get("source", "")).split())
        out += [f"  - model: {model}",
                f"    module: '{st['module']}'",
                "    shape: [" + ", ".join(f'"{x}"' for x in st["shape"]) + "]",
                f"    axis: {st['axis']}",
                f"    field: {st['field']}",
                f"    shape_index: {st['shape_index']}",
                f"    op_type: {st['op_type']}",
                f"    nth: {st['nth']}",
                f"    label: {it['current_label']}",
                f"    expect: {it['size']}",
                f"    rule_id: {rule.get('name')}",
                "    source: >",
                f"      [{rule.get('name')}] {src}"]
    io.open(path, "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")
    return len(rows)


def main():
    ap = argparse.ArgumentParser(
        description="위치 규칙이 인계 항목에 무엇을 하는지 적용 전에 잰다 (읽기 전용)")
    ap.add_argument("spec", nargs="?", help="규칙 YAML")
    ap.add_argument("--demo", action="store_true", help="내장 예시 규칙으로 돌려 본다")
    ap.add_argument("--selftest", action="store_true", help="매처 자체를 검증한다")
    ap.add_argument("--model", default="", help="모델 이름 부분 일치로 한정")
    ap.add_argument("--emit", metavar="OUT.yaml",
                    help="'지금 이름과 같다' 항목만 확인 기록 YAML 로 낸다 "
                         "(다른 항목은 개별 검토 대상이라 절대 내지 않는다)")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.spec and not a.demo:
        ap.error("규칙 YAML 을 주거나 --demo 를 쓸 것")
    spec = DEMO if a.demo else (yaml.safe_load(io.open(a.spec, encoding="utf-8")) or {})
    if a.demo:
        print("※ 내장 예시 규칙이다. 형식을 보여주려는 것이고 근거로 쓰면 안 된다.\n")

    r = evaluate(spec, a.model)
    touched = sum(r["agree"].values()) + sum(r["differ"].values()) + sum(r["clash"].values())
    print(f"인계 항목 {r['total']}건 중 규칙이 건드리는 것 {touched}건 "
          f"(안 건드림 {r['untouched']})")
    print(f"  지금 이름과 같다  {sum(r['agree'].values()):>5}건  (축 {r['axes_agree']:,})"
          f"   -> 확인으로 종결 가능")
    print(f"  지금 이름과 다르다 {sum(r['differ'].values()):>5}건  (축 {r['axes_differ']:,})"
          f"   -> 교정이므로 근거를 요구해야 한다")
    print(f"  규칙끼리 모순    {sum(r['clash'].values()):>5}건"
          f"                -> 적용 전에 없앨 것")
    if sum(r["offcand"].values()):
        print(f"  그중 후보에도 없던 이름 {sum(r['offcand'].values()):>4}건"
              f"           -> 규칙이 과매칭이라는 신호")

    if r["mixed"]:
        print("\n[주의 — 스코프가 층 유형을 넘나들 수 있다]")
        print("  아래 모델은 층마다 유형이 다른데, 인계 항목의 모듈 경로는 층 번호가 지워져")
        print("  있어 항목별로는 가릴 수 없다. 규칙에 layer_types 를 걸어야 하는 자리다.")
        for m, ks in sorted(r["mixed"].items()):
            print(f"  {m.split('__')[-1][:30]:<32} 층 유형 {ks}")

    if r["clash_ex"]:
        print("\n[규칙 모순]")
        for m, sh, ax, says, names in r["clash_ex"]:
            print(f"  {m.split('__')[-1][:24]:<26} ax{ax} {sh} -> {says} ({names})")
    if r["differ_ex"]:
        print("\n[규칙이 지금 이름을 뒤집는 자리 -- 여기가 진짜 검토 대상]")
        for m, sh, ax, cur, want, cand in r["differ_ex"]:
            flag = "" if want in cand else "   ← 후보에도 없던 이름"
            print(f"  {m.split('__')[-1][:24]:<26} ax{ax} {sh}  {cur} -> {want}{flag}")
    if r["agree"]:
        pass
    if a.emit:
        n = _emit_confirmations(r["emit"], a.emit)
        print(f"\n확인 기록 {n}건을 {a.emit} 에 냈다 "
              f"(다르다 {sum(r['differ'].values())}건은 개별 검토 대상이라 내지 않았다)")
    if r["agree"]:
        print("\n[모델별 '지금 이름과 같다']")
        for m, v in r["agree"].most_common(12):
            print(f"  {m.split('__')[-1][:30]:<32}{v:>5}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
