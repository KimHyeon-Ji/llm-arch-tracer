"""단일 검증 하네스 — rules/src를 바꾼 뒤 **반드시** 이걸 돌린다.

왜 있는가 (2026-07-27, 실제로 당한 사고들):
  - PowerShell 정규식으로 rules/symbols.yaml을 고쳤다가 심볼명이 깨지고 `scope:` 키가
    중복됐다. yaml.safe_load는 중복 키를 조용히 마지막 것으로 덮어써서 안 잡혔다.
  - docstring 편집이 summarize.py를 SyntaxError로 만들었는데, 재생성을 돌리기 전까지 몰랐다.
  - 심볼라이저에서 "scope 밖 심볼은 버린다"로 바꿨더니 정상 심볼 88,000개가 bare 정수로
    퇴행했다. 우연히 감사 스크립트를 다시 돌려서 발견했다.
  - 감사 스크립트 자체가 틀려서(수식 `2*d_head` 안의 계수 2까지 셈) 잘못된 결론을 보고했다.
공통 원인은 하나다: **"고쳤다"는 주장에 자동 확인이 없었다.** 그래서 주장 대신 종료 코드로
말하게 만든다.

돌리는 법:
    .venv\\Scripts\\python.exe develop\\verify_all.py              # 검사 + 베이스라인 대조
    .venv\\Scripts\\python.exe develop\\verify_all.py --update-baseline   # 검토 후 기준 갱신

검사 항목:
  STATIC   rules/*.yaml 파싱 + 중복 키, src/*.py import, symbols/derived_dims 스키마
  FLEET    모델별 C체크 FAIL, C17, 미해결 유도상수, bare 정수, 미확인 심볼
  EXTERNAL KV cache 카드 vs 공개 갤러리 수치 (develop/verify/references.yaml)
  BASELINE 이전 기준 대비 **퇴행**이면 실패 (개선은 통과시키되 갱신하라고 알림)
"""
import argparse
import importlib
import json
import os
import re
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJ, "src"))

MODELS = os.path.join(PROJ, "models")
REFS = os.path.join(HERE, "verify", "references.yaml")
BASELINE = os.path.join(HERE, "verify", "baseline.json")

failures: list[str] = []
warnings: list[str] = []


def fail(msg):
    failures.append(msg)
    print(f"   FAIL  {msg}")


def warn(msg):
    warnings.append(msg)
    print(f"   WARN  {msg}")


# ---------------------------------------------------------------- STATIC
class _DupCheckLoader(yaml.SafeLoader):
    """yaml.safe_load silently keeps the LAST of duplicate keys -- that is exactly how a
    corrupted symbols.yaml passed unnoticed. Refuse duplicates instead."""


def _no_dup(loader, node, deep=False):
    out = {}
    for k_node, v_node in node.value:
        key = loader.construct_object(k_node, deep=deep)
        if key in out:
            raise ValueError(f"duplicate key {key!r}")
        out[key] = loader.construct_object(v_node, deep=deep)
    return out


_DupCheckLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_dup)


def check_static():
    print("\n[STATIC] rules/*.yaml 파싱 + 중복 키")
    for fn in sorted(os.listdir(os.path.join(PROJ, "rules"))):
        if fn.endswith((".yaml", ".yml")):
            try:
                yaml.load(open(os.path.join(PROJ, "rules", fn), encoding="utf-8"), _DupCheckLoader)
            except Exception as e:
                fail(f"rules/{fn}: {type(e).__name__}: {str(e)[:140]}")

    print("[STATIC] src/*.py import")
    for fn in sorted(os.listdir(os.path.join(PROJ, "src"))):
        if fn.endswith(".py"):
            try:
                importlib.import_module(fn[:-3])
            except Exception as e:
                fail(f"src/{fn}: {type(e).__name__}: {str(e)[:140]}")
    if failures:
        return  # schema checks below need the modules to import

    import summarize

    print("[STATIC] symbols.yaml 스키마")
    S = summarize.load_symbols()
    prios = []
    for name, spec in S.items():
        if not isinstance(spec, dict):
            fail(f"symbols.yaml {name}: mapping이 아님")
            continue
        if not spec.get("aliases") and not spec.get("from"):
            fail(f"symbols.yaml {name}: aliases도 from도 없음")
        if spec.get("dim"):
            if spec.get("priority") is None:
                fail(f"symbols.yaml {name}: dim인데 priority 없음")
            else:
                prios.append(spec["priority"])
        for f in ("scope",):
            if spec.get(f):
                try:
                    re.compile(spec[f])
                except re.error as e:
                    fail(f"symbols.yaml {name}: {f} 정규식 오류 ({e})")
    dupes = {p for p in prios if prios.count(p) > 1}
    if dupes:
        fail(f"symbols.yaml: dim priority 중복 {sorted(dupes)} (렌더 순서가 비결정적이 됨)")

    print("[STATIC] derived_dims.yaml 스키마")
    D = summarize.load_derived_dims()
    for i, r in enumerate(D.get("rules") or []):
        for f in ("expr", "name", "sym"):
            if not r.get(f):
                fail(f"derived_dims rule[{i}] ({r.get('expr')}): '{f}' 없음")
        if r.get("scope"):
            try:
                re.compile(r["scope"])
            except re.error as e:
                fail(f"derived_dims rule[{i}]: scope 정규식 오류 ({e})")


# ---------------------------------------------------------------- FLEET
def _leaves(v, out):
    if isinstance(v, list):
        for x in v:
            _leaves(x, out)
    elif v is not None:
        out.append(v)


def scan_model(name):
    """Metrics for one published model. bare-int counting looks at shape ELEMENTS only --
    a coefficient inside an expression ("2*d_head") is not a bare integer, and miscounting
    those is how an earlier audit produced a wrong conclusion."""
    d = os.path.join(MODELS, name)
    m = {"c_fail": 0, "c17": "?", "unresolved": 0, "bare": 0, "bare_pct": 0.0,
         "unknown_syms": 0, "kv_card": None}

    report = os.path.join(d, "full", "report.md")
    if os.path.exists(report):
        for line in open(report, encoding="utf-8"):
            if re.match(r"^C\d+\s+FAIL", line):
                m["c_fail"] += 1
            mm = re.match(r"^C17\s+(\w+)", line)
            if mm:
                m["c17"] = mm.group(1)

    summary = os.path.join(d, "model_summary.md")
    if os.path.exists(summary):
        text = open(summary, encoding="utf-8").read()
        m["unresolved"] = text.count("미해결 — 아래 Tier 3")
        m["unknown_syms"] = len(re.findall(r"^\| (\w+) \| _\(미확인", text, re.M))
        km = re.search(r"KV CACHE / TOKEN \(BF16\) \| ([\d.]+ KiB \([A-Za-z ]+\))", text)
        if km:
            m["kv_card"] = km.group(1)

    raw = os.path.join(d, "full", "prefill.trace.raw.jsonl")
    if os.path.exists(raw):
        bare = sym = 0
        for line in open(raw, encoding="utf-8"):
            r = json.loads(line)
            el = []
            for fld in ("input_shape", "output_shape", "weight_shape"):
                _leaves(r.get(fld), el)
            for e in el:
                if str(e).isdigit():
                    if int(e) > 1:
                        bare += 1
                else:
                    sym += 1
        m["bare"] = bare
        m["bare_pct"] = round(100 * bare / max(1, bare + sym), 2)
    return m


def check_fleet():
    print("\n[FLEET] 모델별 지표")
    names = sorted(n for n in os.listdir(MODELS) if os.path.isdir(os.path.join(MODELS, n)))
    out = {}
    print(f"   {'model':46s} {'Cfail':>5} {'C17':>5} {'unres':>5} {'bare':>7} {'bare%':>6} {'?sym':>5}")
    for n in names:
        m = scan_model(n)
        out[n] = m
        print(f"   {n:46s} {m['c_fail']:5d} {m['c17']:>5} {m['unresolved']:5d} "
              f"{m['bare']:7d} {m['bare_pct']:6.2f} {m['unknown_syms']:5d}")
        if m["c_fail"]:
            fail(f"{n}: C체크 FAIL {m['c_fail']}개")
        if m["c17"] not in ("PASS", "?"):
            fail(f"{n}: C17={m['c17']} (온보딩 미완 — 02-new-module-handling.md Phase 0)")
        if m["unresolved"]:
            fail(f"{n}: 미해결 유도 상수 {m['unresolved']}개")
    return out


# ---------------------------------------------------------------- EXTERNAL
def check_external(fleet):
    print("\n[EXTERNAL] KV cache 카드 vs 공개 갤러리 수치")
    refs = yaml.safe_load(open(REFS, encoding="utf-8"))
    want = refs["kv_cache_per_token"]["values"]
    ok = 0
    for name, expected in sorted(want.items()):
        got = (fleet.get(name) or {}).get("kv_card")
        if got is None:
            warn(f"{name}: KV 카드 없음 (모델 미생성?)")
        elif got != expected:
            fail(f"{name}: KV 카드 '{got}' != 공개값 '{expected}'")
        else:
            ok += 1
    print(f"   {ok}/{len(want)} 일치")


# ---------------------------------------------------------------- BASELINE
def check_baseline(fleet, update):
    print("\n[BASELINE] 이전 기준 대비 퇴행 검사")
    cur = {n: {"bare": m["bare"], "unresolved": m["unresolved"],
               "unknown_syms": m["unknown_syms"], "c_fail": m["c_fail"]}
           for n, m in fleet.items()}
    if update:
        os.makedirs(os.path.dirname(BASELINE), exist_ok=True)
        json.dump(cur, open(BASELINE, "w", encoding="utf-8"), indent=2, sort_keys=True)
        print(f"   기준 갱신됨 -> {os.path.relpath(BASELINE, PROJ)}")
        return
    if not os.path.exists(BASELINE):
        warn("기준 파일 없음 — --update-baseline 로 최초 기록 필요")
        return
    old = json.load(open(BASELINE, encoding="utf-8"))
    regressed = improved = 0
    for n, c in sorted(cur.items()):
        o = old.get(n)
        if o is None:
            print(f"   NEW   {n}")
            continue
        for key in ("bare", "unresolved", "unknown_syms", "c_fail"):
            if c[key] > o[key]:
                fail(f"{n}: {key} 퇴행 {o[key]} -> {c[key]}")
                regressed += 1
            elif c[key] < o[key]:
                improved += 1
    for n in sorted(set(old) - set(cur)):
        warn(f"{n}: 기준에는 있는데 지금 없음 (모델이 사라짐)")
    print(f"   퇴행 {regressed}건 / 개선 {improved}건")
    if improved and not regressed:
        print("   -> 개선만 있음. 검토 후 --update-baseline 로 기준을 올릴 것.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--update-baseline", action="store_true",
                    help="현재 지표를 새 기준으로 기록(검토 후에만)")
    args = ap.parse_args()

    check_static()
    if failures:
        print("\n=== STATIC 단계에서 실패 — 이후 검사 생략 ===")
    else:
        fleet = check_fleet()
        check_external(fleet)
        check_baseline(fleet, args.update_baseline)

    print("\n" + "=" * 72)
    print(f"FAIL {len(failures)}건 / WARN {len(warnings)}건")
    for f in failures:
        print(f"  - {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
