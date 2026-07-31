"""게이트 자체를 검증한다 — 폴트 인젝션.

`verify_all.py`는 산출물이 맞는지 판정한다. 그럼 **게이트가 맞는지는 누가 판정하나?**
"FAIL 0"은 두 가지 이유로 나올 수 있다:
  (a) 산출물에 결함이 없다
  (b) 검사가 결함을 못 잡는다
이 둘을 구분하지 않으면 "26/26 통과"는 아무 의미가 없다. 실제로 이 프로젝트에서
그 일이 일어났다 — 2026-07-30에 attention 레이어 수를 이름 기반으로 바꿨다가
falcon(32→0)과 Nemotron(4→0)이 조용히 attention-free로 뒤집혔는데 게이트는 통과시켰다.

그래서 각 검사마다 **그 검사가 잡기로 되어 있는 결함을 일부러 주입**하고, 게이트가
실제로 그걸 잡는지 확인한다. 잡으면 그 검사는 살아 있는 것이고, 못 잡으면 그 검사는
장식이다. 정상 산출물에서 0이 나오는지(오탐 없음)도 같이 본다.

실행:  .venv\\Scripts\\python.exe develop\\verify_selftest.py [--verbose]
종료코드 0 = 모든 검사가 자기 결함을 잡아냄.
"""
import argparse
import copy
import io
import json
import os
import shutil
import sys
import tempfile

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(PROJ, "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import verify_all as V  # noqa: E402

MODELS = V.MODELS


# ------------------------------------------------------------------ 주입 도구
def _edit_jsonl(path, fn, limit=None):
    """jsonl의 각 행에 fn(row)를 적용. fn이 True를 반환한 행만 센다."""
    rows, hit = [], 0
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if (limit is None or hit < limit) and fn(r):
            hit += 1
        rows.append(r)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return hit


def _first_shape_axis(r, field, pred):
    for sh in (r.get(field) or []):
        if isinstance(sh, list):
            for i, v in enumerate(sh):
                if pred(v):
                    return sh, i
    return None, None


# ------------------------------------------------------------------ 결함 정의
# 각 항목: (지표 이름, 사람이 읽는 설명, 어떤 모델에 주입할지, 주입 함수)
# 주입 함수는 모델 폴더 경로를 받아 결함을 심고, 심은 개수를 반환한다.

def inj_weight_T(d):
    """가중치 축에 T — 정적 파라미터가 시퀀스 길이에 의존한다는 물리적 불가능.

    weight_shape는 input/output_shape와 달리 **평면 리스트**(`["V","d_model"]`)다."""
    p = os.path.join(d, "full", "prefill.trace.raw.jsonl")

    def f(r):
        ws = r.get("weight_shape")
        if isinstance(ws, list) and ws:
            ws[0] = "T"
            return True
        return False
    return _edit_jsonl(p, f, limit=1)


def inj_head_excl(d):
    """한 shape에 n_h와 n_kv 동시 등장 — head 개수 축은 둘 중 하나뿐이다."""
    p = os.path.join(d, "full", "prefill.trace.raw.jsonl")

    def f(r):
        for sh in (r.get("output_shape") or []):
            if isinstance(sh, list) and len(sh) >= 3:
                sh[-2], sh[-1] = "n_h", "n_kv"
                return True
        return False
    return _edit_jsonl(p, f, limit=1)


def inj_resid_norm(d):
    """레이어 직속 LayerNorm의 활성 폭이 d_model이 아님 — 잔차 스트림 폭 위반."""
    p = os.path.join(d, "full", "prefill.trace.raw.jsonl")

    def f(r):
        if V._RESID_NORM.match(r.get("module_path") or ""):
            for sh in (r.get("output_shape") or []):
                if isinstance(sh, list) and sh:
                    sh[-1] = "n_h*d_head"
                    return True
        return False
    return _edit_jsonl(p, f, limit=1)


def inj_label_false(d):
    """산술적으로 거짓인 라벨 — 심볼표 대입값이 실측 구체값과 다름."""
    p = os.path.join(d, "full", "prefill.trace.raw.jsonl")

    def f(r):
        sh, i = _first_shape_axis(r, "output_shape", lambda v: str(v) not in ("B", "1"))
        if sh is not None:
            sh[i] = "V"          # vocab size를 아무 축에나 붙이면 거의 항상 거짓
            return True
        return False
    return _edit_jsonl(p, f, limit=1)


def inj_param_incons(d):
    """같은 파라미터가 op마다 다르게 라벨링됨.

    게이트는 `params`가 정확히 1개인 행만 파라미터별로 모으므로(verify_all._label_checks),
    같은 파라미터를 두 번째로 만났을 때 weight_shape를 뒤집어 두 가지 라벨을 만든다."""
    p = os.path.join(d, "full", "prefill.trace.raw.jsonl")
    seen = set()

    def f(r):
        ps, ws = (r.get("params") or []), r.get("weight_shape")
        if len(ps) != 1 or not isinstance(ws, list) or len(ws) < 2 or ws[0] == ws[1]:
            return False
        if ps[0] not in seen:
            seen.add(ps[0])
            return False
        ws[0], ws[1] = ws[1], ws[0]
        return True
    return _edit_jsonl(p, f, limit=1)


def inj_flow_wrong(d):
    """데이터플로우 불일치 — 같은 텐서인데 한쪽만 정수로 방치."""
    p = os.path.join(d, "full", "prefill.trace.raw.jsonl")
    conc = {json.loads(l)["op_id"]: json.loads(l)
            for l in open(os.path.join(d, "full", "prefill.shapes.concrete.jsonl"),
                          encoding="utf-8")}

    def f(r):
        if not (r.get("depends_on") and r.get("input_shape")):
            return False
        c = conc.get(r["op_id"]) or {}
        for sh, csh in zip(r["input_shape"], c.get("input_shape") or []):
            if isinstance(sh, list) and isinstance(csh, list):
                for i, (v, cv) in enumerate(zip(sh, csh)):
                    if not str(v).isdigit() and str(v) != "B" and isinstance(cv, int) and cv > 1:
                        sh[i] = str(cv)      # 이름을 지우고 정수로 되돌린다
                        return True
        return False
    return _edit_jsonl(p, f, limit=1)


def inj_self_contra(d):
    """심볼표가 자기 트레이스와 모순 — 트레이스에 expert가 있는데 E=0."""
    p = os.path.join(d, "structure.yaml")
    y = yaml.safe_load(open(p, encoding="utf-8")) or {}
    if "E" not in (y.get("symbols") or {}):
        return 0
    y["symbols"]["E"] = 0
    yaml.safe_dump(y, open(p, "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
    return 1


def inj_c_fail(d):
    """C 체크 FAIL — report.md에 실패가 있는데 게이트가 못 보면 안 된다."""
    p = os.path.join(d, "full", "report.md")
    t = open(p, encoding="utf-8").read()
    open(p, "w", encoding="utf-8").write(t.replace("C5   PASS", "C5   FAIL", 1))
    return 1


def inj_c17(d):
    """C17 미통과 — 신규 모듈 온보딩이 안 끝난 상태."""
    p = os.path.join(d, "full", "report.md")
    t = open(p, encoding="utf-8").read()
    open(p, "w", encoding="utf-8").write(t.replace("C17  PASS", "C17  WARN", 1))
    return 1


def inj_unresolved(d):
    """미해결 유도 상수가 남은 채로 통과하면 안 된다."""
    p = os.path.join(d, "model_summary.md")
    t = open(p, encoding="utf-8").read()
    open(p, "w", encoding="utf-8").write(t + "\n미해결 — 아래 Tier 3\n")
    return 1


def inj_attn_layers(d):
    """attention 레이어 수 오판 — 이름 기반 규칙이 falcon/Nemotron을 뒤집었던 그 사고."""
    p = os.path.join(d, "full", "prefill.trace.raw.jsonl")

    def f(r):
        mp = r.get("module_path") or ""
        if mp.endswith(("q_proj", "query_key_value", "c_attn", "qkv_proj", "Wqkv")):
            r["module_path"] = mp.rsplit(".", 1)[0] + ".xxx_proj"
            return True
        return False
    return _edit_jsonl(p, f)


def inj_kv_card(d):
    """KV cache 카드 수치가 공개 갤러리와 어긋남."""
    p = os.path.join(d, "model_summary.md")
    t = open(p, encoding="utf-8").read()
    import re as _re
    t2 = _re.sub(r"(KV CACHE / TOKEN \(BF16\) \| )[\d.]+( KiB)", r"\g<1>999.9\2", t, count=1)
    open(p, "w", encoding="utf-8").write(t2)
    return 1 if t2 != t else 0


# 지표 이름 -> (설명, 대상 모델, 주입 함수). 대상 모델은 그 결함이 성립하는 모델로 고른다.
CASES = [
    ("weight_T",     "가중치 축에 T (물리적 불가능)",           "meta-llama__Llama-3.1-8B",  inj_weight_T),
    ("head_excl",    "한 shape에 n_h + n_kv 동시",              "meta-llama__Llama-3.1-8B",  inj_head_excl),
    ("resid_norm",   "레이어 직속 LayerNorm 폭 != d_model",     "meta-llama__Llama-3.1-8B",  inj_resid_norm),
    ("label_false",  "산술적으로 거짓인 라벨",                   "google__gemma-2-2b",        inj_label_false),
    ("param_incons", "같은 파라미터의 라벨 불일치",              "google__gemma-2-2b",        inj_param_incons),
    ("flow_wrong",   "데이터플로우 라벨 불일치(이름->정수)",     "Qwen__Qwen2.5-0.5B",        inj_flow_wrong),
    ("self_contra",  "심볼표가 자기 트레이스와 모순 (E=0)",      "Qwen__Qwen3-30B-A3B",       inj_self_contra),
    ("c_fail",       "C 체크 FAIL",                              "Qwen__Qwen2.5-0.5B",        inj_c_fail),
    ("c17",          "C17 미통과",                               "Qwen__Qwen2.5-0.5B",        inj_c17),
    ("unresolved",   "미해결 유도 상수 잔존",                    "Qwen__Qwen2.5-0.5B",        inj_unresolved),
]

# 외부 대조 검사는 scan_model 지표가 아니라 별도 함수라 따로 돌린다.
EXTERNAL_CASES = [
    ("attention_layers", "attention 레이어 수 오판", "tiiuae__falcon-7b", inj_attn_layers),
    ("kv_cache",         "KV cache 카드 수치 오류",  "openai__gpt-oss-20b", inj_kv_card),
]


# ------------------------------------------------- STATIC / BASELINE 검사 자기검증
# 이 둘은 모델 폴더가 아니라 rules/ 와 baseline.json을 읽으므로 샌드박스 방식이 다르다.
# rules/를 실제로 망가뜨리는 건 위험하므로(2026-07-29에 PowerShell 정규식이 symbols.yaml을
# 손상시킨 전력이 있다) **검사 로직에 직접 결함 입력을 먹인다**.

def _static_cases():
    import summarize
    out = []

    def run_static_with(symbols):
        orig = summarize.load_symbols
        summarize.load_symbols = lambda *a, **k: symbols
        before = len(V.failures)
        buf, real_out = io.StringIO(), sys.stdout
        try:
            sys.stdout = buf
            V.check_static()
        finally:
            sys.stdout = real_out
            summarize.load_symbols = orig
        n = len(V.failures) - before
        del V.failures[before:]
        return n

    good = {"d_model": {"aliases": ["hidden_size"], "dim": True, "priority": 1}}
    base_noise = run_static_with(good)          # 정상 입력에서 나오는 FAIL 수(0이어야 함)

    out.append(("static:priority중복", "dim priority가 겹치면 렌더 순서가 비결정적",
                run_static_with({
                    "a": {"aliases": ["x"], "dim": True, "priority": 3},
                    "b": {"aliases": ["y"], "dim": True, "priority": 3}}) > base_noise))
    out.append(("static:priority누락", "dim인데 priority가 없음",
                run_static_with({"a": {"aliases": ["x"], "dim": True}}) > base_noise))
    out.append(("static:scope정규식", "scope 정규식이 컴파일 불가",
                run_static_with({"a": {"aliases": ["x"], "scope": "(("}}) > base_noise))
    out.append(("static:출처없음", "aliases도 from도 없는 심볼",
                run_static_with({"a": {"meaning": "x"}}) > base_noise))

    # YAML 중복 키: safe_load는 조용히 마지막 것만 남긴다 -> 규칙이 소리없이 사라진다.
    dup = "a: 1\nb: 2\na: 3\n"
    caught = False
    try:
        yaml.load(io.StringIO(dup), V._DupCheckLoader)
    except Exception:
        caught = True
    out.append(("static:YAML중복키", "같은 키 두 번 = 앞 규칙이 조용히 증발", caught))
    return out


def _baseline_case():
    """퇴행 검사: 지표가 나빠졌는데 통과시키면 안 된다."""
    import tempfile as _tf
    # check_baseline이 읽는 지표를 전부 갖춘 스텁(하나라도 빠지면 KeyError로 조용히 못 돈다).
    stub = {"c_fail": 0, "c17": "PASS", "unresolved": 0, "bare": 10, "bare_pct": 0.0,
            "unknown_syms": 0, "kv_card": None, "weight_T": 0, "self_contra": 0,
            "label_false": 0, "param_incons": 0, "flow_wrong": 0, "flow_ambig": 0,
            "head_excl": 0, "resid_norm": 0}
    fleet_before = {"m": dict(stub)}
    fleet_after = {"m": dict(stub, bare=99)}                # bare 악화
    fd = os.path.join(_tf.mkdtemp(), "baseline.json")
    json.dump(fleet_before, open(fd, "w"))
    real, V.BASELINE = V.BASELINE, fd
    before = len(V.failures)
    buf, real_out = io.StringIO(), sys.stdout
    try:
        sys.stdout = buf
        V.check_baseline(fleet_after, False)
    finally:
        sys.stdout = real_out
        V.BASELINE = real
    n = len(V.failures) - before
    del V.failures[before:]
    return n > 0


def _sandbox(name):
    tmp = tempfile.mkdtemp(prefix="vst_")
    shutil.copytree(os.path.join(MODELS, name), os.path.join(tmp, name))
    return tmp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    print("게이트 자기검증 — 각 검사에 그 검사가 잡아야 할 결함을 주입한다\n")
    ok = bad = 0

    print(f"   {'검사':16s} {'모델':38s} {'정상':>6} {'주입후':>7}  판정")
    for metric, desc, model, inject in CASES:
        if not os.path.isdir(os.path.join(MODELS, model)):
            print(f"   {metric:16s} {model:38s}  SKIP (모델 없음)")
            continue
        clean = V.scan_model(model)[metric]
        clean_n = 0 if clean in (0, "PASS") else 1
        tmp = _sandbox(model)
        try:
            n = inject(os.path.join(tmp, model))
            real, V.MODELS = V.MODELS, tmp
            dirty = V.scan_model(model)[metric]
            V.MODELS = real
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        dirty_n = 0 if dirty in (0, "PASS") else 1
        caught = (clean_n == 0) and (dirty_n == 1) and n > 0
        print(f"   {metric:16s} {model:38s} {str(clean):>6} {str(dirty):>7}  "
              f"{'OK — 잡음' if caught else 'FAIL — 못 잡음'}")
        if args.verbose:
            print(f"        {desc} (주입 {n}건)")
        ok, bad = (ok + 1, bad) if caught else (ok, bad + 1)

    print()
    for metric, desc, model, inject in EXTERNAL_CASES:
        if not os.path.isdir(os.path.join(MODELS, model)):
            print(f"   {metric:16s} {model:38s}  SKIP (모델 없음)")
            continue
        tmp = _sandbox(model)
        try:
            n = inject(os.path.join(tmp, model))
            real, V.MODELS = V.MODELS, tmp
            before = len(V.failures)
            # 샌드박스엔 모델이 하나뿐이라 나머지는 전부 "트레이스 없음" WARN이 된다.
            # 그 소음은 삼키고 FAIL 발생 여부만 본다.
            buf, real_out = io.StringIO(), sys.stdout
            try:
                sys.stdout = buf
                V.check_external({model: V.scan_model(model)})
            finally:
                sys.stdout = real_out
            after = len(V.failures)
            V.MODELS = real
            del V.failures[before:]           # 주입 실패를 최종 집계에 남기지 않는다
            del V.warnings[:]
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        caught = after > before and n > 0
        print(f"   {metric:16s} {model:38s} {'':>6} {'':>7}  "
              f"{'OK — 잡음' if caught else 'FAIL — 못 잡음'}")
        if args.verbose:
            print(f"        {desc} (주입 {n}건)")
        ok, bad = (ok + 1, bad) if caught else (ok, bad + 1)

    print()
    for name, desc, caught in _static_cases():
        print(f"   {name:16s} {'(검사 로직에 직접 입력)':38s} {'':>6} {'':>7}  "
              f"{'OK — 잡음' if caught else 'FAIL — 못 잡음'}")
        if args.verbose:
            print(f"        {desc}")
        ok, bad = (ok + 1, bad) if caught else (ok, bad + 1)

    caught = _baseline_case()
    print(f"   {'baseline':16s} {'(지표 악화 시뮬레이션)':38s} {'':>6} {'':>7}  "
          f"{'OK — 잡음' if caught else 'FAIL — 못 잡음'}")
    if args.verbose:
        print("        이전 기준보다 지표가 나빠졌는데 통과시키면 안 된다")
    ok, bad = (ok + 1, bad) if caught else (ok, bad + 1)

    print("\n" + "=" * 72)
    print(f"살아 있는 검사 {ok}개 / 죽은 검사 {bad}개")
    if bad:
        print("죽은 검사가 있다 — 그 검사의 'FAIL 0'은 아무것도 보장하지 않는다.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
