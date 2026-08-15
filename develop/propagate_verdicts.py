"""④층 판정을 **같은 아키텍처의 다른 모델**로 옮긴다.

WHY THIS EXISTS
---------------
판정은 소스를 읽어서 나온다. 그런데 소스는 모델마다 다르지 않다 -- Kimi-K2-Instruct /
K2.6 / K2.7-Code / DeepSeek-V3 / tiny-deepseek-v3 는 **같은 modeling 파일**을 돈다. 그래서
그 다섯에서 같은 질문이 다섯 번 올라오고, 검토자가 같은 코드 줄을 다섯 번 읽어야 했다.

이 스크립트는 이미 내려진 판정을 형제 모델에서 **구조적으로 같은 자리**에 옮겨 붙인다.

무엇을 근거로 "같은 자리"라고 하는가
------------------------------------
`shape` 과 `expect` 는 모델마다 다르다(폭이 다르니까). 옮길 수 있는 것은 **구조**뿐이다:

    module  ·  op_type  ·  nth  ·  field  ·  shape_index  ·  axis  ·  현재 이름

이 일곱이 같으면 그 축은 같은 코드 줄이 만든 같은 자리다. 그리고 옮겨 붙일 때
`shape`/`expect` 는 **그 모델 자신의 인계 목록에서 읽어 온다** -- 남의 숫자를 베끼지 않는다.
그래서 결과 항목은 원본과 똑같이 완전하고, 게이트의 안전장치(`expect` 불일치면 발화 안 함,
발화 0건이면 FAIL)가 그대로 걸린다.

기본은 **보고만** 한다. `--write` 를 줘야 파일에 쓴다 -- 판정을 자동으로 퍼뜨리는 것은
이 프로젝트가 계속 잡아온 실패("근거 없이 그럴듯하게 채우기")와 한 끗 차이다.

실행:
    .venv\\Scripts\\python.exe develop\\propagate_verdicts.py           # 후보만 출력
    .venv\\Scripts\\python.exe develop\\propagate_verdicts.py --write   # 실제로 추가
"""
import argparse
import collections
import io
import json
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
MODELS = os.path.join(PROJ, "models")
OV = os.path.join(PROJ, "rules", "label_overrides.yaml")
CF = os.path.join(PROJ, "rules", "label_confirmed.yaml")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def arch_groups() -> dict:
    """{아키텍처 서명: [모델 폴더명]} — 서명은 그 모델이 쓰는 **모듈 클래스 집합**.

    `model_type` 이 아니라 클래스 집합인 이유: Kimi-K2.6 의 model_type 은 `kimi_k25` 지만
    실제로 도는 코드는 DeepSeek-V3 계열이다. 클래스 집합은 실행된 것에서 나오므로 거짓말을
    하지 않는다.
    """
    g = collections.defaultdict(list)
    for m in sorted(os.listdir(MODELS)):
        p = os.path.join(MODELS, m, "full", "module_classes.json")
        if not os.path.exists(p):
            continue
        try:
            j = json.load(io.open(p, encoding="utf-8")) or {}
        except (ValueError, OSError):
            continue
        sig = frozenset(x for v in j.values() for x in (v if isinstance(v, list) else [v]))
        if sig:
            g[sig].append(m)
    return g


def _unsettled(model: str) -> list:
    out = []
    for ph in ("prefill", "decode"):
        p = os.path.join(MODELS, model, "full", f"{ph}.unsettled.json")
        if not os.path.exists(p):
            continue
        try:
            out += (json.load(io.open(p, encoding="utf-8")) or {}).get("items") or []
        except (ValueError, OSError):
            pass
    return out


_STRUCT = ("op_type", "nth", "field", "shape_index", "axis")


def _key(spec: dict, label: str) -> tuple:
    return (spec.get("module"),) + tuple(spec.get(k) for k in _STRUCT) + (str(label),)


def _shape(spec: dict) -> tuple:
    return tuple(str(x) for x in (spec.get("shape") or []))


def candidates(kind: str, ents=None, groups=None, unsettled=None) -> list:
    """옮겨 붙일 수 있는 항목. kind 는 'override' | 'confirm'.

    출처 쪽은 **구조로 접는다** — 하나의 판정이 prefill/decode 두 항목으로 적혀 있어도
    판정은 하나다. 대상 쪽은 **펼친다** — 형제 모델에서 그 자리가 두 phase 에 각각
    나타나고 shape 가 다르면 항목도 둘이어야 발화한다.

    `ents`/`groups`/`unsettled` 는 자기검증이 결함을 직접 먹이기 위한 주입구다
    (verify_selftest.py). 평소에는 셋 다 실제 파일에서 읽는다.
    """
    path = OV if kind == "override" else CF
    root = "overrides" if kind == "override" else "confirmed"
    if ents is None:
        if not os.path.exists(path):
            return []
        ents = (yaml.safe_load(io.open(path, encoding="utf-8")) or {}).get(root) or []
    _un = unsettled or _unsettled
    where = {}
    for sig, models in (groups or arch_groups()).items():
        for m in models:
            where[m] = (sig, models)

    def _cur(e):
        return e.get("from") if kind == "override" else e.get("label")

    have = {(e.get("model"),) + _key(e, _cur(e)) for e in ents}

    # (model, 구조키) -> (판정, 그 키로 실제 판정된 **원본 shape 들**)
    #
    # shape 을 따로 모으는 이유: 일곱 키가 같아도 **같은 코드 줄이 아닐 수 있다.** `nth` 는
    # "그 모듈 안에서 같은 op_type 의 몇 번째"인데, 레이어 유형마다 앞서 실행되는 op 개수가
    # 다르면 같은 `nth` 가 서로 다른 줄을 가리킨다. DeepSeek-V4-Flash-0731 의
    # `self_attn/slice/nth10/axis1` 이 그렇다 -- 레이어 0~1 에서는 sink 확률을 버리는
    # `scores = probs[..., :-1]`(modeling_deepseek_v4.py:732) 이고, 레이어 2~42 에서는
    # RoPE 출력의 마지막 축을 가르는 slice(:853) 다. `axis 1 = n_h` 라는 **결론**은 네 자리
    # 모두 맞지만, 원본 판정의 `source` 는 RoPE slice 를 인용하므로 sink 제거 자리에 붙이면
    # 근거가 그 코드를 설명하지 못한다. 적용기는 이들을 shape 로 구분해 별개 항목으로 두는데,
    # 전파기만 shape 을 버리고 다시 합치고 있었다. 외부 검토(Codex)가 --write 를 거부하며
    # 이 반례를 제시, 2026-08-15.
    verdicts = {}
    for e in ents:
        if e.get("model") in where and e.get("op_type"):
            k = (e["model"],) + _key(e, _cur(e))
            verdicts.setdefault(k, (e, set()))[1].add(_shape(e))

    out, seen = [], set()
    for (src_model, *key), (e, src_shapes) in verdicts.items():
        key = tuple(key)
        for sib in where[src_model][1]:
            if sib == src_model or (sib,) + key in have:
                continue
            for it in _un(sib):
                st = it["override_stub"]
                if it.get("stub_ambiguous") or _key(st, it["current_label"]) != key:
                    continue              # 지목이 안 되거나 같은 자리가 아니면 옮기지 않는다
                # 원본이 **그 shape 에 대해** 실제로 판정한 적이 있어야 한다. 위 주석의
                # `nth` 충돌이 여기서 걸린다: 0731 의 같은 키가 네 shape 을 내는데, 원본
                # Flash 가 판정한 것은 `[B,n_h,T,d_head-d_rope]` 와 그 decode 짝뿐이고
                # `[B,n_h,T,T]`(sink 제거)·`[B,n_h,1,w_local]` 은 원본에서도 여전히 열린
                # 질문이다. 원본에 없던 shape 은 **새 판정이 필요한 자리**이지 옮길 자리가 아니다.
                if _shape(st) not in src_shapes:
                    continue
                # 그리고 옮기려는 이름이 대상 모델 자신의 후보에 없다면, 그 모델의 규칙은 그
                # 이름을 고려조차 하지 않았다는 뜻이다(= 그 축의 폭이 그 심볼 값과 다르다).
                # 사용자가 DeepSeek-V4-Flash `view/nth2/ax2` 의 후보가 {d_rope, n_h, n_h_I}
                # 인데 V4-Pro 의 같은 자리는 {d_rope, n_h_I} 로 `n_h` 가 없다는 것을 짚어
                # 추가했다. 지금 후보 중엔 위반이 없어 **예방 가드**다, 2026-08-15.
                want = e["to"] if kind == "override" else e["label"]
                if want not in (it.get("candidates") or []):
                    continue
                new = {"model": sib, "module": st["module"], "spread": "class",
                       "shape": st["shape"], "axis": st["axis"], "field": st["field"],
                       "shape_index": st["shape_index"], "op_type": st["op_type"],
                       "nth": st["nth"], "expect": it["size"]}
                if kind == "override":
                    new["from"], new["to"] = e["from"], e["to"]
                else:
                    new.pop("spread", None)
                    new["label"] = e["label"]
                sig = (sib, tuple(sorted((k, repr(v)) for k, v in new.items())))
                if sig in seen:
                    continue              # 같은 자리가 두 phase 에 있어도 shape 가 같으면 하나
                seen.add(sig)
                new["source"] = (str(e.get("source", "")).rstrip() +
                                 f"  (같은 아키텍처의 {src_model} 에서 내린 같은 판정을 "
                                 f"구조적으로 같은 자리에 옮김 — module/op_type/nth/field/"
                                 f"shape_index/axis 와 현재 이름이 모두 일치. shape·expect 는 "
                                 f"이 모델 자신의 값이다.)")
                out.append((src_model, new))
    return out


_ORDER = ("model", "module", "spread", "shape", "axis", "field", "shape_index",
          "op_type", "nth", "from", "to", "label", "expect")


def _emit(e: dict, kind: str) -> str:
    """항목 하나를 손으로 쓴 것과 같은 서식으로 낸다.

    yaml.safe_dump 로 파일 전체를 다시 쓰면 사람이 붙여 놓은 주석과 `>` 접힌 블록이
    전부 날아간다 -- 이 파일은 근거를 읽으려고 여는 파일이므로 서식이 내용이다.
    그래서 덧붙이기만 한다.
    """
    fields = []
    for k in _ORDER:
        if k not in e:
            continue
        v = e[k]
        if k == "shape":
            v = "[" + ", ".join(f'"{x}"' for x in v) + "]"
        elif isinstance(v, str) and not v.replace("_", "").replace(".", "").isalnum():
            v = "'" + v.replace("'", "''") + "'"
        fields.append(f"{k}: {v}")

    wrapped, line = [], ""
    for w in " ".join(str(e.get("source", "")).split()).split(" "):
        if line and len(line) + len(w) + 1 > 88:
            wrapped.append(line)
            line = ""
        line = f"{line} {w}".lstrip()
    wrapped.append(line)

    return ("  - " + fields[0] + "\n"
            + "".join("    " + f + "\n" for f in fields[1:])
            + "    source: >\n"
            + "".join("      " + w + "\n" for w in wrapped))


def main():
    ap = argparse.ArgumentParser(description="④층 판정을 같은 아키텍처의 다른 모델로 옮긴다")
    ap.add_argument("--write", action="store_true", help="실제로 rules/*.yaml 에 추가")
    a = ap.parse_args()

    total = 0
    for kind, path, root in (("override", OV, "overrides"), ("confirm", CF, "confirmed")):
        cands = candidates(kind)
        if not cands:
            print(f"[{kind}] 옮길 수 있는 항목 없음")
            continue
        print(f"\n[{kind}] 옮길 수 있는 항목 {len(cands)}건")
        for src, e in cands:
            what = f"{e.get('from')} -> {e.get('to')}" if kind == "override" else f"label {e.get('label')}"
            print(f"  {src.split('__')[-1][:24]:<24} -> {e['model'].split('__')[-1][:24]:<24} "
                  f"{e['op_type']}/nth{e['nth']}/ax{e['axis']}  {what}  expect {e['expect']}  "
                  f"[{','.join(map(str, e['shape']))}]")
        total += len(cands)
        if a.write:
            body = io.open(path, encoding="utf-8").read().rstrip("\n")
            body += "\n" + "".join(_emit(e, kind) for _, e in cands)
            io.open(path, "w", encoding="utf-8", newline="\n").write(body)
            n = len(yaml.safe_load(io.open(path, encoding="utf-8"))[root])
            print(f"  -> {os.path.relpath(path, PROJ)} 에 {len(cands)}건 추가 (총 {n}건)")

    if total and not a.write:
        print("\n실제로 넣으려면 --write. 넣은 뒤에는 반드시:")
        print("  develop/regen_summaries.py  ->  develop/verify_all.py (EXIT 0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
