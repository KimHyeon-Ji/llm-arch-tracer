r"""**표가 주장하는 구조를 그대로 뽑는다** -- 외부 검토용 부록.

WHY THIS EXISTS
---------------
검토자에게 "이 표가 그 모델의 아키텍처라고 말하는 내용"을 통째로 보여주려면 발행 CSV 를
그대로 옮기면 되지만 너무 길다. 접힌 표는 이미 반복을 한 번으로 줄여 놨으므로, **템플릿마다
한 벌씩** 내면 그것이 곧 주장 전부다.

**`block_type` 만으로 묶으면 안 된다.** 2026-09-16 외부 검토(Codex)가 잡았다 -- 그렇게
묶으면 서로 다른 블록이 같은 문자열 아래 합쳐지고 대표 하나만 남아 **실제 표에 있는 구조가
부록에서 사라진다**:

    Llama-4   NoPE 블록(layers 3,7,...,47)  이 통째로 빠짐
    gpt-oss   홀수층 full attention          이 통째로 빠짐
    V4-Pro    CSA+hash MoE / HCA+동적 MoE / CSA+동적 MoE  세 블록이 빠짐

부록만 보면 Llama-4 가 36층, gpt-oss 가 12층, V4-Pro 가 2층짜리 모델로 보였다. 실제 CSV 에는
전부 있었으므로 산출물의 결함이 아니라 **부록의 결함**이었다.

묶는 키는 `(block_type, repeat, layers)` 다. `layers` 가 서로 다른 레이어 집합을 가르고,
그것이 곧 attention/position/routing 구성이 다르다는 뜻이다.

실행:
    .venv\Scripts\python.exe develop\make_claims_appendix.py <모델폴더> [...] > 부록.md
"""
import collections
import csv
import io
import json
import os
import sys


def _key(r):
    return (r.get("block_type") or "", r.get("repeat") or "", r.get("layers") or "")


def render(d: str) -> list:
    name = os.path.basename(os.path.normpath(d))
    prov = json.load(io.open(os.path.join(d, "full", "provenance.json"), encoding="utf-8"))
    mid = prov.get("model_id") or name.replace("__", "/")
    st = prov.get("symbol_table") or {}
    out = [f"## {mid}", ""]
    out.append(f"발행 좌표 **B={prov.get('capture_batch')}, T={prov.get('seq_len_used')}**. 심볼 표:")
    out.append("")
    out.append("```")
    out.append(", ".join(f"{k}={v}" for k, v in sorted(st.items())))
    out.append("```")
    out.append("")
    for phase in ("prefill", "decode"):
        p = os.path.join(d, f"{phase}.csv")
        if not os.path.isfile(p):
            continue
        rows = list(csv.DictReader(io.open(p, encoding="utf-8", newline="")))
        groups = collections.OrderedDict()
        for r in rows:
            groups.setdefault(_key(r), []).append(r)
        out.append(f"### {phase} — 템플릿 {len(groups)}개 / 전체 {len(rows)}행")
        out.append("")
        for (bt, rep, layers), v in groups.items():
            # 표가 이미 접혀 있으므로 그룹 하나가 곧 대표 한 벌이다. op_id 순서를 지킨다.
            v = sorted(v, key=lambda r: int(r.get("op_id") or 0))
            out.append(f"**`block_type={bt}`  repeat={rep}  layers=`{layers or '-'}`**  ({len(v)}행)")
            out.append("")
            out.append("```")
            for r in v:
                mp = (r.get("module_path") or "").split(".")
                short = ".".join(mp[3:]) if len(mp) > 3 and len(mp) > 1 and mp[1] == "layers" \
                    else (r.get("module_path") or "")
                w = f"  w={r['weight_shape']}" if r.get("weight_shape") else ""
                cav = "  [caveat]" if r.get("caveat") else ""
                out.append(f"{r['op_id']:>4} {r['op_type']:16} {short[:34]:34} "
                           f"{r['input_shape']} -> {r['output_shape']}{w}{cav}")
            out.append("```")
            out.append("")
    return out


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    print("# 부록 — 표가 주장하는 구조 (모델별)")
    print()
    print("각 모델의 발행 CSV 에서 **템플릿마다 한 벌씩** 뽑은 것입니다. 표는 접혀 있으므로")
    print("(같은 구조의 레이어는 한 번만 적고 `repeat`/`layers` 가 몇 개를 대표하는지 말합니다)")
    print("이것이 표가 그 모델의 아키텍처라고 주장하는 내용 전부입니다.")
    print()
    print("묶는 키는 `(block_type, repeat, layers)` 입니다. `block_type` 만으로 묶으면 서로 다른")
    print("블록이 합쳐져 구조가 사라집니다 -- 2026-09-16 검토에서 실제로 그랬습니다.")
    print()
    for a in sys.argv[1:]:
        for line in render(a):
            print(line)
