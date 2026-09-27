r"""`plus_at/DIFF.md` 를 만든다 — 사람이 읽는 변경 요약.

MANIFEST.json 과 actual_footprint.jsonl 에서만 나온다. 그래서 **해시 대상 출력이 아니다**
(자기 해시를 자기 안에 적을 수 없다). 전수 목록은 actual_footprint.jsonl 이 권위다.

실행:  .venv/Scripts/python.exe develop/plus_at_diff.py <모델명>
"""
import collections
import io
import json
import os
import re
import sys

import yaml

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
M = sys.argv[1]
D = os.path.join("models", M, "plus_at")
man = json.load(io.open(os.path.join(D, "MANIFEST.json"), encoding="utf-8"))
fp = [json.loads(l) for l in io.open(os.path.join(D, "actual_footprint.jsonl"),
                                     encoding="utf-8") if l.strip()]
L = []
A = L.append
A(f"# DIFF — `{M}` derived view (+@)")
A("")
A("**이 문서는 사람이 읽는 요약이다.** 전수 목록은 `actual_footprint.jsonl` "
  f"({len(fp)} 줄)이고, 그 digest 가 `expected_footprint.jsonl` 과 같아야 한다.")
A("")
A("```")
A(f"status                {man['status']}")
A(f"base_results_commit   {man['base_results_commit']}")
A(f"tool_source_commit    {man['tool_source_commit']}")
A(f"expected footprint    {man['report']['expected_footprint_sha256'][:32]}…")
A(f"바뀐 셀               {len(fp)}")
A("```")
A("")
A("원본 `{prefill,decode}.{csv,jsonl}` 은 **건드리지 않았다.** 이 디렉터리는 원본 + "
  "`develop/plus_at/overlay-*.yaml` 로 언제든 재생성된다. 손으로 고치지 말 것.")
A("")

A("## 무엇을 무엇으로 바꿨나")
A("")
A("| sub_id | before | after | 셀 | 자리 |")
A("|---|---|---|---:|---|")
per = collections.defaultdict(lambda: collections.Counter())
where = collections.defaultdict(set)
for r in fp:
    per[r["sub_id"]][(r["before"], r["after"])] += 1
    where[r["sub_id"]].add((re.sub(r"layers\.\d+", "layers.N", r["module_path"]),
                            r["op_type"]))
for sid in sorted(per):
    for (b, a), n in sorted(per[sid].items()):
        w = sorted(where[sid])
        loc = w[0][0].split(".")[-1] if len(w) == 1 else f"{len(w)} 개 (module, op)"
        A(f"| `{sid}` | `{b}` | `{a}` | {n} | {loc} |")
A("")

A("## phase / field 별")
A("")
A("```")
cnt = collections.Counter((r["sub_id"], r["phase"], r["field"]) for r in fp)
for k in sorted(cnt):
    A(f"{k[0]:<18}{k[1]:<9}{k[2]:<14}{cnt[k]:>6}")
A(f"{'합계':<41}{len(fp):>6}")
A("```")
A("")

A("## 심볼 (symbols.yaml)")
A("")
A("`kind` 가 **아키텍처**와 **트레이스 제어**를 가른다. 트레이스 제어 심볼은 모델 속성이 "
  "아니라 이 트레이스가 만든 값이다.")
A("")
A("| 심볼 | kind | 식 | 이 트레이스의 값 |")
A("|---|---|---|---|")
sym = json.load(io.open(os.path.join(D, "MANIFEST.json"), encoding="utf-8"))
import yaml
sy = yaml.safe_load(io.open(os.path.join(D, "symbols.yaml"), encoding="utf-8"))["symbols"]
for name, d in sy.items():
    e = d.get("expr") or f"prefill {d.get('expr_prefill')} / decode {d.get('expr_decode')}"
    v = d.get("value")
    if v is None:
        v = f"prefill {d.get('value_prefill')} / decode {d.get('value_decode')}"
    A(f"| `{name}` | {d.get('kind')} | `{e}` | {v} |")
A("")

A("## 바꾸지 않고 남긴 맨 정수 (숨기지 않는다)")
A("")
A("```")
for ph, d in (man["report"].get("residual_literals") or {}).items():
    tot = sum(d.values())
    A(f"{ph:<9}" + "  ".join(f"{k}:{v}" for k, v in sorted(d.items(), key=lambda x: int(x[0])))
      + f"     {tot} 자리")
A("```")
A("")
A("전부 attention-residual 누적 경로다(계열 C). 배치 크기만 스윕하면 이 값은 변하지 "
  "않는다. `d_chunk` 나 층 배치를 스윕하려면 이 자리는 아직 맞지 않는다.")
A("")

A("## V9 게이트")
A("")
A("| 결과 | 처분 | 게이트 | 내용 |")
A("|---|---|---|---|")
for gid in sorted(man["v9"]):
    r = man["v9"][gid]
    A(f"| {r['result']} | {r['decision']} | `{gid}` | {r['detail'][:90]} |")
A("")

A("## 소비자 계약")
A("")
A(man["bundle_contract"])
A("")
if man.get("release_blockers"):
    A("## 왜 provisional 인가")
    A("")
    A("```")
    for b in man["release_blockers"]:
        A(b)
    A("```")

io.open(os.path.join(D, "DIFF.md"), "w", encoding="utf-8",
        newline=chr(10)).write("\n".join(L) + "\n")
print(f"wrote {D}/DIFF.md  ({len(L)} 줄)")
