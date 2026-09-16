r"""**이 산출물에서 확실하지 않은 것 전부를 한 파일로 낸다** -- `UNKNOWNS.md`.

왜 있나
-------
표는 확정 라벨과 미확정 라벨이 똑같이 생겼다. 축 판정 원장(`*.axis_resolution.jsonl`)이
자리마다 등급을 적어 함께 나가지만, 그건 수십만 줄짜리 기계용 파일이라 사람이 열어보지
않는다. 받는 쪽이 **먼저 읽는 자리**에 "무엇을 믿어도 되고 무엇을 믿으면 안 되는지"가
있어야 한다.

출고 게이트의 역할도 "모름이 없는가"가 아니라 **"모름이 공개됐는가"** 로 바뀐다. 모름을
없애는 것은 다음 단계의 일이고, 그때까지 결과물이 묶여 있을 이유는 없다 -- 다만 확정본인
척해서는 안 된다.

담기는 것
---------
  1. 축 판정 등급 요약과 **접힌 질문** (후보가 갈리지 않은 자리들)
  2. ③ 자유 평가가 지금 산출물을 봤는가 (낡았으면 낡았다고 적는다)
  3. 손 안 댄 검토 지적
  4. 모델 동작을 대체한 remedy (표의 숫자가 관측값이 아닌 자리)

실행:
    .venv\Scripts\python.exe develop\make_unknowns.py <모델 디렉터리> [...]
"""
import collections
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJ, "src"))
sys.path.insert(0, HERE)

import axis_ledger as AL                                      # noqa: E402
import review_ledger                                          # noqa: E402

OPEN = (AL.SCOPE_INFERRED, AL.HEURISTIC, AL.OPEN_TIE, AL.UNRESOLVED)

# 등급마다 **무엇을 믿어도 되는지가 다르다.** 이것을 안 적으면 "모름 24%" 라는 숫자만 남아
# 전부 똑같이 위험해 보인다. 실제로는 값 충돌(open_tie)은 숫자가 맞고 이름만 미정이며,
# 지어낸 이름(heuristic)만이 틀린 주장을 실을 수 있다.
GRADE_NOTE = {
    AL.SCOPE_INFERRED: "scope 정규식만이 후보를 갈랐다. 근거는 있으나 아무도 검증 안 했다. "
                       "구체 크기는 맞다",
    AL.HEURISTIC: "**산술로 지어낸 이름.** 값은 맞지만 이름이 틀릴 수 있다 -- "
                  "이 목록에서 가장 먼저 봐야 하는 등급이다",
    AL.OPEN_TIE: "후보 둘 이상이 같은 값이라 트레이스만으로 못 갈랐다. "
                 "구체 크기·FLOPs·바이트는 맞고 **이름만** 미정이다",
    AL.UNRESOLVED: "이름 붙일 근거가 없어 정수로 뒀다. 주장을 안 하므로 틀릴 것도 없다",
}


def _read(d, phase):
    """(요약 레코드, 질문 레코드들). 사이드카가 둘 다 이미 적어 둔다 -- 여기서
    다시 접지 않는다."""
    p = os.path.join(d, "full", f"{phase}.axis_resolution.jsonl")
    if not os.path.isfile(p):
        p = os.path.join(d, f"{phase}.axis_resolution.jsonl")   # 이미 출고된 배치
    if not os.path.isfile(p):
        return None, []
    head, qs = None, []
    with io.open(p, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("kind") == "summary":
                head = r
            elif r.get("kind") == "question":
                qs.append(r)
    return head, qs


def render(d: str, name: str) -> str:
    out = [f"# 확실하지 않은 것 -- {name}", ""]
    out.append("이 파일은 **이 산출물에서 검증되지 않은 부분 전부**를 모은 것이다. "
               "여기 없는 것은 규칙이 결정했고 게이트가 확인했다.")
    out.append("")

    # 1) 축 판정
    out.append("## 1. 축 이름 판정")
    out.append("")
    tot = collections.Counter()
    sites = 0
    for phase in ("prefill", "decode"):
        head, _ = _read(d, phase)
        if head is None:
            continue
        sites += int(head.get("sites") or 0)
        for g, n in (head.get("occurrences") or {}).items():
            tot[g] += int(n or 0)
    if sites:
        conf = tot[AL.CONFIRMED]
        seen = sum(tot.values()) or 1
        out.append(f"축 자리 **{seen:,}개** 중 확정 **{conf:,}개 ({conf / seen:.1%})**, "
                   f"미확정 **{seen - conf:,}개 ({1 - conf / seen:.1%})**.")
        out.append("")
        out.append("| 등급 | 자리 | 무엇을 믿어도 되나 |")
        out.append("|---|---:|---|")
        for g in OPEN:
            if tot[g]:
                out.append(f"| `{g}` | {tot[g]:,} | {GRADE_NOTE[g]} |")
        out.append("")

    # 2) 접힌 질문
    out.append("### 아직 안 푼 질문")
    out.append("")
    out.append("같은 `(등급, 후보, 현재 라벨)` 은 질문 하나다 -- 답 하나가 축 수천 개를 "
               "확정시킨다. 답은 `rules/axis_evidence.yaml` 에 인용과 함께 적는다.")
    out.append("")
    # 게이트가 이 파일이 낡았는지 보려면 **기계가 읽을 수 있는 합계 한 줄**이 필요하다.
    # phase 별 숫자만 있으면 합계를 못 찾아 항상 낡은 것으로 잡힌다.
    n_all = sum(len(_read(d, ph)[1]) for ph in ("prefill", "decode"))
    out.append(f"질문 합계 **{n_all}개**.")
    out.append("")
    any_q = False
    for phase in ("prefill", "decode"):
        _head, qs = _read(d, phase)
        if not qs:
            continue
        any_q = True
        out.append(f"**{phase}** -- 질문 {len(qs)}개")
        out.append("")
        out.append("| 축 수 | 등급 | 후보 | 현재 라벨 |")
        out.append("|---:|---|---|---|")
        for q in sorted(qs, key=lambda r: -int(r.get("occurrences") or 0)):
            # `candidates` 는 "a|b" 꼴 문자열이다. 파이프는 표 칸을 깨므로 escape.
            cands = str(q.get("candidates") or "") or "—"
            cands = cands.replace("|", " \\| ")
            out.append(f"| {int(q.get('occurrences') or 0):,} | `{q.get('grade')}` "
                       f"| {cands} | `{q.get('label')}` |")
        out.append("")
    if not any_q:
        out.append("없다 -- 모든 자리가 확정이거나 이미 근거가 등록돼 있다.")
        out.append("")

    # 3) ③ 자유 평가
    out.append("## 2. 자유 평가(③층) 상태")
    out.append("")
    try:
        st, why = review_ledger.status(d, name)
    except Exception as e:                                     # noqa: BLE001
        st, why = "ERROR", str(e)
    if st == "PASS":
        out.append(f"`PASS` -- {why}")
    else:
        out.append(f"**`{st}`** -- {why}")
        out.append("")
        out.append("즉 규칙이 못 잡는 종류의 오류는 **이 판에서 다시 확인되지 않았다.** "
                   "규칙 게이트가 통과했다는 것과는 별개의 이야기다.")
    out.append("")

    # 4) 손 안 댄 지적
    out.append("## 3. 손 안 댄 검토 지적")
    out.append("")
    rf = os.path.join(d, "review_findings.json")
    finds = []
    if os.path.isfile(rf):
        try:
            finds = (json.load(io.open(rf, encoding="utf-8")) or {}).get("findings") or []
        except Exception:                                      # noqa: BLE001
            finds = []
    openf = [f for f in finds if f.get("status") not in ("fixed", "accepted_limit")]
    if not openf:
        out.append(f"없다 (기록된 지적 {len(finds)}건은 전부 처리됨).")
    else:
        out.append(f"**{len(openf)}건** -- 지적은 됐고 아직 안 고쳤다.")
        out.append("")
        out.append("| 축 | 상태 | 내용 |")
        out.append("|---|---|---|")
        for f in openf:
            note = str(f.get("note") or f.get("why") or f.get("claim") or "").replace("|", "\\|")
            out.append(f"| `{str(f.get('axis'))[:40]}` | {f.get('status')} | {note[:160]} |")
    out.append("")

    # 3.2) 알고 받아들인 한계 -- 표가 모델의 어떤 계산을 안 보여주는가
    limits = [f for f in finds if f.get("status") == "accepted_limit"]
    out.append("### 알고 받아들인 한계")
    out.append("")
    if not limits:
        out.append("없다.")
    else:
        out.append(f"**{len(limits)}건.** 고쳐야 할 결함이 아니라 **요약 표의 범위**다 -- 해당 "
                   "계산은 원시 trace(`full/`)에는 있고 major-op 표에서 빠진다. 표의 행 수로 "
                   "연산량을 세면 과소평가된다.")
        out.append("")
        out.append("| 모듈 | 무엇이 빠졌나 | 근거 |")
        out.append("|---|---|---|")
        for f in limits:
            ev = str(f.get("evidence") or "").replace("|", "\\|")
            out.append(f"| `{f.get('module')}` | {f.get('axis')} | {ev[:230]} |")
    out.append("")

    # 3.5) 의뢰서가 사람 판단을 요청한 항목
    out.append("## 4. 의뢰서의 판단 필요 항목")
    out.append("")
    req = os.path.join(d, "review_request.md")
    n_req, heads = 0, []
    if os.path.isfile(req):
        txt = io.open(req, encoding="utf-8").read()
        mt = re.search(r"판단 필요: \*\*(\d+)건\*\*", txt)
        n_req = int(mt.group(1)) if mt else 0
        heads = re.findall(r"^### (.+)$", txt, re.M)
    if not n_req:
        out.append("없다.")
    else:
        out.append(f"판단 필요 **{n_req}건**. 값 충돌이나 관례로 고른 자리라 규칙이 "
                   "결정하지 못했다 -- 위 1절의 접힌 질문과 같은 종류다. 전문은 "
                   "`review_request.md` 에 있다.")
        out.append("")
        for h in heads:
            out.append(f"* {h}")
    out.append("")

    # 4.5) 더 이상 안 맞는 소스 확인 기록
    out.append("## 5. 더 이상 안 맞는 소스 확인 기록")
    out.append("")
    lc = os.path.join(d, "full", "label_confirmed.json")
    dead_conf = []
    if os.path.isfile(lc):
        try:
            dead_conf = [e for e in (json.load(io.open(lc, encoding="utf-8")) or [])
                         if not e.get("matched")]
        except Exception:                                      # noqa: BLE001
            dead_conf = []
    if not dead_conf:
        out.append("없다 -- 기록된 확인이 전부 지금 산출물의 축에 맞는다.")
    else:
        out.append(f"**{len(dead_conf)}건.** `rules/label_confirmed.yaml` 이 소스를 보고 "
                   "\"이 이름이 맞다\"고 적어 둔 자리인데, 그 앵커가 지금 트레이스에 안 맞는다. "
                   "대부분 발행 배치가 B=1 이 아니게 되면서 접힌 배치 축이 생겨(`n_h` -> "
                   "`B*n_h`) 앵커가 낡은 것이다. **그 축들이 틀렸다는 뜻이 아니라, 지금 판에서 "
                   "소스로 확인된 상태가 아니라는 뜻이다.**")
        out.append("")
        out.append("| 모듈 | 확인한 이름 | 기대값 | 낡은 shape 앵커 |")
        out.append("|---|---|---:|---|")
        for e in dead_conf:
            try:
                anc = json.loads(e["id"])[7]
            except Exception:                                  # noqa: BLE001
                anc = None
            out.append(f"| `{e.get('module')}` | `{e.get('label')}` | {e.get('expect')} | "
                       f"`{anc}` |")
    out.append("")

    # 5) 대체된 동작
    out.append("## 6. 표의 숫자가 관측값이 아닌 자리")
    out.append("")
    prov = os.path.join(d, "full", "provenance.json")
    log = []
    if os.path.isfile(prov):
        try:
            log = (json.load(io.open(prov, encoding="utf-8")) or {}).get("adaptation_log") or []
        except Exception:                                      # noqa: BLE001
            log = []
    subs = [e for e in log if isinstance(e, dict) and e.get("caveat")]
    if not subs:
        out.append("없다 -- 모델 동작을 대체한 remedy 가 없다.")
    else:
        out.append("해당 행에는 `caveat` 열이 차 있다. 그 열로 걸러 보면 된다.")
        out.append("")
        for e in subs:
            out.append(f"* **`{e.get('remedy')}`** -- {e.get('caveat')}")
    out.append("")
    return "\n".join(out)


def write(d: str) -> str:
    name = os.path.basename(os.path.normpath(d))
    p = os.path.join(d, "UNKNOWNS.md")
    io.open(p, "w", encoding="utf-8", newline="\n").write(render(d, name))
    return p


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    for arg in sys.argv[1:]:
        print("wrote", write(arg))
