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
  5. 배치를 키우며 달라진 lowering 과 그 구간 동치 증명이 덮은 범위

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
    # **`current` 는 미수정이 아니다.** 판정이 내려졌고 "지금 라벨이 맞다" 또는 "이름이
    # 없는 게 맞다" 는 기록이다. 그걸 `open` 과 한 묶음으로 "아직 안 고쳤다" 고 세면, 확인이
    # 끝난 자리를 결함으로 발표하게 된다(외부 검토 2026-09-20).
    openf = [f for f in finds if f.get("status") == "open"]
    settled = [f for f in finds
               if f.get("status") not in ("fixed", "accepted_limit", "open")]
    if not openf and not settled:
        out.append(f"없다 (기록된 지적 {len(finds)}건은 전부 처리됨).")
    else:
        out.append(f"아직 안 본 것 **{len(openf)}건**.")
        out.append("")
        if openf:
            out.append("| 축 | 판정 | 내용 |")
            out.append("|---|---|---|")
            for f in openf:
                note = str(f.get("note") or f.get("why") or f.get("claim") or "")
                note = note.replace("|", "\\|")
                out.append(f"| `{str(f.get('axis'))[:40]}` | "
                           f"{f.get('verdict') or '-'} | {note[:160]} |")
            out.append("")
        if settled:
            out.append(f"아래 **{len(settled)}건**은 **판정이 끝난 것**이다 -- 소스를 보고 "
                       "\"지금 라벨이 맞다\" 또는 \"이름이 없는 것이 맞다\" 고 결론 낸 "
                       "자리다. 미수정 결함이 아니다.")
            out.append("")
            out.append("| 축 | 판정 | 상태 |")
            out.append("|---|---|---|")
            for f in settled:
                out.append(f"| `{str(f.get('axis'))[:40]}` | {f.get('verdict') or '-'} | "
                           f"{f.get('status')} |")
    out.append("")

    # 3.2) 알고 받아들인 한계 -- 표가 모델의 어떤 계산을 안 보여주는가
    limits = [f for f in finds if f.get("status") == "accepted_limit"]
    out.append("### 알고 받아들인 한계")
    out.append("")
    if not limits:
        out.append("없다.")
    else:
        # **여기 있는 것이 전부 "표에서 빠진 계산" 은 아니다.** 판정을 그대로 보여 준다 --
        # 배치에 따라 다른 ATen 분해로 내려간 자리를 "계산이 빠졌다" 고 쓰면 사실이 아니다
        # (외부 검토 2026-09-20).
        out.append(f"**{len(limits)}건.** 고쳐야 할 결함이 아니라 **알고 받아들인 것**이다. "
                   "판정별로 뜻이 다르니 `판정` 열을 함께 보라 -- `table_omits_computation` "
                   "은 원시 trace(`full/`)에는 있고 요약 표에서 빠진 계산이고, "
                   "`different_lowering_verified` 는 같은 계산이 다른 ATen 분해로 기록된 "
                   "것이라 빠진 계산이 아니다.")
        out.append("")
        out.append("| 모듈 | 무엇을 받아들였나 | 판정 | 근거 |")
        out.append("|---|---|---|---|")
        for f in limits:
            ev = str(f.get("evidence") or "").replace("|", "\\|")
            out.append(f"| `{f.get('module')}` | {f.get('axis')} | {f.get('verdict') or '-'} | {ev[:230]} |")
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
        # **원인을 뭉뚱그려 주장하지 않는다.** 예전에는 "대부분 배치 접기 때문" 이라고
        # 적었는데, DeepSeek-V4-Pro 의 27 건 중 그 유형은 **6 건**뿐이었다 -- 나머지는
        # 압축 토큰 축의 옛 이름, broadcast singleton, slice/concat 포트와 서수 변경이라
        # 일괄 치환하면 안 되는 것들이었다(외부 검토 2026-09-21). 세어서 적는다.
        _bat = 0
        for _e in dead_conf:
            try:
                _a = json.loads(_e["id"])[7]
            except Exception:                                  # noqa: BLE001
                _a = None
            if isinstance(_a, list) and _a and str(_a[0]) in ("n_h", "n_kv", "n_h_kda"):
                _bat += 1
        _rest = len(dead_conf) - _bat
        out.append(f"**{len(dead_conf)}건.** `rules/label_confirmed.yaml` 이 소스를 보고 "
                   "\"이 이름이 맞다\"고 적어 둔 자리인데, 그 앵커가 지금 트레이스에 안 맞는다. "
                   "**그 축들이 틀렸다는 뜻이 아니라, 지금 판에서 소스로 확인된 상태가 "
                   "아니라는 뜻이다.**")
        out.append("")
        out.append(f"이 중 **{_bat}건**은 선행축이 맨 head 축(`n_h` 계열)인 앵커라, 발행 배치가 "
                   f"B=1 이 아니게 되면서 접힌 축(`B*n_h`)이 생겨 안 맞게 된 것이다. "
                   + (f"나머지 **{_rest}건**의 원인은 자리마다 다르므로 **일괄 치환하면 안 된다** "
                      "-- 하나씩 지금 자리를 찾아 대조해야 한다."
                      if _rest else "이 모델에서는 그 밖의 유형이 없다."))
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

    # 6) 옛 판과의 전환 -- 구간 동치 증명이 무엇을 말하고 무엇을 말하지 않나
    pj = os.path.join(d, "full", "lowering_proof.json")
    if os.path.isfile(pj):
        out.append("## 7. 배치를 키우며 달라진 lowering")
        out.append("")
        try:
            pr = json.load(io.open(pj, encoding="utf-8"))
        except Exception:                                      # noqa: BLE001
            pr = None
        if not pr:
            out.append("증명 파일을 읽지 못했다.")
        else:
            tot = sum(v["records_total"] for v in pr["phases"].values())
            bad = sum(v["records_failed_template"] + sum(v["uncovered"].values())
                      for v in pr["phases"].values())
            # 레코드 0 의 이유를 **추측하지 않는다.** 감사가 내린 판정을 읽는다.
            # 예전에는 "둘 다 같은 배치로 잡았기 때문" 이라고 단정했는데, 라벨만 바뀐
            # 전환에서도 0 이 나온다 -- 그때 그 문장은 틀린 설명이 된다(2026-09-20).
            _claim = None
            _mp = os.path.join(d, "audit_manifest.json")
            if os.path.isfile(_mp):
                try:
                    _claim = ((json.load(io.open(_mp, encoding="utf-8")) or {})
                              .get("lowering_proof") or {}).get("claim")
                except Exception:                              # noqa: BLE001
                    _claim = None
            if tot == 0:
                if _claim == "no_segment_transition":
                    out.append("**직전 발행본과 이 판 사이에는 계산이 달라진 구간이 없다.** "
                               "라벨만 바뀐 전환이라 다시 실행해 대조할 대상이 없었다"
                               "(레코드 0건). **이 전환에 대해 구간 동치는 주장하지 않는다** "
                               "-- 대조를 해서 통과한 것이 아니라, 대조할 것이 없었다.")
                else:
                    out.append("**직전 발행본과 이 판 사이에는** 서명으로 짝이 안 지어지는 "
                               "ATen op 레코드가 없다. 둘 다 같은 배치로 잡았기 때문이다.")
            else:
                out.append(f"직전 발행본과 이 판을 대조하면 서명으로 짝이 안 지어지는 "
                           f"ATen op 레코드가 **{tot:,}건** 있다. 같은 계산이 배치 크기에 "
                           f"따라 다른 연산으로 내려가기 때문이다 -- `einsum` 이 B=1 에서는 "
                           f"피연산자가 교환된 전치 `bmm` 으로, B>1 에서는 교환되지 않은 "
                           f"`bmm` 으로 내려간다.")
            out.append("")
            if tot:
                out.append("짝이 안 지어진 구간은 `develop/lowering_proof.py` 가 **다시 "
                           "실행해서** 대조한다. 모듈의 실제 바깥 경계를 ports 의 "
                           "producer/consumer 로 찾고, 같은 경계 입력을 넣어 **모든 배치 "
                           "조각에서** 같은 경계 출력이 나오는지 본다."
                           + (f" 이 판의 불일치 **{bad:,}건**." if tot else ""))
                out.append("")
            if tot:
                out.append("**이것은 수치 시험이지 증명이 아니다.** 생성한 float64 입력 "
                           "한 벌에 대해 결과가 일치했다는 뜻이고, 모든 입력에 대한 "
                           "대수적 동치를 보인 것이 아니다. 산출물에서는 이 결과를 "
                           "`lowering_replay_consistent` 라고 부른다.")
                out.append("")
            if tot:
                out.append("| phase | 레코드 | 미증명 | template | 검증한 instance |")
                out.append("|---|---:|---:|---:|---|")
                for ph, v in pr["phases"].items():
                    inst = ", ".join(f"{t['proved']}/{t['n']}" for t in v["templates"])
                    out.append(f"| {ph} | {v['records_total']:,} | "
                               f"{v['records_failed_template'] + sum(v['uncovered'].values()):,} | "
                               f"{len(v['templates'])} | {inst} |")
            out.append("")
            if tot:
                out.append("**이 증명이 말하지 않는 것:**")
                out.append("")
                out.append("* 배치 축이 어디인지는 **발행 라벨을 가설로** 삼았다. 라벨이 "
                           "틀렸으면 대조가 깨지므로 이 검사는 라벨의 검사이기도 하지만, "
                           "라벨을 독립적으로 세운 것은 아니다.")
                out.append("* 같은 op 이름 다중집합을 한 template 로 묶는다. DAG 간선과 "
                           "`scalar_args` 까지 같은지는 지문에 들어 있지 않다 -- 다만 이번 "
                           "실행은 모든 instance 를 재실행했으므로 표본 누락은 없다.")
                out.append("* 아무 op 도 소비하지 않는 중간 값은 경계에서 뺐다. 관측할 수 "
                           "없는 값이라 대조 대상이 아니다.")
                out.append("* 트레이스 기록에 dtype 이 없어(`torch.bool` 이 직렬화되지 않는다), "
                           "마스크·색인 자리는 소비 지점에서 맞추고 팩토리 op 의 부동소수점 "
                           "결과는 float64 로 통일했다. 양쪽에 같은 규칙을 쓰므로 대조는 "
                           "성립하지만, 원본의 dtype 자체를 재현한 것은 아니다.")
                out.append("* 부동소수점 bitwise 일치를 요구하지 않는다(rtol=atol=1e-11).")
                out.append("* 추적 범위 밖(실제 GPU kernel 의 op 구성)은 들어 있지 않다.")
                out.append("")
        bp = os.path.join(d, "full", "batch_transition_proof.json")
        if os.path.isfile(bp):
            try:
                b = json.load(io.open(bp, encoding="utf-8"))
            except Exception:                                  # noqa: BLE001
                b = None
            if b:
                bt = sum(v["records_total"] for v in b.get("phases", {}).values())
                bb = sum(v["records_failed_template"] + sum(v["uncovered"].values())
                         for v in b.get("phases", {}).values())
                out.append("### 배치 전환(B=1 -> B=3) 대조")
                out.append("")
                out.append(f"같은 모델을 같은 규칙으로 B=1 과 B=3 두 번 잡아 대조했다. "
                           f"서명으로 짝이 안 지어진 ATen op 레코드 **{bt:,}건**, "
                           f"불일치 **{bb:,}건**. 근거 파일은 `full/{os.path.basename(bp)}`.")
                out.append("")
                out.append("| phase | 레코드 | 불일치 | template | 검증한 instance |")
                out.append("|---|---:|---:|---:|---|")
                for ph, v in b.get("phases", {}).items():
                    inst = ", ".join(f"{t['proved']}/{t['n']}" for t in v["templates"])
                    out.append(f"| {ph} | {v['records_total']:,} | "
                               f"{v['records_failed_template'] + sum(v['uncovered'].values()):,} | "
                               f"{len(v['templates'])} | {inst} |")
                out.append("")
        else:
            # **없는 근거는 조용히 빠지면 안 된다.** 이 절은 예전 발행본에서 B=1 -> B=3
            # 대조를 싣던 자리다. 파일이 없으면 아무것도 안 쓰던 탓에, 근거가 사라진 것과
            # 애초에 없던 것이 독자에게 같아 보였다(2026-09-20).
            out.append("### 배치 전환(B=1 -> B>1) 대조 -- **이 판에는 없다**")
            out.append("")
            out.append("접힌 배치 축(`B`, `B*X`)을 B=1 트레이스와 맞대어 확인한 기록이 "
                       "이 판에는 **실려 있지 않다.** 그 대조는 B=1 로 한 번 더 잡아야 "
                       "만들 수 있는데, 그 뒤로 추적된 계산 자체가 바뀌어(KDA forget "
                       "gate, Q/K 정규화) 옛 기록은 지금 발행하는 계산을 설명하지 "
                       "못한다. 그래서 **옮겨 싣지 않고 없다고 적는다.**")
            out.append("")
            out.append("이 판에서 배치 라벨을 받치는 근거는 **독립 배치 검증** 하나다 -- "
                       "발행한 B 라벨을 다른 배치 크기로 실제로 잡은 shape 과 대조해 "
                       "어긋나는 라벨이 없음을 본다. 두 트레이스를 재실행해 값까지 "
                       "맞춰 보는 대조는 아니다.")
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
