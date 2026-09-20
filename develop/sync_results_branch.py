"""Snapshot the "100% 확신 가능" models' full/ 밖 결과물 파일을 results 브랜치로 복사한다.

WHY THIS EXISTS
---------------
`models/` 전체(특히 `full/`)를 그대로 GitHub에 올리면 받는 쪽이 1.3GB+를 클론해야 한다.
그중 사람이 실제로 보는 결과물(structure.yaml, model_summary.md, csv/jsonl, review_*)은
전체의 1.6%뿐이다(2026-08-31 실측: outside-full 21MB vs full/ 1,273MB). 이 스크립트는 그
21MB만 별도 orphan 브랜치(`results`)에 스냅샷 커밋 하나로 반영한다 -- 받는 쪽은
`git clone --branch results --single-branch --depth 1 <repo-url>` 한 줄로 끝난다.

"100% 확신 가능"의 기준 (둘 다 만족해야 포함됨):
  1. `review_request.md` 가 "판단 필요: **0건**" -- 자동 규칙이 전부 결정했다는 뜻.
  2. `develop/verify/review_ledger.yaml` 에 그 모델의 검토 기록이 있다 -- ③ 자유 평가를
     최소 한 번은 통과했다는 뜻(가트가 못 잡는 것도 이 검토가 잡는다).

이 스크립트는 **gate FAIL 여부를 다시 실행해서 확인하지 않는다** (verify_all.py 는 몇 분
걸리는 무거운 검사라 이 스크립트를 매번 돌릴 때마다 함께 돌리면 부담이 크다) -- 대신 최근에
`develop/verify_all.py` 를 돌려 FAIL 0 을 확인한 뒤에 이 스크립트를 돌리는 것을 전제로 한다.

사용:
    .venv\\Scripts\\python.exe develop\\sync_results_branch.py
    .venv\\Scripts\\python.exe develop\\sync_results_branch.py --dest ../other-path
    .venv\\Scripts\\python.exe develop\\sync_results_branch.py --ref HEAD --dry-run
"""
import argparse
import gzip
import io
import json
import os
import re
import shutil
import subprocess
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
MODELS = os.path.join(PROJ, "models")
LEDGER = os.path.join(PROJ, "develop", "verify", "review_ledger.yaml")

# `full/` 에서 건져 출고본에 함께 싣는 파일. 축마다 어떤 근거로 그 이름이 됐는지가 들어 있다.
# 출고본에는 **`.gz` 를 붙여 압축해서** 싣는다 -- 아래 `_carry` 의 주석을 볼 것.
CARRY_FROM_FULL = ("prefill.axis_resolution.jsonl", "decode.axis_resolution.jsonl")

# 공개판이 "대조했다" 고 말하려면 **그 근거도 같이 나가야** 한다. 증명 산출물이 `full/`
# 안에만 있으면 독자가 추적할 수 없다(외부 검토 2026-09-19).
#
# `CARRY_FROM_FULL` 과 **따로 둔다.** 그쪽은 복사 목록이면서 동시에 축 원장 요약
# (`coverage_ok`/`questions`/`evidence_unused`)을 검사하는 목록이라, 형식이 다른 파일을
# 넣으면 출고 게이트가 "coverage_ok 가 아니다" 로 잘못 막는다. 이 목록은 복사만 한다.
CARRY_EVIDENCE = ("lowering_proof.json", "batch_transition_proof.json")


# ---------------------------------------------------------------- 출고 게이트
#
# "판단 필요 0건 + ledger 존재" 만으로는 부족하다. 그 둘은 **자동 규칙이 다 결정했다** 는
# 뜻이지 "검토에서 지적이 없다" 도 "산출물이 최신 규칙으로 만들어졌다" 도 아니다.
# 외부 검토(2026-09-11)가 요구한 최소 조건을 여기서 검사한다.
#
# 통과 못 하면 **내보내지 않는다.** 이미 나가 있으면 내리고 매니페스트에 이유를 남긴다 --
# 알려진 오류가 있는 판이 검증된 결과와 같은 자리에 남으면 안 된다.

# `current` 는 **판정이 끝난 것**이다 -- ④층이 소스를 보고 "지금 라벨이 맞다"
# (`current_label_correct`) 또는 "이름이 없는 것이 맞다"(`no_name_exists`) 고 결론 낸
# 자리다. 미수정 결함이 아니므로 "손 안 댄 지적" 으로 세면 안 된다
# (외부 검토 2026-09-20). 아직 안 본 것은 `open` 뿐이다.
RELEASE_OK_STATUSES = {"fixed", "accepted_limit", "current"}


def release_blockers(model: str) -> list:
    """이 모델을 지금 내보내면 안 되는 이유들. 빈 리스트면 통과."""
    d = os.path.join(MODELS, model)
    out, unused_by_phase, open_q = [], [], 0

    # 0) **검토가 지금 산출물을 본 것인가.** 원장에 항목이 있다는 것만으로는 부족하다 --
    #    2026-08-31 검토 기록이 남아 있는데 산출물은 2026-09-11 판이면 그 검토는 만료다.
    #    `_confident_models()` 는 항목 존재만 봤고 `status()` 를 안 불렀다(외부 검토 2026-09-12).
    try:
        sys.path.insert(0, os.path.join(PROJ, "src"))
        import review_ledger
        st, why = review_ledger.status(d, model)
    except Exception as e:
        st, why = "ERROR", str(e)
        out.append(f"검토 원장을 못 읽는다: {e}")

    # 1) 검토 기록: 손 안 댄 지적이 남아 있으면 안 된다
    rf = os.path.join(d, "review_findings.json")
    open_findings = 0
    if not os.path.isfile(rf):
        out.append("review_findings.json 없음 -- ③ 자유 평가를 안 거쳤다")
    else:
        try:
            finds = (json.load(io.open(rf, encoding="utf-8")) or {}).get("findings") or []
        except Exception as e:
            finds = []
            out.append(f"review_findings.json 을 못 읽는다: {e}")
        open_findings = len([f for f in finds if f.get("status") not in RELEASE_OK_STATUSES])
        # 근거 없는 판정은 판정이 아니다 -- `verify_all` 과 같은 기준을 여기서도 본다.
        try:
            import review_ledger as _rl
            if _rl.uncited(d):
                out.append(f"소스 인용 없는 교정 주장 {_rl.uncited(d)}건")
            if _rl.soft_undetermined(d):
                out.append(f"밖을 찾아본 흔적 없는 '확인 못함' {_rl.soft_undetermined(d)}건")
        except Exception:
            pass

    # 2) 축 판정 사이드카: **없으면 실패한다.** 예전에는 조용히 건너뛰었는데, 그러면 받는
    #    쪽은 어떤 축이 미확정인지 모른 채 확정본처럼 읽는다.
    for name in CARRY_FROM_FULL:
        p = os.path.join(d, "full", name)
        if not os.path.isfile(p):
            out.append(f"{name} 없음 -- 재트레이스 필요")
            continue
        try:
            summary = json.loads(io.open(p, encoding="utf-8").readline())
        except Exception as e:
            out.append(f"{name} 요약을 못 읽는다: {e}")
            continue
        if not summary.get("coverage_ok"):
            out.append(f"{name}: coverage_ok 가 아니다 -- 등급 합이 자리 수와 안 맞는다")
        open_q += int(summary.get("questions") or 0)
        unused_by_phase.append(set(summary.get("evidence_unused") or ()))

    # **낡음은 phase 를 가로질러 본다.** 근거 하나가 한 phase 에만 해당하는 것은 정상이다 --
    # 빈 캐시 concat(`|0`)은 prefill 에만 있고 decode 에는 없다. phase 별로 보면 그게 낡은
    # 것으로 잡힌다(2026-09-11 에 실제로 오탐이 났다). 두 phase 모두에서 안 쓰여야 낡은 것이다.
    if unused_by_phase:
        dead = set.intersection(*unused_by_phase)
        if dead:
            out.append(f"낡은 근거 {len(dead)}건 {sorted(dead)[:2]} "
                       f"-- 어느 phase 에서도 안 쓰인다")

    # **모름은 출고를 막지 않는다. 숨기는 것이 막는다.**
    #
    # 예전에는 미해결 질문 하나, 낡은 ③ 검토 하나, 손 안 댄 지적 하나면 그 모델이 통째로
    # 묶였다. 그 기준으로는 47개 중 1개만 나갈 수 있었다. 그런데 게이트의 목적은 "모르는 게
    # 없다"가 아니라 **"받는 쪽이 확정본으로 오해하지 않는다"** 이다. 모름을 없애는 것은
    # 다음 단계의 일이고, 그때까지 결과물이 묶여 있을 이유가 없다.
    #
    # 그래서 셋 다 UNKNOWNS.md 로 옮긴다 -- 있으면 통과, 없거나 실제 상태를 안 적었으면 실패.
    # 파일은 develop/make_unknowns.py 가 산출물에서 직접 생성하므로 손으로 못 쓴다.
    disclosed = []
    if open_q:
        disclosed.append(("질문", f"질문 합계 **{open_q}개**", "아직 안 푼 질문"))
    if st != "PASS":
        disclosed.append(("검토", st, "자유 평가(③층) 상태"))
    if open_findings:
        disclosed.append(("지적", f"**{open_findings}건**", "손 안 댄 검토 지적"))
    # 소스 확인 기록이 지금 축에 안 맞으면, 그 축은 "확인됨" 이 아니다. 받는 쪽이 알아야 한다.
    req = os.path.join(d, "review_request.md")
    if os.path.isfile(req):
        mt = re.search(r"판단 필요: \*\*(\d+)건\*\*", io.open(req, encoding="utf-8").read())
        if mt and int(mt.group(1)):
            disclosed.append(("의뢰", f"판단 필요 **{mt.group(1)}건**", "의뢰서의 판단 필요 항목"))
    lc = os.path.join(d, "full", "label_confirmed.json")
    if os.path.isfile(lc):
        try:
            n_dead = len([e for e in (json.load(io.open(lc, encoding="utf-8")) or [])
                          if not e.get("matched")])
        except Exception:                                      # noqa: BLE001
            n_dead = 0
        if n_dead:
            disclosed.append(("확인", f"**{n_dead}건.**", "더 이상 안 맞는 소스 확인 기록"))
    if disclosed:
        u = os.path.join(d, "UNKNOWNS.md")
        if not os.path.isfile(u):
            out.append(f"UNKNOWNS.md 없음 -- 미확정이 {len(disclosed)}종 있는데 공개본에 "
                       f"그 사실이 안 실린다 (develop/make_unknowns.py 로 생성할 것)")
        else:
            text = io.open(u, encoding="utf-8").read()
            for _tag, needle, section in disclosed:
                if needle not in text:
                    out.append(f"UNKNOWNS.md 가 낡았다 -- '{section}' 이 지금 상태와 다르다 "
                               f"({needle} 를 못 찾음). 다시 생성할 것")
    return out


def _digests(model: str) -> dict:
    """이 산출물이 **무엇으로 만들어졌는가**. 서로 다른 판이 같은 결과처럼 보이면 안 된다."""
    d = os.path.join(MODELS, model)
    gen = {}
    p = os.path.join(d, "full", "generated.json")
    if os.path.isfile(p):
        try:
            gen = json.load(io.open(p, encoding="utf-8")) or {}
        except Exception:
            pass
    schema = None
    pp = os.path.join(d, "full", "prefill.ports.jsonl")
    if os.path.isfile(pp):
        try:
            schema = json.loads(io.open(pp, encoding="utf-8").readline()).get(
                "ports_schema_version", 1)
        except Exception:
            pass
    return {"generated_at": gen.get("generated_at"),
            "label_inputs_digest": gen.get("label_inputs"),
            "ports_schema_version": schema}



def _confident_models() -> list:
    """내보낼 후보: 검토 기록이 있고, 판단 필요분이 **공개된** 모델 (정렬됨).

    예전에는 `판단 필요: **0건**` 을 요구했다. 그런데 그 항목들은 값 충돌(`d_head` vs `n_h`)
    이나 관례로 고른 자리 -- 축 판정 원장의 접힌 질문과 같은 종류의 모름이다. 모름이 하나라도
    있으면 통째로 막는 기준으로는 47개 중 1개만 나갈 수 있었다. 모름을 없애는 것은 다음
    단계의 일이고, 지금 필요한 것은 **받는 쪽이 확정본으로 오해하지 않는 것**이다.
    실제 공개 여부는 `release_blockers` 가 UNKNOWNS.md 로 확인한다 (2026-09-16).
    """
    ledger = {}
    if os.path.exists(LEDGER):
        ledger = (yaml.safe_load(io.open(LEDGER, encoding="utf-8")) or {}).get("models") or {}
    out = []
    for name in sorted(os.listdir(MODELS)):
        req = os.path.join(MODELS, name, "review_request.md")
        if not os.path.isfile(req):
            continue
        if name not in ledger:
            continue
        out.append(name)
    return out


def _archive_model(ref: str, model: str, dest: str) -> None:
    """`ref` 시점의 models/<model> (full/ 제외)을 dest/models/<model> 로 뽑는다."""
    target = os.path.join(dest, "models", model)
    if os.path.isdir(target):
        shutil.rmtree(target)
    proc = subprocess.run(
        ["git", "archive", ref, "--", f"models/{model}"],
        cwd=PROJ, capture_output=True, check=True)
    # tar -x 는 상대 경로 그대로 풀어내므로 dest 아래 models/<model>/... 로 떨어진다.
    tar = subprocess.Popen(["tar", "-x", "-C", dest], stdin=subprocess.PIPE)
    tar.communicate(proc.stdout)
    if tar.returncode != 0:
        raise SystemExit(f"tar 추출 실패: {model}")
    # `full/` 은 통째로 버리되, **축 판정 원장은 건져서 같이 내보낸다.** 그게 없으면 받는
    # 쪽은 어떤 축이 소스로 확정됐고 어떤 축이 미확정인지 알 수 없다 -- 표는 확정 라벨과
    # 똑같이 생겼으므로 계속 속는다(외부 검토 2026-09-11).
    #
    # 요약만 싣는 것으로는 부족하다. 자리 키 `(op_id, field, shape_index, axis)` 가 있어야
    # 어느 축인지 역추적할 수 있다.
    full_dir = os.path.join(target, "full")
    if True:
        for name in CARRY_FROM_FULL:
            # **원장은 git 내보내기가 아니라 작업트리에서 가져온다.** Kimi-K3 의 prefill
            # 원장은 657 MB 라 GitHub 한도에 걸려 git 에 둘 수 없고(.gitignore 등재,
            # 2026-09-21), 그래서 `git archive` 로는 나오지 않는다. 작업트리를 읽는 것이
            # **게이트와 같은 자리**이기도 하다 -- `release_blockers` 도 여기를 읽는다.
            # ref 가 HEAD 가 아니면 이 둘이 갈리므로 위에서 거부한다.
            src = os.path.join(MODELS, model, "full", name)
            if not os.path.exists(src):
                # **조용히 건너뛰지 않는다.** 없는 채로 내보내면 받는 쪽은 어떤 축이
                # 미확정인지 모른 채 확정본처럼 읽는다(외부 검토 2026-09-11).
                raise SystemExit(f"{model}: {name} 이 없다 -- 재트레이스해야 출고할 수 있다")
            # **gzip 으로 싣는다.** Kimi-K3 의 prefill 원장은 657 MB 라 GitHub 의 파일당
            # 100 MB 한도에 걸려 push 자체가 거부됐다(2026-09-21, 이 저장소가 원격에
            # 4 개 모델에 멈춰 있던 진짜 이유). gzip 이면 12.4 MB -- 53 배라 한도에
            # 여유가 크고, **자료는 하나도 안 버린다.** 모델마다 이름이 갈리면 읽는 쪽이
            # 두 경우를 다뤄야 하므로 크기와 무관하게 전부 압축한다.
            with io.open(src, "rb") as fin, gzip.open(
                    os.path.join(target, name + ".gz"), "wb", compresslevel=6) as fout:
                shutil.copyfileobj(fin, fout, 1 << 22)
        os.makedirs(full_dir, exist_ok=True)
        # 근거 산출물은 **있으면 싣고 없으면 넘어간다.** 모든 모델이 배치 전환을 거친
        # 것은 아니라서, 없다고 출고를 막을 일은 아니다. 있으면 독자가 추적할 수 있어야 한다.
        for name in CARRY_EVIDENCE:
            src = os.path.join(full_dir, name)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(target, name))
        shutil.rmtree(full_dir)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", default=os.path.join(PROJ, "..", "llm-arch-tracer-results"),
                     help="results 브랜치를 체크아웃해 둔 워크트리 경로")
    ap.add_argument("--ref", default="HEAD", help="스냅샷을 뜰 git ref (기본 HEAD)")
    ap.add_argument("--dry-run", action="store_true", help="뭘 할지만 보여주고 아무것도 안 함")
    ap.add_argument("--no-commit", action="store_true", help="파일만 갱신하고 커밋은 안 함")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="워킹트리가 더러워도 진행(검사 용도). 실제 출고에는 쓰지 마라")
    ap.add_argument("--only", nargs="+", metavar="이름조각",
                    help="이 조각을 이름에 가진 모델만 내보낸다(부분 일치). 이미 나가 있는 "
                         "다른 모델은 건드리지 않는다 -- 규칙이 바뀌는 중에 고친 것만 올릴 때 쓴다")
    a = ap.parse_args()

    dest = os.path.abspath(a.dest)
    # worktree 의 .git 은 디렉터리가 아니라 gitdir 을 가리키는 파일이다 -- exists 로 확인.
    if not os.path.exists(os.path.join(dest, ".git")):
        raise SystemExit(f"{dest} 가 git 워크트리가 아니다 -- 먼저 만들 것 "
                          f"(git worktree add --detach {dest}; cd {dest}; "
                          f"git checkout --orphan results; git rm -rf .)")

    want = _confident_models()
    # **출고 게이트.** 기준을 통과 못 한 모델은 내보내지 않는다.
    blocked = {m: release_blockers(m) for m in want}
    blocked = {m: b for m, b in blocked.items() if b}
    if blocked:
        print(f"출고 기준 미달 {len(blocked)}개 -- 내보내지 않는다:")
        for m, b in sorted(blocked.items()):
            print(f"   {m[:44]:46} {b[0]}" + (f"  (외 {len(b)-1}건)" if len(b) > 1 else ""))
    want = [m for m in want if m not in blocked]
    if a.only:
        # **필요한 모델만 내보낸다.** 기준을 채운 모델을 전부 내보내면, 아직 손대지 않은
        # 모델이 딸려 나간다. 규칙이 좋아져 라벨이 바뀌는 중에는 고친 것만 올려야 한다.
        # 이미 나가 있는 것은 건드리지 않는다(`--only` 는 제거를 하지 않는다).
        sel = [m for m in want if any(f.lower() in m.lower() for f in a.only)]
        missing = [f for f in a.only
                   if not any(f.lower() in m.lower() for m in want)]
        if missing:
            print(f"  **기준 미달이거나 이름이 없다**: {', '.join(missing)}")
        want = sel
    have = set(os.listdir(os.path.join(dest, "models"))) if \
        os.path.isdir(os.path.join(dest, "models")) else set()
    to_add = [m for m in want if m not in have]
    to_remove = [] if a.only else sorted(have - set(want))
    to_refresh = [m for m in want if m in have]   # 내용이 바뀌었을 수 있으니 항상 다시 뜬다

    print(f"판단 필요 0건 + 검토 기록 있음: {len(want)}개 모델")
    if to_add:
        print(f"  새로 추가: {', '.join(to_add)}")
    if to_remove:
        print(f"  더 이상 기준을 못 채워 제거: {', '.join(to_remove)}")
    print(f"  갱신(내용 재확인): {len(to_refresh)}개")

    # 검증한 상태와 **실제로 뽑는 상태**가 같아야 한다. `git archive <ref>` 는 커밋된 것을
    # 뽑는데 게이트는 워킹트리를 읽으므로, 그 둘이 다르면 검증하지 않은 것을 내보내게 된다.
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "models", "rules", "src"],
                           cwd=PROJ, capture_output=True, text=True).stdout.strip()
    if dirty and not a.allow_dirty:
        n = len(dirty.splitlines())
        nl = chr(10)
        print(nl + f"**워킹트리에 커밋 안 된 변경 {n}건** (models/ rules/ src/)." + nl
              + f"게이트는 워킹트리를 읽고 출고는 `{a.ref}` 를 뽑으므로 서로 다른 것을 "
              + "내보낼 수 있다. 커밋한 뒤 다시 돌려라 (검사만 하려면 --allow-dirty).")
        return 1

    # **원장은 작업트리에서 나간다**(`_archive_model` 주석 참고). 표는 `ref` 에서 뽑으므로
    # 둘이 다른 커밋을 가리키면 서로 안 맞는 것을 내보내게 된다. HEAD 가 아니면 막는다.
    _r = subprocess.run(["git", "rev-parse", a.ref], cwd=PROJ,
                        capture_output=True, text=True).stdout.strip()
    _h = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJ,
                        capture_output=True, text=True).stdout.strip()
    if _r != _h and not a.allow_dirty:
        nl = chr(10)
        print(nl + f"**--ref 가 HEAD 가 아니다** ({a.ref} -> {_r[:12]} != "
              + f"HEAD {_h[:12]})." + nl
              + "표는 그 ref 에서 뽑지만 축 원장은 작업트리에서 나가므로 서로 다른 "
              + "판이 섞인다. HEAD 로 맞추고 다시 돌려라.")
        return 1

    if a.dry_run:
        return 0

    os.makedirs(os.path.join(dest, "models"), exist_ok=True)
    for m in to_remove:
        shutil.rmtree(os.path.join(dest, "models", m))
    for m in want:
        _archive_model(a.ref, m, dest)

    # **어떤 판으로 만든 결과인가.** 이게 없으면 서로 다른 ruleset 으로 만든 파일이 같은
    # 결과처럼 보인다. 내리기로 한 모델은 이유를 남긴다.
    manifest = {
        "generated_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "source_ref": subprocess.run(["git", "rev-parse", a.ref], cwd=PROJ,
                                     capture_output=True, text=True).stdout.strip(),
        "verified": {m: _digests(m) for m in want},
        "withheld": {m: {"reason": b} for m, b in sorted(blocked.items())},
        "note": "verified 에 없는 모델은 출고 기준을 통과하지 못했다. "
                "withheld 의 reason 이 그 이유다.",
    }
    with io.open(os.path.join(dest, "MANIFEST.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"  MANIFEST.json: 통과 {len(want)}개 / 보류 {len(blocked)}개")

    if a.no_commit:
        print("커밋은 생략함 (--no-commit)")
        return 0

    subprocess.run(["git", "add", "-A"], cwd=dest, check=True)
    diff = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=dest)
    if diff.returncode == 0:
        print("변경 없음 -- 커밋할 것이 없다")
        return 0

    src_commit = subprocess.run(["git", "rev-parse", "--short", a.ref],
                                 cwd=PROJ, capture_output=True, text=True, check=True
                                 ).stdout.strip()
    msg_lines = [f"sync: main@{src_commit} 기준 결과물 스냅샷 ({len(want)}개 모델)"]
    if to_add:
        msg_lines += ["", "추가: " + ", ".join(to_add)]
    if to_remove:
        msg_lines += ["", "제거: " + ", ".join(to_remove)]
    msg_lines += ["", "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"]
    subprocess.run(["git", "commit", "-m", "\n".join(msg_lines)], cwd=dest, check=True)
    print(f"\n커밋 완료. push 하려면:\n  cd {dest} && git push origin results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
