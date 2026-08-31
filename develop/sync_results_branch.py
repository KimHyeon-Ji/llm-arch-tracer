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
import io
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


def _confident_models() -> list:
    """판단 필요 0건 + 검토 기록 있음, 둘 다인 모델 이름 목록 (정렬됨)."""
    ledger = {}
    if os.path.exists(LEDGER):
        ledger = (yaml.safe_load(io.open(LEDGER, encoding="utf-8")) or {}).get("models") or {}
    out = []
    for name in sorted(os.listdir(MODELS)):
        req = os.path.join(MODELS, name, "review_request.md")
        if not os.path.isfile(req):
            continue
        text = io.open(req, encoding="utf-8").read()
        m = re.search(r"판단 필요: \*\*(\d+)건\*\*", text)
        if not m or int(m.group(1)) != 0:
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
    full_dir = os.path.join(target, "full")
    if os.path.isdir(full_dir):
        shutil.rmtree(full_dir)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", default=os.path.join(PROJ, "..", "llm-arch-tracer-results"),
                     help="results 브랜치를 체크아웃해 둔 워크트리 경로")
    ap.add_argument("--ref", default="HEAD", help="스냅샷을 뜰 git ref (기본 HEAD)")
    ap.add_argument("--dry-run", action="store_true", help="뭘 할지만 보여주고 아무것도 안 함")
    ap.add_argument("--no-commit", action="store_true", help="파일만 갱신하고 커밋은 안 함")
    a = ap.parse_args()

    dest = os.path.abspath(a.dest)
    # worktree 의 .git 은 디렉터리가 아니라 gitdir 을 가리키는 파일이다 -- exists 로 확인.
    if not os.path.exists(os.path.join(dest, ".git")):
        raise SystemExit(f"{dest} 가 git 워크트리가 아니다 -- 먼저 만들 것 "
                          f"(git worktree add --detach {dest}; cd {dest}; "
                          f"git checkout --orphan results; git rm -rf .)")

    want = _confident_models()
    have = set(os.listdir(os.path.join(dest, "models"))) if \
        os.path.isdir(os.path.join(dest, "models")) else set()
    to_add = [m for m in want if m not in have]
    to_remove = sorted(have - set(want))
    to_refresh = [m for m in want if m in have]   # 내용이 바뀌었을 수 있으니 항상 다시 뜬다

    print(f"판단 필요 0건 + 검토 기록 있음: {len(want)}개 모델")
    if to_add:
        print(f"  새로 추가: {', '.join(to_add)}")
    if to_remove:
        print(f"  더 이상 기준을 못 채워 제거: {', '.join(to_remove)}")
    print(f"  갱신(내용 재확인): {len(to_refresh)}개")

    if a.dry_run:
        return 0

    os.makedirs(os.path.join(dest, "models"), exist_ok=True)
    for m in to_remove:
        shutil.rmtree(os.path.join(dest, "models", m))
    for m in want:
        _archive_model(a.ref, m, dest)

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
