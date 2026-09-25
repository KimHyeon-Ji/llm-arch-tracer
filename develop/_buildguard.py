r"""생성 스크립트의 공통 가드. **더러운 트리에서 만든 산출물을 조용히 내보내지 않는다.**

같은 실수를 두 번 했다(2026-09-25). `built_from_commit` 을 필드로만 적었더니, 커밋 **전에**
생성한 산출물이 "그 커밋에서 만들었다" 고 보고됐다. 외부 검토가 두 번 잡았다:

  1차 제출: 보고서엔 2c88c314 / 실제 파일엔 935284c1
  2차 제출: 보고서엔 029a7fef clean / 실제 파일엔 e7613998 dirty

필드가 아니라 **실패**여야 한다. `require_clean_tree()` 를 생성 스크립트 맨 앞에서 부른다.
"""
import hashlib
import io
import os
import subprocess
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _git(*a):
    return subprocess.run(["git", *a], cwd=PROJ, capture_output=True,
                          text=True).stdout.strip()


def sha256_file(path):
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with io.open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def require_clean_tree(allow_env="ALLOW_DIRTY_BUILD"):
    """`src/` · `develop/` · `rules/` 가 깨끗하지 않으면 **멈춘다.**

    산출물(`models/`)의 변경은 허용한다 -- 생성 대상이기 때문이다. 검사만 하려면
    환경변수로 뚫을 수 있지만, 그 경우 `dirty_build: true` 가 기록된다.
    """
    dirty = [l for l in _git("status", "--porcelain").splitlines()
             if l[3:].startswith(("src/", "develop/", "rules/"))]
    if dirty and not os.environ.get(allow_env):
        print("**커밋 안 된 코드가 있다 -- 산출물을 만들지 않는다.**", file=sys.stderr)
        for l in dirty[:10]:
            print("   " + l, file=sys.stderr)
        print(f"\n커밋한 뒤 다시 돌려라. 검사만 하려면 {allow_env}=1 (그 사실이 기록된다).",
              file=sys.stderr)
        raise SystemExit(2)
    return {"built_from_commit": _git("rev-parse", "HEAD"),
            "dirty_build": bool(dirty),
            "generator_sha256": {}}


def stamp(meta, *paths):
    """생성 스크립트 자신들의 SHA-256 을 기록한다."""
    for p in paths:
        meta["generator_sha256"][os.path.basename(p)] = sha256_file(
            p if os.path.isabs(p) else os.path.join(PROJ, p))
    return meta
