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
import shutil
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


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def input_manifest(paths):
    """**모든 입력 파일의 SHA-256.** 생성 대상만 clean 검사해서는 부족하다.

    `models/` 와 sibling `results-labeled` 의 crosswalk·units 는 이 생성기들에게
    **입력**이다. 입력이 미커밋 상태여도 `dirty_build: false` 가 찍힐 수 있다
    (외부 검토 2026-09-25). 그래서 입력을 해시로 고정한다 -- 재현할 때 이 값을 대조하면
    "같은 입력으로 만들었는가" 를 알 수 있다.
    """
    out = {}
    for p in sorted(set(paths)):
        if os.path.exists(p):
            out[os.path.relpath(p, PROJ).replace(os.sep, "/")] = {
                "sha256": sha256_file(p), "bytes": os.path.getsize(p)}
    return out


def worktree_dirty(path):
    """다른 워크트리(results-labeled 등)가 더러운가."""
    r = subprocess.run(["git", "status", "--porcelain"], cwd=path,
                       capture_output=True, text=True)
    return [l for l in r.stdout.splitlines() if l.strip()]


def _porcelain_path(line):
    """`XY path` 또는 `R  old -> new` 에서 경로를 뽑는다."""
    p = line[3:].strip()
    if " -> " in p:
        p = p.split(" -> ")[-1]
    return p.strip().strip('"')


def input_worktrees(specs):
    """워크트리별 HEAD 와 **입력 경로만의** 미커밋 목록.

    `worktree_dirty()` 를 워크트리 전체에 쓰면 **출력 때문에 항상 더럽다.** 생성물인
    `work/review_bundle/` 이 그 워크트리에 있으니 값 1 은 입력 오염을 뜻하지 않았다
    (외부 검토 2026-09-25). 그래서 **실제 입력 prefix 로 좁힌다.**

    입력 SHA-256 과 역할이 다르다 -- 해시는 "무엇으로 만들었나", 이 검사는 "그 입력이
    커밋돼 있나" 다. 둘 다 남긴다.

    specs: `[(name, worktree_path, (input_prefix, ...)), ...]`
    """
    out = {}
    for name, path, prefixes in specs:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path,
                              capture_output=True, text=True).stdout.strip()
        dirty = [l.strip() for l in worktree_dirty(path)
                 if _porcelain_path(l).startswith(tuple(prefixes))]
        out[name] = {"head": head, "input_prefixes": list(prefixes),
                     "dirty_input_paths": dirty}
    return out


def swap_dir(tmp, dest):
    """`tmp` 를 `dest` 로 바꿔 넣는다. **실패하면 기존 것을 되돌린다.**

    예전에는 `rmtree(dest)` 로 먼저 지우고 `os.replace` 했다. 그 사이에 실패하면 기존
    bundle 이 사라진다 -- "검증 후 교체" 는 맞지만 원자적이 아니었다
    (외부 검토 2026-09-25). 이제 기존 것을 `.bak` 으로 rename 해 두고, 교체가 실패하면
    제자리로 돌려놓는다.
    """
    bak = dest + ".bak"
    if os.path.isdir(bak):
        shutil.rmtree(bak)
    had = os.path.isdir(dest)
    if had:
        os.replace(dest, bak)
    try:
        os.replace(tmp, dest)
    except BaseException:
        if had and not os.path.isdir(dest):
            os.replace(bak, dest)           # 되돌린다
        raise
    if had:
        shutil.rmtree(bak)
    return dest


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
            "generator_sha256": {}, "input_sha256": {}}


def salt_fingerprint(salt_bytes):
    """salt **자체가 아니라** 지문만 기록한다. 유실 시 ID 가 전부 바뀌므로 번들 밖에
    백업해 두어야 한다(외부 검토 2026-09-25)."""
    return hashlib.sha256(b"salt-fingerprint:" + salt_bytes).hexdigest()[:16]


def stamp(meta, *paths):
    """생성 스크립트 자신들의 SHA-256 을 기록한다."""
    for p in paths:
        meta["generator_sha256"][os.path.basename(p)] = sha256_file(
            p if os.path.isabs(p) else os.path.join(PROJ, p))
    return meta
