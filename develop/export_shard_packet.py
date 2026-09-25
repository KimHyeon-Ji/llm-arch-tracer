r"""**shard 하나만** 담은 격리 패킷을 저장소 밖으로 내보낸다.

왜 필요한가: `work/priority_bundle/` 전체를 한 검토 세션에 주면 73 개 shard 를 모두 열 수
있어서 "shard 당 새 세션" 조건이 성립하지 않는다. 중복본 shard 도 같이 보이므로 독립
측정이 무의미해진다(외부 검토 2026-09-25).

패킷에 담는 것:
    shard.md          해당 shard 하나. **전역 번호를 지운다**
    source/           그 shard 가 참조하는 frozen source 만
    _packet.json      최소 public manifest (packet id, source 해시, 단위 수)

담지 않는 것:
    다른 shard / 전역 shard 번호·총수 / 배정 manifest / 운영 문서 /
    grade·선정 이유·후보 / 중복 여부

`shard067` 같은 이름은 중복본임을 추측할 수 있으므로 **opaque packet id** 로 바꾼다
(`HMAC(salt, shard + revision)`). 단위 순서도 packet id 에서 나온 seed 로 결정론적으로
섞는다 -- 원본과 중복본은 packet id 가 다르므로 순서도 달라진다. 실제 순서는 대장에만
적는다.

**세션과 shard 는 1:1 이다.** 단위가 겹치지 않는 원본 두 개를 한 세션에 주는 것도 막는다.
겹침만 검사하면 "shard 당 새 세션" 이 아니라 "같은 단위를 두 번 주지 않는다" 만 지켜진다
(외부 검토 2026-09-25 가 반례를 실행해 보였다).

실행:
    .venv\Scripts\python.exe develop\export_shard_packet.py --list
    .venv\Scripts\python.exe develop\export_shard_packet.py --shard shard001.md \
        --session S-001 [--out <디렉터리>]
"""
import hashlib
import hmac
import io
import random
import json
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import _buildguard                                               # noqa: E402

NL = chr(10)
LAB = os.path.join(PROJ, "..", "llm-arch-tracer-results-labeled", "work")
BUNDLE = os.path.join(LAB, "priority_bundle")
ASSIGN = os.path.join(LAB, "priority", "_assignment_manifest.json")
LEDGER = os.path.join(LAB, "priority", "_session_ledger.jsonl")
# 저장소 **밖**. 두 워크트리 어느 쪽에도 들어가지 않는다.
OUT_DEFAULT = os.path.join(PROJ, "..", "llm-arch-tracer-review-packets")


def _salt():
    p = os.path.join(LAB, "_private", "unit_id_salt.txt")
    if not os.path.exists(p):
        die("salt 가 없다: work/_private/unit_id_salt.txt")
    return io.open(p, encoding="utf-8").read().strip().encode()


def die(msg):
    print("**패킷을 만들지 않는다** -- " + msg, file=sys.stderr)
    raise SystemExit(2)


def load_assignment():
    """배정 manifest 를 읽고 **bundle 과 맞는지 확인한다.** 안 맞으면 멈춘다.

    교체와 배정 manifest 쓰기는 한 트랜잭션이 아니다 -- 교체 성공 후 프로세스가 죽으면
    새 bundle + 옛 배정 manifest 가 남는다. 그 조합을 조용히 쓰지 않도록
    `priority_bundle_manifest_sha256` 과 revision 을 대조한다(fail-closed).
    """
    if not os.path.exists(ASSIGN):
        die(f"배정 manifest 가 없다: {ASSIGN}")
    am = json.load(io.open(ASSIGN, encoding="utf-8"))
    bmp = os.path.join(BUNDLE, "_manifest.json")
    if not os.path.exists(bmp):
        die(f"priority bundle 이 없다: {bmp}")
    bman = json.load(io.open(bmp, encoding="utf-8"))
    actual = _buildguard.sha256_file(bmp)
    if am.get("priority_bundle_manifest_sha256") != actual:
        die("배정 manifest 가 이 bundle 을 위한 것이 아니다"
            f"{chr(10)}  기록 {am.get('priority_bundle_manifest_sha256')}"
            f"{chr(10)}  실제 {actual}"
            f"{chr(10)}  -> build_review_bundle.py --stage1 을 다시 돌려라")
    if am.get("priority_bundle_assignment_revision") != bman.get("assignment_revision"):
        die(f"revision 불일치: 배정 {am.get('priority_bundle_assignment_revision')} "
            f"!= bundle {bman.get('assignment_revision')}")
    # 자기 해시도 다시 확인한다 (파일이 손댄 흔적 없이 바뀌지 않았는가)
    chk = {k: v for k, v in am.items() if k != "manifest_payload_sha256"}
    if _buildguard.sha256_bytes(json.dumps(
            chk, ensure_ascii=False, sort_keys=True).encode()) != am.get(
            "manifest_payload_sha256"):
        die("배정 manifest 의 자기 해시가 맞지 않는다")
    return am, bman


def packet_id(salt, shard, revision):
    """opaque. shard 번호와 중복 여부를 추측할 수 없게 만든다."""
    mac = hmac.new(salt, f"packet:{revision}:{shard}".encode(), hashlib.sha256)
    return "P-" + mac.hexdigest()[:12]


def shard_units(text):
    body = text.split("## 심볼 정의와 config 값")[0]
    return re.findall(r"### (\S+)", body)


def unit_roles(am):
    """`unit_id -> {"primary": shard, "duplicates": [shard, ...]}`."""
    dup = am.get("duplicate_assignment") or {}
    dup_shards = {s for v in dup.values() for s in v}
    roles = {}
    for entry in am.get("manifest") or []:
        sh = entry["shard"]
        for uid in entry["unit_ids"]:
            r = roles.setdefault(uid, {"primary": None, "duplicates": []})
            if sh in dup_shards and uid in dup:
                r["duplicates"].append(sh)
            else:
                r["primary"] = sh
    return roles


def read_ledger():
    if not os.path.exists(LEDGER):
        return []
    return [json.loads(l) for l in io.open(LEDGER, encoding="utf-8") if l.strip()]


def write_ledger(records):
    """대장 전체를 임시 파일에 쓰고 교체한다. append 는 중간에 끊기면 반 줄이 남는다."""
    tmp = LEDGER + ".tmp"
    with io.open(tmp, "w", encoding="utf-8", newline=chr(10)) as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + chr(10))
    os.replace(tmp, LEDGER)


class ledger_lock:
    """`os.mkdir` 는 원자적이다. 두 exporter 가 동시에 옛 대장을 읽어 같은 세션을
    예약하는 것을 막는다(외부 검토 2026-09-25)."""

    def __enter__(self):
        self.p = LEDGER + ".lock"
        try:
            os.makedirs(self.p)
        except FileExistsError:
            die(f"다른 반출이 진행 중이다 ({self.p}) -- 끝난 뒤 다시 하거나, "
                "죽은 프로세스가 남긴 것이면 그 디렉터리를 지워라")
        io.open(os.path.join(self.p, "pid"), "w").write(str(os.getpid()))
        return self

    def __exit__(self, *exc):
        shutil.rmtree(self.p, ignore_errors=True)


def check_out_root(out_root):
    """반출 위치가 **저장소 안이면 거부한다.** 기본값만 밖이어서는 강제가 아니다."""
    real = os.path.realpath(out_root)
    labroot = os.path.realpath(os.path.join(LAB, ".."))
    for name, root in (("tracer 워크트리", os.path.realpath(PROJ)),
                       ("results-labeled 워크트리", labroot)):
        if real == root or real.startswith(root + os.sep):
            die(f"반출 위치가 {name} 안이다: {real}{chr(10)}"
                "  검토자에게 저장소 경로를 주지 않는 것이 계약이다")
    return real


def main():
    args = sys.argv[1:]

    def opt(name, default=None):
        return args[args.index(name) + 1] if name in args else default

    am, bman = load_assignment()
    salt = _salt()
    rev = am["assignment_revision"]
    roles = unit_roles(am)

    if "--list" in args:
        ledger = read_ledger()
        dup = am.get("duplicate_assignment") or {}
        dup_shards = {s for v in dup.values() for s in v}
        done = {r["shard"]: r for r in ledger}
        print(f"revision {rev}   shard {len(am['manifest'])}   "
              f"중복 단위 {len(dup)}")
        print(f"{'packet id':<16}{'세션':<12}{'단위':>5}  역할")
        for e in am["manifest"]:
            sh = e["shard"]
            pid = packet_id(salt, sh, rev)
            r = done.get(sh)
            print(f"{pid:<16}{(r['session_id'] if r else '-'):<12}"
                  f"{e['units']:>5}  {'중복본' if sh in dup_shards else '원본'}")
        return 0

    shard = opt("--shard")
    session = opt("--session")
    if not shard or not session:
        print(__doc__)
        die("--shard 와 --session 이 필요하다")
    out_root = check_out_root(opt("--out", OUT_DEFAULT))

    # ---- **대장 읽기부터 갱신까지 한 lock 안에서.** 두 exporter 가 동시에 옛 대장을
    #      읽으면 같은 세션을 두 번 예약할 수 있다.
    with ledger_lock():
        return _export(am, bman, salt, rev, roles, shard, session, out_root)


def _export(am, bman, salt, rev, roles, shard, session, out_root):
    ledger = read_ledger()
    entry = next((e for e in am["manifest"] if e["shard"] == shard), None)
    if entry is None:
        die(f"배정 manifest 에 없는 shard: {shard}")
    sp = os.path.join(BUNDLE, "shards", shard)
    if not os.path.exists(sp):
        die(f"shard 파일이 없다: {sp}")

    # ---- **세션 <-> shard 는 1:1 이다.** 단위가 겹치지 않는 원본 두 개를 한 세션에
    #      주는 것도 막는다. 겹침만 검사하면 "같은 단위를 두 번 주지 않는다" 만 지켜지고
    #      "shard 당 새 세션" 은 지켜지지 않는다 -- 반례가 실제로 통과했다.
    used = next((r for r in ledger if r["session_id"] == session), None)
    if used:
        die(f"세션 {session} 은 이미 {used['shard']} 를 받았다 -- "
            "세션 하나에 shard 하나다 (재전달도 마찬가지)")
    prev = next((r for r in ledger if r["shard"] == shard), None)
    if prev:
        die(f"{shard} 는 이미 세션 {prev['session_id']} 에 나갔다")
    # 겹침 검사도 남긴다 -- 1:1 이 깨지지 않아도 대장이 손으로 편집될 수 있다.
    # **세션 단위로만** 본다. 전역으로 보면 의도된 중복 배정까지 막힌다
    # (중복본은 원본과 단위가 겹치는 것이 정상이다 -- 만들다 실제로 막혔다).
    mine = set(entry["unit_ids"])
    for r in ledger:
        if r["session_id"] == session and mine & set(r["unit_ids"]):
            die(f"세션 {session} 은 이 단위를 이미 받았다 "
                f"({len(mine & set(r['unit_ids']))} 개) -- 원본과 중복본은 다른 세션이다")
    # 한 단위가 세 번 이상 나가는 것은 배정 설계에 없다 (원본 1 + 중복 1)
    times = {u: sum(1 for r in ledger if u in set(r["unit_ids"])) for u in mine}
    over = [u for u, n in times.items() if n >= 2]
    if over:
        die(f"이미 두 세션에 나간 단위가 {len(over)} 개 있다 -- 배정은 원본 1 + 중복 1 이다")

    text = io.open(sp, encoding="utf-8").read()
    pid = packet_id(salt, shard, rev)
    dest = os.path.join(out_root, pid)

    # ---- **전역 번호를 지운다.** `# shard 067 / 73` 은 중복본임을 알려 준다.
    n_units = len(shard_units(text))
    text = re.sub(r"^# shard \d+ / \d+ — 축 판정 \([^)]*\)",
                  f"# 패킷 {pid} — 축 판정 ({n_units} 단위)", text, count=1,
                  flags=re.M)
    if re.search(r"shard\s*\d", text):
        die("shard 본문에 전역 번호가 남아 있다 -- 패턴을 확인해라")

    # ---- **단위 순서를 결정론적으로 섞는다.** 한 세션이 12 단위를 순서대로 보므로 앞
    #      단위가 뒤에 영향을 줄 수 있다(순서 효과). packet id 에서 seed 를 뽑으므로
    #      원본과 중복본은 서로 다른 순서가 된다. 순서는 **대장에만** 적는다.
    SYM = "## 심볼 정의와 config 값"
    body, sep, tail = text.partition(SYM)
    parts = body.split(NL + "### ")
    head, blocks = parts[0], parts[1:]
    order = list(range(len(blocks)))
    seed = int(hmac.new(salt, f"order:{pid}".encode(),
                        hashlib.sha256).hexdigest()[:16], 16)
    random.Random(seed).shuffle(order)
    text = head + "".join(NL + "### " + blocks[i] for i in order) + sep + tail
    unit_order = [blocks[i].split(NL, 1)[0].strip() for i in order]

    # ---- 이 shard 가 참조하는 source 만.
    #      **목록 줄에서만 뽑는다** -- 답 형식 예시에도 `"file": "source/..."` 가 있어서
    #      본문 전체를 긁으면 `...",` 같은 것이 잡힌다(만들다 실제로 겪었다).
    refs = sorted(set(re.findall(r"^source/(\S+)\s+\d+ 줄", text, flags=re.M)))
    want = {}
    for m, files in (bman.get("sources") or {}).items():
        for e in files:
            want[e["bundle_path"].split("/", 1)[1]] = e
    missing = [r for r in refs if r not in want]
    if missing:
        die(f"참조된 source 가 bundle manifest 에 없다: {missing[:3]}")
    if not refs:
        die("shard 가 참조하는 source 를 찾지 못했다 -- 목록 형식이 바뀌었는가")
    # bundle 이 아는 이름이 본문에 나왔는데 목록에서 안 뽑혔으면 형식이 바뀐 것이다
    lost = [n for n in want if n in text and n not in refs]
    if lost:
        die(f"본문에 있으나 목록에서 못 뽑은 source: {lost[:3]}")

    tmp = dest + ".tmp"
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    os.makedirs(os.path.join(tmp, "source"))
    src_meta = []
    for name in refs:
        s = os.path.join(BUNDLE, "source", name)
        d = os.path.join(tmp, "source", name)
        shutil.copyfile(s, d)
        h = _buildguard.sha256_file(d)
        if h != want[name]["sha256"]:
            die(f"source 해시가 bundle manifest 와 다르다: {name}")
        src_meta.append({"path": f"source/{name}", "sha256": h,
                         "lines": want[name]["lines"]})
    io.open(os.path.join(tmp, "shard.md"), "w", encoding="utf-8",
            newline=chr(10)).write(text)
    json.dump({"packet_id": pid, "units": n_units, "sources": src_meta,
               "answer_file": "answers.jsonl",
               "note": "이 패킷 안의 자료만 보고 답해 주세요. 다른 패킷·저장소 자료는 "
                       "주어지지 않습니다."},
              io.open(os.path.join(tmp, "_packet.json"), "w", encoding="utf-8",
                      newline=chr(10)), ensure_ascii=False, indent=1)

    # ---- 누설 검사: 패킷 안에 다른 shard·금지 자료가 없는가
    bad = []
    for root, dirs, files in os.walk(tmp):
        for n in list(dirs) + files:
            if any(b in n for b in ("shard0", "units", "crosswalk", "priority",
                                    "assignment", "salt", "_manifest")):
                bad.append(os.path.relpath(os.path.join(root, n), tmp))
    if bad:
        shutil.rmtree(tmp, ignore_errors=True)
        die(f"패킷에 금지된 이름이 있다: {bad[:3]}")

    _buildguard.swap_dir(tmp, dest)
    ledger.append({
            "packet_id": pid, "shard": shard, "session_id": session,
            "assignment_revision": rev, "units": n_units,
            "unit_ids": entry["unit_ids"], "unit_order": unit_order,
            "order_seed_source": "HMAC(salt, 'order:' + packet_id)",
            "roles": {uid: ("duplicate" if shard in roles.get(uid, {}).get(
                "duplicates", []) else "primary") for uid in entry["unit_ids"]},
            "packet_dir": os.path.relpath(dest, PROJ).replace(os.sep, "/"),
            "source_files": len(src_meta)})
    write_ledger(ledger)
    print(f"패킷 {pid}  ({n_units} 단위 / source {len(src_meta)} 파일)")
    print(f"  세션 {session}   shard {shard} (대장에만 기록)")
    print(f"  -> {dest}")
    print("  **이 디렉터리만** 검토 세션에 준다. 저장소 경로는 주지 않는다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
