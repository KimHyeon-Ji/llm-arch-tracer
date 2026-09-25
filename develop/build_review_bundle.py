r"""0-b.1: 1차 블라인드 판정용 **review bundle** 을 만든다.

왜 별도 bundle 인가: 같은 workspace 에서 리뷰하면 문서를 아무리 가려도 `work/units`,
crosswalk, 현재 symbolic csv/jsonl, 생성 코드를 열어 원본을 찾아볼 수 있다
(외부 검토 2026-09-25). 그래서 **질문과 frozen source/evidence 만** 담은 디렉터리를 따로
만들고, salt·registry·units·crosswalk 는 넣지 않는다.

이 bundle 에 **없는** 것:
    work/units/            (signature 에 현재 라벨이 들어 있다)
    work/crosswalk/        (원장 등급·후보가 들어 있다)
    work/_private/         (unit id salt)
    _family_registry.jsonl (label 을 담는다)
    models/*.csv, *.jsonl  (현재 라벨 그 자체)
    생성 코드

1차는 **후보 없이** 자유 이름 제안이다(승인된 운영 방식). `cannot_determine` 인 단위만
2 라운드에서 후보를 섞어 다시 묻고, 그 결과는 `candidate_assisted=true` 로 약하게 취급한다.

실행:
    .venv\Scripts\python.exe develop\build_review_bundle.py [shard_size] [seed]
"""
import collections
import gzip
import io
import json
import os
import random
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(PROJ, "src"))
if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import yaml                                                      # noqa: E402
import _buildguard                                               # noqa: E402

MODELS = os.path.join(PROJ, "models")
LAB = os.path.join(PROJ, "..", "llm-arch-tracer-results-labeled", "work")
UNITS = os.path.join(LAB, "units")
BUNDLE = os.path.join(LAB, "review_bundle")

PUBLISHED = (
    "deepseek-ai__DeepSeek-V4-Pro",
    "meta-llama__Llama-4-Maverick-17B-128E",
    "moonshotai__Kimi-K3",
    "openai__gpt-oss-120b",
    "openai__gpt-oss-20b",
)

# 모델마다 **실행된** 구현체. Kimi 는 remote modeling 과 shim 도 함께 준다
# (naive.py 만으로는 모델 의미를 정의할 수 없다 -- 외부 검토 2026-09-25).
SOURCES = {
    "deepseek-ai__DeepSeek-V4-Pro": [
        ".venv/Lib/site-packages/transformers/models/deepseek_v4/modeling_deepseek_v4.py",
        ".venv/Lib/site-packages/transformers/models/deepseek_v4/configuration_deepseek_v4.py",
        ".venv/Lib/site-packages/transformers/integrations/moe.py"],
    "meta-llama__Llama-4-Maverick-17B-128E": [
        ".venv/Lib/site-packages/transformers/models/llama4/modeling_llama4.py",
        ".venv/Lib/site-packages/transformers/models/llama4/configuration_llama4.py"],
    "openai__gpt-oss-120b": [
        ".venv/Lib/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py",
        ".venv/Lib/site-packages/transformers/integrations/moe.py"],
    "openai__gpt-oss-20b": [
        ".venv/Lib/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py",
        ".venv/Lib/site-packages/transformers/integrations/moe.py"],
    "moonshotai__Kimi-K3": [
        ".venv/Lib/site-packages/fla/ops/kda/naive.py",
        ".venv/Lib/site-packages/fla/ops/kda/gate.py",
        "src/kda_shim.py"],
}
# 모델마다 config 의 원본 (HF 캐시). 경로는 provenance 에서 읽는다.
FIELD_KO = {"i": "입력", "o": "출력", "w": "가중치"}
FIELD_KEY = {"i": "input_shape", "o": "output_shape", "w": "weight_shape"}


def _kimi_remote(model):
    """Kimi 의 remote modeling 파일 경로 (HF 캐시)."""
    pv = os.path.join(MODELS, model, "full", "provenance.json")
    if not os.path.exists(pv):
        return []
    rev = (json.load(io.open(pv, encoding="utf-8")) or {}).get("revision_resolved")
    if not rev:
        return []
    base = os.path.expanduser(
        f"~/.cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/{rev}")
    return [os.path.join(base, n) for n in
            ("modeling_kimi_linear.py", "configuration_kimi_k3.py", "config.json")
            if os.path.exists(os.path.join(base, n))]


def _assign(positions, target, linked):
    """가릴 자리마다 placeholder 를 배정한다.

    `X_same` 을 쓰면 **"이 두 자리는 같은 축이다" 라는 현재 시스템의 가정을 검토자에게
    미리 알려 준다.** 라벨 문자열이 같다는 이유만으로 그렇게 표시하면 블라인드 검토가
    아니다(외부 검토 2026-09-25).

    * `X`         판정 대상
    * `X_linked`  provenance/dataflow 로 **같은 축임이 독립적으로 입증된** 자리
                  (축 등가류가 대상과 겹친다)
    * `Y1`, `Y2`  그 밖 -- 라벨만 같을 뿐 동일 축임이 입증되지 않았다. **서로 다른**
                  placeholder 를 준다
    """
    out, n = {}, 0
    for pos in positions:
        if pos == target:
            out[pos] = "X"
        elif pos in linked:
            out[pos] = "X_linked"
        else:
            n += 1
            out[pos] = f"Y{n}"
    return out


def _mask(shape, secret, ph, field, sidx, rx=None):
    """`ph` 는 `(field, shape_index, axis) -> placeholder`."""
    out = []
    for ax, e in enumerate(shape):
        t = str(e)
        key = (field, sidx, ax)
        if t == secret:
            out.append(ph.get(key, "Y?"))
        elif rx.search(t):
            out.append(rx.sub(ph.get(key, "Y?"), t))
        else:
            out.append(t)
    return out


def _lineage(model, phase):
    """`(published op_id, field, si, axis) -> 축 등가류 root 집합`.

    crosswalk 의 `raw_sites` 를 축 등가류(`axis_classes.build`)에 넣어 얻는다.
    "같은 축임이 독립적으로 입증됐다" 의 근거이고, 이것이 있을 때만 `X_linked` 를 쓴다.
    """
    import axis_classes as AC
    rows = [json.loads(l) for l in io.open(
        os.path.join(MODELS, model, "full", f"{phase}.trace.raw.jsonl"), encoding="utf-8")]
    conc = {}
    with io.open(os.path.join(MODELS, model, "full",
                              f"{phase}.shapes.concrete.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            conc[r["op_id"]] = r
    uf = AC.build(rows, conc)
    out = {}
    with gzip.open(os.path.join(LAB, "crosswalk", f"{model}.{phase}.jsonl.gz"),
                   "rt", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            if not c.get("is_question_cell") and c["field"] != "w":
                pass
            roots = {uf.find(tuple(rs)) for rs in c["raw_sites"] if rs[1] in ("i", "o")}
            out[(c["op_id"], c["field"], c["shape_index"], c["axis"])] = roots
    return out


def _shapes(row, field):
    v = row.get(FIELD_KEY[field]) if isinstance(row, dict) else None
    if not v:
        return []
    return [v] if (field == "w" and not isinstance(v[0], list)) else v


INPUTS = []


def _salt_bytes():
    p = os.path.join(LAB, "_private", "unit_id_salt.txt")
    return (io.open(p, encoding="utf-8").read().strip().encode()
            if os.path.exists(p) else b'')


def main():
    shard_size = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 20260925
    meta = _buildguard.require_clean_tree()
    _buildguard.stamp(meta, "develop/build_review_bundle.py", "develop/_buildguard.py")
    rnd = random.Random(seed)

    units = []
    pubs, sts = {}, {}
    for m in PUBLISHED:
        st = yaml.safe_load(io.open(os.path.join(MODELS, m, "structure.yaml"),
                                    encoding="utf-8"))
        sts[m] = st
        for ph in ("prefill", "decode"):
            up = os.path.join(UNITS, f"{m}.{ph}.units.jsonl")
            if not os.path.exists(up):
                continue
            pubs[(m, ph)] = {r["op_id"]: r for r in
                             (json.loads(l) for l in
                              io.open(os.path.join(MODELS, m, f"{ph}.jsonl"),
                                      encoding="utf-8"))}
            INPUTS.extend([up, os.path.join(MODELS, m, f"{ph}.jsonl"),
                           os.path.join(MODELS, m, "structure.yaml"),
                           os.path.join(MODELS, m, "full", "provenance.json"),
                           os.path.join(MODELS, m, "full",
                                        f"{ph}.shapes.concrete.jsonl"),
                           os.path.join(LAB, "crosswalk", f"{m}.{ph}.jsonl.gz")])
            for u in (json.loads(l) for l in io.open(up, encoding="utf-8")):
                if u["affects_published_cells"] > 0:
                    u["_phase"] = ph
                    units.append(u)

    lin = {}
    for (m, ph) in pubs:
        print(f"   lineage {m} {ph} …", flush=True)
        lin[(m, ph)] = _lineage(m, ph)

    # ---- 층화 셔플: 같은 family·cohort·module 을 한 shard 에 몰지 않는다
    units.sort(key=lambda u: -u["affects_published_cells"])
    buckets = collections.defaultdict(list)
    for u in units:
        buckets[(u["question_family_id"], u["signature"]["layer_cohort_id"],
                 u["signature"]["module"])].append(u)
    keys = list(buckets)
    rnd.shuffle(keys)
    ordered, i = [], 0
    while any(buckets[k] for k in keys):
        for k in keys:
            if buckets[k]:
                ordered.append(buckets[k].pop(0))
    shards = [ordered[i:i + shard_size] for i in range(0, len(ordered), shard_size)]

    # ---- **임시 디렉터리에 새로 만들고 원자적으로 교체한다.** 제자리에서 갱신하면
    #      옛 사본이 남고 manifest 의 해시와 어긋날 수 있다(외부 검토 2026-09-25).
    TMP = BUNDLE + ".tmp"
    if os.path.isdir(TMP):
        shutil.rmtree(TMP)
    os.makedirs(os.path.join(TMP, "source"))
    src_meta = {}
    for m in PUBLISHED:
        paths = [os.path.join(PROJ, p) for p in SOURCES.get(m, [])]
        if m == "moonshotai__Kimi-K3":
            paths += _kimi_remote(m)
        INPUTS.extend(paths)
        for p in paths:
            if not os.path.exists(p):
                continue
            name = f"{m.split('__')[-1][:10]}__{os.path.basename(p)}"
            dst = os.path.join(TMP, "source", name)
            # **항상 덮어쓴다.** 예전에는 목적지가 있으면 건너뛰면서 manifest 에는
            # 원본의 새 해시를 적어, 원본이 바뀌면 사본과 어긋날 수 있었다.
            io.open(dst, "wb").write(io.open(p, "rb").read())
            assert (_buildguard.sha256_file(dst)
                    == _buildguard.sha256_file(p)), name
            src_meta.setdefault(m, []).append(
                {"bundle_path": f"source/{name}",
                 "original": os.path.relpath(p, PROJ) if p.startswith(PROJ) else p,
                 "sha256": _buildguard.sha256_file(p),
                 "lines": sum(1 for _ in io.open(p, encoding="utf-8", errors="replace"))})

    os.makedirs(os.path.join(TMP, "shards"))
    manifest = []
    for si, sh in enumerate(shards, 1):
        L = [f"# shard {si:03d} / {len(shards)} — 축 판정 ({len(sh)} 단위)", "",
             "각 단위의 **`X` 로 표시된 축이 무엇인지** 답해 주세요.", "",
             "* `X`        판정 대상 축",
             "* `X_linked` **같은 축임이 독립적으로 입증된** 자리 (축 계보가 대상과 겹칩니다)",
             "* `Y1`, `Y2` 함께 가린 그 밖의 자리. **서로 같다는 보장이 없습니다** --",
             "             값이 같아 보여도 다른 축일 수 있으니 각각 판단하세요",
             "* placeholder 가 합성식 안에 있으면(`X*d_model`) 그 식 안에 그 축이 들어 있다는 뜻입니다",
             "",
             "**현재 붙어 있는 이름·등급은 알려 드리지 않고, 후보 목록도 주지 않습니다.**",
             "소스와 아래 정보만으로 판단해 주세요. 이름이 없는 것이 맞다고 판단되면",
             "`no_name` 으로, 근거가 부족하면 `cannot_determine` 으로 답해 주세요.",
             "**추측해서 채우지 마세요** -- `cannot_determine` 이 틀린 이름보다 낫습니다.", "",
             "## 답 형식", "", "```json",
             json.dumps({"decision_unit_id": "...",
                         "proposal": "named | no_name | cannot_determine",
                         "proposed_expr": "(named 일 때만)",
                         "evidence": [{"kind": "source", "file": "source/...",
                                       "lines": "120-138", "source_sha256": "<64 hex>",
                                       "claim": "..."},
                                      {"kind": "trace_or_metamorphic",
                                       "artifact": "...", "claim": "..."}],
                         "rejected_candidates": [{"expr": "...", "reason": "..."}],
                         "assumptions": [], "confidence": "low | medium | high"},
                        ensure_ascii=False, indent=1),
             "```", "",
             "근거는 **두 종류 이상**(선언부 / 변환부 / callsite) 을 대 주세요.",
             "`source_sha256` 은 아래 목록의 값을 그대로 적으면 됩니다.", "", "---", ""]
        seen_models = []
        for u in sh:
            m, ph = u["model"], u["_phase"]
            if m not in seen_models:
                seen_models.append(m)
            sig = u["signature"]
            secret = sig["old_expr"]
            rx = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(secret) + r"(?![A-Za-z0-9_])")
            oid, field, sidx, ax = tuple(u["published_cells"][0])
            prow = pubs[(m, ph)][oid]
            cs = u.get("concrete_shapes") or {}
            L += [f"### {u['decision_unit_id']}", "",
                  f"* 모델 / phase: `{m}` / `{ph}`",
                  f"* block: `{sig['block_type']}`   층 cohort: `{sig['layer_cohort_id']}`",
                  f"* 모듈: `{sig['module']}`   연산: `{sig['op_type']}` (`{sig['raw_op']}`)",
                  f"* 판정할 자리: **{FIELD_KO[field]} shape[{sidx}] 의 축 {ax}**",
                  f"* 이 축의 concrete 값: **{u.get('concrete_value')}**",
                  f"* 이 단위가 대표하는 발행 셀 {u['affects_published_cells']:,} / "
                  f"raw 자리 {u['represents_raw_sites']:,}", ""]
            if sig.get("param_role"):
                L += [f"* parameter 역할: `{', '.join(sig['param_role'])}`", ""]
            # ---- 가릴 자리를 모아 lineage 로 placeholder 를 배정한다
            LN = lin.get((m, ph), {})
            tgt_roots = LN.get((oid, field, sidx, ax)) or set()
            masked, target = [], (field, sidx, ax)
            for f in ("i", "w", "o"):
                for j, sh in enumerate(_shapes(prow, f)):
                    for a2, e2 in enumerate(sh):
                        if str(e2) == secret or rx.search(str(e2)):
                            masked.append((f, j, a2))
            linked = {pos for pos in masked
                      if pos != target
                      and (LN.get((oid, pos[0], pos[1], pos[2])) or set()) & tgt_roots}
            ph_map = _assign(masked, target, linked)
            L += ["shape (가린 형태):", "", "```"]
            for f in ("i", "w", "o"):
                shs = _shapes(prow, f)
                if not shs:
                    continue
                parts = ["[" + ", ".join(_mask(sh, secret, ph_map, f, j, rx)) + "]"
                         for j, sh in enumerate(shs)]
                L.append(f"  {FIELD_KO[f]:4} {', '.join(parts)}")
            L += ["```", "", "같은 자리의 concrete shape:", "", "```"]
            for f in ("i", "w", "o"):
                v = cs.get(f)
                if not v:
                    continue
                shs = [v] if (f == "w" and not isinstance(v[0], list)) else v
                L.append(f"  {FIELD_KO[f]:4} " + ", ".join(
                    "[" + ", ".join(str(x) for x in s) + "]" for s in shs))
            L += ["```", ""]
            nb = sig.get("neighbours") or {}
            if nb.get("prev") or nb.get("next"):
                L += ["dataflow:", "", "```"]
                for t in nb.get("prev", []):
                    L.append(f"  직전  {t[0]:16} {t[1]}")
                for t in nb.get("next", []):
                    L.append(f"  직후  {t[0]:16} {t[1]}")
                L += ["```", ""]
            L += ["---", ""]
        # shard 에 등장한 모델의 심볼표·소스만 싣는다
        L += ["## 심볼 정의와 config 값", ""]
        for m in seen_models:
            st = sts[m]
            syms = {k: v for k, v in (st.get("symbols") or {}).items() if v is not None}
            L += [f"### {m}", "", "```",
                  json.dumps(syms, ensure_ascii=False, indent=1), "```", ""]
            lo = st.get("symbols_label_only") or {}
            if lo:
                L += ["심볼표에 없지만 표에 쓰이는 식:", "", "```",
                      json.dumps(lo, ensure_ascii=False, indent=1), "```", ""]
            sc = st.get("scope") or {}
            L += [f"추적 범위: B={sc.get('batch')}, prefill 길이 {sc.get('prefill_len')}, "
                  f"decode query 1 / 캐시 {sc.get('decode_cache_len')}", "",
                  "볼 소스 (이 bundle 안, SHA-256 전체):", "", "```"]
            for e in src_meta.get(m, []):
                L.append(f"{e['bundle_path']:56} {e['lines']:>6} 줄")
                L.append(f"  sha256 {e['sha256']}")
            L += ["```", ""]
        io.open(os.path.join(TMP, "shards", f"shard{si:03d}.md"), "w",
                encoding="utf-8", newline="\n").write("\n".join(L))
        manifest.append({"shard": f"shard{si:03d}.md", "units": len(sh),
                         "unit_ids": [u["decision_unit_id"] for u in sh],
                         "models": seen_models,
                         "affects_published_cells": sum(
                             u["affects_published_cells"] for u in sh),
                         "candidate_seed": rnd.randint(1, 10 ** 9),
                         "status": "pending"})
    # ---- **선언이 아니라 검사다.** 금지 경로·비밀 노출·ID 충돌을 실제로 센다.
    secrets_by_uid = {u["decision_unit_id"]: u["signature"]["old_expr"]
                      for u in units}
    checks = {"forbidden_paths": [], "secret_exposed_units": [],
              "candidate_list_shards": 0, "duplicate_unit_ids": 0,
              "shard_unit_total": sum(len(x) for x in shards)}
    seen_ids = set()
    for u in units:
        if u["decision_unit_id"] in seen_ids:
            checks["duplicate_unit_ids"] += 1
        seen_ids.add(u["decision_unit_id"])
    FORBID = ("units", "crosswalk", "_private", "_family_registry", "salt")
    for root, dirs, files in os.walk(TMP):
        for n in list(dirs) + files:
            if any(b in n for b in FORBID):
                checks["forbidden_paths"].append(
                    os.path.relpath(os.path.join(root, n), TMP))
    for sp in sorted(os.listdir(os.path.join(TMP, "shards"))):
        doc = io.open(os.path.join(TMP, "shards", sp), encoding="utf-8").read()
        if "참고 후보" in doc:
            checks["candidate_list_shards"] += 1
        body = doc.split("## 심볼 정의와 config 값")[0]
        for uid in re.findall(r"### (\S+)", body):
            sec = secrets_by_uid.get(uid)
            if not sec:
                continue
            i = body.find("### " + uid)
            j = body.find("### ", i + 4)
            blk = body[i:j if j > 0 else len(body)]
            if re.search(r"(?<![A-Za-z0-9_])" + re.escape(sec)
                         + r"(?![A-Za-z0-9_])", blk):
                checks["secret_exposed_units"].append(uid)
    meta.update({"shard_size": shard_size, "shuffle_seed": seed,
                 "shards": len(shards), "units": len(units),
                 "sources": src_meta, "manifest": manifest,
                 "salt_fingerprint": _buildguard.salt_fingerprint(_salt_bytes()),
                 "checks": {k: (len(v) if isinstance(v, list) else v)
                            for k, v in checks.items()},
                 "check_detail": {k: v[:10] for k, v in checks.items()
                                  if isinstance(v, list) and v},
                 "excluded_from_bundle": [
                     "work/units/", "work/crosswalk/", "work/_private/",
                     "_family_registry.jsonl", "models/*.csv", "models/*.jsonl",
                     "생성 코드"]})
    meta["input_sha256"] = _buildguard.input_manifest(INPUTS)
    meta["input_worktree_dirty"] = len(
        _buildguard.worktree_dirty(os.path.dirname(LAB)))
    json.dump(meta, io.open(os.path.join(TMP, "_manifest.json"), "w",
                            encoding="utf-8", newline=chr(10)),
              ensure_ascii=False, indent=1)

    bad = (checks["forbidden_paths"] or checks["secret_exposed_units"]
           or checks["candidate_list_shards"] or checks["duplicate_unit_ids"]
           or checks["shard_unit_total"] != len(units))
    if bad:
        print("**검사 실패 -- bundle 을 교체하지 않는다**", file=sys.stderr)
        print(json.dumps(meta["checks"], ensure_ascii=False), file=sys.stderr)
        raise SystemExit(3)

    if os.path.isdir(BUNDLE):
        shutil.rmtree(BUNDLE)
    os.replace(TMP, BUNDLE)
    print(f"단위 {len(units):,} -> shard {len(shards)} 개 (shard 당 {shard_size})")
    print(f"frozen source {sum(len(v) for v in src_meta.values())} 파일")
    print("검사 " + json.dumps(meta["checks"], ensure_ascii=False))
    print("입력 해시 " + str(len(meta["input_sha256"])) + " 파일"
          + "  입력 워크트리 미커밋 " + str(meta["input_worktree_dirty"]))
    print(f"-> {os.path.relpath(BUNDLE, PROJ)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
