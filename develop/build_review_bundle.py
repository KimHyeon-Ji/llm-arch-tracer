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

**`X_linked` 는 철회했다(외부 검토 2026-09-25).** "같은 축임이 독립적으로 입증됐다" 고
표시했지만 근거가 없었다: `_lineage()` 가 `axis_classes.build()` 를 mode 없이 불러
`DEFAULT_MODE = "legacy"` 로 돌았고, legacy 는 포트 provenance 가 아니라 `depends_on` 과
concrete shape 일치에 기반한 **값 간선**을 쓴다. 게다가 대상 10 개 model/phase 의 raw
trace 에 `input_sources` 가 **0 / 773,497 행**이고 `*.ports.jsonl` 은 전부 **0 바이트**다.
즉 `X_same` 을 이름 기반 추정에서 값 기반 추정으로 바꾼 것에 불과했다.

그래서 기본값은 `lineage_mode="none"` 이고, 대상 `X` 밖의 가린 자리는 **전부 서로 다른**
`Y1..Yn` 이다. 되살리려면 `--lineage=provenance` 를 명시해야 하고, fallback 없이
port coverage 가 1.0 이어야 통과한다(0 이면 실패).

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
# 1 단계 배정용 **별도** bundle. 모집단 bundle 과 한 디렉터리에 두면 두 shard 집합을
# 맞춰 보는 것만으로 "어느 단위가 위험 등급인가" 가 드러난다(외부 검토 2026-09-25).
PRIORITY_BUNDLE = os.path.join(LAB, "priority_bundle")

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
    * `Y1`, `Y2`  그 밖 -- **서로 같다는 표시를 하지 않는다.** 각각 다른 placeholder 다
    * `X_linked`  `--lineage=provenance` 로 **실제 포트 provenance** 가 있을 때만 쓴다.
                  기본값에서는 `linked` 가 비어 있으므로 등장하지 않는다

    `linked` 를 값·이름 추정으로 채우면 안 된다. 그러면 "이 둘은 같은 축" 이라는 현재
    시스템의 가정을 검토자에게 알려 주는 셈이다(외부 검토 2026-09-25).
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


# 렌더러에 넘길 수 있는 필드 **화이트리스트**. `"참고 후보"` 라는 문구 하나만 찾는 검사로는
# 제목이나 형식이 바뀌면 통과한다(외부 검토 2026-09-25). 그래서 **민감 필드를 제거한
# public DTO 만 렌더 입력으로 쓰고**, 그 DTO 에 금지 키가 없음을 구조적으로 검사한다.
PUBLIC_UNIT_KEYS = ("decision_unit_id", "model", "_phase", "published_cells",
                    "concrete_value", "concrete_shapes",
                    "affects_published_cells", "represents_raw_sites")
PUBLIC_SIG_KEYS = ("block_type", "layer_cohort_id", "module", "op_type", "raw_op",
                   "param_role", "neighbours")
# 어느 깊이에서든 나오면 안 되는 키
PRIVATE_KEYS = ("old_expr", "candidates", "grade", "reason", "reasons", "expr",
                "label", "question_family_id", "selection_reasons",
                "stage1_reasons", "population_shard")


def _public(u, stage1=False):
    """렌더 입력을 만든다. **원본 단위 dict 를 렌더러에 넘기지 않는다.**

    `stage1` 에서는 발행 영향 수도 뺀다 -- 1 단계 대상은 위험 등급이 전부 포함돼 있어서
    "셀 1 개" 가 곧 "위험 등급으로 뽑혔다" 를 알려 준다.
    """
    v = {k: u[k] for k in PUBLIC_UNIT_KEYS if k in u}
    if stage1:
        v.pop("affects_published_cells", None)
        v.pop("represents_raw_sites", None)
    sig = u.get("signature") or {}
    v["signature"] = {k: sig[k] for k in PUBLIC_SIG_KEYS if k in sig}
    return v


def _has_private(o):
    if isinstance(o, dict):
        return (any(k in PRIVATE_KEYS for k in o)
                or any(_has_private(x) for x in o.values()))
    if isinstance(o, list):
        return any(_has_private(x) for x in o)
    return False


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


def _count_lines(p):
    if not os.path.exists(p):
        return 0
    n = 0
    with io.open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            n += b.count(chr(10).encode())
    return n


def port_coverage(model, phase):
    """포트 provenance 가 **실제로 있는가.** 없으면 lineage 를 쓸 수 없다."""
    pp = os.path.join(MODELS, model, "full", f"{phase}.ports.jsonl")
    raw = os.path.join(MODELS, model, "full", f"{phase}.trace.raw.jsonl")
    pl, rl = _count_lines(pp), _count_lines(raw)
    return {"ports_lines": pl, "raw_lines": rl,
            "ports_bytes": os.path.getsize(pp) if os.path.exists(pp) else None,
            "coverage": (round(pl / rl, 6) if rl else None)}


def _lineage(model, phase, mode):
    """`(published op_id, field, si, axis) -> 축 등가류 root 집합`.

    **`mode` 는 반드시 `"provenance"` 다.** legacy/migration/hybrid 는 값 간선을 쓰므로
    "독립적으로 입증된 계보" 의 근거가 될 수 없다. fallback 을 두지 않는다 --
    포트가 없으면 `X_linked` 를 쓰지 않는 것이 맞고, 조용히 legacy 로 내려가서는 안 된다
    (외부 검토 2026-09-25).
    """
    if mode != "provenance":
        raise ValueError(f"lineage mode 는 provenance 여야 한다 (받은 값: {mode!r})")
    cov = port_coverage(model, phase)
    if not cov["coverage"] or cov["coverage"] < 1.0:
        print(f"**포트 provenance 가 없다 -- lineage 를 쓸 수 없다**: {model} {phase} "
              f"ports {cov['ports_lines']} 행 / raw {cov['raw_lines']} 행",
              file=sys.stderr)
        raise SystemExit(4)
    import axis_classes as AC
    rows = [json.loads(l) for l in io.open(
        os.path.join(MODELS, model, "full", f"{phase}.trace.raw.jsonl"), encoding="utf-8")]
    conc = {}
    with io.open(os.path.join(MODELS, model, "full",
                              f"{phase}.shapes.concrete.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            conc[r["op_id"]] = r
    uf = AC.build(rows, conc, mode="provenance")
    out = {}
    with gzip.open(os.path.join(LAB, "crosswalk", f"{model}.{phase}.jsonl.gz"),
                   "rt", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
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
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    stage1 = "--stage1" in flags
    lineage_mode = "none"
    for f in flags:
        if f.startswith("--lineage="):
            lineage_mode = f.split("=", 1)[1]
    shard_size = int(argv[0]) if argv else 12
    seed = int(argv[1]) if len(argv) > 1 else 20260925
    dest = PRIORITY_BUNDLE if stage1 else BUNDLE
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

    # ---- 1 단계 대상만 남긴다. **선정 이유·등급은 읽고 버린다** (shard 에 넣지 않는다)
    stage1_ids = []
    if stage1:
        sp = os.path.join(LAB, "priority", "stage1_units.jsonl")
        INPUTS.append(sp)
        stage1_ids = [json.loads(l)["decision_unit_id"]
                      for l in io.open(sp, encoding="utf-8")]
        keep = set(stage1_ids)
        units = [u for u in units if u["decision_unit_id"] in keep]
        assert len(units) == len(keep), f"{len(units)} != {len(keep)}"

    # ---- lineage: 기본은 쓰지 않는다. 포트 provenance 가 없으므로 `X_linked` 도 없다.
    lin, cov = {}, {}
    for (m, ph) in pubs:
        cov[f"{m}.{ph}"] = port_coverage(m, ph)
    if lineage_mode != "none":
        for (m, ph) in pubs:
            print(f"   lineage[{lineage_mode}] {m} {ph} …", flush=True)
            lin[(m, ph)] = _lineage(m, ph, lineage_mode)
            INPUTS.extend([os.path.join(MODELS, m, "full", f"{ph}.ports.jsonl"),
                           os.path.join(MODELS, m, "full", f"{ph}.semantic.jsonl")])

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

    # ---- 10% 중복 배정: **같은 단위를 다른 세션에** 준다. shard 문서에는 중복이라는
    #      표시를 넣지 않는다(이전 답도, 중복 여부도 알리지 않는다 -- 외부 검토 2026-09-25).
    dup_map = {}
    if stage1:
        dup = rnd.sample(ordered, max(1, round(0.10 * len(ordered))))
        dup.sort(key=lambda u: u["decision_unit_id"])
        rnd.shuffle(dup)
        base = len(shards)
        for i in range(0, len(dup), shard_size):
            shards.append(dup[i:i + shard_size])
        for si in range(base, len(shards)):
            for u in shards[si]:
                dup_map.setdefault(u["decision_unit_id"], []).append(
                    f"shard{si + 1:03d}.md")

    # ---- **임시 디렉터리에 새로 만들고 원자적으로 교체한다.** 제자리에서 갱신하면
    #      옛 사본이 남고 manifest 의 해시와 어긋날 수 있다(외부 검토 2026-09-25).
    TMP = dest + ".tmp"
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
    for si, shard_units in enumerate(shards, 1):
        L = [f"# shard {si:03d} / {len(shards)} — 축 판정 ({len(shard_units)} 단위)", "",
             "각 단위의 **`X` 로 표시된 축이 무엇인지** 답해 주세요.", "",
             "* `X`        판정 대상 축",
             "* `Y1`, `Y2` 함께 가린 그 밖의 자리. **서로 같다는 보장이 전혀 없습니다** --",
             "             값이 같아 보여도 다른 축일 수 있으니 각각 따로 판단하세요.",
             "             같은 축인지 아닌지는 알려 드리지 않습니다",
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
        for u in shard_units:
            # `secret` 은 **가리기에만** 쓴다. 문서에 넣지 않는다.
            secret = u["signature"]["old_expr"]
            view = _public(u, stage1)
            assert not _has_private(view), u["decision_unit_id"]
            m, ph = view["model"], view["_phase"]
            if m not in seen_models:
                seen_models.append(m)
            sig = view["signature"]
            rx = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(secret) + r"(?![A-Za-z0-9_])")
            oid, field, sidx, ax = tuple(view["published_cells"][0])
            prow = pubs[(m, ph)][oid]
            cs = view.get("concrete_shapes") or {}
            L += [f"### {view['decision_unit_id']}", "",
                  f"* 모델 / phase: `{m}` / `{ph}`",
                  f"* block: `{sig['block_type']}`   층 cohort: `{sig['layer_cohort_id']}`",
                  f"* 모듈: `{sig['module']}`   연산: `{sig['op_type']}` (`{sig['raw_op']}`)",
                  f"* 판정할 자리: **{FIELD_KO[field]} shape[{sidx}] 의 축 {ax}**",
                  f"* 이 축의 concrete 값: **{view.get('concrete_value')}**"]
            if "affects_published_cells" in view:
                L.append(f"* 이 단위가 대표하는 발행 셀 "
                         f"{view['affects_published_cells']:,} / "
                         f"raw 자리 {view['represents_raw_sites']:,}")
            L.append("")
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
        manifest.append({"shard": f"shard{si:03d}.md", "units": len(shard_units),
                         "unit_ids": [u["decision_unit_id"] for u in shard_units],
                         "models": seen_models,
                         "affects_published_cells": sum(
                             u["affects_published_cells"] for u in shard_units),
                         "candidate_seed": rnd.randint(1, 10 ** 9),
                         "status": "pending"})
    # ---- **선언이 아니라 검사다.** 금지 경로·비밀 노출·ID 충돌을 실제로 센다.
    secrets_by_uid = {u["decision_unit_id"]: u["signature"]["old_expr"]
                      for u in units}
    checks = {"forbidden_paths": [], "secret_exposed_units": [],
              "candidate_text_in_unit_blocks": [],
              "private_keys_in_render_input": 0, "duplicate_unit_ids": 0,
              "shard_unit_total": sum(len(x) for x in shards)}
    # **구조적 검사.** 렌더 입력(public DTO)에 금지 키가 남아 있으면 문구 검색과 무관하게
    # 실패한다 -- 후보가 다른 제목·형식으로 새는 경로를 문구로 막을 수는 없다.
    for u in units:
        if _has_private(_public(u, stage1)):
            checks["private_keys_in_render_input"] += 1
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
        body = doc.split("## 심볼 정의와 config 값")[0]
        # 고정 머리글 뒤의 **단위 블록만** 본다. 머리글에는 `rejected_candidates` 처럼
        # "후보" 가 정당하게 들어가지만, 단위 블록에는 어떤 형태로도 들어갈 이유가 없다.
        blocks = body.split(chr(10) + "### ")[1:]
        for blk in blocks:
            if "후보" in blk:
                checks["candidate_text_in_unit_blocks"].append(
                    f"{sp}:{blk.splitlines()[0]}")
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
                 "bundle_kind": "stage1_assignment" if stage1 else "population",
                 "shards": len(shards), "units": len(units),
                 "sources": src_meta, "manifest": manifest,
                 "salt_fingerprint": _buildguard.salt_fingerprint(_salt_bytes()),
                 # lineage 를 **쓰지 않았음**을 산출물에 남긴다. 되살릴 때 무엇이
                 # 충족돼야 하는지도 함께 적는다(외부 검토 2026-09-25).
                 "lineage_mode": lineage_mode,
                 "lineage_placeholder_x_linked_used": bool(lin),
                 "port_coverage": cov,
                 "legacy_fallback_count": 0,
                 "lineage_revival_requires": [
                     "mode=provenance 명시", "legacy/migration/hybrid fallback 금지",
                     "port coverage == 1.0", "ports·semantic sidecar 를 입력 해시에 포함"],
                 "checks": {k: (len(v) if isinstance(v, list) else v)
                            for k, v in checks.items()},
                 "check_detail": {k: v[:10] for k, v in checks.items()
                                  if isinstance(v, list) and v},
                 "excluded_from_bundle": [
                     "work/units/", "work/crosswalk/", "work/_private/",
                     "_family_registry.jsonl", "models/*.csv", "models/*.jsonl",
                     "생성 코드"]})
    meta["input_sha256"] = _buildguard.input_manifest(INPUTS)
    # **입력 경로만** 본다. 예전에는 워크트리 전체를 봐서 출력(`work/review_bundle/`)
    # 때문에 항상 1 이었고 입력 오염을 뜻하지 않았다(외부 검토 2026-09-25).
    meta["input_worktrees"] = _buildguard.input_worktrees([
        ("tracer", PROJ, ("models/", "src/", "develop/", "rules/")),
        ("results-labeled", os.path.dirname(LAB),
         ("work/units/", "work/crosswalk/", "work/priority/"))])
    if stage1:
        bm = os.path.join(BUNDLE, "_manifest.json")
        prev = os.path.join(dest, "_manifest.json")
        rev = 1
        if os.path.exists(prev):
            rev = int((json.load(io.open(prev, encoding="utf-8")) or {}).get(
                "assignment_revision", 0)) + 1
        meta.update({
            "assignment_revision": rev,
            "source_bundle_manifest_sha256": _buildguard.sha256_file(bm),
            "stage1_unit_ids": stage1_ids,
            "duplicate_assignment": dup_map,
            "duplicate_ratio": round(len(dup_map) / max(1, len(units)), 4),
            "not_given_to_reviewer": [
                "work/priority/stage1_units.jsonl", "work/priority/_priority.json",
                "grade", "선정 이유", "후보", "모집단 shard 번호", "중복 배정 여부"]})
        payload = json.dumps(meta, ensure_ascii=False, sort_keys=True)
        meta["manifest_payload_sha256"] = _buildguard.sha256_bytes(payload.encode())
    json.dump(meta, io.open(os.path.join(TMP, "_manifest.json"), "w",
                            encoding="utf-8", newline=chr(10)),
              ensure_ascii=False, indent=1)

    bad = (checks["forbidden_paths"] or checks["secret_exposed_units"]
           or checks["candidate_text_in_unit_blocks"]
           or checks["private_keys_in_render_input"]
           or checks["duplicate_unit_ids"]
           or checks["shard_unit_total"] != len(units) + sum(
               len(v) for v in dup_map.values()))
    if bad:
        print("**검사 실패 -- bundle 을 교체하지 않는다**", file=sys.stderr)
        print(json.dumps(meta["checks"], ensure_ascii=False), file=sys.stderr)
        raise SystemExit(3)

    _buildguard.swap_dir(TMP, dest)
    print(f"단위 {len(units):,} -> shard {len(shards)} 개 (shard 당 {shard_size})")
    if stage1:
        print(f"중복 배정 {len(dup_map)} 단위 (배정본 {sum(len(v) for v in dup_map.values())})")
    print(f"lineage_mode {lineage_mode}  X_linked "
          + ("사용" if lin else "**미사용**"))
    print(f"frozen source {sum(len(v) for v in src_meta.values())} 파일")
    print("검사 " + json.dumps(meta["checks"], ensure_ascii=False))
    print("입력 해시 " + str(len(meta["input_sha256"])) + " 파일")
    print("입력 워크트리 미커밋 " + json.dumps(
        {k: len(v["dirty_input_paths"]) for k, v in meta["input_worktrees"].items()},
        ensure_ascii=False))
    print(f"-> {os.path.relpath(dest, PROJ)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
