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
import io
import json
import os
import random
import re
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


def _mask(shape, secret, target_ax=None, rx=None):
    out = []
    for ax, e in enumerate(shape):
        t = str(e)
        if t == secret:
            out.append("X" if ax == target_ax else "X_same")
        else:
            out.append(rx.sub("X", t))
    return out


def _shapes(row, field):
    v = row.get(FIELD_KEY[field]) if isinstance(row, dict) else None
    if not v:
        return []
    return [v] if (field == "w" and not isinstance(v[0], list)) else v


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
            for u in (json.loads(l) for l in io.open(up, encoding="utf-8")):
                if u["affects_published_cells"] > 0:
                    u["_phase"] = ph
                    units.append(u)

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

    # ---- frozen source 사본 + 해시
    os.makedirs(os.path.join(BUNDLE, "source"), exist_ok=True)
    src_meta = {}
    for m in PUBLISHED:
        paths = [os.path.join(PROJ, p) for p in SOURCES.get(m, [])]
        if m == "moonshotai__Kimi-K3":
            paths += _kimi_remote(m)
        for p in paths:
            if not os.path.exists(p):
                continue
            name = f"{m.split('__')[-1][:10]}__{os.path.basename(p)}"
            dst = os.path.join(BUNDLE, "source", name)
            if not os.path.exists(dst):
                io.open(dst, "wb").write(io.open(p, "rb").read())
            src_meta.setdefault(m, []).append(
                {"bundle_path": f"source/{name}",
                 "original": os.path.relpath(p, PROJ) if p.startswith(PROJ) else p,
                 "sha256": _buildguard.sha256_file(p),
                 "lines": sum(1 for _ in io.open(p, encoding="utf-8", errors="replace"))})

    os.makedirs(os.path.join(BUNDLE, "shards"), exist_ok=True)
    manifest = []
    for si, sh in enumerate(shards, 1):
        L = [f"# shard {si:03d} / {len(shards)} — 축 판정 ({len(sh)} 단위)", "",
             "각 단위의 **`X` 로 표시된 축이 무엇인지** 답해 주세요.", "",
             "* `X`       판정 대상 축",
             "* `X_same`  같은 값/식이 그 shape 의 다른 자리에도 있어 함께 가린 자리",
             "* `X` 가 합성식 안에 있으면(`X*d_model`) 그 식 안에 대상이 들어 있다는 뜻입니다",
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
            L += ["shape (가린 형태):", "", "```"]
            for f in ("i", "w", "o"):
                shs = _shapes(prow, f)
                if not shs:
                    continue
                parts = ["[" + ", ".join(_mask(s, secret,
                                               ax if (f == field and j == sidx) else None, rx))
                         + "]" for j, s in enumerate(shs)]
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
        io.open(os.path.join(BUNDLE, "shards", f"shard{si:03d}.md"), "w",
                encoding="utf-8", newline="\n").write("\n".join(L))
        manifest.append({"shard": f"shard{si:03d}.md", "units": len(sh),
                         "unit_ids": [u["decision_unit_id"] for u in sh],
                         "models": seen_models,
                         "affects_published_cells": sum(
                             u["affects_published_cells"] for u in sh),
                         "candidate_seed": rnd.randint(1, 10 ** 9),
                         "status": "pending"})
    meta.update({"shard_size": shard_size, "shuffle_seed": seed,
                 "shards": len(shards), "units": len(units),
                 "sources": src_meta, "manifest": manifest,
                 "excluded_from_bundle": [
                     "work/units/", "work/crosswalk/", "work/_private/",
                     "_family_registry.jsonl", "models/*.csv", "models/*.jsonl",
                     "생성 코드"]})
    json.dump(meta, io.open(os.path.join(BUNDLE, "_manifest.json"), "w",
                            encoding="utf-8", newline="\n"),
              ensure_ascii=False, indent=1)
    print(f"단위 {len(units):,} -> shard {len(shards)} 개 (shard 당 {shard_size})")
    print(f"frozen source {sum(len(v) for v in src_meta.values())} 파일")
    print(f"-> {os.path.relpath(BUNDLE, PROJ)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
