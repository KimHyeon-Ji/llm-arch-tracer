r"""0-b 4단계: 1차 판정용 질문 문서를 만든다. **답을 보여주지 않는다.**

승인된 규칙(외부 검토 2026-09-25):

  반드시 준다 -- 익명 `decision_unit_id` / 모델·phase·block·layer cohort / 상대 module
  path·op type / 판정할 축을 `?` 로 표시한 shape / concrete shape 와 그 축의 concrete 값 /
  입력·출력·가중치 전체 주변 shape / 직전·직후 dataflow / parameter path / 최소 소스와
  source hash / 심볼 정의와 config 값 / 후보는 세션마다 무작위 순서 / 이 unit 이 대표하는
  raw site 수와 발행 셀 수

  숨긴다 -- 현재 `expr`·label / 현재 grade·reason / "대부분 맞을 것" 같은 기대치 /
  예시 정답 / 정답을 암시하는 기존 질문 ID·설명

**누설 차단**: 대상 축만 `?` 로 바꾸면 안 된다. 같은 식이 같은 shape 의 다른 자리나
주변 shape 에 또 나오면 거기서 답이 드러난다. 그래서 **보여주는 모든 shape 에서 그 식의
모든 등장을 가린다** (`?` = 판정 대상, `?same` = 같은 식이 나온 다른 자리).

실행:
    .venv\Scripts\python.exe develop\build_question_docs.py
"""
import collections
import hashlib
import io
import json
import os
import random
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJ, "src"))
if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import yaml                                                      # noqa: E402

MODELS = os.path.join(PROJ, "models")
LAB = os.path.join(PROJ, "..", "llm-arch-tracer-results-labeled", "work")
UNITS = os.path.join(LAB, "units")
OUT = os.path.join(LAB, "ask")

PUBLISHED = (
    "deepseek-ai__DeepSeek-V4-Pro",
    "meta-llama__Llama-4-Maverick-17B-128E",
    "moonshotai__Kimi-K3",
    "openai__gpt-oss-120b",
    "openai__gpt-oss-20b",
)

# 모델마다 실행된 구현체 (work/SOURCES.md 와 같은 목록). 해시를 함께 싣는다.
SOURCE_FILES = {
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
        ".venv/Lib/site-packages/fla/ops/kda/naive.py"],
}
FIELD_KEY = {"i": "input_shape", "o": "output_shape", "w": "weight_shape"}
FIELD_KO = {"i": "입력", "o": "출력", "w": "가중치"}


def _sha256(path):
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with io.open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _shapes(prow, field):
    v = prow.get(FIELD_KEY[field])
    if not v:
        return []
    return [v] if (field == "w" and v and not isinstance(v[0], list)) else v


_TOK = {}


def _mask(shape, secret, target_ax=None):
    """보여줄 shape. `secret` 의 모든 등장을 가린다 (누설 차단).

    **합성식 안의 토큰까지 가린다.** 정확히 같은 식만 가리면
    `n_hc*d_model` 같은 주변 축에서 답이 드러난다 -- V4-Pro 에서 실제로 4 건 샜다
    (2026-09-25). `?tok` 은 "대상 식이 이 합성식 안에 들어 있다" 는 표시다.
    """
    rx = _TOK.get(secret)
    if rx is None:
        rx = _TOK[secret] = re.compile(
            r"(?<![A-Za-z0-9_])" + re.escape(secret) + r"(?![A-Za-z0-9_])")
    out = []
    for ax, e in enumerate(shape):
        t = str(e)
        if t == secret:
            out.append("?" if ax == target_ax else "?same")
        else:
            out.append(rx.sub("?tok", t))
    return out


def _render_operands(prow, field, si, ax, secret):
    """입력·출력·가중치 전체를 가린 형태로 렌더."""
    lines = []
    for f in ("i", "o", "w"):
        shs = _shapes(prow, f)
        if not shs:
            continue
        parts = []
        for j, sh in enumerate(shs):
            tgt = ax if (f == field and j == si) else None
            parts.append("[" + ", ".join(_mask(sh, secret, tgt)) + "]")
        lines.append(f"  {FIELD_KO[f]:4} {', '.join(parts)}")
    return "\n".join(lines)


def build(model, rnd):
    pub_by_phase, conc_by_phase = {}, {}
    for phase in ("prefill", "decode"):
        with io.open(os.path.join(MODELS, model, f"{phase}.jsonl"), encoding="utf-8") as f:
            pub_by_phase[phase] = {r["op_id"]: r for r in (json.loads(x) for x in f)}
        c = {}
        p = os.path.join(MODELS, model, "full", f"{phase}.shapes.concrete.jsonl")
        if os.path.exists(p):
            with io.open(p, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    c[r["op_id"]] = r
        conc_by_phase[phase] = c

    st = yaml.safe_load(io.open(os.path.join(MODELS, model, "structure.yaml"),
                                encoding="utf-8"))
    syms = {k: v for k, v in (st.get("symbols") or {}).items() if v is not None}
    lab_only = st.get("symbols_label_only") or {}
    scope = st.get("scope") or {}
    src = {os.path.basename(p): _sha256(os.path.join(PROJ, p))
           for p in SOURCE_FILES.get(model, [])}

    os.makedirs(OUT, exist_ok=True)
    n_units = 0
    for phase in ("prefill", "decode"):
        up = os.path.join(UNITS, f"{model}.{phase}.units.jsonl")
        if not os.path.exists(up):
            continue
        units = [json.loads(l) for l in io.open(up, encoding="utf-8")]
        units = [u for u in units if u["affects_published_cells"] > 0]
        pub, conc = pub_by_phase[phase], conc_by_phase[phase]

        L = [f"# {model} / {phase} — 축 판정 요청 ({len(units)} 단위)", "",
             "각 단위마다 **`?` 로 표시된 축이 무엇인지** 답해 주세요.", "",
             "* `?`      판정 대상 축",
             "* `?same`  같은 값/식이 그 shape 의 다른 자리에도 있어 함께 가린 자리",
             "* `?tok`   대상 식이 그 합성식 **안에** 들어 있어 가린 자리 (예: `?tok*d_model`)",
             "  (대상 축만 가리면 다른 자리에서 답이 드러납니다)", "",
             "**현재 이 축에 붙어 있는 이름은 알려 드리지 않습니다.** 소스와 아래 정보만으로",
             "판단해 주세요. 이름이 없는 것이 맞다고 판단되면 그렇게 답해 주세요.", "",
             "## 이 모델의 심볼 정의와 config 값", "", "```",
             json.dumps(syms, ensure_ascii=False, indent=1), "```", ""]
        if lab_only:
            L += ["심볼표에 없지만 표에 쓰이는 식:", "", "```",
                  json.dumps(lab_only, ensure_ascii=False, indent=1), "```", ""]
        L += [f"추적 범위: 배치 B={scope.get('batch')}, prefill 길이 "
              f"{scope.get('prefill_len')}, decode query 1 / 캐시 "
              f"{scope.get('decode_cache_len')}", "",
              "## 볼 소스 (실행된 구현체, SHA-256 고정)", "", "```"]
        for k, v in src.items():
            L.append(f"{k:44} {v[:32] if v else '(없음)'}")
        L += ["```", "", "---", ""]

        for u in units:
            sig = u["signature"]
            secret = sig["old_expr"]
            ex = u["published_cells"][0]
            oid, field, si, ax = ex
            prow = pub.get(oid) or {}
            # **사이드카를 발행 op_id 로 조회하면 안 된다** -- raw op_id 키다. units 빌더가
            # crosswalk 의 raw_sites 로 이미 풀어 두었으므로 그것을 쓴다(2026-09-25).
            cval = u.get("concrete_value")
            cr = u.get("concrete_shapes") or {}
            # 후보 -- 세션마다 순서를 섞는다
            cands = [c for c in (sig.get("candidates") or "").split("|") if c]
            rnd.shuffle(cands)
            # dataflow
            prev = [pub.get(d) for d in (prow.get("depends_on") or [])][:3]
            nxt = [r for r in pub.values() if oid in (r.get("depends_on") or [])][:3]

            L += [f"### {u['decision_unit_id']}", "",
                  f"* 모델 / phase: `{model}` / `{phase}`",
                  f"* block_type: `{sig['block_type']}`   층: `{prow.get('layers') or '(비층)'}`",
                  f"* 모듈: `{sig['module']}`   연산: `{sig['op_type']}`  "
                  f"(`{prow.get('raw_op')}`)",
                  f"* 판정할 자리: **{FIELD_KO[field]} shape[{si}] 의 축 {ax}**",
                  f"* 이 축의 concrete 값: **{cval}**",
                  f"* 이 단위가 대표하는 발행 셀 {u['affects_published_cells']:,} 개 / "
                  f"raw 자리 {u['represents_raw_sites']:,} 개", ""]
            if prow.get("params"):
                L += [f"* parameter: `{', '.join(prow['params'])}`", ""]
            L += ["주변 shape (가린 형태):", "", "```", _render_operands(prow, field, si, ax, secret),
                  "```", ""]
            if any(cr.get(k) for k in ("i", "o", "w")):
                L += ["같은 자리의 concrete shape:", "", "```"]
                for f in ("i", "o", "w"):
                    v = cr.get(f)
                    if not v:
                        continue
                    shs = [v] if (f == "w" and not isinstance(v[0], list)) else v
                    L.append(f"  {FIELD_KO[f]:4} " + ", ".join(
                        "[" + ", ".join(str(x) for x in sh) + "]" for sh in shs))
                L += ["```", ""]
            if prev or nxt:
                L += ["dataflow:", "", "```"]
                for r in prev:
                    if r:
                        L.append(f"  직전  {r.get('op_type'):16} {r.get('module_path')}")
                for r in nxt:
                    if r:
                        L.append(f"  직후  {r.get('op_type'):16} {r.get('module_path')}")
                L += ["```", ""]
            if cands:
                L += [f"참고 후보 (무작위 순서): {', '.join(cands)}", "",
                      "이 목록에 없는 이름이나 '이름 없음' 도 답으로 가능합니다.", ""]
            L += ["---", ""]
            n_units += 1

        io.open(os.path.join(OUT, f"{model}.{phase}.ask.md"), "w",
                encoding="utf-8", newline="\n").write("\n".join(L))
    return n_units


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 20260925
    rnd = random.Random(seed)
    total = 0
    for m in PUBLISHED:
        n = build(m, rnd)
        total += n
        print(f"  {m[:34]:36} 단위 {n}")
    meta = {"seed": seed,
            "built_from_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=PROJ,
                capture_output=True, text=True).stdout.strip(),
            "built_from_tree_clean": not subprocess.run(
                ["git", "status", "--porcelain"], cwd=PROJ,
                capture_output=True, text=True).stdout.strip(),
            "total_units": total,
            "note": "후보 순서는 seed 로 섞었다. 다른 세션에는 다른 seed 를 쓴다."}
    json.dump(meta, io.open(os.path.join(OUT, "_ask_meta.json"), "w",
                            encoding="utf-8", newline="\n"),
              ensure_ascii=False, indent=1)
    print(f"\n총 {total} 단위 -> {os.path.relpath(OUT, PROJ)}  (seed {seed})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
