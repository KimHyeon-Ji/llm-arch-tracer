r"""독립 reference matcher — expected footprint 를 **적용기와 다른 방식으로** 만든다.

왜 있는가
---------
외부 검토(2026-09-27): "expected footprint 를 만든 selector 와 actual 적용기가 같은 구현을
공유한다면 공통 버그가 날 수 있으므로, 최소한 fixture 또는 독립 reference matcher 로 한 번
더 확인해야 합니다."

그래서 이 파일은 **overlay 의 `match:` 문법을 해석하지 않는다.** 규칙의 뜻을 사람이 읽고
직접 술어로 쓴다. overlay 에서 가져오는 것은 `sub_id`, `from`, `to` 뿐이다.

  적용기  develop/plus_at_apply.py     -- overlay 의 match 블록을 일반 해석
  이 파일                              -- 규칙을 손으로 구현한 술어

두 결과가 다르면 둘 중 하나가 틀렸다. 어느 쪽인지는 fixture 가 가른다
(develop/fixtures/plus_at/cases.yaml -- 사람이 직접 쓴다).

**이 파일은 이 모델의 규칙 전용이다.** 일반화하지 않는다 -- 일반화하면 적용기와 같아진다.

실행:
    .venv\Scripts\python.exe develop\plus_at_refmatch.py <모델디렉터리> <overlay.yaml> [출력.jsonl]
"""
import hashlib
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import plus_at_canon as C                                      # noqa: E402
import yaml                                                    # noqa: E402

FIELDS = ("input_shape", "output_shape")       # 계열 A/B 는 weight 를 안 건드린다


# --------------------------------------------------------------------- 술어
def _self_attn(mp):
    return mp.endswith(".self_attn")


def _expert_idx(mp):
    """`...block_sparse_moe.experts.<n>` 의 n. 아니면 None."""
    m = re.search(r"\.block_sparse_moe\.experts\.(\d+)(?:\.|$)", mp or "")
    return int(m.group(1)) if m else None


def _moe_root(mp):
    return (mp or "").endswith(".block_sparse_moe")


def cells_kda_nchunk(phase, rows):
    """계열 A -- KDA 청크 축.

    self_attn 의 `exp` op 에서 shape 가 정확히 [B, n_h_kda, 5, d_chunk, d_head_kda] 인
    input/output 의 축 2. prefill 만.
    """
    if phase != "prefill":
        return
    want = ["B", "n_h_kda", "5", "d_chunk", "d_head_kda"]
    for r in rows:
        if r.get("op_type") != "exp" or not _self_attn(r.get("module_path") or ""):
            continue
        for field in FIELDS:
            # **피연산자 0 만.** overlay 가 shape_index: 0 으로 못 박았다. exp 는 단항이라
            # 실제 자리에는 차이가 없지만, 규칙을 넓게 두면 두 구현이 갈린다 -- fixture 가
            # 바로 그 divergence 를 잡았다(2026-09-27). 좁은 쪽으로 맞춘다.
            for si, sh in enumerate(C.parse_jsonl_shape(r.get(field), field)):
                if si != 0:
                    continue
                if sh == want:
                    yield (phase, int(r["op_id"]), field, si, 2), "5", "n_chunk"


def cells_moe_expert(phase, rows, last_index):
    """계열 B -- 전문가 자신의 토큰 축.

    experts.<n> 아래 모든 op 의 input/output 축 0 이 라우팅 토큰 수다.
    n < last_index -> regular,  n == last_index -> last.
    """
    lit = "3840" if phase == "prefill" else "12"
    for r in rows:
        n = _expert_idx(r.get("module_path") or "")
        if n is None:
            continue
        to = "n_trace_last" if n == last_index else "n_trace_regular"
        for field in FIELDS:
            for si, sh in enumerate(C.parse_jsonl_shape(r.get(field), field)):
                if sh and sh[0] == lit:
                    yield (phase, int(r["op_id"]), field, si, 0), lit, to


def cells_moe_concat(phase, rows, n_operands_last):
    """계열 B -- block_sparse_moe 자신의 concat 이 전문가 출력을 이어 붙이는 자리.

    피연산자 순서가 전문가 순서다(`_t.cat(outputs, dim=0)`). 마지막 피연산자만 last.
    출력 축은 이미 B*k*T / B*k 로 렌더돼 있어 건드리지 않는다.
    """
    lit = "3840" if phase == "prefill" else "12"
    for r in rows:
        if r.get("op_type") != "concat" or not _moe_root(r.get("module_path") or ""):
            continue
        shapes = C.parse_jsonl_shape(r.get("input_shape"), "input_shape")
        if not shapes:
            continue
        last_si = len(shapes) - 1
        for si, sh in enumerate(shapes):
            if not sh or sh[0] != lit:
                continue
            to = "n_trace_last" if si == last_si else "n_trace_regular"
            yield (phase, int(r["op_id"]), "input_shape", si, 0), lit, to


RULES = {
    "k3-kda-nchunk": lambda ph, rows, ov: cells_kda_nchunk(ph, rows),
    "k3-moe-regular": lambda ph, rows, ov: (
        c for c in cells_moe_expert(ph, rows, ov["last_index"])
        if c[2] == "n_trace_regular"),
    "k3-moe-last": lambda ph, rows, ov: (
        c for c in cells_moe_expert(ph, rows, ov["last_index"])
        if c[2] == "n_trace_last"),
    "k3-moe-concat": lambda ph, rows, ov: cells_moe_concat(ph, rows, None),
}


def canonical_record(key, before, after, row, sub_id):
    phase, op_id, field, si, ax = key
    return {"phase": phase, "op_id": op_id, "field": field, "shape_index": si,
            "axis": ax, "before": before, "after": after,
            "module_path": row.get("module_path") or "",
            "op_type": row.get("op_type") or "", "sub_id": sub_id}


def canonical_bytes(records):
    """digest 대상. 키 순서·정렬을 고정한다."""
    def k(r):
        return (r["phase"], r["op_id"], r["field"], r["shape_index"], r["axis"])
    lines = [json.dumps(r, ensure_ascii=False, sort_keys=True)
             for r in sorted(records, key=k)]
    return ("\n".join(lines) + "\n").encode("utf-8")


def build(model_dir: str, overlay_path: str):
    ov = yaml.safe_load(io.open(overlay_path, encoding="utf-8"))
    c_trace = int(ov["symbols"]["C_trace"]["value"])
    ctx = {"last_index": c_trace - 1}
    subs = {s["sub_id"]: s for s in ov["substitutions"]}
    unknown = set(subs) - set(RULES)
    if unknown:
        raise SystemExit(f"이 matcher 가 모르는 sub_id: {sorted(unknown)} "
                         f"-- 규칙을 손으로 추가해야 한다")
    out, seen = [], {}
    for phase in ("prefill", "decode"):
        jp = os.path.join(model_dir, f"{phase}.jsonl")
        if not os.path.exists(jp):
            continue
        rows = C.read_jsonl_rows(jp)
        by_id = {int(r["op_id"]): r for r in rows}
        for sub_id, spec in subs.items():
            if phase not in (spec.get("phases") or []):
                continue
            for key, before, after in RULES[sub_id](phase, rows, ctx):
                if key in seen:
                    raise SystemExit(
                        f"두 규칙이 같은 셀을 노린다: {key}  "
                        f"{seen[key]} vs {sub_id}")
                seen[key] = sub_id
                out.append(canonical_record(key, before, after, by_id[key[1]], sub_id))
    return out


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if len(sys.argv) < 3:
        print(__doc__.strip().splitlines()[-1])
        return 2
    md, ovp = sys.argv[1], sys.argv[2]
    outp = sys.argv[3] if len(sys.argv) > 3 else None
    recs = build(md, ovp)
    blob = canonical_bytes(recs)
    digest = hashlib.sha256(blob).hexdigest()
    per = {}
    for r in recs:
        per[(r["sub_id"], r["phase"])] = per.get((r["sub_id"], r["phase"]), 0) + 1
    print(f"expected footprint  셀 {len(recs)}   digest {digest[:16]}")
    for k in sorted(per):
        print(f"  {k[0]:<16} {k[1]:<8} {per[k]:>6}")
    if outp:
        with io.open(outp, "wb") as f:
            f.write(blob)
        print(f"wrote {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
