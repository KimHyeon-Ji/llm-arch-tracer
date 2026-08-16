"""각 인계 축이 **어디서 생겼는지**(origin) 를 거슬러 찾는다. 아무것도 쓰지 않는다.

WHY THIS EXISTS
---------------
위치만으로는 역할을 못 가린다. `[B, n_h_lin_v, T, d_head_lin_k]` 의 마지막 축은 그 텐서가
query/key 면 K, value 면 V 인데 **shape 은 셋 다 똑같다**. 실제로 이 착각이 출하됐다:
Qwen3-Next 의 `transpose nth 2` 를 value 로 읽은 판정이 query 를 뒤집어 216축을 거꾸로
고치고 있었다(2026-08-16, 외부 검토가 소스의 서수를 다시 세어 잡음).

그래서 규칙은 "rank+축"이 아니라 **"역할+rank+축"** 이어야 하고, 역할은 그 축이 처음
만들어진 자리 -- origin -- 가 정한다.

무엇을 origin 으로 보는가
------------------------
축 등가류 안에서 **출력으로 처음 나타난 자리**다. `axis_classes` 의 등가류는 생산자 출력과
소비자 입력을 이미 이어 두었으므로, 그 안에서 `field == 'o'` 인 가장 이른 op 이 그 축을 만든
연산이다. (`unsettled` 의 `anchor` 는 `min(sites)` 라 같은 op_id 에서 `'i'` 가 먼저 잡힐 수
있다 -- 지목용이지 출처용이 아니다.)

origin 종류는 외부 검토가 정리한 다섯 가지로 나눈다:

    partition        split/chunk 출력이 역할을 정한다 (q/k/v, hidden/B/C, gate, dt)
    axis_constructor view/reshape 가 head·head폭·chunk 축을 새로 만든다
    replication      repeat_interleave/expand 가 n_h_lin_k -> n_h_lin_v 로 바꾼다
    factory          zeros/ones/arange/파라미터/캐시 -- 경계에서 태어난 축
    multi_parent     matmul/broadcast 처럼 출력 축마다 부모가 다른 연산

`multi_parent` 는 **단일 origin 을 붙이면 안 되는 자리**라서 따로 표시한다.

실행:
    .venv\\Scripts\\python.exe develop\\axis_origin.py --model Qwen3-Next
    .venv\\Scripts\\python.exe develop\\axis_origin.py --model Qwen3-Next --verify
    .venv\\Scripts\\python.exe develop\\axis_origin.py            # 전 모델 요약
"""
import argparse
import collections
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
MODELS = os.path.join(PROJ, "models")
sys.path.insert(0, os.path.join(PROJ, "src"))

import axis_classes as AC          # noqa: E402
import build_table as BT           # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

_PARTITION = {"split_with_sizes", "split", "chunk", "unbind"}
_CONSTRUCTOR = {"view", "reshape", "_unsafe_view"}
_REPLICATION = {"repeat_interleave", "expand", "repeat", "broadcast_to"}
_FACTORY = {"zeros", "ones", "full", "empty", "arange", "eye", "rand", "randn",
            "zeros_like", "ones_like", "scalar_tensor", "lift_fresh"}
_MULTI = {"matmul", "batched_matmul", "linear", "bmm", "einsum", "mm", "addmm"}


def origin_kind(op_type: str) -> str:
    if op_type in _PARTITION:
        return "partition"
    if op_type in _CONSTRUCTOR:
        return "axis_constructor"
    if op_type in _REPLICATION:
        return "replication"
    if op_type in _FACTORY:
        return "factory"
    if op_type in _MULTI:
        return "multi_parent"
    return "other"


def trace(model: str, phase: str):
    """{class_root: origin} 과 보조 자료. 산출물은 읽기만 한다."""
    d = os.path.join(MODELS, model)
    raw = os.path.join(d, "full", f"{phase}.trace.raw.jsonl")
    if not os.path.exists(raw):
        return None
    rows = [json.loads(l) for l in io.open(raw, encoding="utf-8")]
    conc = BT.load_concrete(d, phase) or {}
    uf = AC.build(rows, conc)
    ordinals = AC.op_ordinals(rows)
    byid = {r["op_id"]: r for r in rows}

    # 등가류마다 '출력으로 처음 나타난 자리'
    first = {}
    for r in rows:
        oid = r["op_id"]
        for si, sh in enumerate(r.get("output_shape") or []):
            if not isinstance(sh, list):
                continue
            for ax in range(len(sh)):
                root = uf.find((oid, "o", si, ax))
                cur = first.get(root)
                if cur is None or (oid, si, ax) < cur:
                    first[root] = (oid, si, ax)
    org = {}
    for root, (oid, si, ax) in first.items():
        r = byid[oid]
        org[root] = {"op_id": oid, "op_type": r.get("op_type"),
                     "module": AC.module_key_of(r.get("module_path") or ""),
                     "nth": ordinals.get(oid), "output_index": si, "axis": ax,
                     "kind": origin_kind(r.get("op_type")),
                     "out_shape": [str(x) for x in (r.get("output_shape") or [[]])[si]]
                                  if si < len(r.get("output_shape") or []) else []}
    return {"rows": rows, "uf": uf, "ordinals": ordinals, "origin": org}


def item_origins(model: str, phase: str, tr: dict) -> list:
    """인계 항목마다 그 항목이 덮는 등가류들의 origin 을 모은다.

    한 항목의 등가류들이 **서로 다른 origin** 을 가리키면 그 초안은 역할을 넘나든다는 뜻이다 --
    규칙으로 일괄 종결하면 안 되는 자리다.
    """
    p = os.path.join(MODELS, model, "full", f"{phase}.unsettled.json")
    if not os.path.exists(p):
        return []
    items = (json.load(io.open(p, encoding="utf-8")) or {}).get("items") or []
    rows, uf, ordn, org = tr["rows"], tr["uf"], tr["ordinals"], tr["origin"]
    out = []
    for it in items:
        st = it["override_stub"]
        rx = re.compile(st["module"])
        fld = "input_shape" if st["field"] == "i" else "output_shape"
        kinds = collections.Counter()
        for r in rows:
            if not rx.search(AC.module_key_of(r.get("module_path") or "") or ""):
                continue
            if r.get("op_type") != st["op_type"] or ordn.get(r["op_id"]) != st["nth"]:
                continue
            shs = r.get(fld) or []
            si = st["shape_index"]
            if si >= len(shs) or not isinstance(shs[si], list):
                continue
            if [str(x) for x in shs[si]] != [str(x) for x in st["shape"]]:
                continue
            if st["axis"] >= len(shs[si]):
                continue
            o = org.get(uf.find((r["op_id"], st["field"], si, st["axis"])))
            if o:
                kinds[(o["op_type"], o["nth"], o["output_index"], o["kind"],
                       o["module"])] += 1
        out.append((it, kinds))
    return out


# 외부 검토가 소스에서 확정한 종점. 도구가 **소스와 같은 말을 하는지** 대조하는 데 쓴다.
GROUND_TRUTH = {
    "Qwen__Qwen3-Next-80B-A3B-Instruct": {
        "note": "post-conv split 출력 0/1/2 = query/key/value "
                "(modeling_qwen3_next.py:659). pre-conv qkvz split(:566)까지 더 올라가면 안 된다.",
        "expect_partition_outputs": {0, 1, 2}},
    "tiiuae__Falcon-H1-7B-Instruct": {
        "note": "post-conv split 출력 0/1/2 = hidden/B/C "
                "(modeling_falcon_h1.py:725).",
        "expect_partition_outputs": {0, 1, 2}},
}


def main():
    ap = argparse.ArgumentParser(description="인계 축의 origin 을 거슬러 찾는다 (읽기 전용)")
    ap.add_argument("--model", default="", help="모델 이름 부분 일치")
    ap.add_argument("--verify", action="store_true",
                    help="외부 검토가 준 종점과 대조한다")
    ap.add_argument("--limit", type=int, default=14)
    a = ap.parse_args()

    names = [m for m in sorted(os.listdir(MODELS))
             if os.path.isdir(os.path.join(MODELS, m))
             and (not a.model or a.model.lower() in m.lower())]
    grand = collections.Counter()
    split_ok = collections.Counter()
    for m in names:
        per = collections.Counter()
        mixed, shown = 0, 0
        for phase in ("prefill", "decode"):
            tr = trace(m, phase)
            if not tr:
                continue
            for it, kinds in item_origins(m, phase, tr):
                if not kinds:
                    per["origin불명"] += 1
                    grand["origin불명"] += 1
                    continue
                if len(kinds) > 1:
                    mixed += 1
                    per["역할혼재"] += 1
                    grand["역할혼재"] += 1
                (op, nth, oi, kind, mod), _n = kinds.most_common(1)[0]
                per[kind] += 1
                grand[kind] += 1
                if kind == "partition":
                    split_ok[(m, oi)] += 1
                if a.model and shown < a.limit and len(kinds) == 1:
                    print(f"  {it['override_stub']['op_type']:<16}"
                          f"ax{it['override_stub']['axis']} {it['current_label']:<14}"
                          f"<- {kind}/{op}/nth{nth}/out{oi}  {it['anchor_shape']}")
                    shown += 1
        if per:
            print(f"{m.split('__')[-1][:34]:<36} {dict(per)}"
                  + (f"   역할혼재 {mixed}" if mixed else ""))

    print(f"\n전체: {dict(grand)}")
    if a.verify:
        print("\n[소스가 말한 종점과 대조]")
        for m, gt in GROUND_TRUTH.items():
            outs = {oi for (mm, oi) in split_ok if mm == m}
            if not outs:
                print(f"  {m.split('__')[-1][:30]:<32} partition origin 없음 (SKIP)")
                continue
            want = gt["expect_partition_outputs"]
            ok = outs <= want
            print(f"  {m.split('__')[-1][:30]:<32} 출력 인덱스 {sorted(outs)} "
                  f"(소스: {sorted(want)}) {'OK' if ok else 'FAIL — 소스에 없는 출력'}")
            print(f"      {gt['note']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
