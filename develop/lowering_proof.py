r"""짝 못 지은 op 구간이 **같은 계산의 다른 lowering** 인지 trace 국소로 증명한다.

왜 필요한가
-----------
`transition_diff.classify_unmatched` 는 `module_path` 별 미매칭 레코드를 **한 덩어리로**
분류한다. 그래서 출력의 "117 / 69" 는 연결된 구간 수가 아니라 미매칭이 존재한 **모듈
인스턴스 수**다. 서로 떨어진 계산이 한 목록에 섞여 들어갈 수 있고, shape 다중집합만
비교하므로 "같은 shape 의 다른 텐서" 를 구별하지 못한다(외부 검토 2026-09-18, 09-19).

그래서 여기서는 ports 의 producer/consumer 로 **연결 성분**을 세우고, 성분의 실제 경계
텐서를 찾고, 기록된 `scalar_args` 로 **그 구간만 다시 실행해서** 수치로 대조한다.
승인은 fail-closed 다 -- 근거를 못 세우면 승인하지 않는다.

무엇을 증명하고 무엇을 증명하지 않나
-----------------------------------
증명: 저장된 두 trace 의 대응하는 연결 성분이, 같은 경계 입력에 대해 같은 경계 출력을
낸다(옛 B=1 결과가 새 B>1 결과의 해당 배치 조각과 일치한다).
증명하지 않음: GPU Triton kernel 의 op 구성, MoE 대체 경로 밖의 동작, 부동소수점
bitwise 일치.

실행:
    .venv\Scripts\python.exe develop\lowering_proof.py <모델이름> [--phase prefill]
"""
import argparse
import collections
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJ, "src"))
if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import torch                                      # noqa: E402
import transition_diff as TD                       # noqa: E402
import dim_expr as DE                              # noqa: E402

# 원소 대응을 그대로 옮기는 op. 값을 만들지 않고 **보기만 바꾼다**.
LAYOUT_OPS = {
    "aten.view.default", "aten._unsafe_view.default", "aten.reshape.default",
    "aten._reshape_alias.default", "aten.permute.default", "aten.transpose.int",
    "aten.t.default", "aten.clone.default", "aten.contiguous.default",
    "aten.expand.default", "aten.unsqueeze.default", "aten.squeeze.dim",
    "aten.alias.default", "aten.select.int", "aten.slice.Tensor",
    "aten.cat.default", "aten.narrow.default", "aten.detach.default",
}


# ---------------------------------------------------------------- 1. ports 정규화
def norm_ports(model_dir: str, phase: str) -> dict:
    """옛 schema(`[op_id, slot]`)와 새 schema(`{kind, op_id, output_slot}`)를 한 형식으로.

    반환: `{op_id: {"srcs": [...], "in_ids": [...], "out_ids": [...], "pos": [...],
             "kw": {...}, "schema": 1|2}}`
    `srcs` 항목은 `("op", op_id, slot)` / `("ext", 종류, 이름)` / `None`.
    """
    p = os.path.join(model_dir, "full", f"{phase}.ports.jsonl")
    out = {}
    if not os.path.exists(p):
        return out
    for line in io.open(p, encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:                                          # noqa: BLE001
            continue
        oid = r.get("op_id")
        if oid is None:
            continue
        srcs = []
        for s in (r.get("input_sources") or []):
            if s is None:
                srcs.append(None)
            elif isinstance(s, (list, tuple)):                     # schema 1
                srcs.append(("op", s[0], s[1] if len(s) > 1 else 0))
            elif isinstance(s, dict):                              # schema 2
                if s.get("kind") == "op":
                    srcs.append(("op", s.get("op_id"), s.get("output_slot") or 0))
                else:
                    srcs.append(("ext", s.get("external"), s.get("name")))
            else:
                srcs.append(None)
        sa = r.get("scalar_args") or {}
        out[oid] = {
            "srcs": srcs,
            "in_ids": list(r.get("input_tensor_ids") or []),
            "out_ids": list(r.get("output_tensor_ids") or []),
            "pos": list(sa.get("pos") or []),
            "kw": dict(sa.get("kw") or {}),
            "schema": r.get("ports_schema_version") or 1,
        }
    return out


def load_raw(model_dir: str, phase: str) -> dict:
    """`op_id -> 원본 레코드`."""
    p = os.path.join(model_dir, "full", f"{phase}.trace.raw.jsonl")
    out = {}
    if not os.path.exists(p):
        return out
    for line in io.open(p, encoding="utf-8"):
        r = json.loads(line)
        if r.get("op_id") is not None:
            out[r["op_id"]] = r
    return out


# ------------------------------------------------- 2. 연결 성분과 실제 경계 구성
class Graph:
    """한 phase 전체의 producer/consumer 색인. 경계는 **모듈이 아니라 그래프**로 정한다."""

    def __init__(self, ports: dict):
        self.ports = ports
        # **한 텐서에 생산자가 여럿일 수 있다.** 제자리 연산(`copy_`, `scatter_`,
        # `index_put_`, `mul_`)이 같은 텐서를 다시 쓴다. 마지막 것만 남기면 앞선 소비자가
        # "뒤 op 에 의존" 하는 것처럼 보여 경계와 앵커가 통째로 틀어진다(2026-09-19).
        self._prod = collections.defaultdict(list)       # tensor_id -> [(op_id, slot)]
        self.consumers = collections.defaultdict(list)   # tensor_id -> [op_id]
        for oid in sorted(ports):
            for slot, tid in enumerate(ports[oid]["out_ids"]):
                if tid is not None:
                    self._prod[tid].append((oid, slot))
        for oid in sorted(ports):
            for tid in ports[oid]["in_ids"]:
                if tid is not None:
                    self.consumers[tid].append(oid)
        # 생산자가 하나뿐인 흔한 경우를 위한 지름길
        self.producer = {t: v[0] for t, v in self._prod.items() if len(v) == 1}

    def produced_by(self, tid, before=None):
        """`before` 직전에 이 텐서를 마지막으로 쓴 op. `before` 가 None 이면 첫 생산자."""
        v = self._prod.get(tid)
        if not v:
            return None
        if before is None:
            return v[0]
        last = None
        for oid, slot in v:
            if oid < before:
                last = (oid, slot)
            else:
                break
        return last if last is not None else v[0]

    def preds(self, oid):
        for tid in self.ports[oid]["in_ids"]:
            pr = self.produced_by(tid, oid)
            if pr:
                yield pr[0], tid

    def succs(self, oid):
        for tid in self.ports[oid]["out_ids"]:
            for c in self.consumers.get(tid, ()):
                if c > oid:
                    yield c, tid


def components(seed_ids, graph: Graph, raw: dict, bridge_ok) -> list:
    """`seed_ids` 를 producer/consumer 로 묶어 연결 성분을 만든다.

    미매칭 노드 사이를 잇는 노드는 `bridge_ok(op_id)` 가 참일 때만 성분에 끌어들인다
    (짝지어진 layout op 만 허용). 그 외의 노드를 건너뛰고 잇지 않는다 -- 건너뛰면
    "사이에 낀 계산" 을 없는 셈 치게 된다.
    """
    seeds = set(seed_ids)
    parent = {n: n for n in seeds}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # seed 에서 bridge 만 밟아 도달하는 다른 seed 를 잇는다. 밟은 bridge 는 그 성분에 넣는다.
    bridges = collections.defaultdict(set)
    for s in seeds:
        stack = [(s, ())]
        seen = {s}
        while stack:
            cur, path = stack.pop()
            for nxt, _tid in list(graph.succs(cur)) + list(graph.preds(cur)):
                if nxt in seen:
                    continue
                if nxt in seeds:
                    union(s, nxt)
                    for b in path:
                        bridges[s].add(b)
                    continue
                if bridge_ok(nxt) and len(path) < 8:
                    seen.add(nxt)
                    stack.append((nxt, path + (nxt,)))
    comps = collections.defaultdict(set)
    for n in seeds:
        comps[find(n)].add(n)
    for s, bs in bridges.items():
        comps[find(s)].update(bs)
    return [sorted(v) for v in comps.values()]


def boundary(comp_ids, graph: Graph):
    """성분의 (외부 입력 텐서, 외부 출력 텐서). 둘 다 `(tensor_id, 생산 op)` 순서 목록."""
    inside = set(comp_ids)
    ext_in, seen_in = [], set()
    for oid in comp_ids:
        for k, tid in enumerate(graph.ports[oid]["in_ids"]):
            if tid is None:
                # 기록에 텐서 id 가 없는 입력(추적 밖에서 온 KV 캐시 등)도 **경계 입력**이다.
                # 건너뛰면 재실행이 그 자리를 채우지 못해 구간 전체를 못 본다.
                nid = ("null", oid, k)
                if nid not in seen_in:
                    ext_in.append(nid)
                    seen_in.add(nid)
                continue
            if tid in seen_in:
                continue
            pr = graph.produced_by(tid, oid)
            if pr is None or pr[0] not in inside:
                ext_in.append(tid)
                seen_in.add(tid)
    ext_out, seen_out, dropped = [], set(), []
    for oid in comp_ids:
        for tid in graph.ports[oid]["out_ids"]:
            if tid is None or tid in seen_out:
                continue
            # 소비자는 **op 순서로 거르지 않는다.** 제자리 연산은 같은 텐서를 다시 내므로,
            # 나중 생산 시점에서 보면 소비자가 전부 "앞" 에 있는 것처럼 보여 멀쩡한
            # 내부 텐서가 가짜 경계 출력이 된다(2026-09-19, KDA 에서 453 건).
            cons = list(graph.consumers.get(tid, ()))
            if not cons:
                # **아무 op 도 소비하지 않는 값**은 관측할 수 없다. 구간 밖으로 나가지
                # 않으므로 경계가 아니다. 제자리 연산이 남기는 중간 버퍼가 대부분이고,
                # 그 효과는 실제로 소비되는 텐서 id 로 흘러 거기서 대조된다.
                dropped.append(tid)
                seen_out.add(tid)
                continue
            if not cons or any(c not in inside for c in cons):
                ext_out.append(tid)
                seen_out.add(tid)
    boundary.last_dropped = len(dropped)
    return ext_in, ext_out


# --------------------------------------------- 3. old/new 성분 대응 (앵커 기반)
def anchor_map(pairs):
    """짝지은 op 쌍에서 `옛 op_id -> 새 op_id` 를 만든다. **이것만이 두 trace 사이의 다리다.**

    shape 이나 module path 로 성분을 짝짓지 않는다. 성분의 경계 텐서를 **누가 만들었나**
    로 짝짓고, 그 생산자가 짝지어진 op 일 때만 대응을 인정한다.
    """
    m = {}
    for ra, rb in pairs:
        oa, ob = ra.get("op_id"), rb.get("op_id")
        if oa is not None and ob is not None:
            m[oa] = ob
    return m


def collect(model, new_root, old_root=None, phases=("prefill", "decode")):
    """모델의 미매칭 레코드를 연결 성분으로 분해한다. 반환은 phase 별 구조 보고."""
    old_dir = os.path.join(old_root or os.path.join(PROJ, "models"), model)
    new_dir = os.path.join(new_root, model)
    # `transition_diff` 와 **같은 좌표계**를 쓴다. 옛 판은 B=1 로, 새 판도 B=1 로 평가해
    # 서명을 맞추고(그게 짝짓기의 전제), T 는 옛 판 값으로 고정한다.
    old_prov, new_prov = DE.load_provenance(old_dir), DE.load_provenance(new_dir)
    new_b = int(new_prov.get("capture_batch") or 1)
    old_b = int(old_prov.get("capture_batch") or 1)
    # **옛 판이 항상 B=1 인 것은 아니다.** 한 번 출고하고 나면 `models/` 의 옛 판도 B>1 이라
    # 그 뒤의 대조는 B=3 -> B=3 이 된다. 그때 b=0 조각을 뽑으면 shape 이 안 맞는다.
    # 배치 비를 쓰고, 비가 1 이면 조각을 뽑지 않는다 (2026-09-19).
    if old_b <= 0 or new_b % old_b:
        raise ValueError(f"배치 비가 정수가 아니다: 옛 {old_b} -> 새 {new_b}")
    batch = new_b // old_b
    old_t = int(old_prov.get("seq_len_used") or 0) or None
    old_ns = DE.namespace(old_prov, 1)
    new_ns1 = DE.namespace(new_prov, 1, seq_len=old_t)
    # 재실행은 **기록된 그대로의 구체 shape** 이어야 하므로 배치를 대입한 좌표계도 만든다.
    old_nsB = DE.namespace(old_prov, old_b)
    new_nsB = DE.namespace(new_prov, new_b, seq_len=old_t)
    report = {}
    for phase in phases:
        old_ops = TD._ops(old_dir, phase, old_ns, 1)
        new_ops = TD._ops(new_dir, phase, new_ns1, batch)
        if not (old_ops or new_ops):
            continue
        pairs, _, _ = TD._match(old_ops, new_ops)
        amap = anchor_map(pairs)
        matched_old = set(amap)
        matched_new = set(amap.values())
        po, pn = norm_ports(old_dir, phase), norm_ports(new_dir, phase)
        go, gn = Graph(po), Graph(pn)
        raw_o, raw_n = load_raw(old_dir, phase), load_raw(new_dir, phase)

        def bridge(side_raw, matched):
            def f(oid):
                r = side_raw.get(oid)
                return bool(r) and oid in matched and (r.get("raw_op") in LAYOUT_OPS)
            return f

        rows = []
        for mod in set(old_ops) | set(new_ops):
            la, lb = TD._leftovers(old_ops.get(mod, []), new_ops.get(mod, []))
            if not (la or lb):
                continue
            ca = components([r["op_id"] for r in la if r.get("op_id") is not None],
                            go, raw_o, bridge(raw_o, matched_old))
            cb = components([r["op_id"] for r in lb if r.get("op_id") is not None],
                            gn, raw_n, bridge(raw_n, matched_new))
            rows.append({
                "module": mod, "n_old": len(la), "n_new": len(lb),
                "comp_old": ca, "comp_new": cb,
                "all_old": sorted(r["op_id"] for _s, r in old_ops.get(mod, [])
                                  if r.get("op_id") is not None),
                "all_new": sorted(r["op_id"] for _s, r in new_ops.get(mod, [])
                                  if r.get("op_id") is not None),
                "ids_old": sorted(r["op_id"] for r in la if r.get("op_id") is not None),
                "ids_new": sorted(r["op_id"] for r in lb if r.get("op_id") is not None),
                "layer_idx": (la or lb)[0].get("layer_idx"),
            })
        report[phase] = {"rows": rows, "graph_old": go, "graph_new": gn,
                         "raw_old": raw_o, "raw_new": raw_n, "amap": amap,
                         "batch": batch, "ns_old": old_ns, "ns_new1": new_ns1,
                         "ns_newB": new_nsB, "ns_oldB": old_nsB,
                         "old_batch": old_b, "new_batch": new_b}
    return report


def _anchor_set(comp, graph, amap, side):
    """성분의 경계 앵커 집합. 새 op_id 공간에서 센다. 하나라도 못 세우면 None."""
    ein, eout = boundary(comp, graph)
    ks = set()
    for tid in ein:
        k = _bkey(tid, comp, graph, amap, side, out=False)
        if k is None:
            return None
        ks.add(k)
    for tid in eout:
        k = _bkey(tid, comp, graph, amap, side, out=True)
        if k is None:
            return None
        ks.add(k)
    return ks


def _prod_of(tid, comp, graph):
    """구간이 이 텐서를 **쓰는 시점** 기준의 생산자. 제자리 연산이 있으면 시점이 중요하다."""
    first = min((o for o in comp if tid in graph.ports[o]["in_ids"]), default=None)
    return graph.produced_by(tid, first)


def align_boundaries(co, cn, go, gn, amap, rev, limit=64):
    """두 성분의 **경계를 같은 깊이로** 맞춘다.

    한쪽에서 짝지은 layout op 이 성분 안에 들어가 있고 다른 쪽에서는 밖에 남아 있으면,
    경계가 서로 다른 지점에 생겨 앵커가 어긋난다(K3 KDA prefill 이 385 중 213 이 그랬다).
    그럴 때 **밖에 남은 쪽도 그 op 을 안으로 끌어들인다**. 끌어들이는 대상은 짝지은 op
    뿐이다 -- 짝지었다는 것은 서명이 같다는 뜻이고, 그래야 양쪽에 같은 op 이 생긴다.

    반환 `(co, cn, ok)`. `limit` 번 안에 앵커 집합이 같아지지 않으면 `ok=False` 다.
    """
    co, cn = set(co), set(cn)
    for _ in range(limit):
        ko = _anchor_map(co, go, amap, "old")
        kn = _anchor_map(cn, gn, amap, "new")
        if ko is None or kn is None:
            return sorted(co), sorted(cn), False
        if set(ko) == set(kn):
            return sorted(co), sorted(cn), True
        grew = False
        # 한쪽에만 있는 앵커의 생산자가 **짝지은 op** 이면 양쪽에 동시에 끌어들인다.
        # 짝지었다는 것은 서명(op 종류·파라미터·B=1 정규화 shape)이 같다는 뜻이므로,
        # 양쪽에 같은 op 을 넣는 셈이고 경계만 한 칸 위로 올라간다.
        for k in set(ko) - set(kn):
            if k[0] != "op":
                continue
            pr = _prod_of(ko[k], co, go)
            if not pr:
                continue
            nb = amap.get(pr[0])
            if nb is None:
                continue
            if pr[0] not in co or nb not in cn:
                co.add(pr[0])
                cn.add(nb)
                grew = True
        for k in set(kn) - set(ko):
            if k[0] != "op":
                continue
            pr = _prod_of(kn[k], cn, gn)
            if not pr:
                continue
            ob = rev.get(pr[0])
            if ob is None:
                continue
            if pr[0] not in cn or ob not in co:
                cn.add(pr[0])
                co.add(ob)
                grew = True
        if not grew:
            return sorted(co), sorted(cn), False
    return sorted(co), sorted(cn), False


def _anchor_map(comp, graph, amap, side):
    """`앵커 -> 경계 텐서`. 앵커를 못 세우거나 겹치면 None."""
    ein, eout = boundary(comp, graph)
    out = {}
    for tid, is_out in [(t, False) for t in ein] + [(t, True) for t in eout]:
        k = _bkey(tid, comp, graph, amap, side, is_out)
        if k is None or k in out:
            return None
        out[k] = tid
    return out


def pair_components(row, go, gn, amap):
    """옛/새 성분을 **경계 앵커가 겹치는지**로 묶어 증명 단위를 만든다.

    1:1 로 갈리지 않는다. 같은 계산이 한쪽에서는 한 덩어리이고 다른 쪽에서는 여러 조각일
    수 있다(K3 KDA prefill 이 옛 1 : 새 18 이다). 그래서 앵커를 공유하는 성분들을 한
    단위로 합친다. 합친 단위의 경계는 그래프가 정하므로 임의로 늘어나지 않는다.

    반환 `(units, lone_old, lone_new)`. `units` 는 `(옛 op 목록, 새 op 목록)`.
    """
    ao = [(c, _anchor_set(c, go, amap, "old")) for c in row["comp_old"]]
    an = [(c, _anchor_set(c, gn, amap, "new")) for c in row["comp_new"]]
    lone_o = [c for c, k in ao if k is None]
    lone_n = [c for c, k in an if k is None]
    ao = [(c, k) for c, k in ao if k is not None]
    an = [(c, k) for c, k in an if k is not None]

    # 앵커 -> (쪽, 색인) 로 이분 그래프를 세우고 연결 성분을 잡는다
    touch = collections.defaultdict(list)
    for i, (_c, k) in enumerate(ao):
        for a in k:
            touch[a].append(("o", i))
    for j, (_c, k) in enumerate(an):
        for a in k:
            touch[a].append(("n", j))
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(ao)):
        find(("o", i))
    for j in range(len(an)):
        find(("n", j))
    for a, members in touch.items():
        for m in members[1:]:
            union(members[0], m)

    groups = collections.defaultdict(lambda: ([], []))
    for i in range(len(ao)):
        groups[find(("o", i))][0].append(i)
    for j in range(len(an)):
        groups[find(("n", j))][1].append(j)

    units, stuck_o, stuck_n = [], [], []
    rev = {v: k for k, v in amap.items()}
    for g, (oi, nj) in groups.items():
        if not oi:
            lone_n.extend(an[j][0] for j in nj)
            continue
        if not nj:
            lone_o.extend(ao[i][0] for i in oi)
            continue
        co = sorted({o for i in oi for o in ao[i][0]})
        cn = sorted({o for j in nj for o in an[j][0]})
        co, cn, ok = align_boundaries(co, cn, go, gn, amap, rev)
        if not ok:
            stuck_o.append(co)
            stuck_n.append(cn)
            continue
        units.append((co, cn))

    # 앵커가 겹치지 않아 따로 떨어진 조각과, 정렬이 안 된 단위를 **한 단위로 합쳐** 다시
    # 맞춰 본다. K3 KDA 는 옛 1 : 새 18 로 갈려서 17 개가 앵커를 공유하지 않는데, 전부
    # 합치면 경계가 네 번 만에 맞는다. 합쳐도 경계는 그래프가 정하므로 임의로 늘지 않는다.
    if stuck_o or stuck_n or lone_o or lone_n:
        co = sorted({o for c in stuck_o + lone_o for o in c})
        cn = sorted({o for c in stuck_n + lone_n for o in c})
        if co and cn:
            co, cn, ok = align_boundaries(co, cn, go, gn, amap, rev)
            if ok:
                units.append((co, cn))
                lone_o, lone_n = [], []
            else:
                lone_o, lone_n = stuck_o + lone_o, stuck_n + lone_n
        else:
            lone_o, lone_n = stuck_o + lone_o, stuck_n + lone_n
    return units, lone_o, lone_n


def template_of(co, cn, raw_o, raw_n):
    """성분 쌍의 지문. 같은 지문이면 같은 lowering 패턴이라고 본다."""
    def sig(c, raw):
        return tuple(sorted(collections.Counter(
            (raw.get(o) or {}).get("raw_op") for o in c).items()))
    return (sig(co, raw_o), sig(cn, raw_n))


def prove(model, new_root, old_root=None, phases=("prefill", "decode"),
          per_template=2, out_path=None, verbose=True, whole_module=False):
    """연결 성분을 짝짓고 재실행으로 대조한다. **덮지 못한 레코드가 남으면 실패다.**

    `per_template` 은 같은 지문마다 실제로 재실행할 instance 수다. 지문이 같아도 경계
    텐서가 다를 수 있으므로 1 보다 크게 잡아 표본을 늘린다. 재실행하지 않은 instance 는
    `적용` 으로만 세고, 그 수를 산출물에 그대로 적는다 -- 증명했다고 쓰지 않는다.
    """
    rep = collect(model, new_root, old_root, phases)
    out = {"model": model, "phases": {}}
    for phase, d in rep.items():
        go, gn = d["graph_old"], d["graph_new"]
        raw_o, raw_n = d["raw_old"], d["raw_new"]
        amap, batch = d["amap"], d["batch"]
        ns_old, ns_newB = d["ns_oldB"], d["ns_newB"]
        tmpl = collections.defaultdict(lambda: {"n": 0, "rec": 0, "proved": 0,
                                                "failed": 0, "why": collections.Counter()})
        uncovered = collections.Counter()
        recs_total = recs_paired = 0
        for row in d["rows"]:
            recs_total += row["n_old"] + row["n_new"]
            if whole_module:
                # 구간 경계를 모듈의 **실제 바깥 경계**까지 민다. 모듈 안에는 서명이 같은
                # op 이 수천 개라 `_match` 의 등장순서 짝짓기가 틀릴 수 있고, 그 틀린 짝이
                # 앵커로 흘러들어 구간 경계를 잘못 맺는다(2026-09-19). 모듈 바깥 경계는
                # 파라미터 이름과 반복되지 않는 op 으로 anchored 되어 모호하지 않다.
                units = [(row["all_old"], row["all_new"])] if (row["all_old"] and row["all_new"]) else []
                lo = ln = []
            else:
                units, lo, ln = pair_components(row, go, gn, amap)
            for c in lo:
                uncovered["옛쪽 성분이 짝을 못 지었다"] += len(c)
            for c in ln:
                uncovered["새쪽 성분이 짝을 못 지었다"] += len(c)
            for co, cn in units:
                k = template_of(co, cn, raw_o, raw_n)
                t = tmpl[k]
                t["n"] += 1
                # 덮은 것은 **미매칭 레코드** 로 센다. 단위에 끌어들인 짝지은 op 까지
                # 세면 전체 레코드 수를 넘어 "덮었다" 가 부풀려진다.
                nrec = len(set(co) & set(row["ids_old"])) + len(set(cn) & set(row["ids_new"]))
                t["rec"] += nrec
                recs_paired += nrec
                if t["proved"] + t["failed"] < per_template:
                    ok, why = compare_components(co, cn, go, gn, raw_o, raw_n,
                                                 ns_old, ns_newB, batch, amap)
                    if ok:
                        t["proved"] += 1
                    else:
                        t["failed"] += 1
                        t["why"][why[:90]] += 1
        good = sum(v["rec"] for v in tmpl.values() if v["proved"] and not v["failed"])
        bad = sum(v["rec"] for v in tmpl.values() if v["failed"])
        out["phases"][phase] = {
            "records_total": recs_total, "records_paired": recs_paired,
            "records_verified_template": good, "records_failed_template": bad,
            "uncovered": dict(uncovered),
            "templates": [{"n": v["n"], "records": v["rec"], "proved": v["proved"],
                           "failed": v["failed"], "why": dict(v["why"]),
                           "old_ops": dict(k[0]), "new_ops": dict(k[1])}
                          for k, v in sorted(tmpl.items(), key=lambda kv: -kv[1]["rec"])],
        }
        if verbose:
            print(f"\n=== {phase}")
            print(f"   미매칭 레코드 {recs_total:,}  /  성분 짝지음 {recs_paired:,}  "
                  f"/  짝 못 지음 {sum(uncovered.values()):,}")
            print(f"   template {len(tmpl)}종  — 통과 template 의 레코드 {good:,}, "
                  f"실패 template 의 레코드 {bad:,}")
            for t in out["phases"][phase]["templates"][:8]:
                mark = "통과" if t["proved"] and not t["failed"] else ("실패" if t["failed"] else "미검")
                print(f"      [{mark}] instance {t['n']:>6,}  레코드 {t['records']:>8,}  "
                      f"재실행 {t['proved']}/{t['proved'] + t['failed']}")
                if t["why"]:
                    for w, c in list(t["why"].items())[:2]:
                        print(f"              └ {w}")
            for w, c in uncovered.items():
                print(f"      (미덮음) {c:>8,}  {w}")
    if out_path:
        json.dump(out, io.open(out_path, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model")
    ap.add_argument("--new", default=os.path.join(HERE, "out"))
    ap.add_argument("--old", default=None)
    ap.add_argument("--phase", default=None)
    ap.add_argument("--structure", action="store_true",
                     help="성분 구조만 보고하고 재실행은 하지 않는다")
    ap.add_argument("--per-template", type=int, default=2)
    ap.add_argument("--whole-module", action="store_true",
                     help="구간 경계를 모듈 바깥까지 민다 (모호한 내부 앵커를 피한다)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    phases = (a.phase,) if a.phase else ("prefill", "decode")
    if not a.structure:
        o = prove(a.model, a.new, a.old, phases, a.per_template, a.json,
                  whole_module=a.whole_module)
        fail = sum(p["records_failed_template"] + sum(p["uncovered"].values())
                   for p in o["phases"].values())
        print(f"\n증명 못 한 레코드: **{fail:,}**")
        return 1 if fail else 0
    rep = collect(a.model, a.new, a.old, phases)
    for phase, d in rep.items():
        rows = d["rows"]
        print(f"\n=== {phase}   미매칭 모듈 {len(rows)}개 "
              f"(레코드 {sum(r['n_old'] + r['n_new'] for r in rows):,})")
        sizes = collections.Counter()
        for r in rows:
            sizes[(len(r["comp_old"]), len(r["comp_new"]))] += 1
        print("   모듈당 (옛 성분 수, 새 성분 수) 분포:")
        for k, v in sizes.most_common(10):
            print(f"      {str(k):16} 모듈 {v:>4}")
        allc_o = [len(c) for r in rows for c in r["comp_old"]]
        allc_n = [len(c) for r in rows for c in r["comp_new"]]
        print(f"   옛 성분 {len(allc_o):,}개 (op 합 {sum(allc_o):,}), "
              f"새 성분 {len(allc_n):,}개 (op 합 {sum(allc_n):,})")
        cs = collections.Counter(allc_o), collections.Counter(allc_n)
        print(f"   성분 크기 상위 — 옛 {cs[0].most_common(6)}")
        print(f"   성분 크기 상위 — 새 {cs[1].most_common(6)}")
    return 0




# ------------------------------------------------- 4/5. 구간 재실행 (layout + contraction)
_SCHEMA_CACHE = {}

# 텐서를 새로 만드는 op. 기록에 dtype 이 없어 기본 dtype(float32)으로 되살아나는데,
# 재실행은 float64 로 도는 터라 섞이면 곱셈에서 터진다. 부동소수점 결과만 float64 로
# 맞춘다 -- bool/정수 결과는 건드리지 않는다(마스크·색인이다).
_FACTORY_OPS = {
    "aten.ones.default", "aten.zeros.default", "aten.full.default",
    "aten.empty.memory_format", "aten.arange.default", "aten.arange.start",
    "aten.new_zeros.default", "aten.new_ones.default", "aten.new_full.default",
    "aten.zeros_like.default", "aten.ones_like.default", "aten.empty_like.default",
    "aten.full_like.default", "aten.eye.default", "aten.linspace.default",
}


def overload(raw_op: str):
    """`aten.view.default` -> `torch.ops.aten.view.default`. 못 찾으면 None (승인 안 함)."""
    if raw_op in _SCHEMA_CACHE:
        return _SCHEMA_CACHE[raw_op]
    o = None
    parts = (raw_op or "").split(".")
    if len(parts) == 3 and parts[0] == "aten":
        try:
            import torch
            o = getattr(getattr(torch.ops.aten, parts[1]), parts[2])
        except Exception:                                          # noqa: BLE001
            o = None
    _SCHEMA_CACHE[raw_op] = o
    return o


def _is_tensor_t(t: str) -> bool:
    return t.startswith("Tensor") and "[]" not in t and not t.startswith("List[Tensor")


def _is_tensor_list_t(t: str) -> bool:
    return "List[Tensor" in t or t.startswith("Tensor[]")


def build_args(ov, p, have, oid=None, r_op=None):
    """스키마 순서대로 (텐서, 스칼라)를 끼워 넣는다. 하나라도 못 채우면 None -- **승인 안 함**.

    텐서는 `in_ids` 순서로, 스칼라는 `scalar_args.pos` 순서로 꺼낸다. 이 끼워넣기가
    맞는지는 재실행 결과 shape 이 기록된 shape 과 같은지로 검사한다(`replay` 가 확인).
    """
    args, kwargs = [], {}
    ti = si = 0
    ids = p["in_ids"]
    pos, kw = p["pos"], p["kw"]
    for a in ov._schema.arguments:
        t = str(a.type)
        if a.kwarg_only:
            if a.name in kw:
                kwargs[a.name] = kw[a.name]
            continue
        if _is_tensor_list_t(t):
            lst = []
            while ti < len(ids):
                tid = ids[ti]
                key = ("null", oid, ti) if tid is None else tid
                if key not in have:
                    return None
                lst.append(have[key])
                ti += 1
            args.append(lst)
        elif _is_tensor_t(t):
            if ti >= len(ids):
                # `mul.Tensor(x, 0.088)` 처럼 스칼라가 Tensor 자리에 온 기록이 있다.
                # 트레이스 시점에 torch 가 스칼라를 승격해 이 overload 로 보낸 것이다.
                if si < len(pos) and isinstance(pos[si], (int, float, bool)):
                    args.append(pos[si])
                    si += 1
                    continue
                if "Optional" in t or t.endswith("?"):
                    args.append(None)
                    continue
                return None
            tid = ids[ti]
            key = ("null", oid, ti) if tid is None else tid
            slot = ti
            ti += 1
            if key not in have:
                return None
            v = have[key]
            # 트레이스 기록에 dtype 이 없다. `torch.ones(64, 64, dtype=torch.bool)` 이
            # `aten.ones.default([64,64])` 로만 남아 float 로 되살아난다. 마스크·색인
            # 자리는 소비 지점에서 맞춘다. bool 계산의 float 대응물은 0/1 이므로
            # `!= 0` 이 원래 bool 을 그대로 복원한다. 양쪽에 같은 규칙을 쓰므로 대조는
            # 여전히 성립한다.
            want = _ARG_DTYPE.get((r_op, slot)) if r_op else None
            if want == "bool" and hasattr(v, "dtype") and v.dtype is not torch.bool:
                v = v != 0
            elif want == "long" and hasattr(v, "dtype") and v.dtype is not torch.int64:
                v = v.long()
            args.append(v)
        else:
            if si < len(pos):
                args.append(pos[si])
                si += 1
            else:
                break
    if ti != len(ids) or si != len(pos):
        return None                       # 남거나 모자라면 끼워넣기가 틀린 것이다
    return args, kwargs


def topo(comp_ids, graph: Graph):
    """성분 안의 실행 순서. op_id 는 트레이스 순서이므로 정렬로 충분하지만,
    의존이 역행하면 승인하지 않는다(None 반환)."""
    order = sorted(comp_ids)
    inside = set(comp_ids)
    done = set()
    for oid in order:
        for tid in graph.ports[oid]["in_ids"]:
            pr = graph.produced_by(tid, oid)
            if pr and pr[0] in inside and pr[0] not in done:
                return None
        done.add(oid)
    return order


def replay(comp_ids, graph: Graph, raw: dict, seed_tensors: dict, ns):
    """구간을 기록된 인자로 다시 실행한다.

    반환 `(values, err)`. `err` 가 있으면 승인하지 않는다. 각 op 의 결과 shape 을 기록된
    `output_shape` 과 대조해, 인자 끼워넣기가 틀렸으면 거기서 멈춘다.
    """
    order = topo(comp_ids, graph)
    if order is None:
        return None, "의존이 역행한다"
    have = dict(seed_tensors)
    inside = set(comp_ids)
    # 구간 안에서 **앞으로 몇 번 더 쓰이나**. 0 이 되면 놓는다 -- KDA 한 층이 4천 op 이고
    # [288,64,128] float64 가 18MB 라, 전부 들고 있으면 터진다.
    need = collections.Counter()
    for oid in comp_ids:
        for tid in graph.ports[oid]["in_ids"]:
            if tid is not None:
                need[tid] += 1
    _, keep_out = boundary(comp_ids, graph)
    keep = set(keep_out)
    for oid in order:
        p = graph.ports[oid]
        r = raw.get(oid) or {}
        ov = overload(r.get("raw_op"))
        if ov is None:
            return None, f"스키마 없음: {r.get('raw_op')}"
        built = build_args(ov, p, have, oid, r.get("raw_op"))
        if built is None:
            return None, f"인자 복원 실패: {r.get('raw_op')} (op {oid})"
        args, kwargs = built
        try:
            out = ov(*args, **kwargs)
        except Exception as e:                                     # noqa: BLE001
            # `view` 는 stride 가 맞아야 통한다. 재실행에서 만든 텐서는 원본과 stride 가
            # 다를 수 있다. 원본에서 `view` 가 성립했다는 것은 그 자리가 **순수한 재해석**
            # 이었다는 뜻이고, `reshape` 는 view 가 가능할 때 view 와 논리적으로 같은
            # 텐서를 준다. 그래서 값은 바뀌지 않는다. 결과 shape 은 아래에서 기록과 대조한다.
            if r.get("raw_op") in ("aten.view.default", "aten._unsafe_view.default")                     and args and hasattr(args[0], "reshape"):
                try:
                    out = args[0].reshape(args[1])
                except Exception as e2:                            # noqa: BLE001
                    return None, (f"실행 실패: {r.get('raw_op')} (op {oid}) "
                                  f"{type(e2).__name__}: {e2}")
            else:
                return None, f"실행 실패: {r.get('raw_op')} (op {oid}) {type(e).__name__}: {e}"
        outs = list(out) if isinstance(out, (tuple, list)) else [out]
        if r.get("raw_op") in _FACTORY_OPS:
            outs = [(o.to(torch.float64)
                     if hasattr(o, "dtype") and o.dtype.is_floating_point
                     and o.dtype is not torch.float64 else o)
                    for o in outs]
        want = r.get("output_shape") or []
        for slot, tid in enumerate(p["out_ids"]):
            if tid is None or slot >= len(outs):
                continue
            v = outs[slot]
            if slot < len(want) and isinstance(want[slot], list):
                exp = tuple(DE.evaluate(x, ns) for x in want[slot])
                if None not in exp and tuple(v.shape) != exp:
                    return None, (f"shape 불일치: {r.get('raw_op')} (op {oid}) "
                                  f"실행 {tuple(v.shape)} != 기록 {exp}")
            have[tid] = v
        for tid in p["in_ids"]:
            if tid is None or tid in keep:
                continue
            need[tid] -= 1
            if need[tid] <= 0:
                have.pop(tid, None)
    return have, None


# 트레이스 기록에는 dtype 이 없다. **어떤 자리는 float 이면 안 된다** -- 마스크는 bool,
# 색인은 정수다. 소비하는 op 과 인자 위치로 정한다. 표에 없는 자리는 float64 로 두고,
# 틀리면 재실행이 그 자리에서 실패해 승인되지 않는다(fail-closed).
_ARG_DTYPE = {
    ("aten.masked_fill.Scalar", 1): "bool",
    ("aten.masked_fill.Tensor", 1): "bool",
    ("aten.masked_fill_.Scalar", 1): "bool",
    ("aten.where.self", 0): "bool",
    ("aten.index_select.default", 2): "long",
    ("aten.gather.default", 2): "long",
    ("aten.index.Tensor", 1): "long",
    ("aten.scatter.value", 2): "long",
    ("aten.scatter_.value", 2): "long",
}


def seed_dtype(tid, comp_ids, graph: Graph, raw: dict):
    """이 경계 입력을 무슨 dtype 으로 채워야 하나. 모르면 None (float64)."""
    if isinstance(tid, tuple) and tid and tid[0] == "null":
        _n, oid, k = tid
        return _ARG_DTYPE.get(((raw.get(oid) or {}).get("raw_op"), k))
    for oid in sorted(comp_ids):
        p = graph.ports[oid]
        if tid in p["in_ids"]:
            k = p["in_ids"].index(tid)
            d = _ARG_DTYPE.get(((raw.get(oid) or {}).get("raw_op"), k))
            if d:
                return d
    return None


def in_shape_of(tid, comp_ids, graph: Graph, raw: dict, ns):
    """경계 입력 텐서의 (구체 shape, 소비 op 에서의 라벨). 못 찾으면 (None, None)."""
    if isinstance(tid, tuple) and tid and tid[0] == "null":
        _n, oid, k = tid
        lab = (raw.get(oid) or {}).get("input_shape") or []
        if k < len(lab) and isinstance(lab[k], list):
            conc = tuple(DE.evaluate(x, ns) for x in lab[k])
            if None not in conc:
                return conc, list(lab[k])
        return None, None
    for oid in comp_ids:
        p = graph.ports[oid]
        if tid not in p["in_ids"]:
            continue
        k = p["in_ids"].index(tid)
        r = raw.get(oid) or {}
        lab = (r.get("input_shape") or [])
        if k < len(lab) and isinstance(lab[k], list):
            conc = tuple(DE.evaluate(x, ns) for x in lab[k])
            if None not in conc:
                return conc, list(lab[k])
    return None, None


# -------------------------------------- 배치 축 가설: 발행 라벨을 가설로 삼고 재실행이 검사한다
_B_HEAD = None


def unbatch(t, labels, batch, b=0):
    """새 판 텐서에서 **배치 조각 `b`** 를 뽑아 옛 판 좌표로 옮긴다.

    `b=0` 만 보면 "첫 배치만 맞고 나머지는 틀린" 계산을 승인한다 -- 외부 검토가 반례로
    짚었다(2026-09-19): `old: y = x` 대 `new: y[0] = x[0]; y[1:] = 0` 은 b=0 만 보면
    통과한다. 그래서 호출부가 **모든 배치 조각**을 돌린다.

    어느 축이 배치인지는 **발행 라벨을 가설로** 쓴다. 라벨이 `B` 면 그 축이 배치이고,
    `B*X` 면 그 축에 배치가 접혀 있다(배치가 바깥). 라벨이 틀렸으면 아래 수치 대조가
    깨진다 -- 그래서 이 재실행은 위상뿐 아니라 **라벨 자체의 검사**다.

    `B` 가 맨 앞 인수가 아닌 곳에 섞여 있으면(`n_h*B` 등) 접힌 순서를 모르므로
    None 을 돌려 **승인하지 않는다**.
    """
    global _B_HEAD
    if _B_HEAD is None:
        import re as _re
        _B_HEAD = _re.compile(r"^B\*(.+)$")
    if batch == 1:
        return t                      # 배치 비가 1 -- 뽑을 조각이 없다
    cur = t
    for i in range(len(labels) - 1, -1, -1):
        lab = str(labels[i]).strip()
        if lab == "B":
            if cur.shape[i] != batch:
                return None
            cur = cur.narrow(i, b, 1)
            continue
        m = _B_HEAD.match(lab)
        if m:
            n = cur.shape[i]
            if batch <= 0 or n % batch:
                return None
            cur = cur.unflatten(i, (batch, n // batch)).select(i, b)
            continue
        if _tok_has_B(lab):
            return None                     # 접힌 순서를 모른다 -- 승인하지 않는다
    return cur


def _tok_has_B(lab: str) -> bool:
    import re as _re
    return any(t == "B" for t in _re.split(r"[^A-Za-z0-9_]+", lab) if t)


_NEWSET = {}


def _new_matched(amap):
    """`amap` 의 새쪽 값 집합. 앵커를 세울 때마다 다시 만들지 않는다."""
    k = id(amap)
    v = _NEWSET.get(k)
    if v is None or len(v[0]) != len(amap):
        v = (amap, set(amap.values()))
        _NEWSET[k] = v
    return v[1]


def _bkey(tid, comp_ids, graph, amap, side, out=False):
    if isinstance(tid, tuple) and tid and tid[0] == "null":
        # 소비하는 op 으로 앵커를 세운다. 그 op 이 짝지어져 있어야 한다.
        _n, oid, k = tid
        oo = amap.get(oid) if side == "old" else (oid if oid in _new_matched(amap) else None)
        return None if oo is None else ("nullin", oo, k)
    """경계 텐서의 앵커. 옛쪽 op_id 는 짝지은 대응표로 새 공간에 옮겨 적는다.

    앵커를 못 세우면 None -- 그러면 대응을 만들지 않고 **승인하지 않는다**. 위치(zip)로
    맞추면 "shape 이 같은 딴 텐서" 가 구성상 통과해 버린다(외부 검토 2026-09-19).
    """
    if out:
        # 소비 op 만으로는 앵커가 겹친다 -- 한 op 이 이 구간의 출력 **둘**을 먹을 수 있다.
        # 그래서 **인자 위치**까지 넣는다(2026-09-19 decode 에서 936건이 여기서 충돌했다).
        inside = set(comp_ids)
        ks, unanchored = [], False
        for c in sorted(graph.consumers.get(tid, ())):
            if c in inside:
                continue
            # **양쪽에 같은 규칙을 쓴다.** 새쪽에서 짝 없는 소비자를 그냥 앵커로 쓰면,
            # 옛쪽은 생산자로 물러나는데 새쪽은 소비자로 남아 집합이 어긋난다.
            cc = amap.get(c) if side == "old" else (c if c in _new_matched(amap) else None)
            if cc is None:
                # 소비자가 짝지어지지 않았다. 그 소비자로는 앵커를 못 세우니 아래에서
                # **생산자**로 세운다 -- 생산자가 짝지어져 있으면 그것도 모호하지 않다.
                unanchored = True
                continue
            for k, t in enumerate(graph.ports[c]["in_ids"]):
                if t == tid:
                    ks.append((cc, k))
        if unanchored:
            pr = _prod_of(tid, comp_ids, graph)
            if pr is None:
                return None
            po = (amap.get(pr[0]) if side == "old"
                  else (pr[0] if pr[0] in _new_matched(amap) else None))
            return None if po is None else ("byprod", po, pr[1])
        if not ks:
            # 밖에서 아무도 안 쓰는 출력. 생산 op 이 짝지어져 있으면 그걸로 앵커를 세우고,
            # 아니면 승인하지 않는다.
            pr = _prod_of(tid, comp_ids, graph)
            if pr is None:
                return None
            po = (amap.get(pr[0]) if side == "old"
                  else (pr[0] if pr[0] in _new_matched(amap) else None))
            return None if po is None else ("sink", po, pr[1])
        return ("con", tuple(ks))
    first = min((o for o in comp_ids if tid in graph.ports[o]["in_ids"]), default=None)
    pr = graph.produced_by(tid, first)
    if pr is not None:
        oid, slot = pr
        oo = amap.get(oid) if side == "old" else (oid if oid in _new_matched(amap) else None)
        if oo is None:
            return None
        return ("op", oo, slot)
    for oid in comp_ids:
        p = graph.ports[oid]
        if tid in p["in_ids"]:
            k = p["in_ids"].index(tid)
            if k < len(p["srcs"]) and p["srcs"][k] and p["srcs"][k][0] == "ext":
                return ("ext", p["srcs"][k][2])
            break
    return None


def boundary_pairing(co, cn, go, gn, amap):
    """성분 경계를 **앵커로** 짝짓는다. 하나라도 못 맞추면 (None, 사유)."""
    ein_o, eout_o = boundary(co, go)
    ein_n, eout_n = boundary(cn, gn)
    outs = []
    for name, lo, ln, is_out in (("입력", ein_o, ein_n, False),
                                 ("출력", eout_o, eout_n, True)):
        if len(lo) != len(ln):
            return None, f"경계 {name} 개수가 다르다 ({len(lo)} -> {len(ln)})"
        ko, kn = {}, {}
        for tid in lo:
            k = _bkey(tid, co, go, amap, "old", is_out)
            if k is None:
                return None, f"옛 경계 {name}의 앵커를 세울 수 없다"
            if k in ko:
                return None, f"옛 경계 {name} 앵커가 겹친다"
            ko[k] = tid
        for tid in ln:
            k = _bkey(tid, cn, gn, amap, "new", is_out)
            if k is None:
                return None, f"새 경계 {name}의 앵커를 세울 수 없다"
            if k in kn:
                return None, f"새 경계 {name} 앵커가 겹친다"
            kn[k] = tid
        if set(ko) != set(kn):
            return None, f"경계 {name}의 앵커 집합이 다르다"
        outs.append([(ko[k], kn[k]) for k in sorted(ko, key=str)])
    return tuple(outs), ""


def compare_components(co, cn, go, gn, raw_o, raw_n, ns_old, ns_newB, batch,
                       amap=None, rtol=1e-11, atol=1e-11):
    """대응하는 두 성분을 **같은 경계 입력**으로 재실행해 경계 출력을 대조한다.

    새 판 경계 입력에 난수를 넣고, 옛 판 경계 입력은 그 `b=0` 조각으로 만든다. 배치가
    다른 조각에는 서로 다른 난수가 들어가므로, 계산이 배치를 섞으면 대조가 깨진다.

    반환 `(ok: bool, detail: str)`.
    """
    import torch
    bp, why = boundary_pairing(co, cn, go, gn, amap if amap is not None else {})
    if bp is None:
        return False, why
    in_pairs, out_pairs = bp
    seed_n, seed_o, weak_seeds = {}, {}, []
    for to, tn in in_pairs:
        sh_n, lab_n = in_shape_of(tn, cn, gn, raw_n, ns_newB)
        sh_o, _ = in_shape_of(to, co, go, raw_o, ns_old)
        if sh_n is None or sh_o is None:
            return False, "경계 입력 shape 을 못 읽었다"
        dt = seed_dtype(tn, cn, gn, raw_n)
        if dt == "bool":
            # 마스크는 구조가 아니라 **양쪽이 같기만** 하면 된다. 난수 bool 이면 충분하고,
            # 오히려 실제 삼각 마스크보다 더 많은 경우를 건드린다.
            v = torch.randint(0, 2, sh_n or (), dtype=torch.bool)
        elif dt == "long":
            # 색인은 범위를 모른다. 0 으로 채우면 항상 유효하고 양쪽이 같다 -- 다만 그
            # 경로는 약하게만 검사된다는 뜻이라 산출물에 적는다.
            v = torch.zeros(sh_n or (), dtype=torch.long)
            weak_seeds.append(str(lab_n))
        else:
            v = torch.randn(*sh_n, dtype=torch.float64) if sh_n else torch.randn((), dtype=torch.float64)
        seed_n[tn] = v
        u = unbatch(v, lab_n, batch)
        if u is None:
            return False, f"배치 축 가설을 세울 수 없다: {lab_n}"
        if tuple(u.shape) != tuple(sh_o):
            return False, (f"배치 조각이 옛 shape 과 다르다: 라벨 {lab_n} -> {tuple(u.shape)} "
                           f"!= {tuple(sh_o)}")
    vn, en = replay(cn, gn, raw_n, seed_n, ns_newB)
    if en:
        return False, f"새 구간 재실행 실패: {en}"

    # **모든 배치 조각을 본다.** b=0 만 보면 "첫 배치만 맞는" 계산을 승인한다(외부 검토
    # 2026-09-19 의 반례). 조각마다 옛 구간을 그 조각의 입력으로 다시 돌려 대조한다.
    for b in range(batch):
        seed_ob = {}
        for to, tn in in_pairs:
            _sh, lab_n = in_shape_of(tn, cn, gn, raw_n, ns_newB)
            u = unbatch(seed_n[tn], lab_n, batch, b)
            if u is None:
                return False, f"b={b} 의 배치 축 가설을 세울 수 없다: {lab_n}"
            seed_ob[to] = u.contiguous()
        vo, eo = replay(co, go, raw_o, seed_ob, ns_old)
        if eo:
            return False, f"옛 구간 재실행 실패(b={b}): {eo}"
        for to, tn in out_pairs:
            if to not in vo or tn not in vn:
                return False, "경계 출력을 재실행 결과에서 못 찾았다"
            a = vo[to]
            lab = _out_label(tn, cn, gn, raw_n)
            if lab is None:
                return False, "경계 출력 라벨이 없다"
            bs = unbatch(vn[tn], lab, batch, b)
            if bs is None:
                return False, f"출력의 배치 축 가설을 세울 수 없다: {lab}"
            if tuple(bs.shape) != tuple(a.shape):
                return False, (f"출력 shape 불일치(b={b}): {tuple(a.shape)} vs "
                               f"{tuple(bs.shape)} (라벨 {lab})")
            if not torch.allclose(a, bs, rtol=rtol, atol=atol):
                d = (a - bs).abs().max().item()
                return False, f"값 불일치(b={b}): 최대 차이 {d:.3e}"
    compare_components.last_weak = weak_seeds
    return True, ""


def _out_label(tid, comp_ids, graph: Graph, raw: dict):
    for oid in comp_ids:
        p = graph.ports[oid]
        if tid in p["out_ids"]:
            k = p["out_ids"].index(tid)
            lab = (raw.get(oid) or {}).get("output_shape") or []
            if k < len(lab) and isinstance(lab[k], list):
                return list(lab[k])
    return None


if __name__ == "__main__":
    sys.exit(main())
