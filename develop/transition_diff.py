r"""B=1 판과 비퇴화 배치 판의 **라벨 차이를 원인별로 가른다.**

왜 필요한가
-----------
47개를 전량 재트레이스하면 라벨이 대거 바뀐다. 기준선을 그대로 다시 박으면 **진짜 회귀가
배치 변화에 묻힌다**(외부 검토 2026-09-13). 그래서 차이를 분류해 사람이 볼 것만 남긴다.

    batch_expected   B=1 을 대입하면 옛 식과 값이 같고, 비배치 심볼 구성도 그대로다
                     예: `n_h` -> `B*n_h`, `T` -> `B*T`, `E*T` -> `B*E*T`
    literal_resolved 옛 판이 정수였는데 이름이 붙었다(배치가 드러나 가릴 수 있게 됐다)
    singleton_fixed  옛 판이 `B` 였는데 `1` 이 됐다 -- 방송 싱글턴을 배치로 오인했던 자리
    alignment_drift  그 그룹의 라벨 다중집합이 그대로다 -- 자리만 밀렸지 의미는 그대로
    semantic_change  비배치 심볼 구성이 바뀌었다. **사람이 봐야 한다**
    unexplained      위 어디에도 안 맞는다. **사람이 봐야 한다**

`op_id` 로 짝짓지 않는다 -- 배치가 바뀌면 op 구성이 조금 달라져 번호가 밀린다.
`(module_key, raw_op, 등장순서, field, shape_index, axis)` 로 맞춘다.

실행:
    .venv\Scripts\python.exe develop\transition_diff.py <모델이름> [--new develop/out]
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
sys.path.insert(0, os.path.join(PROJ, "src"))
if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import build_table
import dim_expr as DE                     # noqa: E402
from anchors import module_key            # noqa: E402

_FIELDS = (("input_shape", "i"), ("output_shape", "o"), ("weight_shape", "w"))


def _synthetic_scale(r, batch: int):
    """이 행의 리터럴 축을 배치로 나눠야 하나 -- 나눠야 하면 배치, 아니면 1.

    `caveat` 이 붙은 행은 **대체 remedy 가 만든 숫자**를 싣고 있다고 remedy 스스로 선언한
    자리다(build_table.apply_caveats). Kimi-K3 의 MoE 는 정렬된 토큰을 균등 분할하므로
    전문가에 들어가는 토큰 수가 `B*k*T/traced` 인데, 라벨이 맨 정수라 B=1 정규화가 통하지
    않는다. 옛 판 1280 과 새 판 3840 이 서로 다른 서명이 되어 expert op 전부가 미짝으로
    떨어졌다(2026-09-14). 선언이 없는 행은 건드리지 않는다.
    """
    return batch if (r.get("caveat") and batch > 1) else 1


def _ops(model_dir: str, phase: str, ns1, batch: int = 1):
    """모듈별 op 목록. 각 op 은 **B=1 로 정규화한 구체 shape** 을 서명으로 갖는다.

    서수로 짝지으면 안 된다 -- 배치를 바꾸면 연속성 때문에 `view` 가 `clone`+`_unsafe_view`
    로 갈려 op 이 끼어들고, 그 뒤가 전부 밀린다. 그 밀림이 `T -> d_head`(16 -> 128) 같은
    "있을 수 없는 재라벨" 로 나타났다(외부 검토 2026-09-13).

    새 판은 비퇴화 배치로 잡혔지만 **라벨을 B=1 로 평가하면** 옛 판과 같은 좌표가 된다.
    그 값으로 서명을 만들어 맞춘다.
    """
    p = os.path.join(model_dir, "full", f"{phase}.trace.raw.jsonl")
    if not os.path.exists(p):
        return {}
    per = collections.defaultdict(list)
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        shapes = {}
        for fld, tag in _FIELDS:
            v = r.get(fld)
            if v is None:
                continue
            group = [v] if fld == "weight_shape" else v
            scale = _synthetic_scale(r, batch)
            got = []
            for sh in group:
                if not isinstance(sh, list):
                    got.append(None)
                    continue
                vals = []
                for x in sh:
                    v = DE.evaluate(x, ns1)
                    # 맨 정수이고 배치로 딱 나눠떨어질 때만 나눈다. 안 나눠떨어지면 그 축은
                    # 배치와 무관한 상수이므로 그대로 둔다.
                    if scale > 1 and str(x).isdigit() and v is not None and v % scale == 0:
                        v //= scale
                    vals.append(v)
                got.append(tuple(vals))
            shapes[tag] = tuple(got)
        sig = (r.get("raw_op") or "", tuple(sorted(r.get("params") or [])),
               shapes.get("i"), shapes.get("o"), shapes.get("w"))
        per[r.get("module_path") or "(root)"].append((sig, r))
    return per


# 배치를 바꾸면 메모리 연속성이 달라져 reshape 경로가 갈린다 -- `view` 하나가
# `clone` + `_unsafe_view` 가 된다. 이 계열만으로 이루어지고 경계가 같으면 **의미 변화가
# 아니다**. 다만 "reshape 계열이다" 만으로 승인하면 안 되고 **경계가 동치인지** 봐야 한다
# (외부 검토 2026-09-13).
RESHAPE_OPS = {"aten.view.default", "aten._unsafe_view.default", "aten.clone.default",
               "aten.contiguous.default", "aten.reshape.default",
               "aten._reshape_alias.default", "aten.expand.default"}

# 시퀀스를 내림으로 가를 때 생기는 op. 의미를 바꿀 수 있는 op 이므로 **경계 동치 검사를
# 통과할 때만** 인정한다 -- 그 검사가 곧 "조각이 토큰을 잃지도 겹치지도 않았다" 이다.
PARTITION_OPS = {"aten.slice.Tensor", "aten.cat.default", "aten.alias.default",
                 "aten.narrow.default"}


def _shapes_of(rec, ns1):
    """이 op 의 (입력 shape **다중집합**, 출력 shape 다중집합). 전부 B=1 로 정규화한 값이다.

    집합이 아니라 다중집합이다. 집합은 `cat([a, a])` 의 중복을 지워서, 같은 조각을 두 번
    쓰는 것과 서로 다른 두 조각을 쓰는 것을 구별하지 못했다(외부 검토 2026-09-18).
    """
    def norm(v, weight=False):
        out = collections.Counter()
        for sh in ([v] if weight else (v or [])):
            if isinstance(sh, list):
                out[tuple(DE.evaluate(x, ns1) for x in sh)] += 1
        return out
    ins = norm(rec.get("input_shape"))
    if rec.get("weight_shape"):
        ins.update(norm(rec.get("weight_shape"), weight=True))
    return ins, norm(rec.get("output_shape"))


def _ports(model_dir: str, phase: str) -> dict:
    """`full/<phase>.ports.jsonl` 을 op_id 로 색인한다. 없으면 빈 dict -- 그러면 분할 검증이
    통과하지 못하고, 그것이 의도다(근거가 없으면 승인하지 않는다)."""
    p = os.path.join(model_dir, "full", f"{phase}.ports.jsonl")
    out = {}
    if not os.path.exists(p):
        return out
    for line in open(p, encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:                                      # noqa: BLE001
            continue
        if r.get("op_id") is not None:
            out[r["op_id"]] = r
    return out


def _tiling_ok(rows, ports, ns=None) -> bool:
    """이 구간의 `slice` 들이 부모 축을 **정확히 한 번씩** 덮는가.

    shape 만 보는 검사는 토큰 중복을 승인한다. 외부 검토(Codex 2026-09-18)가 코드로 짚었다:

        옛: alias(x)                    # [4] -> [4]
        새: a = x[:2]; cat([a, a])      # [4] -> [2] -> [4]
        x = [0,1,2,3] 인데 새 판은 [0,1,0,1] 이다.

    들어오는 shape 집합도 나가는 shape 집합도 같아서 통과했다. 값이 다른데 shape 이 같은
    것을 shape 으로는 못 가른다. 그래서 **실제 범위**를 본다 -- ports 의 `scalar_args`
    `pos = [dim, start, end]` 와 `input_tensor_ids` 로 어느 텐서의 어느 구간인지 읽는다.

    같은 (부모 텐서, dim) 의 조각들이 겹치거나 빈틈을 남기면 분할이 아니다. 범위를 읽을 수
    없으면 **승인하지 않는다** -- 모르는 것을 통과시키는 쪽이 아니라 막는 쪽으로 넘어진다.
    """
    groups = collections.defaultdict(list)
    for r in rows:
        if r.get("raw_op") not in ("aten.slice.Tensor", "aten.narrow.default"):
            continue
        pt = ports.get(r.get("op_id")) or {}
        pos = (pt.get("scalar_args") or {}).get("pos")
        tids = pt.get("input_tensor_ids") or []
        if not pos or len(pos) < 3 or not tids:
            return False                       # 범위를 모르면 승인 못 한다
        dim, start, end = pos[0], pos[1], pos[2]
        ins = [x for x in (r.get("input_shape") or []) if isinstance(x, list)]
        if not ins or dim >= len(ins[0]):
            return False
        parent = DE.evaluate(ins[0][dim], ns) if ns is not None else ins[0][dim]
        if not isinstance(parent, int) or parent <= 0:
            return False
        # 음수 인덱스와 `sys.maxsize` 꼴의 열린 끝을 실제 범위로 되돌린다.
        s0 = start + parent if start < 0 else start
        e0 = end + parent if end < 0 else min(end, parent)
        if not (0 <= s0 < e0 <= parent):
            return False
        groups[(tids[0], dim, parent)].append((s0, e0))
    if not groups:
        return False                           # 자를 것이 없으면 분할이라 부르지 않는다
    for (_tid, _dim, parent), pieces in groups.items():
        # **정확히 한 번씩 덮어야 한다.** 겹치면 중복이고, 빈틈이 있으면 누락이다.
        # 겹침만 보던 판은 `cat([a, a])` 를 막지 못했다 -- 조각 하나가 부모의 절반만
        # 덮는데도 통과했다.
        at = 0
        for s0, e0 in sorted(pieces):
            if s0 != at:
                return False
            at = e0
        if at != parent:
            return False
    return True


def classify_unmatched(left_old, left_new, ns_old, ns_new1, ports_old=None, ports_new=None):
    """짝 못 지은 구간이 **연속성 때문에 갈린 reshape** 인가.

    승인 조건(전부 만족해야 한다):
      * 양쪽 모두 reshape 계열 op 만 (분할 op 은 아래 추가 조건까지 만족할 때만)
      * 파라미터를 새로 읽거나 놓지 않음
      * 구간에 들어오는 텐서 **다중집합**이 같음(B=1 정규화)
      * 구간에서 나가는 텐서 다중집합이 같음
      * 새로 생긴 중간 텐서는 구간 안에서 소비됨
      * 분할 op 이 끼면 `_tiling_ok` -- 조각이 부모를 정확히 한 번씩 덮어야 한다

    **집합이 아니라 다중집합으로 센다.** 집합은 `cat([a, a])` 의 중복을 지운다.
    """
    if not (left_old or left_new):
        return "none", ""
    ops = {r.get("raw_op") for r in left_old} | {r.get("raw_op") for r in left_new}
    partition = ops <= (RESHAPE_OPS | PARTITION_OPS) and not (ops <= RESHAPE_OPS)
    if not ops <= (RESHAPE_OPS | PARTITION_OPS):
        bad = sorted(ops - RESHAPE_OPS - PARTITION_OPS)[:3]
        return "semantic_topology_change", f"reshape 아닌 op: {bad}"
    if any(r.get("params") for r in left_old + left_new):
        return "parameter_access_change", "파라미터 접근이 달라졌다"

    oi, oo = collections.Counter(), collections.Counter()
    for r in left_old:
        a, b = _shapes_of(r, ns_old)
        oi.update(a); oo.update(b)
    ni, no = collections.Counter(), collections.Counter()
    for r in left_new:
        a, b = _shapes_of(r, ns_new1)
        ni.update(a); no.update(b)
    if oi - ni:
        return "unexplained_topology_change", "들어오는 텐서가 다르다"
    if oo - no:
        return "unexplained_topology_change", "나가는 텐서가 다르다"
    extra = no - oo
    if extra - ni:
        return "unexplained_topology_change", f"소비되지 않는 새 텐서 {sum((extra - ni).values())}개"
    if partition:
        # 분할은 **범위**로 검증한다. shape 으로는 중복과 분할을 못 가른다.
        if not _tiling_ok(left_new, ports_new or {}, ns_new1):
            return "unexplained_topology_change", "분할 범위를 확인 못 했다 (겹침·빈틈·근거 없음)"
        return "sequence_partition_expected", ""
    return "layout_lowering_verified", ""


def _reviewed_transitions(model: str) -> list:
    """`develop/verify/references.yaml` 의 `transition_reviewed` 중 이 모델 것.

    값이 같아 트레이스로는 못 가르는 이름 교체를 **소스를 보고** 받아들인 기록이다.
    `source` 가 없는 항목은 읽지 않는다 -- 근거 없이 통과시키지 않는다.
    """
    p = os.path.join(HERE, "verify", "references.yaml")
    if not os.path.isfile(p):
        return []
    try:
        import yaml
        d = yaml.safe_load(io.open(p, encoding="utf-8")) or {}
    except Exception:                                          # noqa: BLE001
        return []
    out = []
    for e in (d.get("transition_reviewed") or []):
        if e.get("model") != model or not e.get("source"):
            continue
        out.append((e.get("scope") or "", set(e.get("from") or []),
                    set(e.get("to") or [])))
    return out


def _applied_verdicts(model_dir: str) -> set:
    """실제로 발화한 ④층 교정의 `(module 정규식, from, to)` 집합.

    인용을 달아 등록한 교정이 라벨을 바꾼 것은 **회귀가 아니라 반영**이다. 그걸 그냥
    `semantic_change` 로 세면, 근거를 갖춰 고칠수록 사람 검토 건수가 늘어난다.
    발화 0건인 항목은 넣지 않는다 -- 아무것도 안 바꿨으므로 설명할 것도 없다.
    """
    p = os.path.join(model_dir, "full", "label_overrides.json")
    out = set()
    if not os.path.exists(p):
        return out
    try:
        for e in json.load(open(p, encoding="utf-8")) or []:
            if e.get("applied") and e.get("from") and e.get("to"):
                out.add((e.get("module") or "", str(e["from"]), str(e["to"])))
    except Exception:                                          # noqa: BLE001
        pass
    return out


def _unconfirmed_sites(model_dir: str, phase: str) -> set:
    """축 판정 사이드카가 **확정이 아니라고** 적은 자리들.

    사이드카는 확정이 아닌 자리만 기록하므로, 여기 없으면 확정이다. `literal_resolved`
    (정수였던 것이 이름을 얻음)를 "B=1 에서 값이 같다" 만으로 승인하면 안 된다는 외부 검토
    지적(2026-09-13)을 이걸로 검사한다.
    """
    p = os.path.join(model_dir, "full", f"{phase}.axis_resolution.jsonl")
    out = set()
    if not os.path.exists(p):
        return out
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        if r.get("kind") == "site":
            out.add((r.get("op_id"), r.get("field"), r.get("shape_index"), r.get("axis")))
    return out


def _leftovers(a, b):
    """서명으로 맞추고 남은 것. `_match` 와 같은 규칙을 쓴다."""
    byb = collections.defaultdict(collections.deque)
    for sig, r in b:
        byb[sig].append(r)
    la = []
    for sig, r in a:
        if byb.get(sig):
            byb[sig].popleft()
        else:
            la.append(r)
    lb = [r for q in byb.values() for r in q]
    return la, lb


def _match(old_ops, new_ops):
    """(짝지은 쌍, 옛쪽 미짝, 새쪽 미짝). 서명이 같은 것끼리 등장 순서대로 맞춘다."""
    pairs, un_a, un_b = [], 0, 0
    for mod in set(old_ops) | set(new_ops):
        a, b = old_ops.get(mod, []), new_ops.get(mod, [])
        by_b = collections.defaultdict(collections.deque)
        for sig, r in b:
            by_b[sig].append(r)
        used = collections.Counter()
        for sig, ra in a:
            q = by_b.get(sig)
            if q:
                pairs.append((ra, q.popleft()))
                used[sig] += 1
            else:
                un_a += 1
        un_b += sum(len(q) for q in by_b.values())
    return pairs, un_a, un_b


def classify(old_lab, new_lab, old_ns, new_ns1, confirmed=True, scale=1):
    """이 차이는 무엇 때문인가."""
    if old_lab == new_lab:
        return None
    # 대체 remedy 가 만든 숫자(`caveat` 이 붙은 행)가 배치에 정비례해 커진 것. 이름이 아니라
    # **대체물의 크기**가 바뀐 것이므로 의미 변화가 아니다. 다만 그것만으로 승인하지는 않고,
    # 토큰 보존은 `check_dispatch_conservation` 이 따로 본다.
    if (scale > 1 and old_lab.isdigit() and new_lab.isdigit()
            and int(new_lab) == int(old_lab) * scale):
        return "synthetic_dispatch_scaling"
    ov, nv1 = DE.evaluate(old_lab, old_ns), DE.evaluate(new_lab, new_ns1)
    old_syms = DE.free_symbols(old_lab) - {"B"}
    new_syms = DE.free_symbols(new_lab) - {"B"}

    if old_lab == "B" and new_lab == "1":
        return "singleton_fixed"
    # **이름에서 맨 정수로 물러난 것은 회귀가 아니다.** 지어낸 이름(`3*d_conv`, `n_h_kda/2`,
    # 등록을 물린 `n_chunk`)을 거두고 숫자로 남긴 자리다 -- 주장이 **줄어든다**. 값이 같으면
    # 새로 틀릴 것이 없다. `literal_resolved`(정수 -> 이름)의 정확한 거울이다 (2026-09-19).
    if new_lab.lstrip("-").isdigit() and not old_lab.lstrip("-").isdigit():
        if ov is not None and ov == nv1:
            return "fabrication_withdrawn"
    if old_lab.lstrip("-").isdigit() and not new_lab.lstrip("-").isdigit():
        # 정수였던 것이 이름을 얻었다. B=1 대입값이 같아야 한다.
        if ov is None or ov != nv1:
            return "unexplained"
        # 값이 같다는 것만으로는 부족하다. 새 라벨이 **확정 등급**이어야 한다.
        return "literal_resolved" if confirmed else "literal_resolved_unconfirmed"
    # **B=1 을 대입하면 같은 값이고, 비배치 심볼 구성도 그대로다.**
    if ov is not None and nv1 is not None and ov == nv1 and old_syms == new_syms:
        if DE.batch_degree(new_lab, new_ns1) != DE.batch_degree(old_lab, old_ns):
            return "batch_expected"
        return "unexplained"        # 값도 심볼도 같은데 문자열이 다르다 -- 표기 흔들림
    # 구조 심볼이 런타임 심볼(/)로, 또는 그 반대로 바뀌는 것은 정상적인 배치 전환이
    # 아니다. 정상 전환은 구조 심볼을 **보존**한다(). 별도 범주로 둔다.
    # **옛 T 에서는 두 이름이 수치적으로 구별 불가능했던 자리.** 내림식이 그렇다 --
    # `m*(T//m)` 은 T 가 m 의 배수면 곧 T 라서, 옛 발행본(T=2048, m_hca=128)은 그 축을
    # `T` 라고 부를 수밖에 없었다. 새 T=2056 에서 2048+8 로 갈리며 내림이 드러난 것이고,
    # 새 이름이 더 정확하다. 옛 이름이 틀렸던 게 아니라 **덜 정해져 있었다**.
    #
    # 판정 조건: 새 라벨을 **옛 좌표**에 넣으면 옛 라벨과 값이 같고(그래서 옛 트레이스는
    # 둘을 가를 수 없었다), 새 라벨이 시퀀스에서 유도된 양이다.
    if "T" in new_syms and ov is not None:
        nv_old = DE.evaluate(new_lab, old_ns)
        if nv_old is not None and nv_old == ov:
            return "sequence_partition_expected"

    RT = {"B", "T", "1"}
    if (old_lab in RT) != (new_lab in RT):
        return "runtime_role_change"
    if old_syms != new_syms:
        return "semantic_change"
    return "unexplained"


def check_dispatch_conservation(model_dir: str, phase: str) -> list:
    """대체된 디스패치가 **토큰을 보존하는지** 구체 shape 으로 검사한다. 실패 목록을 돌려준다.

    합성 축을 "배치에 비례하니 괜찮다" 로만 승인하면, 분할이 토큰을 흘리거나 겹쳐도 통과한다.
    remedy 가 `caveat` 으로 표시한 모듈마다 다음을 본다(외부 검토가 요구한 보존식):

      * 분할 조각의 합 == 디스패치에 들어간 토큰 수 (누락·중복 없음)
      * 조각이 전부 양수 (빈 전문가로 축이 사라지지 않음)
      * 되모으는 `cat` 의 출력이 다시 그 토큰 수
      * 최종 토큰 축이 `B*T` (가중합이 토큰당 하나를 낸다)

    구조만 보고 값은 안 본다 -- 값은 애초에 트레이스에 없다. 그래서 "분할이 대체물이다" 를
    감추지 않으면서 "대체물이 최소한 일관된다" 는 말할 수 있다.
    """
    conc = build_table.load_concrete(model_dir, phase)
    raw = os.path.join(model_dir, "full", f"{phase}.trace.raw.jsonl")
    if not os.path.exists(raw):
        return []
    prov = DE.load_provenance(model_dir)
    # decode 의 쿼리 길이는 1 이다 -- 토큰 축은 `B*T` 가 아니라 `B`. 처음에 phase 를 안 보고
    # `B*seq_len_used` 로 기대해서 decode 92건이 전부 거짓 실패였다(2026-09-15).
    bt_tokens = int(prov.get("capture_batch") or 1) * (
        int(prov.get("seq_len_used") or 0) if phase == "prefill" else 1)
    per = collections.defaultdict(lambda: {"index": set(), "slice": [], "cat": [], "tail": set()})
    for line in open(raw, encoding="utf-8"):
        r = json.loads(line)
        if not r.get("caveat"):
            continue
        c = conc.get(r.get("op_id")) or {}
        ins = [x for x in (c.get("input_shape") or []) if isinstance(x, list) and x]
        outs = [x for x in (c.get("output_shape") or []) if isinstance(x, list) and x]
        if not outs:
            continue
        # 전문가 아래가 아니라 **블록 자체**의 행만 본다 -- 분할과 되모으기가 거기 있다
        mp = r.get("module_path") or ""
        key = mp.split(".experts.")[0]
        op = (r.get("raw_op") or "").split(".")
        name = op[1] if len(op) > 1 else ""
        if ".experts." in mp:
            continue
        if name == "index":
            per[key]["index"].add(outs[0][0])
        elif name == "slice" and ins and len(ins[0]) == len(outs[0]) and ins[0][1:] == outs[0][1:]:
            per[key]["slice"].append((ins[0][0], outs[0][0]))
        elif name == "cat" and len(ins) > 1:
            per[key]["cat"].append((tuple(x[0] for x in ins), outs[0][0]))
        elif name == "sum":
            per[key]["tail"].add(outs[0][0])

    bad = []
    for mod, d in sorted(per.items()):
        if not d["index"]:
            continue
        n = max(d["index"])
        pieces = [o for i, o in d["slice"] if i == n]
        if not pieces:
            bad.append(f"{mod}: 디스패치 {n} 토큰인데 분할 조각을 못 찾았다")
            continue
        if any(x <= 0 for x in pieces):
            bad.append(f"{mod}: 빈 조각 {pieces}")
        if sum(pieces) != n:
            bad.append(f"{mod}: 조각 합 {sum(pieces)} != 디스패치 {n} (누락 또는 중복)")
        rejoin = [o for ins, o in d["cat"] if set(ins) <= set(pieces) and len(ins) == len(pieces)]
        if not rejoin:
            bad.append(f"{mod}: 조각을 되모으는 cat 이 없다")
        elif max(rejoin) != n:
            bad.append(f"{mod}: 되모은 길이 {max(rejoin)} != 디스패치 {n}")
        if bt_tokens and d["tail"] and bt_tokens not in d["tail"]:
            bad.append(f"{mod}: 최종 토큰 축 {sorted(d['tail'])} 에 B*T={bt_tokens} 가 없다")
    return bad


def run(model, new_root, show=8, old_root=None):
    old_dir = os.path.join(old_root or os.path.join(PROJ, "models"), model)
    new_dir = os.path.join(new_root, model)
    if not (os.path.isdir(old_dir) and os.path.isdir(new_dir)):
        print(f"{model}: 옛 판 또는 새 판이 없다")
        return 1
    old_prov, new_prov = DE.load_provenance(old_dir), DE.load_provenance(new_dir)
    old_b = int(old_prov.get("capture_batch") or 1)
    # **양쪽을 같은 좌표에서 평가한다 -- B=1.**
    #
    # 원래는 `DE.namespace(old_prov, old_b)` 였다. 옛 판이 항상 B=1 이던 시절에는 그게 곧
    # B=1 이라 새 판(B=1 로 고정)과 좌표가 같았다. 그런데 B=3 판을 한 번 승격하고 나면 옛
    # 판도 B=3 이 되어, 옛쪽만 B=3 으로 평가되고 새쪽은 B=1 로 평가된다. 그러면 같은 op 의
    # 서명이 `(3,264,2880)` vs `(1,264,2880)` 로 갈려 **전부 미짝**이 된다 -- gpt-oss-20b 에서
    # 짝지은 op 이 440 개로 떨어지고 사람 검토가 12,596 건으로 튀었다 (2026-09-17).
    #
    # 실제 capture 배치는 `_ops(batch=...)` 로 따로 넘긴다. 그쪽은 좌표가 아니라 **합성 축
    # 리터럴을 몇으로 나눌지**에 쓰이므로 진짜 배치여야 한다.
    old_ns = DE.namespace(old_prov, 1)
    # 새 라벨을 **옛 판의 좌표**(B=1, 옛 T)로 평가한다. 발행점이 `(B, T)` 를 함께 고르므로
    # T 도 달라질 수 있고, 그러면 T 를 품은 모든 shape 이 미짝으로 떨어진다.
    old_t = int(old_prov.get("seq_len_used") or 0) or None
    new_ns1 = DE.namespace(new_prov, 1, seq_len=old_t)
    if old_t and old_t != new_prov.get("seq_len_used"):
        print(f"   (T 가 {old_t} -> {new_prov.get('seq_len_used')} 로 바뀌었다 -- "
              f"새 라벨을 옛 T 로 평가해 맞춘다)")
    print(f"=== {model}   옛 B={old_b} -> 새 B={new_prov.get('capture_batch')}")

    tot = collections.Counter()
    samples = collections.defaultdict(collections.Counter)
    for phase in ("prefill", "decode"):
        new_b = int(new_prov.get("capture_batch") or 1)
        old_ops = _ops(old_dir, phase, old_ns, batch=old_b)
        new_ops = _ops(new_dir, phase, new_ns1, batch=new_b)   # 새 라벨을 B=1 로 평가해 정규화
        if not (old_ops and new_ops):
            continue
        unconf = _unconfirmed_sites(new_dir, phase)
        verdicts = _applied_verdicts(new_dir)
        reviewed = _reviewed_transitions(model)
        po, pn = _ports(old_dir, phase), _ports(new_dir, phase)
        pairs, un_a, un_b = _match(old_ops, new_ops)
        tot["짝지은 op"] += len(pairs)
        # **짝 못 지은 구간도 분류한다.** 개수만 세면 "검사했다" 가 아니다.
        for mod in set(old_ops) | set(new_ops):
            la, lb = _leftovers(old_ops.get(mod, []), new_ops.get(mod, []))
            if not (la or lb):
                continue
            kind, why = classify_unmatched(la, lb, old_ns, new_ns1,
                                           ports_old=po, ports_new=pn)
            tot[kind] += len(la) + len(lb)
            if why:
                samples[kind][(mod.split(".")[-1][:24], why, "")] += 1
            else:
                samples[kind][(f"{len(la)} -> {len(lb)}",
                               ",".join(sorted({r.get("raw_op", "").split(".")[-2]
                                                for r in lb})),
                               mod.split(".")[-1][:22])] += 1
        for ra, rb in pairs:
            for fld, tag in _FIELDS:
                va, vb = ra.get(fld), rb.get(fld)
                if va is None or vb is None:
                    continue
                ga = [va] if fld == "weight_shape" else va
                gb = [vb] if fld == "weight_shape" else vb
                for gi, (sa, sb) in enumerate(zip(ga, gb)):
                    if not (isinstance(sa, list) and isinstance(sb, list)) or len(sa) != len(sb):
                        continue
                    for ax, (lo, ln) in enumerate(zip(sa, sb)):
                        lo, ln = str(lo), str(ln)
                        site = (rb.get("op_id"), tag, gi, ax)
                        c = classify(lo, ln, old_ns, new_ns1,
                                     confirmed=site not in unconf,
                                     scale=new_b if rb.get("caveat") else 1)
                        # 인용 달린 교정이 실제로 바꾼 자리인가. 맞으면 반영이지 회귀가 아니다.
                        if c == "semantic_change" and verdicts:
                            mp = rb.get("module_path") or ""
                            if any(f == lo and t == ln and (not mx or re.search(mx, mp))
                                   for mx, f, t in verdicts):
                                c = "verdict_applied"
                        # 소스를 보고 받아들인 이름 교체. 등재된 **정확한 from/to 쌍**과
                        # 모듈 scope 에만 적용한다 -- 넓게 잡으면 진짜 회귀를 삼킨다.
                        if c == "semantic_change" and reviewed:
                            mp = rb.get("module_path") or ""
                            # 이름 교체와 **배치 접힘이 같은 자리에 겹칠 수 있다**
                            # (`n_h_kda` -> `B*n_h`). 배치는 `batch_expected` 가 이미
                            # 인정하는 변화이므로, 비교 전에 `B*` 한 겹만 벗긴다.
                            _ln = ln[2:] if ln.startswith("B*") else ln
                            _lo = lo[2:] if lo.startswith("B*") else lo
                            if any(_lo in fr and _ln in to and (not mx or re.search(mx, mp))
                                   for mx, fr, to in reviewed):
                                c = "transition_reviewed"
                        if c is None:
                            tot["같음"] += 1
                            continue
                        tot[c] += 1
                        samples[c][(lo, ln, ra.get("raw_op"))] += 1

    for k, v in tot.most_common():
        print(f"   {k:22} {v:>9,}")

    # 합성 디스패치가 나왔으면 **토큰 보존**을 따로 본다. 비례만으로 승인하면 분할이 토큰을
    # 흘려도 통과한다.
    cons = []
    if tot["synthetic_dispatch_scaling"]:
        for phase in ("prefill", "decode"):
            cons += [f"{phase}: {m}" for m in check_dispatch_conservation(new_dir, phase)]
        if cons:
            # 비례한다는 것만으로 승인하지 않는다 -- 보존이 깨졌으면 사람이 봐야 한다.
            # `audit` 이 읽는 두 토큰 형식으로 낸다(AUTO_OK 에 없는 이름).
            print(f"   synthetic_dispatch_unconserved {len(cons):>9,}")
            print("\n   **합성 디스패치 보존 검사 실패**")
            for m in cons[:10]:
                print(f"      {m}")
        else:
            print("\n   합성 디스패치 보존 검사: 통과 (조각 합 == 디스패치, 되모으기 일치, "
                  "최종 토큰 축 B*T)")
    # `alignment_suspected` 는 **자동 승인 범주가 아니다.** 그룹 다중집합이 같다는 것만으로는
    # 진짜 축 맞바뀜(전치 오라벨)과 구별되지 않는다(외부 검토 2026-09-13).
    review = (len(cons) + tot["semantic_change"] + tot["unexplained"]
              + tot["runtime_role_change"] + tot["alignment_suspected"]
              + tot["semantic_topology_change"] + tot["unexplained_topology_change"]
              + tot["parameter_access_change"] + tot["literal_resolved_unconfirmed"])
    for kind in ("verdict_applied", "fabrication_withdrawn", "synthetic_dispatch_scaling",
                 "sequence_partition_expected",
                 "parameter_access_change",
                 "semantic_topology_change",
                 "unexplained_topology_change", "runtime_role_change",
                 "semantic_change", "unexplained", "literal_resolved_unconfirmed",
                 "alignment_suspected", "batch_expected", "literal_resolved",
                 "singleton_fixed", "layout_lowering_verified",
                 "transition_reviewed"):
        if not samples[kind]:
            continue
        print(f"\n   --- {kind}")
        for (o, n, op), cnt in samples[kind].most_common(show):
            print(f"      {cnt:7,}  `{o}` -> `{n}`   {op}")
    print(f"\n사람이 봐야 하는 것: **{review:,}**")
    return 1 if review else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model")
    ap.add_argument("--new", default=os.path.join(HERE, "out"))
    ap.add_argument("--old", default=None,
                     help="옛 판 루트 (기본 models/). 두 진단용 트레이스를 직접 비교할 때 쓴다")
    ap.add_argument("--show", type=int, default=8)
    a = ap.parse_args()
    return run(a.model, a.new, a.show, old_root=a.old)


if __name__ == "__main__":
    sys.exit(main())
