r"""**발행 표만으로 DAG 와 roofline 을 세울 수 있나** -- 실측으로 답하는 탐침.

주장 대신 해 본다. 발행된 `csv`/`jsonl` 과 `symbol_table` 만 읽어서

  1. DAG 를 만든다            (`depends_on`, `layers`/`repeat` 펼침)
  2. 심볼에 숫자를 대입한다    (`B=128` 처럼 덮어쓴다)
  3. op 마다 FLOPs·바이트를 센다
  4. **대입이 안 되는 자리를 전부 보고한다**

4 번이 핵심이다. "되는 것 같다" 가 아니라 **어디가 막히는지** 세어야 판단이 된다.

실행:
    .venv\Scripts\python.exe develop\plus_at_roofline_probe.py \
        models/moonshotai__Kimi-K3/plus_at prefill --B 128
    .venv\Scripts\python.exe develop\plus_at_roofline_probe.py \
        models/openai__gpt-oss-120b prefill --B 128 --dtype 2

한계 (정직하게)
---------------
* **dtype 열이 없다.** `--dtype` 로 바이트/원소를 받는다. 기본 2 (bf16).
* **텐서 재사용을 모른다.** 발행 표에는 tensor id 가 없다(그건 `full/ports.jsonl` 에 있다).
  그래서 바이트는 **상한**(op 마다 입력+출력을 다 더한 값)이고, 가중치는 `params` 로
  중복을 제거한 하한도 함께 낸다. 진짜 traffic 은 그 사이에 있다.
* conv1d 는 커널 폭을 표에서 못 읽어 근사다. 표시한다.
* `T` 는 트레이스 시점에 고정된다 -- `--T` 로 덮어쓰면 **그 길이로 다시 트레이스한 것과
  다르다**(attention 길이 의존 구조가 안 바뀐다). 그래서 기본은 거부하고, 굳이 하려면
  `--force-T` 를 받는다.
"""
import argparse
import collections
import io
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import plus_at_canon as C                                        # noqa: E402

IDENT = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
SAFE = {"ceil": math.ceil, "floor": math.floor, "min": min, "max": max}

# op 마다 FLOPs 를 어떻게 세나. `exact` 가 아닌 것은 결과에 그렇게 적는다.
# `linear` 은 bias 가 피연산자에 섞여 있고 `grouped_matmul` 은 전문가 축(E)이 앞에 붙는다.
# 그래서 **rank >= 2 인 마지막 두 피연산자**를 골라 M,K,N 을 읽는다.
EXACT_MM = ("matmul", "batched_matmul", "linear", "grouped_matmul")
ELEMENTWISE = {"elementwise_mul": 1, "elementwise_add": 1, "mul_": 1, "add_": 1,
               "exp": 1, "tanh": 4, "sigmoid": 4, "silu": 5, "relu": 1, "gelu": 5}
REDUCE = {"softmax": 5, "rmsnorm": 4, "sum": 1}
NO_FLOP = ("concat", "embedding")
APPROX = ("conv1d",)


def numel(shape, env, fails, where):
    n = 1
    for ax in shape:
        v = value(ax, env, fails, where)
        if v is None:
            return None
        n *= v
    return n


def value(tok, env, fails, where):
    """축 토큰 하나를 수로. 못 하면 fails 에 적고 None."""
    t = str(tok)
    if t.lstrip("-").isdigit():
        return int(t)
    try:
        unknown = sorted({x for x in IDENT.findall(t)} - set(env) - set(SAFE))
        if unknown:
            fails.append((where, t, f"미선언 심볼 {unknown}"))
            return None
        return int(eval(t.replace("/", "//"), {"__builtins__": {}},   # noqa: S307
                       dict(env, **SAFE)))
    except Exception as e:                                       # noqa: BLE001
        fails.append((where, t, f"평가 실패 {e}"))
        return None


def op_flops(r, env, fails):
    """(flops, 정확도표시). 못 세면 (None, 이유)."""
    ot = r["op_type"]
    ins = r["input_shape"] or []
    outs = r["output_shape"] or []
    w = f"op{r['op_id']} {ot}"
    if ot in NO_FLOP:
        return 0, "none"
    if ot in EXACT_MM:
        mats = [g for g in ins if len(g) >= 2]
        if len(mats) < 2:
            fails.append((w, "", f"{ot} 인데 rank>=2 피연산자가 둘 미만이다"))
            return None, "fail"
        a, b = mats[-2], mats[-1]
        M = value(a[-2], env, fails, w)
        K = value(a[-1], env, fails, w)
        N = value(b[-1], env, fails, w)
        if None in (M, K, N):
            return None, "fail"
        batch = 1
        for ax in a[:-2]:
            v = value(ax, env, fails, w)
            if v is None:
                return None, "fail"
            batch *= v
        return 2 * batch * M * K * N, "exact"
    n = numel(outs[0], env, fails, w) if outs and outs[0] else None
    if n is None:
        return None, "fail"
    if ot in ELEMENTWISE:
        return n * ELEMENTWISE[ot], "elementwise"
    if ot in REDUCE:
        src = numel(ins[0], env, fails, w) if ins and ins[0] else n
        return (src or n) * REDUCE[ot], "reduce"
    if ot in APPROX:
        # 커널 폭을 표에서 못 읽는다. d_conv 가 환경에 있으면 쓰고, 없으면 1 로 둔다.
        k = env.get("d_conv", 1)
        return 2 * n * k, "approx(conv1d: 커널을 d_conv 로 가정)"
    fails.append((w, "", f"FLOPs 규칙이 없는 op_type {ot!r}"))
    return None, "fail"


def op_bytes(r, env, fails, dtype):
    """op 하나가 만지는 바이트 (상한: 입력 + 출력 전부)."""
    w = f"op{r['op_id']} {r['op_type']}"
    tot = 0
    for grp in ((r["input_shape"] or []) + (r["output_shape"] or [])):
        n = numel(grp, env, fails, w)
        if n is None:
            return None
        tot += n
    return tot * dtype


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bundle", help="발행 디렉터리 (models/<m> 또는 .../plus_at)")
    ap.add_argument("phase", choices=("prefill", "decode"))
    ap.add_argument("--B", type=int, help="배치를 이 값으로 덮어쓴다")
    ap.add_argument("--T", type=int, help="시퀀스 길이 (기본 거부. --force-T 필요)")
    ap.add_argument("--force-T", action="store_true")
    ap.add_argument("--dtype", type=int, default=2, help="원소당 바이트 (기본 2=bf16)")
    ap.add_argument("--moe-fix", metavar="EXPR",
                    help="MoE 균등분할 대체값을 이 식으로 바꾼다 "
                         "(예: 'B*T*k/4'). 근거는 overlay 의 moe_aggregate -- "
                         "**공개 번들에는 그 식이 없어서 손으로 줘야 한다**")
    ap.add_argument("--expand", action="store_true",
                    help="repeat/layers 를 펼쳐 층마다 센다 (기본은 repeat 곱셈)")
    a = ap.parse_args()

    bd = os.path.join(PROJ, a.bundle) if not os.path.isabs(a.bundle) else a.bundle
    rows = C.read_jsonl_rows(os.path.join(bd, f"{a.phase}.jsonl"))

    # 심볼 환경: base_symbols.json(번들 안) 또는 full/provenance.json
    env = None
    snap = os.path.join(bd, "base_symbols.json")
    if os.path.isfile(snap):
        d = json.load(io.open(snap, encoding="utf-8"))
        env = {k: v for k, v in (d.get("symbols") or {}).items()
               if isinstance(v, int)}
        src = "번들 안 base_symbols.json"
    if env is None:
        prov = os.path.join(os.path.dirname(bd.rstrip("/\\")), "full",
                            "provenance.json")
        prov = prov if os.path.isfile(prov) else os.path.join(bd, "full",
                                                             "provenance.json")
        if not os.path.isfile(prov):
            print("심볼표를 못 찾았다 (base_symbols.json 도 provenance.json 도 없다)")
            return 1
        d = json.load(io.open(prov, encoding="utf-8"))
        env = {k: v for k, v in (d.get("symbol_table") or {}).items()
               if isinstance(v, int)}
        src = "full/provenance.json"

    # **번들이 새로 넣은 심볼도 읽어야 한다.** base_symbols.json 은 값을 안 담는다
    # (kind 만). 값과 식은 symbols.yaml 에 있다 -- bundle_contract 의
    # symbol_namespaces 가 말하는 그대로다. 둘 중 하나만 읽으면 막힌다(실측).
    added = {}
    sy = os.path.join(bd, "symbols.yaml")
    if os.path.isfile(sy):
        import yaml
        added = (yaml.safe_load(io.open(sy, encoding="utf-8")) or {}).get(
            "symbols") or {}
    print(f"심볼표 {len(env)}개  ({src})"
          + (f" + 번들 선언 {sorted(added)}" if added else ""))
    traced = dict(env)
    if a.B is not None:
        print(f"  B: {env.get('B')} -> {a.B}")
        env["B"] = a.B
    if a.T is not None:
        if not a.force_T:
            print(f"\n**T 를 {env.get('T')} -> {a.T} 로 바꾸는 것은 거부한다.**")
            print("  T 는 트레이스 시점에 고정된다. 표의 구조(attention 길이 의존,")
            print("  residual 누적 단계 수)는 그 T 로 관측된 것이라 숫자만 바꾸면")
            print("  구조가 안 따라온다. 정말 하려면 --force-T.")
            return 1
        print(f"  T: {env.get('T')} -> {a.T}  **구조는 안 바뀐다 (--force-T)**")
        env["T"] = a.T

    # 번들 심볼은 **식을 우선 쓴다** -- 그래야 T 를 바꿀 때 따라온다.
    for name, d in sorted(added.items()):
        v = None
        if d.get("expr"):
            v = value(d["expr"], env, [], f"symbols.yaml/{name}")
        if v is None:
            v = d.get("value")
        env[name] = v
        dom = d.get("validity_domain") or []
        bad_dom = [c for c in dom
                   if not eval(c, {"__builtins__": {}}, dict(env, **SAFE))]  # noqa: S307
        note = f"= {v}" + (f"  (식 {d['expr']!r})" if d.get("expr") else "")
        if bad_dom:
            note += f"  **validity_domain 위반 {bad_dom}**"
        print(f"    {name} {note}")

    # ---------------------------------------------------------------- DAG
    by_id = {r["op_id"]: r for r in rows}
    dup = len(rows) - len(by_id)
    edges = sum(len(r["depends_on"] or []) for r in rows)
    dangling = [(r["op_id"], d) for r in rows for d in (r["depends_on"] or [])
                if d not in by_id]
    # 위상 정렬로 순환 확인
    indeg = {i: 0 for i in by_id}
    adj = collections.defaultdict(list)
    for r in rows:
        for d in (r["depends_on"] or []):
            if d in by_id:
                adj[d].append(r["op_id"])
                indeg[r["op_id"]] += 1
    q = [i for i, v in indeg.items() if v == 0]
    seen = 0
    while q:
        n = q.pop()
        seen += 1
        for m in adj[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                q.append(m)
    print()
    print("=== DAG")
    print(f"  노드 {len(by_id)}  간선 {edges}  중복 op_id {dup}  "
          f"없는 의존 {len(dangling)}")
    print(f"  위상 정렬 {seen}/{len(by_id)}  -> {'비순환' if seen == len(by_id) else '**순환 있음**'}")
    roots = [i for i, r in by_id.items() if not (r["depends_on"] or [])]
    leaves = [i for i in by_id if not adj[i]]
    print(f"  뿌리 {len(roots)}  잎 {len(leaves)}")

    # ---------------------------------------------------------------- 대입
    # **MoE 균등분할 대체값 보정.** 표의 전문가 행은 토큰 축이 리터럴(prefill 3840,
    # decode 12)이다. 그건 shim 이 전문가 4 개로 균등분할한 대체값이라 B 에 안 따라온다.
    # overlay 의 moe_aggregate 가 "총 라우팅 토큰은 B*T*k 로 보존된다" 고 하므로
    # 리터럴을 B*T*k/traced 로 되돌리면 총 FLOPs 가 맞는다. 개별 전문가 분포는 여전히
    # 복원되지 않는다 -- 총량만 맞다.
    moe_fixed = 0
    if a.moe_fix:
        want = value(a.moe_fix, env, [], "--moe-fix")
        if want is None:
            print(f"**--moe-fix 식을 평가할 수 없다: {a.moe_fix!r}**")
            return 1
        traced_lit = str(value(a.moe_fix, traced, [], "--moe-fix(traced)"))
        print(f"  MoE 보정: 리터럴 {traced_lit} -> {want}  (식 {a.moe_fix!r})")
        for r in rows:
            if "moe_infer_even_split" not in (r.get("caveat") or ""):
                continue
            for fld in ("input_shape", "output_shape"):
                for grp in (r.get(fld) or []):
                    for i, t in enumerate(grp):
                        if str(t) == traced_lit:
                            grp[i] = str(want)
                            moe_fixed += 1
        print(f"  보정한 축 {moe_fixed} 자리")

    fails = []
    flops = collections.Counter()
    byts = collections.Counter()
    kinds = collections.Counter()
    ok_rows = bad_rows = 0
    weights = {}
    for r in rows:
        mult = 1
        if a.expand:
            mult = len(C.expand_layers(r.get("layers"))) or 1
        else:
            mult = int(r.get("repeat") or 1)
        f, kind = op_flops(r, env, fails)
        b = op_bytes(r, env, fails, a.dtype)
        kinds[kind] += 1
        if f is None or b is None:
            bad_rows += 1
            continue
        ok_rows += 1
        flops[r["block_type"] or "?"] += f * mult
        byts[r["block_type"] or "?"] += b * mult
        # 가중치는 이름으로 중복 제거 (하한 계산용)
        ws = r.get("weight_shape")
        if ws:
            n = numel(ws, env, fails, f"op{r['op_id']} w")
            for p in (r.get("params") or []):
                if n is not None:
                    weights[p] = n

    print()
    print("=== 대입")
    print(f"  성공 {ok_rows} 행 / 실패 {bad_rows} 행")
    print(f"  FLOPs 정확도: " + ", ".join(f"{k} {v}" for k, v in kinds.most_common()))
    if fails:
        agg = collections.Counter(f"{t} ({why})" for _w, t, why in fails)
        print(f"  **막힌 자리 {len(fails)}건, 종류 {len(agg)}**")
        for k, v in agg.most_common(8):
            print(f"    {v:>6}  {k}")
    else:
        print("  막힌 자리 없음")

    # **맨정수를 담은 행을 따로 센다.** 대입이 "성공" 해도, 원래 B 에 따라 커져야 할
    # 자리가 리터럴로 굳어 있으면 결과가 조용히 틀린다. 실패보다 위험하다.
    lit_flops = 0
    lit_rows = collections.Counter()
    lit_vals = collections.Counter()
    for r in rows:
        toks = [t for grp in ((r["input_shape"] or []) + (r["output_shape"] or []))
                for t in grp]
        bare = [t for t in toks if str(t).isdigit() and str(t) != "1"]
        if not bare:
            continue
        f, _k = op_flops(r, env, [])
        m = (len(C.expand_layers(r.get("layers"))) or 1) if a.expand             else int(r.get("repeat") or 1)
        if f:
            lit_flops += f * m
        lit_rows[r["block_type"] or "?"] += 1
        for t in bare:
            lit_vals[t] += 1

    TF = sum(flops.values())
    BY = sum(byts.values())
    WB = sum(weights.values()) * a.dtype
    print()
    print(f"=== roofline 입력 (B={env.get('B')}, T={env.get('T')}, dtype={a.dtype}B"
          + (", repeat 펼침" if a.expand else ", repeat 곱셈") + ")")
    print(f"  FLOPs        {TF:>20,}   ({TF/1e12:.2f} TFLOP)")
    print(f"  바이트 상한   {BY:>20,}   ({BY/1e9:.2f} GB)  op 마다 입출력 전부")
    print(f"  가중치 하한   {WB:>20,}   ({WB/1e9:.2f} GB)  params 이름으로 중복 제거")
    if BY:
        print(f"  산술 강도    상한 기준 {TF/BY:>8.2f} FLOP/byte"
              f"   가중치만 {TF/WB if WB else 0:>8.2f} FLOP/byte")
    print()
    print("=== **B 를 바꿔도 안 따라오는 자리** (맨정수. 대입 성공과 무관하게 위험)")
    print(f"  맨정수를 담은 행 {sum(lit_rows.values())}  "
          f"그 행의 FLOPs {lit_flops:,} ({lit_flops*100/TF if TF else 0:.1f}%)")
    for k, v in lit_vals.most_common(8):
        print(f"    {k:>8} × {v}")
    print("  이 값들이 B 에 비례해야 하는 것이면 결과가 조용히 틀린다. "
          "MANIFEST 의 caveat / moe_aggregate 를 봐야 한다.")
    print()
    print("  block_type 별 FLOPs")
    for k, v in flops.most_common():
        print(f"    {k:<14}{v:>20,}  {v*100/TF if TF else 0:5.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
