r"""모델마다 **쓸 수 있는 심볼 이름**과 선언된 유도식을 모은다.

`expr_compare` 는 값을 절대 보지 않는다. 그래서 이름 집합만 따로 만들어 넘긴다 --
이 모듈이 값에 닿는 유일한 곳이고, **이름만 내보낸다**(`names()` 는 `set[str]`).

이름은 세 곳에서 온다.

    structure.yaml `symbols`          config 에서 읽은 폭·개수
    structure.yaml `symbols_label_only`  표에는 쓰이지만 심볼표에 없던 식 (`n_chunk` 등)
    `dim_expr.namespace(provenance)`  `B`, `T`, `ctx`, `ceil` … 추적 범위에서 오는 것

`B` 와 `T` 를 빼먹으면 발행 라벨의 **절반이 "모르는 심볼"** 이 된다(49 / 132 식). 실제로
그렇게 만들었다가 전체에 돌려 보고 알았다.

실행:
    .venv\Scripts\python.exe develop\label_universe.py [model]
"""
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(PROJ, "src"))

import yaml                                                      # noqa: E402

MODELS = os.path.join(PROJ, "models")
PUBLISHED = (
    "deepseek-ai__DeepSeek-V4-Pro",
    "meta-llama__Llama-4-Maverick-17B-128E",
    "moonshotai__Kimi-K3",
    "openai__gpt-oss-120b",
    "openai__gpt-oss-20b",
)


def structure(model):
    return yaml.safe_load(io.open(os.path.join(MODELS, model, "structure.yaml"),
                                  encoding="utf-8")) or {}


def names(model, st=None):
    """**이름만.** 값은 내보내지 않는다."""
    st = st or structure(model)
    out = set(st.get("symbols") or {})
    out |= set(st.get("symbols_label_only") or {})
    try:
        import dim_expr
        prov = dim_expr.load_provenance(os.path.join(MODELS, model))
        sc = st.get("scope") or {}
        ns = dim_expr.namespace(prov, batch=sc.get("batch") or 1,
                                seq_len=sc.get("prefill_len"))
        out |= set(ns)                          # 키만 -- 값은 버린다
    except Exception as e:
        print(f"  ! namespace 를 못 읽었다 ({model}): {e}", file=sys.stderr)
    return out


def aliases(model, st=None, extra=None):
    """선언된 유도식. `symbols_label_only` 의 `expr` 와 `rules/label_aliases.yaml`."""
    st = st or structure(model)
    out = {}
    for k, v in (st.get("symbols_label_only") or {}).items():
        if isinstance(v, dict) and v.get("expr"):
            out[k] = v["expr"]
    out.update(extra or {})
    return out


def universe(model):
    """`(names, aliases)`. 비교에 넘길 것."""
    st = structure(model)
    return names(model, st), aliases(model, st)


def published_exprs(model, phases=("prefill", "decode")):
    """발행된 표에 실제로 쓰인 축 식(고유). 정수 리터럴은 뺀다."""
    seen = set()
    for ph in phases:
        p = os.path.join(MODELS, model, f"{ph}.jsonl")
        if not os.path.exists(p):
            continue
        for r in (json.loads(l) for l in io.open(p, encoding="utf-8")):
            for f in ("input_shape", "output_shape", "weight_shape"):
                v = r.get(f) or []
                shs = v if (v and isinstance(v[0], list)) else ([v] if v else [])
                for sh in shs:
                    for e in sh:
                        t = str(e)
                        if not t.lstrip("-").isdigit():
                            seen.add(t)
    return seen


def audit(models=PUBLISHED):
    """발행 라벨이 **전부 정규화되는가.** 안 되는 것은 이름을 대라."""
    import expr_compare as E
    rows, bad = [], []
    for m in models:
        nm, al = universe(m)
        ex = published_exprs(m)
        ok = 0
        for e in sorted(ex):
            try:
                E.canonical(e, nm, al)
                ok += 1
            except E.Undecidable as u:
                bad.append((m, e, u.reason))
        rows.append((m, len(ex), ok, len(ex) - ok, len(nm), len(al)))
    return rows, bad


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")
    ms = [sys.argv[1]] if len(sys.argv) > 1 else list(PUBLISHED)
    rows, bad = audit(ms)
    print(f"{'모델':<26}{'식':>5}{'정규화':>7}{'실패':>5}{'심볼':>6}{'유도':>5}")
    for m, n, ok, ng, ns, na in rows:
        print(f"{m.split('__')[-1]:<26}{n:>5}{ok:>7}{ng:>5}{ns:>6}{na:>5}")
    if bad:
        print()
        print("정규화 실패:")
        for m, e, r in bad[:40]:
            print(f"  {m.split('__')[-1]:<24}{e:<30}{r}")
        print(f"  총 {len(bad)}")
    sys.exit(1 if bad else 0)
