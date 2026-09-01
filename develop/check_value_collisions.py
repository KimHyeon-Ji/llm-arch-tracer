"""실제 트레이스 축에서 2개 이상의 서로 무관한 이름(plain 심볼 + 유도식)이 같은 값으로
겹치면서, 그 축이 slice/split/concat/reshape류(부분 분할이 여러 가지로 설명될 수 있는 연산)에
있는 자리를 찾는다 -- 등록 전에 외부 검토가 필요한 고위험 신호.
review/05-overrides.md 「값이 2개 이상 겹치면」 참고.

WHY THIS EXISTS (그리고 왜 v1이 아니라 v2인가)
-----------------------------------------------
v1(2026-09-01 오전)은 `structure.yaml`의 plain 심볼 테이블만 보고 "같은 모델 안에 값이
같은 심볼이 3개 이상"을 찾았다. 외부 검토(Codex, 2026-09-01 오후)가 실행해서 확인한 결과,
이 v1은 **자신이 잡으려던 실제 사고를 잡지 못한다**:

  - DeepSeek-V4-Pro의 실제 사고는 `n_h_I`(64)/`d_rope`(64) 단 **2개**의 plain 심볼이 겹치는
    자리였다 -- 세 번째 이름 `c_I-d_rope`는 유도식(`rules/derived_dims.yaml`)이라
    structure.yaml에 없고, v1은 애초에 유도식을 후보에 넣지 않았다.
  - 문턱을 단순히 "2개+"로 낮추면 함대 전체가 39개 모델/100건으로 3.7배 뛴다(실측,
    2026-09-01) -- `d_ff`↔`d_moe`, `n_h`↔`n_kv`처럼 정상적으로 반복되는 동치까지 전부
    잡혀서 신호가 묻힌다.

그래서 v2는 두 가지를 바꾼다:
  1. 후보 풀에 **유도식도 포함**한다 (`src/summarize.derived_symbols()`, `rule_coverage.py`가
     이미 쓰는 것과 같은 함수) -- `c_I-d_rope` 같은 이름도 이제 후보가 된다.
  2. "모델 어딘가에 값이 겹치는 심볼이 있다"가 아니라 **실제 트레이스의 특정 축**에서 겹치는지
     본다 -- 그리고 그 축이 `slice`/`split`/`concat`/`reshape` 류(부분 분할이 여러 방식으로
     설명될 수 있는 연산 -- 정확히 오늘 사고가 난 op 부류)일 때만 신호로 센다. 이러면 흔한
     "값은 같지만 아무 위험한 연산에도 안 걸리는" 동치는 자동으로 걸러진다.

이 스크립트는 **아무것도 고치지 않는다.** 여기 나온 자리는 대량 등록(positional rule,
`spread: class` 배치) 전에 외부 검토를 한 번 거치는 게 review/05-overrides.md의 방침이다.

실행:
    .venv\\Scripts\\python.exe develop\\check_value_collisions.py            # 함대 전체
    .venv\\Scripts\\python.exe develop\\check_value_collisions.py --model X  # 모델 이름 부분 일치
    .venv\\Scripts\\python.exe develop\\check_value_collisions.py --min-size 4  # 이 크기 미만 축은 제외
"""
import argparse
import collections
import io
import json
import os
import re
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
MODELS = os.path.join(PROJ, "models")
sys.path.insert(0, os.path.join(PROJ, "src"))
import summarize as _summarize  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# slice/split 류: 한 텐서를 여러 조각으로 나누는 연산 -- 조각의 경계가 "정확히 반"인지
# "특정 필드 폭만큼"인지는 소스를 읽어야 갈리고, op 모양만으로는 둘 다 그럴듯해 보인다.
# concat 은 그 반대(합치는 쪽)라 같은 이유로 위험하다. reshape/view/transpose/permute 는
# 축 자체를 재배열해 이름이 자리를 옮길 수 있다. DeepSeek-V4-Pro 사고의 op 1874(slice)/
# 1887(concat)이 정확히 이 목록의 첫 두 항목이다.
_RISK_OPS = {"slice", "split", "split_with_sizes", "narrow", "chunk",
             "concat", "cat", "view", "reshape", "transpose", "permute"}


def _symbols(model: str, models_dir: str) -> dict:
    p = os.path.join(models_dir, model, "structure.yaml")
    if not os.path.exists(p):
        return {}
    try:
        return (yaml.safe_load(io.open(p, encoding="utf-8")) or {}).get("symbols") or {}
    except (ValueError, OSError):
        return {}


def _derived_for(model: str, models_dir: str):
    """(glob, scoped) 유도식 맵 -- `rule_coverage.py:_derived_symbols_for`와 같은 방식으로
    구한다(모델 자체 심볼 + seq_len만 필요, 실 config 로드 없음)."""
    syms = _symbols(model, models_dir)
    seq_len = None
    prov_p = os.path.join(models_dir, model, "full", "provenance.json")
    if os.path.exists(prov_p):
        try:
            seq_len = (json.load(io.open(prov_p, encoding="utf-8")) or {}).get("seq_len_used")
        except (ValueError, OSError):
            pass
    try:
        return _summarize.derived_symbols(syms, cfg=None, seq_len=seq_len)
    except Exception:
        return {}, []


def _candidates_at(value: int, module_path: str, plain: dict, glob_d: dict, scoped_d: list) -> set:
    """이 (값, 모듈) 자리에 붙을 수 있는 서로 무관한 이름들 -- plain 심볼 + 유도식(전역/스코프)."""
    cands = {n for n, v in plain.items() if v == value}
    cands |= {sym for val, sym in glob_d.items() if val == value}
    for rx, m in scoped_d:
        if rx.search(module_path or ""):
            cands |= {sym for val, sym in m.items() if val == value}
    return cands


def collisions(model: str, models_dir: str | None = None, min_size: int = 3,
               min_candidates: int = 2) -> list:
    """[(value, [name,...], [(module_path, op_type, op_id, phase), ...]), ...] --
    실제 트레이스에서 위험 op에 걸린, `min_candidates`개 이상의 후보가 겹치는 축.
    `models_dir`는 테스트가 실제 모델 디렉터리를 바꿔치기할 수 있도록 받는다(모듈 전역
    `MODELS`에 의존하면 셀프테스트가 이 모듈과 호출자 양쪽을 따로 패치해야 하는 함정이
    생긴다 -- 외부 검토, 2026-09-01)."""
    mdir = models_dir if models_dir is not None else MODELS
    d = os.path.join(mdir, model)
    plain = _symbols(model, mdir)
    glob_d, scoped_d = _derived_for(model, mdir)

    by_key = collections.defaultdict(list)   # (value, frozenset(cands)) -> [(mp, op_type, op_id, phase), ...]
    for phase in ("prefill", "decode"):
        raw_p = os.path.join(d, "full", f"{phase}.trace.raw.jsonl")
        conc_p = os.path.join(d, "full", f"{phase}.shapes.concrete.jsonl")
        if not (os.path.exists(raw_p) and os.path.exists(conc_p)):
            continue
        conc = {}
        for line in io.open(conc_p, encoding="utf-8"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            conc[r.get("op_id")] = r
        for line in io.open(raw_p, encoding="utf-8"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("op_type") not in _RISK_OPS:
                continue
            c = conc.get(r.get("op_id"))
            if not c:
                continue
            mp = r.get("module_path") or ""
            for fld in ("input_shape", "output_shape", "weight_shape"):
                sv_all, cv_all = r.get(fld), c.get(fld)
                if sv_all is None or cv_all is None:
                    continue
                pairs = [(sv_all, cv_all)] if fld == "weight_shape" else list(zip(sv_all, cv_all))
                for sv, cv in pairs:
                    if not isinstance(sv, list) or not isinstance(cv, list):
                        continue
                    for cc in cv:
                        if not isinstance(cc, int) or isinstance(cc, bool) or cc < min_size:
                            continue
                        cands = _candidates_at(cc, mp, plain, glob_d, scoped_d)
                        if len(cands) >= min_candidates:
                            key = (cc, frozenset(cands))
                            sites = by_key[key]
                            if len(sites) < 6:   # 예시는 몇 개만 -- 신호가 필요하지 표가 아니다
                                sites.append((mp, r.get("op_type"), r.get("op_id"), phase))
    return sorted(([v, sorted(cands), sites] for (v, cands), sites in by_key.items()),
                  key=lambda x: (-len(x[1]), -x[0]))


def all_keys(min_size: int = 3, min_candidates: int = 2) -> set:
    """{(model, value, (name, ...)), ...} -- 지금 함대 전체에서 검출되는 모든 위험 자리.
    `develop/verify_all.py`의 게이트와 `--dump-baseline`이 공유하는 정규 형태다."""
    out = set()
    for m in sorted(mm for mm in os.listdir(MODELS) if os.path.isdir(os.path.join(MODELS, mm))):
        for val, names, _sites in collisions(m, min_size=min_size, min_candidates=min_candidates):
            out.add((m, val, tuple(names)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="", help="모델 이름 부분 일치로 한정")
    ap.add_argument("--min-size", type=int, default=3,
                    help="이 크기 미만인 축은 무시 (기본 3 -- 0/1/2는 배치·브로드캐스트로 흔하다)")
    ap.add_argument("--min-candidates", type=int, default=2,
                    help="이 개수 미만으로 겹치는 자리는 무시 (기본 2)")
    ap.add_argument("--dump-baseline", metavar="OUT.json",
                    help="지금 검출되는 전체 자리를 베이스라인 스냅샷으로 낸다 (게이트 도입 "
                         "시점에 한 번, 또는 검토 후 갱신할 때만 쓸 것 -- develop/verify_all.py "
                         "가 이 파일에 없는 '새로' 생긴 자리만 FAIL로 잡는다)")
    a = ap.parse_args()

    if a.dump_baseline:
        keys = sorted(all_keys(a.min_size, a.min_candidates))
        import json as _json
        _json.dump([list(k[:2]) + [list(k[2])] for k in keys],
                   io.open(a.dump_baseline, "w", encoding="utf-8", newline="\n"),
                   ensure_ascii=False, indent=1)
        print(f"{len(keys)}건을 {a.dump_baseline} 에 스냅샷으로 냈다.")
        return 0

    models = sorted(m for m in os.listdir(MODELS) if os.path.isdir(os.path.join(MODELS, m)))
    if a.model:
        models = [m for m in models if a.model.lower() in m.lower()]
        if not models:
            print(f"'{a.model}' 에 맞는 모델이 없다.")
            return 2

    total_flagged = 0
    for m in models:
        rows = collisions(m, min_size=a.min_size, min_candidates=a.min_candidates)
        if not rows:
            continue
        total_flagged += 1
        print(f"\n{m}")
        for val, names, sites in rows:
            ex = "; ".join(f"{mp.split('.')[-1]}:{op}(#{oid},{ph})" for mp, op, oid, ph in sites[:3])
            print(f"  값 {val}: {len(names)}개 이름 겹침 -- {', '.join(names)}  [{ex}]")

    print(f"\n{'=' * 60}")
    if total_flagged:
        print(f"{total_flagged}개 모델에 위험 자리 있음 -- 대량 등록 전에 외부 검토 권장 "
              f"(review/05-overrides.md)")
    else:
        print("위험 자리 없음.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
