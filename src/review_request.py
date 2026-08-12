"""Write the hand-off the label review starts from: `models/<model>/review_request.md`.

The pipeline is deterministic Python and stops at the point where a judgement is required. It
does not call an LLM -- doing that from inside `src/` would need an API key and would tie the
tool to one vendor. Instead it finishes by assembling everything a reviewer needs and naming
what is left undecided, so the review can be run afterwards by any LLM (or by a person) with
no preparation: see `review/`.

The request has two halves. The first is what the rules could NOT settle -- the short list. The
second is a COMPLETE INVENTORY of every name the model uses and every integer left unnamed.

The inventory exists because filtering by "what looks unresolved" has a blind spot, and it cost
us a real defect: DeepSeek-V4-Pro's mHC mixing rendered `[B, 1, 4, d_model]` where 4 is `n_hc`,
and it appeared in NONE of the four unresolved categories -- it was not a heuristic, not a square
axis, not an alias gap, not an unregistered field. It was simply an integer nobody had a reason
to look at, and a reader found it in the CSV (2026-08-10). A list of open questions can only ever
surface the questions we already know how to ask.

The inventory is small enough to read end to end: across the fleet a model averages ~22 distinct
labels, ~8 (module, unnamed integer) pairs, and ~384 (module, output shape) pairs. Accuracy is
worth those lines.
"""
import collections
import csv
import os
import re

import source_check


def _fmt_axis(a: dict) -> str:
    where = a.get("module") or a.get("op") or ""
    return f"| `{a.get('shape', '')}` | {a.get('why', '')} | {where} |"


def collect(structure: dict, sc_res: dict, fields: dict) -> dict:
    """The open questions, grouped by what kind of answer each needs."""
    q = {"unregistered": [], "square": [], "alias": [], "heuristic": [], "membership": [],
         "ambiguous": []}
    # 5. MODULE-FIELD MEMBERSHIP -- the only question raised without looking at a value. Folded to
    # one row per (module, symbol): the same accusation repeated for `t`, `matmul` and `_to_copy`
    # of one projection is one question about that projection.
    seen = {}
    for g in (sc_res.get("membership_gaps") or []):
        key = (g["module"], g["symbol"])
        agg = seen.setdefault(key, dict(g, axes=0))
        agg["axes"] += g["axes"]
    for (mod, sym), g in sorted(seen.items(), key=lambda kv: -kv[1]["axes"]):
        q["membership"].append(
            f"`{sym}` (← config `{g['field']}`) in `{mod}` — 이 모듈도 이 모듈을 만든 "
            f"`{g['owner']}` 도 그 필드를 읽지 않는다, {g['axes']}축")
    q["unregistered"] = list(structure.get("unregistered_fields") or [])
    q["square"] = list(sc_res.get("square_unconfirmed") or [])
    q["alias"] = [f"{s} ← {f}" for s, f in (sc_res.get("alias_gaps") or [])]
    # `heuristic` is a count; the axes themselves are in heuristic_examples, which is what a
    # reviewer can actually open a source file against. Fold the layer index out first: the same
    # module in 12 layers is ONE question about that module, and listing it 12 times both inflates
    # the open count and buries the other kinds of question under it.
    folded = {}
    for ex in ((structure.get("label_provenance") or {}).get("heuristic_examples") or []):
        mod = re.sub(r"\.\d+\.", ".*.", str(ex.get("module") or ""))
        key = (str(ex.get("label")), mod, str(ex.get("rule")))
        agg = folded.setdefault(key, {"axes": 0, "layers": 0})
        agg["axes"] += int(ex.get("axes") or 0)
        agg["layers"] += 1
    for (label, mod, rule), agg in folded.items():
        where = f"{mod} (레이어 {agg['layers']}개)" if agg["layers"] > 1 else mod
        q["heuristic"].append(f"`{label}` in `{where}` — {rule}, {agg['axes']}축")
    return q


def _scan_tables(model_dir: str, symbols: dict) -> dict:
    """Everything the rendered tables actually contain, folded to a readable size.

    Layer indices are normalised away (`layers.7.` -> `layers.*.`) so a 100-layer model does not
    produce 100 identical rows. Reads the FULL tables, not the major view, so nothing is hidden by
    the roll-up.
    """
    labels = collections.defaultdict(collections.Counter)   # label -> {module: axes}
    bare = collections.Counter()                            # (module, int) -> axes
    shapes = collections.defaultdict(set)                   # module -> {output shape}
    for phase in ("prefill", "decode"):
        path = os.path.join(model_dir, "full", f"{phase}.csv")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                mk = re.sub(r"\.\d+\.", ".*.", row.get("module_path") or "") or "(root)"
                if row.get("output_shape"):
                    shapes[mk].add(row["output_shape"])
                for fld in ("input_shape", "weight_shape", "output_shape"):
                    for grp in re.findall(r"\[([^\[\]]*)\]", row.get(fld) or ""):
                        for a in (x.strip() for x in grp.split(",")):
                            if not a:
                                continue
                            if a.isdigit():
                                if int(a) >= 2:
                                    bare[(mk, int(a))] += 1
                            else:
                                labels[a][mk] += 1
    by_value = collections.defaultdict(list)
    for name, val in (symbols or {}).items():
        if isinstance(val, int) and not isinstance(val, bool) and val >= 2:
            by_value[val].append(name)
    return {"labels": labels, "bare": bare, "shapes": shapes, "by_value": by_value}


def _inventory(model_dir: str, symbols: dict) -> list:
    """The full-inventory half of the request."""
    sc = _scan_tables(model_dir, symbols)
    labels, bare, shapes, by_value = sc["labels"], sc["bare"], sc["shapes"], sc["by_value"]
    L = ["## 전수 점검 — 이 모델이 쓰는 이름 전부", "",
         "위 절이 '풀리지 않은 것'이라면 여기는 **전부**다. 규칙이 자신 있게 붙인 이름도 틀릴 수 "
         "있고, 그런 건 미결 목록에 절대 오르지 않는다. 한 줄씩 읽고 **그 모듈에서 그 이름이 "
         "말이 되는지** 보라.", ""]

    L += ["### A. 붙은 이름 전부 (%d종)" % len(labels), "",
          "| 라벨 | 값 | 나타나는 모듈 | 축 수 |", "|---|---|---|---|"]
    def _val(lab):
        v = (symbols or {}).get(lab)
        return str(v) if isinstance(v, int) and not isinstance(v, bool) else ""
    for lab, mods in sorted(labels.items(), key=lambda kv: -sum(kv[1].values())):
        where = ", ".join(f"`{m}`" for m, _ in mods.most_common(4))
        if len(mods) > 4:
            where += f" 외 {len(mods) - 4}개"
        L.append(f"| `{lab}` | {_val(lab)} | {where} | {sum(mods.values())} |")
    L.append("")

    L += ["### B. 이름 없이 남은 정수 전부 (%d쌍)" % len(bare), "",
          "**여기가 필터가 못 보던 자리다.** 정수가 남는 것 자체는 정상이다(루프 인덱스, "
          "피연산자 개수, 브로드캐스트 축). 문제는 **이름이 있어야 하는데 없는 경우**이고, "
          "마지막 열이 그 신호다 — 이 모델의 심볼과 값이 같다면 스코프가 그 모듈을 못 덮고 "
          "있을 수 있다. 실제로 `n_hc`(=4)가 그렇게 정수로 남아 있었다.", "",
          "| 모듈 | 정수 | 축 수 | 같은 값의 심볼 |", "|---|---|---|---|"]
    for (mk, val), cnt in sorted(bare.items(), key=lambda kv: -kv[1]):
        hit = ", ".join(f"`{x}`" for x in by_value.get(val, []))
        L.append(f"| `{mk}` | {val} | {cnt} | {hit or '—'} |")
    L.append("")

    n_pairs = sum(len(v) for v in shapes.values())
    L += [f"### C. 모듈이 내는 출력 shape 전부 ({len(shapes)}개 모듈 / {n_pairs}종)", "",
          "모듈 하나가 어떤 모양을 내놓는지 전부 적었다. 어떤 모듈에 **있을 수 없는 이름**이 "
          "섞여 있는지 보는 자리다(예: attention head 수가 Mamba mixer 안에, 전문가 수가 "
          "self_attn 안에).", ""]
    for mk in sorted(shapes):
        L.append(f"- `{mk}`")
        for sh in sorted(shapes[mk]):
            L.append(f"  - `{sh}`")
    L.append("")
    return L

def build(model_dir: str, model_id: str, model_type: str, structure: dict,
          sc_res: dict, fields: dict) -> str:
    q = collect(structure, sc_res, fields)
    # 6. AMBIGUITY. Written by build_table from the resolver's tie log: an axis where two symbols
    # held the same value, so whichever won did so by global priority. Folded to one line per
    # (module, value, candidates) -- 352,048 such axes across the fleet collapse to 110 questions.
    _amb = os.path.join(model_dir, "full", "ambiguous.json")
    if os.path.exists(_amb):
        import json as _json
        try:
            with open(_amb, encoding="utf-8") as _fh:
                for a in (_json.load(_fh) or []):
                    q["ambiguous"].append(
                        "`%s` in `%s` — 값 %s 를 두고 후보가 %d개, %d축"
                        % (" vs ".join(a["candidates"]), a["module"], a["value"],
                           len(a["candidates"]), a["axes"]))
        except (ValueError, OSError):
            pass
    mt = model_type or "(미확인)"
    open_n = sum(len(v) for v in q.values())

    L = [f"# 검토 의뢰서 — {model_id}", "",
         "파이썬 파이프라인이 규칙으로 결정할 수 있는 것을 전부 결정하고, **판단이 필요한 것만** "
         "여기 남겼다. 절차와 출력 형식은 `review/` 에 있다.", "",
         f"- transformers 모듈: `{mt}`",
         f"- 판단 필요: **{open_n}건**", ""]

    L += ["## 증거 — 이미 받아둔 실제 소스", ""]
    for kind, ok in (("modeling", sc_res.get("modeling_ok")), ("configuration", sc_res.get("config_ok"))):
        path = f"develop/sources/{kind}_{mt}.py"   # forward slashes: this file is a deliverable
        L.append(f"- `{path}` — {'있음, 이 파일을 열어서 판정한다' if ok else '**없음** (네트워크 불가 또는 transformers 본체에 없는 아키텍처)'}")
    L += ["", f"- 온라인 원본: https://github.com/huggingface/transformers/tree/main/src/transformers/models/{mt}",
          "", "그 밖의 재료: `full/review.md`(리뷰 패킷 — shape 별 실제 행 표본), "
          "`structure.yaml`(이 모델의 심볼 표), `full/<phase>.csv`(전체 operator 표).", ""]

    L += ["## 판단이 필요한 것", ""]
    if not open_n:
        L += ["없다. 이 모델의 축은 전부 등록된 규칙이 이름을 냈고, 소스 대조도 어긋난 곳이 없다.",
              "", "그래도 검토를 돌린다면 `full/review.md` 의 표본을 보고 규칙 자체가 틀리지 "
              "않았는지를 본다 — 그것이 규칙 게이트가 구조적으로 못 보는 부분이다.", ""]
    else:
        if q["alias"]:
            L += ["### 1. 이 config 필드가 정말 이 뜻인가", "",
                  "값은 로드된 config 에 있지만 이 모델의 config 클래스가 선언한 필드가 아니다 "
                  "(체크포인트 `config.json` 에서 온 값). 클래스가 뜻을 보증하지 않으므로 "
                  "modeling 소스에서 이 필드가 실제로 어떻게 쓰이는지 확인해야 한다.", ""]
            L += [f"- `{x}`" for x in q["alias"]] + [""]
        if q["square"]:
            L += ["### 2. 이 정사각 축이 정말 같은 이름 두 번인가", "",
                  "`[..., X, X]` 로 렌더됐는데, 그 이름이 읽은 config 필드에서 나온 정사각 "
                  "reshape 을 modeling 소스에서 찾지 못했다. 두 축 크기가 우연히 같은 것일 수 있다.", ""]
            L += [f"- `{x}`" for x in q["square"]] + [""]
        if q["unregistered"]:
            L += ["### 3. 이름 붙일 근거가 없는 config 필드", "",
                  "모듈 폭으로 쓰이는데 심볼 표에 등록돼 있지 않다. 소스에서 무엇인지 확인하고 "
                  "`rules/symbols.yaml` 에 등록하면 다음 실행부터 자동으로 잡힌다.", ""]
            L += [f"- `{x}`" for x in q["unregistered"][:40]] + [""]
        if q["heuristic"]:
            L += ["### 4. 규칙 없이 산술로 지은 이름", "",
                  "값이 맞아떨어져서 붙인 이름이다. 산술적으로 참이어도 틀린 이름일 수 있으므로 "
                  "(예: RoPE 절반 차원) 소스에서 확인이 필요하다.", ""]
            L += [f"- {x}" for x in q["heuristic"][:40]] + [""]
        if q["ambiguous"]:
            L += ["### 6. 값이 겹쳐 **임의로** 고른 축", "",
                  "두 심볼이 같은 값을 갖는 자리다. 규칙에는 고를 근거가 없고, 이긴 쪽은 전역 "
                  "우선순위 — 즉 **관례**로 정해졌다. 이름이 맞을 수도 있지만 파이프라인은 그걸 "
                  "알지 못한다. 표에서는 확신 있는 라벨과 똑같이 보인다.", "",
                  "**소스를 열어 어느 쪽인지 확정하는 것이 여기서 할 일이다.** 확정되면 "
                  "`rules/label_overrides.yaml` 에 근거와 함께 못 박는다(review/05-overrides.md). "
                  "출신으로만 구별되는 경우라면 그렇게 적고 `open` 으로 남긴다.", ""]
            L += [f"- {x}" for x in q["ambiguous"][:40]] + [""]
        if q["membership"]:
            L += ["### 5. 그 모듈이 읽지도 않는 필드의 이름이 가중치 축에 붙어 있다", "",
                  "**값을 전혀 보지 않는 안건이다.** 파라미터의 shape 은 그것을 소유한 모듈이 "
                  "선언한 것이므로, 그 모듈도 그 모듈을 만든 부모도 읽지 않는 config 필드의 "
                  "이름이 붙어 있으면 산술이 맞아도 근거가 없다. 소스에서 그 "
                  "`nn.Linear`/`nn.Parameter` 를 만드는 줄을 찾아 실제 폭이 무엇인지 확인하라.", ""]
            L += [f"- {x}" for x in q["membership"][:40]] + [""]

    L += ["## 기계적으로 이미 확인된 것 — 다시 묻지 말 것", ""]
    gaps_n = len(sc_res.get("alias_gaps") or [])
    conf = sc_res.get("square_confirmed") or []
    L += [f"- **심볼이 읽은 config 필드**: {'전부 이 모델의 config 클래스(또는 상속/프로퍼티/getattr 기본값)에 존재한다' if not gaps_n else f'{gaps_n}건이 클래스 선언 밖 — 위 1절 참고'}",
          f"- **정사각 축**: " + (", ".join(f"`{lab}` ← 소스의 `{ident}` ← `{fld}`" for lab, ident, fld in conf)
                                  if conf else "소스에서 정사각 생성/reshape 과 대응이 확인된 축 없음"),
          f"- **모듈이 읽는 config 속성**: `__init__` 에서 config 를 읽는 클래스 {sc_res.get('module_reads', 0)}개를 소스에서 확인했다. "
          "그 목록이 각 모듈의 폭이 가질 수 있는 이름의 전부다."]
    if not sc_res.get("membership_ran"):
        L.append("- **가중치 축 ↔ 모듈 소속**: **수행되지 않았다** — `full/module_classes.json` 또는 "
                 "modeling 소스가 없다. 통과가 아니다.")
    elif not (sc_res.get("membership_gaps") or []):
        L.append("- **가중치 축 ↔ 모듈 소속**: 가중치 축의 이름이 전부 그 모듈(또는 그 부모)이 "
                 "실제로 읽는 config 필드에서 나왔다. 이 축들은 값이 아니라 소스로 확인된 것이다.")
    L.append("")

    L += _inventory(model_dir, structure.get("symbols") or {})

    L += ["## 이 의뢰서를 처리하는 법", "",
          "`review/prompt.md` 를 LLM 에 넘기고 이 모델을 지정한다. 판정 4종과 근거 요건, "
          "결과를 어디에 어떤 형식으로 쓰는지는 전부 `review/` 안에 있다.", ""]

    path = os.path.join(model_dir, "review_request.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    return path
