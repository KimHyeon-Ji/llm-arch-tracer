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
import json
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


def _rows(model_dir: str) -> list:
    """Every distinct ROW of the folded tables, both phases. The first thing to read.

    A/B/C below fold the tables by (module, label), and that fold destroys the row -- which is
    exactly where the defects an outside reviewer found were living. All three of them were visible
    only by reading `input_shape`, `weight_shape` and `output_shape` side by side:

      * `[T, d_model] @ [d_model, n_kv*d_head]` while the SAME weight's `weight_shape` said
        `[n_kv*d_head, n_h*d_head]` -- one tensor, two names (4,406 rows, 25 models)
      * `transpose [B, 32, T] -> [B, T, d_head/2]` -- a transpose cannot rename an axis
      * MLA's `q_b_proj` carrying a fused-QKV width, visible as `[T, c_q] @ [c_q, ·]` where the
        output width should come from q_head_dim

    None of them appear in a per-label view. The reviewer was also FAST, and this is why: folding
    only the layer index leaves a median of 62 rows per model (max 136). That is smaller than the
    label inventory and says more. Read it top to bottom.
    """
    sigs = collections.OrderedDict()
    for phase in ("prefill", "decode"):
        path = os.path.join(model_dir, f"{phase}.jsonl")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                mk = re.sub(r"\.\d+\.", ".*.", r.get("module_path") or "") or "(root)"
                mk = re.sub(r"\.\d+$", ".*", mk)
                key = (phase, mk, r.get("op_type"), str(r.get("input_shape")),
                       str(r.get("weight_shape")), str(r.get("output_shape")))
                sigs[key] = sigs.get(key, 0) + 1
    L = ["## 행 단위 전건 — 여기부터 읽는다", "",
         "접힌 표(`<phase>.jsonl`)의 **고유 행 전부**다. 레이어 인덱스만 접었고 그 밖에는 아무것도 "
         "합치지 않았다. 아래 A/B/C 절은 (모듈, 라벨)로 접은 뷰라 **한 행 안의 어긋남이 보이지 "
         "않는다** — 실제로 외부 검토가 찾아낸 결함 세 건이 전부 그 자리에 있었다.", "",
         "**한 행씩 읽고 이것만 물어라: 입력·가중치·출력이 서로 말이 되는가.**", "",
         "- 같은 텐서가 `input_shape` 와 `weight_shape` 에서 다른 이름을 쓰지 않는가 "
         "(가중치는 `[out, in]` 으로 저장되고 피연산자는 전치돼 있다)",
         "- 전치·view 처럼 **이름을 바꿀 수 없는 op** 이 이름을 바꾸지 않았는가",
         "- 행렬곱의 수축 축이 양쪽에서 같은 이름인가 — `[m,k] @ [k,n] -> [m,n]`",
         "- 이 모듈이 그 이름을 가질 수 있는가 (소스에서 그 `nn.Linear` 를 만드는 줄을 찾아라)", "",
         f"고유 행 {len(sigs)}개.", "",
         "| phase | 모듈 | op | input_shape | weight_shape | output_shape |",
         "|---|---|---|---|---|---|"]
    for (phase, mk, op, ins, w, outs) in sigs:
        L.append(f"| {phase} | `{mk}` | {op} | `{ins}` | `{w}` | `{outs}` |")
    L.append("")
    return L


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
    if sc_res.get("source_from") == "repo":
        _sf = sc_res.get("source_files") or {}
        L += ["", "> 이 아키텍처는 transformers 본체에 없다. 위 소스는 **모델 저장소의 remote "
              "code** 에서 받은 것이고(" + ", ".join(f"`{v}`" for v in _sf.values()) +
              "), 그게 실제로 도는 코드다. 파일 이름은 이 모델이 갈라져 나온 아키텍처를 따르므로 "
              "model_type 과 다를 수 있다."]
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

    # 규칙이 스스로 "여기까지"라고 선언한 축. 정규식과 우선순위를 더 비트는 대신 소스를 읽는
    # 층에 넘긴다 -- 그 길로 세 번 갔다가 세 번 되돌렸다(2026-08-13~14).
    _uns = []
    for _ph in ("prefill", "decode"):
        _p = os.path.join(model_dir, "full", f"{_ph}.unsettled.json")
        if os.path.exists(_p):
            try:
                with open(_p, encoding="utf-8") as _f:
                    _uns += (json.load(_f) or {}).get("items") or []
            except (ValueError, OSError):
                pass
    if _uns:
        _seen, _uq = set(), []
        for _it in sorted(_uns, key=lambda x: -x.get("axes", 0)):
            _k = (_it["module"], _it["size"], _it["current_label"],
                  _it.get("axis_pos"), _it.get("anchor_shape"))
            if _k in _seen:
                continue
            _seen.add(_k)
            _uq.append(_it)
        L += ["### 0. 규칙이 끝내지 못한 축 — **여기부터 답한다**", "",
              "값으로는 결정할 수 없어 파이프라인이 판단을 넘긴 자리다. 세 가지뿐이다:",
              "`tie`(두 심볼이 같은 값이라 관례로 골랐다) · `heur`(등록 규칙이 없어 산술로 "
              "지어냈다) · `bare`(이름을 못 붙였는데 크기가 커서 진짜 차원일 수 있다).", "",
              "**답이 나오면 `override_stub` 을 채워 `rules/label_overrides.yaml` 에 넣는다.** "
              "`spread: class` 라 그 축이 지나는 모든 자리가 한 번에 바뀐다 — 모듈 경계에서 "
              "멈추지 않는다(그것이 예전에 교정을 막던 유일한 이유였다).", "",
              "**값이 같은 심볼이 여럿이면 값으로는 영원히 못 가른다. shape 안의 위치가 "
              "말해 준다** — `[B, n_h, T, d_head]` 의 축 1 은 head 개수, 축 3 은 head 폭이다.", "",
              "아래 `shape` 과 `축` 은 **그 축을 처음 만든 자리(앵커)** 의 것이다. 초안은 "
              "`shape`/`axis`/`field`/`shape_index`/`op_type`/`nth` 여섯으로 그 앵커를 "
              "지목한다 — `shape`+`axis` 만으로는 부족하다(Kimi 의 `[B, n_h, T, d_nope]` 축 3 은 "
              "**366개 등가류**에 걸쳐 있다: q 의 q_pass, KV 의 k_nope, value_states …). "
              "`nth` 는 그 모듈 안에서 같은 op_type 의 몇 번째인지다 — MLA 는 `self_attn` 안에 "
              "`split_with_sizes` 가 q용·kv용 둘이라 그것 없이는 못 가른다.", "",
              "**유일성은 실제로 돌려 봐서 검증한다**: 그 조건에 맞는 자리들이 몇 개의 등가류에 "
              "속하는지 세고, **한 레이어 안에서 둘 이상**이면 `stub_ambiguous` 를 붙인다. "
              "그 초안은 쓰지 말고 `open` 으로 남길 것.", "",
              "| 왜 | 모듈 | 크기 | 지금 이름 | 후보 | 축 | 앵커 shape | 축 수 |",
              "|---|---|---|---|---|---|---|---|"]
        for _it in _uq[:40]:
            _c = ", ".join(f"`{x}`" for x in _it.get("candidates") or []) or "—"
            _sh = _it.get("anchor_shape") or "—"
            L += [f"| `{_it['why']}` | `{_it['module']}` | {_it['size']} | "
                  f"`{_it['current_label']}` | {_c} | {_it.get('anchor_axis', '—')} | "
                  f"`{_sh}` | {_it.get('axes', 0)} |"]
            if _it.get("stub_ambiguous"):
                # 초안이 유일하지 않으면 **쓰지 말라고 표에 적는다.** 못 쓰는 초안을 조용히
                # 주는 것은 안 주느니만 못하다 -- 외부 검토가 그 이유로 작업을 거부했다.
                L += [f"| ⚠ | `{_it['module']}` | | | | | **{_it['stub_ambiguous']}** | |"]
        L += ["", "**고칠 것과 맞는 것 둘 다 적는다.** 이름이 틀렸으면 아래 초안의 `to`/"
              "`source` 를 채워 `rules/label_overrides.yaml` 에, **지금 이름이 맞으면** 같은 "
              "앵커에 `to` 대신 `label: <지금 이름>` 과 `source` 를 적어 "
              "`rules/label_confirmed.yaml` 에 넣는다. 확인을 적지 않으면 그 축은 재생성마다 "
              "다시 질문으로 올라온다.", "",
              "초안(그대로 복사해 `to` 와 `source` 만 채운다):", "", "```yaml"]
        for _it in _uq[:6]:
            _st = _it["override_stub"]
            L += [f"  - model: {os.path.basename(os.path.normpath(model_dir))}",
                  f"    module: '{_st['module']}'",
                  f"    spread: class",
                  f"    shape: {json.dumps(_st['shape'], ensure_ascii=False)}",
                  f"    axis: {_st['axis']}",
                  f"    field: {_st['field']}",
                  f"    shape_index: {_st['shape_index']}",
                  f"    op_type: {_st['op_type']}",
                  f"    nth: {_st['nth']}",
                  f"    from: {_st['from']}",
                  f"    to: {_st['to']}",
                  f"    expect: {_st['expect']}",
                  f"    source: {_st['source']}"]
        L += ["```", ""]

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

    L += _rows(model_dir)
    L += _inventory(model_dir, structure.get("symbols") or {})

    L += ["## 이 의뢰서를 처리하는 법", "",
          "`review/prompt.md` 를 LLM 에 넘기고 이 모델을 지정한다. 판정 4종과 근거 요건, "
          "결과를 어디에 어떤 형식으로 쓰는지는 전부 `review/` 안에 있다.", ""]

    path = os.path.join(model_dir, "review_request.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    return path
