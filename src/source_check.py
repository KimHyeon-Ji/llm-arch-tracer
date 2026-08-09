"""Cross-check the labels against the model's real modeling/configuration source. Deterministic.

The gate proves the output is consistent with our RULES. It cannot prove the rules are right --
and where two config values happen to be equal at the traced seq_len, no amount of internal
consistency decides which name an axis deserves. That check has to come from outside, and the
authority is the source HuggingFace actually ran.

Most of that comparison is mechanical, so it runs on every regeneration with no LLM and no API
key. Three checks, each decidable by reading the source:

  A. ALIAS GROUNDING  -- every config field our symbol table read for this model must exist in
     that model's own `configuration_*.py`. If `d_model` came from `hidden_size`, the string
     `hidden_size` has to be there. A symbol resolved from a field the config class does not
     define means the alias list matched something incidental.

  B. SQUARE MATRICES  -- a trailing `[..., X, X]` is only legitimate when the source really
     builds a square tensor there. Look for a reshape whose last two arguments are the same
     expression (`view(*shape[:-1], hc, hc)`). Confirms the repeat rather than assuming it.

  C. MODULE WIDTHS    -- collect the config attributes each module class reads in `__init__`.
     Those are the only names that module's widths can legitimately carry, which is what
     `anchors.py` claims from the live model. Agreement between a static read of the source and
     a dynamic read of the built module is real corroboration; disagreement is a finding.

Sources are cached under `develop/sources/` so the check keeps working offline and so a reader
can see exactly which text a verdict was based on. No network and no cache -> the report says
the source was unavailable, never that the check passed.
"""
import ast
import os
import re
import urllib.error
import urllib.request

_ROOT = "https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/"
_RAW = _ROOT + "models/{mt}/{fn}_{mt}.py"
CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "develop", "sources")


def fetch(model_type: str, kind: str = "modeling", refresh: bool = False) -> str | None:
    """Source text for `{kind}_{model_type}.py`, from cache or GitHub. None if unavailable.

    An empty model_type means a top-level transformers file (`configuration_utils.py`).
    """
    if not model_type and kind != "configuration_utils":
        return None
    os.makedirs(CACHE, exist_ok=True)
    name = f"{kind}.py" if not model_type else f"{kind}_{model_type}.py"
    url = _ROOT + name if not model_type else _RAW.format(mt=model_type, fn=kind)
    path = os.path.join(CACHE, name)
    if os.path.exists(path) and not refresh:
        with open(path, encoding="utf-8") as f:
            return f.read()
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            text = r.read().decode("utf-8", "replace")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        # The text tower of a multimodal model has its own model_type (`gemma3_text`) but no
        # file of its own -- Gemma3TextConfig / Gemma3TextModel live in the parent's file. Without
        # this the check reports "source unavailable" for exactly the models we do trace.
        if model_type.endswith("_text"):
            return fetch(model_type[:-len("_text")], kind, refresh)
        return None
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


def _config_fields(config_src: str) -> set:
    """Every field a configuration class declares, across all three forms transformers uses.

    transformers 5.x declares config fields as class-level annotations on an `@strict`
    dataclass (`hidden_size: int = 4096`); 4.x wrote them as `__init__` kwargs assigned to
    `self.`. Both appear in the wild -- a checked-out `main` is 5.x, but a model whose file
    predates the migration still carries the old shape. Reading only one form made every field
    look undefined, which would turn this check into noise that is always red.
    """
    names = set(re.findall(r"^\s*self\.([A-Za-z_][A-Za-z_0-9]*)\s*=", config_src, re.M))
    try:
        tree = ast.parse(config_src)
    except SyntaxError:
        return names
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__init__":
            a = node.args
            names.update(x.arg for x in list(a.args) + list(a.kwonlyargs))
        elif isinstance(node, ast.ClassDef):
            for stmt in node.body:
                # dataclass field: `vocab_size: int = 151936`
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    names.add(stmt.target.id)
                # computed field: `@property def head_dim(self): return hidden_size // n_heads`.
                # Falcon and xLSTM expose head_dim / v_head_dim this way, so reading only
                # declarations reported five models as ungrounded on a field the class defines.
                elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
                        (isinstance(d, ast.Name) and d.id in ("property", "cached_property"))
                        or (isinstance(d, ast.Attribute) and d.attr in ("property", "cached_property"))
                        for d in stmt.decorator_list):
                    names.add(stmt.name)
                # attribute_map = {"n_embd": "hidden_size"} -- both sides are valid config names
                elif isinstance(stmt, ast.Assign) and any(
                        isinstance(t, ast.Name) and t.id == "attribute_map" for t in stmt.targets):
                    if isinstance(stmt.value, ast.Dict):
                        for k, v in zip(stmt.value.keys, stmt.value.values):
                            for c in (k, v):
                                if isinstance(c, ast.Constant) and isinstance(c.value, str):
                                    names.add(c.value)
    return names


def base_fields() -> set:
    """Fields every config inherits from `PreTrainedConfig` (tie_word_embeddings, ...).

    A model's own file does not redeclare inherited fields, so without these a symbol reading a
    perfectly ordinary base field would be reported as ungrounded.
    """
    src = fetch("", "configuration_utils")
    return _config_fields(src) if src else set()


_OPTIONAL_READ = re.compile(r"getattr\s*\(\s*config\s*,\s*[\"'](\w+)[\"']")


def optional_config_reads(modeling_src: str) -> set:
    """Fields the modeling code reads as `getattr(config, "X", <default>)`.

    An optional override is grounded even though the config class never declares it: Qwen3-MoE and
    GLM-4.5 both do `self.head_dim = getattr(config, "head_dim", hidden_size // num_heads)`, so a
    checkpoint that sets head_dim IS the authority for that width. Treating it as ungrounded
    flagged a field the model demonstrably uses.
    """
    return set(_OPTIONAL_READ.findall(modeling_src or ""))


def check_aliases(symbols_used: dict, config_src: str, extra: set | None = None) -> list:
    """[(symbol, field)] whose field is NOT defined by this model's configuration class."""
    fields = _config_fields(config_src) | (extra or set())
    return [(sym, fld) for sym, fld in symbols_used.items()
            if fld and fld not in fields]


_SQUARE = re.compile(r"\.(?:view|reshape)\s*\([^)]*?([A-Za-z_][\w.]*)\s*,\s*\1\s*\)")
# A square tensor is not always reshaped into existence -- a causal mask is BUILT square:
# `torch.tril(torch.ones(chunk_size, chunk_size, ...))` in every Mamba2 chunked scan. Looking
# only for reshapes left Zamba2 and Nemotron-3 with an unanswerable question about `d_chunk`
# whose answer was sitting in the source (③ 라벨 검토 2026-08-09).
_SQUARE_NEW = re.compile(r"\b(?:ones|zeros|empty|full|rand|randn|eye)\s*\(\s*"
                         r"([A-Za-z_][\w.]*)\s*,\s*\1\s*[,)]")


def square_reshapes(modeling_src: str) -> set:
    """Identifiers X that a square tensor is built from -- reshaped `(..., X, X)` or allocated."""
    out = {m.group(1).split(".")[-1] for m in _SQUARE.finditer(modeling_src)}
    out |= {m.group(1).split(".")[-1] for m in _SQUARE_NEW.finditer(modeling_src)}
    return out


def ident_to_field(modeling_src: str) -> dict:
    """{local identifier: the config field its value came from}.

    A width reaches its use site through a short chain of rebindings -- DeepSeek-V4 writes
    `self.hc_mult = config.hc_mult` in `__init__` and `hc = self.hc_mult` in `forward`, then
    reshapes with `hc`. Following that chain is what turns "the source has *a* square reshape"
    into "the source builds a square from *this* field", which is the claim a label needs.
    """
    direct, via_self = {}, {}
    for m in re.finditer(r"^\s*(?:self\.)?([A-Za-z_]\w*)\s*=\s*(?:self\.)?config\.([A-Za-z_]\w*)",
                         modeling_src, re.M):
        direct[m.group(1)] = m.group(2)
    for m in re.finditer(r"^\s*([A-Za-z_]\w*)\s*=\s*self\.([A-Za-z_]\w*)\s*$",
                         modeling_src, re.M):
        via_self[m.group(1)] = m.group(2)
    out = dict(direct)
    for local, attr in via_self.items():
        if attr in direct:
            out.setdefault(local, direct[attr])
    return out


def module_config_reads(modeling_src: str) -> dict:
    """{class name: {config attribute it reads in __init__}} -- e.g. from `config.hidden_size`."""
    out = {}
    try:
        tree = ast.parse(modeling_src)
    except SyntaxError:
        return out
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        reads = set()
        for node in ast.walk(cls):
            if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                    and node.value.id == "config"):
                reads.add(node.attr)
        if reads:
            out[cls.name] = reads
    return out


def run(model_dir: str, model_id: str, model_type: str, symbols_used: dict,
        square_labels: set) -> dict:
    """Everything the static sources can say about this model's labels."""
    cfg = fetch(model_type, "configuration")
    mdl = fetch(model_type, "modeling")
    res = {"model_type": model_type, "config_ok": bool(cfg), "modeling_ok": bool(mdl),
           "alias_gaps": [], "square_confirmed": [], "square_unconfirmed": [],
           "module_reads": 0}
    if cfg:
        res["alias_gaps"] = check_aliases(symbols_used, cfg,
                                          base_fields() | optional_config_reads(mdl))
    if mdl:
        found = square_reshapes(mdl)
        chain = ident_to_field(mdl)
        # Our symbol names are ours; the source uses its own identifiers. The two meet at the
        # config field: our symbol records which field it read, and the chain records which field
        # the reshaped identifier came from. Equal fields is a confirmation; anything else is
        # left unconfirmed rather than waved through.
        for lab in sorted(square_labels):
            fld = symbols_used.get(lab)
            # A label with no config field behind it cannot be checked this way, and the common
            # case is not even a reshape: `[..., T, T]` is an attention score matrix, square
            # because Q @ K^T contracts the head dim, not because anything was reshaped. Asking
            # the source for a `view(T, T)` would report every attention model as unconfirmed.
            if not fld:
                continue
            hit = next((i for i in found if chain.get(i) == fld), None)
            if hit:
                res["square_confirmed"].append((lab, hit, fld))
            else:
                res["square_unconfirmed"].append(lab)
        res["square_idents"] = sorted(found)
        res["module_reads"] = len(module_config_reads(mdl))
    return res


def write_report(model_dir: str, res: dict) -> str:
    L = ["# 소스 대조 결과 (자동)", "",
         "모델의 실제 `modeling_*.py` / `configuration_*.py` 를 받아 라벨을 대조한 결과다. "
         "LLM 없이 매 재생성마다 돌며, 받은 소스는 `develop/sources/` 에 남는다.", ""]
    mt = res.get("model_type") or "(미확인)"
    L += [f"- transformers 모듈: `{mt}`",
          f"- configuration 소스: {'확보' if res.get('config_ok') else '**미확보**'}",
          f"- modeling 소스: {'확보' if res.get('modeling_ok') else '**미확보**'}", ""]
    if not (res.get("config_ok") and res.get("modeling_ok")):
        L += ["> 소스를 못 받았다(네트워크 없음, 또는 이 아키텍처가 transformers 본체에 없음). "
              "**검사를 통과한 것이 아니라 수행되지 않은 것이다.**", ""]

    gaps = res.get("alias_gaps") or []
    L += ["## A. 심볼이 읽은 config 필드가 실제로 존재하는가", ""]
    if not res.get("config_ok"):
        L.append("소스 미확보로 판정 불가.")
    elif gaps:
        # Precisely what was and was not established: the value IS on the loaded config object
        # (that is where it was read from), but the class does not declare the field -- it came
        # from the checkpoint's config.json. That is common and often fine; it just means the
        # config class contract does not vouch for what the field means.
        L += ["아래 필드는 로드된 config 객체에는 있지만 **이 모델의 config 클래스가 선언하지 "
              "않는다** — 값의 출처가 체크포인트 `config.json` 이라는 뜻이다. 대개 정상이지만, "
              "클래스가 뜻을 보증하지 않으므로 modeling 소스에서 실제 쓰임을 확인해야 한다.", "",
              "| 심볼 | 읽은 필드 |", "|---|---|"]
        L += [f"| `{s}` | `{f}` |" for s, f in gaps]
    else:
        L.append("이 모델이 쓴 심볼의 config 필드가 전부 `configuration_*.py` 에 존재한다.")
    L.append("")

    L += ["## B. 정사각 축이 소스의 정사각 reshape 과 맞는가", ""]
    if not res.get("modeling_ok"):
        L.append("소스 미확보로 판정 불가.")
    else:
        conf, unconf = res.get("square_confirmed") or [], res.get("square_unconfirmed") or []
        if not (conf or unconf):
            L.append("정사각으로 렌더된 축이 없다.")
        else:
            idents = ", ".join(f"`{x}`" for x in (res.get("square_idents") or [])) or "없음"
            L.append(f"소스에서 찾은 정사각 reshape 식별자: {idents}")
            L.append("")
            if conf:
                L += ["| 축 이름 | 소스 식별자 | config 필드 |", "|---|---|---|"]
                L += [f"| `{lab}` | `{ident}` | `{fld}` |" for lab, ident, fld in conf]
                L.append("")
            if unconf:
                L += ["**미확인** — 이 이름이 읽은 config 필드에서 나온 정사각 reshape 을 "
                      "소스에서 찾지 못했다(확인 필요): "
                      + ", ".join(f"`{x}`" for x in unconf)]
    L += ["", "## C. 모듈이 읽는 config 속성", "",
          f"`__init__` 에서 config 를 읽는 클래스 {res.get('module_reads', 0)}개를 소스에서 확인했다. "
          "이 목록이 그 모듈의 폭이 가질 수 있는 이름의 전부다 — `src/anchors.py` 가 "
          "빌드된 모델에서 읽어오는 값과 같은 출처이며, 서로 어긋나면 그것이 발견이다.", ""]
    path = os.path.join(model_dir, "source_check.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    return path
