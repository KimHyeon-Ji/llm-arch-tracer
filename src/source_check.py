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


def _installed(model_type: str, kind: str) -> str | None:
    """The file the INSTALLED transformers would actually import, read off disk.

    This is the code the trace ran. GitHub `main` is not: it moves, and this environment pins
    transformers 5.14.1, so the two drift. The drift is not theoretical -- Falcon-H1's SSD scan
    carries `G_intermediate = ... # shape: (b, c, l, s, h, n)` at modeling_falcon_h1.py:818 in
    the installed 5.14.1 (1,297 lines), and the `main` copy that had been cached into
    develop/sources/ (1,204 lines) has no such line. Two reviewers reading "the source" reached
    different conclusions about what the evidence said, and both were right about their own file
    (2026-08-15). A review layer whose evidence is a DIFFERENT VERSION than the traced code can
    confirm a label that the traced code does not support, which is the one failure this whole
    layer exists to prevent.
    """
    try:
        import transformers
    except ImportError:
        return None
    base = os.path.dirname(os.path.abspath(transformers.__file__))
    cands = ([os.path.join(base, "models", model_type, f"{kind}_{model_type}.py")]
             if model_type else [os.path.join(base, f"{kind}.py")])
    for p in cands:
        if os.path.exists(p):
            try:
                with open(p, encoding="utf-8") as f:
                    return f.read()
            except OSError:
                return None
    return None


def fetch(model_type: str, kind: str = "modeling", refresh: bool = False) -> str | None:
    """Source text for `{kind}_{model_type}.py`. None if unavailable.

    Order: the INSTALLED transformers file (what actually ran) -> local cache -> GitHub `main`.
    An empty model_type means a top-level transformers file (`configuration_utils.py`).
    """
    if not model_type and kind != "configuration_utils":
        return None
    os.makedirs(CACHE, exist_ok=True)
    name = f"{kind}.py" if not model_type else f"{kind}_{model_type}.py"
    url = _ROOT + name if not model_type else _RAW.format(mt=model_type, fn=kind)
    path = os.path.join(CACHE, name)
    # The installed file wins, and it is written into the cache so the ④-layer reviewer -- who is
    # told to open develop/sources/ -- reads the same bytes this check reasoned about.
    text = _installed(model_type, kind)
    if text is not None:
        try:
            if not os.path.exists(path) or open(path, encoding="utf-8").read() != text:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(text)
        except OSError:
            pass
        return text
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


def fetch_from_repo(model_id: str, kind: str = "modeling", refresh: bool = False):
    """(source text, filename) for the model's OWN repo, or (None, None).

    `fetch` only tries transformers `main`, so an architecture that lives as remote code in the
    model repository came back empty -- and every check that needs the source then reported
    "수행되지 않음" rather than a verdict. Kimi-K2.6 / K2.7 sat there for the whole fleet: their
    `model_type` is `kimi_k2`, transformers has no such file, and the code that actually runs is
    `modeling_deepseek.py` inside the repo itself. An outside reviewer's methodology named this
    directly -- list the repo's files first, then fetch the ones you need (2026-08-12).

    The filename is returned because it is NOT derivable from model_type: the repo names its file
    after whatever architecture it forked from, and a verdict has to cite the file it actually read.
    """
    import glob
    tag = model_id.replace("/", "__")
    cached = sorted(glob.glob(os.path.join(CACHE, f"{tag}__{kind}_*.py")))
    if cached and not refresh:
        with open(cached[0], encoding="utf-8") as f:
            return f.read(), os.path.basename(cached[0]).split("__", 1)[1]
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
        names = [f for f in list_repo_files(model_id)
                 if f.endswith(".py") and os.path.basename(f).startswith(kind + "_")]
        if not names:
            return None, None
        # a repo may ship several; the shortest name is the base architecture, which is the one
        # `AutoModelForCausalLM` loads for a text model
        names.sort(key=len)
        path = hf_hub_download(model_id, names[0])
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except Exception:
        return None, None
    os.makedirs(CACHE, exist_ok=True)
    out = os.path.join(CACHE, f"{tag}__{os.path.basename(names[0])}")
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    return text, os.path.basename(names[0])


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


def square_labels(model_dir: str) -> set:
    """Labels sitting on the trailing pair of a shape that is REALLY square -- `[..., 256, 256]`.

    These are the axes no internal check can settle: two widths that are equal cannot be told
    apart by value, so the only authority is whether the source really builds a square there.

    "Really" is the word that had to be fixed. Until 2026-08-15 this asked whether the two
    trailing **labels** were the same string, which quietly assumes the answer to the question
    being asked. Falcon-H1 builds `torch.ones(chunk_size, chunk_size)` and the tie-break named
    the two axes `d_state` and `d_chunk`; because the strings differed, the square check --
    the one check written for exactly this situation (Zamba2 / Nemotron-3, ③ 라벨 검토
    2026-08-09) -- skipped it. A check that only sees the squares that are already labelled
    consistently cannot catch a square that was labelled inconsistently.

    So squareness is decided on the CONCRETE sizes (full/<phase>.shapes.concrete.jsonl) and
    BOTH labels are returned -- if they disagree, both go to the source for confirmation and
    at most one can come back confirmed. Where no sidecar row exists the old string test is
    kept, so a model traced before the sidecar existed still reports what it used to.

    Only ACTIVATIONS are asked about. A square WEIGHT is not a question: `nn.Linear(d, d)` is an
    ordinary projection whenever a model sizes n_h*d_head == d_model, and its parameter honestly
    has the same width twice. Scanning weights too made the review re-ask about q_proj on
    Qwen2.5-0.5B, v/out_proj on xLSTM, and every square projection in Zamba2 -- six models whose
    answer was "yes, it is square" each time (③ 라벨 검토 2026-08-09).
    """
    import csv
    import json
    out = set()
    for phase in ("prefill", "decode"):
        path = os.path.join(model_dir, "full", f"{phase}.csv")
        if not os.path.exists(path):
            continue
        conc = {}
        cp = os.path.join(model_dir, "full", f"{phase}.shapes.concrete.jsonl")
        if os.path.exists(cp):
            for line in open(cp, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    conc[int(r["op_id"])] = r
                except (ValueError, KeyError, TypeError):
                    pass
        for row in csv.DictReader(open(path, encoding="utf-8")):
            wl = (row.get("weight_shape") or "").strip()
            # A weight reaches an op already transposed (`t` -> `linear`/`matmul`), so its shape
            # arrives REVERSED and a plain string compare misses it. Missing it re-asks the
            # square-weight question this function exists to suppress -- 96 such rows in
            # Qwen2.5-0.5B alone, every one of them `nn.Linear(896, 896)`.
            wsh = [x.strip() for x in wl.strip("[]").split(",") if x.strip()] if wl else []
            try:
                cr = conc.get(int(row.get("op_id")))
            except (TypeError, ValueError):
                cr = None
            for fld in ("input_shape", "output_shape"):
                real = (cr or {}).get(fld) or []
                for k, grp in enumerate(re.findall(r"\[([^\[\]]*)\]", row.get(fld) or "")):
                    sh = [x.strip() for x in grp.split(",") if x.strip()]
                    if len(sh) < 2:
                        continue
                    if wsh and sh in (wsh, wsh[::-1]):
                        continue          # this operand IS the weight, in either orientation
                    r = real[k] if k < len(real) else None
                    if isinstance(r, list) and len(r) == len(sh):
                        if r[-1] != r[-2]:
                            continue
                    elif sh[-1] != sh[-2]:
                        continue          # no sidecar row -> fall back to the old string test
                    out |= {x for x in (sh[-1], sh[-2]) if not x.isdigit()}
    return out


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


# `config.hidden_size`, `self.config.hidden_size`, and `config.text_config.hidden_size` are the
# same read written three ways; a chain is walked to its root so the nesting of a multimodal
# config does not hide the field.
def _config_chain_attr(node: ast.Attribute) -> str | None:
    attr, cur = node.attr, node.value
    while isinstance(cur, ast.Attribute):
        if cur.attr == "config":
            return attr
        cur = cur.value
    return attr if isinstance(cur, ast.Name) and cur.id == "config" else None


_BASE_STOP = {"object", "nn.Module", "Module", "PreTrainedModel", "GenerationMixin",
              "ABC", "Enum"}


def class_config_reads(modeling_src: str) -> tuple[dict, set]:
    """({class: fields it reads from config}, {classes whose full read set is unknown}).

    Three things `module_config_reads` does not do, each of which turns a real read into an
    apparent absence -- and an apparent absence is what would accuse a correct label:

    * **inheritance.** `Qwen3MoeAttention(Qwen3Attention)` reads `num_attention_heads` in the base
      class only. Reads are unioned along the chain, transitively.
    * **optional reads.** `getattr(config, "head_dim", hidden_size // num_heads)` never appears as
      `config.head_dim`.
    * **nested configs.** `config.text_config.hidden_size` is a read of `hidden_size`.

    The second return value is the honest half: a class whose base is imported from another file
    (`from ..llama.modeling_llama import LlamaAttention`) has reads we cannot see, so it is listed
    as unknown and no conclusion is drawn about it. Silence about a class is not evidence.
    """
    reads, bases = {}, {}
    try:
        tree = ast.parse(modeling_src)
    except SyntaxError:
        return {}, set()
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        own = set()
        for node in ast.walk(cls):
            if isinstance(node, ast.Attribute):
                got = _config_chain_attr(node)
                if got:
                    own.add(got)
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "getattr" and len(node.args) >= 2
                    and isinstance(node.args[1], ast.Constant)
                    and isinstance(node.args[1].value, str)):
                tgt = node.args[0]
                root = tgt.id if isinstance(tgt, ast.Name) else getattr(tgt, "attr", None)
                if root == "config":
                    own.add(node.args[1].value)
        reads[cls.name] = own
        bases[cls.name] = [b.id if isinstance(b, ast.Name)
                           else (f"{b.value.id}.{b.attr}" if isinstance(b, ast.Attribute)
                                 and isinstance(b.value, ast.Name) else None)
                           for b in cls.bases]

    unknown = set()
    resolved: dict = {}

    def walk(name, seen):
        if name in resolved:
            return resolved[name]
        if name in seen:                       # defensive: a cycle cannot happen in valid Python
            return set()
        seen = seen | {name}
        acc = set(reads.get(name, ()))
        for b in bases.get(name, ()):
            if b is None or b in _BASE_STOP or b.split(".")[-1] in _BASE_STOP:
                continue
            if b in reads:
                acc |= walk(b, seen)
            else:
                unknown.add(name)              # base defined elsewhere -> reads not fully visible
        resolved[name] = acc
        return acc

    for name in list(reads):
        walk(name, frozenset())
    return resolved, unknown


def field_equivalents(config_src: str) -> dict:
    """{field: other names for the SAME field}, from `attribute_map` and from `@property` bodies.

    Two ways a class reads a field without writing its name, both of which would otherwise be
    reported as "this module never reads that field":

    * `attribute_map = {"num_local_experts": "n_routed_experts"}` -- DeepSeek's expert module reads
      `config.num_local_experts`; our alias table resolved `E` to `n_routed_experts`. Same field.
    * `@property def head_dim(self): return self.hidden_size // self.num_attention_heads` -- falcon
      and Zamba2 compute the width the module builds from the fields the module does read.
    """
    eq: dict = {}
    try:
        tree = ast.parse(config_src)
    except SyntaxError:
        return eq
    def link(a, b):
        eq.setdefault(a, set()).add(b)
        eq.setdefault(b, set()).add(a)
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        for stmt in cls.body:
            if isinstance(stmt, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "attribute_map" for t in stmt.targets):
                if isinstance(stmt.value, ast.Dict):
                    for k, v in zip(stmt.value.keys, stmt.value.values):
                        if (isinstance(k, ast.Constant) and isinstance(v, ast.Constant)
                                and isinstance(k.value, str) and isinstance(v.value, str)):
                            link(k.value, v.value)
            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
                    (isinstance(d, ast.Name) and d.id in ("property", "cached_property"))
                    or (isinstance(d, ast.Attribute) and d.attr in ("property", "cached_property"))
                    for d in stmt.decorator_list):
                for n in ast.walk(stmt):
                    if (isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                            and n.value.id == "self" and n.attr != stmt.name):
                        link(stmt.name, n.attr)
    return {k: v for k, v in eq.items()}


def _ancestors(module_key: str):
    """`model.layers.*.mixer.in_proj` -> itself, `...mixer`, `model.layers.*`, `model.layers`,
    `model`, `(root)`."""
    if module_key in ("", "(root)"):
        return ["(root)"]
    parts = module_key.split(".")
    out = [".".join(parts[:i]) for i in range(len(parts), 0, -1)]
    return out + ["(root)"]


_IDENT = re.compile(r"[A-Za-z_]\w*")


def _owner_reads(mk: str, module_classes: dict, reads: dict, unknown: set, levels: int = 2):
    """(fields the module may legitimately name, [owning module paths]) -- or (None, None).

    An `nn.Linear` reads no config; its two widths were chosen by whichever class constructed it.
    So responsibility runs up the module path to the nearest class that reads config at all, and
    then ONE more, because a width is just as often passed in by the parent:
    `DeepseekV3MLP(config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts)`
    -- the MLP reads `intermediate_size`, and only its parent knows that is the MoE width.
    Stopping at one level accused every shared-expert projection in the fleet.

    Two levels and no further, deliberately: the root model class reads nearly every field there
    is, so unioning it in would make the check unable to fire at all.
    """
    got, owners = set(), []
    for anc in _ancestors(mk):
        classes = module_classes.get(anc) or []
        if any(c in unknown for c in classes):
            return None, None                   # reads not fully visible -> no conclusion at all
        here = set()
        for c in classes:
            here |= set(reads.get(c) or ())
        if here:
            got |= here
            owners.append(anc)
            if len(owners) >= levels:
                break
    return (got, owners) if owners else (None, None)


def membership_gaps(module_classes: dict, labels_by_module: dict, symbol_fields: dict,
                    reads: dict, unknown: set, equivalents: dict | None = None) -> list:
    """Labels a module carries that the SOURCE never lets that module read. One row each.

    This is the only label check that never looks at a value. A width is `moe_intermediate_size`
    because the module that built it read `moe_intermediate_size`; if neither that class nor the
    one that constructed it ever touches the field, the name is unsupported however well the
    arithmetic works out. That is exactly the failure mode value matching cannot see -- Nemotron-3's
    Mamba mixer carried `k` (=2) because two is also a chunk count, and no metric moved.

    `labels_by_module` should be WEIGHT axes. A parameter's shape is declared by the module that
    owns it, so an unreadable field there is a defect. Activation axes are different in kind: a
    tensor flows through modules that never declared its width (`d_model` reaches every norm in
    the stack), and judging those by declared reads accuses correct labels.

    Returns [] rather than a guess whenever the source cannot settle it: no map, no class, an
    unknown base class, or a symbol with no config field behind it.
    """
    if not (module_classes and reads):
        return []
    equivalents = equivalents or {}
    gaps = []
    for mk, labels in sorted(labels_by_module.items()):
        owner_fields, owners = _owner_reads(mk, module_classes, reads, unknown)
        if owner_fields is None:
            continue
        for label in sorted(labels):
            for sym in sorted(set(_IDENT.findall(label))):
                # EVERY field this symbol may stand for, not just the one it resolved from. One
                # symbol legitimately spans several fields -- `d_moe` names both
                # `moe_intermediate_size` and `shared_expert_intermediate_size`, and the shared
                # expert reads the second. Checking only the resolved field accused every
                # shared-expert projection in four models of a name that is exactly right.
                flds = symbol_fields.get(sym) or set()
                if isinstance(flds, str):
                    flds = {flds}
                if not flds or flds & owner_fields:
                    continue
                # the same field under another name: an attribute_map alias, or a @property whose
                # ingredients the module demonstrably reads
                if any((equivalents.get(f) or set()) & owner_fields for f in flds):
                    continue
                gaps.append({"module": mk, "label": label, "symbol": sym,
                             "field": "/".join(sorted(flds)),
                             "owner": " / ".join(owners), "axes": labels.get(label, 0)})
    return gaps


def _weight_axes(model_dir: str) -> dict:
    """{module path: {label: axes}} over WEIGHT shapes only, layer indices folded away."""
    import collections
    import csv
    out = collections.defaultdict(collections.Counter)
    for phase in ("prefill", "decode"):
        path = os.path.join(model_dir, "full", f"{phase}.csv")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                ws = row.get("weight_shape") or ""
                if not ws or ws == "[]":
                    continue
                mk = re.sub(r"\.\d+\.", ".*.", row.get("module_path") or "") or "(root)"
                for grp in re.findall(r"\[([^\[\]]*)\]", ws):
                    for a in (x.strip() for x in grp.split(",")):
                        if a and not a.isdigit():
                            out[mk][a] += 1
    return out


def run(model_dir: str, model_id: str, model_type: str, symbols_used: dict,
        square_labels: set, alias_map: dict | None = None) -> dict:
    """Everything the static sources can say about this model's labels."""
    cfg = fetch(model_type, "configuration")
    mdl = fetch(model_type, "modeling")
    # transformers 본체에 없으면 **모델 저장소의 remote code** 를 본다. 그게 실제로 도는 코드다.
    src_from = "transformers"
    src_files = {}
    if not mdl:
        mdl, _fn = fetch_from_repo(model_id, "modeling")
        if mdl:
            src_from, src_files["modeling"] = "repo", _fn
    if not cfg:
        _c, _fn = fetch_from_repo(model_id, "configuration")
        if _c:
            cfg, src_from, src_files["configuration"] = _c, "repo", _fn
    res = {"model_type": model_type, "config_ok": bool(cfg), "modeling_ok": bool(mdl),
           "source_from": src_from, "source_files": src_files,
           "alias_gaps": [], "square_confirmed": [], "square_unconfirmed": [],
           "module_reads": 0, "membership_gaps": [], "membership_ran": False}
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

        # D. MODULE-FIELD MEMBERSHIP. Needs the module path -> class map, which only exists once
        # the model has been built (introspect.module_classes, written to full/module_classes.json
        # at trace time and backfilled by develop/backfill_module_classes.py). No map means the
        # check did not run -- which is reported as such, never as a pass.
        mcp = os.path.join(model_dir, "full", "module_classes.json")
        if alias_map and os.path.exists(mcp):
            import json
            with open(mcp, encoding="utf-8") as fh:
                classes = json.load(fh)
            reads, unknown = class_config_reads(mdl)
            # Only symbols whose config field we actually know are checked. `symbols_used` is that
            # record; a symbol missing from it took its value by some other route (Llama-4's
            # `d_moe` comes off the nested text config, and neither of its aliases is on the
            # object we resolved against). We cannot say which field such a symbol read, so we say
            # nothing about it rather than accuse a name of not matching a field we guessed.
            checkable = {s: f for s, f in alias_map.items() if symbols_used.get(s)}
            res["membership_gaps"] = membership_gaps(
                classes, _weight_axes(model_dir), checkable, reads, unknown,
                field_equivalents(cfg or ""))
            res["membership_ran"] = True
            res["membership_unknown"] = sorted(unknown)

    # Persisted so develop/verify_all.py can enforce this every run without needing the config
    # object, the network, or a model build. Written even when the check could not run -- the
    # gate must be able to tell "clean" from "never happened".
    import json
    with open(os.path.join(model_dir, "full", "membership.json"), "w", encoding="utf-8") as fh:
        json.dump({"ran": res["membership_ran"], "gaps": res["membership_gaps"],
                   "unknown_classes": res.get("membership_unknown") or []},
                  fh, ensure_ascii=False, indent=1)
    return res
