"""Step 7 -- raw aten -> human-readable op_type. Rules live in rules/optype_map.yaml so
Tier 2/3 (02-new-module-handling.md) can extend them permanently without touching code."""
import os
import yaml

_DEFAULT_RULES_PATH = os.path.join(
    os.path.dirname(__file__), "..", "rules", "optype_map.yaml"
)


def load_rules(path: str = _DEFAULT_RULES_PATH) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def optype(raw_op: str, rules: dict) -> tuple[str, bool]:
    """Returns (op_type, unmapped). unmapped=True surfaces the op instead of hiding it
    (P8) -- these get reported via C16 and routed to Tier 2/3 if they matter."""
    if raw_op in rules.get("exact", {}):
        return rules["exact"][raw_op], False
    for pattern, label in rules.get("contains", {}).items():
        if pattern in raw_op:
            return label, False
    fallback = raw_op.replace("aten.", "").split(".")[0] if raw_op.startswith("aten.") else raw_op
    return fallback, True


def normalize_rows(rows: list[dict], rules_path: str = _DEFAULT_RULES_PATH) -> list[dict]:
    rules = load_rules(rules_path)
    for row in rows:
        op_type, unmapped = optype(row["raw_op"], rules)
        row["op_type"] = op_type
        row["unmapped"] = unmapped
    return rows
