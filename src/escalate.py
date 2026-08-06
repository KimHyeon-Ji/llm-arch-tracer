"""Tier 3 -- human verification packet. Minimal repro + a closed-form question, never
a free-form "what is this?". See 02-new-module-handling.md Tier 3.

**NOT WIRED (verified 2026-08-06): nothing in the codebase imports this module.** Kept, not
deleted, because the packet shape is the right one -- but read the two reasons before wiring it:

  - `unmapped` is NOT the trigger. It means "no entry in rules/optype_map.yaml", not "we cannot
    tell what this op is": 177,411 of 260,812 fleet ops are unmapped and every one is ordinary
    plumbing (view / t / expand / transpose / clone / slice) whose ATen name is already the right
    label. Wiring on that condition produces 177k packets and zero information.
  - The real trigger is `adapt.AdaptationExhausted` -- the retry loop meeting an error no
    registered remedy matches. But that fires DURING the trace, so neither `op_id` nor `rows`
    exists yet, and `build_packet()` below assumes both. The packet would have to be
    "error + remedy history + partial trace", which is a different shape than what is built here.

Across 26 models the adaptation loop applied exactly one remedy kind (a BF16 RuntimeError, twice)
and never exhausted, so there has been no real case to design against. Wire it when one appears
-- guessing the design now is how a module ends up unwired in the first place.
"""
import json
import os
import time


def extract_upstream(rows: list[dict], op_id: int, depth: int = 3) -> list[dict]:
    """Smallest subgraph needed to reproduce the failing op: walk depends_on backwards
    up to `depth` hops."""
    by_id = {r["op_id"]: r for r in rows}
    frontier = {op_id}
    seen = set()
    for _ in range(depth):
        nxt = set()
        for oid in frontier:
            row = by_id.get(oid)
            if not row:
                continue
            seen.add(oid)
            nxt.update(row.get("depends_on", []))
        frontier = nxt - seen
        if not frontier:
            break
    seen.update(frontier)
    return [by_id[i] for i in sorted(seen) if i in by_id]


def build_packet(op_id, rows, cfg_diff, remedy_history, research_findings) -> dict:
    by_id = {r["op_id"]: r for r in rows}
    row = by_id.get(op_id, {})
    return {
        "op_id": op_id,
        "raw_op": row.get("raw_op"),
        "module_path": row.get("module_path"),
        "min_repro_subgraph": extract_upstream(rows, op_id, depth=3),
        "relevant_config_fields": cfg_diff,
        "tried": remedy_history,
        "candidates": research_findings,  # from Tier 2, each item should cite its source
        "question": None,
        "options": None,
    }


def ask_human(packet: dict) -> dict:
    packet["question"] = (
        "Is this op (A) a deterministic structural computation, or "
        "(B) data-dependent routing/control-flow?"
    )
    packet["options"] = [
        "A: structural -- only needs an op_type label",
        "B: data-dependent -- needs symbolic treatment (see MoE handling in 01-main.md C8)",
    ]
    return packet


def write_packet(model_dir: str, packet: dict) -> str:
    os.makedirs(os.path.join(model_dir, "escalations"), exist_ok=True)
    n = int(time.time())
    path = os.path.join(model_dir, "escalations", f"packet_{n}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(packet, f, indent=2, default=str)
    return path


def record_human_answer(packet_path: str, decision: str):
    with open(packet_path, encoding="utf-8") as f:
        packet = json.load(f)
    packet["human_decision"] = decision
    packet["source"] = "human"
    with open(packet_path, "w", encoding="utf-8") as f:
        json.dump(packet, f, indent=2, default=str)
    # Caller is responsible for also folding this into rules/optype_map.yaml or
    # rules/error_remedies.yaml so the pattern is handled automatically next time.
