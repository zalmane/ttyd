#!/usr/bin/env python3
"""Talk to your data — agentic CLI.

Decomposes questions into: reuse existing concepts, patch policies, or build fresh.
Logs every step: plan, CRL, SQL, results.

Usage:
    python ask.py                              # interactive mode
    python ask.py "What is our total ARR?"     # single question
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from setup_db import create_db
from compact_rl import (
    riverlang_to_compact, compact_to_modelset, split_crl, merge_crl,
)
from llm_runner import _call_llm, run_query
from riverlang.ast.parser import RiverLangParser
from riverlang.compiler.compiler import Compiler
from riverlang.compiler.fqn import FQN

# ---------------------------------------------------------------------------
# Colors (ANSI)
# ---------------------------------------------------------------------------
DIM = "\033[2m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"

# ---------------------------------------------------------------------------
# Schema & policy catalog
# ---------------------------------------------------------------------------

POLICY_DIR = Path(__file__).parent / "policies"
POLICIES_RL = {p.stem: p.read_text() for p in sorted(POLICY_DIR.glob("*.river"))}
POLICIES_CRL = {k: riverlang_to_compact(v) for k, v in POLICIES_RL.items()}

MODEL_ID = "query"

ALL_SOURCES = """src orders = table("orders") {
  order_id:str
  customer_id:str
  product_id:str
  amount:float
  order_date:date
  status:str
  rel to_customer -> one query.customers on base.customer_id == target.customer_id
  rel to_product -> one query.products on base.product_id == target.product_id
}
src customers = table("customers") {
  customer_id:str
  name:str
  segment:str
  region:str
  signup_date:date
}
src subscriptions = table("subscriptions") {
  subscription_id:str
  customer_id:str
  product_id:str
  mrr:float
  start_date:date
  end_date:date
  status:str
  rel to_customer -> one query.customers on base.customer_id == target.customer_id
  rel to_product -> one query.products on base.product_id == target.product_id
}
src products = table("products") {
  product_id:str
  name:str
  category:str
  list_price:float
}"""


def _build_concept_catalog() -> tuple[str, list[dict]]:
    """Build concept catalog from all policies. Returns (catalog_text, structured_concepts)."""
    concepts = []
    lines = []
    for policy_key, rl_text in POLICIES_RL.items():
        ms = RiverLangParser.from_str(rl_text)
        model = ms.models[0]
        for elem in model.elements:
            el = elem.element
            if type(el).__name__ != "Entity":
                continue
            is_output = el.is_output
            props = []
            for p in el.properties:
                sf = p.schema_field
                ann = sf.annotations[0] if sf.annotations else None
                props.append({"id": sf.id, "type": sf.type.to_grammar() if sf.type else "?", "ann": ann})

            concept = {
                "policy": policy_key,
                "entity": elem.id,
                "output": is_output,
                "metrics": [p["id"] for p in props if p["ann"] == "metric"],
                "dimensions": [p["id"] for p in props if p["ann"] == "dimension"],
                "keys": [p["id"] for p in props if p["ann"] == "key"],
                "fields": [p["id"] for p in props if p["ann"] is None],
            }
            concepts.append(concept)

            mark = " [output]" if is_output else ""
            lines.append(f"  {policy_key}.{elem.id}{mark}")
            if concept["metrics"]:
                lines.append(f"    metrics: {', '.join(concept['metrics'])}")
            if concept["dimensions"]:
                lines.append(f"    dimensions: {', '.join(concept['dimensions'])}")
            if concept["fields"]:
                lines.append(f"    fields: {', '.join(concept['fields'][:6])}")

    return "\n".join(lines), concepts


CONCEPT_CATALOG_TEXT, CONCEPT_CATALOG = _build_concept_catalog()

# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

PLANNER_PROMPT = """You are a query planner. Given a business question and a catalog of existing data concepts, decide how to answer it.

Available tables: orders (30 rows), customers (10), subscriptions (13), products (5)
Period: Jan-Apr 2025

Existing concepts:
{catalog}

Output a JSON plan:
{{
  "approach": "reuse" | "patch" | "fresh",
  "reasoning": "brief explanation",
  "policy": "subscription_arr" | "revenue_analytics" | "customer_health" | "order_classification" | null,
  "entities": ["entity_id_to_reuse"],
  "description": "what the user is asking for"
}}

Valid policy names: subscription_arr, revenue_analytics, customer_health, order_classification
- "reuse": an existing [output] entity answers this EXACTLY as-is. Set policy and entities.
- "patch": policy has the right base data but needs a new/modified entity. Set policy.
- "fresh": nothing relevant. Set policy to null.

Output ONLY valid JSON."""


def plan(question: str) -> dict:
    t0 = time.time()
    text, usage = _call_llm(
        PLANNER_PROMPT.format(catalog=CONCEPT_CATALOG_TEXT),
        [{"role": "user", "content": question}],
        max_tokens=300,
    )
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        result = {"approach": "fresh", "reasoning": "Could not parse plan", "policy": None, "entities": [], "description": question}

    result["_time"] = round(time.time() - t0, 1)
    result["_tokens"] = usage.get("output_tokens", 0)
    return result


# ---------------------------------------------------------------------------
# Executors
# ---------------------------------------------------------------------------

def execute_reuse(plan_result: dict, con) -> None:
    policy_key = plan_result.get("policy")
    if not policy_key or policy_key not in POLICIES_RL:
        log(f"{YELLOW}Policy '{policy_key}' not found, falling back to fresh{RESET}")
        execute_fresh(plan_result, con)
        return

    t0 = time.time()
    rl = POLICIES_RL[policy_key]
    ms = RiverLangParser.from_str(rl)
    cr = Compiler(ms).verify()
    if cr.has_errors():
        log(f"{YELLOW}Policy has compile errors, falling back to fresh{RESET}")
        execute_fresh(plan_result, con)
        return

    model = ms.models[0]
    sql_gen = cr.get_sql_generator()
    raw_entities = plan_result.get("entities", [])
    # Strip policy prefix if present (planner may return "policy.entity" or just "entity")
    entities = [e.split(".")[-1] if "." in e else e for e in raw_entities]
    target_ents = [
        e for e in model.elements
        if hasattr(e.element, "is_output") and e.element.is_output
        and (not entities or e.id in entities)
    ]

    _last["model_set"] = ms
    _last["compiler_result"] = cr
    _last["con"] = con

    log(f"{DIM}Compile: {time.time() - t0:.2f}s (local, no LLM call){RESET}")

    for ent in target_ents:
        _run_entity(sql_gen, model.id, ent, con)


def execute_patch(plan_result: dict, con) -> None:
    policy_key = plan_result.get("policy")
    description = plan_result.get("description", "")

    if not policy_key or policy_key not in POLICIES_CRL:
        log(f"{YELLOW}Policy '{policy_key}' not found, falling back to fresh{RESET}")
        execute_fresh(plan_result, con)
        return

    crl = POLICIES_CRL[policy_key]
    header, sources, entities = split_crl(crl)

    t0 = time.time()
    qr = run_query(description, policy_key, sources, entities)
    elapsed = time.time() - t0

    log(f"{DIM}LLM: {elapsed:.1f}s, {qr.get('total_output_tokens', 0)} tok, {qr['attempts']} attempt(s){RESET}")

    if qr["status"] != "compiled":
        log(f"{YELLOW}Patch failed, falling back to fresh{RESET}")
        if "last_errors" in qr:
            for e in qr["last_errors"][:2]:
                log(f"  {DIM}{e[:100]}{RESET}")
        execute_fresh(plan_result, con)
        return

    # Show only NEW CRL entities (not existing ones)
    orig_ids = _get_entity_ids(crl)
    new_crl_lines = _extract_new_entities(qr.get("entities", ""), orig_ids)
    _last["crl"] = "\n".join(new_crl_lines) if new_crl_lines else qr.get("entities", "")
    if new_crl_lines:
        log(f"\n{DIM}CRL (new entities):{RESET}")
        for line in new_crl_lines:
            log(f"  {DIM}{line}{RESET}")

    _run_new_output_entities(qr, orig_ids, con)


def execute_fresh(plan_result: dict, con) -> None:
    description = plan_result.get("description", "")

    t0 = time.time()
    qr = run_query(description, MODEL_ID, ALL_SOURCES, entities="")
    elapsed = time.time() - t0

    log(f"{DIM}LLM: {elapsed:.1f}s, {qr.get('total_output_tokens', 0)} tok, {qr['attempts']} attempt(s){RESET}")

    if qr["status"] != "compiled":
        print(f"\n  {RED}Error: Could not compile the answer.{RESET}")
        if "last_errors" in qr:
            for e in qr["last_errors"][:3]:
                log(f"  {DIM}{e[:120]}{RESET}")
        return

    # Show CRL (all entities are new in fresh mode)
    _last["crl"] = qr.get("entities", "")
    log(f"\n{DIM}CRL:{RESET}")
    for line in qr.get("entities", "").split("\n"):
        if line.strip():
            log(f"  {DIM}{line}{RESET}")

    _run_new_output_entities(qr, set(), con)


def _get_entity_ids(crl_text: str) -> set[str]:
    """Get entity IDs from CRL text."""
    import re
    return set(re.findall(r"^ent\s+(\w+)\s", crl_text, re.MULTILINE))


def _extract_new_entities(entities_text: str, orig_ids: set[str]) -> list[str]:
    """Extract lines belonging to entities not in orig_ids."""
    lines = entities_text.split("\n")
    result = []
    in_new = False
    depth = 0
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("ent "):
            ent_id = stripped.split()[1]
            in_new = ent_id not in orig_ids
            depth = 0
        if in_new:
            result.append(line)
            depth += stripped.count("{") - stripped.count("}")
            if depth <= 0 and "{" in stripped or stripped == "}":
                if depth <= 0:
                    in_new = False
    return result


def _run_new_output_entities(qr: dict, orig_ids: set[str], con) -> None:
    """Run SQL only for NEW output entities (not existing ones from the policy)."""
    cr = qr["compiler_result"]
    ms = qr["model_set"]
    model = ms.models[0]

    _last["model_set"] = ms
    _last["compiler_result"] = cr
    _last["con"] = con

    # Prefer new entities; fall back to any output if none are new
    new_output = [
        e for e in model.elements
        if hasattr(e.element, "is_output") and e.element.is_output
        and e.id not in orig_ids
    ]
    if not new_output:
        new_output = [e for e in model.elements if hasattr(e.element, "is_output") and e.element.is_output]

    sql_gen = cr.get_sql_generator()
    for ent in new_output:
        _run_entity(sql_gen, model.id, ent, con)


# Last query state (for /sql, /crl, /plan, /debug commands)
_last = {"sql": None, "crl": None, "plan": None, "model_set": None, "compiler_result": None, "con": None}


def _run_entity(sql_gen, model_id: str, ent, con) -> None:
    try:
        sql = sql_gen.generate_sql(FQN((model_id, ent.id)), dialect="duckdb")
        _last["sql"] = sql
        # Show SQL
        log(f"\n{DIM}SQL ({ent.id}):{RESET}")
        for line in sql.split("\n"):
            log(f"  {DIM}{line}{RESET}")

        rows = con.execute(sql).fetchall()
        cols = [d[0] for d in con.description]
        print(f"\n{BOLD}  Result:{RESET}")
        print(format_table(cols, rows))
    except Exception as e:
        print(f"\n  {RED}SQL Error: {e}{RESET}")


# ---------------------------------------------------------------------------
# Logging & formatting
# ---------------------------------------------------------------------------

def log(msg: str):
    print(f"  {msg}", flush=True)


def format_table(cols: list[str], rows: list[tuple]) -> str:
    if not rows:
        return f"  {DIM}(no results){RESET}"
    str_rows = [[str(v) for v in row] for row in rows]
    widths = [max(len(c), *(len(r[i]) for r in str_rows)) for i, c in enumerate(cols)]
    lines = []
    header = " | ".join(f"{c:>{w}}" for c, w in zip(cols, widths))
    lines.append(f"  {CYAN}{header}{RESET}")
    lines.append(f"  {'-' * len(header)}")
    for row in str_rows:
        lines.append(f"  {' | '.join(f'{v:>{w}}' for v, w in zip(row, widths))}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def ask(question: str, con) -> None:
    t_total = time.time()
    print(f"\n{BOLD}  Q: {question}{RESET}")

    # Step 1: Plan
    log(f"{CYAN}[plan]{RESET} Analyzing question...")
    p = plan(question)
    _last["plan"] = p
    approach = p.get("approach", "fresh")
    policy = p.get("policy")
    reasoning = p.get("reasoning", "")

    approach_color = GREEN if approach == "reuse" else YELLOW if approach == "patch" else CYAN
    log(f"{CYAN}[plan]{RESET} {approach_color}{approach}{RESET}"
        + (f" -> {BOLD}{policy}{RESET}" if policy else "")
        + f" {DIM}({p['_time']}s, {p['_tokens']} tok){RESET}")
    log(f"{DIM}{reasoning}{RESET}")

    # Step 2: Execute
    log(f"{CYAN}[exec]{RESET} Running...")
    if approach == "reuse":
        execute_reuse(p, con)
    elif approach == "patch":
        execute_patch(p, con)
    else:
        execute_fresh(p, con)

    elapsed = time.time() - t_total
    log(f"\n{DIM}Total: {elapsed:.1f}s{RESET}")


def print_welcome():
    print(f"\n{BOLD}{'=' * 65}{RESET}")
    print(f"{BOLD}  Talk to Your Data{RESET}")
    print(f"{BOLD}{'=' * 65}{RESET}")

    print(f"\n{BOLD}  Data:{RESET}")
    print(f"    {CYAN}orders{RESET} (30)  {CYAN}customers{RESET} (10)  {CYAN}subscriptions{RESET} (13)  {CYAN}products{RESET} (5)")
    print(f"    {DIM}Period: Jan-Apr 2025 | Segments: Enterprise, Mid-Market, SMB | Regions: US, EMEA, APAC{RESET}")

    print(f"\n{BOLD}  Glossary (pre-built concepts — instant reuse):{RESET}")
    for c in CONCEPT_CATALOG:
        if not c["output"]:
            continue
        # Build a human-readable concept description
        entity_name = c["entity"].replace("_", " ").title()
        metrics = c["metrics"]
        dims = c["dimensions"]
        fields = c["fields"]
        parts = []
        if metrics:
            parts.append(", ".join(m.replace("_", " ") for m in metrics))
        if dims:
            parts.append(f"by {', '.join(d.replace('_', ' ') for d in dims)}")
        if not metrics and not dims and fields:
            parts.append(", ".join(f.replace("_", " ") for f in fields[:5]))
        desc = " ".join(parts)
        print(f"    {GREEN}{entity_name}{RESET}  {DIM}{desc}{RESET}")

    print(f"\n  {DIM}Ask anything — reuses concepts when possible, builds new queries otherwise.{RESET}")
    print(f"  {DIM}Type /help for commands.{RESET}\n")


HISTORY_FILE = Path(__file__).parent / ".ask_history"

COMMANDS = {
    "/help":     "Show this help",
    "/sql":      "Show last generated SQL",
    "/crl":      "Show last generated CRL entities",
    "/plan":     "Show last planner output",
    "/debug":    "Trace last query — run each entity in the lineage step by step",
    "/glossary": "Show pre-built concepts",
    "/tables":   "Show available tables",
    "/clear":    "Clear screen",
    "/quit":     "Exit",
}


def handle_command(cmd: str) -> bool:
    """Handle a / command. Returns True if handled."""
    cmd = cmd.strip().lower()

    if cmd in ("/quit", "/exit", "/q"):
        print("  Bye!")
        return True  # signal exit

    if cmd == "/help":
        print(f"\n{BOLD}  Commands:{RESET}")
        for c, desc in COMMANDS.items():
            print(f"    {CYAN}{c:<12s}{RESET} {desc}")
        print()
        return False

    if cmd == "/sql":
        if _last["sql"]:
            print(f"\n{DIM}{_last['sql']}{RESET}\n")
        else:
            print(f"  {DIM}No SQL yet — ask a question first.{RESET}")
        return False

    if cmd == "/crl":
        if _last["crl"]:
            print(f"\n{DIM}{_last['crl']}{RESET}\n")
        else:
            print(f"  {DIM}No CRL yet — ask a question first.{RESET}")
        return False

    if cmd == "/plan":
        if _last["plan"]:
            p = _last["plan"]
            print(f"\n  {BOLD}approach:{RESET} {p.get('approach')}")
            print(f"  {BOLD}policy:{RESET}   {p.get('policy', '-')}")
            print(f"  {BOLD}entities:{RESET} {p.get('entities', [])}")
            print(f"  {BOLD}reason:{RESET}   {p.get('reasoning', '-')}")
            print(f"  {DIM}{p.get('_time', 0)}s, {p.get('_tokens', 0)} tok{RESET}\n")
        else:
            print(f"  {DIM}No plan yet — ask a question first.{RESET}")
        return False

    if cmd == "/glossary":
        print(f"\n{BOLD}  Glossary:{RESET}")
        for c in CONCEPT_CATALOG:
            if not c["output"]:
                continue
            entity_name = c["entity"].replace("_", " ").title()
            metrics = c["metrics"]
            dims = c["dimensions"]
            fields = c["fields"]
            parts = []
            if metrics:
                parts.append(", ".join(m.replace("_", " ") for m in metrics))
            if dims:
                parts.append(f"by {', '.join(d.replace('_', ' ') for d in dims)}")
            if not metrics and not dims and fields:
                parts.append(", ".join(f.replace("_", " ") for f in fields[:5]))
            print(f"    {GREEN}{entity_name}{RESET}  {DIM}{' '.join(parts)}{RESET}")
        print()
        return False

    if cmd == "/tables":
        print(f"\n{BOLD}  Tables:{RESET}")
        print(f"    {CYAN}orders{RESET}         30 rows  order_id, customer_id, product_id, amount, order_date, status")
        print(f"    {CYAN}customers{RESET}      10 rows  customer_id, name, segment, region, signup_date")
        print(f"    {CYAN}subscriptions{RESET}  13 rows  subscription_id, customer_id, product_id, mrr, start/end_date, status")
        print(f"    {CYAN}products{RESET}        5 rows  product_id, name, category, list_price")
        print()
        return False

    if cmd == "/debug":
        _debug_trace()
        return False

    if cmd == "/clear":
        print("\033[2J\033[H", end="")
        return False

    print(f"  {DIM}Unknown command. Type /help for options.{RESET}")
    return False


def _debug_trace():
    """Trace last query by running SQL for each entity in the dependency chain."""
    ms = _last.get("model_set")
    cr = _last.get("compiler_result")
    con = _last.get("con")

    if not ms or not cr or not con:
        print(f"  {DIM}No query to debug — ask a question first.{RESET}")
        return

    model = ms.models[0]
    sql_gen = cr.get_sql_generator()

    # Build dependency order: walk entities, skipping sources
    from riverlang.ast.riverlang_ast import Entity, Source
    entities = []
    for elem in model.elements:
        if isinstance(elem.element, Entity):
            entities.append(elem)

    print(f"\n{BOLD}  Debug trace — {len(entities)} entities in lineage{RESET}\n")

    for i, elem in enumerate(entities):
        fqn = FQN((model.id, elem.id))
        is_output = elem.element.is_output
        marker = f" {GREEN}[output]{RESET}" if is_output else ""

        # Get generator info
        gen = elem.element.generator
        base_str = ".".join(gen.base.value) if hasattr(gen, "base") else "?"
        filter_str = f" filter {gen.filter.to_grammar()}" if hasattr(gen, "filter") and gen.filter else ""
        group_str = f" {gen.grouping.to_grammar()}" if hasattr(gen, "grouping") and gen.grouping else ""
        order_str = ""
        if hasattr(gen, "order_by") and gen.order_by:
            order_str = f" order by ..."
        limit_str = f" limit {gen.limit}" if hasattr(gen, "limit") and gen.limit else ""

        print(f"  {CYAN}[{i+1}/{len(entities)}]{RESET} {BOLD}{elem.id}{RESET}{marker}")
        print(f"  {DIM}from {base_str}{filter_str}{group_str}{order_str}{limit_str}{RESET}")

        # Generate and run SQL for this specific entity
        try:
            sql = sql_gen.generate_sql(fqn, dialect="duckdb")
            rows = con.execute(sql).fetchall()
            cols = [d[0] for d in con.description]

            print(f"  {DIM}SQL: {sql[:120].replace(chr(10), ' ')}{'...' if len(sql) > 120 else ''}{RESET}")
            print(f"  {DIM}{len(rows)} rows, {len(cols)} cols{RESET}")

            # Show first few rows
            if rows:
                print(format_table(cols, rows[:5]))
                if len(rows) > 5:
                    print(f"  {DIM}... and {len(rows) - 5} more rows{RESET}")
            else:
                print(f"  {DIM}(empty){RESET}")
        except Exception as e:
            print(f"  {RED}Error: {e}{RESET}")

        print()


def main():
    import argparse
    import readline

    parser = argparse.ArgumentParser(description="Talk to your data")
    parser.add_argument("question", nargs="?", help="Question (interactive if omitted)")
    args = parser.parse_args()

    con = create_db(":memory:")

    if args.question:
        ask(args.question, con)
        con.close()
        return

    # Setup readline history
    if HISTORY_FILE.exists():
        readline.read_history_file(str(HISTORY_FILE))
    readline.set_history_length(200)
    # Tab completion for / commands
    readline.parse_and_bind("tab: complete")
    readline.set_completer(lambda text, state: (
        [c for c in COMMANDS if c.startswith(text)] + [None]
    )[state])

    print_welcome()
    while True:
        try:
            q = input(f"  \001{BOLD}\002>\001{RESET}\002 ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Bye!")
            break
        if not q:
            continue
        if q.startswith("/"):
            if handle_command(q):
                break
            continue
        ask(q, con)
        print()

    # Save history
    try:
        readline.write_history_file(str(HISTORY_FILE))
    except OSError:
        pass
    con.close()


if __name__ == "__main__":
    main()
