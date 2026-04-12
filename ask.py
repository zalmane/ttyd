#!/usr/bin/env python3
"""Talk to your data — agentic CLI.

Scans models/ directory for RiverLang models. Plans each question as
reuse (existing output entity) or patch (LLM adds grouping/filter).

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

import duckdb
from setup_db import create_db
from compact_rl import riverlang_to_compact, compact_to_modelset, split_crl, merge_crl
from llm_runner import _call_llm, run_query
from riverlang.ast.parser import RiverLangParser
from riverlang.compiler.compiler import Compiler
from riverlang.compiler.fqn import FQN

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
DIM = "\033[2m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"

# ---------------------------------------------------------------------------
# Load all models from models/ directory
# ---------------------------------------------------------------------------

MODELS_DIR = Path(__file__).parent / "models"

def _load_models():
    """Load all .river models, compile, build CRL versions and concept catalog."""
    # Read and parse all models together (cross-model references need joint compilation)
    model_files = sorted(MODELS_DIR.glob("*.river"))
    all_rl_text = "\n".join(f.read_text() for f in model_files)
    all_rl = all_rl_text
    ms = RiverLangParser.from_str(all_rl)
    cr = Compiler(ms).verify()
    if cr.has_errors():
        print(f"{RED}Model compilation errors:{RESET}")
        for e in cr.errors:
            print(f"  {e}")

    # Build per-model CRL and catalog
    models_rl = {}
    models_crl = {}
    catalog = []
    catalog_lines = []

    for model in ms.models:
        # Find the .river file for this model
        rl_text = None
        for f in model_files:
            if f.stem == model.id or f.stem == model.id.replace("orders", "order"):
                rl_text = f.read_text()
                break
        if rl_text:
            models_rl[model.id] = rl_text
            models_crl[model.id] = riverlang_to_compact(rl_text)

        for elem in model.elements:
            el = elem.element
            if type(el).__name__ not in ("Entity",):
                continue

            is_output = getattr(el, "is_output", False)
            props = []
            if hasattr(el, "properties"):
                for p in el.properties:
                    sf = p.schema_field
                    ann = sf.annotations[0] if sf.annotations else None
                    props.append({"id": sf.id, "ann": ann})

            concept = {
                "model": model.id,
                "entity": elem.id,
                "description": elem.description,
                "output": is_output,
                "metrics": [p["id"] for p in props if p["ann"] == "metric"],
                "dimensions": [p["id"] for p in props if p["ann"] == "dimension"],
                "fields": [p["id"] for p in props if p["ann"] is None],
            }
            catalog.append(concept)

            mark = " [output]" if is_output else ""
            desc = f" — {elem.description}" if elem.description else ""
            catalog_lines.append(f"  {model.id}.{elem.id}{mark}{desc}")
            if concept["metrics"]:
                catalog_lines.append(f"    metrics: {', '.join(concept['metrics'])}")
            if concept["dimensions"]:
                catalog_lines.append(f"    dimensions: {', '.join(concept['dimensions'])}")
            if concept["fields"] and not concept["metrics"] and not concept["dimensions"]:
                catalog_lines.append(f"    fields: {', '.join(concept['fields'][:6])}")

    return ms, cr, models_rl, models_crl, catalog, "\n".join(catalog_lines), all_rl_text


MODELS_MS, MODELS_CR, MODELS_RL, MODELS_CRL, CONCEPT_CATALOG, CONCEPT_CATALOG_TEXT, ALL_MODELS_RL = _load_models()

# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

PLANNER_PROMPT = """You are a query planner. Given a business question and a catalog of data models, decide how to answer it.

Available data: orders (54), customers (10), subscriptions (17), products (5). Period: Jul 2024 - Apr 2025.

Models (organized as a DAG — each model is one concept):
{catalog}

Output a JSON plan:
{{
  "approach": "reuse" | "patch",
  "reasoning": "brief explanation",
  "model": "model_name",
  "entities": ["entity_id_to_reuse"],
  "description": "what the user is asking for"
}}

Valid model names: {model_names}
- "reuse": an existing [output] entity answers this EXACTLY as-is with NO filtering or grouping needed. The result rows ARE the answer.
- "patch": the model has the right base data but needs a new grouping, filter, or calculation added by the LLM.

Pick the most specific model that has the data needed. For example:
- "NRR last month" → reuse nrr (already computed)
- "Total ARR" → patch arr (needs grouping all rows)
- "Revenue in March" → patch revenue (needs month filter)
- "MRR by segment" → patch customer_monthly_mrr (needs grouping by segment)

Output ONLY valid JSON."""


def plan(question: str) -> dict:
    model_names = ", ".join(sorted(MODELS_CRL.keys()))
    t0 = time.time()
    text, usage = _call_llm(
        PLANNER_PROMPT.format(catalog=CONCEPT_CATALOG_TEXT, model_names=model_names),
        [{"role": "user", "content": question}],
        max_tokens=300,
    )
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        result = {"approach": "patch", "reasoning": "Could not parse plan", "model": None, "entities": [], "description": question}

    result["_time"] = round(time.time() - t0, 1)
    result["_tokens"] = usage.get("output_tokens", 0)
    return result


# ---------------------------------------------------------------------------
# Executors
# ---------------------------------------------------------------------------

def execute_reuse(plan_result: dict, con) -> None:
    model_name = plan_result.get("model")
    if not model_name or model_name not in MODELS_RL:
        log(f"{YELLOW}Model '{model_name}' not found, switching to patch{RESET}")
        execute_patch(plan_result, con)
        return

    t0 = time.time()
    # Use the pre-compiled model set (all models compiled together)
    sql_gen = MODELS_CR.get_sql_generator()

    raw_entities = plan_result.get("entities", [])
    entities = [e.split(".")[-1] if "." in e else e for e in raw_entities]

    # Find the target model and its output entities
    target_model = None
    for m in MODELS_MS.models:
        if m.id == model_name:
            target_model = m
            break

    if not target_model:
        log(f"{YELLOW}Model '{model_name}' not found in compiled models{RESET}")
        execute_patch(plan_result, con)
        return

    target_ents = [
        e for e in target_model.elements
        if hasattr(e.element, "is_output") and e.element.is_output
        and (not entities or e.id in entities)
    ]

    _last["model_set"] = MODELS_MS
    _last["compiler_result"] = MODELS_CR
    _last["con"] = con

    log(f"{DIM}Compile: {time.time() - t0:.2f}s (local, no LLM call){RESET}")

    for ent in target_ents:
        _run_entity(sql_gen, target_model.id, ent, con)


def execute_patch(plan_result: dict, con) -> None:
    model_name = plan_result.get("model")
    description = plan_result.get("description", "")

    # Build list of models to try: planner's choice first, then others
    models_to_try = []
    if model_name and model_name in MODELS_CRL:
        models_to_try.append(model_name)
    for m in MODELS_CRL:
        if m not in models_to_try:
            models_to_try.append(m)

    for i, mk in enumerate(models_to_try):
        crl = MODELS_CRL[mk]
        header, sources, entities = split_crl(crl)

        t0 = time.time()
        qr = run_query(description, mk, sources, entities, all_models_rl=ALL_MODELS_RL)
        elapsed = time.time() - t0

        log(f"{DIM}LLM ({mk}): {elapsed:.1f}s, {qr.get('total_output_tokens', 0)} tok, {qr['attempts']} attempt(s){RESET}")

        if qr["status"] == "compiled":
            orig_ids = _get_entity_ids(crl)
            new_crl_lines = _extract_new_entities(qr.get("entities", ""), orig_ids)
            _last["crl"] = "\n".join(new_crl_lines) if new_crl_lines else qr.get("entities", "")
            if new_crl_lines:
                log(f"\n{DIM}CRL (new entities):{RESET}")
                for line in new_crl_lines:
                    log(f"  {DIM}{line}{RESET}")
            _run_new_output_entities(qr, orig_ids, con, model_id=mk)
            return

        errors = qr.get("last_errors", [])
        if i < len(models_to_try) - 1:
            log(f"{YELLOW}Patch {mk} failed, trying next model...{RESET}")
            if errors:
                log(f"  {DIM}{errors[0][:80]}{RESET}")
        else:
            log(f"{RED}All models failed{RESET}")
            if errors:
                for e in errors[:2]:
                    log(f"  {DIM}{e[:100]}{RESET}")


def _get_entity_ids(crl_text: str) -> set[str]:
    import re
    return set(re.findall(r"^ent\s+(\w+)\s", crl_text, re.MULTILINE))


def _extract_new_entities(entities_text: str, orig_ids: set[str]) -> list[str]:
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


def _run_new_output_entities(qr: dict, orig_ids: set[str], con, model_id: str = None) -> None:
    cr = qr["compiler_result"]
    ms = qr["model_set"]

    _last["model_set"] = ms
    _last["compiler_result"] = cr
    _last["con"] = con

    # Find the target model (the one that was patched)
    target_model = None
    for m in ms.models:
        if model_id and m.id == model_id:
            target_model = m
            break
    if not target_model:
        target_model = ms.models[0]

    new_output = [
        e for e in target_model.elements
        if hasattr(e.element, "is_output") and e.element.is_output
        and e.id not in orig_ids
    ]
    if not new_output:
        new_output = [e for e in target_model.elements if hasattr(e.element, "is_output") and e.element.is_output]

    sql_gen = cr.get_sql_generator()
    for ent in new_output:
        _run_entity(sql_gen, target_model.id, ent, con)


# Last query state
_last = {"sql": None, "crl": None, "plan": None, "model_set": None, "compiler_result": None, "con": None}


def _run_entity(sql_gen, model_id: str, ent, con) -> None:
    try:
        sql = sql_gen.generate_sql(FQN((model_id, ent.id)), dialect="duckdb")
        _last["sql"] = sql
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

    log(f"{CYAN}[plan]{RESET} Analyzing question...")
    p = plan(question)
    _last["plan"] = p
    approach = p.get("approach", "patch")
    model = p.get("model")
    reasoning = p.get("reasoning", "")

    approach_color = GREEN if approach == "reuse" else YELLOW
    log(f"{CYAN}[plan]{RESET} {approach_color}{approach}{RESET}"
        + (f" -> {BOLD}{model}{RESET}" if model else "")
        + f" {DIM}({p['_time']}s, {p['_tokens']} tok){RESET}")
    log(f"{DIM}{reasoning}{RESET}")

    log(f"{CYAN}[exec]{RESET} Running...")
    if approach == "reuse":
        execute_reuse(p, con)
    else:
        execute_patch(p, con)

    elapsed = time.time() - t_total
    log(f"\n{DIM}Total: {elapsed:.1f}s{RESET}")


def print_welcome():
    print(f"\n{BOLD}{'=' * 65}{RESET}")
    print(f"{BOLD}  Talk to Your Data{RESET}")
    print(f"{BOLD}{'=' * 65}{RESET}")

    print(f"\n{BOLD}  Data:{RESET}")
    print(f"    {CYAN}orders{RESET} (54)  {CYAN}customers{RESET} (10)  {CYAN}subscriptions{RESET} (17)  {CYAN}products{RESET} (5)")
    print(f"    {DIM}Period: Jul 2024 - Apr 2025 | Segments: Enterprise, Mid-Market, SMB | Regions: US, EMEA, APAC{RESET}")

    print(f"\n{BOLD}  Concepts ({len(CONCEPT_CATALOG)} entities across {len(MODELS_CRL)} models):{RESET}")
    for c in CONCEPT_CATALOG:
        if not c["output"]:
            continue
        entity_name = c["entity"].replace("_", " ").title()
        desc = c.get("description", "")
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
        detail = " ".join(parts)
        if desc:
            print(f"    {GREEN}{entity_name}{RESET}  {DIM}{desc}{RESET}")
        else:
            print(f"    {GREEN}{entity_name}{RESET}  {DIM}{detail}{RESET}")

    print(f"\n  {DIM}Type /help for commands.{RESET}\n")


HISTORY_FILE = Path(__file__).parent / ".ask_history"

COMMANDS = {
    "/help":     "Show this help",
    "/sql":      "Show last generated SQL",
    "/crl":      "Show last generated CRL entities",
    "/plan":     "Show last planner output",
    "/debug":    "Trace last query — run each entity in the lineage step by step",
    "/glossary": "Show all concepts",
    "/tables":   "Show available tables",
    "/clear":    "Clear screen",
    "/quit":     "Exit",
}


def handle_command(cmd: str) -> bool:
    cmd = cmd.strip().lower()

    if cmd in ("/quit", "/exit", "/q"):
        print("  Bye!")
        return True

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
            print(f"  {BOLD}model:{RESET}    {p.get('model', '-')}")
            print(f"  {BOLD}entities:{RESET} {p.get('entities', [])}")
            print(f"  {BOLD}reason:{RESET}   {p.get('reasoning', '-')}")
            print(f"  {DIM}{p.get('_time', 0)}s, {p.get('_tokens', 0)} tok{RESET}\n")
        else:
            print(f"  {DIM}No plan yet — ask a question first.{RESET}")
        return False

    if cmd == "/glossary":
        print(f"\n{BOLD}  Concepts:{RESET}")
        for c in CONCEPT_CATALOG:
            entity_name = c["entity"].replace("_", " ").title()
            mark = f" {GREEN}[output]{RESET}" if c["output"] else ""
            desc = f" — {c['description']}" if c.get("description") else ""
            print(f"    {c['model']}.{BOLD}{entity_name}{RESET}{mark}{DIM}{desc}{RESET}")
        print()
        return False

    if cmd == "/tables":
        print(f"\n{BOLD}  Tables:{RESET}")
        print(f"    {CYAN}orders{RESET}         54 rows  order_id, customer_id, product_id, amount, order_date, status")
        print(f"    {CYAN}customers{RESET}      10 rows  customer_id, name, segment, region, signup_date")
        print(f"    {CYAN}subscriptions{RESET}  17 rows  subscription_id, customer_id, product_id, mrr, start/end_date, status")
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
    ms = _last.get("model_set")
    cr = _last.get("compiler_result")
    con = _last.get("con")

    if not ms or not cr or not con:
        print(f"  {DIM}No query to debug — ask a question first.{RESET}")
        return

    from riverlang.ast.riverlang_ast import Entity, Source
    model = ms.models[0]
    sql_gen = cr.get_sql_generator()
    entities = [e for e in model.elements if isinstance(e.element, Entity)]

    print(f"\n{BOLD}  Debug trace — {len(entities)} entities in lineage{RESET}\n")

    for i, elem in enumerate(entities):
        fqn = FQN((model.id, elem.id))
        is_output = elem.element.is_output
        marker = f" {GREEN}[output]{RESET}" if is_output else ""
        gen = elem.element.generator
        base_str = ".".join(gen.base.value) if hasattr(gen, "base") else "?"
        filter_str = f" filter {gen.filter.to_grammar()}" if hasattr(gen, "filter") and gen.filter else ""
        group_str = f" {gen.grouping.to_grammar()}" if hasattr(gen, "grouping") and gen.grouping else ""

        print(f"  {CYAN}[{i+1}/{len(entities)}]{RESET} {BOLD}{elem.id}{RESET}{marker}")
        print(f"  {DIM}from {base_str}{filter_str}{group_str}{RESET}")

        try:
            sql = sql_gen.generate_sql(fqn, dialect="duckdb")
            rows = con.execute(sql).fetchall()
            cols = [d[0] for d in con.description]
            print(f"  {DIM}{len(rows)} rows, {len(cols)} cols{RESET}")
            if rows:
                print(format_table(cols, rows[:5]))
                if len(rows) > 5:
                    print(f"  {DIM}... and {len(rows) - 5} more rows{RESET}")
        except Exception as e:
            print(f"  {RED}Error: {e}{RESET}")
        print()


def main():
    import argparse
    import readline

    parser = argparse.ArgumentParser(description="Talk to your data")
    parser.add_argument("question", nargs="?", help="Question (interactive if omitted)")
    args = parser.parse_args()

    db_path = Path(__file__).parent / "demo.db"
    if not db_path.exists():
        create_db(db_path)
    con = duckdb.connect(str(db_path), read_only=True)

    if args.question:
        ask(args.question, con)
        con.close()
        return

    if HISTORY_FILE.exists():
        readline.read_history_file(str(HISTORY_FILE))
    readline.set_history_length(200)
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

    try:
        readline.write_history_file(str(HISTORY_FILE))
    except OSError:
        pass
    con.close()


if __name__ == "__main__":
    main()
