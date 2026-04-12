#!/usr/bin/env python3
"""Run 20 questions in comparison mode using the same code as the CLI /compare.

Outputs a summary table showing timing, tokens, results, and match status.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import duckdb
from setup_db import create_db

# Import everything from ask.py — same code path as interactive CLI
import ask
from ask import (
    plan, execute_reuse, execute_patch, _run_direct_sql, _last,
    MODELS_CRL, ALL_MODELS_RL, CONCEPT_CATALOG_TEXT,
    split_crl, run_query, format_table,
    log, DIM, BOLD, CYAN, GREEN, YELLOW, RED, RESET,
)

QUESTIONS = [
    "What is our total ARR?",
    "ARR by segment",
    "ARR excluding EMEA",
    "Total revenue in Q1 2025",
    "Revenue by product category",
    "Top 5 customers by total spend",
    "How many Strategic deals were there?",
    "Total revenue from Small orders",
    "Which customer has the highest ARR?",
    "Average discount percentage by segment",
    "Revenue in March 2025 by region",
    "How many unique customers placed orders in 2025?",
    "ARR from Platform products only",
    "Total ARR in February 2025",
    "Which region has the highest revenue?",
    "Average order size for Enterprise customers",
    "How many subscriptions are active?",
    "Revenue growth: Q4 2024 vs Q1 2025",
    "NRR last month",
    "Customers with ARR above 5000",
]


def extract_first_value(con):
    """Extract a comparable value from _last['sql'] result."""
    sql = _last.get("sql")
    if not sql:
        return None
    try:
        rows = con.execute(sql).fetchall()
        if not rows:
            return None
        if len(rows) == 1 and len(rows[0]) == 1:
            return rows[0][0]
        return rows
    except Exception:
        return None


def fmt_val(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:,.1f}"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, list):
        return f"{len(v)} rows"
    return str(v)[:15]


def values_match(v1, v2) -> str:
    if v1 is None or v2 is None:
        return "n/a"
    if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
        return "match" if abs(float(v1) - float(v2)) < 1 else "differ"
    if isinstance(v1, list) and isinstance(v2, list):
        def nums(rows):
            s = set()
            for r in rows:
                for v in r:
                    if isinstance(v, (int, float)):
                        s.add(round(float(v), 1))
            return s
        n1, n2 = nums(v1), nums(v2)
        if n1 and n2:
            overlap = n1 & n2
            return "match" if len(overlap) >= min(len(n1), len(n2)) * 0.5 else "differ"
    # scalar vs rows — try to find the scalar in the rows
    if isinstance(v1, (int, float)) and isinstance(v2, list):
        for r in v2:
            for v in r:
                if isinstance(v, (int, float)) and abs(float(v1) - float(v)) < 1:
                    return "match"
    if isinstance(v2, (int, float)) and isinstance(v1, list):
        for r in v1:
            for v in r:
                if isinstance(v, (int, float)) and abs(float(v2) - float(v)) < 1:
                    return "match"
    return "n/a"


def run_one_crl(question: str, con) -> dict:
    """Run CRL approach using the same ask.py code path. Returns result dict."""
    import json
    _last["sql"] = None
    t0 = time.time()

    # Plan (same as ask.py)
    p = plan(question)
    model_name = p.get("model")
    approach = p.get("approach", "patch")

    if approach == "reuse" and model_name:
        execute_reuse(p, con)
    elif model_name and model_name in MODELS_CRL:
        # Inline patch without the print noise
        crl = MODELS_CRL[model_name]
        header, sources, entities = split_crl(crl)
        qr = run_query(p.get("description", question), model_name, sources, entities, all_models_rl=ALL_MODELS_RL)
        if qr["status"] == "compiled":
            from ask import _get_entity_ids, _run_new_output_entities
            orig_ids = _get_entity_ids(crl)
            _run_new_output_entities(qr, orig_ids, con, model_id=model_name, show_sql=False)
            # Now re-run with show_sql to populate _last["sql"]
            if _last.get("sql") is None:
                _run_new_output_entities(qr, orig_ids, con, model_id=model_name, show_sql=False)
        else:
            elapsed = round(time.time() - t0, 1)
            return {"status": "fail", "elapsed_s": elapsed, "output_tokens": qr.get("total_output_tokens", 0), "value": None}

    elapsed = round(time.time() - t0, 1)
    value = extract_first_value(con)
    tok = 0  # hard to get from this path
    return {"status": "ok" if value is not None else "fail", "elapsed_s": elapsed, "output_tokens": tok, "value": value}


def main():
    db_path = Path(__file__).parent / "demo.db"
    if not db_path.exists():
        create_db(db_path)
    con = duckdb.connect(str(db_path), read_only=True)

    n = len(QUESTIONS)
    results = []

    # Header
    print(f"\n{BOLD}Comparing CRL vs Direct SQL — {n} questions{RESET}\n")

    for i, q in enumerate(QUESTIONS):
        print(f"{CYAN}[{i+1}/{n}]{RESET} {BOLD}{q}{RESET}")

        # CRL approach (silent — no SQL/result output)
        sys.stdout.write(f"  CRL: ")
        sys.stdout.flush()

        _last["sql"] = None
        t0 = time.time()
        p = plan(q)
        model_name = p.get("model")
        crl_value = None
        crl_tok = 0

        if p.get("approach") == "reuse" and model_name:
            execute_reuse(p, con)
            crl_value = extract_first_value(con)
        elif model_name and model_name in MODELS_CRL:
            from ask import _get_entity_ids, _run_new_output_entities
            crl = MODELS_CRL[model_name]
            header, sources, entities = split_crl(crl)
            qr = run_query(p.get("description", q), model_name, sources, entities, all_models_rl=ALL_MODELS_RL)
            crl_tok = qr.get("total_output_tokens", 0)
            if qr["status"] == "compiled":
                orig_ids = _get_entity_ids(crl)
                _run_new_output_entities(qr, orig_ids, con, model_id=model_name, show_sql=False)
                crl_value = extract_first_value(con)

        crl_elapsed = round(time.time() - t0, 1)
        crl_status = "ok" if crl_value is not None else "FAIL"
        print(f"{crl_elapsed}s, {crl_tok} tok → {fmt_val(crl_value)}  {'✓' if crl_status == 'ok' else '✗'}")

        # Direct SQL (same as /compare in ask.py)
        sys.stdout.write(f"  SQL: ")
        sys.stdout.flush()

        sql_r = _run_direct_sql(q, con)
        sql_value = None
        if sql_r["status"] == "ok" and sql_r["rows"]:
            if len(sql_r["rows"]) == 1 and len(sql_r["rows"][0]) == 1:
                sql_value = sql_r["rows"][0][0]
            else:
                sql_value = sql_r["rows"]
        sql_status = "ok" if sql_value is not None else "FAIL"
        print(f"{sql_r['elapsed_s']}s, {sql_r['output_tokens']} tok → {fmt_val(sql_value)}  {'✓' if sql_status == 'ok' else '✗'}")

        # Match
        match = values_match(crl_value, sql_value)
        if match == "match":
            print(f"  {GREEN}→ MATCH{RESET}")
        elif match == "differ":
            print(f"  {RED}→ DIFFER: CRL={fmt_val(crl_value)}, SQL={fmt_val(sql_value)}{RESET}")
        else:
            reason = ""
            if crl_status == "FAIL" and sql_status == "FAIL":
                reason = " (both failed)"
            elif crl_status == "FAIL":
                reason = " (CRL failed)"
            elif sql_status == "FAIL":
                reason = " (SQL failed)"
            print(f"  {DIM}→ n/a{reason}{RESET}")

        results.append({"q": q, "crl_t": crl_elapsed, "crl_tok": crl_tok, "crl_val": crl_value,
                         "sql_t": sql_r["elapsed_s"], "sql_tok": sql_r["output_tokens"], "sql_val": sql_value,
                         "match": match})
        print()

    # Summary table
    print(f"\n{BOLD}{'=' * 100}{RESET}")
    print(f"{BOLD}{'#':>2} {'Question':<42s} {'CRL':>6s} {'SQL':>6s} {'CRL val':>12s} {'SQL val':>12s} {'':>7s}{RESET}")
    print(f"{'-' * 100}")

    for i, r in enumerate(results):
        match_sym = f"{GREEN}={RESET}" if r["match"] == "match" else f"{RED}X{RESET}" if r["match"] == "differ" else f"{DIM}?{RESET}"
        print(f"{i+1:>2} {r['q']:<42s} {r['crl_t']:>5.1f}s {r['sql_t']:>5.1f}s {fmt_val(r['crl_val']):>12s} {fmt_val(r['sql_val']):>12s} {match_sym:>7s}")

    print(f"{'-' * 100}")

    crl_ok = sum(1 for r in results if r["crl_val"] is not None)
    sql_ok = sum(1 for r in results if r["sql_val"] is not None)
    matches = sum(1 for r in results if r["match"] == "match")
    differs = sum(1 for r in results if r["match"] == "differ")
    crl_total_t = sum(r["crl_t"] for r in results)
    sql_total_t = sum(r["sql_t"] for r in results)

    print(f"\n{BOLD}Summary:{RESET}")
    print(f"  CRL:  {crl_ok}/{n} succeeded, {crl_total_t:.1f}s total")
    print(f"  SQL:  {sql_ok}/{n} succeeded, {sql_total_t:.1f}s total")
    print(f"  {GREEN}Match: {matches}{RESET}  {RED}Differ: {differs}{RESET}  N/A: {n - matches - differs}")

    if differs > 0:
        print(f"\n{RED}  Differing results:{RESET}")
        for r in results:
            if r["match"] == "differ":
                print(f"    {r['q']}: CRL={fmt_val(r['crl_val'])} vs SQL={fmt_val(r['sql_val'])}")

    con.close()


if __name__ == "__main__":
    main()
