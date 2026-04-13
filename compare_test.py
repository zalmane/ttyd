#!/usr/bin/env python3
"""Run 50 questions x 3 runs each: CRL vs SQL with stability measurement.

Each question runs 3 times. Reports:
- Per-question: match/differ, CRL vs SQL values, consistency across runs
- Summary: total matches, differs, stability scores
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import duckdb
from setup_db import create_db
import ask
from ask import (
    plan, execute_reuse, execute_patch, _run_direct_sql, _last,
    MODELS_CRL, ALL_MODELS_RL, CONCEPT_CATALOG_TEXT,
    split_crl, run_query, format_table, _get_entity_ids, _run_new_output_entities,
    DIM, BOLD, CYAN, GREEN, YELLOW, RED, RESET,
)
from compact_rl import compact_to_modelset

RUNS_PER_QUESTION = 3

QUESTIONS = [
    # --- Simple aggregations ---
    "What is our total ARR?",
    "Total revenue across all orders",
    "How many subscriptions are active?",
    "How many completed orders do we have?",
    "What is total MRR?",

    # --- Group by dimension ---
    "ARR by segment",
    "Revenue by product category",
    "Revenue by region",
    "ARR by product category",
    "Order count by segment",

    # --- Filters ---
    "ARR excluding EMEA",
    "Revenue from US only",
    "ARR from Mid-Market customers",
    "Revenue from orders above 2000",
    "How many Enterprise customers have active subscriptions?",

    # --- Governed concepts (need policy logic) ---
    "How many Strategic deals were there?",
    "Total revenue from Small orders",
    "Revenue from Transactional deals only",
    "Average deal size for Strategic deals",
    "How many Growth deals in 2025?",

    # --- Discount / effective MRR (complex policy) ---
    "Average discount percentage across all subscriptions",
    "Which customer gets the highest discount?",
    "ARR from subscriptions with discount above 15%",
    "Total effective MRR for Enterprise segment",
    "List all subscriptions where discount is zero",

    # --- Date filtering (parse_date overlap) ---
    "Total ARR in February 2025",
    "ARR in January 2025 by segment",
    "Revenue in Q4 2024",
    "Revenue in August 2024 by region",
    "How many orders in December 2024?",

    # --- Top N / ranking ---
    "Top 3 customers by ARR",
    "Top 5 customers by total spend",
    "Largest single order amount",
    "Customer with the most orders",
    "Bottom 3 regions by revenue",

    # --- Cross-concept ---
    "Which customer has the highest ARR?",
    "Revenue in March 2025 by region",
    "How many unique customers placed orders in 2025?",
    "ARR from Platform products only",
    "Average order size for Enterprise customers",

    # --- Time comparison ---
    "Revenue growth: Q4 2024 vs Q1 2025",
    "Was January 2025 revenue higher than December 2024?",
    "Which month had the highest revenue?",
    "MRR trend: list MRR for each month a subscription started",

    # --- NRR / MRR movements (derived chain) ---
    "NRR last month",
    "How many customers churned?",
    "Total expansion MRR",
    "Which customers are new in the MRR movements?",

    # --- Tricky / ambiguous ---
    "Which customers were active in 2025?",
    "Show me the first order we ever got",
]


def fmt_val(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:,.1f}"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, list):
        return f"{len(v)}r"
    return str(v)[:12]


def normalize_val(v):
    """Normalize value for consistency comparison."""
    if v is None:
        return None
    if isinstance(v, float):
        return round(v, 1)
    if isinstance(v, int):
        return v
    if isinstance(v, list):
        return len(v)
    return str(v)


def run_crl_once(question, con, db_path):
    """Run CRL approach once. Returns (value, elapsed, tokens)."""
    import json
    _last["sql"] = None
    t0 = time.time()

    p = plan(question)
    model_name = p.get("model")
    approach = p.get("approach", "patch")

    if approach == "reuse" and model_name:
        execute_reuse(p, con)
        val = _extract_value(con)
        return val, round(time.time() - t0, 1), 0

    if model_name and model_name in MODELS_CRL:
        crl = MODELS_CRL[model_name]
        header, sources, entities = split_crl(crl)
        qr = run_query(p.get("description", question), model_name, sources, entities, all_models_rl=ALL_MODELS_RL)
        tok = qr.get("total_output_tokens", 0)
        if qr["status"] == "compiled":
            orig_ids = _get_entity_ids(crl)
            _run_new_output_entities(qr, orig_ids, con, model_id=model_name, show_sql=False)
            val = _extract_value(con)
            return val, round(time.time() - t0, 1), tok

    return None, round(time.time() - t0, 1), 0


def run_sql_once(question, con):
    """Run direct SQL once. Returns (value, elapsed, tokens)."""
    r = _run_direct_sql(question, con)
    val = None
    if r["status"] == "ok" and r["rows"]:
        if len(r["rows"]) == 1 and len(r["rows"][0]) == 1:
            val = r["rows"][0][0]
        else:
            val = r["rows"]
    return val, r["elapsed_s"], r["output_tokens"]


def _extract_value(con):
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
    except:
        return None


def values_match(v1, v2) -> str:
    if v1 is None or v2 is None:
        return "n/a"
    n1, n2 = normalize_val(v1), normalize_val(v2)
    if isinstance(n1, (int, float)) and isinstance(n2, (int, float)):
        return "match" if abs(float(n1) - float(n2)) < 1 else "differ"
    if n1 == n2:
        return "match"
    # rows comparison
    if isinstance(v1, list) and isinstance(v2, list):
        def nums(rows):
            s = set()
            for r in rows:
                for v in r:
                    if isinstance(v, (int, float)):
                        s.add(round(float(v), 1))
            return s
        n1s, n2s = nums(v1), nums(v2)
        if n1s and n2s:
            overlap = n1s & n2s
            return "match" if len(overlap) >= min(len(n1s), len(n2s)) * 0.5 else "differ"
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


def consistency(vals):
    """Return consistency score: 'stable', 'unstable', or 'all_fail'."""
    normalized = [normalize_val(v) for v in vals]
    non_none = [v for v in normalized if v is not None]
    if not non_none:
        return "all_fail"
    if len(set(str(v) for v in non_none)) == 1 and len(non_none) == len(vals):
        return "stable"
    if len(non_none) < len(vals):
        return "flaky"  # some runs fail
    return "unstable"  # different values across runs


def main():
    db_path = str(Path(__file__).parent / "demo.db")
    if not Path(db_path).exists():
        create_db(db_path)
    con = duckdb.connect(db_path, read_only=True)

    n = len(QUESTIONS)
    results = []

    print(f"\n{BOLD}CRL vs SQL — {n} questions x {RUNS_PER_QUESTION} runs each{RESET}\n")

    for i, q in enumerate(QUESTIONS):
        print(f"{CYAN}[{i+1}/{n}]{RESET} {BOLD}{q}{RESET}")

        crl_vals, crl_times, crl_toks = [], [], []
        sql_vals, sql_times, sql_toks = [], [], []

        for run in range(RUNS_PER_QUESTION):
            cv, ct, ctok = run_crl_once(q, con, db_path)
            sv, st, stok = run_sql_once(q, con)
            crl_vals.append(cv); crl_times.append(ct); crl_toks.append(ctok)
            sql_vals.append(sv); sql_times.append(st); sql_toks.append(stok)
            sys.stdout.write(".")
            sys.stdout.flush()

        crl_con = consistency(crl_vals)
        sql_con = consistency(sql_vals)

        # Use first non-None value as representative
        crl_rep = next((v for v in crl_vals if v is not None), None)
        sql_rep = next((v for v in sql_vals if v is not None), None)
        match = values_match(crl_rep, sql_rep)

        crl_avg_t = sum(crl_times) / len(crl_times)
        sql_avg_t = sum(sql_times) / len(sql_times)
        crl_ok = sum(1 for v in crl_vals if v is not None)
        sql_ok = sum(1 for v in sql_vals if v is not None)

        match_sym = f"{GREEN}={RESET}" if match == "match" else f"{RED}X{RESET}" if match == "differ" else f"{DIM}?{RESET}"
        crl_con_sym = f"{GREEN}S{RESET}" if crl_con == "stable" else f"{YELLOW}F{RESET}" if crl_con == "flaky" else f"{RED}U{RESET}" if crl_con == "unstable" else f"{DIM}-{RESET}"
        sql_con_sym = f"{GREEN}S{RESET}" if sql_con == "stable" else f"{YELLOW}F{RESET}" if sql_con == "flaky" else f"{RED}U{RESET}" if sql_con == "unstable" else f"{DIM}-{RESET}"

        print(f" CRL:{crl_avg_t:>5.1f}s {crl_ok}/{RUNS_PER_QUESTION} {fmt_val(crl_rep):>10s} {crl_con_sym}  SQL:{sql_avg_t:>5.1f}s {sql_ok}/{RUNS_PER_QUESTION} {fmt_val(sql_rep):>10s} {sql_con_sym}  {match_sym}")

        results.append({
            "q": q, "match": match,
            "crl_vals": crl_vals, "crl_times": crl_times, "crl_con": crl_con, "crl_rep": crl_rep, "crl_avg_t": crl_avg_t, "crl_ok": crl_ok,
            "sql_vals": sql_vals, "sql_times": sql_times, "sql_con": sql_con, "sql_rep": sql_rep, "sql_avg_t": sql_avg_t, "sql_ok": sql_ok,
        })

    # Summary table
    print(f"\n{BOLD}{'=' * 115}{RESET}")
    print(f"{BOLD}{'#':>2} {'Question':<42s} {'CRL':>5s} {'ok':>4s} {'con':>3s} {'val':>10s}  {'SQL':>5s} {'ok':>4s} {'con':>3s} {'val':>10s} {'':>3s}{RESET}")
    print(f"{'-' * 115}")

    for i, r in enumerate(results):
        match_sym = "=" if r["match"] == "match" else "X" if r["match"] == "differ" else "?"
        crl_c = "S" if r["crl_con"] == "stable" else "F" if r["crl_con"] == "flaky" else "U" if r["crl_con"] == "unstable" else "-"
        sql_c = "S" if r["sql_con"] == "stable" else "F" if r["sql_con"] == "flaky" else "U" if r["sql_con"] == "unstable" else "-"
        print(f"{i+1:>2} {r['q']:<42s} {r['crl_avg_t']:>4.1f}s {r['crl_ok']:>1d}/{RUNS_PER_QUESTION}  {crl_c:>1s} {fmt_val(r['crl_rep']):>10s}  {r['sql_avg_t']:>4.1f}s {r['sql_ok']:>1d}/{RUNS_PER_QUESTION}  {sql_c:>1s} {fmt_val(r['sql_rep']):>10s}  {match_sym:>1s}")

    print(f"{'-' * 115}")

    # Aggregates
    crl_total_ok = sum(r["crl_ok"] for r in results)
    sql_total_ok = sum(r["sql_ok"] for r in results)
    crl_stable = sum(1 for r in results if r["crl_con"] == "stable")
    sql_stable = sum(1 for r in results if r["sql_con"] == "stable")
    crl_flaky = sum(1 for r in results if r["crl_con"] == "flaky")
    sql_flaky = sum(1 for r in results if r["sql_con"] == "flaky")
    crl_unstable = sum(1 for r in results if r["crl_con"] == "unstable")
    sql_unstable = sum(1 for r in results if r["sql_con"] == "unstable")
    matches = sum(1 for r in results if r["match"] == "match")
    differs = sum(1 for r in results if r["match"] == "differ")
    crl_total_t = sum(r["crl_avg_t"] for r in results)
    sql_total_t = sum(r["sql_avg_t"] for r in results)

    print(f"\n{BOLD}Summary:{RESET}")
    print(f"  CRL: {crl_total_ok}/{n*RUNS_PER_QUESTION} runs succeeded, {crl_total_t:.0f}s avg total")
    print(f"       Stability: {GREEN}{crl_stable} stable{RESET}, {YELLOW}{crl_flaky} flaky{RESET}, {RED}{crl_unstable} unstable{RESET}, {n - crl_stable - crl_flaky - crl_unstable} all_fail")
    print(f"  SQL: {sql_total_ok}/{n*RUNS_PER_QUESTION} runs succeeded, {sql_total_t:.0f}s avg total")
    print(f"       Stability: {GREEN}{sql_stable} stable{RESET}, {YELLOW}{sql_flaky} flaky{RESET}, {RED}{sql_unstable} unstable{RESET}, {n - sql_stable - sql_flaky - sql_unstable} all_fail")
    print(f"  {GREEN}Match: {matches}{RESET}  {RED}Differ: {differs}{RESET}  N/A: {n - matches - differs}")

    if differs > 0:
        print(f"\n{RED}  Differing results:{RESET}")
        for r in results:
            if r["match"] == "differ":
                print(f"    {r['q']}: CRL={fmt_val(r['crl_rep'])} vs SQL={fmt_val(r['sql_rep'])}")

    if crl_unstable > 0 or sql_unstable > 0:
        print(f"\n{RED}  Unstable results (different values across runs):{RESET}")
        for r in results:
            if r["crl_con"] == "unstable":
                vals = [fmt_val(v) for v in r["crl_vals"]]
                print(f"    CRL {r['q']}: {vals}")
            if r["sql_con"] == "unstable":
                vals = [fmt_val(v) for v in r["sql_vals"]]
                print(f"    SQL {r['q']}: {vals}")

    con.close()


if __name__ == "__main__":
    main()
