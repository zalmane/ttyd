"""Benchmark: compare response times between agent (RL) and direct LLM (CRL) modes.

RL mode: calls the policy-model-agent API at localhost:8090
CRL mode: calls Bedrock directly with compact grammar + compilation loop

Usage:
    python benchmark.py              # run default subset
    python benchmark.py --all        # run all 25 tests
    python benchmark.py --ids arr_total,rev_repeat_customers
    python benchmark.py --crl-only   # skip RL baseline
    python benchmark.py --workers 5  # parallel workers
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from setup_db import create_db
from compact_rl import (
    riverlang_to_compact,
    compact_to_modelset,
    split_crl,
    merge_crl,
)
from llm_runner import run_query

from riverlang.ast.parser import RiverLangParser
from riverlang.compiler.compiler import Compiler
from riverlang.compiler.fqn import FQN

PORT = 8090
BP = "5aa7bb42-cc55-425f-afae-a2c36474bbd8"
USER_ID = "00000000-0000-0000-0000-000000000001"
POLICY_DIR = Path(__file__).parent / "policies"
GOLDEN_DIR = Path(__file__).parent / "golden"
RUNS_DIR = Path(__file__).parent / "runs"

POLICIES = {p.stem: p.read_text() for p in sorted(POLICY_DIR.glob("*.river"))}
CRL_POLICIES = {k: riverlang_to_compact(v) for k, v in POLICIES.items()}

ALL_IDS = [
    "arr_total", "arr_excl_apac", "arr_multi_sub", "avg_mrr",
    "arr_excl_pre2025", "arr_expiring_2025", "arr_pct_enterprise", "arr_midmarket_us",
    "rev_jan_feb", "rev_excl_below_500", "avg_order_by_segment", "rev_repeat_customers",
    "max_order", "rev_by_region", "rev_march_vs_jan", "unique_customers_march",
    "top3_by_spend", "customers_above_5k", "customers_emea", "most_orders_customer",
    "avg_spend_per_customer", "transactional_total", "count_by_size_tier",
    "avg_enterprise_standard", "large_deals_march",
]

DEFAULT_SUBSET = [
    "arr_total", "top3_by_spend", "arr_expiring_2025", "avg_order_by_segment",
    "customers_above_5k", "arr_pct_enterprise", "rev_repeat_customers", "rev_march_vs_jan",
]


def api_call(path, data):
    req = urllib.request.Request(
        f"http://localhost:{PORT}{path}",
        data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:200]
        return {"detail": f"HTTP {e.code}: {body}"}
    except (TimeoutError, OSError) as e:
        return {"detail": f"Timeout: {e}"}


def load_golden(test_id: str) -> dict | None:
    path = GOLDEN_DIR / f"{test_id}.yaml"
    if not path.exists():
        return None
    with open(path) as f:
        return yaml.safe_load(f)


def match_expected(expected, all_results):
    if isinstance(expected, (int, float)):
        target = float(expected)
        for _eid, cols, rows in all_results:
            if not rows:
                continue
            for row in rows:
                for val in row:
                    if isinstance(val, (int, float)) and abs(float(val) - target) < 1:
                        return True
            for col_idx in range(len(cols)):
                col_vals = [r[col_idx] for r in rows if isinstance(r[col_idx], (int, float))]
                if col_vals and abs(sum(col_vals) - target) < 1:
                    return True
            if abs(len(rows) - target) < 1:
                return True
        return False
    if isinstance(expected, list):
        for _eid, cols, rows in all_results:
            if not rows:
                continue
            agent_dicts = [dict(zip(cols, row)) for row in rows]
            matched = 0
            for exp_row in expected:
                for agent_row in agent_dicts:
                    if all(
                        str(agent_row.get(k, "")) == str(v)
                        or (isinstance(v, (int, float)) and isinstance(agent_row.get(k), (int, float))
                            and abs(float(agent_row[k]) - float(v)) < 1)
                        for k, v in exp_row.items()
                        if k in agent_row
                    ):
                        matched += 1
                        break
            if matched == len(expected):
                return True
        return False
    return False


# ---------------------------------------------------------------------------
# RL mode: agent API at localhost:8090
# ---------------------------------------------------------------------------

def run_rl(golden_id: str) -> dict:
    golden = load_golden(golden_id)
    if not golden:
        return {"status": "SKIP", "elapsed_s": 0}

    question = golden["question"]
    policy_key = golden["policy"]
    expected = golden.get("golden_answer")

    thread_resp = api_call("/v1/threads", {"metadata": {"blueprint_id": BP, "user_id": USER_ID}})
    if "thread_id" not in thread_resp:
        return {"status": "ERROR", "elapsed_s": 0, "error": str(thread_resp)[:100]}
    tid = thread_resp["thread_id"]

    t0 = time.time()
    resp = api_call(f"/v1/threads/{tid}/runs/wait", {
        "assistant_id": "policy-model-agent",
        "input": {
            "operation": "update",
            "should_compile": True,
            "should_analyze": False,
            "current_model_str": POLICIES[policy_key],
            "messages": [{"role": "user", "content": question}],
        },
    })
    elapsed = time.time() - t0
    result = {"elapsed_s": round(elapsed, 1), "mode": "rl"}

    if resp.get("detail"):
        result["status"] = "ERROR"
        result["error"] = resp["detail"][:100]
        return result

    agent_rl = resp.get("current_model_str", "")
    result["output_chars"] = len(agent_rl)

    try:
        ms = RiverLangParser.from_str(agent_rl)
        cr = Compiler(ms).verify()
        if cr.has_errors():
            result["status"] = "COMPILE_FAIL"
            return result
    except Exception as e:
        result["status"] = "COMPILE_FAIL"
        result["error"] = str(e)[:100]
        return result

    con = create_db(":memory:")
    status, result_extras = _run_sql(cr, ms, POLICIES[policy_key], expected, con, parse_orig_fn=RiverLangParser.from_str)
    con.close()
    result["status"] = status
    result.update(result_extras)
    return result


# ---------------------------------------------------------------------------
# CRL mode: direct Bedrock LLM calls
# ---------------------------------------------------------------------------

def run_crl(golden_id: str) -> dict:
    golden = load_golden(golden_id)
    if not golden:
        return {"status": "SKIP", "elapsed_s": 0}

    question = golden["question"]
    policy_key = golden["policy"]
    expected = golden.get("golden_answer")
    crl_policy = CRL_POLICIES[policy_key]
    header, sources, entities = split_crl(crl_policy)

    # Direct LLM call with compilation loop
    qr = run_query(question, policy_key, sources, entities)

    result = {
        "mode": "crl",
        "elapsed_s": qr["elapsed_s"],
        "attempts": qr["attempts"],
        "output_tokens": qr["total_output_tokens"],
        "input_tokens": qr["total_input_tokens"],
        "output_chars": len(qr.get("entities", "")),
    }

    if qr["status"] != "compiled":
        result["status"] = "COMPILE_FAIL"
        result["error"] = "; ".join(qr.get("last_errors", [])[:2])
        return result

    # Run SQL
    cr = qr["compiler_result"]
    ms = qr["model_set"]
    con = create_db(":memory:")
    status, result_extras = _run_sql(cr, ms, crl_policy, expected, con, parse_orig_fn=compact_to_modelset)
    con.close()
    result["status"] = status
    result.update(result_extras)
    return result


# ---------------------------------------------------------------------------
# Shared SQL execution
# ---------------------------------------------------------------------------

def _run_sql(cr, ms, orig_policy_str, expected, con, parse_orig_fn):
    """Run SQL for output entities. Returns (status, extras_dict)."""
    model = ms.models[0]
    try:
        orig_ms = parse_orig_fn(orig_policy_str)
        orig_ids = {e.id for e in orig_ms.models[0].elements}
    except Exception:
        orig_ids = set()

    new_ents = [
        e for e in model.elements
        if e.id not in orig_ids and hasattr(e.element, "is_output") and e.element.is_output
    ]
    if not new_ents:
        new_ents = [e for e in model.elements if hasattr(e.element, "is_output") and e.element.is_output]

    sql_gen = cr.get_sql_generator()
    all_results = []
    for ent in new_ents:
        try:
            sql = sql_gen.generate_sql(FQN((model.id, ent.id)), dialect="duckdb")
            rows = con.execute(sql).fetchall()
            cols = [d[0] for d in con.description]
            all_results.append((ent.id, cols, rows))
        except Exception as e:
            return "SQL_ERROR", {"sql_error": str(e)[:200]}

    if expected is not None:
        matched = match_expected(expected, all_results)
        return ("PASS" if matched else "FAIL"), {}
    if all_results:
        return "PRODUCED", {}
    return "NO_RESULT", {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark RL (agent) vs CRL (direct LLM)")
    parser.add_argument("--all", action="store_true", help="Run all 25 tests")
    parser.add_argument("--ids", type=str, help="Comma-separated test IDs")
    parser.add_argument("--crl-only", action="store_true", help="Only run CRL mode")
    parser.add_argument("--rl-only", action="store_true", help="Only run RL mode")
    parser.add_argument("--workers", type=int, default=5, help="Parallel workers")
    args = parser.parse_args()

    if args.ids:
        test_ids = [x.strip() for x in args.ids.split(",")]
    elif args.all:
        test_ids = ALL_IDS
    else:
        test_ids = DEFAULT_SUBSET

    modes = []
    if not args.crl_only:
        modes.append("rl")
    if not args.rl_only:
        modes.append("crl")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / f"bench_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Benchmark: {len(test_ids)} tests x {len(modes)} modes, {args.workers} workers")
    print(f"Output: {run_dir}\n", flush=True)

    all_data: dict[str, dict] = {gid: {"id": gid} for gid in test_ids}

    for mode in modes:
        runner = run_rl if mode == "rl" else run_crl
        label = "RL (agent API)" if mode == "rl" else "CRL (direct LLM)"
        print(f"--- {label}: {len(test_ids)} tests, {args.workers} workers ---", flush=True)
        t_mode_start = time.time()

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(runner, gid): gid for gid in test_ids}
            for future in as_completed(futures):
                gid = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = {"status": "ERROR", "elapsed_s": 0, "error": str(e)[:100]}
                all_data[gid][mode] = result
                status = result.get("status", "?")
                elapsed = result.get("elapsed_s", 0)
                out_ch = result.get("output_chars", 0)
                out_tok = result.get("output_tokens", 0)
                attempts = result.get("attempts", "")
                att_str = f" x{attempts}" if attempts and attempts > 1 else ""
                tok_str = f" {out_tok}tok" if out_tok else f" {out_ch}ch"
                print(f"  {mode.upper():3s} {gid:<25s} {status:12s} {elapsed:6.1f}s{tok_str}{att_str}", flush=True)

        t_wall = time.time() - t_mode_start
        print(f"  Wall time: {t_wall:.1f}s\n", flush=True)

    # Summary table
    print(f"{'=' * 90}")
    print(f"{'Test ID':<25s} ", end="")
    for m in modes:
        print(f"  {m.upper():>6s}        ", end="")
    if len(modes) == 2:
        print(f"  {'Speedup':>8s}", end="")
    print()
    print("-" * 90)

    total = {m: 0.0 for m in modes}
    pass_count = {m: 0 for m in modes}
    for gid in test_ids:
        row = all_data[gid]
        print(f"{gid:<25s} ", end="")
        for m in modes:
            r = row.get(m, {})
            t = r.get("elapsed_s", 0)
            s = r.get("status", "?")
            att = r.get("attempts", 1)
            icon = "v" if s == "PASS" else "x"
            att_str = f"x{att}" if att and att > 1 else "  "
            print(f"  {icon} {t:5.1f}s {att_str}   ", end="")
            total[m] += t
            if s == "PASS":
                pass_count[m] += 1
        if len(modes) == 2:
            rl_t = row.get("rl", {}).get("elapsed_s", 0)
            crl_t = row.get("crl", {}).get("elapsed_s", 0)
            if rl_t > 0:
                speedup = (rl_t - crl_t) / rl_t * 100
                print(f"  {speedup:+6.0f}%", end="")
        print()

    print("-" * 90)
    print(f"{'TOTAL':<25s} ", end="")
    for m in modes:
        print(f"    {total[m]:5.1f}s        ", end="")
    if len(modes) == 2 and total.get("rl", 0) > 0:
        print(f"  {(total['rl']-total['crl'])/total['rl']*100:+6.0f}%", end="")
    print()
    for m in modes:
        print(f"  {m.upper()} pass: {pass_count[m]}/{len(test_ids)}", end="")
    print(flush=True)

    # Save
    with open(run_dir / "benchmark.yaml", "w") as f:
        yaml.dump({
            "timestamp": timestamp,
            "modes": modes,
            "tests": [all_data[gid] for gid in test_ids],
        }, f, default_flow_style=False, sort_keys=False)
    print(f"\nSaved to: {run_dir}/benchmark.yaml")


if __name__ == "__main__":
    main()
