"""Run 25 ad-hoc query tests against the policy-model-agent using CRL compact format.

Same test suite as run_25_tests.py but sends policies in CRL (Compact RiverLang)
format to reduce output tokens and response time. A transpiler converts the agent's
CRL response back to a RiverLang AST for local compilation.

Each run creates a timestamped output folder under runs/ with:
  - summary.yaml: pass/fail counts, per-test status, timing comparison
  - one YAML file per test with full details
"""

import difflib
import json
import sys
import time
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import yaml
from riverlang.ast.parser import RiverLangParser
from riverlang.compiler.compiler import Compiler
from riverlang.compiler.fqn import FQN

sys.path.insert(0, str(Path(__file__).parent))
from setup_db import create_db
from compact_rl import riverlang_to_compact, compact_to_modelset, is_riverlang
from crl_grammar_spec import CRL_GRAMMAR_SPEC

PORT = 8090
BP = "5aa7bb42-cc55-425f-afae-a2c36474bbd8"
USER_ID = "00000000-0000-0000-0000-000000000001"
POLICY_DIR = Path(__file__).parent / "policies"
GOLDEN_DIR = Path(__file__).parent / "golden"
RUNS_DIR = Path(__file__).parent / "runs"

POLICIES = {p.stem: p.read_text() for p in sorted(POLICY_DIR.glob("*.river"))}
CRL_POLICIES = {k: riverlang_to_compact(v) for k, v in POLICIES.items()}

GOLDEN_IDS = [
    "arr_total", "arr_excl_apac", "arr_multi_sub", "avg_mrr",
    "arr_excl_pre2025", "arr_expiring_2025", "arr_pct_enterprise", "arr_midmarket_us",
    "rev_jan_feb", "rev_excl_below_500", "avg_order_by_segment", "rev_repeat_customers",
    "max_order", "rev_by_region", "rev_march_vs_jan", "unique_customers_march",
    "top3_by_spend", "customers_above_5k", "customers_emea", "most_orders_customer",
    "avg_spend_per_customer", "transactional_total", "count_by_size_tier",
    "avg_enterprise_standard", "large_deals_march",
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
    """Check if expected value appears anywhere in agent results."""
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


def summarize_results(all_results):
    parts = []
    for eid, cols, rows in all_results:
        if not rows:
            parts.append(f"{eid}: (empty)")
        elif len(rows) == 1 and len(cols) == 1:
            v = rows[0][0]
            parts.append(f"{eid}: {v:,.1f}" if isinstance(v, float) else f"{eid}: {v}")
        elif len(rows) == 1:
            vals = [f"{v:,.1f}" if isinstance(v, float) else str(v) for v in rows[0]]
            parts.append(f"{eid}: [{', '.join(vals)}]")
        else:
            sums = []
            for ci, c in enumerate(cols):
                col_vals = [r[ci] for r in rows if isinstance(r[ci], (int, float))]
                if col_vals:
                    sums.append(f"SUM({c})={sum(col_vals):,.1f}")
            parts.append(f"{eid}: {len(rows)} rows, {', '.join(sums)}")
    return " | ".join(parts)


def results_to_serializable(all_results):
    out = []
    for eid, cols, rows in all_results:
        entity_data = {
            "entity_id": eid,
            "columns": cols,
            "row_count": len(rows),
            "rows": [dict(zip(cols, [float(v) if isinstance(v, float) else v for v in row])) for row in rows],
        }
        out.append(entity_data)
    return out


def run_test(i, golden_id, con):
    """Run a single test using CRL format. Returns (status, detail_dict)."""
    golden = load_golden(golden_id)
    if not golden:
        return "SKIP", {"error": f"No golden file for {golden_id}"}

    question = golden["question"]
    policy_key = golden["policy"]
    expected = golden.get("golden_answer")

    crl_policy = CRL_POLICIES[policy_key]
    original_rl = POLICIES[policy_key]

    detail = {
        "id": golden_id,
        "question": question,
        "policy": policy_key,
        "golden_answer": expected,
        "mode": "crl",
        "input_chars": len(crl_policy),
        "original_chars": len(original_rl),
    }

    # Create thread
    thread_resp = api_call("/v1/threads", {"metadata": {"blueprint_id": BP, "user_id": USER_ID}})
    if "thread_id" not in thread_resp:
        detail["error"] = str(thread_resp)
        return "ERROR", detail
    tid = thread_resp["thread_id"]

    # Build message with CRL grammar spec
    crl_message = f"{CRL_GRAMMAR_SPEC}\n\nQuestion: {question}"

    # Invoke agent with CRL policy
    t0 = time.time()
    resp = api_call(f"/v1/threads/{tid}/runs/wait", {
        "assistant_id": "policy-model-agent",
        "input": {
            "operation": "update",
            "should_compile": False,
            "should_analyze": False,
            "current_model_str": crl_policy,
            "messages": [{"role": "user", "content": crl_message}],
        },
    })
    elapsed = time.time() - t0
    detail["elapsed_s"] = round(elapsed, 1)

    if resp.get("detail"):
        detail["error"] = resp["detail"][:200]
        return "ERROR", detail

    agent_output = resp.get("current_model_str", "")
    detail["output_chars"] = len(agent_output)

    # Detect format and parse
    if is_riverlang(agent_output):
        detail["format"] = "riverlang_fallback"
        try:
            ms = RiverLangParser.from_str(agent_output)
        except Exception as e:
            detail["error"] = f"RL parse error: {e}"
            return "COMPILE_FAIL", detail
    else:
        detail["format"] = "crl"
        try:
            ms = compact_to_modelset(agent_output)
        except Exception as e:
            detail["error"] = f"CRL parse error: {e}"
            detail["agent_output"] = agent_output[:500]
            return "COMPILE_FAIL", detail

    # Compile
    try:
        cr = Compiler(ms).verify()
        if cr.has_errors():
            detail["compile_errors"] = [str(e) for e in cr.errors]
            return "COMPILE_FAIL", detail
    except Exception as e:
        detail["error"] = f"Compile error: {e}"
        return "COMPILE_FAIL", detail

    detail["compiled"] = True

    # CRL diff
    diff_lines = [
        line for line in difflib.unified_diff(
            crl_policy.split("\n"), agent_output.split("\n"), lineterm="", n=1
        )
        if (line.startswith("+") or line.startswith("-"))
        and not line.startswith("+++") and not line.startswith("---")
    ]
    detail["crl_diff"] = diff_lines

    # Find new/output entities
    model = ms.models[0]
    orig_ms = compact_to_modelset(crl_policy)
    orig_ids = {e.id for e in orig_ms.models[0].elements}
    new_ents = [
        e for e in model.elements
        if e.id not in orig_ids and hasattr(e.element, "is_output") and e.element.is_output
    ]
    if not new_ents:
        new_ents = [e for e in model.elements if hasattr(e.element, "is_output") and e.element.is_output]

    # Generate SQL and run
    sql_gen = cr.get_sql_generator()
    all_results = []
    sql_error = None
    for ent in new_ents:
        try:
            sql = sql_gen.generate_sql(FQN((model.id, ent.id)), dialect="duckdb")
            rows = con.execute(sql).fetchall()
            cols = [d[0] for d in con.description]
            all_results.append((ent.id, cols, rows))
            detail.setdefault("generated_sql", {})[ent.id] = sql
        except Exception as e:
            sql_error = str(e)[:200]

    detail["agent_results"] = results_to_serializable(all_results)
    detail["agent_summary"] = summarize_results(all_results)

    if sql_error:
        detail["sql_error"] = sql_error
        return "SQL_ERROR", detail

    if expected is not None:
        matched = match_expected(expected, all_results)
        return "PASS" if matched else "FAIL", detail

    if all_results:
        return "PRODUCED", detail

    return "NO_RESULT", detail


def main():
    con = create_db(":memory:")

    # Create run output folder
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / f"crl_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {run_dir}")
    print(f"Mode: CRL (Compact RiverLang)\n")

    # Show CRL size savings
    for key in sorted(CRL_POLICIES):
        orig = len(POLICIES[key])
        compact = len(CRL_POLICIES[key])
        print(f"  {key}: {orig} -> {compact} chars ({100-compact*100//orig}% reduction)")
    print()

    results = []
    icons = {
        "PASS": "  OK", "FAIL": "FAIL", "ERROR": " ERR", "COMPILE_FAIL": "COMP",
        "SQL_ERROR": " SQL", "PRODUCED": " RAN", "NO_RESULT": "NONE", "SKIP": "SKIP",
    }

    for i, golden_id in enumerate(GOLDEN_IDS):
        golden = load_golden(golden_id)
        desc = golden.get("question", golden_id) if golden else golden_id
        print(f"[{i + 1:2d}/25] {golden_id}")
        print(f"  Q: \"{desc}\"")

        status, detail = run_test(i, golden_id, con)
        results.append((status, golden_id, detail))

        elapsed = detail.get("elapsed_s", 0)
        fmt = detail.get("format", "?")
        if status == "PASS":
            print(f"  PASS ({elapsed}s, {fmt}) — {detail.get('agent_summary', '')}")
        elif status == "FAIL":
            print(f"  FAIL ({elapsed}s, {fmt}) — {detail.get('agent_summary', '')}")
        elif status in ("ERROR", "COMPILE_FAIL", "SQL_ERROR"):
            print(f"  {status} ({elapsed}s, {fmt}) — {detail.get('error', detail.get('sql_error', ''))[:100]}")
        else:
            print(f"  {status} ({elapsed}s, {fmt})")

        # Write per-test output file
        test_output = {k: v for k, v in detail.items() if k != "agent_output"}
        test_output["status"] = status
        with open(run_dir / f"{golden_id}.yaml", "w") as f:
            yaml.dump(test_output, f, default_flow_style=False, sort_keys=False, allow_unicode=True, width=120)

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY (CRL mode)")
    print(f"{'=' * 70}")
    counts = Counter(s for s, _, _ in results)
    total_elapsed = sum(d.get("elapsed_s", 0) for _, _, d in results)
    crl_count = sum(1 for _, _, d in results if d.get("format") == "crl")
    fallback_count = sum(1 for _, _, d in results if d.get("format") == "riverlang_fallback")

    for status, gid, detail in results:
        elapsed = detail.get("elapsed_s", 0)
        fmt = detail.get("format", "?")
        print(f"  [{icons.get(status, '  ? ')}] {gid} ({elapsed}s, {fmt})")

    print(
        f"\nPASS: {counts['PASS']} | FAIL: {counts['FAIL']}"
        f" | SQL_ERROR: {counts['SQL_ERROR']} | COMPILE_FAIL: {counts['COMPILE_FAIL']}"
        f" | ERROR: {counts['ERROR']} | NO_RESULT: {counts['NO_RESULT']}"
        f" | Total: {len(results)}"
    )
    print(f"Total elapsed: {total_elapsed:.1f}s")
    print(f"Format: CRL={crl_count}, RL_fallback={fallback_count}")

    # Write summary file
    summary = {
        "timestamp": timestamp,
        "mode": "crl",
        "total": len(results),
        "counts": dict(counts),
        "total_elapsed_s": round(total_elapsed, 1),
        "crl_format_count": crl_count,
        "fallback_count": fallback_count,
        "tests": [
            {"id": gid, "status": s, "elapsed_s": d.get("elapsed_s", 0), "format": d.get("format", "?")}
            for s, gid, d in results
        ],
    }
    with open(run_dir / "summary.yaml", "w") as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False)

    print(f"\nResults saved to: {run_dir}")
    con.close()


if __name__ == "__main__":
    main()
