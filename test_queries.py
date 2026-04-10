"""Validate talk-to-your-data test dataset.

For each .river policy: parse, compile, generate DuckDB SQL for each output entity,
and run against the demo database. Then run example filtered queries.

Usage:
    python test_queries.py
"""

from __future__ import annotations

from pathlib import Path

import duckdb
from riverlang.ast.parser import RiverLangParser
from riverlang.compiler.compiler import Compiler
from riverlang.compiler.fqn import FQN

from setup_db import create_db

POLICY_DIR = Path(__file__).parent / "policies"


def print_table(rows: list[tuple], cols: list[str]) -> None:
    """Pretty-print query results."""
    widths = [max(len(str(c)), *(len(str(r[i])) for r in rows)) for i, c in enumerate(cols)]
    header = " | ".join(f"{c:>{w}}" for c, w in zip(cols, widths))
    print(f"  {header}")
    print(f"  {'-' * len(header)}")
    for row in rows:
        print(f"  {' | '.join(f'{str(v):>{w}}' for v, w in zip(row, widths))}")


def compile_policy(path: Path) -> tuple[str, dict[str, str]]:
    """Compile a .river file and return (model_id, {entity_id: sql})."""
    rl = path.read_text()
    ms = RiverLangParser.from_str(rl)
    result = Compiler(ms).verify()
    if result.has_errors():
        errors = [str(e) for e in result.errors]
        raise RuntimeError(f"Compilation failed for {path.name}:\n" + "\n".join(errors))

    model = ms.models[0]
    sql_gen = result.get_sql_generator()
    sqls: dict[str, str] = {}
    for elem in model.elements:
        if hasattr(elem.element, "is_output") and elem.element.is_output:
            fqn = FQN((model.id, elem.id))
            sqls[elem.id] = sql_gen.generate_sql(fqn, dialect="duckdb")
    return model.id, sqls


def run_query(con: duckdb.DuckDBPyConnection, sql: str, label: str) -> list[tuple]:
    """Run a SQL query and print results."""
    print(f"\n  {label}")
    rows = con.execute(sql).fetchall()
    cols = [d[0] for d in con.description]
    if rows:
        print_table(rows, cols)
    else:
        print("  (no results)")
    return rows


def main() -> None:
    # Create demo database
    print("Creating demo database...")
    con = create_db(":memory:")
    print()

    # Compile all policies and run base queries
    all_sqls: dict[str, dict[str, str]] = {}
    for f in sorted(POLICY_DIR.glob("*.river")):
        model_id, sqls = compile_policy(f)
        all_sqls[model_id] = sqls
        print(f"=== {f.stem} ({model_id}) ===")
        for entity_id, sql in sqls.items():
            rows = run_query(con, sql, f"{entity_id}:")
            assert rows, f"Expected results for {model_id}.{entity_id}"
        print()

    # Example filtered queries (simulating "talk to your data")
    print("=" * 60)
    print("FILTERED QUERIES (simulating user questions)")
    print("=" * 60)

    # Q1: "What's our revenue in March?"
    sql = all_sqls["revenue_analytics"]["monthly_revenue"]
    filtered = f"SELECT * FROM ({sql}) t WHERE t.month = 3 AND t.year = 2025"
    rows = run_query(con, filtered, 'Q: "What\'s our revenue in March?"')
    assert len(rows) == 1 and rows[0][2] > 0, "Expected 1 row with positive revenue for March"

    # Q2: "Revenue from Enterprise customers"
    sql = all_sqls["revenue_analytics"]["revenue_by_segment"]
    filtered = f"SELECT * FROM ({sql}) t WHERE t.segment = 'Enterprise'"
    rows = run_query(con, filtered, 'Q: "Revenue from Enterprise customers"')
    assert len(rows) == 1, "Expected 1 row for Enterprise segment"

    # Q3: "What's our total ARR?"
    sql = all_sqls["subscription_arr"]["arr_by_customer"]
    total = f"SELECT SUM(t.arr) as total_arr, SUM(t.sub_count) as total_subs FROM ({sql}) t"
    rows = run_query(con, total, 'Q: "What\'s our total ARR?"')
    assert rows[0][0] > 0, "Expected positive total ARR"

    # Q4: "Top 5 customers by spend"
    sql = all_sqls["customer_health"]["customer_summary"]
    top5 = f"SELECT * FROM ({sql}) t ORDER BY t.total_spend DESC LIMIT 5"
    rows = run_query(con, top5, 'Q: "Top 5 customers by spend"')
    assert len(rows) == 5, "Expected 5 rows"
    assert rows[0][4] >= rows[1][4], "Expected descending order"

    # Q5: "ARR by segment"
    sql = all_sqls["subscription_arr"]["arr_by_segment"]
    rows = run_query(con, sql, 'Q: "ARR by segment"')
    assert len(rows) >= 2, "Expected multiple segments"

    print("\n" + "=" * 60)
    print("ALL QUERIES PASSED")
    print("=" * 60)

    con.close()


if __name__ == "__main__":
    main()
