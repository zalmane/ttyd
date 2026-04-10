"""Create a DuckDB with SaaS demo data for talk-to-your-data experiments.

Tables: orders, customers, subscriptions, products
Data: ~100 rows per table, deterministic (seeded), Jan-Apr 2025, 10 customers, 5 products.

Usage:
    python setup_db.py           # creates demo.db in current dir
    python setup_db.py :memory:  # returns in-memory connection (for import)
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb

DB_PATH = Path(__file__).parent / "demo.db"

PRODUCTS = [
    ("PROD-01", "Platform Pro", "Platform", 599.00),
    ("PROD-02", "Platform Starter", "Platform", 199.00),
    ("PROD-03", "Analytics Add-on", "Add-on", 149.00),
    ("PROD-04", "Security Add-on", "Add-on", 99.00),
    ("PROD-05", "Professional Services", "Services", 2500.00),
]

CUSTOMERS = [
    ("CUST-01", "Acme Corp", "Enterprise", "US", "2024-01-15"),
    ("CUST-02", "Globex Inc", "Enterprise", "EMEA", "2024-02-20"),
    ("CUST-03", "Initech", "Mid-Market", "US", "2024-03-10"),
    ("CUST-04", "Umbrella Ltd", "Mid-Market", "APAC", "2024-04-05"),
    ("CUST-05", "Stark Industries", "Enterprise", "US", "2024-05-18"),
    ("CUST-06", "Wayne Enterprises", "Enterprise", "EMEA", "2024-06-22"),
    ("CUST-07", "Cyberdyne Systems", "SMB", "US", "2024-07-30"),
    ("CUST-08", "Soylent Corp", "SMB", "APAC", "2024-08-14"),
    ("CUST-09", "Aperture Science", "Mid-Market", "EMEA", "2024-09-01"),
    ("CUST-10", "Massive Dynamic", "Enterprise", "US", "2024-10-12"),
]

# fmt: off
ORDERS = [
    # Jan 2025
    ("ORD-001", "CUST-01", "PROD-01", 4500.00, "2025-01-05", "completed"),
    ("ORD-002", "CUST-02", "PROD-01", 4500.00, "2025-01-08", "completed"),
    ("ORD-003", "CUST-03", "PROD-02", 1200.00, "2025-01-12", "completed"),
    ("ORD-004", "CUST-07", "PROD-02", 800.00,  "2025-01-15", "completed"),
    ("ORD-005", "CUST-05", "PROD-03", 900.00,  "2025-01-20", "completed"),
    ("ORD-006", "CUST-01", "PROD-05", 2500.00, "2025-01-25", "completed"),
    ("ORD-007", "CUST-04", "PROD-04", 600.00,  "2025-01-28", "pending"),
    # Feb 2025
    ("ORD-008", "CUST-06", "PROD-01", 4500.00, "2025-02-03", "completed"),
    ("ORD-009", "CUST-03", "PROD-03", 450.00,  "2025-02-07", "completed"),
    ("ORD-010", "CUST-08", "PROD-02", 800.00,  "2025-02-10", "completed"),
    ("ORD-011", "CUST-01", "PROD-03", 900.00,  "2025-02-14", "completed"),
    ("ORD-012", "CUST-05", "PROD-01", 4500.00, "2025-02-18", "completed"),
    ("ORD-013", "CUST-09", "PROD-02", 1200.00, "2025-02-22", "completed"),
    ("ORD-014", "CUST-02", "PROD-04", 600.00,  "2025-02-25", "cancelled"),
    # Mar 2025
    ("ORD-015", "CUST-10", "PROD-01", 4500.00, "2025-03-01", "completed"),
    ("ORD-016", "CUST-01", "PROD-01", 4500.00, "2025-03-05", "completed"),
    ("ORD-017", "CUST-03", "PROD-05", 2500.00, "2025-03-08", "completed"),
    ("ORD-018", "CUST-06", "PROD-03", 900.00,  "2025-03-12", "completed"),
    ("ORD-019", "CUST-04", "PROD-01", 4500.00, "2025-03-15", "completed"),
    ("ORD-020", "CUST-07", "PROD-04", 400.00,  "2025-03-18", "completed"),
    ("ORD-021", "CUST-02", "PROD-05", 2500.00, "2025-03-22", "completed"),
    ("ORD-022", "CUST-05", "PROD-02", 1200.00, "2025-03-25", "completed"),
    ("ORD-023", "CUST-08", "PROD-03", 450.00,  "2025-03-28", "pending"),
    # Apr 2025
    ("ORD-024", "CUST-01", "PROD-04", 600.00,  "2025-04-02", "completed"),
    ("ORD-025", "CUST-09", "PROD-01", 4500.00, "2025-04-05", "completed"),
    ("ORD-026", "CUST-10", "PROD-03", 900.00,  "2025-04-08", "completed"),
    ("ORD-027", "CUST-06", "PROD-05", 2500.00, "2025-04-12", "completed"),
    ("ORD-028", "CUST-03", "PROD-01", 4500.00, "2025-04-15", "completed"),
    ("ORD-029", "CUST-05", "PROD-04", 600.00,  "2025-04-18", "pending"),
    ("ORD-030", "CUST-02", "PROD-02", 1200.00, "2025-04-22", "completed"),
]

SUBSCRIPTIONS = [
    # Active subscriptions
    ("SUB-001", "CUST-01", "PROD-01", 500.00,  "2024-06-01", "2025-05-31", "active"),
    ("SUB-002", "CUST-02", "PROD-01", 500.00,  "2024-07-01", "2025-06-30", "active"),
    ("SUB-003", "CUST-03", "PROD-02", 170.00,  "2024-08-01", "2025-07-31", "active"),
    ("SUB-004", "CUST-05", "PROD-01", 500.00,  "2024-09-01", "2025-08-31", "active"),
    ("SUB-005", "CUST-06", "PROD-01", 500.00,  "2024-10-01", "2025-09-30", "active"),
    ("SUB-006", "CUST-01", "PROD-03", 130.00,  "2024-11-01", "2025-10-31", "active"),
    ("SUB-007", "CUST-04", "PROD-02", 170.00,  "2025-01-01", "2025-12-31", "active"),
    ("SUB-008", "CUST-10", "PROD-01", 500.00,  "2025-01-15", "2026-01-14", "active"),
    ("SUB-009", "CUST-09", "PROD-02", 170.00,  "2025-02-01", "2026-01-31", "active"),
    ("SUB-010", "CUST-05", "PROD-03", 130.00,  "2025-03-01", "2026-02-28", "active"),
    # Churned
    ("SUB-011", "CUST-07", "PROD-02", 170.00,  "2024-03-01", "2024-12-31", "churned"),
    ("SUB-012", "CUST-08", "PROD-02", 170.00,  "2024-05-01", "2025-01-31", "churned"),
    # Trial
    ("SUB-013", "CUST-07", "PROD-01", 0.00,    "2025-03-01", "2025-03-31", "trial"),
]
# fmt: on


def create_db(path: str | Path = DB_PATH) -> duckdb.DuckDBPyConnection:
    """Create and populate the demo DuckDB database."""
    con = duckdb.connect(str(path))

    con.execute("DROP TABLE IF EXISTS products")
    con.execute("DROP TABLE IF EXISTS customers")
    con.execute("DROP TABLE IF EXISTS orders")
    con.execute("DROP TABLE IF EXISTS subscriptions")

    con.execute("""
        CREATE TABLE products (
            product_id VARCHAR PRIMARY KEY,
            name VARCHAR,
            category VARCHAR,
            list_price DOUBLE
        )
    """)
    con.executemany("INSERT INTO products VALUES (?, ?, ?, ?)", PRODUCTS)

    con.execute("""
        CREATE TABLE customers (
            customer_id VARCHAR PRIMARY KEY,
            name VARCHAR,
            segment VARCHAR,
            region VARCHAR,
            signup_date DATE
        )
    """)
    con.executemany("INSERT INTO customers VALUES (?, ?, ?, ?, ?)", CUSTOMERS)

    con.execute("""
        CREATE TABLE orders (
            order_id VARCHAR PRIMARY KEY,
            customer_id VARCHAR,
            product_id VARCHAR,
            amount DOUBLE,
            order_date DATE,
            status VARCHAR
        )
    """)
    con.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?)", ORDERS)

    con.execute("""
        CREATE TABLE subscriptions (
            subscription_id VARCHAR PRIMARY KEY,
            customer_id VARCHAR,
            product_id VARCHAR,
            mrr DOUBLE,
            start_date DATE,
            end_date DATE,
            status VARCHAR
        )
    """)
    con.executemany("INSERT INTO subscriptions VALUES (?, ?, ?, ?, ?, ?, ?)", SUBSCRIPTIONS)

    return con


def print_summary(con: duckdb.DuckDBPyConnection) -> None:
    """Print a summary of the database contents."""
    for table in ["products", "customers", "orders", "subscriptions"]:
        count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table}: {count} rows")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else str(DB_PATH)
    print(f"Creating DuckDB at {target}")
    con = create_db(target)
    print_summary(con)
    con.close()
    print("Done.")
