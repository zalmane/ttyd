"""Unit tests for CRL (Compact RiverLang) transpiler round-trip correctness."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import duckdb
from riverlang.ast.parser import RiverLangParser
from riverlang.compiler.compiler import Compiler
from riverlang.compiler.fqn import FQN

from compact_rl import (
    compact_to_modelset,
    is_riverlang,
    riverlang_to_compact,
    _expand_dot_shorthand,
    _id_to_display,
)
from setup_db import create_db

POLICY_DIR = Path(__file__).parent / "policies"


def _get_output_entities(ms):
    """Get (model_id, element_id) for output entities."""
    result = []
    for model in ms.models:
        for elem in model.elements:
            if hasattr(elem.element, "is_output") and elem.element.is_output:
                result.append((model.id, elem.id))
    return result


def test_roundtrip_all_policies():
    """For each .river policy: RL -> CRL -> ModelSet -> compile -> SQL -> same results."""
    con = create_db(":memory:")

    for policy_path in sorted(POLICY_DIR.glob("*.river")):
        rl = policy_path.read_text()
        crl = riverlang_to_compact(rl)

        # Parse CRL back to ModelSet
        rt_ms = compact_to_modelset(crl)
        rt_cr = Compiler(rt_ms).verify()
        assert not rt_cr.has_errors(), f"Compile errors for {policy_path.stem}: {[str(e) for e in rt_cr.errors]}"

        # Compare with original
        orig_ms = RiverLangParser.from_str(rl)
        orig_cr = Compiler(orig_ms).verify()

        for model_id, elem_id in _get_output_entities(orig_ms):
            fqn = FQN((model_id, elem_id))
            orig_sql = orig_cr.get_sql_generator().generate_sql(fqn, dialect="duckdb")
            rt_sql = rt_cr.get_sql_generator().generate_sql(fqn, dialect="duckdb")

            orig_rows = sorted(con.execute(orig_sql).fetchall())
            rt_rows = sorted(con.execute(rt_sql).fetchall())
            assert orig_rows == rt_rows, f"Results differ for {policy_path.stem}.{elem_id}"

    con.close()
    print("PASS: All policies round-trip correctly")


def test_size_reduction():
    """CRL should be significantly smaller than RiverLang."""
    for policy_path in sorted(POLICY_DIR.glob("*.river")):
        rl = policy_path.read_text()
        crl = riverlang_to_compact(rl)
        ratio = len(crl) / len(rl)
        assert ratio < 0.55, f"{policy_path.stem}: only {(1-ratio)*100:.0f}% reduction (expected >45%)"
        print(f"  {policy_path.stem}: {len(rl)} -> {len(crl)} chars ({(1-ratio)*100:.0f}% reduction)")
    print("PASS: All policies have >45% size reduction")


def test_if_else_roundtrip():
    """If/else expressions in order_classification survive round-trip."""
    rl = (POLICY_DIR / "order_classification.river").read_text()
    crl = riverlang_to_compact(rl)

    # CRL should contain if/else
    assert "if (" in crl, "Expected if/else in CRL"
    assert "else" in crl, "Expected else in CRL"

    ms = compact_to_modelset(crl)
    cr = Compiler(ms).verify()
    assert not cr.has_errors(), f"Compile errors: {[str(e) for e in cr.errors]}"

    con = create_db(":memory:")
    fqn = FQN(("order_classification", "classified_orders"))
    sql = cr.get_sql_generator().generate_sql(fqn, dialect="duckdb")
    rows = con.execute(sql).fetchall()
    assert len(rows) == 26, f"Expected 26 rows, got {len(rows)}"

    # Check size_tier values exist
    size_tiers = {r[6] for r in rows}  # size_tier column
    assert "Large" in size_tiers
    assert "Small" in size_tiers
    con.close()
    print("PASS: if/else expressions round-trip correctly")


def test_annotations_preserved():
    """@key, @dimension, @metric annotations survive round-trip."""
    rl = (POLICY_DIR / "customer_health.river").read_text()
    crl = riverlang_to_compact(rl)

    # Check CRL has annotations
    assert "@key" in crl
    assert "@dim" in crl
    assert "@met" in crl

    ms = compact_to_modelset(crl)
    model = ms.models[0]
    # Find customer_summary entity
    for elem in model.elements:
        if elem.id == "customer_summary":
            for prop in elem.element.properties:
                sf = prop.schema_field
                if sf.id == "customer_id":
                    assert "key" in sf.annotations, f"Expected @key on customer_id"
                elif sf.id == "total_spend":
                    assert "metric" in sf.annotations, f"Expected @metric on total_spend"
                elif sf.id == "name":
                    assert "dimension" in sf.annotations, f"Expected @dimension on name"
            break
    else:
        assert False, "customer_summary entity not found"
    print("PASS: annotations preserved")


def test_dot_shorthand():
    """Test .X shorthand expansion to base.X."""
    assert _expand_dot_shorthand(".customer_id") == "base.customer_id"
    assert _expand_dot_shorthand(".foo") == "base.foo"
    assert _expand_dot_shorthand("base.to_customer.name") == "base.to_customer.name"
    assert _expand_dot_shorthand("sum(base.by_cust.mrr)") == "sum(base.by_cust.mrr)"
    print("PASS: dot shorthand works")


def test_is_riverlang_detection():
    """Test format detection for RiverLang vs CRL."""
    rl = (POLICY_DIR / "subscription_arr.river").read_text()
    crl = riverlang_to_compact(rl)

    assert is_riverlang(rl) is True, "Should detect RiverLang"
    assert is_riverlang(crl) is False, "Should detect CRL"
    print("PASS: format detection works")


def test_display_name_generation():
    """Test ID to display name conversion."""
    assert _id_to_display("customer_id") == "Customer Id"
    assert _id_to_display("arr") == "Arr"
    assert _id_to_display("total_spend") == "Total Spend"
    assert _id_to_display("mrr") == "Mrr"
    print("PASS: display name generation works")


def test_group_expression_fields():
    """Test group with expression fields like (year(base.order_date)) as yr."""
    rl = (POLICY_DIR / "revenue_analytics.river").read_text()
    crl = riverlang_to_compact(rl)

    # Should contain group expressions
    assert "group (year(base.order_date)) as yr" in crl

    ms = compact_to_modelset(crl)
    cr = Compiler(ms).verify()
    assert not cr.has_errors()

    con = create_db(":memory:")
    fqn = FQN(("revenue_analytics", "monthly_revenue"))
    sql = cr.get_sql_generator().generate_sql(fqn, dialect="duckdb")
    rows = con.execute(sql).fetchall()
    assert len(rows) == 4, f"Expected 4 monthly rows, got {len(rows)}"
    con.close()
    print("PASS: group expression fields round-trip correctly")


if __name__ == "__main__":
    tests = [
        test_dot_shorthand,
        test_display_name_generation,
        test_is_riverlang_detection,
        test_annotations_preserved,
        test_if_else_roundtrip,
        test_group_expression_fields,
        test_size_reduction,
        test_roundtrip_all_policies,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"FAIL: {t.__name__}: {e}")
            failed += 1
    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
