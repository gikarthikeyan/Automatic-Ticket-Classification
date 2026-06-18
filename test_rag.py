# test_rag.py
# Full test suite for RAG service

import logging
import sys
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.rag import RAGService


def test_rag():
    print("=" * 60)
    print("  RAG Service Tests")
    print("=" * 60)

    rag = RAGService()
    results = {"passed": 0, "failed": 0}

    # ─── Test 1: Setup ───
    print("\nTest 1: Setup from metadata JSONs")
    print("-" * 40)
    result = rag.setup(force_reload=True)
    print(f"  Status:  {result.get('status')}")
    print(f"  Loaded:  {result.get('loaded', 0)}")
    print(f"  Skipped: {result.get('skipped', 0)}")
    print(f"  Total:   {result.get('total', 0)}")

    if result.get("status") == "success":
        print("PASSED - Setup complete")
        results["passed"] += 1
    else:
        print("FAILED - Check metadata JSONs")
        results["failed"] += 1
        print("Fix metadata JSONs first!")
        return

    # ─── Test 2: Evaluator - FE task ───
    print("\nTest 2: Evaluator - Feature Engineering task")
    print("-" * 40)
    decision = rag.evaluate(
        "Suggest new recency feature for Bucket 2-3"
    )
    print(f"  Needed:     {decision['needed']}")
    print(f"  Task type:  {decision['task_type']}")
    print(f"  Confidence: {decision['confidence']}")
    print(f"  Reason:     {decision['reason']}")

    if decision["needed"]:
        print("PASSED - RAG triggered correctly for FE")
        results["passed"] += 1
    else:
        print("FAILED - Should have triggered RAG")
        results["failed"] += 1

    # ─── Test 3: Evaluator - Execute query ───
    print("\nTest 3: Evaluator - Execute query task")
    print("-" * 40)
    decision = rag.evaluate(
        "Execute this query in BQ"
    )
    print(f"  Needed:     {decision['needed']}")
    print(f"  Task type:  {decision['task_type']}")
    print(f"  Confidence: {decision['confidence']}")

    if not decision["needed"]:
        print("PASSED - RAG skipped for execution")
        results["passed"] += 1
    else:
        print("FAILED - Should skip RAG for execution")
        results["failed"] += 1

    # ─── Test 4: Evaluator - Validate query ───
    print("\nTest 4: Evaluator - Validate query task")
    print("-" * 40)
    decision = rag.evaluate(
        "Validate this SQL query"
    )
    print(f"  Needed:     {decision['needed']}")
    print(f"  Task type:  {decision['task_type']}")

    if not decision["needed"]:
        print("PASSED - RAG skipped for validation")
        results["passed"] += 1
    else:
        print("FAILED - Should skip RAG for validation")
        results["failed"] += 1

    # ─── Test 5: Evaluator - Complex query ───
    print("\nTest 5: Evaluator - Complex query generation")
    print("-" * 40)
    decision = rag.evaluate(
        "Generate optimized query combining "
        "campaign and purchase history for segments"
    )
    print(f"  Needed:     {decision['needed']}")
    print(f"  Task type:  {decision['task_type']}")
    print(f"  Confidence: {decision['confidence']}")

    if decision["needed"]:
        print("PASSED - RAG triggered for complex query")
        results["passed"] += 1
    else:
        print("FAILED - Should trigger RAG for complex query")
        results["failed"] += 1

    # ─── Test 6: Retrieval ───
    print("\nTest 6: Retrieval - Find matching patterns")
    print("-" * 40)
    result = rag.retrieve(
        requirement="recency feature days since last purchase",
        top_k=3
    )
    print(f"  Status: {result['status']}")
    print(f"  Found:  {result['total']} patterns")
    for p in result.get("patterns", []):
        print(
            f"  [{p['rank']}] "
            f"{p['feature_name']} "
            f"(sim: {p['similarity']})"
        )

    if result["status"] == "success":
        print("PASSED - Retrieval works")
        results["passed"] += 1
    else:
        print("FAILED - Retrieval failed")
        results["failed"] += 1

    # ─── Test 7: Context built for LLM ───
    print("\nTest 7: LLM context built correctly")
    print("-" * 40)
    context = result.get("context", "")
    print(f"  Context length: {len(context)} chars")
    print(f"  Has patterns:   {'PATTERNS' in context}")
    print(f"  Has task:       {'REQUIREMENT' in context}")

    if len(context) > 100:
        print("PASSED - Context built")
        results["passed"] += 1
    else:
        print("FAILED - Context empty")
        results["failed"] += 1

    # ─── Test 8: Add new feature ───
    print("\nTest 8: Add new approved feature")
    print("-" * 40)
    result = rag.add(
        feature_name="test_recency_score",
        signal_type="RECENCY",
        description="Test recency feature - days since last purchase",
        query="""
            SELECT customer_id,
            DATE_DIFF(
                CURRENT_DATE(),
                MAX(purchase_date),
                DAY
            ) as days_since_last_purchase
            FROM `project.dataset.purchases`
            WHERE day_string = '20260616'
            GROUP BY customer_id
        """,
        table_name="purchases",
        status="TESTING"
    )
    print(f"  Status: {result['status']}")
    print(f"  Total:  {result['total']}")

    if result["status"] in ["loaded", "skipped"]:
        print("PASSED - Feature added")
        results["passed"] += 1
    else:
        print("FAILED - Could not add feature")
        results["failed"] += 1

    # ─── Test 9: Stats ───
    print("\nTest 9: RAG DB statistics")
    print("-" * 40)
    stats = rag.stats()
    print(f"  Total:     {stats['total']}")
    print(f"  By status: {stats.get('by_status', {})}")
    print(f"  By signal: {stats.get('by_signal', {})}")
    print(f"  By type:   {stats.get('by_type', {})}")

    if stats["total"] > 0:
        print("PASSED - Stats work")
        results["passed"] += 1
    else:
        print("FAILED - Stats empty")
        results["failed"] += 1

    # ─── Test 10: Needs setup check ───
    print("\nTest 10: Needs setup check")
    print("-" * 40)
    needs = rag.needs_setup()
    print(f"  Needs setup: {needs}")

    if not needs:
        print("PASSED - DB is populated")
        results["passed"] += 1
    else:
        print("FAILED - DB shows empty")
        results["failed"] += 1

    # ─── Summary ───
    print("\n" + "=" * 60)
    print("  Test Summary")
    print("=" * 60)
    total = results["passed"] + results["failed"]
    print(f"  Passed: {results['passed']}/{total}")
    print(f"  Failed: {results['failed']}/{total}")

    if results["failed"] == 0:
        print("\nALL TESTS PASSED!")
    else:
        print(f"\n{results['failed']} TESTS FAILED!")
    print("=" * 60)


if __name__ == "__main__":
    test_rag()
