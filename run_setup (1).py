# run_setup.py
# Run ONCE to populate ChromaDB from metadata JSONs
# Or run again when new features added to metadata

import logging
import sys
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.rag import RAGService


if __name__ == "__main__":
    print("=" * 50)
    print("  RAG Setup")
    print("=" * 50)

    rag = RAGService()

    if rag.needs_setup():
        print("\nDB empty - running setup...")
        result = rag.setup(force_reload=False)
    else:
        current = rag.stats()
        print(
            f"\nDB already has "
            f"{current['total']} patterns."
        )
        ans = input("Force reload? (y/n): ").lower()
        result = rag.setup(force_reload=(ans == 'y'))

    print("\nSetup Result:")
    print(f"  Loaded:  {result.get('loaded', 0)}")
    print(f"  Skipped: {result.get('skipped', 0)}")
    print(f"  Failed:  {result.get('failed', 0)}")
    print(f"  Total:   {result.get('total', 0)}")

    print("\nRAG DB Stats:")
    stats = rag.stats()
    print(f"  Total:     {stats['total']}")
    print(f"  By status: {stats.get('by_status', {})}")
    print(f"  By signal: {stats.get('by_signal', {})}")
    print(f"  By type:   {stats.get('by_type', {})}")
    print("\nDone! ChromaDB is ready.")
