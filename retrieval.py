# services/rag/retrieval.py
# Only job: Read embeddings from ChromaDB
#           Match against requirement
#           Return context for LLM

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class RetrievalService:
    """
    Only job:
    1. Convert requirement to embedding
    2. Match against ChromaDB vectors
    3. Return matched patterns
    4. Build LLM context
    """

    def __init__(self, collection):
        self.collection = collection

    def search_and_match(
        self,
        requirement: str,
        top_k: int = 5,
        exclude_rejected: bool = True,
        signal_type: Optional[str] = None,
        entry_type: Optional[str] = None
    ) -> dict:
        """
        Main method.
        Search ChromaDB for matching patterns.
        Returns context ready for LLM prompt.

        Args:
            requirement:      What user wants
            top_k:            How many patterns to return
            exclude_rejected: Skip rejected features
            signal_type:      Filter by signal type
            entry_type:       Filter by entry type

        Returns:
            {
                status:   success/error/empty
                patterns: matched patterns
                context:  ready for LLM prompt
                total:    count matched
            }
        """
        logger.info(
            f"Retrieval: '{requirement[:60]}...'"
        )

        # Check DB not empty
        if self.collection.count() == 0:
            logger.warning("ChromaDB empty!")
            return {
                "status":   "empty",
                "patterns": [],
                "context":  "",
                "total":    0,
                "message":  "Run setup first!"
            }

        # Build filter
        where = self._build_filter(
            exclude_rejected=exclude_rejected,
            signal_type=signal_type,
            entry_type=entry_type
        )

        # Query ChromaDB
        try:
            results = self.collection.query(
                query_texts=[requirement],
                n_results=min(
                    top_k,
                    self.collection.count()
                ),
                where=where,
                include=[
                    "metadatas",
                    "distances"
                ]
            )

        except Exception as e:
            logger.error(
                f"ChromaDB query failed: {str(e)}"
            )
            return {
                "status":   "error",
                "patterns": [],
                "context":  "",
                "total":    0,
                "error":    str(e)
            }

        # Process matches
        patterns = self._process_results(results)

        # Build LLM context
        context = self._build_context(
            patterns=patterns,
            requirement=requirement
        )

        logger.info(
            f"Matched {len(patterns)} patterns"
        )

        return {
            "status":   "success",
            "patterns": patterns,
            "context":  context,
            "total":    len(patterns)
        }

    def _build_filter(
        self,
        exclude_rejected: bool,
        signal_type: Optional[str],
        entry_type: Optional[str]
    ) -> Optional[dict]:
        """Build ChromaDB where filter"""

        conditions = []

        if exclude_rejected:
            conditions.append(
                {"status": {"$ne": "REJECTED"}}
            )

        if signal_type:
            conditions.append(
                {"signal_type": {"$eq": signal_type}}
            )

        if entry_type:
            conditions.append(
                {"entry_type": {"$eq": entry_type}}
            )

        if not conditions:
            return None
        elif len(conditions) == 1:
            return conditions[0]
        else:
            return {"$and": conditions}

    def _process_results(
        self,
        results: dict
    ) -> list:
        """Process ChromaDB results into patterns"""

        patterns = []

        for i, (meta, dist) in enumerate(zip(
            results["metadatas"][0],
            results["distances"][0]
        )):
            similarity = round(1 - dist, 3)

            pattern = {
                "rank":         i + 1,
                "feature_name": meta["feature_name"],
                "signal_type":  meta["signal_type"],
                "description":  meta["description"],
                "query":        meta["query"],
                "status":       meta["status"],
                "entry_type":   meta["entry_type"],
                "complexity":   meta["complexity"],
                "similarity":   similarity,
                "notes":        meta.get("notes", "")
            }

            patterns.append(pattern)

            logger.info(
                f"  [{i+1}] "
                f"{meta['feature_name']} | "
                f"sim: {similarity} | "
                f"{meta['status']}"
            )

        return patterns

    def _build_context(
        self,
        patterns: list,
        requirement: str
    ) -> str:
        """
        Build LLM context from matched patterns.
        Groups by status for clarity.
        """

        if not patterns:
            return (
                "No existing patterns found.\n"
                "Use SQL knowledge and metadata\n"
                "to generate query."
            )

        # Group by status
        production = [
            p for p in patterns
            if p["status"] == "IN_PRODUCTION"
        ]
        testing = [
            p for p in patterns
            if p["status"] == "TESTING"
        ]
        examples = [
            p for p in patterns
            if p["status"] == "EXAMPLE"
        ]
        patterns_only = [
            p for p in patterns
            if p["entry_type"] == "PATTERN"
        ]
        rejected = [
            p for p in patterns
            if p["status"] == "REJECTED"
        ]

        context = """
===============================================
PATTERNS FROM KNOWLEDGE BASE
Use as STYLE GUIDE.
Follow patterns but create NEW!
===============================================
"""

        # Production first
        if production:
            context += "\nPRODUCTION (proven):\n"
            for p in production:
                context += f"""
{p['feature_name']} | {p['signal_type']}
Similarity: {p['similarity']}
{p['query']}
{'-' * 30}
"""

        # Testing patterns
        if testing:
            context += "\nTESTING (promising):\n"
            for p in testing:
                context += f"""
{p['feature_name']}
{p['query']}
{'-' * 30}
"""

        # Query patterns
        if patterns_only:
            context += "\nQUERY PATTERNS (templates):\n"
            for p in patterns_only:
                context += f"""
{p['feature_name']}
Use when: {p['notes']}
{p['query']}
{'-' * 30}
"""

        # Examples
        if examples:
            context += "\nEXAMPLES (learn style):\n"
            for p in examples:
                context += f"""
{p['signal_type']}:
{p['query']}
{'-' * 30}
"""

        # Rejected - avoid!
        if rejected:
            context += "\nDO NOT REPEAT:\n"
            for p in rejected:
                context += (
                    f"REJECTED: {p['feature_name']}"
                    f" - {p['notes']}\n"
                )

        context += f"""
===============================================
REQUIREMENT:
{requirement}

Follow style above.
Use metadata for exact columns.
Create something NEW!
===============================================
"""
        return context
