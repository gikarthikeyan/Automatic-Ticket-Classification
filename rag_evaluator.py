# services/rag/rag_evaluator.py
# Only job: Decide if RAG is needed
#           for incoming request

import logging

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────
# RAG Needed Keywords
# ─────────────────────────────────────────
RAG_NEEDED = [
    # Feature engineering
    "feature",
    "suggest",
    "new feature",
    "create feature",
    "feature engineering",
    "novel",
    "new signal",

    # Signal types
    "recency",
    "frequency",
    "monetary",
    "engagement",
    "propensity",
    "affinity",
    "weighted",
    "score",
    "rate",

    # Complex query
    "combine",
    "complex",
    "optimize",
    "optimized",
    "efficient",
    "bucket 2",
    "bucket 3",
    "improve model",
    "log loss",

    # Domain specific
    "campaign",
    "purchase history",
    "customer lifetime",
    "rolling window",
    "trend",
    "segment"
]

# ─────────────────────────────────────────
# RAG NOT Needed Keywords
# ─────────────────────────────────────────
RAG_NOT_NEEDED = [
    # Simple fetch
    "show me",
    "list all",
    "how many",
    "count records",
    "get records",
    "fetch data",

    # Direct execution
    "execute",
    "run query",
    "run this",

    # Validation only
    "validate",
    "check syntax",
    "is this valid",

    # Simple tools
    "current time",
    "what time",
    "list tables",
    "available tables"
]


class RAGEvaluator:
    """
    Only job:
    Evaluate if RAG retrieval is needed
    for incoming user request.

    Returns:
    -> needed: True/False
    -> reason: why
    -> confidence: HIGH/MEDIUM/LOW
    -> task_type: what kind of request
    """

    def evaluate(
        self,
        user_request: str
    ) -> dict:
        """
        Main method.
        Evaluates if RAG should be triggered.
        """
        request_lower = user_request.lower()

        # Count matches
        rag_matches = [
            kw for kw in RAG_NEEDED
            if kw in request_lower
        ]
        no_rag_matches = [
            kw for kw in RAG_NOT_NEEDED
            if kw in request_lower
        ]

        # Detect task type
        task_type = self._detect_task(request_lower)

        logger.info(
            f"RAG eval | "
            f"Task: {task_type} | "
            f"RAG matches: {rag_matches} | "
            f"No-RAG matches: {no_rag_matches}"
        )

        # Decision
        result = self._decide(
            rag_matches=rag_matches,
            no_rag_matches=no_rag_matches,
            task_type=task_type,
            request=user_request
        )

        self._log_decision(result)
        return result

    def _detect_task(
        self,
        request: str
    ) -> str:
        """Detect what type of task this is"""

        if any(
            kw in request for kw in [
                "feature", "suggest", "novel",
                "new signal", "create feature"
            ]
        ):
            return "FEATURE_ENGINEERING"

        elif any(
            kw in request for kw in [
                "execute", "run query", "run this"
            ]
        ):
            return "EXECUTE_QUERY"

        elif any(
            kw in request for kw in [
                "validate", "check syntax",
                "is this valid"
            ]
        ):
            return "VALIDATE_QUERY"

        elif any(
            kw in request for kw in [
                "current time", "what time"
            ]
        ):
            return "GET_TIME"

        elif any(
            kw in request for kw in [
                "list tables", "available tables"
            ]
        ):
            return "LIST_TABLES"

        elif any(
            kw in request for kw in [
                "generate", "create query",
                "write query", "get", "show",
                "find", "fetch"
            ]
        ):
            return "GENERATE_QUERY"

        return "UNKNOWN"

    def _decide(
        self,
        rag_matches: list,
        no_rag_matches: list,
        task_type: str,
        request: str
    ) -> dict:
        """Make final RAG decision"""

        # Always needs RAG
        if task_type == "FEATURE_ENGINEERING":
            return {
                "needed":     True,
                "task_type":  task_type,
                "reason":     "Feature engineering always needs RAG",
                "confidence": "HIGH"
            }

        # Never needs RAG
        if task_type in [
            "EXECUTE_QUERY",
            "VALIDATE_QUERY",
            "GET_TIME",
            "LIST_TABLES"
        ]:
            return {
                "needed":     False,
                "task_type":  task_type,
                "reason":     f"{task_type} does not need RAG",
                "confidence": "HIGH"
            }

        # Query generation - depends on complexity
        if task_type == "GENERATE_QUERY":

            # Complex query needs RAG
            if len(rag_matches) >= 2:
                return {
                    "needed":     True,
                    "task_type":  task_type,
                    "reason":     f"Complex query: {rag_matches}",
                    "confidence": "HIGH"
                }

            # Some complexity - maybe RAG
            elif len(rag_matches) == 1:
                return {
                    "needed":     True,
                    "task_type":  task_type,
                    "reason":     f"Domain specific: {rag_matches}",
                    "confidence": "MEDIUM"
                }

            # Simple query - no RAG
            elif no_rag_matches:
                return {
                    "needed":     False,
                    "task_type":  task_type,
                    "reason":     "Simple query",
                    "confidence": "HIGH"
                }

            # Unknown - use RAG to be safe
            else:
                return {
                    "needed":     True,
                    "task_type":  task_type,
                    "reason":     "Uncertain - using RAG to be safe",
                    "confidence": "LOW"
                }

        # Unknown task - use RAG to be safe
        return {
            "needed":     True,
            "task_type":  "UNKNOWN",
            "reason":     "Unknown task - using RAG",
            "confidence": "LOW"
        }

    def _log_decision(self, result: dict):
        """Log RAG decision"""
        icon = "RAG ON" if result["needed"] \
               else "RAG OFF"

        logger.info(
            f"{icon} | "
            f"Task: {result['task_type']} | "
            f"Confidence: {result['confidence']} | "
            f"Reason: {result['reason']}"
        )
