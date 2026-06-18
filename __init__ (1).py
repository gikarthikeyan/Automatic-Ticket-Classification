# services/rag/__init__.py
# Entry point - ties all three together

import os
import logging
import chromadb
from pathlib import Path
from dotenv import load_dotenv

from services.rag.embeddings import (
    EmbeddingService,
    LiteLLMEmbeddingFunction
)
from services.rag.rag_evaluator import RAGEvaluator
from services.rag.retrieval import RetrievalService

load_dotenv()
logger = logging.getLogger(__name__)

CHROMA_DIR = os.getenv(
    "CHROMA_DIR",
    str(
        Path(__file__).parent.parent.parent
        / "chroma_db"
    )
)


class RAGService:
    """
    Main entry point for RAG.

    Three components:
    1. EmbeddingService  -> JSON -> ChromaDB
    2. RAGEvaluator      -> Is RAG needed?
    3. RetrievalService  -> Match & return context

    Usage:
        rag = RAGService()

        # Once at startup
        if rag.needs_setup():
            rag.setup()

        # Every request
        decision = rag.evaluate(requirement)
        if decision["needed"]:
            result = rag.retrieve(requirement)

        # After approval
        rag.add(feature_name, ...)
    """

    def __init__(self):

        # Shared ChromaDB
        os.makedirs(CHROMA_DIR, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=CHROMA_DIR
        )

        self._collection = (
            self._client.get_or_create_collection(
                name="fe_patterns",
                embedding_function=(
                    LiteLLMEmbeddingFunction()
                ),
                metadata={"hnsw:space": "cosine"}
            )
        )

        # Three services
        self._embeddings = EmbeddingService(
            collection=self._collection
        )
        self._evaluator = RAGEvaluator()
        self._retrieval = RetrievalService(
            collection=self._collection
        )

        logger.info(
            f"RAG Service ready: "
            f"{self._collection.count()} patterns"
        )

    # ─────────────────────────────────────────
    # File 1: Embeddings
    # ─────────────────────────────────────────
    def setup(
        self,
        force_reload: bool = False
    ) -> dict:
        """
        One time: JSON -> Embed -> ChromaDB
        Run once or when new features added!
        """
        return self._embeddings.create_and_save_all(
            force_reload=force_reload
        )

    def add(
        self,
        feature_name: str,
        signal_type: str,
        description: str,
        query: str,
        table_name: str,
        status: str = "TESTING"
    ) -> dict:
        """Add newly approved feature to ChromaDB"""
        return self._embeddings.save_single(
            feature_name=feature_name,
            signal_type=signal_type,
            description=description,
            query=query,
            table_name=table_name,
            status=status
        )

    def needs_setup(self) -> bool:
        """Check if ChromaDB is empty"""
        return self._embeddings.is_empty()

    # ─────────────────────────────────────────
    # File 2: Evaluator
    # ─────────────────────────────────────────
    def evaluate(
        self,
        requirement: str
    ) -> dict:
        """
        Check if RAG is needed.
        Call for EVERY incoming request!

        Returns:
        {
            needed: True/False,
            task_type: what kind of task,
            reason: why,
            confidence: HIGH/MEDIUM/LOW
        }
        """
        return self._evaluator.evaluate(requirement)

    # ─────────────────────────────────────────
    # File 3: Retrieval
    # ─────────────────────────────────────────
    def retrieve(
        self,
        requirement: str,
        top_k: int = 5,
        exclude_rejected: bool = True,
        signal_type: str = None
    ) -> dict:
        """
        Find matching patterns.
        Call ONLY if evaluate() returns needed=True!

        Returns:
        {
            status: success/error/empty,
            patterns: matched patterns,
            context: ready for LLM prompt,
            total: count
        }
        """
        return self._retrieval.search_and_match(
            requirement=requirement,
            top_k=top_k,
            exclude_rejected=exclude_rejected,
            signal_type=signal_type
        )

    # ─────────────────────────────────────────
    # Stats
    # ─────────────────────────────────────────
    def stats(self) -> dict:
        """Get RAG DB statistics"""

        total = self._collection.count()
        if total == 0:
            return {
                "total":   0,
                "message": "Empty - run setup()!"
            }

        all_data = self._collection.get(
            include=["metadatas"]
        )

        statuses    = {}
        signals     = {}
        entry_types = {}

        for meta in all_data["metadatas"]:
            s = meta.get("status", "UNKNOWN")
            statuses[s] = statuses.get(s, 0) + 1

            sig = meta.get("signal_type", "UNKNOWN")
            signals[sig] = signals.get(sig, 0) + 1

            et = meta.get("entry_type", "UNKNOWN")
            entry_types[et] = entry_types.get(et, 0) + 1

        return {
            "total":       total,
            "by_status":   statuses,
            "by_signal":   signals,
            "by_type":     entry_types,
            "chroma_path": CHROMA_DIR
        }
