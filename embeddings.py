# services/rag/embeddings.py
# Only job: Convert queries to embeddings
#           and save to ChromaDB

import os
import json
import logging
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

METADATA_DIR = os.getenv(
    "METADATA_DIR",
    str(Path(__file__).parent.parent.parent / "metadata")
)
CHROMA_DIR = os.getenv(
    "CHROMA_DIR",
    str(Path(__file__).parent.parent.parent / "chroma_db")
)


# ─────────────────────────────────────────
# Embedding Function
# ─────────────────────────────────────────
class LiteLLMEmbeddingFunction(
    embedding_functions.EmbeddingFunction
):
    """Converts text to vector using LiteLLM"""

    def __call__(
        self,
        texts: list[str]
    ) -> list[list[float]]:

        embeddings = []
        for text in texts:
            try:
                import litellm
                response = litellm.embedding(
                    model="text-embedding-ada-002",
                    input=text
                )
                embeddings.append(
                    response.data[0]["embedding"]
                )
                logger.debug(
                    f"Embedded: {text[:50]}..."
                )
            except Exception as e:
                logger.error(
                    f"Embedding failed: {str(e)}"
                )
                embeddings.append([0.0] * 1536)

        return embeddings


# ─────────────────────────────────────────
# Embedding Service
# ─────────────────────────────────────────
class EmbeddingService:
    """
    Only job:
    1. Read metadata JSONs
    2. Convert queries to embeddings
    3. Save to ChromaDB
    """

    def __init__(self, collection):
        self.collection = collection
        logger.info(
            f"EmbeddingService ready: "
            f"{collection.count()} patterns"
        )

    def create_and_save_all(
        self,
        force_reload: bool = False
    ) -> dict:
        """
        Main method.
        Read JSONs -> Embed -> Save ChromaDB.
        Run ONCE or when new features added!
        """
        logger.info("Creating embeddings from JSONs...")

        if not os.path.exists(METADATA_DIR):
            return {
                "status": "error",
                "error":  "Metadata dir not found"
            }

        if force_reload:
            self._clear()

        stats = {
            "loaded":  0,
            "skipped": 0,
            "failed":  0
        }

        json_files = list(
            Path(METADATA_DIR).glob("*.json")
        )
        logger.info(
            f"Found {len(json_files)} JSON files"
        )

        for json_file in json_files:
            table = json_file.stem
            logger.info(f"Processing: {table}")

            try:
                with open(json_file) as f:
                    metadata = json.load(f)

                tm = metadata.get(
                    "table_metadata", {}
                )

                # Existing features
                for feature in tm.get(
                    "existing_features", []
                ):
                    r = self._embed_and_save(
                        feature=feature,
                        table=table,
                        status=feature.get(
                            "status",
                            "IN_PRODUCTION"
                        )
                    )
                    stats[r] += 1

                # Rejected features
                for feature in tm.get(
                    "rejected_features", []
                ):
                    r = self._embed_and_save(
                        feature=feature,
                        table=table,
                        status="REJECTED"
                    )
                    stats[r] += 1

                # Few shot examples
                for example in tm.get(
                    "few_shot_examples", []
                ):
                    r = self._embed_and_save_example(
                        example=example,
                        table=table
                    )
                    stats[r] += 1

                # Query patterns
                for pattern in tm.get(
                    "query_patterns", []
                ):
                    r = self._embed_and_save_pattern(
                        pattern=pattern,
                        table=table
                    )
                    stats[r] += 1

            except json.JSONDecodeError as e:
                logger.error(
                    f"JSON error {table}: {e}"
                )
                stats["failed"] += 1

            except Exception as e:
                logger.error(
                    f"Failed {table}: {str(e)}"
                )
                stats["failed"] += 1

        total = self.collection.count()
        logger.info(
            f"Done! Loaded:{stats['loaded']} "
            f"Skipped:{stats['skipped']} "
            f"Total:{total}"
        )

        return {
            "status":  "success",
            "loaded":  stats["loaded"],
            "skipped": stats["skipped"],
            "failed":  stats["failed"],
            "total":   total
        }

    def save_single(
        self,
        feature_name: str,
        signal_type: str,
        description: str,
        query: str,
        table_name: str,
        status: str = "TESTING"
    ) -> dict:
        """
        Save single new feature.
        Call after new feature approved!
        """
        feature = {
            "feature_name": feature_name,
            "signal_type":  signal_type,
            "description":  description,
            "query":        query
        }
        result = self._embed_and_save(
            feature=feature,
            table=table_name,
            status=status
        )
        return {
            "status": result,
            "total":  self.collection.count()
        }

    def _embed_and_save(
        self,
        feature: dict,
        table: str,
        status: str
    ) -> str:
        """Embed one feature and save to ChromaDB"""

        name = feature.get("feature_name", "")
        if not name:
            return "failed"

        doc_id = f"{table}__{name}"

        if self.collection.get(
            ids=[doc_id]
        )["ids"]:
            return "skipped"

        text = f"""
Feature: {name}
Signal: {feature.get('signal_type', '')}
Description: {feature.get('description', '')}
Query: {feature.get('query', '')}
Notes: {feature.get('notes', '')}
""".strip()

        meta = {
            "feature_name": name,
            "table_name":   table,
            "signal_type":  str(
                feature.get("signal_type", "")
            ),
            "entry_type":   "FEATURE",
            "status":       status,
            "query":        str(
                feature.get("query", "")
            ),
            "description":  str(
                feature.get("description", "")
            ),
            "complexity":   self._detect_complexity(
                feature.get("query", "")
            ),
            "notes":        str(
                feature.get("notes", "")
            )
        }

        try:
            self.collection.add(
                ids=[doc_id],
                documents=[text],
                metadatas=[meta]
            )
            logger.info(f"  Saved: {name}")
            return "loaded"

        except Exception as e:
            logger.error(f"  Failed {name}: {e}")
            return "failed"

    def _embed_and_save_example(
        self,
        example: dict,
        table: str
    ) -> str:
        """Embed few shot example"""

        pattern = example.get("pattern", "")
        if not pattern:
            return "failed"

        doc_id = f"{table}__example__{pattern}"

        if self.collection.get(
            ids=[doc_id]
        )["ids"]:
            return "skipped"

        text = f"""
Pattern: {pattern}
Description: {example.get('description', '')}
Input: {example.get('input', '')}
Query: {example.get('output', '')}
""".strip()

        try:
            self.collection.add(
                ids=[doc_id],
                documents=[text],
                metadatas=[{
                    "feature_name": f"example_{pattern}",
                    "table_name":   table,
                    "signal_type":  pattern,
                    "entry_type":   "EXAMPLE",
                    "status":       "EXAMPLE",
                    "query":        str(
                        example.get("output", "")
                    ),
                    "description":  str(
                        example.get("description", "")
                    ),
                    "complexity":   "SIMPLE",
                    "notes":        ""
                }]
            )
            return "loaded"

        except Exception as e:
            logger.error(f"Example failed: {e}")
            return "failed"

    def _embed_and_save_pattern(
        self,
        pattern: dict,
        table: str
    ) -> str:
        """Embed query pattern"""

        name = pattern.get("pattern_name", "")
        if not name:
            return "failed"

        doc_id = f"{table}__pattern__{name}"

        if self.collection.get(
            ids=[doc_id]
        )["ids"]:
            return "skipped"

        text = f"""
Pattern Name: {name}
Description: {pattern.get('description', '')}
Use when: {pattern.get('use_when', '')}
Template: {pattern.get('template', '')}
""".strip()

        try:
            self.collection.add(
                ids=[doc_id],
                documents=[text],
                metadatas=[{
                    "feature_name": name,
                    "table_name":   table,
                    "signal_type":  "PATTERN",
                    "entry_type":   "PATTERN",
                    "status":       "IN_PRODUCTION",
                    "query":        str(
                        pattern.get("template", "")
                    ),
                    "description":  str(
                        pattern.get("description", "")
                    ),
                    "complexity":   self._detect_complexity(
                        pattern.get("template", "")
                    ),
                    "notes":        str(
                        pattern.get("use_when", "")
                    )
                }]
            )
            return "loaded"

        except Exception as e:
            logger.error(f"Pattern failed: {e}")
            return "failed"

    def _detect_complexity(
        self,
        query: str
    ) -> str:
        """Detect SQL query complexity"""
        if not query:
            return "UNKNOWN"
        q = query.upper()
        if (
            "PERCENTILE" in q or
            "STDDEV" in q or
            q.count("WITH") >= 3
        ):
            return "COMPLEX"
        elif (
            "JOIN" in q or
            "WITH" in q or
            "OVER (" in q
        ):
            return "MEDIUM"
        return "SIMPLE"

    def _clear(self):
        """Clear ChromaDB"""
        logger.info("Clearing ChromaDB...")
        self.collection.delete(
            where={"table_name": {"$ne": ""}}
        )
        logger.info("Cleared!")

    def is_empty(self) -> bool:
        return self.collection.count() == 0

    def count(self) -> int:
        return self.collection.count()
