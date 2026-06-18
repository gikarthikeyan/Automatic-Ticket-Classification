# RAG Service

## Folder Structure
```
rag_service/
├── services/
│   ├── __init__.py
│   └── rag/
│       ├── __init__.py       <- Entry point
│       ├── embeddings.py     <- File 1: JSON -> ChromaDB
│       ├── rag_evaluator.py  <- File 2: Is RAG needed?
│       └── retrieval.py      <- File 3: Match & return
├── run_setup.py              <- Run once!
├── test_rag.py               <- Tests
└── README.md
```

## Install
```
pip install chromadb python-dotenv
```

## .env file
```
METADATA_DIR=./metadata
CHROMA_DIR=./chroma_db
```

## Run Order
```
# Step 1: Setup once
python run_setup.py

# Step 2: Test
python test_rag.py
```

## Usage in MCP Server
```python
from services.rag import RAGService

rag = RAGService()

if rag.needs_setup():
    rag.setup()

# Every request
decision = rag.evaluate(requirement)
if decision["needed"]:
    result = rag.retrieve(requirement)
    context = result["context"]

# After approval
rag.add(feature_name, signal_type, ...)
```

## Each File Does ONE Thing
- embeddings.py   -> Convert JSON queries to vectors, save ChromaDB
- rag_evaluator.py -> Decide if RAG is needed for this request
- retrieval.py    -> Search ChromaDB, return LLM context
- __init__.py     -> Single entry point tying all together
