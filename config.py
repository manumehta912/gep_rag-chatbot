import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.5-flash-lite")
EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")

PDF_PATH = BASE_DIR / "data" / "GEP-Jun-2026.pdf"
INDEX_DIR = BASE_DIR / "index"

# Printed page number = PDF page - PAGE_OFFSET (Chapter 1 starts on PDF page 23 = report page 1)
PAGE_OFFSET = 22

# Chunking
CHUNK_SIZE = 1200          # characters per chunk
CHUNK_OVERLAP = 150        # characters carried into the next chunk (same section only)
EMBED_DIM = 768

# Embedding throttling (free tier is limited per minute; each text counts as a request)
EMBED_BATCH = 10
EMBED_PAUSE = 12           # seconds between batches (keeps tokens-per-minute low)

# Retrieval
CANDIDATES = 30            # per retriever (dense + BM25) before fusion
FINAL_K = 6                # chunks sent to the LLM (before neighbour expansion)
RERANK_POOL = 15           # how many fused candidates the optional LLM reranker sees
MAX_CONTEXT_CHARS = 14000  # context budget for the answer prompt
