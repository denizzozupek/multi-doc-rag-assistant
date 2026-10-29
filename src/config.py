import os
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pathlib import Path

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY is not set in the environment variables.")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")

# MODELS
DEFAULT_LLM = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    timeout=30,
    max_retries=3,
)

CONTEXTUAL_LLM = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    timeout=30,
    max_retries=6,
)

EMBEDDING_MODEL_NAME = "text-embedding-3-small"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# FILE PATHS
PDF_PATH = Path("data/arxiv1.pdf")
LOCAL_CHAT_HISTORY_FILE = Path("data/chat_history/chat_history.json")
PERSIST_DIRECTORY = Path("data/vector_db")
SQLITE_DB_PATH = Path("data/graph.db")
BM25_PERSIST_DIRECTORY = Path("data/bm25_index")

# INGESTION CHUNKING PARAMETERS
CHUNK_SIZE = 1000
OVERLAP_SIZE = 200


# RETRIEVAL PARAMETERS AND BACKWARD COMPATIBILITY
RETRIEVAL_CANDIDATE_K = 20  # Candidate pool for BM25 and dense retrievers
DEFAULT_RERANK_TOP_K = 3  # Last stage re-ranking top K results
RRF_K = 60  # Smoothing constant for Reciprocal Rank Fusion ranking
MMR_FETCH_K = 50  # Only used for MMR retriever, fetch K results from dense retriever before MMR re-ranking
RETRIEVAL_FETCH_K = RETRIEVAL_CANDIDATE_K
HYBRID_CANDIDATE_K = RETRIEVAL_CANDIDATE_K
FINAL_RERANK_TOP_K = DEFAULT_RERANK_TOP_K
RETRIEVER_SEARCH_TYPE = "hybrid"  # Options: "hybrid", "similarity", "mmr", "bm25"


#Async Semaphore for controlling concurrent access to the LLM
CONCURRENT_SEMAPHORE = 10
MAX_RERANK_CONCURRENCY = 2 # Maximum number of concurrent reranking tasks
