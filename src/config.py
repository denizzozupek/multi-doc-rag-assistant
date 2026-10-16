import os 
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pathlib import Path

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")

DEFAULT_LLM = ChatOpenAI(
    model="gpt-6-luna",
    temperature=0,
    timeout=30,
    max_retries=3,
)

PDF_PATH = Path("data/arxiv1.pdf")
LOCAL_CHAT_HISTORY_FILE = Path("data/chat_history/chat_history.json")
PERSIST_DIRECTORY = Path("data/vector_db")
SQLITE_DB_PATH = Path("data/graph.db")
BM25_PERSIST_DIRECTORY = Path("data/bm25_index")
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
CHUNK_SIZE = 1000
OVERLAP_SIZE = 200
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

RETRIEVAL_CANDIDATE_K = 20      # Dense ve BM25'in çekeceği aday havuzu
DEFAULT_RERANK_TOP_K = 3        # Reranker'ın LLM'e göndereceği son bağlam sayısı
MMR_FETCH_K = 50                # Yalnızca search_type='mmr' için çeşitlilik havuzu

# Geriye dönük uyumluluk alias'ları
RETRIEVAL_FETCH_K = RETRIEVAL_CANDIDATE_K
HYBRID_CANDIDATE_K = RETRIEVAL_CANDIDATE_K
FINAL_RERANK_TOP_K = DEFAULT_RERANK_TOP_K
RETRIEVER_SEARCH_TYPE = "hybrid"  # Options: "hybrid", "similarity", "mmr", "bm25"