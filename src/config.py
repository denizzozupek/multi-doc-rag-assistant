import os 
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REDIS_URL = os.getenv("REDIS_URL")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")

if not REDIS_URL:
    raise ValueError("REDIS_URL is not set in the environment variables.")


DEFAULT_LLM = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    timeout=30,
    max_retries=3,
)

PDF_PATH = "data/arxiv1.pdf"
PERSIST_DIRECTORY = "data/vector_db"
SQLITE_DB_PATH = "data/graph.db"
BM25_PERSIST_DIRECTORY = "data/bm25_index"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
CHUNK_SIZE = 1000
OVERLAP_SIZE = 200
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

RETRIEVAL_FETCH_K = 20
RETRIEVAL_TO_RERANK_TOP_K = 15
DEFAULT_RERANK_TOP_K = 3
RETRIEVER_SEARCH_TYPE = "hybrid"  # Options: "similarity", "mmr", "hybrid"