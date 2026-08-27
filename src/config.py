import os 
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REDIS_URL = os.getenv("REDIS_URL")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")

if not REDIS_URL:
    raise ValueError("REDIS_URL is not set in the environment variables.")

PDF_PATH = "data/arxiv1.pdf"
PERSIST_DIRECTORY = "data/vector_db"
BM25_PERSIST_DIRECTORY = "data/bm25_index"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
CHUNK_SIZE = 1000
OVERLAP_SIZE = 200
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"