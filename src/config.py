import os 
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REDIS_URL = os.getenv("REDIS_URL")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")

if not REDIS_URL:
    raise ValueError("REDIS_URL is not set in the environment variables.")

PDF_PATH = "data/time-clocks.pdf"
PERSIST_DIRECTORY = "data/vector_db"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"