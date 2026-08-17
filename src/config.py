import os 
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")

PDF_PATH = "data/time-clocks.pdf"
PERSIST_DIRECTORY = "data/vector_db"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"