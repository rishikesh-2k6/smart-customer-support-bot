import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from current dir, code/ dir, or project root
load_dotenv()
load_dotenv(Path(__file__).parent / ".env")
load_dotenv(Path(__file__).parent.parent / ".env")

# API — load multiple Groq keys for rotation
GROQ_API_KEYS = []
for i in range(1, 20):
    key = os.environ.get(f"GROQ_API_KEY_{i}", "")
    if key:
        GROQ_API_KEYS.append(key)
# Fallback: single key
if not GROQ_API_KEYS:
    single = os.environ.get("GROQ_API_KEY", "")
    if single:
        GROQ_API_KEYS.append(single)
if not GROQ_API_KEYS:
    print("WARNING: No GROQ_API_KEY found. LLM calls will fail.")

# Model — Groq Llama 3.3 70B (best quality, 6 independent account keys)
MODEL = "llama-3.3-70b-versatile"
MAX_TOKENS = 1000
TEMPERATURE = 0  # deterministic

# Project root = parent of the code/ directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# RAG
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_DB_PATH = str(PROJECT_ROOT / "chroma_db")
TOP_K = 3
CHUNK_SIZE = 400        # tokens per chunk (approximate, in words)
CHUNK_OVERLAP = 80      # word overlap between chunks
SIMILARITY_THRESHOLD = 0.35  # below this = no useful match → escalate

# Paths
DATA_DIR = PROJECT_ROOT / "data"
TICKETS_PATH = PROJECT_ROOT / "support_tickets" / "support_tickets.csv"
OUTPUT_PATH = PROJECT_ROOT / "support_tickets" / "output.csv"

# Logging — required path for submission
LOG_DIR = Path.home() / "hackerrank_orchestrate"
LOG_PATH = LOG_DIR / "log.txt"

# Companies
COMPANIES = ["hackerrank", "claude", "visa"]

# Output field allowed values
VALID_STATUSES = {"replied", "escalated"}
VALID_REQUEST_TYPES = {"product_issue", "feature_request", "bug", "invalid"}
