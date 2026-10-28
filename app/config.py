"""Settings from environment / .env."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv(BASE_DIR / ".env")

DB_PATH = os.getenv("DB_PATH", str(BASE_DIR / "evals.db"))
EXAMPLES_DIR = BASE_DIR / "examples"
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "60"))

# LLM-as-a-judge (any OpenAI-compatible API). Empty key = judge metric is skipped.
JUDGE_API_KEY = os.getenv("JUDGE_API_KEY", "")
JUDGE_BASE_URL = os.getenv("JUDGE_BASE_URL", "https://api.groq.com/openai/v1")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "llama-3.3-70b-versatile")

# Optional embeddings for semantic_similarity. Empty key = lexical similarity.
EMBED_API_KEY = os.getenv("EMBED_API_KEY", "")
EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "https://api.openai.com/v1")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
