import os

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://jobs:jobs@localhost:5432/jobs")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dims, matches schema.sql
