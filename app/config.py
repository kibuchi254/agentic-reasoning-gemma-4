import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "ollama")

MODEL = os.getenv("MODEL", "gemma4:latest")
CONTEXT_WINDOW = int(os.getenv("CONTEXT_WINDOW", "8192"))
MAX_PROMPT_TOKENS = int(os.getenv("MAX_PROMPT_TOKENS", "6144"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "2048"))

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

AGENT_MAX_ITERATIONS = int(os.getenv("AGENT_MAX_ITERATIONS", "10"))
AGENT_TEMPERATURE = float(os.getenv("AGENT_TEMPERATURE", "0.4"))
