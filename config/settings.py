import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load .env file
env_path = os.path.join(os.getcwd(), '.env')
loaded = load_dotenv(env_path)
print(f"DEBUG: Searching for .env at: {env_path}")
print(f"DEBUG: .env loaded successfully: {loaded}")

@dataclass
class Settings:
    # ---------------------------------------------------------
    # System Resource & Performance Settings
    # ---------------------------------------------------------
    # Adjust these values based on your local machine specifications (VRAM/RAM).
    
    # Recommendations:
    # - Low-end (8GB RAM / CPU only): MAX_QUESTIONS=2, TOP_K_RESULTS=2, MAX_CHUNKS_PER_URL=3
    # - Entry-level GPU (16GB RAM / 4GB VRAM GPU): MAX_QUESTIONS=2, TOP_K_RESULTS=3, MAX_CHUNKS_PER_URL=4
    # - Mid-range (16GB RAM / 8GB VRAM GPU): MAX_QUESTIONS=3, TOP_K_RESULTS=3, MAX_CHUNKS_PER_URL=5 (Default)
    # - High-end (32GB+ RAM / 16GB VRAM GPU): MAX_QUESTIONS=5, TOP_K_RESULTS=5, MAX_CHUNKS_PER_URL=15
    # - Ultra-end (64GB+ RAM / 24GB VRAM GPU): MAX_QUESTIONS=8, TOP_K_RESULTS=7, MAX_CHUNKS_PER_URL=30
    
    # Maximum number of research questions the planner will generate
    MAX_QUESTIONS: int = 3
    
    # Maximum number of search results (URLs) to scrape per question
    TOP_K_RESULTS: int = 3
    
    # Maximum number of text chunks to evaluate per URL
    MAX_CHUNKS_PER_URL: int = 5
    
    # ---------------------------------------------------------
    # Core Application Settings
    # ---------------------------------------------------------
    LLM_MODEL: str = "openai/gpt-oss-20b"
    EMBEDDING_MODEL: str = "BAAI/bge-small-en"
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    DB_DIR: str = os.path.join(os.getcwd(), "chroma_db")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

settings = Settings()
