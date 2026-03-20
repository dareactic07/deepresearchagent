import threading

# A global lock to ensure the local Ollama LLM is only invoked by one thread at a time.
# This prevents out-of-memory crashes when LangGraph concurrency maps multiple nodes in parallel.
llm_lock = threading.Lock()
