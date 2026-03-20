# Multi-Agent Deep Researcher (Streamlit Web App) 🕵️‍♂️🤖

A fully modular, production-quality multi-agent research assistant built with **LangGraph, LangChain, Ollama (LLaMA 3), HuggingFace embeddings, ChromaDB, and Streamlit**. 

This system takes a user research topic, automatically decomposes it into high-value sub-questions, parallel-searches the web, extracts fact-based knowledge, evaluates the findings, and synthesizes a comprehensive final markdown report. It also features a sleek, **Perplexity-style web UI** for historical chat tracking!

---

## 🌟 Features

- **Multi-Agent Orchestration**: Utilizes LangGraph for stateful, cyclic, and parallel execution.
- **Sleek Web Interface**: Runs a pure-Python Streamlit UI with sidebar chat-histories, dynamic text auto-resizing, and seamless visual feedback.
- **Persistent Long-Term Memory**: High-confidence facts are embedded using `BAAI/bge-small-en` and stored locally in a persistent, session-isolated ChromaDB vector store.
- **Human-in-the-Loop (HITL)**: The AI pauses before diving into internet scrapes, allowing you to explicitly modify its research Plan via the UI!
- **Deep Search Follow Ups**: Chatting inside an active session utilizes Local RAG context isolation. However, you can toggle **Deep Search** on, which automatically kicks off a DuckDuckGo investigation to find live answers to new questions and inject them directly into your database.
- **Local AI Engines**: Powered entirely by local `llama3` via Ollama. No paid APIs required!

---

## 🏗️ Architecture Workflow

1. **Planner Agent**: Uses structured output to decompose the main topic into a specific set of optimized research questions.
2. **Human Approval Checkpoint**: Streamlit natively pauses Langgraph execution and renders the plan for user approval.
3. **Search Agent**: Executes DuckDuckGo searches for each generated research question.
4. **Scraper Agent**: Fetches raw HTML via Jina Reader or BeautifulSoup, chunking text cleanly.
5. **Evaluator Agent**: Inspects the scraped chunks. Extracts high-quality, non-redundant facts, assigning Float-based Confidence Scores.
6. **Memory Store**: Persists validated high-confidence facts into an isolated ChromaDB collection.
7. **Synthesizer Agent**: Aggregates the validated truth data to compile a beautifully formatted, exhaustive Markdown report.

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/) installed and running locally.

### Get Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/dareactic07/deepresearchagent.git
   cd deepresearchagent
   ```

2. **Set up a virtual environment (Recommended):**
   ```bash
   python -m venv venv
   # On Windows
   .\venv\Scripts\activate
   # On Mac/Linux
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Pull the Local LLM:**
   Make sure Ollama is open/running in the background, and pull the target model:
   ```bash
   ollama pull llama3
   ```

---

## ⚙️ Configuration (System Scaling)

Because this agent runs completely locally, it can be computationally heavy. Open `config/settings.py` to adjust performance parameters perfectly to your hardware!

- `MAX_QUESTIONS`: Defines how many sub-topics the planner explores.
- `TOP_K_RESULTS`: Defines how many DuckDuckGo URLs are scraped per question.
- `MAX_CHUNKS_PER_URL`: Defines how deeply the LLM reads into each article.

**Hardware Recommendations:**
- 🟢 **Entry-level (4GB VRAM)**: `MAX_QUESTIONS=2, TOP_K_RESULTS=3, MAX_CHUNKS=4`
- 🟡 **Mid-range (8GB VRAM)**: `MAX_QUESTIONS=3, TOP_K_RESULTS=3, MAX_CHUNKS=5`
- 🔴 **High-end (16GB+ VRAM)**: `MAX_QUESTIONS=5, TOP_K_RESULTS=5, MAX_CHUNKS=15`

---

## 🖥️ Usage

We have entirely deprecated the legacy terminal interface `main.py` in favor of a gorgeous `Streamlit` Web App UI with integrated SQLite Session routing.

Simply boot up the frontend:
```bash
python -m streamlit run app.py
```
*(The UI will automatically open in your default browser at `localhost:8501`)*
