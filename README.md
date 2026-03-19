# Multi-Agent Deep Researcher 🕵️‍♂️🤖

A fully modular, production-quality multi-agent research assistant built with **LangGraph, LangChain, Ollama (LLaMA 3), HuggingFace embeddings, ChromaDB, and DuckDuckGo search**. 

This system takes a user research topic, automatically decomposes it into high-value sub-questions, parallel-searches the web, extracts fact-based knowledge, evaluates the findings, and synthesizes a comprehensive final markdown report—all while running **100% locally**.

---

## 🌟 Features

- **Multi-Agent Orchestration**: Utilizes LangGraph for stateful, cyclic, and parallel execution.
- **Local AI Engines**: Powered entirely by local `llama3` via Ollama. No paid APIs required!
- **Persistent Long-Term Memory**: High-confidence facts are embedded using `BAAI/bge-small-en` and stored locally in a persistent ChromaDB vector store.
- **Dynamic Configuration**: Easily scales from low-end 4GB VRAM/CPU setups up to ultra-high-end 24GB GPUs via centralized settings.
- **Source & Confidence Tracking**: Every fact presented in the final report maps back to its exact DuckDuckGo source URL alongside a mathematically calculated Confidence Score.

---

## 🏗️ Architecture Workflow

1. **Planner Agent**: Uses structured output to decompose the main topic into a specific set of optimized research questions.
2. **Search Agent**: (Parallel) Executes DuckDuckGo searches for each generated research question.
3. **Scraper Agent**: (Parallel) Fetches raw HTML via Trafilatura, stripping out ads, navigation wrappers, and scripts, then chunks the text down to context-friendly token sizes.
4. **Evaluator Agent**: (Sequential/Parallel) Inspects the scraped chunks. Extracts high-quality, non-redundant facts, assigning scores based on Relevance and Clarity.
5. **Memory Store**: Persists validated high-confidence facts long-term into a local ChromaDB instance.
6. **Synthesizer Agent**: Aggregates the validated truth data to compile a beautifully formatted, exhaustive Markdown report.

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

Execute the main entry script. You can optionally provide your research topic directly over the command line:

```bash
python main.py "The impact of Quantum Computing on RSA Cryptography"
```

The console will stream live updates as the agents progress through the graph nodes, ultimately printing a massive Deep Research Markdown report directly to the terminal!
