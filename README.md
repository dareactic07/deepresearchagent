# Deep Research Agent 🕵️‍♂️🤖
### Powered by Groq LPUs & Tavily Search

A production-grade multi-agent research assistant built with **LangGraph, LangChain, Groq (Llama 3.3/70B), Tavily, and Streamlit**. 

This system takes a complex research topic, decomposes it into targeted sub-questions, parallel-searches the live web using Tavily, extracts truth-based facts, evaluates findings, and synthesizes a comprehensive final markdown paper. It features a **Perplexity-style chat interface** for historical exploration and real-time knowledge injection!

---

## 🌟 Key Features

- **Ultra-Fast LLM Inference**: Migrated from local execution to **Groq Cloud LPUs**, providing sub-second reasoning and report generation.
- **High-Fidelity Web Search**: Integrated **Tavily Search API** for clean, AI-ready raw content and scientific-grade discovery.
- **Deep Search Follow-Ups**: Toggle "Deep Search" in chat to trigger autonomous web investigations for follow-up questions.
- **Adaptive Context Fragmentation**: Automatically detects massive web pages and fragments them into searchable windows to survive strict **TPM (Tokens Per Minute)** limits.
- **Persistent Knowledge Base**: All discovered facts are embedded via `BAAI/bge-small-en` and stored in a session-isolated **ChromaDB** vector store.
- **Human-in-the-Loop (HITL)**: The agent pauses for your approval of the research plan before spending search credits!

---

## 🏗️ The Multi-Agent Architecture

1.  **Planner**: Decomposes the topic into $N$ specific, high-entropy research questions. 
2.  **Search Node**: Paralellizes Tavily API calls for every question simultaneously.
3.  **Evaluator**: (The Critic) Inspects raw HTML fragments, assigns confidence scores, and filters out hallucinations/redundancy.
4.  **Memory Store**: Persists validated truths into long-term vector memory.
5.  **Synthesizer**: Compiles the final exhaustive Markdown report with citations and confidence metrics.

---

## 🚀 Installation & Quickstart

### Prerequisites
- Python 3.10+
- **Groq API Key** ([Get one here](https://console.groq.com/))
- **Tavily API Key** ([Get one here](https://tavily.com/))

### Get Started

1.  **Clone & Enter:**
    ```bash
    git clone https://github.com/dareactic07/deepresearchagent.git
    cd deepresearchagent
    ```

2.  **Environment Setup:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate  # Windows
    source venv/bin/activate # Mac/Linux
    pip install -r requirements.txt
    ```

3.  **API Configuration:**
    Create a `.env` file in the root directory:
    ```env
    GROQ_API_KEY=your_groq_key_here
    TAVILY_API_KEY=your_tavily_key_here
    ```

4.  **Launch the UI:**
    ```bash
    python -m streamlit run app.py
    ```

---

## ⚙️ Performance Tuning (`config/settings.py`)

Since the agent uses cloud APIs, the hardware load is minimal, but you can scale the research depth:

-   `MAX_QUESTIONS`: How many broad sub-topics the planner explores (Default: 3)
-   `TOP_K_RESULTS`: How many high-ranking URLs are scraped per question (Default: 3)
-   `MAX_CHUNKS_PER_URL`: How deeply the evaluator reads into each article (Default: 5)
-   `LLM_MODEL`: Set to `llama-3.3-70b-versatile` or `llama3-8b-8192` depending on your account tiers/limits.

---

## 🛡️ Reliability Notes
- **TPM Management**: If you hit "Request too large" errors on low-tier Groq accounts, the agent now automatically fragments text into 300-word windows to stay under 8,000 TPM limits.
- **Structured Output**: Uses explicit `JsonOutputParser` chains to bypass Groq's tool-call handshake bugs for 100% stable formatting.

