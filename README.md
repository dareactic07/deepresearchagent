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

## 🏗️ How It Works: The Complete Multi-Agent Workflow

![Deep Research Architecture Diagram](Gemini_Generated_Image_9zwzs49zwzs49zwz.png)

The core architecture relies on a **LangGraph** state machine that coordinates specialized LLM nodes. Here is exactly how a research topic flows through the system:

### 1. Topic Initiation & Planning (The Planner)
- **Agent**: `planner_node` (`agents/planner.py`)
- **Action**: When a user inputs a broad topic, the Planner analyzes it and writes a detailed strategic rationale. It then decomposes the topic into a strict JSON array of $N$ high-entropy, highly specific research questions (based on the `MAX_QUESTIONS` config).
- **Goal**: Breadth-first exploration targets.

### 2. Human-in-the-Loop (HITL) 
- **Agent**: `human_approval_node` (`graph/builder.py`)
- **Action**: Execution pauses (interrupt) to present the user with the proposed strategy and sub-questions. The user can either **Approve** the plan (to proceed) or **Modify** it by providing feedback. If modified, the feedback loops back into the Planner to dynamically formulate refined questions.

### 3. Parallel Web Discovery (The Searcher)
- **Agent**: `search_node` (`agents/search.py` + `tools/tavily_tool.py`)
- **Action**: For every approved sub-question, LangGraph spins up a concurrent thread (`Send`). The Searcher uses the **Tavily API** to scour the web, fetching the top URLs (`TOP_K_RESULTS`) and extracting raw HTML/clean text content for each target. 
- **Goal**: Gather massive amounts of unstructured raw data simultaneously.

### 4. Evaluation & Context Fragmentation (The Critic)
- **Agent**: `evaluator_node` (`agents/evaluator.py`)
- **Action**: Unstructured web pages can easily exceed Groq's TPM limits (Request too large). The Critic automatically detects large documents and fragments them into ~300-word sliding windows. 
- Using **ChromaDB's** vector similarity (`rank_texts`), it selects only the most relevant snippets to fit perfectly into the LLM context.
- The LLM then strictly acts as a JSON parser: it reads the chunks, extracts explicitly factual claims, assigns a **Relevance Score** (0.0-1.0) and a **Clarity Score** (0.0-1.0). Only facts surpassing strict thresholds (e.g., score >= 0.6) survive.

### 5. Persistent Knowledge Base (The Memory Store)
- **Agent**: `memory_store_node` (`memory/vector_store.py`)
- **Action**: All surviving, high-confidence facts are cross-referenced and deduplicated. They are then embedded using local `SentenceTransformer` models (`BAAI/bge-small-en` via ChromaDB).
- Each chat receives its own **isolated session prefix**, ensuring answers from one research topic don't pollute the context of another.

### 6. Report Synthesis (The Writer)
- **Agent**: `synthesizer_node` (`agents/synthesizer.py`)
- **Action**: The Synthesizer reads the complete, curated factual memory. Given prompt instructions for maximum verbosity, it compiles an exhaustive Markdown research paper. 
- **Academic Rigor**: It enforces numerical bracketed citations (e.g., `[1]`, `[2]`) next to **every single claim** made, and appends a `References` section mapping the index to the exact Source URLs saved in the facts database.

---

## 💬 Interactive Chat & "Deep Search"

Beyond the static initial report, the UI acts as an interactive exploration tool:
- **Local RAG Memory**: Ask questions, and the agent retrieves context *only* from the vetted ChromaDB facts to prevent hallucinations.
- **Deep Search Toggle**: If turned on, the agent triggers a sub-workflow:
  1. Compiles the last 4 messages to intelligently **Rewrite** your vague query into a standalone Web Search string.
  2. Hits Tavily live and runs the **Critic (Evaluator)** against the brand new chunks.
  3. **Dynamically injects** newly discovered facts directly into the Session's Vector Database.
  4. Finally answers your question using both old and newly ingested knowledge!

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

    ```bash
    python -m venv venv
    .\venv\Scripts\activate  # Windows
    source venv/bin/activate # Mac/Linux
    
    # We use a clean, top-level requirements file
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
-   `LLM_MODEL`: Can be overridden via `.env`. Defaults to `llama-3.3-70b-versatile` but you can set it to `llama3-8b-8192` depending on your Groq tier's Tokens Per Minute limits.

---

## 🛡️ Reliability Notes
- **TPM Management**: If you hit "Request too large" errors on low-tier Groq accounts, the agent now automatically fragments text into 300-word windows to stay under 8,000 TPM limits.
- **Structured Output**: Uses explicit `JsonOutputParser` chains to bypass Groq's tool-call handshake bugs for 100% stable formatting.
