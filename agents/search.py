from tools.tavily_tool import search_tavily
from utils.chunking import chunk_text

def search_node(state: dict) -> dict:
    """
    Called individually for each question via the Send mapping.
    Uses Tavily to search, extract, and chunk content natively.
    """
    question = state.get("question", "")
    results = search_tavily(question)
    
    urls = []
    extracted = []
    
    for r in results:
        url = r.get('url')
        if url:
            urls.append(url)
            text = r.get('raw_content') or r.get('content')
            if text:
                # Chunk the massive 15k-word webpage down to 500-token slices for safe local-LLM injection!
                chunks = chunk_text(text)
                for c in chunks:
                    extracted.append({"url": url, "text": c})
    
    return {
        "urls_per_question": {question: urls},
        "extracted_content": {question: extracted}
    }
