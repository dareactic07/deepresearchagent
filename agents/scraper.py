from tools.scraper_tool import scrape_url
from utils.chunking import chunk_text

def scraper_node(state: dict) -> dict:
    """
    Processes urls for a single question.
    Receives {"question": str, "urls": List[str]}
    """
    question = state.get("question", "")
    urls = state.get("urls", [])
    
    all_chunks = []
    
    for url in urls:
        content = scrape_url(url)
        if content:
            chunks = chunk_text(content)
            for c in chunks:
                all_chunks.append({"text": c, "url": url})
            
    return {"extracted_content": {question: all_chunks}}
