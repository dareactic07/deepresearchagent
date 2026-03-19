from ddgs import DDGS
from typing import List
from config.settings import settings

def search_duckduckgo(query: str, max_results: int = None) -> List[str]:
    """Search DuckDuckGo and return top URLs."""
    if max_results is None:
        max_results = settings.TOP_K_RESULTS
        
    try:
        with DDGS() as ddg:
            results = list(ddg.text(query, max_results=max_results))
            return [result['href'] for result in results if 'href' in result]
    except Exception as e:
        print(f"Error searching for {query}: {e}")
        return []
