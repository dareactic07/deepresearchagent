from tavily import TavilyClient
from config.settings import settings
from typing import List, Dict, Any

def search_tavily(query: str, max_results: int = None) -> List[Dict[str, Any]]:
    """
    Search using Tavily API and return results with raw content.
    Returns a list of dicts: [{'url': str, 'content': str, 'raw_content': str}, ...]
    """
    if not settings.TAVILY_API_KEY:
        print("TAVILY_API_KEY not found in settings. Skipping search.")
        return []

    if max_results is None:
        max_results = settings.TOP_K_RESULTS

    try:
        tavily = TavilyClient(api_key=settings.TAVILY_API_KEY)
        # include_raw_content=True allows us to skip the scraper node!
        response = tavily.search(
            query=query, 
            search_depth="advanced", 
            max_results=max_results,
            include_raw_content=True
        )
        return response.get('results', [])
    except Exception as e:
        print(f"Tavily Search failed for '{query}': {e}")
        return []
