import trafilatura
from typing import Optional

def scrape_url(url: str) -> Optional[str]:
    """Scrape and clean content from a URL using trafilatura."""
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded is None:
            return None
        
        # extract text, remove navigation, headers, footers
        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False,
            no_fallback=False
        )
        return text
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None
