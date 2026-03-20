import trafilatura
import requests
from typing import Optional

def scrape_url(url: str) -> Optional[str]:
    """Scrape and clean content from a URL using Jina Reader with Trafilatura fallback."""
    try:
        # Jina Reader bypasses bot-protections and renders JS to clean Markdown
        response = requests.get(f"https://r.jina.ai/{url}", timeout=15)
        if response.status_code == 200 and len(response.text) > 100:
            return response.text
        else:
            print(f"Jina Reader returned status {response.status_code}. Falling back.")
    except Exception as e:
        print(f"Jina Reader failed: {e}. Falling back.")
        
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=False,
                no_fallback=False
            )
            if text and len(text) > 100:
                return text
    except Exception as e:
        print(f"Trafilatura failed: {e}. Falling back.")
        
    try:
        # Tertiary fallback: RAW request with standard user-agent + BeautifulSoup
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            # Extract paragraphs
            paragraphs = soup.find_all('p')
            text = "\n".join([p.get_text() for p in paragraphs])
            if text and len(text) > 100:
                return text
    except Exception as e:
        print(f"Raw BS4 fallback failed: {e}")
        
    return None
