import requests
from bs4 import BeautifulSoup
from typing import Optional

def scrape_url(url: str) -> Optional[str]:
    """Scrape and clean content from a URL using only BeautifulSoup4 and requests."""
    try:
        # RAW request with standard user-agent + BeautifulSoup
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5'
        }
        
        # 10 second timeout to prevent thread hanging
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.extract()
                
            # Extract text
            text = soup.get_text(separator=' ', strip=True)
            
            if text and len(text) > 100:
                return text
        else:
            print(f"BS4 Scraper returned status {response.status_code} for {url}")
            
    except Exception as e:
        print(f"BS4 Scraper failed for {url}: {e}")
        
    return None
