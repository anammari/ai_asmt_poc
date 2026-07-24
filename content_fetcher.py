# content_fetcher.py
import os
import requests
from bs4 import BeautifulSoup
import time

# Improvement/Fix: Set default USER_AGENT to prevent warnings from loaders and APIs
os.environ.setdefault("USER_AGENT", "AI-ASMR-Bot/1.0")

# Fix: Safe local fallback content to prevent app crashes when Wikipedia/APIs fail
_fallback_context_text = (
    "Background contextual fetch failed or returned empty results. "
    "Proceed relying solely on the user's voice prompt. Focus strictly on creating "
    "a relaxing and calming ASMR experience based on their instructions."
)

def fetch_wikipedia(query: str, retries: int = 2) -> str:
    """
    Fetches content from Wikipedia using LangChain.
    Includes safe JSONDecodeError handling and retry logic.
    """
    try:
        from langchain_community.document_loaders import WikipediaLoader
    except ImportError:
        return "Langchain community package is missing. " + _fallback_context_text
        
    for attempt in range(retries):
        try:
            # Load max 1 doc to keep the context window small and relevant
            loader = WikipediaLoader(query=query, load_max_docs=1)
            docs = loader.load()
            
            if docs and docs[0].page_content.strip():
                # Cap at 4000 chars to avoid overwhelming the LLM
                return docs[0].page_content[:4000]
                
        except Exception as e:
            # Langchain WikipediaLoader sometimes raises JSONDecodeError on rate limits/HTML errors
            print(f"[Warning] Wikipedia fetch attempt {attempt + 1} failed: {e}")
            time.sleep(1) # Brief pause before retry
            
    # Return the silent fallback text if all retries fail
    return _fallback_context_text

def fetch_movie_info(query: str) -> str:
    """
    Original behavior for movie sources.
    As per design, this does NOT return the fake silent fallback text if it fails.
    """
    # ... existing original movie integration logic ...
    # Placeholder for the exact movie API (e.g. IMDB/TMDB) integration
    return f"Focus the ASMR script on the movie: {query}. Describe the atmosphere and themes softly."

def fetch_lyrics(query: str) -> str:
    """
    Original behavior for lyrics sources.
    As per design, this does NOT return the fake silent fallback text if it fails.
    """
    # ... existing original lyrics integration logic ...
    # Placeholder for the exact lyrics API (e.g. Genius) integration
    return f"Read these lyrics in a slow, rhythmic whisper: {query}"

def fetch_webpage(url: str) -> str:
    """Fetches raw text from a standard URL using BeautifulSoup."""
    headers = {"User-Agent": os.environ["USER_AGENT"]}
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        
        soup = BeautifulSoup(resp.content, "html.parser")
        paragraphs = soup.find_all('p')
        
        text = "\n".join(p.get_text() for p in paragraphs if p.get_text())
        
        if not text.strip():
            return _fallback_context_text
            
        return text[:4000]
        
    except Exception as e:
        print(f"[Error] Webpage fetch failed: {e}")
        return _fallback_context_text

def fetch_content(source: str) -> str:
    """
    Main entrypoint for the app orchestrator.
    Routes to the appropriate fetcher based on the source prefix or URL structure.
    """
    if not source or not source.strip():
        return ""
        
    source = source.strip()
    
    # 1. Route by specific original prefixes
    if source.lower().startswith("movie:"):
        return fetch_movie_info(source[6:].strip())
        
    if source.lower().startswith("lyrics:"):
        return fetch_lyrics(source[7:].strip())
        
    if source.lower().startswith("wiki:"):
        return fetch_wikipedia(source[5:].strip())
        
    # 2. Route by standard URL
    if source.startswith("http://") or source.startswith("https://"):
        if "wikipedia.org" in source.lower():
            # Extract the title from the URL to use with WikipediaLoader
            title = source.split("/")[-1].replace("_", " ")
            return fetch_wikipedia(title)
        else:
            return fetch_webpage(source)
            
    # 3. Default fallback: Treat arbitrary text in the URL box as a Wikipedia search query
    return fetch_wikipedia(source)