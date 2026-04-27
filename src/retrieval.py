import os
from tavily import TavilyClient
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup

load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def search_web(query: str, max_results: int = 5) -> list[dict]:
    """
    Search the web for the given query and return top results.

    Args:
        query (str): The search query.
        max_results (int): The maximum number of results to return.

    Returns:
        list[dict]: A list of search results,  Each result has: title, url, content (snippet), score (relevance).
    """
    results = tavily.search(query=query, 
                max_results=max_results,
                include_answer=True,
                include_domains=[
                    "news.ycombinator.com",
                    "github.com",
                    "medium.com",
                    "dev.to",
                    "stackoverflow.com",
                    "martinfowler.com" 
                ])

    cleaned_results = []
    for result in results.get("results", []):
        cleaned_results.append({
            "title" : result.get("title"),
            "url" : result.get("url"),
            "content" : result.get("content"),
            "score" : result.get("score")
        })

    return cleaned_results, results.get("answer", "")

def fetch_content(url:str, max_chars:int = 5000) -> dict: 
    """
    Fetches the full readable content from a URL.
    Returns a dict with url, title, and content.
    max_chars limits content length to avoid overwhelming Claude's context.
    """

    try:
        result = tavily.extract(urls=[url])
        if result and result.get("results"):
            extracted = result["results"][0]
            return {
                "url": url,
                "title": extracted.get("url", url),
                "content": extracted.get("raw_content", "")[:5000]
            }
        else:
            return {
                "url": url,
                "title": "Could not extract",
                "content": "Tavily could not extract content from this URL"
            }
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return {
            "url": url,
            "title": "Error fetching content",
            "content": f"Could not fetch content : {str(e)}"
        }

if __name__ == "__main__":
    results, tavily_summary = search_web("Kafka vs RabbitMQ high throughput")

    print(f"Tavily summary: {tavily_summary}\n")
    print(f"Found {len(results)} results:\n")
    
    for r in results:
        print(f"Title: {r['title']}")
        print(f"URL: {r['url']}")
        print(f"Relevance: {r['score']:.2f}")
        print(f"Content: {r['content'][:300]}...")
        print()