import os
from tavily import TavilyClient
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import anthropic
import voyageai   
import numpy as np
import time
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

def chunk_text(text:str, chunk_size:int = 150, overlap:int = 2) -> list[str]:
    """
    Split text into overlapping chunks of roughly chunk_size words.
    Why overlap? If a key sentence sits at the boundary between two chunks,
    overlap ensures it appears in at least one complete chunk.
    
    Example with chunk_size=5, overlap=2:
    Text: "the cat sat on the mat near the door"
    Chunk 1: "the cat sat on the"
    Chunk 2: "on the mat near the"  <- overlaps by 2 words
    Chunk 3: "near the door
    """

    words = text.split()

    if not words:
        return []
    
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  # Move forward by chunk_size minus the overlap
    
    return chunks

def embed_text(text: str) -> list[float]:
    """
    Convert text into a list of numbers (embedding vector) that 
    captures its semantic meaning using Anthropic's embedding model.
    
    Similar meanings → similar vectors
    Different meanings → different vectors
    
    Returns a list of ~1024 floats.
    """
    """
    Embed a single text using Voyage AI.
    """
    voyage = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
    
    response = voyage.embed(
        texts=[text],
        model="voyage-3"
    )
    
    return response.embeddings[0]

def embed_chunks(chunks: list[str]) -> list[dict]:
    """
    Embed all chunks in a single API call using Voyage's batch support.
    Much more efficient than one call per chunk.
    """
    voyage = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
    
    response = voyage.embed(
        texts=chunks,
        model="voyage-3"
    )
    
    embedded = []
    for chunk, vector in zip(chunks, response.embeddings):
        embedded.append({
            "text": chunk,
            "embedding": vector
        })
    
    return embedded

def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Measure how similar two vectors are.
    
    Returns a score between -1 and 1:
      1.0  = identical meaning
      0.0  = unrelated
     -1.0  = opposite meaning
    
    We compute this by measuring the angle between two vectors.
    Small angle = similar direction = similar meaning.
    """
    a = np.array(vec_a)
    b = np.array(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def retrieve_relevant_chunks(question: str,embedded_chunks: list[dict], top_k: int = 3) -> list[dict]:
    """
    Find the most relevant chunks for a question.
    
    Steps:
    1. Embed the question into a vector
    2. Compare that vector against every chunk's vector
    3. Return the top_k chunks with highest similarity scores
    """
    time.sleep(60)
    embedded_question = embed_text(question)
    scored_chunks = []
    for chunk in embedded_chunks:
        score = cosine_similarity(embedded_question, chunk["embedding"])
        scored_chunks.append({
            "text": chunk["text"],
            "score": score
        })
    scored_chunks.sort(key=lambda x : x["score"], reverse=True)
    return scored_chunks[:top_k]



if __name__ == "__main__":
    # Test chunking
    sample_text = """Kafka is a distributed event streaming platform 
    that excels at high throughput scenarios. It can handle millions of 
    messages per second and is designed for durability and replay. 
    RabbitMQ on the other hand is a traditional message broker that 
    excels at complex routing and lower latency scenarios. It supports 
    multiple messaging protocols and has a simpler operational model 
    for smaller scale deployments. The choice between them depends 
    heavily on your specific requirements around throughput, latency, 
    and operational complexity."""
    chunks = chunk_text(sample_text, chunk_size=30, overlap=5)
    print(f"Chunked text into {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}: {chunk}\n")
    
    print("\nEmbedding first chunk...")
    first_chunk = chunks[0]
    vector = embed_text(first_chunk)

    print(f"Text: {first_chunk[:100]}...")
    print(f"Embedding length: {len(vector)} numbers")
    print(f"First 5 numbers: {vector[:5]}")
    print(f"\nThis is what 'meaning as math' looks like.")