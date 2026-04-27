from src.agent import ask_claude
from src.retrieval import search_web

query = "Kafka vs RabbitMQ architecture decision"

print(f"Searching for: {query}\n")
results, summary = search_web(query)

print(f"Tavily summary: {summary}\n")
print(f"Found {len(results)} sources:")
for r in results:
    print(f"\n  Title: {r['title']}")
    print(f"  URL: {r['url']}")
    print(f"  Preview: {r['content'][:200]}...")