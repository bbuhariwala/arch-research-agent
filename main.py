from src.agent import ask_claude
from src.retrieval import search_web, chunk_text, embed_chunks, retrieve_relevant_chunks

# Step 1: Search
question = "Behram or Buhariwala"
print(f"Question: {question}\n")

print("Step 1: Searching the web...")
results, tavily_summary = search_web(question)
print(f"Found {len(results)} results\n")

# Step 2: Chunk all snippets
print("Step 2: Chunking content...")
all_chunks = []
for r in results:
    chunks = chunk_text(r["content"])
    all_chunks.extend(chunks)
print(f"Created {len(all_chunks)} chunks total\n")

# Step 3: Embed all chunks
print("Step 3: Embedding chunks (this takes a few seconds)...")
embedded_chunks = embed_chunks(all_chunks)
print(f"Embedded {len(embedded_chunks)} chunks\n")

# Step 4: Retrieve most relevant chunks
print("Step 4: Retrieving most relevant chunks...")
relevant = retrieve_relevant_chunks(question, embedded_chunks, top_k=3)

print(f"Top 3 most relevant chunks:\n")
for i, chunk in enumerate(relevant):
    print(f"--- Chunk {i+1} (similarity: {chunk['score']:.3f}) ---")
    print(chunk["text"])
    print()