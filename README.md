# Architecture Research Agent

A multi-agent system that researches software architecture decisions using 
real-world sources from GitHub, Hacker News, and technical blogs.

## What It Does

Given an architecture question like "Should I use Kafka or RabbitMQ?", 
the agent:
1. Autonomously searches relevant technical sources
2. Retrieves the most semantically relevant content using RAG
3. Dynamically replans — searching again if it identifies gaps
4. Synthesizes a structured analysis with citations

## Architecture

Question → Claude (orchestrator)
↓ decides to search
search_web() → Tavily API (HN, GitHub, blogs)
↓
chunk_text() → overlapping 150-word chunks
↓
embed_chunks() → Voyage AI embeddings
↓
retrieve_relevant_chunks() → cosine similarity
↓
Claude (synthesizer) → structured analysis + citations

## Tech Stack

- **Claude API** — orchestration and synthesis
- **Tavily** — web search across HN, GitHub, technical blogs
- **Voyage AI** — text embeddings (voyage-3 model)
- **RAG pipeline** — chunking, embedding, cosine similarity retrieval
- **Dynamic replanning** — agent searches multiple times based on what it finds

## Setup

1. Clone the repo
2. Create a virtual environment: `python -m venv venv && source venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Copy `.env.example` to `.env` and add your API keys
5. Run: `python main.py`

## API Keys Required

- `ANTHROPIC_API_KEY` — [console.anthropic.com](https://console.anthropic.com)
- `TAVILY_API_KEY` — [tavily.com](https://tavily.com)
- `VOYAGE_API_KEY` — [dash.voyageai.com](https://dash.voyageai.com)

## Example Output

Question: "Should I use Kafka or RabbitMQ for a high-throughput pipeline?"

The agent searches 4-6 times across different angles, retrieves the most 
relevant excerpts, and produces a structured recommendation with tradeoffs, 
operational considerations, and cited sources.

## Design Decisions

- **Snippet-based RAG over full-page extraction** — Tavily snippets (~500 words 
  per result) provide sufficient context without the complexity and cost of full 
  page extraction. Evaluated full extraction via Tavily's extract API but chose 
  snippets as the right tradeoff.
  
- **In-memory vector store** — embeddings are computed per request rather than 
  persisted. Appropriate for a research tool where queries are unique each time. 
  A persistent store like ChromaDB or Pinecone would be better for a fixed 
  document corpus.

- **Dynamic replanning** — Claude decides how many searches to run based on 
  what it finds. This produces more thorough analyses than a fixed number of 
  searches but costs more per query.