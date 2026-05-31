# Architecture Research Agent

A multi-agent system that researches software architecture decisions 
using real-world sources from GitHub, Hacker News, and technical blogs.

## What It Does

Given an architecture question like "Should I use Kafka or RabbitMQ?":
1. Asks 3-4 clarifying questions to understand your specific context
2. Autonomously searches relevant technical sources via MCP
3. Uses RAG to retrieve the most semantically relevant content
4. Reasons about whether it has enough information to synthesize
5. Produces a structured analysis with inline citations
6. Has a critic agent review the draft for errors before delivery
7. Iterates based on critic feedback until approved

## Architecture

![Architecture Diagram](docs/architecture.png)

## Tech Stack

- **Claude API** — orchestration, synthesis, and critique
- **LangGraph** — agent graph with typed state management
- **Tavily** — web search across HN, GitHub, technical blogs
- **Voyage AI** — text embeddings (voyage-3 model)
- **MCP** — standard protocol for tool integration
- **RAG pipeline** — chunking, embedding, cosine similarity retrieval

## Key Design Decisions

**Why LangGraph over a while loop**
Adding the critic agent was one new node and two edges — 
no existing code changed. A while loop would have required 
modifying existing logic. LangGraph makes the system extensible.

**Why multi-agent**
A single agent optimized to produce answers isn't optimized 
to critique them. The critic caught a cost estimate off by 
50-100x and a recommendation physically impossible at the 
user's stated scale. Same model, different mandate.

**Why MCP**
Search is exposed as an MCP server — any MCP-compatible 
client can use it without knowing the implementation. 
Claude discovers tools at runtime and decides which to call.

**Why snippet-based RAG**
5 results × ~500 words = 2,500 words chunked into ~15 pieces. 
Top 3 retrieved = ~450 focused words to Claude. Better signal, 
lower cost, faster responses than passing everything.

**Why clarification loop**
Without context the system recommended self-managed Kafka 
to a 2-person team. With context it recommended MSK Serverless 
and flagged operational risk. 4 questions materially changed 
the recommendation.

**Critic temperature at 0.3**
Approval decisions need consistency not creativity. 
Lower temperature produces more reliable judgment calls.

## Setup

1. Clone the repo
2. `python -m venv venv && source venv/bin/activate`
3. `pip install -r requirements.txt`
4. Copy `.env.example` to `.env` and add API keys
5. `python main.py`

## API Keys Required

- `ANTHROPIC_API_KEY` — console.anthropic.com
- `TAVILY_API_KEY` — tavily.com
- `VOYAGE_API_KEY` — dash.voyageai.com

## What I Learned

The most important lesson: LLM systems fail in non-obvious ways. 
Claude will confidently cite benchmarks that don't exist and 
recommend services that can't handle your load. You need 
systematic checks — a critic agent, citation rules, 
defensive guards at every API boundary — not just good prompts.