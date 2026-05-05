from typing import TypedDict,Annotated
import operator
from langgraph.graph import StateGraph, END
from src.retrieval import search_web, chunk_text, embed_chunks, retrieve_relevant_chunks
from src.tools import TOOLS
import anthropic
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are a senior software architect specializing in analyzing 
architecture decisions for engineering teams.

When searching, be specific and search for different angles:
- Performance benchmarks and comparisons
- Production experiences and war stories  
- Operational complexity and maintenance
- Community sentiment on Hacker News and GitHub

When synthesizing, produce analysis in this format:

## Recommendation
[Clear, direct recommendation]

## Why
[Core reasoning in 2-3 sentences]

## Key Tradeoffs
[Tradeoffs for each option]

## When To Choose Each
[Specific conditions for each choice]

## Operational Considerations
[What teams deal with in production]

## Sources
[Numbered list of all URLs referenced]"""


def search_node(state: ResearchState):
    """
    Search node: asks Claude what to search for next,
    executes the search, and adds results to state.
    
    This node is responsible for one thing only: gathering information.
    """
    client = anthropic.Anthropic()

    searches_done = state.get("searches_done", [])
    search_count = state.get("search_count", 0)

    if searches_done:
        context = f"""You are researching: {state['question']}
                You have already searched for:
                {chr(10).join(f'- {s}' for s in searches_done)}
                What should you search for next to get a more complete picture?
                Respond with ONLY the search query, nothing else."""
    else:
        context = f"""You are researching: {state['question']}
                What is the most important thing to search for first?
                Respond with ONLY the search query, nothing else."""
    
    # Ask Claude what to search for
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=100,
        messages=[{"role": "user", "content": context}]
    )

    query = response.content[0].text.strip()
    print(f"\n  🔍 Search {search_count + 1}: '{query}'")
    # Execute the search
    results, summary = search_web(query)
    # Chunk and embed
    all_chunks = []
    for r in results:
        chunks = chunk_text(r["content"])
        all_chunks.extend(chunks)
    
    embedded = embed_chunks(all_chunks)
    relevant = retrieve_relevant_chunks(query, embedded, top_k=3)

    # Format results
    formatted_results = {
        "query": query,
        "summary": summary,
        "chunks": [{"text": c["text"], "score": c["score"]} for c in relevant]
    }

    current_results = state.get("search_results", [])
    current_searches = state.get("searches_done", [])

    return {
        **state,
        "search_results": current_results + [formatted_results],
        "searches_done": current_searches + [query],
        "search_count": search_count + 1
    }
    
def synthesize_node(state: ResearchState) -> ResearchState:
    """
    Synthesis node: takes all search results collected so far
    and produces a structured architecture analysis.
    This node is responsible for one thing only: producing the final answer.
    """
    client = anthropic.Anthropic()
    all_context = ""
    for i , result in enumerate(state["search_results"]):
        all_context += f"\n--- Search {i+1}: '{result['query']}' ---\n"
        all_context += f"Summary: {result['summary']}\n\n"
        for chunk in result["chunks"]:
            all_context += f"Excerpt (relevance: {chunk['score']:.2f}):\n"
            all_context += f"{chunk['text']}\n\n"
        
    prompt = f"""Based on the following research, provide a structured architecture analysis.
        Question: {state['question']}
        Research collected:
        {all_context}
        Produce a complete structured analysis with your recommendation, tradeoffs, 
        and numbered sources."""
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}]
    )

    draft = response.content[0].text
    print(f"\n  ✓ Synthesis complete ({len(draft)} chars)")

    return {
        **state,
        "current_draft": draft
    }
    
def should_continue_searching(state: ResearchState) -> str :
    """
    Decides whether to search again or synthesize.
    This function is a conditional edge in the graph.
    Returns the name of the next node to go to.
    """

    search_count = state.get("search_count", 0)
    max_searches = state.get("max_searches", 3)
    
    if search_count >= max_searches:
        print(f"\n  → Reached max searches ({max_searches}), moving to synthesis")
        return "synthesize"
    else:
        print(f"\v → Searches done: {search_count} / {max_searches}, searching again")
        return "search"

def build_research_graph():
    """
    Build and compile the research agent graph.

    Graph structure:

    START → search → [should_continue_searching] → search (loop)
                                                    → synthesize → END
    """
    graph = StateGraph(ResearchState)
    graph.add_node("search", search_node)
    graph.add_node("synthesize", synthesize_node)

    # Add edges
    # Always start with a search
    graph.set_entry_point("search")

    graph.add_conditional_edges(
        "search",
        should_continue_searching,  
        {
            "search" : "search",
            "synthesize" : "synthesize"
        }
    )

    # After synthesis, always end
    graph.add_edge("synthesize", END)
    # Compile the graph — this validates your structure
    return graph.compile()

def run_research_agent(question: str) -> str:
    """
    Run the research agent graph for a given question.
    """
    print(f"\n{'='*60}")
    print(f"Research Question: {question}")
    print(f"{'='*60}")

    # Build the graph
    app = build_research_graph()

    initial_state = {
        "question": question,
        "search_results": [],
        "searches_done": [],
        "current_draft": "",
        "search_count": 0,
        "max_searches": 3      # Claude will search 3 times before synthesizing
    }

    # Run the graph
    final_state = app.invoke(initial_state)

    return final_state["current_draft"]

# State is the shared memory that flows through every node
# Every node reads from this and writes back to it
# Think of it as the agent's working memory
class ResearchState(TypedDict):
    question: str                    # The original architecture question
    search_results: list[dict]       # All search results collected so far
    searches_done: list[str]         # Track what queries were already searched
    current_draft: str               # The current synthesis/answer
    search_count: int                # How many searches have been done
    max_searches: int                # Maximum searches allowed (prevents infinite loops)