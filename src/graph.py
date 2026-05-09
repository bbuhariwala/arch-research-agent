from typing import TypedDict,Annotated
import operator
from langgraph.graph import StateGraph, END
from src.retrieval import search_web, chunk_text, embed_chunks, retrieve_relevant_chunks
from src.tools import TOOLS
import anthropic
from dotenv import load_dotenv
import json

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
    Search node: executes a search and adds results to state.
    Uses next_search_query from reasoning node if available,
    otherwise asks Claude what to search for.
    """
    client = anthropic.Anthropic()

    search_count = state.get("search_count", 0)

    # check if reasoning node was executed
    next_query = state.get("next_search_query","")
    if next_query:
        query = next_query
        print(f"\n Search {search_count + 1} (targeted): '{query}'")
    else:
        searches_done = state.get("searches_done", [])
        if searches_done:
            context = f"""You are researching: {state['question']}
                    You have already searched for:
                    {chr(10).join(f'- {s}' for s in searches_done)}
                    What should you search for next to get a more complete picture?
                    Respond with ONLY the search query, nothing else."""
        else:
            context = f"""You are researching: {state['question']}
                    User context:
                    {state.get('clarifications', 'No additional context')}
                    What is the most important thing to search for first given this context?
                    Respond with ONLY the search query, nothing else."""
    
            # Ask Claude what to search for
            response = client.messages.create(
                model="claude-opus-4-6",
                max_tokens=100,
                messages=[{"role": "user", "content": context}]
            )

            query = response.content[0].text.strip()
            print(f"  DEBUG: Raw query from Claude: '{query}' | Length: {len(query)}")
            print(f"\n  🔍 Search {search_count + 1}: '{query}'")
            if not query:
                print(f"  ⚠️ Claude returned empty query, falling back to original question")
                query = state["question"]
    
    # Execute the search
    results, summary = search_web(query)
    #print("***Results***", results)
    #print(f"\n Summary: ", summary)
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
        "search_count": search_count + 1,
        "next_search_query": ""  # Clear it after use
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

    clarifications = state.get("clarifications", "")
    critic_feedback = state.get("critic_feedback", "")
    revision_count = state.get("revision_count", 0)
    revision_section = ""
    if critic_feedback and revision_count > 0:
        revision_section = f"""
        CRITIC FEEDBACK FROM PREVIOUS DRAFT:
        {critic_feedback}
        Address these gaps specifically in this revised analysis."""
    
    context_section = ""
    if clarifications:
        context_section = f"""Additional context from the user:
                            {clarifications}
                            Use this context to make your recommendation specific to their situation."""
    else:
        context_section = "No additional context provided."

    prompt = f"""Based on the following research, provide a structured architecture analysis.
        Question: {state['question']}
        {context_section}

        {revision_section}

        Research collected:
        {all_context}
        Produce a complete structured analysis tailored to the user's specific context.
        If this is a revision, directly address the critic's feedback."""
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}]
    )

    draft = response.content[0].text
    print(f"\n  Synthesis complete ({len(draft)} chars)")

    return {
        **state,
        "current_draft": draft
    }


def reasoning_node(state: ResearchState) -> ResearchState:
    """
    Reasoning node: Claude reads all search results collected so far
    and decides whether it has enough information to synthesize,
    or whether it needs to search for something specific.
    
    This is what makes the agent intelligent instead of mechanical.
    It sets a flag in state that the conditional edge reads.
    """

    client = anthropic.Anthropic()
    searches_summary = ""
    for i,result in enumerate(state["search_results"]):
        searches_summary += f"\nSearch {i+1}: '{result['query']}'\n"
        searches_summary += f"Summary: {result['summary']}\n"
        if result['chunks']:
            searches_summary += f"Sample content: {result['chunks'][0]['text'][:200]}...\n"
    
    prompt = f"""You are researching this architecture question:
        "{state['question']}"
        Here is what you have found so far:
        {searches_summary}

        Searches completed: {state['search_count']}
        Maximum searches allowed: {state['max_searches']}

        Evaluate the research collected and decide:
        1. Do you have enough information to write a thorough architecture analysis?
        2. Are there important gaps or angles not yet covered?

        Respond in this exact JSON format:
        {{
            "has_enough_info": true or false,
            "reasoning": "brief explanation of your decision",
            "next_search_query": "specific query if you need more info, empty string if not"
        }}

        Be honest — if you have good coverage of tradeoffs, performance, and operational 
        concerns, say you have enough. Only search again if there is a specific important gap."""

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=300,
        messages=[{"role":"user","content": prompt}]
    )

    try:
        raw = response.content[0].text.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        decision = json.loads(raw)
    
    except Exception as e:
        print(e)
        print("Could not parse this json, defaulting to synthesize")
        decision = {
            "has_enough_data" : True,
            "reasoning": "Parse error, defaulting to synthesis",
            "next_search_query": ""
        }
    print(f"\n Reasoning: {decision['reasoning']}")
    print(f"\n Has enough info: {decision['has_enough_info']}")
    if not decision["has_enough_info"] and decision.get("next_search_query"):
        print(f"  → Will search for: '{decision['next_search_query']}'")
    
        # Store the decision in state
    return {
        **state,
        "has_enough_info": decision["has_enough_info"],
        "next_search_query": decision['next_search_query']
        
    }

def clarify_node(state: ResearchState) -> ResearchState:
    """
    Clarification node: before researching, ask the user targeted
    questions to make the analysis more specific and useful.
    """
    client = anthropic.Anthropic()
    prompt = f"""You are a senior software architect about to research this question:
    "{state['question']}"

    Before researching, identify 3-4 specific clarifying questions that would 
    significantly change your recommendation. Focus on:
    - Scale and volume requirements
    - Team experience and operational capacity  
    - Existing infrastructure and constraints
    - Specific technical requirements

    Format as a numbered list. Be concise — one line per question."""
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )

    questions = response.content[0].text.strip()
    # Print questions and get user input
    print(f"\n{'='*60}")
    print("Before I research this, a few quick questions:")
    print(f"{'='*60}")
    print(questions)
    print(f"\n{'='*60}")
    print("Your answers (press Enter after each, type 'done' when finished):")
    answers = []
    while True:
        answer = input("> ").strip()
        if answer.lower() == "done" or not answer:
            break
        answers.append(answer)
    clarifications = "\n".join(answers)

    return {
        **state,
        "clarifications": clarifications,
        "asked_clarifications": True
    }

def critic_node(state: ResearchState) -> ResearchState: 
    """
    Critic agent: a second Claude instance that reviews the Researcher's
    draft and identifies gaps, weak reasoning, or missing considerations.
    
    This agent has a completely different system prompt — its job is to
    challenge and critique, not to synthesize.
    
    Key insight: same model, different persona, different job.
    """
    client = anthropic.Anthropic()
    CRITIC_SYSTEM_PROMPT = """You are a principal engineer and architecture reviewer 
    with 20 years of experience. Your job is to critically evaluate architecture 
    recommendations before they go to engineering teams.

    You are skeptical, rigorous, and direct. You look for:
    - Overly generic recommendations that ignore the user's specific context
    - Missing tradeoffs or one-sided analysis
    - Unsupported claims without evidence
    - Important considerations that weren't addressed
    - Recommendations that would fail in production

    You are NOT trying to rewrite the analysis. You are identifying specific gaps
    that require additional research or clarification.

    Be direct and specific. Vague feedback like "could be more detailed" is not useful.
    Point to exactly what is missing and why it matters."""

    clarifications = state.get("clarifications", "No additional context provided")
    revision_count = state.get("revision_count", 0)
    
    prompt = f"""Review this architecture recommendation:
    ORIGINAL QUESTION: {state['question']}

    USER CONTEXT:
    {clarifications}

    DRAFT RECOMMENDATION:
    {state['current_draft']}

    RESEARCH CONDUCTED:
    {len(state['search_results'])} searches: {', '.join(state['searches_done'])}

    Evaluate this recommendation critically. Consider:
    1. Is it specific enough given the user's context?
    2. Are all major tradeoffs covered fairly?
    3. Are there production concerns not addressed?
    4. Are the sources sufficient for this recommendation?
    5. Would you be comfortable if your team shipped based on this?

    Respond in this exact JSON format:
    {{
        "approved": true or false,
        "overall_quality": "strong/adequate/weak",
        "gaps": [
            "specific gap 1",
            "specific gap 2"
        ],
        "required_searches": [
            "specific search query to fill gap 1",
            "specific search query to fill gap 2"
        ],
        "feedback_summary": "2-3 sentence summary of your assessment"
    }}

    Be concise in your JSON. Keep gaps and feedback brief and specific.
    Only approve if you would genuinely be comfortable with this recommendation
    going to an engineering team. Be honest."""
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=2000,
        system=CRITIC_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}]
    )
    try:
        raw = response.content[0].text.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        decision = json.loads(raw)
    
    except Exception as e:
        print(e)
        print(f"Could not parse critic response, defaulting to approve")
        decision = {
            "approved": True,
            "overall_quality": "adequate",
            "gaps": [],
            "required_searches": [],
            "feedback_summary": "Parse error, defaulting to approval"
        }
    
    approved = decision.get("approved", True)
    quality = decision.get("overall_quality", "adequate")
    gaps = decision.get("gaps", [])
    required_searches = decision.get("required_searches", [])
    feedback = decision.get("feedback_summary", "")
    print(f"\n  🔍 Critic Review:")
    print(f"  Quality: {quality}")
    print(f"  Approved: {approved}")
    print(f"  Feedback: {feedback}")
    
    if gaps:
        print(f"  Gaps found:")
        for gap in gaps:
            print(f"    - {gap}")
    
    if required_searches:
        print(f"  Required searches:")
        for s in required_searches:
            print(f"    - {s}")
    
    # If not approved and we have required searches,
    # set the next search query so the search node picks it up
    next_query = ""
    if not approved and required_searches:
        next_query = required_searches[0]  # Address first gap first
    
    return {
        **state,
        "critic_feedback": feedback,
        "critic_approved": approved,
        "revision_count": revision_count + 1,
        "next_search_query": next_query
    }


def should_continue_searching(state: ResearchState) -> str :
    """
    Decides whether to search again or synthesize.
    This function is a conditional edge in the graph.
    Returns the name of the next node to go to.
    """

    search_count = state.get("search_count", 0)
    max_searches = state.get("max_searches", 5)
    has_enough_info = state.get("has_enough_info", False)
    
    if search_count >= max_searches:
        print(f"\n  → Reached max searches ({max_searches}), moving to synthesis")
        return "synthesize"
    if has_enough_info:
        return "synthesize"
    else:
        return "search"


def should_revise(state: ResearchState) -> str:
    """
    Conditional edge after critic node.
    Decides whether to send back for revision or end.
    """
    critic_approved = state.get("critic_approved", True)
    revision_count = state.get("revision_count", 0)
    max_revisions = state.get("max_revisions", 2)
    if revision_count >= max_revisions:
        print(f"\n  → Max revisions reached ({max_revisions}), ending")
        return "end"
    if critic_approved:
        print(f"\n  → Critic approved, ending")
        return "end"
    else:
        print(f"\n  → Critic rejected, sending back for research")
        return "revise"

def print_run_summary(final_state: ResearchState) -> None:
    """
    Print a human-readable summary of what the agent did.
    This is your visibility into the agent's decision making.
    In production this would be logged to an observability system.
    """
    print(f"\n{'='*60}")
    print("AGENT RUN SUMMARY")
    print(f"{'='*60}")
    print(f"Question: {final_state['question']}")
    print(f"Total searches: {final_state['search_count']}")
    print(f"Revision cycles: {final_state['revision_count']}")
    print(f"Critic approved: {final_state['critic_approved']}")
    print(f"Critic feedback: {final_state.get('critic_feedback', 'N/A')}")
    print(f"\nSearch trail:")
    for i, query in enumerate(final_state['searches_done']):
        print(f"  {i+1}. {query}")
    total_chunks = sum(
        len(r['chunks']) for r in final_state['search_results']
    )
    print(f"\nTotal chunks retrieved: {total_chunks}")
    print(f"{'='*60}\n")

def build_research_graph():
    """
    Graph structure:
    
    START → clarify → search → reason → enough? → YES → synthesize → END
                         ↑                → NO  → search ──┘
                         └────────────────────────────────┘
    """
    graph = StateGraph(ResearchState)
    graph.add_node("clarify", clarify_node)
    graph.add_node("search", search_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("reason", reasoning_node)
    graph.add_node("critic", critic_node)

    # Add edges
    # Always start with a search
    graph.set_entry_point("clarify")
    graph.add_edge("clarify", "search")
    graph.add_edge("search", "reason")

    graph.add_conditional_edges(
        "reason",
        should_continue_searching,  
        {
            "search" : "search",
            "synthesize" : "synthesize"
        }
    )

    graph.add_edge("synthesize", "critic")
    graph.add_conditional_edges(
        "critic",
        should_revise,  
        {
            "end" : END,
            "revise" : "search"
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
        "max_searches": 5,
        "has_enough_info": False,
        "next_search_query": "",
        "clarifications": "",
        "asked_clarifications": False,
        "critic_feedback": "",
        "critic_approved": False,
        "revision_count": 0,
        "max_revisions": 2
    }

    # Run the graph
    final_state = app.invoke(initial_state)

    # Print what the agent did
    print_run_summary(final_state)

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
    has_enough_info: bool            #  Reasoning node's decision
    next_search_query: str           # what to search for next if needed
    clarifications: str              # user's answers to clarifying questions
    asked_clarifications: bool       # Have we asked yet?
    critic_feedback: str             # What the critic found
    critic_approved: bool            # Did the critic approve?
    revision_count: int              # How many revision cycles
    max_revisions: int               # Hard limit on revisions