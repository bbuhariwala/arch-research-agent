import anthropic
import json
from dotenv import load_dotenv
from tools import TOOLS
from retrieval import search_web, chunk_text, embed_chunks, retrieve_relevant_chunks

load_dotenv()

SYSTEM_PROMPT = """You are a senior software architect specializing in analyzing 
architecture decisions. When given a technical question you:

1. Search for current real-world discussions and benchmarks using your search tool
2. Analyze the retrieved information critically  
3. Produce a structured analysis covering:
   - Recommendation (clear and direct)
   - Key tradeoffs
   - When to choose each option
   - Operational considerations
   - cited sources

Always base your analysis on the information you retrieve, not just your training data.
Be direct and opinionated — engineers need clear guidance, not "it depends" without explanation."""


def execute_tool(tool_name:str, tool_input:dict) -> str:
    """
    Execute a tool call requested by Claude and return the result as a string.
    This is the bridge between Claude's decisions and your actual functions.
    """
    if tool_name == "search_web":
        query = tool_input["query"]
        max_results = tool_input.get("max_results", 5)
        print(f"\n  🔍 Searching: '{query}'")
        # Search the web
        results, tavily_summary = search_web(query, max_results)
        all_chunks = []
        source_map = {}  # track which chunk came from which URL

        for r in results:
            chunks = chunk_text(r["content"])
            for chunk in chunks:
                all_chunks.append(chunk)
                source_map[chunk] = r["url"]
        
        embedded_chunks = embed_chunks(all_chunks)

        # Retrieve most relevant chunks for this specific query
        relevant = retrieve_relevant_chunks(query, embedded_chunks, top_k=3)

        # Format results for Claude — clear and structured
        formatted = f"Search results for: '{query}'\n\n"
        formatted += f"Summary: {tavily_summary}\n\n"
        formatted += "Most relevant excerpts:\n\n"

        for i,chunk in enumerate(relevant):
            url = source_map.get(chunk["text"], "Unknown source")
            formatted += f"[Source {i+1}] {url}\n"
            formatted += f"Relevance score: {chunk['score']:.3f}\n"
            formatted += f"{chunk['text']}\n\n"

        print(f"  ✓ Retrieved {len(relevant)} relevant chunks")
        return formatted
    else:
        raise f"Unknown tool: {tool_name}"


def run_research_agent(question: str) -> str:
    """
    Run the full research agent loop for a given architecture question.
    
    The loop:
    1. Send question to Claude with tool definitions
    2. Claude decides to search → we execute the search
    3. Pass search results back to Claude
    4. Claude produces final analysis
    5. Return the analysis
    """
    client = anthropic.Anthropic()
    print(f"\n{'='*60}")
    print(f"Research Question: {question}")
    print(f"{'='*60}")

    # Start with the user's question
    messages = [
        {"role": "user", "content": question}
    ]

    while True:
        print("\nClaude is thinking...")
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages
        )

        print(f"Stop reason: {response.stop_reason}")

          # If Claude is done — no more tool calls — return the final answer
        if response.stop_reason == "end_turn":
            final_answer = ""
            for block in response.content:
                if hasattr(block, "text"):
                    final_answer += block.text
            return final_answer
        
        if response.stop_reason == "tool_use":
            # Add Claude's response to message history
            messages.append({
                "role": "assistant",
                "content": response.content
            })

            # Find all tool calls in this response
            # Claude can request multiple tools in one response
            tool_results = []

            print(response)

            for block in response.content:
                 if block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input
                    tool_use_id = block.id
                
                    # Execute the tool
                    result = execute_tool(tool_name, tool_input)

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": result
                    })
            
            # Pass tool results back to Claude
            messages.append({
                "role": "user",
                "content": tool_results
            })

if __name__ == "__main__":
    answer = run_research_agent(
        "Should I use Kafka or RabbitMQ for a high-throughput event pipeline?"
    )
    print(f"\n{'='*60}")
    print("FINAL ANALYSIS:")
    print('='*60)
    print(answer)
                    
