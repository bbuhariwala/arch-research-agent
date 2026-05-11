"""
MCP Server for Architecture Research Agent

This server exposes search functionality as MCP tools.
Any MCP-compatible client can connect to this server and use these tools
without knowing anything about the underlying implementation.

Transport: stdio (for local development)
For production: swap to SSE transport for persistent HTTP connections
"""
import asyncio
import os
from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types
from tavily import TavilyClient
from retrieval import search_web, chunk_text, embed_chunks, retrieve_relevant_chunks

load_dotenv()

# Initialize the MCP server
# The name identifies this server to clients
app = Server("arch-research-search")

@app.list_tools()
async def list_tools() -> list[types.Tool]:
    """
    Tell clients what tools this server provides.
    
    This is called by MCP clients when they first connect —
    they ask "what can you do?" and you respond with this list.
    """
    return [types.Tool(
        name="search_architecture_sources",
        description="""Search for architecture decision information from 
            Hacker News, GitHub, and technical blogs. Use this to find 
            real-world experiences, benchmarks, and community opinions about 
            specific technology choices. Be specific in your query — include 
            technology names and the specific aspect you want to research.""",
        inputSchema={
            "type" : "object",
            "properties" : {
                "query" : {
                    "type": "string",
                    "description": "Specific search query including technology names and aspect to research"
                },
                 "max_results": {
                    "type": "integer",
                    "description": "Number of results to return (default 5)",
                    "default": 5
                }
            },
            "required" : ["query"]
        }
    ),
    types.Tool(
        name="search_hacker_news",
        description="""Search specifically for Hacker News discussions about 
        architecture decisions and technology choices. Best for finding 
        practitioner opinions, war stories, and community sentiment.""",
        inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for Hacker News discussions"
                    }
                },
                "required": ["query"]
        }
    )]

@app.call_tool()
async def call_tool(name: str,arguments: dict) ->list[types.TextContent]:
    """
    Execute a tool call from an MCP client.
    
    The client sends: tool name + arguments
    You execute the actual function and return results
    
    Know This: MCP uses JSON-RPC 2.0 as its message format.
    Every tool call is a JSON-RPC request with a method name and params.
    The server responds with a JSON-RPC response containing the result.
    This is why the protocol is transport-agnostic — JSON-RPC works 
    over stdio, HTTP, WebSockets, anything that can carry text.
    """

    if name == "search_architecture_sources":
        query = arguments["query"]
        max_results = arguments.get("max_results", 5)
        results,summary = search_web(query, max_results)

        all_chunks = []
        for result in results:
            chunks = chunk_text(result["content"])
            for chunk in chunks:
                all_chunks.append({
                    "text": chunk,
                    "url": result["url"],
                    "title": result["title"]
                })
        embedded = embed_chunks([c["text"] for c in all_chunks])
        for i,emb  in enumerate(embedded):
            emb["url"] = all_chunks[i]["url"]
            emb["title"] = all_chunks[i]["title"]
        
        relevant = retrieve_relevant_chunks(query, embedded, top_k=3)
        # Format response
        response_text = f"Search results for: '{query}'\n\n"
        response_text += f"Summary: {summary}\n\n"
        response_text += "Most relevant excerpts:\n\n"
        
        for i, chunk in enumerate(relevant):
            response_text += f"[Source {i+1}] {chunk.get('url', 'unknown')}\n"
            response_text += f"Relevance: {chunk['score']:.3f}\n"
            response_text += f"{chunk['text']}\n\n"

        return [types.TextContent(type="text", text=response_text)]
    elif name == "search_hacker_news":
        query = arguments["query"]
        tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        # Search only Hacker News
        results = tavily.search(
            query=query,
            max_results=5,
            include_domains=["news.ycombinator.com"]
        )
        response_text = f"Hacker News discussions for: '{query}'\n\n"
        for r in results.get("results", []):
            response_text += f"Title: {r.get('title')}\n"
            response_text += f"URL: {r.get('url')}\n"
            response_text += f"Content: {r.get('content', '')[:300]}...\n\n"
        return [types.TextContent(type="text", text=response_text)]
    else:
        return [types.TextContent(
            type="text",
            text=f"Unknown tool: {name}"
        )]

async def main():
    """
    Start the MCP server using stdio transport.
    """
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())