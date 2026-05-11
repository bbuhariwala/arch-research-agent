"""
MCP Client for Architecture Research Agent

Connects to the MCP server and executes tool calls.
This replaces the direct search_web() calls in the agent.

📌 Know This: In a real multi-agent system you might have multiple
MCP servers — one for search, one for GitHub API, one for internal
docs. The client discovers tools from all of them and the LLM
picks which tool to call based on the descriptions.
"""
import asyncio
import os
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def call_mcp_tool(tool_name:str , arguments: dict) -> str:
    """
    Connect to MCP server, call a tool, return the result.
    
    Uses stdio transport — spawns the server as a subprocess,
    sends the tool call, gets the result, closes the connection.
    """
    # Point to your MCP server script
    server_params = StdioServerParameters(
        command=sys.executable,  # Python interpreter
        args=["src/mcp_server.py"],  # Derver script
        env=dict(os.environ)  # Pass environment variables (API keys)
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read,write) as session:
            #Initialize
            await session.initialize()
            # Call the tool
            result = await session.call_tool(tool_name, arguments)
            # Extract text from result
            if result.content:
                return result.content[0].text
            return "No results returned"

def search_via_mcp(query:str, max_results:int = 5) -> str:
    """
    Synchronous wrapper around the async MCP call.
    Makes it easy to use from your existing synchronous agent code.
    """
    return asyncio.run(
        call_mcp_tool(
            "search_architecture_sources",
            {"query": query, "max_results": max_results}
        )
    )

def search_hacker_news_via_mcp(query: str) -> str:
    """
    Search specifically on Hacker News via MCP.
    """
    return asyncio.run(
        call_mcp_tool(
            "search_hacker_news",
            {"query": query}
        )
    )

if __name__ == "__main__":
    # Test the MCP client directly
    print("Testing MCP client...")
    result = search_via_mcp("Kafka vs RabbitMQ throughput")
    print(result[:500])