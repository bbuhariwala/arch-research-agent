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

async def discover_and_call_tools(context: str) -> str:
    """
    Full MCP flow:
    1. Connect to server
    2. Discover available tools
    3. Pass tools + full context to Claude
    4. Claude decides which tool to call
    5. Execute and return results
    """
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["src/mcp_server.py"],
        env=dict(os.environ)
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Step 1 — discover tools from MCP server
            tools_response = await session.list_tools()
            
            # Step 2 — convert MCP tool definitions to Claude tool format
            claude_tools = []
            for tool in tools_response.tools:
                claude_tools.append({
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                })
            
            print(f"\n  📡 Discovered {len(claude_tools)} tools from MCP server:")
            for t in claude_tools:
                print(f"    - {t['name']}")
            
            # Step 3 — pass full context to Claude with discovered tools
            import anthropic
            client = anthropic.Anthropic()
            
            # context already contains everything — question, clarifications,
            # previous searches, critic feedback. No need to pass question separately.
            messages = [{"role": "user", "content": context}]
            
            all_results = []
            
            # Mini agentic loop — Claude picks and calls tools until done
            while True:
                response = client.messages.create(
                    model="claude-opus-4-6",
                    max_tokens=4096,
                    tools=claude_tools,
                    messages=messages
                )
                
                if response.stop_reason == "end_turn":
                    break
                
                if response.stop_reason == "tool_use":
                    messages.append({
                        "role": "assistant",
                        "content": response.content
                    })
                    
                    tool_results = []
                    for block in response.content:
                        if block.type == "tool_use":
                            print(f"\n  🔧 Claude picked tool: {block.name}")
                            print(f"  Query: {block.input.get('query', '')}")
                            
                            # Execute via MCP
                            result = await session.call_tool(
                                block.name,
                                block.input
                            )
                            
                            result_text = result.content[0].text if result.content else ""
                            all_results.append(result_text)
                            
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result_text
                            })
                    
                    messages.append({
                        "role": "user",
                        "content": tool_results
                    })
            
            return "\n\n".join(all_results)

def run_mcp_research(context: str) -> str:
    """Synchronous wrapper for the full MCP discovery flow."""
    return asyncio.run(discover_and_call_tools(context))
    

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