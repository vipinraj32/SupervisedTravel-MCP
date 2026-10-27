import os
import asyncio
import certifi
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


client = MultiServerMCPClient(
    {
        "tavily": {
            "transport": "streamable_http",
            "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={TAVILY_API_KEY}",
        }
    }
)


async def get_all_tools():
    tools=await client.get_tools()
    print("Available tools:\n")

    for tool in tools:
        print("tool Name:", tool.name)


tavily_search_tool = None

async def get_tavily_search_tool():
    global tavily_search_tool
    if tavily_search_tool is not None:
        return tavily_search_tool

    tools = await client.get_tools()
    # print("\nAvailable MCP Tools:")

    # for tool in tools:
    #     print(tool.name)

    tavily_search_tool = next(
        tool
        for tool in tools
        if tool.name == "tavily_search"
    )



# This function can be used to call the tavily_search tool with a query in backend.py
async def tavily_mcp_search(query: str):
    await get_tavily_search_tool()
    result = await tavily_search_tool.ainvoke(
        {
            "query": query
        }
    )

    # print("\nMCP Tavily Search Result:\n")
    # print(result)
    return result