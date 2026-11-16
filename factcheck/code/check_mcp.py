"""Run: python check_mcp.py   -- tests both MCP servers start and the free sources respond (no LLM needed)."""
import asyncio

from mcp_client import MCPHub


async def main():
    async with MCPHub() as hub:
        print("Tools:", await hub.list_tools())
        for tool, q in (("search_wikipedia", "Great Wall of China"), ("search_arxiv", "mixture of experts language model"),
                        ("search_openalex", "low-rank adaptation"), ("search_pubmed", "aspirin heart attack prevention"),
                        ("search_web", "Great Wall of China visible from space"), ("search_huggingface", "mixtral")):
            r = await hub.call("search", tool, {"query": q})
            print(f"{tool:20s} sources={len(r['sources'])} error={r['error']}")


asyncio.run(main())
