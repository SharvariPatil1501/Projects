import asyncio

from graph import factcheck_graph
from mcp_client import MCPHub


async def run_factcheck(claim: str, on_update=None) -> dict:
    """Start both MCP servers, run the graph, stop the servers. on_update(node, state) is called per stage."""
    state = {"claim": claim}
    async with MCPHub() as hub:
        async for update in factcheck_graph.astream({"claim": claim}, config={"configurable": {"hub": hub}},
                                                    stream_mode="updates"):
            for node, out in update.items():
                state.update(out)
                if on_update:
                    on_update(node, state)
    return state


def run_sync(claim: str, on_update=None) -> dict:
    return asyncio.run(run_factcheck(claim, on_update))
