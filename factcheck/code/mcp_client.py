"""MCP client: starts both servers as subprocesses (stdio) with the SAME Python/venv and talks to them."""
import json
import os
import sys
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from config import ROOT


class MCPHub:
    async def __aenter__(self):
        self._stack = AsyncExitStack()
        self.sessions = {
            "search": await self._start("search_server.py"),
            "verify": await self._start("verify_server.py"),
        }
        return self

    async def _start(self, script: str) -> ClientSession:
        params = StdioServerParameters(command=sys.executable, args=[str(ROOT / script)],
                                       env=dict(os.environ), cwd=str(ROOT))  # pass API keys through
        read, write = await self._stack.enter_async_context(stdio_client(params))
        session = await self._stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        return session

    async def __aexit__(self, *exc):
        await self._stack.aclose()

    async def call(self, server: str, tool: str, args: dict) -> dict:
        res = await self.sessions[server].call_tool(tool, args)
        text = res.content[0].text if res.content else "{}"
        if res.isError:
            raise RuntimeError(f"{tool} failed: {text[:150]}")
        return json.loads(text)

    async def list_tools(self) -> dict:
        return {name: [t.name for t in (await s.list_tools()).tools] for name, s in self.sessions.items()}
