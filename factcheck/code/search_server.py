"""MCP server #1: Search. Exposes free source connectors as tools and stores every retrieved
source by id, readable as the resource source://{id}. Run by the MCP client over stdio."""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mcp.server.fastmcp import FastMCP

import sources as S

mcp = FastMCP("FactCheckSearch")
STORE: dict[str, dict] = {}


async def _run(fn, *args) -> str:
    try:
        found = await asyncio.to_thread(fn, *args)
        for s in found:
            STORE[s["id"]] = s
        return json.dumps({"sources": found, "error": None})
    except Exception as e:  # report, never crash the server
        return json.dumps({"sources": [], "error": f"{type(e).__name__}: {str(e)[:150]}"})


@mcp.tool()
async def search_web(query: str) -> str:
    """Search the general web (DuckDuckGo). Returns JSON {sources, error}."""
    return await _run(S.search_web, query)


@mcp.tool()
async def search_wikipedia(query: str) -> str:
    """Search Wikipedia and return article text. Returns JSON {sources, error}."""
    return await _run(S.search_wikipedia, query)


@mcp.tool()
async def search_arxiv(query: str) -> str:
    """Search arXiv papers (abstracts). Returns JSON {sources, error}."""
    return await _run(S.search_arxiv, query)


@mcp.tool()
async def search_openalex(query: str) -> str:
    """Search OpenAlex scholarly works (abstracts). Returns JSON {sources, error}."""
    return await _run(S.search_openalex, query)


@mcp.tool()
async def search_pubmed(query: str) -> str:
    """Search PubMed biomedical abstracts. Returns JSON {sources, error}."""
    return await _run(S.search_pubmed, query)


@mcp.tool()
async def search_huggingface(query: str) -> str:
    """Find Hugging Face models by name and return their model cards. Returns JSON {sources, error}."""
    return await _run(S.search_huggingface, query)


@mcp.tool()
async def search_factchecks(query: str) -> str:
    """Look up existing professional fact-checks (needs optional GOOGLE_FACTCHECK_API_KEY). Returns JSON {sources, error}."""
    return await _run(S.search_factchecks, query)


@mcp.tool()
async def fetch_page(url: str) -> str:
    """Download a web page and extract its main text. Returns JSON {text, error}."""
    try:
        return json.dumps({"text": await asyncio.to_thread(S.fetch_page, url), "error": None})
    except Exception as e:
        return json.dumps({"text": "", "error": f"{type(e).__name__}: {str(e)[:150]}"})


@mcp.resource("source://{source_id}")
def get_source(source_id: str) -> str:
    """Return the stored text of a previously retrieved source."""
    s = STORE.get(source_id)
    return s["text"] if s else "Source not found."


if __name__ == "__main__":
    mcp.run()
