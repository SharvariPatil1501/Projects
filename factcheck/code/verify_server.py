"""MCP server #2: Verification. Tool classify_batch checks one statement against many source texts
using a local NLI model (no cloud calls). The model loads in the background at startup."""
import asyncio
import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mcp.server.fastmcp import FastMCP

import verifier as V

mcp = FastMCP("FactCheckVerify")


def _safe(fn):
    try:
        fn()
    except Exception as e:
        print(f"NLI preload failed: {e}", file=sys.stderr)


@mcp.tool()
async def classify_batch(claim: str, sources_json: str) -> str:
    """Classify each source as supports / contradicts / unrelated for the claim.
    sources_json: JSON list of {uri, text}. Returns JSON {results, error}."""
    try:
        sources = json.loads(sources_json)
        results = await asyncio.to_thread(V.classify_sources, claim, sources)
        return json.dumps({"results": results, "error": None})
    except Exception as e:
        return json.dumps({"results": [], "error": f"{type(e).__name__}: {str(e)[:200]}"})


if __name__ == "__main__":
    threading.Thread(target=lambda: _safe(V.load_nli), daemon=True).start()  # warm up while searching runs
    mcp.run()
