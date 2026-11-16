"""LangGraph pipeline (async):  analyst -> search -> classify -> judge -> explain
LLM is used twice (plan, explanation). Evidence gathering and verification run through MCP servers."""
import asyncio
import json
from typing import Dict, List, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

from config import CLASSIFY_PER_PART, FETCH_FULL_TOP, MAX_PARTS, MAX_SOURCES, MAX_TEXT_CHARS
from judge import VERDICT_TEXT, judge_part, overall_verdict
from llm import generate, parse_json
from textutil import overlap
from trust import TIER_NAMES

SOURCE_TOOLS = {"web": ["search_web"], "wikipedia": ["search_wikipedia"],
                "scholarly": ["search_arxiv", "search_openalex"],
                "pubmed": ["search_pubmed"], "huggingface": ["search_huggingface"]}


class FactState(TypedDict, total=False):
    claim: str
    plan: dict
    sources: List[dict]
    factchecks: List[dict]
    parts: List[dict]
    verdict: str
    explanation: str
    verifier_error: str
    debug: List[str]


PLAN_PROMPT = """You are a fact-checking analyst. Do NOT judge whether the statement is true. Return ONLY JSON:
{"parts": [{"text": "<one self-contained checkable assertion>",
            "support_query": "<web search query likely to find evidence that this part is TRUE>",
            "contradict_query": "<web search query likely to find evidence that this part is FALSE or different>"}],
 "entities": ["<named models, people, organisations or products>"],
 "source_types": ["web", "wikipedia", "scholarly", "pubmed", "huggingface"]}
Rules:
- First read the statement and decide how many separate factual assertions it contains. Make exactly one part per assertion.
- If it is a single fact, return ONE part. Never split one fact into pieces and never merge unrelated facts into one part.
- Make every part self-contained: repeat names instead of using "it" or "they".
- entities: at most 3. source_types: always include "web" and "wikipedia"; add "scholarly" for science, technology or research;
  add "pubmed" only for health or biology; add "huggingface" only for claims about AI models.
Statement: <<CLAIM>>"""

EXPLAIN_PROMPT = """Write a 3-4 sentence plain-language explanation of this fact-check result.
Use ONLY the evidence below. Do not add outside knowledge. Do not change the verdict. If evidence is missing or weak, say so.
Statement: <<CLAIM>>
Overall verdict: <<VERDICT>>
Evidence by part:
<<EVIDENCE>>"""


def _hub(config: RunnableConfig):
    return config["configurable"]["hub"]


def _relevance(text: str, s: dict) -> float:
    return overlap(text, s["title"] + " " + s["text"]) + 0.15 * (3 - s["tier"])


def _uniq(items):
    return list(dict.fromkeys(items))


async def analyst_node(state: FactState, config: RunnableConfig):
    claim, debug = state["claim"], list(state.get("debug", []))
    plan = {}
    try:
        plan = parse_json(await asyncio.to_thread(generate, PLAN_PROMPT.replace("<<CLAIM>>", claim), True))
    except Exception as e:  # search still works without the LLM
        debug.append(f"planner LLM failed, using a basic plan ({type(e).__name__})")
    parts, support_q, contra_q = [], [], []
    for p in plan.get("parts", []) if isinstance(plan, dict) else []:
        if isinstance(p, str):
            p = {"text": p}
        if isinstance(p, dict) and str(p.get("text", "")).strip():
            t = p["text"].strip()
            parts.append(t)
            support_q.append(str(p.get("support_query") or t))
            contra_q.append(str(p.get("contradict_query") or f"{t} false OR incorrect OR debunked"))
    if len(parts) > MAX_PARTS:
        debug.append(f"statement has {len(parts)} parts; only the first {MAX_PARTS} were checked (raise MAX_PARTS in config.py)")
        parts, support_q, contra_q = parts[:MAX_PARTS], support_q[:MAX_PARTS], contra_q[:MAX_PARTS]
    if not parts:
        parts, support_q, contra_q = [claim], [claim], [f"{claim} false OR incorrect OR debunked"]
    types = [t for t in plan.get("source_types", []) if t in SOURCE_TOOLS] if isinstance(plan, dict) else []
    types = _uniq(["web", "wikipedia", *types]) if plan else ["web", "wikipedia", "scholarly"]
    entities = [e for e in plan.get("entities", []) if isinstance(e, str)][:3] if isinstance(plan, dict) else []
    return {"plan": {"parts": parts, "entities": entities, "support_queries": support_q,
                     "contradict_queries": contra_q, "source_types": types},
            "debug": debug}


async def search_node(state: FactState, config: RunnableConfig):
    hub, plan, debug = _hub(config), state["plan"], list(state.get("debug", []))
    sem = asyncio.Semaphore(4)

    async def call(tool, query):
        async with sem:
            try:
                r = await hub.call("search", tool, {"query": query})
                return {"sources": r["sources"], "error": f'{tool}: {r["error"]}' if r.get("error") else None}
            except Exception as e:
                return {"sources": [], "error": f"{tool}: {e}"}

    jobs = []
    for q in plan["support_queries"] + plan["contradict_queries"]:
        for t in plan["source_types"]:
            if t != "huggingface":
                jobs += [call(tool, q) for tool in SOURCE_TOOLS[t]]
    if "huggingface" in plan["source_types"]:
        jobs += [call("search_huggingface", e) for e in plan["entities"]]
    jobs.append(call("search_factchecks", state["claim"]))

    found, checks, seen = [], [], set()
    for r in await asyncio.gather(*jobs):
        if r["error"]:
            debug.append(r["error"])
        for s in r["sources"]:
            key = s["url"].split("#")[0].rstrip("/").lower()
            if key not in seen:
                seen.add(key)
                (checks if s["source_type"] == "factcheck" else found).append(s)

    found.sort(key=lambda s: -_relevance(state["claim"], s))
    found = found[:MAX_SOURCES]

    # upgrade the best web snippets to full page text
    targets = [s for s in found if s["source_type"] == "web"][:FETCH_FULL_TOP]

    async def fetch(s):
        async with sem:
            try:
                return (await hub.call("search", "fetch_page", {"url": s["url"]})).get("text", "")
            except Exception:
                return ""

    for s, page in zip(targets, await asyncio.gather(*[fetch(s) for s in targets])):
        if page:
            s["text"] = f'{s["text"]}\n\n{page}'[:MAX_TEXT_CHARS]
            s["evidence"] = "full_page"
    return {"sources": found, "factchecks": checks, "debug": _uniq(debug)}


async def classify_node(state: FactState, config: RunnableConfig):
    hub, parts, verifier_error = _hub(config), [], ""
    for part in state["plan"]["parts"]:
        cands = sorted(state["sources"], key=lambda s: -_relevance(part, s))[:CLASSIFY_PER_PART]
        labels = {}
        if cands:
            try:
                payload = json.dumps([{"uri": s["uri"], "text": s["text"]} for s in cands])
                res = await hub.call("verify", "classify_batch", {"claim": part, "sources_json": payload})
                if res.get("error"):
                    raise RuntimeError(res["error"])
                labels = {r["uri"]: r for r in res["results"]}
            except Exception as e:
                verifier_error = str(e)[:200]
        evidence = []
        for s in cands:
            r = labels.get(s["uri"])
            if r:
                evidence.append({**{k: s[k] for k in ("uri", "url", "title", "domain", "tier", "source_type", "evidence")},
                                 **{k: r[k] for k in ("label", "confidence", "quote", "reason", "method")}})
        parts.append({"text": part, "evidence": evidence})
    return {"parts": parts, "verifier_error": verifier_error}


def judge_node(state: FactState):
    parts = [{**p, **judge_part(p["evidence"])} for p in state["parts"]]
    verdict = "ERROR" if state.get("verifier_error") else overall_verdict([p["verdict"] for p in parts])
    return {"parts": parts, "verdict": verdict}


async def explain_node(state: FactState):
    ev = []
    for p in state["parts"]:
        ev.append(f'- Part: {p["text"]} -> {p["verdict"]}')
        for kind in ("supports", "contradicts"):
            for e in p[kind][:3]:
                ev.append(f'   {kind}: [{TIER_NAMES[e["tier"]]}] {e["title"][:70]}: "{e["quote"][:200]}"')
        if not p["supports"] and not p["contradicts"]:
            ev.append("   no decisive evidence found")
    prompt = (EXPLAIN_PROMPT.replace("<<CLAIM>>", state["claim"]).replace("<<VERDICT>>", VERDICT_TEXT[state["verdict"]])
              .replace("<<EVIDENCE>>", "\n".join(ev)))
    try:
        text = (await asyncio.to_thread(generate, prompt)).strip()
    except Exception:
        text = f'Result: {VERDICT_TEXT[state["verdict"]]}. See the evidence lists below for the sources and quotes behind each part.'
    return {"explanation": text}


g = StateGraph(FactState)
for name, fn in (("analyst", analyst_node), ("search", search_node), ("classify", classify_node),
                 ("judge", judge_node), ("explain", explain_node)):
    g.add_node(name, fn)
g.set_entry_point("analyst")
for a, b in (("analyst", "search"), ("search", "classify"), ("classify", "judge"), ("judge", "explain")):
    g.add_edge(a, b)
g.add_edge("explain", END)
factcheck_graph = g.compile()
