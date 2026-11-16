import streamlit as st

import cache
from judge import VERDICT_TEXT, to_markdown
from pipeline import run_sync
from trust import TIER_NAMES

st.set_page_config(page_title="Fact-Checker", layout="wide")
st.title("Source-Based Fact-Checker")
st.caption("Enter a statement. The app gathers sources from many free places, then reports which sources "
           "support or contradict it. Verdicts describe what the sources say, not absolute truth.")

claim = st.text_area("Statement to check", "The Great Wall of China is visible from space with the naked eye.", height=80)
fresh = st.checkbox("Ignore cache")

STAGE = {
    "analyst": lambda s: f'Planned {len(s["plan"]["parts"])} part(s), {len(s["plan"]["support_queries"]) + len(s["plan"]["contradict_queries"])} queries, sources: {", ".join(s["plan"]["source_types"])}',
    "search": lambda s: f'Collected {len(s["sources"])} sources (+{len(s.get("factchecks", []))} existing fact-checks)',
    "classify": lambda s: "Classified each source as supports / contradicts / unrelated",
    "judge": lambda s: f'Verdict computed by code: {VERDICT_TEXT[s["verdict"]]}',
    "explain": lambda s: "Explanation written",
}
BOX = {"SUPPORTED": st.success, "CONTRADICTED": st.error, "MIXED": st.warning,
       "PARTIALLY_SUPPORTED": st.warning, "NOT_ENOUGH_EVIDENCE": st.info, "ERROR": st.error}


def show_evidence(items, icon):
    if not items:
        st.caption("None found.")
    for e in items[:6]:
        with st.expander(f'{icon} {TIER_NAMES[e["tier"]]} · {e["title"][:80]} · {e["domain"]} · {e["confidence"]:.0%}'):
            st.write(f'"{e["quote"]}"')
            st.markdown(f'[Open source]({e["url"]}) · type: {e["source_type"]} · text used: {e["evidence"]} · method: {e["method"]}')


if st.button("Check statement", type="primary") and claim.strip():
    cached = None if fresh else cache.get(claim)
    if cached:
        st.session_state["fc"] = cached
        st.toast("Loaded from cache")
    else:
        with st.status("Starting MCP servers and checking...", expanded=True) as status:
            status.write("Started Search + Verification MCP servers (first run downloads the NLI model)")
            res = run_sync(claim.strip(), lambda node, s: status.write(STAGE[node](s)))
            status.update(label="Done", state="complete", expanded=False)
        if res.get("sources"):
            cache.put(claim, res)
        st.session_state["fc"] = res

r = st.session_state.get("fc")
if r:
    BOX[r["verdict"]](f'**{VERDICT_TEXT[r["verdict"]]}**')
    if r["verdict"] == "ERROR":
        st.error(f'The verification server failed: {r.get("verifier_error")}')
    st.write(r.get("explanation", ""))

    counts = {}
    for s in r.get("sources", []):
        counts[s["source_type"]] = counts.get(s["source_type"], 0) + 1
    st.caption("Sources searched: " + (", ".join(f"{k}: {v}" for k, v in sorted(counts.items())) or "none"))

    for i, p in enumerate(r.get("parts", []), 1):
        st.subheader(f'Part {i}: {p["text"]}')
        BOX[p["verdict"]](VERDICT_TEXT[p["verdict"]] + (f' ({p["note"]})' if p.get("note") else ""))
        a, b = st.columns(2)
        with a:
            st.markdown(f'**Evidence for** (weight {p["support_weight"]})')
            show_evidence(p["supports"], "✅")
        with b:
            st.markdown(f'**Evidence against** (weight {p["contradict_weight"]})')
            show_evidence(p["contradicts"], "❌")
        st.caption(f'{p["unrelated"]} other sources were checked and judged unrelated or neutral.')

    if r.get("factchecks"):
        st.subheader("Existing professional fact-checks")
        for f in r["factchecks"]:
            st.markdown(f'- [{f["title"]}]({f["url"]}) — {f["text"]}')
    if r.get("debug"):
        with st.expander(f' {len(r["debug"])} pipeline warnings'):
            for d in r["debug"]:
                st.write(f"- {d}")
    st.download_button("Download report (.md)", to_markdown(r), file_name="fact_check_report.md", mime="text/markdown")
