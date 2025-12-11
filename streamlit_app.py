# streamlit_app.py
import streamlit as st
import pandas as pd
from src.memory import VectorMemory
from src.agents import ResearchAgent, AnalysisAgent, MemoryAgent, Coordinator
from src.main import MOCK_KB

st.set_page_config(page_title="Multi-Agent Chat System", layout="wide")
st.title("Simple Multi-Agent Chat System — Prototype")

# --- Initialize singletons in session_state ---
if "vm" not in st.session_state:
    st.session_state.vm = VectorMemory()
if "research" not in st.session_state:
    st.session_state.research = ResearchAgent(MOCK_KB)
if "analysis" not in st.session_state:
    st.session_state.analysis = AnalysisAgent()
if "memory_agent" not in st.session_state:
    st.session_state.memory_agent = MemoryAgent(st.session_state.vm)
if "coord" not in st.session_state:
    st.session_state.coord = Coordinator(st.session_state.research,
                                         st.session_state.analysis,
                                         st.session_state.memory_agent)

coord: Coordinator = st.session_state.coord

# --- Layout ---
col_left, col_right = st.columns((2, 1))

with col_left:
    st.subheader("Ask a question")
    query = st.text_input("Enter your question here", value="What are the main types of neural networks?")
    update_memory = st.checkbox("Store synthesis to memory (if produced)", value=False)
    run_btn = st.button("Run Query")

    if run_btn and query.strip():
        # run and optionally store
        res = coord.handle_query(query)
        # if user requested explicit memory store, store the synthesis
        if update_memory and res.get("results", {}).get("synthesis"):
            synth = res["results"]["synthesis"]
            coord.memory.store_finding(topic=query, text=synth.get("text",""), source="ui", agent="Coordinator", confidence=synth.get("confidence", 0.5))
            st.success("Synthesis stored to memory.")
        st.success("Query processed — see results below.")
        st.session_state._last_run = res

    st.markdown("### Latest Result")
    if "_last_run" in st.session_state:
        res = st.session_state._last_run
        # show synthesis prominently
        synth = res.get("results", {}).get("synthesis", {})
        if synth:
            st.markdown("**Synthesis:**")
            st.info(synth.get("text", "No synthesis produced."))
            st.write(f"Confidence: {synth.get('confidence', 0.0)}")
            st.button("Copy synthesis to clipboard", key="copy_synth")
        # research results table
        research = res.get("results", {}).get("research", [])
        if research:
            st.markdown("**Research results**")
            df = pd.DataFrame([{
                "title": r.get("title"),
                "excerpt": (r.get("text")[:200] + "..." if len(r.get("text",""))>200 else r.get("text","")),
                "tags": ", ".join(r.get("tags", [])),
                "source": r.get("source"),
                "confidence": r.get("confidence")
            } for r in research])
            st.dataframe(df)
            # expand each result
            for i, r in enumerate(research):
                with st.expander(f"Result {i+1}: {r.get('title')} ({r.get('confidence')})"):
                    st.write(r.get("text"))
                    st.write("tags:", r.get("tags"))
                    st.write("source:", r.get("source"))
        # analysis results
        analysis = res.get("results", {}).get("analysis", {})
        if analysis:
            st.markdown("**Analysis (ranked)**")
            ranked = analysis.get("ranked", [])
            if ranked:
                df2 = pd.DataFrame(ranked)
                st.table(df2[["title", "score", "confidence", "explanation"]])
            st.write("Summary:", analysis.get("summary_text"))
            st.write("Overall confidence:", analysis.get("confidence"))

    else:
        st.info("No query run yet. Enter a question and press 'Run Query'.")

    st.markdown("---")
    st.markdown("### Conversation Context (last 6 messages)")
    for msg in coord.context[-6:]:
        role = msg.get("role", "system")
        st.write(f"**{role}** @ {msg.get('ts')}: {msg.get('text')}")

with col_right:
    st.subheader("Trace Log")
    traces = coord.trace[-15:]
    if traces:
        for t in traces[::-1]:
            ts = t.get("ts")
            actor = t.get("actor")
            action = t.get("action")
            with st.expander(f"{ts} | {actor} | {action}", expanded=False):
                payload = t.get("payload", {})
                st.json(payload)
    else:
        st.info("No trace entries yet.")

    st.markdown("---")
    st.subheader("Memory Inspector")
    mem_q = st.text_input("Search memory (keyword or free text)", value="")
    col_mem1, col_mem2 = st.columns(2)
    with col_mem1:
        if st.button("Keyword search memory"):
            if mem_q.strip():
                kw = coord.memory.retrieve_by_topic(mem_q, top_k=10)
                if not kw:
                    st.warning("No keyword matches.")
                else:
                    for r in kw:
                        st.write(f"- [{r.id}] {r.topic} (by {r.agent}) @ {r.timestamp}")
                        st.write(r.text[:300] + ("..." if len(r.text)>300 else ""))
                        st.write("confidence:", r.confidence, "metadata:", r.metadata)
            else:
                st.info("Type a search term first.")
    with col_mem2:
        if st.button("Vector search memory"):
            if mem_q.strip():
                vec = coord.memory.retrieve_similar(mem_q, top_k=10)
                if not vec:
                    st.warning("No vector-similar records.")
                else:
                    for rec, score in vec:
                        st.write(f"- [{rec.id}] {rec.topic} (score={score:.3f}) - {rec.agent} @ {rec.timestamp}")
                        st.write(rec.text[:250] + ("..." if len(rec.text)>250 else ""))
            else:
                st.info("Type a search term first.")

    st.markdown("---")
    st.subheader("Memory Stats")
    all_recs = coord.memory.vm.get_all()
    st.write(f"Total records: {len(all_recs)}")
    if len(all_recs) > 0:
        dfm = pd.DataFrame([{
            "id": r.id, "topic": r.topic, "agent": r.agent, "confidence": r.confidence, "ts": r.timestamp
        } for r in all_recs])
        st.table(dfm.sort_values("ts", ascending=False).head(10))
