import streamlit as st
import uuid
import os

from api import database
from graph.builder import build_graph
from memory.vector_store import vector_store
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from tools.search_tool import search_duckduckgo
from tools.scraper_tool import scrape_url
from utils.chunking import chunk_text
from agents.evaluator import evaluator_node
from config.settings import settings

# Must be the very first Streamlit command
st.set_page_config(page_title="Deep Research Agent", page_icon="🔍", layout="wide")

# Ensure SQLite DB is ready
database.init_db()

# State initialization
if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = None
if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()

# -----------------
# UI: Sidebar Layout
# -----------------
with st.sidebar:
    st.title("🌐 Deep Research")
    
    if st.button("➕ New Topic", use_container_width=True, type="primary"):
        st.session_state.active_session_id = None
        st.rerun()
    
    st.divider()
    st.markdown("### Library")
    
    sessions = database.get_sessions()
    
    if not sessions:
        st.caption("No history yet. Start a topic!")
        
    for session in sessions:
        c1, c2 = st.columns([5, 1])
        with c1:
            is_active = (st.session_state.active_session_id == session['id'])
            
            # Truncate long topics for UI clarity
            safe_topic = session.get('topic') or "Unknown"
            display_topic = safe_topic[:30] + ("..." if len(safe_topic) > 30 else "")
            
            if st.button(
                display_topic, 
                key=f"load_{session['id']}", 
                use_container_width=True,
                type="primary" if is_active else "secondary"
            ):
                st.session_state.active_session_id = session['id']
                st.rerun()
        with c2:
            if st.button("🗑️", key=f"del_{session['id']}", help="Delete this topic entirely"):
                database.delete_session(session['id'])
                vector_store.delete_session(session['id'])
                if st.session_state.active_session_id == session['id']:
                    st.session_state.active_session_id = None
                st.rerun()

# -----------------
# UI: Main Content View
# -----------------
if not st.session_state.active_session_id:
    # Landing Page
    st.markdown("<h1 style='text-align: center; margin-top: 15vh;'>Where knowledge begins</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray; margin-bottom: 2rem;'>Ask a complex question. The Deep Research Agent will automatically scrape the web and synthesize a complete markdown paper.</p>", unsafe_allow_html=True)
    
    # Bottom sticky input handled natively by st.chat_input
    if query := st.chat_input("Ask anything to begin your deep research..."):
        new_session_id = str(uuid.uuid4())
        
        # Save to SQLite and swap session state immediately
        database.create_session(new_session_id, query)
        database.add_message(new_session_id, "user", query)
        
        st.session_state.active_session_id = new_session_id
        st.rerun()
        
else:
    # Active Session Chat View
    session_id = st.session_state.active_session_id
    
    # Grab the topic title safely
    sessions = database.get_sessions()
    topic_title = next((s['topic'] for s in sessions if s['id'] == session_id), "Deep Research Session")
    
    st.header(topic_title)
    
    # Render historical messages from SQLite
    messages = database.get_messages(session_id)
    
    for msg in messages:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])
            
    # ------
    # EXECUTION STAGE
    # ------
    # Trigger graph: If the user just asked the FIRST question, there is only 1 message in history (the user prompt).
    if len(messages) == 1:
        # Context Isolation
        vector_store.set_session(session_id)
        
        # Compile graph dynamically with the persistent memory checkpointer
        workflow = build_graph(checkpointer=st.session_state.memory, interrupt_before=["human_approval"])
        config = {"configurable": {"thread_id": session_id}}
        
        state = workflow.get_state(config)
        
        if not state.values:
            # We haven't started planning yet! Run until the interrupt trigger.
            with st.spinner("⏳ Agent is planning your research strategy..."):
                initial_state = {
                    "topic": messages[0]['content'],
                    "questions": [],
                    "urls_per_question": {},
                    "extracted_content": {},
                    "validated_facts": {},
                    "scores": {},
                    "approved": False,
                    "research_plan": "",
                    "final_report": ""
                }
                try:
                    workflow.invoke(initial_state, config=config)
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to plan: {str(e)}")
                    
        elif "human_approval" in state.next:
            # The agent paused for our approval! Display the proposed plan natively.
            st.info("⚠️ **Action Required:** Review the Deep Research Agent's proposed strategy below.")
            
            with st.container(border=True):
                st.markdown("### 📋 Proposed Strategy")
                st.write(state.values.get("research_plan", "No rationale."))
                st.markdown("### 🔍 Questions to Investigate")
                for q in state.values.get("questions", []):
                    st.markdown(f"- {q}")
                    
            c1, c2 = st.columns(2)
            with c1:
                if st.button("✅ Approve & Begin Deep Research", type="primary", use_container_width=True):
                    workflow.update_state(config, {"approved": True})
                    with st.spinner("⏳ Executing Deep Research... (May take 1 to 3 minutes)"):
                        try:
                            final_state = workflow.invoke(None, config=config)
                            report = final_state.get("final_report", "⚠️ Empty mapping.")
                            database.add_message(session_id, "assistant", report)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Execution Error: {str(e)}")
            with c2:
                with st.popover("✏️ Modify Plan", use_container_width=True):
                    feedback = st.text_area("Provide feedback to adjust the planner's scope:")
                    if st.button("Submit Feedback", type="primary"):
                        new_topic = state.values["topic"] + "\nUser Feedback: " + feedback
                        workflow.update_state(config, {"approved": False, "topic": new_topic})
                        with st.spinner("⏳ Re-planning..."):
                            try:
                                workflow.invoke(None, config=config)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Replanning Error: {str(e)}")
                    
    # Standard Chat Follow ups
    elif len(messages) > 1:
        # Mini Deep Search Toggle
        st.markdown("<br>", unsafe_allow_html=True)
        deep_search = st.toggle("🌐 **Deep Search:** Fetch live web results to answer this question")
        
        if query := st.chat_input("Ask a follow up..."):
            # Render user immediately
            with st.chat_message("user"):
                st.markdown(query)
            database.add_message(session_id, "user", query)
            
            with st.chat_message("assistant"):
                vector_store.set_session(session_id)
                
                # --- DEEP SEARCH INJECTION ---
                if deep_search:
                    with st.spinner("🌐 Deep Searching the web..."):
                        try:
                            # 1. Search
                            urls = search_duckduckgo(query)
                            all_chunks = []
                            # 2. Scrape Top 2 URLs for speed
                            for url in urls[:2]:
                                text = scrape_url(url)
                                if text:
                                    chunks = chunk_text(text)
                                    all_chunks.extend([{"text": c, "url": url} for c in chunks])
                            
                            # 3. Evaluate and inject facts
                            if all_chunks:
                                eval_state = {"question": query, "chunks": all_chunks}
                                eval_result = evaluator_node(eval_state)
                                facts = eval_result["validated_facts"].get(query, [])
                                if facts:
                                    vector_store.add_facts(facts)
                                    st.toast(f"✅ Injected {len(facts)} new facts into knowledge base!")
                        except Exception as e:
                            st.warning(f"Deep Search failed (falling back to local memory): {str(e)}")
                            
                # --- RAG QUERY ---
                with st.spinner("Searching isolated context..."):
                    results = vector_store.search_facts(query, n_results=5)
                    
                    context = ""
                    for r in results:
                        context += f"- Fact: {r['fact']}\n"
                        
                    llm = ChatOllama(model=settings.LLM_MODEL, temperature=0.3)
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", "You are a helpful AI answering questions based strictly on the extracted research context below:\n\n{context}\n\nIf the answer is not in the context, politely state you don't have that information from the research."),
                        ("human", "{question}")
                    ])
                    
                    chain = prompt | llm
                    try:
                        response = chain.invoke({"context": context, "question": query})
                        ai_reply = response.content
                        st.markdown(ai_reply)
                        database.add_message(session_id, "assistant", ai_reply)
                    except Exception as e:
                        st.error(f"Error querying local context: {str(e)}")
