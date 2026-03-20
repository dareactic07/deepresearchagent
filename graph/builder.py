from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send
from langgraph.checkpoint.memory import MemorySaver
from typing import List

from graph.state import ResearchState
from agents.planner import planner_node
from agents.search import search_node
from agents.evaluator import evaluator_node
from agents.synthesizer import synthesizer_node
from memory.vector_store import vector_store

def human_approval_node(state: ResearchState) -> dict:
    """Takes user input to approve or modify plan. Bypassed automatically in UI API mode."""
    import os, sys
    if os.environ.get("API_MODE") == "1":
        return {"topic": state["topic"], "questions": state["questions"], "approved": True}
        
    if os.environ.get("STREAMLIT_MODE") == "1" or not sys.stdin.isatty():
        # Non-interactive mode (Streamlit). State is modified externally by the UI.
        return {}

        
    print("\n" + "="*50)
    print("📋 PROPOSED RESEARCH PLAN")
    print("="*50)
    print(state.get("research_plan", "No rationale given."))
    print("\nQuestions to investigate:")
    for q in state.get("questions", []):
        print(f"- {q}")
    
    while True:
        choice = input("\nApprove this plan? [y/modify]: ").strip().lower()
        if choice in ['y', 'yes']:
            # Signal explicit approval via a dedicated state flag! None of this string parsing mess.
            return {"topic": state["topic"], "questions": state["questions"], "approved": True}
        else:
            feedback = input("\nEnter feedback to adjust the research plan: ").strip()
            # Append to topic AND reset approval flag
            return {"topic": state["topic"] + "\nUser Feedback: " + feedback, "questions": state["questions"], "approved": False}

def schedule_search(state: ResearchState) -> List[Send]:
    """Map each question to a search_node call."""
    return [Send("search", {"question": q}) for q in state["questions"]]

def schedule_evaluator(state: ResearchState) -> List[Send]:
    """Map each question and its chunks to an evaluator_node call."""
    sends = []
    for q in state["questions"]:
        # content is now nested correctly as List[Dict[str, str]] from search_node
        chunks = state["extracted_content"].get(q, [])
        sends.append(Send("evaluator", {"question": q, "chunks": chunks}))
    return sends

def memory_store_node(state: ResearchState) -> dict:
    """Store validated facts into short-term (state) and long-term (chroma)."""
    validated_facts_dict = state.get("validated_facts", {})
    all_facts = []
    for facts in validated_facts_dict.values():
        if facts:
            all_facts.extend(facts)
    
    if all_facts:
        vector_store.add_facts(all_facts)
        
    return {}

def gather_search_node(state: ResearchState) -> dict:
    """Empty node to force LangGraph map-reduce synchronization before the next split."""
    return {}

def route_approval(state: ResearchState):
    # Reliable routing using the explicit boolean flag to prevent infinite loops!
    if state.get("approved"):
        return schedule_search(state)
    return "planner"

def build_graph(checkpointer=None, interrupt_before=None):
    builder = StateGraph(ResearchState)
    
    builder.add_node("planner", planner_node)
    builder.add_node("human_approval", human_approval_node)
    builder.add_node("search", search_node)
    builder.add_node("gather_search", gather_search_node)
    builder.add_node("evaluator", evaluator_node)
    builder.add_node("memory_store", memory_store_node)
    builder.add_node("synthesizer", synthesizer_node)
    
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "human_approval")
    builder.add_conditional_edges("human_approval", route_approval, ["planner", "search"])
    builder.add_edge("search", "gather_search")
    builder.add_conditional_edges("gather_search", schedule_evaluator, ["evaluator"])
    builder.add_edge("evaluator", "memory_store")
    builder.add_edge("memory_store", "synthesizer")
    builder.add_edge("synthesizer", END)
    
    kwargs = {}
    if checkpointer is not None:
        kwargs["checkpointer"] = checkpointer
    if interrupt_before is not None:
        kwargs["interrupt_before"] = interrupt_before
        
    return builder.compile(**kwargs)
