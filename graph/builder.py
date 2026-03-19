from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send
from typing import List

from graph.state import ResearchState
from agents.planner import planner_node
from agents.search import search_node
from agents.scraper import scraper_node
from agents.evaluator import evaluator_node
from agents.synthesizer import synthesizer_node
from memory.vector_store import vector_store

def schedule_search(state: ResearchState) -> List[Send]:
    """Map each question to a search_node call."""
    return [Send("search", {"question": q}) for q in state["questions"]]

def schedule_scraper(state: ResearchState) -> List[Send]:
    """Map each question and its urls to a scraper_node call."""
    sends = []
    for q in state["questions"]:
        urls = state["urls_per_question"].get(q, [])
        sends.append(Send("scraper", {"question": q, "urls": urls}))
    return sends

def schedule_evaluator(state: ResearchState) -> List[Send]:
    """Map each question and its chunks to an evaluator_node call."""
    sends = []
    for q in state["questions"]:
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

def build_graph():
    builder = StateGraph(ResearchState)
    
    builder.add_node("planner", planner_node)
    builder.add_node("search", search_node)
    builder.add_node("scraper", scraper_node)
    builder.add_node("evaluator", evaluator_node)
    builder.add_node("memory_store", memory_store_node)
    builder.add_node("synthesizer", synthesizer_node)
    
    builder.add_edge(START, "planner")
    builder.add_conditional_edges("planner", schedule_search, ["search"])
    builder.add_conditional_edges("search", schedule_scraper, ["scraper"])
    builder.add_conditional_edges("scraper", schedule_evaluator, ["evaluator"])
    builder.add_edge("evaluator", "memory_store")
    builder.add_edge("memory_store", "synthesizer")
    builder.add_edge("synthesizer", END)
    
    return builder.compile()
