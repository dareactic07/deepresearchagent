from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from config.settings import settings
from graph.state import ResearchState
from utils.llm_lock import llm_lock

def synthesizer_node(state: ResearchState) -> dict:
    topic = state.get("topic", "")
    validated_facts_dict = state.get("validated_facts", {})
    
    facts_text = ""
    sources = set()
    for question, facts in validated_facts_dict.items():
        if facts:
            facts_text += f"\n### {question}\n"
            for f in facts:
                facts_text += f"- (Confidence: {f['confidence']}) {f['fact']}\n"
                sources.add(f.get("source", "Unknown"))
                
    source_list = "\n".join([f"- {s}" for s in sources])
    
    llm = ChatOllama(model=settings.LLM_MODEL, temperature=0.4)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert research analyst and technical report writer. "
                   "Using the provided extracted facts, write a HIGHLY DETAILED, EXTENSIVE, AND COMPREHENSIVE markdown research report.\n"
                   "You must expand on each fact and explain its significance in deep detail. The report should be several pages long if possible.\n"
                   "Format exactly as follows:\n\n"
                   "# <Topic>\n\n"
                   "## Executive Summary\n<1-2 paragraphs summarizing the entire topic>\n\n"
                   "## Key Findings\n<Detailed bullet points with deep context>\n\n"
                   "## Detailed Insights\n<Multiple extensive sections explaining the facts in depth, categorized logically>\n\n"
                   "## Sources and Confidence Breakdown\n"
                   "<Provide a detailed bulleted breakdown that explicitly lists every fact utilized, the exact valid Source URL it came from, and its exact confidence score (e.g., - Fact: [fact text] | Source: [URL] | Confidence: [score])>\n"),
        ("human", "Topic: {topic}\n\nExtracted Facts:\n{facts}\n\nWrite the most detailed, extensive report possible based on these facts. Do NOT hallucinate URLs; use only the sources provided in the facts.")
    ])
    
    chain = prompt | llm
    
    try:
        with llm_lock:
            response = chain.invoke({"topic": topic, "facts": facts_text})
        final_report = response.content
    except Exception as e:
        print(f"Error in synthesizer: {e}")
        final_report = f"# {topic}\n\nError generating report. {e}"
        
    return {"final_report": final_report}
