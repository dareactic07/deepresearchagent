from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
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
                facts_text += f"- Fact: {f['fact']} | Source: {f.get('source', 'Unknown')}\n"
                sources.add(f.get("source", "Unknown"))
                
    source_list = "\n".join([f"- {s}" for s in sources])
    
    llm = ChatGroq(model=settings.LLM_MODEL, api_key=settings.GROQ_API_KEY, temperature=0.4, max_tokens=6000)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert research analyst and technical report writer. "
                   "Using the provided extracted facts, write a HIGHLY DETAILED, EXTENSIVE, AND COMPREHENSIVE markdown research report.\n"
                   "CRITICAL INSTRUCTION: You must complete the report. DO NOT get cut off due to length. Ensure you manage your length so the 'References' section is ALWAYS generated at the very end.\n"
                   "CRITICAL INSTRUCTION: Format the report like an academic research paper. For EVERY single claim or fact, you MUST include a bracketed numerical citation inline (e.g., [1], [2]).\n"
                   "At the very end of the report, you MUST include a 'References' section that maps each number to its exact Source URL.\n"
                   "Format exactly as follows:\n\n"
                   "# <Topic>\n\n"
                   "## Abstract\n<1-2 paragraphs summarizing the entire topic. Use numerical citations [1]>\n\n"
                   "## Key Findings\n<Detailed bullet points with deep context. Use numerical citations [2]>\n\n"
                   "## Detailed Analysis\n<Multiple extensive sections explaining the facts in depth, categorized logically. Use numerical citations [3]>\n\n"
                   "## References\n"
                   "<A numbered list mapping your inline citations to the exact Source URLs provided in the facts. E.g., [1] https://...>\n"),
        ("human", "Topic: {topic}\n\nExtracted Facts:\n{facts}\n\nWrite the most detailed, extensive report possible based on these facts. Do NOT hallucinate URLs. You MUST use academic numerical citations [1] inline, and list the exact sources in the References section at the end. Failure to cite the source for a claim is unacceptable.")
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
