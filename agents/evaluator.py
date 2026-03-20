from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from config.settings import settings
from utils.scoring import calculate_score
from utils.llm_lock import llm_lock

class ExtractedFact(BaseModel):
    fact: str = Field(description="The extracted factual information")
    source: str = Field(description="The source URL or chunk reference")
    relevance: float = Field(description="Relevance score to the question (0.0 to 1.0)")
    clarity: float = Field(description="Clarity score of the fact (0.0 to 1.0)")

class FactList(BaseModel):
    facts: List[ExtractedFact] = Field(description="List of extracted facts")

def evaluator_node(state: dict) -> dict:
    """
    Evaluates chunks for a single question.
    Receives {"question": str, "chunks": List[str]}
    """
    question = state.get("question", "")
    chunks = state.get("chunks", [])
    
    validated_facts = []
    
    if not chunks:
        return {"validated_facts": {question: []}}
        
    llm = ChatOllama(model=settings.LLM_MODEL, temperature=0.1)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert Critic and Validation Agent. Review the provided text chunks to answer the research question. Extract only high-quality, non-redundant facts. Actively identify and EXCLUDE low-quality sources, contradictions, or unverified claims. Rate relevance and clarity from 0.0 to 1.0. For the 'source' field, YOU MUST USE THE EXACT URL provided with the chunk. If no reliable facts are found or if the sources are too contradictory, return an empty list."),
        ("human", "Question: {question}\n\nText Chunks:\n{chunks}")
    ])
    
    # Process a highly focused amount of chunks.
    # Format chunks to include the URL clearly
    formatted_chunks = []
    for i, c in enumerate(chunks[:settings.MAX_CHUNKS_PER_URL]):
        formatted_chunks.append(f"URL: {c.get('url', 'Unknown')}\nTEXT: {c.get('text', '')}")
        
    chunk_text = "\n\n---\n\n".join(formatted_chunks)
    
    try:
        structured_llm = llm.with_structured_output(FactList)
        chain = prompt | structured_llm
        print(f"Evaluating chunks for question: {question}")
        with llm_lock:
            response = chain.invoke({"question": question, "chunks": chunk_text})
        
        for f in response.facts:
            score = calculate_score(f.relevance, f.clarity)
            if score >= 0.6: # Configurable threshold
                validated_facts.append({
                    "fact": f.fact,
                    "source": f.source,
                    "confidence": score
                })
    except Exception as e:
        print(f"Error in evaluator for question '{question}': {e}")
        
    return {"validated_facts": {question: validated_facts}}
