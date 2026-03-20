from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from config.settings import settings
from utils.scoring import calculate_score
from utils.llm_lock import llm_lock

class ExtractedFact(BaseModel):
    fact: str = Field(description="The extracted factual information")
    source_id: int = Field(description="The integer ID of the chunk that provided this fact (e.g., 0, 1, 2)")
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
        print(f"⚠️ No scraped text chunks available for question: '{question}'. Skipping evaluation.")
        return {"validated_facts": {question: []}}
        
    llm = ChatOllama(model=settings.LLM_MODEL, temperature=0.1)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert Critic and Validation Agent. Extract high-quality facts from the chunks based on the question. EXCLUDE contradictions. Rate relevance and clarity on a strict, granular decimal scale from 0.0 to 1.0 (e.g., 0.72, 0.85, 0.94). DO NOT default to 1.0; be fiercely critical and realistic. For the 'source_id' field, YOU MUST return the exact INTEGER ID (e.g. 0, 1, 2) that corresponds to the chunk where you found the fact."),
        ("human", "Question: {question}\n\nText Chunks:\n{chunks}")
    ])
    
    # Process a highly focused amount of chunks using RAG semantic filtering.
    from memory.vector_store import vector_store
    
    formatted_chunks = []
    if chunks:
        texts_only = [c.get('text', '') for c in chunks]
        top_indices = vector_store.rank_texts(question, texts_only, top_k=settings.MAX_CHUNKS_PER_URL)
        ranked_chunks = [chunks[i] for i in top_indices]
    else:
        ranked_chunks = []

    for i, c in enumerate(ranked_chunks):
        formatted_chunks.append(f"CHUNK ID: {i}\nTEXT: {c.get('text', '')}")
        
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
                # Safely map the integer ID back to the exact URL!
                mapped_url = "Unknown"
                if 0 <= f.source_id < len(ranked_chunks):
                    mapped_url = ranked_chunks[f.source_id].get('url', 'Unknown')
                    
                validated_facts.append({
                    "fact": f.fact,
                    "source": mapped_url,
                    "confidence": score
                })
    except Exception as e:
        print(f"Error in evaluator for question '{question}': {e}")
        
    return {"validated_facts": {question: validated_facts}}
