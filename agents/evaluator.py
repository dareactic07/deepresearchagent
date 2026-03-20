from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
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
    try:
        question = state.get("question", "")
        chunks = state.get("chunks", [])
        
        validated_facts = []
        
        if not chunks:
            print(f"⚠️ No scraped text chunks available for question: '{question}'. Skipping evaluation.")
            return {"validated_facts": {question: []}}
            
        llm = ChatGroq(model=settings.LLM_MODEL, api_key=settings.GROQ_API_KEY, temperature=0.1)
        
        parser = JsonOutputParser(pydantic_object=FactList)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert Critic and Validation Agent. Extract high-quality facts from the chunks based on the question. EXCLUDE contradictions.\n\n"
                       "INSTRUCTIONS:\n"
                       "1. Rate relevance and clarity on a strict scale (0.0 to 1.0).\n"
                       "2. For 'source_id', use the exact integer ID from the chunks provided.\n"
                       "3. YOU MUST RETURN ONLY A VALID JSON OBJECT. No conversational filler.\n\n"
                       "{format_instructions}"),
            ("human", "Question: {question}\n\nText Chunks:\n{chunks}")
        ])
        
        from memory.vector_store import vector_store
        
        # 1. Fragment large text blobs (e.g. from Deep Search) into manageable snippets
        fragmented_pool = []
        for c in chunks:
            text = c.get('text', '')
            url = c.get('url', 'Unknown')
            # If a chunk is massive (un-chunked page), break it up!
            if len(text) > 2000:
                words = text.split()
                for i in range(0, len(words), 300): # ~300 word windows
                    fragmented_pool.append({
                        "text": " ".join(words[i:i+350]), # overlapping windows
                        "url": url
                    })
            else:
                fragmented_pool.append(c)

        # 2. Rank and select only the absolute best snippets to fit TPM limits
        formatted_chunks = []
        if fragmented_pool:
            texts_only = [f.get('text', '') for f in fragmented_pool]
            # Strict limit to avoid TPM/Context issues
            top_k = min(len(fragmented_pool), 10) 
            top_indices = vector_store.rank_texts(question, texts_only, top_k=top_k)
            ranked_chunks = [fragmented_pool[i] for i in top_indices]
        else:
            ranked_chunks = []

        for i, c in enumerate(ranked_chunks):
            formatted_chunks.append(f"CHUNK ID: {i}\nTEXT: {c.get('text', '')}")
            
        chunk_text = "\n\n---\n\n".join(formatted_chunks)
        
        # Use explicit JSON parser chain for Groq reliability
        chain = prompt | llm | parser
        print(f"Evaluating chunks for question: {question}")
        with llm_lock:
            response_dict = chain.invoke({
                "question": question, 
                "chunks": chunk_text,
                "format_instructions": parser.get_format_instructions()
            })
        
        if response_dict and "facts" in response_dict:
            for f_data in response_dict["facts"]:
                # Manual safety check for data types (LLM can hallucinate)
                fact_text = f_data.get("fact", "")
                sid = f_data.get("source_id", -1)
                rel = float(f_data.get("relevance", 0))
                cla = float(f_data.get("clarity", 0))
                
                score = calculate_score(rel, cla)
                if score >= 0.6:
                    mapped_url = "Unknown"
                    if 0 <= sid < len(ranked_chunks):
                        mapped_url = ranked_chunks[sid].get('url', 'Unknown')
                        
                    validated_facts.append({
                        "fact": fact_text,
                        "source": mapped_url,
                        "confidence": score
                    })
        return {"validated_facts": {question: validated_facts}}
    except Exception as e:
        print(f"Error in evaluator for question '{state.get('question','unknown')}': {e}")
        return {"validated_facts": {state.get("question","unknown"): []}}
