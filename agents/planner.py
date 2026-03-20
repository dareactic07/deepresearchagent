from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from typing import List
from config.settings import settings
from utils.llm_lock import llm_lock

class ResearchQuestions(BaseModel):
    plan_rationale: str = Field(description="A detailed paragraph explaining your overall research strategy and why you chose these specific questions.")
    questions: List[str] = Field(description=f"List of exactly {settings.MAX_QUESTIONS} research questions")

def planner_node(state: dict) -> dict:
    topic = state["topic"]
    
    llm = ChatOllama(model=settings.LLM_MODEL, temperature=0.7)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are an expert research assistant. Analyze the given research topic. First, write a detailed 'plan_rationale' explaining your overall research strategy. Then, break down the topic into exactly {settings.MAX_QUESTIONS} highly relevant, structured research questions that cover different key aspects."),
        ("human", "Topic: {topic}")
    ])
    
    try:
        structured_llm = llm.with_structured_output(ResearchQuestions)
        chain = prompt | structured_llm
        print(f"Planning research questions for: {topic}")
        with llm_lock:
            response = chain.invoke({"topic": topic})
        questions = response.questions
        plan_rationale = response.plan_rationale
        # Enforce limit just in case LLM outputs too many
        questions = questions[:settings.MAX_QUESTIONS]
    except Exception as e:
        print(f"Error in planner (structured output failed?): {e}")
        plan_rationale = "Fallback standard strategy due to structured parsing error."
        questions = [
            f"What is the fundamental basis and current state of {topic}?", 
            f"What are the most significant impacts, consequences, or future trends of {topic}?",
            f"Who are the key players or main sources studying {topic}?",
            f"What are the major challenges or limitations regarding {topic}?",
            f"What historical context or background is necessary to understand {topic}?",
            f"How does {topic} compare to alternative or opposing concepts?",
            f"What are the ethical or societal implications of {topic}?",
            f"How is {topic} expected to evolve in the next decade?"
        ][:settings.MAX_QUESTIONS]
        
    return {"questions": questions, "research_plan": plan_rationale}
