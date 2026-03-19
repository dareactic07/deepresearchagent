from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from typing import List
from config.settings import settings

class ResearchQuestions(BaseModel):
    questions: List[str] = Field(description=f"List of exactly {settings.MAX_QUESTIONS} research questions")

def planner_node(state: dict) -> dict:
    topic = state["topic"]
    
    llm = ChatOllama(model=settings.LLM_MODEL, temperature=0.7)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are an expert research assistant. Break down the given research topic into exactly {settings.MAX_QUESTIONS} highly relevant, structured research questions that cover different key aspects of the topic."),
        ("human", "Topic: {topic}")
    ])
    
    try:
        structured_llm = llm.with_structured_output(ResearchQuestions)
        chain = prompt | structured_llm
        print(f"Planning research questions for: {topic}")
        response = chain.invoke({"topic": topic})
        questions = response.questions
        # Enforce limit just in case LLM outputs too many
        questions = questions[:settings.MAX_QUESTIONS]
    except Exception as e:
        print(f"Error in planner (structured output failed?): {e}")
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
        
    return {"questions": questions}
