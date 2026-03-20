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
    previous_questions = state.get("questions", [])
    
    llm = ChatOllama(model=settings.LLM_MODEL, temperature=0.7)
    
    prev_q_text = "\n".join([f"- {q}" for q in previous_questions]) if previous_questions else "None"
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are an expert research assistant. Analyze the given research topic and any user feedback. First, write a detailed 'plan_rationale' explaining your overall research strategy. Then, formulate the research questions. \n\nCRITICAL RULES:\n1. You MUST generate a MINIMUM of 3 highly specific, detailed questions. Never output placeholders like 'Question 1' or 'Question 2'.\n2. If 'Previous Questions' are provided below, you MUST RETAIN THEM exactly as they are, and merely APPEND any new questions requested by the user's feedback, UNLESS the user explicitly asks to replace or delete them. Do not wipe out the old questions.\n3. Keep the total questions under {settings.MAX_QUESTIONS}."),
        ("human", "Topic / Feedback: {topic}\n\nPrevious Questions:\n{prev_q_text}")
    ])
    
    try:
        structured_llm = llm.with_structured_output(ResearchQuestions)
        chain = prompt | structured_llm
        print(f"Planning research questions for: {topic}")
        with llm_lock:
            response = chain.invoke({"topic": topic, "prev_q_text": prev_q_text})
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
