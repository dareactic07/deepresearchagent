from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from typing import List
from config.settings import settings
from utils.llm_lock import llm_lock

class ResearchQuestions(BaseModel):
    # This must match the schema exactly for json_mode!
    plan_rationale: str = Field(description="A detailed paragraph explaining your overall research strategy.")
    questions: List[str] = Field(description="List of specific research questions")

def planner_node(state: dict) -> dict:
    try:
        topic = state["topic"]
        previous_questions = state.get("questions", [])
        
        # GROQ: Use ultra-fast LPU inference
        llm = ChatGroq(model=settings.LLM_MODEL, api_key=settings.GROQ_API_KEY, temperature=0.7)
        
        max_q = settings.MAX_QUESTIONS
        prev_q_text = "\n".join([f"- {q}" for q in previous_questions]) if previous_questions else "None"
        
        parser = JsonOutputParser(pydantic_object=ResearchQuestions)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert research assistant. Analyze the given research topic and any user feedback.\n\n"
                       "INSTRUCTIONS:\n"
                       "1. Write a detailed 'plan_rationale' explaining your strategy.\n"
                       "2. Formulate exactly {max_q} research questions.\n"
                       "3. YOU MUST RETURN ONLY A VALID JSON OBJECT matching the schema below. No conversational filler.\n\n"
                       "{format_instructions}"),
            ("human", "Topic / Feedback: {topic}\n\nPrevious Questions:\n{prev_q_text}")
        ])
        
        chain = prompt | llm | parser
        print(f"Planning research questions for: {topic}")
        with llm_lock:
            # We pass the strings directly to the prompt variables
            response_dict = chain.invoke({
                "topic": topic, 
                "max_q": max_q,
                "prev_q_text": prev_q_text,
                "format_instructions": parser.get_format_instructions()
            })
            
        questions = response_dict.get("questions", [])
        plan_rationale = response_dict.get("plan_rationale", "No strategy provided.")
        # Enforce limit just in case LLM outputs too many
        questions = questions[:max_q]
        return {"questions": questions, "research_plan": plan_rationale}
    except Exception as e:
        print(f"Error in planner (JSON parsing or API failed?): {e}")
        topic = state.get("topic", "the topic")
        max_q = settings.MAX_QUESTIONS
        return {
            "questions": [
                f"What is the fundamental basis and current state of {topic}?", 
                f"What are the most significant impacts or future trends of {topic}?",
                f"Who are the key players or main sources studying {topic}?",
            ][:max_q], 
            "research_plan": "Fallback strategy due to error."
        }


