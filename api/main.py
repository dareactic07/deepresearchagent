from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
import os

from api import database
from graph.builder import build_graph
from memory.vector_store import vector_store
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from config.settings import settings

app = FastAPI(title="Deep Research API")

# Setup CORS for local Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure DB is initialized
database.init_db()

# Models
class ResearchRequest(BaseModel):
    topic: str

class ChatRequest(BaseModel):
    message: str

@app.get("/api/sessions")
def get_all_sessions():
    return database.get_sessions()

@app.get("/api/sessions/{session_id}")
def get_session_history(session_id: str):
    messages = database.get_messages(session_id)
    if not messages:
        # Check if session exists at all
        sessions = [s for s in database.get_sessions() if s["id"] == session_id]
        if not sessions:
            raise HTTPException(status_code=404, detail="Session not found")
    return {"messages": messages}

@app.post("/api/research")
def start_research(request: ResearchRequest):
    session_id = str(uuid.uuid4())
    
    # Create session in SQLite
    database.create_session(session_id, request.topic)
    
    # Log user's initial prompt
    database.add_message(session_id, "user", request.topic)

    # Tell the vector DB to isolate all scraped facts to this session specifically!
    vector_store.set_session(session_id)
    
    # Initialize the workflow graph
    workflow = build_graph()
    initial_state = {
        "topic": request.topic,
        "questions": [],
        "urls_per_question": {},
        "extracted_content": {},
        "validated_facts": {},
        "scores": {},
        "approved": False,
        "research_plan": "",
        "final_report": ""
    }
    config = {"configurable": {"thread_id": session_id}}
    
    # Set API_MODE to automatically bypass the CLI input() in human_approval
    os.environ["API_MODE"] = "1"
    
    try:
        final_state = workflow.invoke(initial_state, config=config)
        report = final_state.get("final_report", "Error generating report.")
        
        # Save AI's massive research report to the sqlite chat history
        database.add_message(session_id, "assistant", report)
        
        return {"session_id": session_id, "report": report}
    except Exception as e:
        error_msg = f"Failed to complete research: {str(e)}"
        database.add_message(session_id, "assistant", error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/api/sessions/{session_id}/chat")
def chat_with_session(session_id: str, request: ChatRequest):
    # Retrieve isolation context
    vector_store.set_session(session_id)
    
    # Save user message
    database.add_message(session_id, "user", request.message)
    
    # Perform RAG strictly within this namespace
    results = vector_store.search_facts(request.message, n_results=5)
    
    context = ""
    for r in results:
        context += f"- Fact: {r['fact']}\n"
        
    llm = ChatOllama(model=settings.LLM_MODEL, temperature=0.3)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI answering questions based strictly on the extracted research context below:\n\n{context}\n\nIf the answer is not in the context, politely state you don't have that information from the research."),
        ("human", "{question}")
    ])
    
    chain = prompt | llm
    
    try:
        response = chain.invoke({"context": context, "question": request.message})
        ai_reply = response.content
        
        database.add_message(session_id, "assistant", ai_reply)
        return {"reply": ai_reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/sessions/{session_id}")
def delete_chat_session(session_id: str):
    try:
        # Delete from SQLite (Cascades to messages)
        database.delete_session(session_id)
        
        # Completely drop the VectorDB knowledge base namespace!
        vector_store.delete_session(session_id)
        
        return {"status": "success", "message": f"Session {session_id} obliterared."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Entrypoint to standard uvicorn runner
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
