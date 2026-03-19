from tools.search_tool import search_duckduckgo

def search_node(state: dict) -> dict:
    """
    Called individually for each question via the Send mapping.
    State receives {"question": str}
    """
    question = state.get("question", "")
    urls = search_duckduckgo(question)
    
    return {"urls_per_question": {question: urls}}
