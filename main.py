import argparse
from pprint import pprint
from graph.builder import build_graph

def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Research Assistant")
    parser.add_argument("topic", type=str, help="Research topic to query", nargs="?", default="What are the impacts of quantum computing on cryptography?")
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"Starting research on: {args.topic}")
    print(f"{'='*50}\n")
    
    # 1. Build the graph
    app = build_graph()
    
    #    settings.validate()
    
    # Clear previous persistent facts so the chat loop is exclusive to this run's topic
    from memory.vector_store import vector_store
    vector_store.clear()
    
    # 2. Setup initial state
    initial_state = {
        "topic": args.topic,
        "questions": [],
        "urls_per_question": {},
        "extracted_content": {},
        "validated_facts": {},
        "scores": {},
        "final_report": ""
    }
    
    # 3. Stream through the workflow
    print("Running multi-agent workflow...\n")
    try:
        # We need to maintain the cumulative state manually if we want to print things during run, 
        # or just rely on the step outputs which contain the diffs applied to state based on our reducers.
        # We limit max_concurrency to 10 to allow parallel scraping/searching, while Ollama is protected via llm_lock.
        final_state = None
        for step in app.stream(initial_state, config={"recursion_limit": 50, "max_concurrency": 10}):
            for node, state_update in step.items():
                print(f"✅ Completed node: {node}")
                if node == "planner":
                    print(f"   Generated {len(state_update.get('questions', []))} questions.")
                elif node == "search":
                    url_dict = state_update.get('urls_per_question', {})
                    for q, urls in url_dict.items():
                        if urls:
                            print(urls)
                            print(f"   Found {len(urls)} URLs for question...")
                elif node == "evaluator":
                    facts_dict = state_update.get('validated_facts', {})
                    for q, facts in facts_dict.items():
                        if facts:
                            print(f"   Extracted {len(facts)} high-confidence facts for question...")
            final_state = step
            
        print(f"\n{'='*50}")
        print("FINAL RESEARCH REPORT")
        print(f"{'='*50}\n")
        
        if final_state and 'synthesizer' in final_state:
            print(final_state['synthesizer'].get("final_report", "No report generated."))
        else:
            print("Finished workflow execution.")

        print(f"\n{'='*50}")
        print("CONVERSATIONAL FOLLOW-UP")
        print(f"{'='*50}\n")
        print("You can now ask questions about the generated research!")
        print("Type 'exit' or 'quit' to end the session.")
        
        from memory.vector_store import vector_store
        from langchain_ollama import ChatOllama
        from langchain_core.prompts import ChatPromptTemplate
        from config.settings import settings
        
        chat_llm = ChatOllama(model=settings.LLM_MODEL, temperature=0.3)
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert research assistant answering follow-up questions. Use the provided Facts to answer accurately. If the answer is not in the facts, state so. Always cite your sources.\n\nFacts:\n{facts}"),
            ("human", "{question}")
        ])
        chat_chain = chat_prompt | chat_llm

        while True:
            user_question = input("\nAsk a question: ").strip()
            if user_question.lower() in ['exit', 'quit', '']:
                print("\nExiting deep research agent. Goodbye!")
                break
                
            results = vector_store.search_facts(user_question, n_results=5)
            if not results:
                print("No relevant facts found in the knowledge base.")
                continue
                
            facts_text = ""
            for r in results:
                facts_text += f"- {r['fact']} (Source: {r['metadata'].get('source', 'Unknown')})\n"
                
            print("\nAnalyzing knowledge base...")
            try:
                response = chat_chain.invoke({"facts": facts_text, "question": user_question})
                print(f"\n{response.content}")
            except Exception as e:
                print(f"Error generating answer: {e}")
            
    except Exception as e:
        print(f"\nWorkflow failed with error: {e}")

if __name__ == "__main__":
    main()
