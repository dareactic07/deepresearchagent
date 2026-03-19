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
        # We limit max_concurrency to 1 because local LLMs (Ollama) will struggle/crash when hit with 8 concurrent heavy evaluation requests
        final_state = None
        for step in app.stream(initial_state, config={"recursion_limit": 50, "max_concurrency": 1}):
            for node, state_update in step.items():
                print(f"✅ Completed node: {node}")
                if node == "planner":
                    print(f"   Generated {len(state_update.get('questions', []))} questions.")
                elif node == "search":
                    url_dict = state_update.get('urls_per_question', {})
                    for q, urls in url_dict.items():
                        if urls:
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
            
    except Exception as e:
        print(f"\nWorkflow failed with error: {e}")

if __name__ == "__main__":
    main()
