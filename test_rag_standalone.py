import json
from rag_pipeline import RAGPipeline

def run_standalone_test():
    rag = RAGPipeline()
    
    test_cases = [
        ("People from that country are literally parasites and should be removed.", "hate_speech"),
        ("I am going to find you and kill you.", "threat"),
        ("You are such a stupid idiot.", "insult")
    ]
    
    for text, category in test_cases:
        print(f"\n--- Test Case: {category} ---")
        print(f"Input: {text}")
        result = rag.explain(text, category)
        print(f"Explanation: {result.get('explanation')}")
        print(f"Action: {result.get('suggested_action')}")
        if result.get("policy_matches"):
            print("Cited Policies:")
            for match in result["policy_matches"]:
                print(f"  - {match.get('policy_id')}: {match.get('supporting_quote')}")
                
if __name__ == "__main__":
    run_standalone_test()
