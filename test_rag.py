from fastapi.testclient import TestClient
from main import app
import json

client = TestClient(app)

def run_tests():
    test_cases = [
        "People from that country are literally parasites and should be removed.",
        "I am going to find you and kill you.",
        "You are such a stupid idiot.",
        "Have a nice day!"
    ]
    
    # Manually trigger startup events in TestClient context
    with client as c:
        for i, text in enumerate(test_cases):
            print(f"\n--- Test Case {i+1} ---")
            print(f"Input: {text}")
            try:
                response = c.post("/explain", json={"text": text})
                if response.status_code == 200:
                    result = response.json()
                    print(f"Route: {result['route']}")
                    print(f"Dominant Category: {result['dominant_category']}")
                    print(f"Explanation: {result.get('explanation')}")
                    print(f"Action: {result.get('suggested_action')}")
                    if result.get("policy_matches"):
                        print("Cited Policies:")
                        for match in result["policy_matches"]:
                            print(f"  - {match.get('policy_id')}: {match.get('supporting_quote')}")
                else:
                    print(f"Error: {response.status_code}")
                    print(response.text)
            except Exception as e:
                print(f"Failed to connect or run test: {e}")

if __name__ == "__main__":
    run_tests()
