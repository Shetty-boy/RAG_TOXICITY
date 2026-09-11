import json
import os

def reformulate_query(toxic_text: str, category: str) -> str:
    """
    Mock LLM: Rewrites the raw toxic text into formal policy language.
    """
    mappings = {
        "toxicity": "general harassment, harmful language, or abusive behavior",
        "hate_speech": "content targeting protected groups, hate speech, or discriminatory language",
        "threat": "credible threats of violence or statements of intent to harm",
        "insult": "personal attacks, name-calling, or targeted insults",
        "obscene": "obscene, vulgar, or excessively profane language",
        "sexual_content": "sexual harassment, explicit remarks, or unwanted sexualization"
    }
    return mappings.get(category, "policy violation")

def generate_explanation(text: str, retrieved_docs: list) -> dict:
    """
    Mock LLM: Generates an explanation based on the retrieved documents.
    """
    if not retrieved_docs:
        return {
            "policy_matches": [],
            "suggested_action": "Requires human review",
            "explanation": "No relevant policies found.",
            "insufficient_evidence": True
        }
        
    top_doc = retrieved_docs[0]
    category = top_doc.get("category", "")
    
    if category == "threat":
        action = "Immediate ban and escalation to law enforcement."
    elif category in ["hate_speech", "sexual_content"]:
        action = "Immediate content removal and account strike."
    elif category == "obscene":
        action = "Content removal."
    else:
        action = "Warning on first offense, temporary suspension for repeated violations."
        
    return {
        "policy_matches": [
            {
                "policy_id": top_doc["policy_id"],
                "section": f"Policy regarding {category}",
                "evidence_doc": f"Version {top_doc.get('policy_version', 'Unknown')}",
                "supporting_quote": top_doc["text"]
            }
        ],
        "suggested_action": action,
        "explanation": f"The provided text violates our {category} policy. As per the policy, {action}",
        "insufficient_evidence": False
    }

def validate_citations(explanation: dict, retrieved_docs: list) -> dict:
    """
    Server-side validation to ensure cited policies were actually in the context.
    """
    retrieved_ids = {doc["policy_id"] for doc in retrieved_docs}
    for match in explanation.get("policy_matches", []):
        if match["policy_id"] not in retrieved_ids:
            return {
                "error": "generation_hallucination_detected",
                "action": "escalated_to_human_review",
                "original_explanation": explanation
            }
    return explanation

class RAGPipeline:
    def __init__(self, db_path="chroma_db", bm25_dir="bm25_indices"):
        # MOCK IMPLEMENTATION due to Windows Python 3.13 C++ build limitations for ChromaDB
        self.policies = []
        self._initialize_mock_db()
        
    def _initialize_mock_db(self):
        policies_path = "policies/mock_policies.json"
        if os.path.exists(policies_path):
            with open(policies_path, "r") as f:
                self.policies = json.load(f)
                            
    def hybrid_retrieval(self, query: str, category: str, k: int = 3):
        """
        Mock retrieval: Returns policies that match the category.
        """
        category_policies = [p for p in self.policies if p["category"] == category and p.get("approved", True) and not p.get("deprecated", False)]
        return category_policies[:k]

    def explain(self, text: str, category: str) -> dict:
        """
        Executes the full RAG explanation pipeline.
        """
        reformulated_query = reformulate_query(text, category)
        retrieved_docs = self.hybrid_retrieval(reformulated_query, category)
        explanation = generate_explanation(text, retrieved_docs)
        final_explanation = validate_citations(explanation, retrieved_docs)
        
        return final_explanation
