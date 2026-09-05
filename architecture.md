# NLP Toxicity Detection + RAG Architecture

This document outlines the architecture for extending the existing Toxicity Detection API with Retrieval-Augmented Generation (RAG) capabilities to provide explanations, context, and suggested actions based on platform policies.

## 1. High-Level Architecture

The system consists of two main phases:
1. **Classification (Existing):** A fast, lightweight check to determine if text is toxic.
2. **RAG Pipeline (Proposed):** If toxic, retrieve relevant policies and generate a detailed report.

```text
       [User Text Input]
               │
               ▼
       [FastAPI Endpoint]
               │
=======================================
   PHASE 1: CLASSIFICATION
=======================================
               │
               ▼
     [DistilBERT Classifier]
               │
       (Probability Score)
               │
               ▼
          {Is Toxic?} ────(No)────> [Return: Not Toxic]
               │
             (Yes)
               │
=======================================
   PHASE 2: RAG PIPELINE
=======================================
               │
               ├──> [Embed Text]
               │          │
               │          ▼
               │   [(Vector Database)]
               │   [(ChromaDB/FAISS )]
               │          │
               │     (Retrieves)
               │          ▼
               │  [Relevant Policies]
               │  [& Past Cases     ]
               │          │
               ▼          ▼
          [LLM Generation (Gemini/GPT)]
=======================================
               │
               ▼
      [Rich JSON Response]
```

## 2. Component Breakdown

### A. The Classifier (Current Stack)
*   **Model:** Fine-tuned DistilBERT (`distilbert-base-uncased`)
*   **Purpose:** Fast, binary sequence classification (Toxic / Not Toxic).
*   **Optimization:** CPU-only inference, `torch.no_grad()`, pre-loaded into memory via FastAPI lifespan.

### B. The Knowledge Base (Vector DB)
*   **Data Types:** 
    *   Platform Community Guidelines
    *   Terms of Service
    *   Past moderated examples (e.g., "Case #4421")
    *   Regulatory guidelines (e.g., EU DSA rules)
*   **Embedding Model:** e.g., `sentence-transformers/all-MiniLM-L6-v2`
*   **Storage:** Local vector store (like ChromaDB) for easy deployment alongside the API.

### C. The Generator (LLM)
*   **Model:** A generative LLM (e.g., Google Gemini, OpenAI GPT-4o-mini, or a local model if resources permit).
*   **Prompt Structure:**
    *   **Context:** Retrieved policies from the Vector DB.
    *   **Input:** The flagged toxic text.
    *   **Task:** Explain *why* the text violates the policy and suggest an action (warn, suspend, ban).

## 3. Data Flow Example

**Input:**
`"You are all a bunch of worthless [redacted], get off this platform."`

**Step 1: Classifier Output**
*   `is_toxic`: true
*   `toxic_probability`: 0.98

**Step 2: Vector Search Retrieves:**
*   *Document A: Policy Section 4.1 - Hate Speech & Harassment.*
*   *Document B: Action Guidelines - Severity 8+ requires immediate 24h suspension.*

**Step 3: LLM Generation Output (Final API Response):**
```json
{
  "is_toxic": true,
  "toxic_probability": 0.98,
  "explanation": "The text contains targeted harassment and insulting language, violating the platform's hate speech policy.",
  "policy_reference": "Section 4.1 - Hate Speech & Harassment",
  "suggested_action": "24-hour suspension (Severity 8+)",
  "similar_past_cases": ["Case #8922 - Similar derogatory language, resulted in suspension."]
}
```

## 4. Tech Stack Additions Needed

To implement this, the following libraries would be added to the project:
*   `sentence-transformers` (for generating embeddings)
*   `chromadb` (for the vector database)
*   `google-generativeai` or `langchain` (for LLM API calls and orchestration)
