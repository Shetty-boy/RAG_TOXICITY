# NLP Toxicity Detection + RAG Architecture

This document outlines the architecture for extending the existing Toxicity Detection API with Retrieval-Augmented Generation (RAG) capabilities to provide explanations, context, and suggested actions based on platform policies.

## 1. High-Level Architecture

The system consists of four phases, using **multi-label classification** and **confidence-based routing** to enable precise, category-guided policy retrieval:
1. **Classification:** A multi-label DistilBERT model that returns a score per toxicity category (6 outputs).
2. **Confidence Gate:** A three-tier decision that routes the text based on per-category thresholds.
3. **Category Router:** Uses the dominant toxic category to query only the relevant policy namespace in the Vector DB.
4. **RAG Pipeline:** Retrieves targeted policies and generates a detailed, actionable report.

```text
       [User Text Input]
               │
               ▼
       [FastAPI Endpoint]
               │
=======================================
   PHASE 1: MULTI-LABEL CLASSIFICATION
=======================================
               │
               ▼
     [DistilBERT Classifier]
     (Multi-Label, 6 outputs)
               │
               ▼
  ┌────────────────────────────────┐
  │  toxicity:          0.91       │
  │  hate_speech:       0.12       │
  │  threat:            0.05       │
  │  insult:            0.88       │
  │  obscene:           0.82       │
  │  sexual_content:    0.03       │
  └────────────────────────────────┘
               │
      (max_score = 0.91)
               │
               ▼
=======================================
   PHASE 2: CONFIDENCE GATE (Routing)
=======================================
               │
     ┌─────────┴─────────────────────────┐
     │                    │              │
  any > threshold    max >= 0.30     max < 0.30
  (TOXIC)       (AMBIGUOUS)         (CLEAN)
     │                    │              │
     ▼                    ▼              ▼
  [Flag as           [Send for      [Return:
   Toxic]         LLM Deep Review]  Not Toxic]
     │                    │
     └────────┬───────────┘
              │
              ▼
=======================================
   PHASE 3: CATEGORY ROUTER
=======================================
              │
    (dominant category = insult)
              │
    ┌─────────┴──────────────────────────────────┐
    │                                            │
    │  Which namespace to query in Vector DB?    │
    │                                            │
    │  toxicity          ──> [General Policies]  │
    │  hate_speech       ──> [Hate Speech Docs]  │
    │  threat            ──> [Threat Policies]   │
    │  insult            ──> [Personal Attack Docs]  │
    │  obscene           ──> [Obscenity Policies]│
    │  sexual_content    ──> [Safety Policies]   │
    │                                            │
    └────────────────────────────────────────────┘
              │
              ▼
=======================================
   PHASE 4: RAG PIPELINE (Hybrid Retrieval)
=======================================
              │
              ├──> [Step 1: Query Reformulation]
              │          │
              │    LLM rewrites raw toxic text into
              │    policy-friendly language:
              │    "parasites from country X"
              │       ──> "hate speech targeting
              │            national or ethnic group"
              │          │
              ├──> [Step 2a: Dense Search]   ──┐
              │    Semantic embedding query      │
              │    on reformulated text          │
              │    in category namespace         │
              │                                 │
              ├──> [Step 2b: Sparse Search]  ──┤
              │    BM25 keyword search           │
              │    on same namespace             │  (Reciprocal
              │                                 │   Rank Fusion)
              │    ┌────────────────────────────┘
              │    │
              │    ▼
              │  [Step 3: RRF Score Fusion]
              │  Combines dense + sparse ranks
              │  into a single ranked result set
              │          │
              │    (Top-K policy docs)
              │          ▼
              │  [Insult & Personal Attack Policies]
              │  [Past Cases of Similar Insults    ]
              │          │
              ▼          ▼
         [LLM Generation (Gemini/GPT)]
              │    ┌──────────────────────────────────┐
              │    │  Input: text + category scores   │
              │    │  + dominant category + route     │
              │    │  - TOXIC:   Explain + Action     │
              │    │  - AMBIGUOUS: Verdict + Reason   │
              │    └──────────────────────────────────┘
=======================================
              │
              ▼
     [Rich JSON Response]
```

## 2. Component Breakdown

### A. The Classifier (Requires Retraining)
*   **Model:** Fine-tuned DistilBERT (`distilbert-base-uncased`)
*   **Current limitation:** Single-label binary output (`toxic: true/false`) — cannot distinguish category.
*   **Upgraded purpose:** Multi-label classification returning a score (0.0–1.0) per toxicity category.
*   **What changes in training:**
    ```python
    # Old config
    num_labels = 2
    problem_type = "single_label_classification"
    # loss = CrossEntropyLoss + Softmax

    # New config
    num_labels = 6
    problem_type = "multi_label_classification"
    # loss = BCEWithLogitsLoss(pos_weight=...) + Sigmoid (each label is independent)
    ```
*   **Dataset needed:** Google Civil Comments dataset (free on Hugging Face).
*   **Optimization:** CPU-only inference, `torch.no_grad()`, pre-loaded into memory via FastAPI lifespan.

### B. The Confidence Gate (UPGRADED)
*   **Purpose:** Routes text through the system based on the classifier's confidence to avoid false negatives.
*   **Logic:**
    ```python
    if any(scores[cat] > thresholds[cat] for cat in label_names):
        route = "TOXIC"       # Threshold breached — go directly to RAG + action
    elif max(scores.values()) >= 0.30:
        route = "AMBIGUOUS"   # Classifier is unsure — send to LLM for deep review
    else:
        route = "CLEAN"       # High confidence — return safe immediately
    ```
*   **Why this matters:** A flat max threshold across all labels misses nuanced signals (like a threat at 0.79). Individual thresholds ensure dangerous content is caught appropriately.

### C. The Category Router (NEW)
*   **Purpose:** Uses the dominant toxicity category from the classifier output to query only the relevant namespace in the Vector DB — avoiding a broad, noisy search across all policies.
*   **Logic:**
    ```python
    scores = {
        "toxicity": 0.91, "hate_speech": 0.12,
        "threat": 0.05,   "insult": 0.88,
        "obscene": 0.82, "sexual_content": 0.03
    }
    dominant_category = max(scores, key=scores.get)  # → "toxicity"
    collection = db.get_collection(dominant_category)  # query only that namespace
    ```
*   **Why this matters:** Searching "hate speech policies" for an insult returns noisy, irrelevant results. Targeted namespace retrieval is faster and produces a better LLM context.

### D. The Knowledge Base — Hybrid Retrieval (UPGRADED)
*   **Core Problem:** Raw toxic text has a different vocabulary than policy documents.
    *   User says: `"parasites from country X"` → embedding is about insects/biology
    *   Policy says: `"content targeting national or ethnic groups"` → totally different vector space
    *   Pure semantic search on raw text retrieves the **wrong documents**.
*   **Solution: Three-Layer Hybrid Retrieval**

    **Layer 1 — Metadata Filter (Hard Constraint)**
    ```python
    CURRENT_POLICY_VERSION = "v3.2"

    # All four filters enforced simultaneously by ChromaDB
    collection.query(
        query_texts=[policy_query],
        where={
            "$and": [
                {"category":       {"$eq": dominant_category}},
                {"policy_version": {"$eq": CURRENT_POLICY_VERSION}},
                {"approved":       {"$eq": True}},
                {"deprecated":     {"$eq": False}}
            ]
        }
    )
    ```

    **Layer 2 — Query Reformulation (Vocabulary Bridge)**
    ```python
    # LLM rewrites raw toxic text into policy-document language BEFORE embedding
    raw_text   = "People from country X are parasites."
    policy_query = llm.rewrite(raw_text, category="hate_speech")
    # → "hate speech targeting a national or ethnic group"
    # NOW embed policy_query, not raw_text
    ```

    **Layer 3 — Dense + Sparse Score Fusion (RRF)**
    ```python
    # Dense: semantic similarity on reformulated query
    dense_results  = chroma_collection.query(query_texts=[policy_query], n_results=10)
    # Sparse: BM25 keyword match on same namespace
    sparse_results = bm25_index.get_top_n(policy_query.split(), policy_docs, n=10)
    # Fuse rankings with Reciprocal Rank Fusion
    final_docs = reciprocal_rank_fusion([dense_results, sparse_results], top_k=5)
    ```

*   **Structure:** Separate ChromaDB collection + BM25 index per category:
    *   `toxicity` — General platform abuse & harassment policies
    *   `hate_speech` — Hate speech, slurs, protected group policies
    *   `threat` — Threat escalation, law enforcement referral policies
    *   `insult` — Personal attack, name-calling, severity guidelines
    *   `obscene` — Obscenity and profanity policies
    *   `sexual_content` — Safety, reporting, and content removal policies
*   **Embedding Model:** e.g., `sentence-transformers/all-MiniLM-L6-v2`
*   **Sparse Index:** `rank_bm25` — one BM25 index per category, built from policy document tokens.

### E. The Generator (LLM) — Hallucination-Resistant Design
*   **Model:** A generative LLM (e.g., Google Gemini, OpenAI GPT-4o-mini, or a local model if resources permit).
*   **Core Risk:** Even with RAG, the LLM can hallucinate policy IDs, section numbers, or severity levels that don't exist in any retrieved document. In a moderation system, this means wrongful enforcement backed by a fabricated citation — a legal and trust liability.

*   **Confidence Score — Source and Design**

    Every API response exposes a top-level `confidence` field. It is critical to understand **where this number comes from**:

    | Source | Reliable? | Reason |
    |---|---|---|
    | `max(category_scores)` from DistilBERT classifier ✅ | Yes | Sigmoid outputs are calibrated probabilities |
    | LLM self-reported confidence (asking "how sure are you?") ❌ | No | LLMs are poorly calibrated — may say "95% confident" while hallucinating |

    ```python
    # confidence = max of all category sigmoid scores from the classifier
    confidence = max(scores.values())  # e.g., max(0.91, 0.88, 0.12, ...) → 0.91
    ```
    This is already computed implicitly for the confidence gate — it is simply promoted to a named top-level field in the response for API consumers.

    **Why downstream systems need it:**
    ```
    confidence >= 0.95  →  auto-remove content
    confidence  0.80–0.95  →  auto-hide + flag for human review
    confidence  0.50–0.80  →  flag only, no automated action
    ```

*   **Mitigation 1 — Constrained System Prompt**

    The LLM is given a strict system instruction that prohibits reasoning beyond the retrieved context:
    ```
    SYSTEM PROMPT:
    You are a content moderation assistant. You must:
    1. Base ALL conclusions ONLY on the retrieved policy documents provided below.
    2. Cite the exact policy_id and document name for every claim you make.
    3. If the retrieved documents do not contain sufficient evidence to reach
       a conclusion, respond with "insufficient_evidence": true and explain
       what information is missing. Do NOT infer or fabricate policy details.
    4. Never reference policies, section numbers, or case IDs that do not
       appear in the documents provided to you.
    ```

*   **Mitigation 2 — Citation-Based Structured Output**

    The LLM is forced to respond in a strict JSON schema where every claim must reference a specific retrieved document. Free-form text responses are not allowed:
    ```json
    {
      "route": "TOXIC",
      "dominant_category": "hate_speech",
      "confidence": 0.93,
      "toxic": true,
      "insufficient_evidence": false,
      "predicted_labels": ["toxicity", "hate_speech"],
      "category_scores": {
        "toxicity": 0.93, "hate_speech": 0.85,
        "threat": 0.02,   "insult": 0.10, "obscene": 0.11, "sexual_content": 0.01
      },
      "policy_matches": [
        {
          "policy_id": "HATE_001",
          "section": "Section 4.1 - Hate Speech & Harassment",
          "evidence_doc": "Retrieved Document 2",
          "supporting_quote": "Content targeting persons based on national origin is prohibited."
        }
      ],
      "suggested_action": "24-hour suspension",
      "explanation": "The text targets a national group using dehumanizing language, directly violating HATE_001 as stated in Retrieved Document 2."
    }
    ```

*   **Mitigation 3 — Insufficient Evidence Fallback**

    If retrieved documents don't clearly support a verdict, the LLM must say so rather than guess. This prevents forced conclusions on edge cases:
    ```json
    {
      "route": "AMBIGUOUS",
      "toxic": null,
      "insufficient_evidence": true,
      "missing_context": "Retrieved documents do not contain policies covering this specific type of sarcastic language. Recommend human review.",
      "suggested_action": "Escalate to human moderator."
    }
    ```

*   **Mitigation 4 — Post-Generation Citation Validation (Server-Side)**

    Before the API response is returned to the caller, the server programmatically checks that every `policy_id` cited by the LLM actually exists in the set of retrieved documents. If a hallucinated ID is found, the response is **not returned** — it is flagged for human review instead:
    ```python
    retrieved_policy_ids = {doc["metadata"]["policy_id"] for doc in retrieved_docs}

    for citation in llm_output["policy_violated"]:
        if citation["policy_id"] not in retrieved_policy_ids:
            # Hallucination detected — do not return this response
            return {
                "error": "generation_hallucination_detected",
                "flagged_policy_id": citation["policy_id"],
                "action": "escalated_to_human_review"
            }
    ```
    This converts hallucination from a **silent failure** into a **caught and escalated failure**.


### F. Past Case Curation (NEW)
*   **Problem:** Retrieving past moderation cases without filtering can actively harm LLM output.

    | Risk | What Happens |
    |---|---|
    | **Inconsistent judgments** | Case #500 (old policy): insult → warning. Case #800 (new policy): same insult → ban. LLM gets both and generates a confused, inconsistent recommendation. |
    | **Outdated policy citations** | LLM cites Section 3.1 v1.0, which was superseded by v3.2. The output is factually wrong and potentially a liability. |
    | **Unreviewed cases** | A QA-rejected case slips through and becomes a precedent in the LLM's reasoning. |

*   **Solution: Metadata-Controlled Case Store**

    Every past case stored in the Vector DB **must carry this metadata schema:**
    ```json
    {
      "case_id":        1002,
      "policy_version": "v3.2",
      "category":       "harassment",
      "approved":       true,
      "deprecated":     false,
      "outcome":        "suspension",
      "date":           "2024-08-01"
    }
    ```

    | Field | Purpose |
    |---|---|
    | `policy_version` | Only retrieve cases decided under the **current** policy version (or one prior for transitional context) |
    | `approved` | Human QA-reviewed and signed off before entering the retrieval pool — `false` cases are never retrieved |
    | `deprecated` | Old cases preserved for **audit trail only** — excluded from live retrieval without deleting them |
    | `outcome` | The action taken (warning / suspension / ban) — gives the LLM a concrete precedent to cite |
    | `date` | Allows optional recency filtering if needed |

*   **Operational rule:** When a policy is updated (e.g., v3.1 → v3.2), all cases under v3.1 are bulk-marked `deprecated: true`. They remain in the DB for auditing but are invisible to retrieval. New cases are ingested under `v3.2` only after human approval (`approved: true`).

## 3. Data Flow Examples

### Case A — Confident Toxic, Category: Insult
**Input:** `"You are all a bunch of worthless idiots, get off this platform."`

**Step 1: Multi-Label Classifier Output:**
```json
{ "toxicity": 0.91, "hate_speech": 0.08, "threat": 0.02, "insult": 0.88, "sexual_harassment": 0.01 }
```
**max_score = 0.91 → Route: TOXIC | dominant_category: toxicity**

**Step 2: Category Router →** queries `toxicity` + `insult` namespaces in Vector DB

**Step 3: RAG retrieves:** Policy §4.1 (General Harassment), §3.2 (Personal Attacks & Insults)

**Step 4: LLM Response:**
```json
{
  "route": "TOXIC",
  "dominant_category": "toxicity",
  "confidence": 0.91,
  "toxic": true,
  "insufficient_evidence": false,
  "category_scores": { "toxicity": 0.91, "insult": 0.88, "hate_speech": 0.08, "threat": 0.02, "sexual_harassment": 0.01 },
  "policy_matches": [
    {
      "policy_id": "TOX_004",
      "section": "Section 4.1 - General Harassment",
      "evidence_doc": "Retrieved Document 1",
      "supporting_quote": "Sustained personal attacks targeting individuals or groups are prohibited."
    },
    {
      "policy_id": "INS_002",
      "section": "Section 3.2 - Personal Attacks",
      "evidence_doc": "Retrieved Document 3",
      "supporting_quote": "Use of derogatory labels to demean users constitutes a personal attack."
    }
  ],
  "suggested_action": "24-hour suspension (Severity 8+)",
  "explanation": "The text contains sustained personal insults targeting a group, violating TOX_004 and INS_002 as cited."
}
```

---

### Case B — Confident Toxic, Category: Threat (Different Response Path)
**Input:** `"I know where you live. You'll regret this."`

**Step 1: Multi-Label Classifier Output:**
```json
{ "toxicity": 0.85, "hate_speech": 0.10, "threat": 0.92, "insult": 0.15, "sexual_harassment": 0.02 }
```
**max_score = 0.92 → Route: TOXIC | dominant_category: threat**

**Step 2: Category Router →** queries `threat` namespace ONLY — does NOT retrieve insult or hate speech policies.

**Step 3: RAG retrieves:** Threat escalation policy, law enforcement referral guidelines

**Step 4: LLM Response:**
```json
{
  "route": "TOXIC",
  "dominant_category": "threat",
  "confidence": 0.92,
  "toxic": true,
  "insufficient_evidence": false,
  "category_scores": { "threat": 0.92, "toxicity": 0.85, "insult": 0.15, "hate_speech": 0.10, "sexual_harassment": 0.02 },
  "policy_matches": [
    {
      "policy_id": "THR_001",
      "section": "Section 7.1 - Threats & Physical Safety",
      "evidence_doc": "Retrieved Document 1",
      "supporting_quote": "Any statement implying physical harm or knowledge of a user's location is a credible threat."
    }
  ],
  "suggested_action": "Immediate account suspension + flag for law enforcement referral review.",
  "explanation": "The message constitutes a credible personal threat implying knowledge of location, violating THR_001 as cited."
}
```

---

### Case C — Ambiguous (0.4 < max_score < 0.8) — The False Negative Safety Net
**Input:** `"You're a worthless idiot."`

**Step 1: Multi-Label Classifier Output:**
```json
{ "toxicity": 0.55, "hate_speech": 0.04, "threat": 0.01, "insult": 0.60, "sexual_harassment": 0.01 }
```
**max_score = 0.60 → Route: AMBIGUOUS | dominant_category: insult**
*(Would have been silently returned as clean under the old binary gate)*

**Step 2: Category Router →** queries `insult` namespace

**Step 3: LLM Response:**
```json
{
  "route": "AMBIGUOUS",
  "dominant_category": "insult",
  "confidence": 0.60,
  "toxic": null,
  "insufficient_evidence": false,
  "category_scores": { "insult": 0.60, "toxicity": 0.55, "hate_speech": 0.04, "threat": 0.01, "sexual_harassment": 0.01 },
  "llm_verdict": "TOXIC",
  "policy_matches": [
    {
      "policy_id": "INS_002",
      "section": "Section 3.2 - Personal Attacks & Insults",
      "evidence_doc": "Retrieved Document 2",
      "supporting_quote": "Calling a user 'worthless' or similar derogatory terms constitutes a personal attack."
    }
  ],
  "suggested_action": "Issue a formal warning.",
  "explanation": "Direct personal insult ('worthless idiot') constitutes a personal attack under INS_002. Confidence is moderate — human review recommended before action."
}
```

---

### Case D — Clean (max_score ≤ 0.4)
**Input:** `"I disagree with your point, I think the approach is flawed."`

**Step 1: Multi-Label Classifier Output:**
```json
{ "toxicity": 0.04, "hate_speech": 0.01, "threat": 0.01, "insult": 0.05, "sexual_harassment": 0.01 }
```
**max_score = 0.05 → Route: CLEAN (no RAG or LLM call — saves compute)**

**Step 2: Returned immediately (no RAG or LLM call — saves compute):**
```json
{
  "route": "CLEAN",
  "confidence": 0.05,
  "toxic": false,
  "category_scores": { "toxicity": 0.04, "insult": 0.05, "hate_speech": 0.01, "threat": 0.01, "sexual_harassment": 0.01 }
}
```

## 4. Tech Stack Additions Needed

To implement this, the following libraries would be added to the project:
*   `sentence-transformers` (for dense semantic embeddings)
*   `chromadb` (for vector database with per-category namespaces + metadata filtering)
*   `rank_bm25` (for sparse/keyword BM25 retrieval per category)
*   `google-generativeai` or `langchain` (for LLM API calls, query reformulation, and generation)
*   `datasets` + `scikit-learn` (for retraining with Jigsaw multi-label data)

### Hybrid Retrieval Summary

| Retrieval Layer | Method | Catches | Library |
|---|---|---|---|
| Hard filter | Metadata where-clause | Wrong category docs | ChromaDB `where={}` |
| Query reformulation | LLM rewrite | Vocabulary mismatch | Gemini / GPT |
| Dense search | Semantic embedding | Conceptual similarity | `sentence-transformers` |
| Sparse search | BM25 keyword match | Exact policy term overlap | `rank_bm25` |
| Score fusion | RRF | Best of both signals | Custom (5 lines of code) |

### Retraining Checklist
- [ ] Download Jigsaw Toxic Comment Classification dataset from Kaggle
- [ ] Remap Jigsaw labels to: `toxicity`, `hate_speech`, `threat`, `insult`, `sexual_harassment`
- [ ] Change `num_labels=5`, `problem_type="multi_label_classification"` in DistilBERT config
- [ ] Switch loss to `BCEWithLogitsLoss`, activations to `Sigmoid`
- [ ] Evaluate with `micro-F1` and `ROC-AUC` per label (not just accuracy)
- [ ] Rebuild ChromaDB collections with metadata filters — one per category
- [ ] Build one BM25 index per category from policy documents
- [ ] Add query reformulation step (LLM rewrites toxic text into policy language before embedding)
- [ ] Implement RRF fusion of dense + sparse results
- [ ] Update `main.py` to use `max(scores)` for the confidence gate and `dominant_category` for routing
