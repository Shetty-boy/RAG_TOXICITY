# Project Decision Log — RAG Toxicity Detection System

> **Purpose:** This file is a living record of every decision made in this project.
> Every architectural choice, design change, and improvement is logged here with
> the problem it solves and the reasoning behind it. Update this file whenever
> a change is made to the project.
>
> **How to read this:** Each entry follows the format:
> - **What** was changed or added
> - **Why** — the specific problem it solves
> - **Decision** — what was chosen and why over alternatives

---

## Project Overview

This project started as a **binary toxicity detection API** — a fine-tuned DistilBERT model
served via FastAPI that classifies text as toxic or not toxic. It is being extended into a
**full RAG (Retrieval-Augmented Generation) pipeline** that not only detects toxicity but
also explains *why* content is toxic, cites the specific policy violated, and recommends
a moderation action.

**Target users:** Companies and platforms that need automated, explainable content moderation
(social media, gaming platforms, HR tools, customer support systems).

---

## Phase 1 — Foundation (Completed)

### [1.1] Initial Toxicity Detection API

**What:** A FastAPI REST API wrapping a fine-tuned DistilBERT model for binary toxicity classification.

**Stack:**
- Model: `distilbert-base-uncased` fine-tuned for sequence classification
- Framework: FastAPI + Uvicorn
- Inference: PyTorch (CPU-only, `torch.no_grad()`)
- Deployment: Docker (Python 3.10-slim)

**Key implementation decisions:**
- **Model loaded once at startup** (via FastAPI `lifespan` context manager) rather than
  per-request. This avoids reloading ~255MB weights on every API call, which would make
  the API unusably slow.
- **CPU-only PyTorch build** (`torch==2.1.1+cpu`) to keep the Docker image lightweight
  and deployable on standard cloud instances without GPU costs.
- **`token_type_ids` stripped from DistilBERT inputs** — DistilBERT does not use segment
  embeddings (unlike BERT), so passing `token_type_ids` causes a runtime error. The model
  inference code explicitly deletes this key if present.
- **`model.eval()` called at startup** — disables dropout layers, reducing memory usage
  and ensuring deterministic inference output.

**API endpoints:**
- `GET /health` — liveness check
- `POST /predict` — accepts `{"text": "..."}`, returns `{"toxic_probability": 0.87, "is_toxic": true}`

---

## Phase 2 — Architecture Design

### [2.1] Decision to Extend with RAG

**Problem:** The binary API (`"is_toxic": true`) is too blunt for real-world moderation.
It tells you *that* content is toxic but not *why*, *which policy* was violated, or *what
action* to take. Human moderators reviewing flagged content need this context.

**Decision:** Add a RAG pipeline on top of the classifier. The classifier acts as a fast
first-pass filter; the RAG pipeline adds explanation, policy citation, and action recommendation.

**Why RAG and not just an LLM alone:**
- An LLM alone would hallucinate policy details it was never given
- RAG grounds the LLM's output in actual retrieved policy documents
- The classifier handles the binary decision cheaply; the LLM only runs when needed

**Real-world equivalents:** Google Jigsaw/Perspective API, Microsoft Azure Content Safety,
ActiveFence — all build this kind of multi-layer pipeline.

---

### [2.2] Architecture Documented in `architecture.md`

**What:** Created `architecture.md` as the single source of truth for the system design.

**Decision:** Use a plain text ASCII flowchart instead of Mermaid diagrams.

**Why:** Mermaid requires a special markdown renderer to visualize. ASCII art is readable
in any text editor, terminal, or IDE without plugins — making the document universally
accessible.

---

## Phase 3 — Architecture Improvements (Iterative)

### [3.1] Confidence-Based Routing (Replacing the Hard Binary Gate)

**Problem:** The original design used a hard binary gate:
```
score >= 0.5 → Toxic → RAG pipeline
score < 0.5  → Not Toxic → Return immediately
```
This creates a **false negative vulnerability**: a text scored at `0.45` would be silently
returned as "not toxic" and never reach the RAG pipeline or LLM review. Real toxic content
near the decision boundary gets missed entirely.

**Example:** `"You're a worthless idiot."` might score `0.55` — borderline but still
personally abusive language that a moderator would want to flag.

**Decision:** Replace the binary gate with a **three-tier confidence router:**
```python
if score > 0.8:
    route = "TOXIC"      # High confidence — act directly
elif 0.4 < score <= 0.8:
    route = "AMBIGUOUS"  # Classifier unsure — send to LLM for deeper review
else:
    route = "CLEAN"      # High confidence — return safe immediately
```

**Why this works:** The AMBIGUOUS zone acts as a safety net for false negatives. Content
the classifier is uncertain about is escalated to the LLM for a second opinion rather than
silently passed.

**Industry pattern:** This is called "confidence-based routing" — used in production ML
systems at Google, Meta, and Microsoft Trust & Safety teams.

**Note:** The thresholds (`0.4`, `0.8`) are starting values. They should be calibrated
on a validation set by analysing the score distribution in `eda.ipynb`.

---

### [3.2] Multi-Label Classification (Replacing Binary Output)

**Problem:** The original model outputs a single score: `{"toxic": true}`.
This tells you content is toxic but gives no information about *what kind* of toxicity.
Without category information, the RAG pipeline cannot target the right policy documents —
it has to search across all policies, introducing noise.

**Example of the problem:**
- A threat and an insult both return `{"toxic": true}`
- But a threat should escalate to law enforcement review; an insult should trigger a warning
- Without categories, the system cannot make this distinction

**Decision:** Retrain DistilBERT as a **multi-label classifier** outputting a score per
toxicity category:
```json
{
  "toxicity": 0.91,
  "hate_speech": 0.12,
  "threat": 0.05,
  "insult": 0.88,
  "sexual_harassment": 0.03
}
```

**What changes in the model:**
| Setting | Old (Binary) | New (Multi-Label) |
|---|---|---|
| `num_labels` | 2 | 5 |
| `problem_type` | `single_label_classification` | `multi_label_classification` |
| Loss function | `CrossEntropyLoss` + Softmax | `BCEWithLogitsLoss` + Sigmoid |
| Activation | Softmax (sum to 1) | Sigmoid (each label independent) |

**Dataset needed:** [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
(free on Kaggle) — provides labels: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`.

**Evaluation metrics to use:** `micro-F1` and `ROC-AUC` per label — not accuracy (accuracy
is misleading on imbalanced multi-label datasets).

---

### [3.3] Category Router (Phase 3 in the Pipeline)

**Problem:** Even with multi-label output, if the Vector DB is queried without filtering,
the search runs across all policy documents regardless of category. A query for insult-
related text might retrieve hate speech policies — irrelevant and noisy context.

**Decision:** Add a **Category Router** between the Confidence Gate and RAG Pipeline.
The dominant category (highest-scoring label) determines which Vector DB namespace
(collection) is queried:
```python
dominant_category = max(scores, key=scores.get)
collection = db.get_collection(dominant_category)
```

**Namespace structure in ChromaDB:**
- `toxicity` → General harassment and abuse policies
- `hate_speech` → Slurs, protected group targeting policies
- `threat` → Escalation procedures, law enforcement referral guidelines
- `insult` → Personal attack and name-calling severity guidelines
- `sexual_harassment` → Safety, reporting, and content removal policies

**Why this matters:** A threat detection case that searches the `threat` namespace
exclusively retrieves law enforcement escalation policies — exactly what's needed.
Without routing, it might retrieve generic harassment guidelines that don't inform
the right action.

---

### [3.4] Hybrid Retrieval — Fixing the Vocabulary Mismatch Problem

**Problem:** Within a category namespace, retrieval is still done by embedding the
**raw toxic text** and finding semantically similar policy documents. This fails because:

- **User text vocabulary:** `"parasites from country X"`, `"get out"`, `"vermin"`
- **Policy document vocabulary:** `"content targeting protected groups"`, `"national origin"`,
  `"hate speech"`, `"dehumanizing language"`

These vocabularies barely overlap. A semantic embedding of toxic slang will NOT reliably
match the formal language of policy documents. Pure semantic search retrieves wrong documents.

**Decision:** Replace pure semantic search with **three-layer hybrid retrieval:**

**Layer 1 — Metadata Hard Filter**
Before any search, filter the collection by category, policy version, approval status,
and deprecation flag using ChromaDB's `where={}` clause. This eliminates irrelevant
documents before any ranking happens.
```python
collection.query(
    query_texts=[policy_query],
    where={"$and": [
        {"category":       {"$eq": dominant_category}},
        {"policy_version": {"$eq": CURRENT_POLICY_VERSION}},
        {"approved":       {"$eq": True}},
        {"deprecated":     {"$eq": False}}
    ]}
)
```

**Layer 2 — Query Reformulation (Vocabulary Bridge)**
Before embedding, an LLM rewrites the raw toxic text into formal policy language.
This bridges the vocabulary gap before any vector search happens:
```
Raw:       "People from country X are parasites."
Rewritten: "hate speech targeting a national or ethnic group"
```
The rewritten query is embedded — not the raw toxic text.

**Layer 3 — Dense + Sparse Score Fusion (RRF)**
Two retrievers run in parallel on the reformulated query:
- **Dense (semantic):** `sentence-transformers` + ChromaDB — catches conceptual similarity
- **Sparse (BM25):** `rank_bm25` keyword matching — catches exact policy term overlap

Results are fused using **Reciprocal Rank Fusion (RRF)**, a mathematically proven method
for combining ranked lists that outperforms either retriever alone.

**Why all three are needed:**
| Layer | What it catches | Without it |
|---|---|---|
| Metadata filter | Wrong category documents | Insult queries get threat policies |
| Query reformulation | Vocabulary mismatch | Raw slang doesn't match policy text |
| Dense search | Conceptual/paraphrase similarity | Misses semantically related but differently worded docs |
| Sparse BM25 | Exact policy term overlap | Dense search misses specific legal terms |
| RRF fusion | Best of both signals | Single retriever leaves relevant docs behind |

---

### [3.5] Past Case Curation — Metadata-Controlled Case Store

**Problem:** The RAG knowledge base contains two types of documents:
1. Policy documents (static, versioned)
2. Past moderation cases (dynamic, potentially outdated or inconsistent)

Retrieving past cases **without filtering** introduces serious risks:
- **Inconsistent judgments:** Case #500 (old policy) gave a warning for language that
  current policy (v3.2) mandates a ban for. LLM gets both as context and produces an
  inconsistent recommendation.
- **Outdated citations:** LLM cites Section 3.1 v1.0 which was superseded. The output
  is factually wrong and potentially a legal liability.
- **Unreviewed cases:** A QA-rejected case slips into the retrieval pool and becomes
  a precedent in the LLM's reasoning.

**Decision:** Every past case stored in the Vector DB must carry a mandatory metadata schema:
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

**Operational rule:** When a policy is updated (e.g., v3.1 → v3.2), all cases under v3.1
are bulk-marked `deprecated: true`. They remain in the DB for audit trail and compliance
purposes (required by regulations like the EU Digital Services Act) but are invisible to
live retrieval. New cases are only ingested under `v3.2` after human approval (`approved: true`).

**Why not just delete old cases?** Deletion destroys the audit trail. Legal and compliance
teams need evidence of past moderation decisions. The `deprecated` flag preserves the data
while excluding it from influencing live LLM outputs.

---

### [3.6] LLM Hallucination Mitigations (Four-Layer Defence)

**Problem:** Even with RAG, an LLM can hallucinate policy details — citing `"Policy 4.2"`
when no such policy exists in the retrieved documents. In a moderation system this means:
- A user is wrongfully suspended/banned citing a fabricated policy
- The platform has no legal basis for the action
- If audited, the decision trail is fraudulent

**Decision:** Four mitigation layers in combination:

**Mitigation 1 — Constrained System Prompt**
The LLM system prompt explicitly prohibits reasoning beyond retrieved documents and
requires citing exact policy IDs for every claim. The first line of defence.

**Mitigation 2 — Citation-Based Structured Output**
The LLM must respond in a strict JSON schema where every claim references a specific
retrieved document (`policy_id`, `evidence_doc`, `supporting_quote`). The schema makes
fabrication structurally harder — you cannot fill a `supporting_quote` field without
actually having the text from a retrieved document.

**Mitigation 3 — Insufficient Evidence Fallback**
The LLM is explicitly taught that `"insufficient_evidence": true` is a valid and preferred
response over a guessed verdict. Edge cases (sarcasm, coded language, cultural context)
are escalated to a human moderator rather than forced into a potentially wrong decision.

**Mitigation 4 — Post-Generation Citation Validation (Server-Side)**
After the LLM responds, the server programmatically checks every cited `policy_id` in
the output against the actual retrieved document set:
```python
retrieved_policy_ids = {doc["metadata"]["policy_id"] for doc in retrieved_docs}
for citation in llm_output["policy_matches"]:
    if citation["policy_id"] not in retrieved_policy_ids:
        return {"error": "generation_hallucination_detected", "action": "escalated_to_human_review"}
```
Mitigations 1–3 reduce hallucination probability. Mitigation 4 **catches what still slips
through** before it reaches the end user. This turns hallucination from a silent failure
into a caught and escalated failure.

---

### [3.7] Confidence Scores in API Response

**Problem:** The API response returned `"is_toxic": true/false` — a binary that loses all
information about how certain the system is. Downstream consumers (dashboards, auto-action
pipelines, human review queues) cannot distinguish a `0.51` call from a `0.99` call.

**Decision:** Add a top-level `confidence` field to every API response.

**Critical design decision — source of confidence:**
`confidence` = `max(category_scores)` from the DistilBERT classifier — **NOT** self-reported
LLM confidence. LLMs are poorly calibrated and may say "95% confident" while hallucinating.
The classifier's sigmoid probability outputs are reliable, calibrated numbers.

```python
confidence = max(scores.values())  # e.g., max(0.91, 0.88, 0.12, ...) → 0.91
```

**Why downstream systems need it — tiered automated actions:**
```
confidence >= 0.95  →  auto-remove content
confidence 0.80–0.95 →  auto-hide + flag for human review
confidence 0.50–0.80 →  flag only, no automated action
```

**Final unified API response schema:**
```json
{
  "route": "TOXIC",
  "dominant_category": "hate_speech",
  "confidence": 0.93,
  "toxic": true,
  "insufficient_evidence": false,
  "category_scores": {
    "toxicity": 0.93, "hate_speech": 0.85,
    "threat": 0.02, "insult": 0.10, "sexual_harassment": 0.01
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
  "explanation": "..."
}
```

---

### [3.8] Multi-Label Output Labels and Loss

**Problem:** Determining the exact set of labels for the multi-label classifier and dealing with class imbalance (e.g. threats are extremely rare).

**Decision:**
1. **6 Labels chosen:** We use `toxicity`, `hate_speech`, `threat`, `insult`, `obscene`, and `sexual_content`. We kept `obscene` separate from `insult` as they require different policy responses. `sexual_harassment` was renamed to `sexual_content` since the dataset measures explicit content rather than harassment specifically.
2. **Weighted loss over oversampling:** To handle class imbalance, we use the natural distribution from the `google/civil_comments` dataset but apply `pos_weight` in the `BCEWithLogitsLoss`. This teaches the model realistic priors while still penalizing misses on rare classes like `threat`.
3. **Per-category threshold routing:** Rather than a flat score gate, the confidence router triggers `TOXIC` if any category exceeds its configured threshold (stored in `label_config.json`). This ensures that a `threat` with a score of 0.79 is properly flagged.
4. **Predicted labels array:** The API response now includes `predicted_labels` listing all active categories, acknowledging that text can trigger multiple violations (e.g. both insult and obscene).

---

## Current System Architecture Summary

The system now has **4 phases:**

```
Phase 1: Multi-Label Classification
         → DistilBERT outputs 5 category scores (0.0–1.0 each)

Phase 2: Confidence Gate
         → max_score > 0.8    : TOXIC   → continue to Phase 3
         → 0.4 < max_score <= 0.8 : AMBIGUOUS → continue to Phase 3
         → max_score <= 0.4   : CLEAN   → return immediately (no LLM cost)

Phase 3: Category Router
         → dominant_category = max(scores)
         → routes to category-specific Vector DB namespace

Phase 4: RAG Pipeline (Hybrid Retrieval)
         → Query reformulation (LLM rewrites toxic text into policy language)
         → Dense search (sentence-transformers + ChromaDB)
         → Sparse search (BM25 via rank_bm25)
         → RRF fusion → top-K policy documents
         → LLM generation with constrained prompt
         → Server-side citation validation
         → Return rich JSON response
```

---

## Files in this Repository

| File | Purpose |
|---|---|
| `main.py` | FastAPI app — current binary classifier API |
| `architecture.md` | Full system architecture with ASCII flowcharts |
| `DECISIONS.md` | This file — decision log and project journal |
| `eda.ipynb` | Exploratory Data Analysis and model training notebook |
| `best_toxicity_model/` | Saved best checkpoint of the DistilBERT classifier |
| `distilbert-toxicity-model/` | Training checkpoints (steps 500, 1000, 1500) |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Container definition for production deployment |
| `.gitignore` | Excludes large model weights from Git (>100MB GitHub limit) |

---

## Pending Work / Next Steps

- [x] Retrain DistilBERT as multi-label classifier on Civil Comments dataset
- [ ] Build ChromaDB collections — one per category, with policy documents
- [ ] Build BM25 index per category from policy document tokens
- [ ] Implement query reformulation step in `main.py`
- [ ] Implement RRF fusion of dense + sparse results
- [ ] Add `/explain` endpoint that triggers the full RAG pipeline
- [ ] Implement server-side citation validation (Mitigation 4)
- [ ] Calibrate confidence thresholds on validation set
- [ ] Build a simple frontend dashboard to visualize moderation decisions

---

*Last updated: 2026-09-06*
*Update this file whenever an architectural decision is made or a significant change is implemented.*
