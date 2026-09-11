import json
import os
import chromadb
from rank_bm25 import BM25Okapi
import pickle

def main():
    print("Initializing RAG Database build...")
    
    # 1. Load policies
    policies_path = "policies/mock_policies.json"
    if not os.path.exists(policies_path):
        print(f"Error: {policies_path} not found.")
        return
        
    with open(policies_path, "r") as f:
        policies = json.load(f)
        
    print(f"Loaded {len(policies)} policy documents.")
    
    # Group policies by category
    category_policies = {}
    for policy in policies:
        cat = policy["category"]
        if cat not in category_policies:
            category_policies[cat] = []
        category_policies[cat].append(policy)
        
    # 2. Initialize ChromaDB
    # Use SentenceTransformers embedding function natively supported by Chroma
    from chromadb.utils import embedding_functions
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    
    db_path = "chroma_db"
    client = chromadb.PersistentClient(path=db_path)
    
    # 3. Create BM25 index directory
    bm25_dir = "bm25_indices"
    os.makedirs(bm25_dir, exist_ok=True)
    
    # 4. Build databases per category
    for category, docs in category_policies.items():
        print(f"Processing category: {category} ({len(docs)} docs)")
        
        # --- ChromaDB (Dense) ---
        # Get or create collection
        collection = client.get_or_create_collection(name=category, embedding_function=emb_fn)
        
        ids = [doc["policy_id"] for doc in docs]
        texts = [doc["text"] for doc in docs]
        metadatas = [{
            "policy_id": doc["policy_id"],
            "policy_version": doc["policy_version"],
            "category": doc["category"],
            "approved": doc["approved"],
            "deprecated": doc["deprecated"]
        } for doc in docs]
        
        collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        # --- BM25 (Sparse) ---
        tokenized_corpus = [doc.split(" ") for doc in texts] # Simple whitespace tokenization
        bm25 = BM25Okapi(tokenized_corpus)
        
        # Save BM25 index and corresponding mapping
        index_data = {
            "bm25": bm25,
            "docs": docs
        }
        
        with open(os.path.join(bm25_dir, f"{category}.pkl"), "wb") as f:
            pickle.dump(index_data, f)
            
    print("Database build complete!")
    print(f"ChromaDB stored at: {db_path}")
    print(f"BM25 indices stored at: {bm25_dir}")

if __name__ == "__main__":
    main()
