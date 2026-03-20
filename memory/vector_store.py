import os
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from config.settings import settings

class VectorStore:
    def __init__(self):
        # Create persistent ChromaDB client
        os.makedirs(settings.DB_DIR, exist_ok=True)
        self.client = chromadb.PersistentClient(path=settings.DB_DIR)
        
        # Initialize embedding model
        from sentence_transformers import SentenceTransformer
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        
        # We start without an active collection. 
        # The API must call set_session(session_id) before querying/adding facts!
        self.collection = None
        self.active_session_id = None

    def set_session(self, session_id: str):
        """Swaps the active in-memory collection to the target session."""
        self.active_session_id = session_id
        # Collection names in Chroma must be alpha-numeric, so we prefix them
        safe_name = f"session_{session_id.replace('-', '_')}"
        self.collection = self.client.get_or_create_collection(
            name=safe_name,
            metadata={"hnsw:space": "cosine"}
        )

    def delete_session(self, session_id: str):
        """Completely obliterates a session's knowledge base."""
        safe_name = f"session_{session_id.replace('-', '_')}"
        try:
            self.client.delete_collection(name=safe_name)
            if self.active_session_id == session_id:
                self.collection = None
                self.active_session_id = None
        except Exception as e:
            print(f"Error deleting collection {safe_name}: {e}")

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.embedding_model.encode(texts)
        return embeddings.tolist()

    def clear(self):
        """Clears the active collection (useful for deep-resets of the current chat)."""
        if not self.collection or not self.active_session_id:
            return
            
        safe_name = f"session_{self.active_session_id.replace('-', '_')}"
        try:
            self.client.delete_collection(safe_name)
            self.collection = self.client.create_collection(
                name=safe_name,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception:
            pass

    def rank_texts(self, query: str, texts: List[str], top_k: int = 5) -> List[int]:
        """Returns the indices of the top_k most similar texts to the query."""
        if not texts:
            return []
            
        import numpy as np
        
        query_emb = np.array(self._get_embeddings([query])[0])
        doc_embs = np.array(self._get_embeddings(texts))
        
        # Cosine similarity
        query_norm = np.linalg.norm(query_emb)
        doc_norms = np.linalg.norm(doc_embs, axis=1)
        
        # Avoid division by zero
        sims = np.dot(doc_embs, query_emb) / (doc_norms * query_norm + 1e-9)
        
        # Get top_k indices (sorted descending)
        top_indices = np.argsort(sims)[::-1][:top_k]
        return top_indices.tolist()

    def add_facts(self, facts: List[Dict[str, Any]]):
        """
        Store facts in the vector database.
        Expected fact format: {"fact": str, "source": str, "confidence": float}
        """
        if not facts:
            return

        # Deduplicate facts within the current batch to prevent ChromaDB ID collision crashes
        unique_facts = {}
        for f in facts:
            if f["fact"] not in unique_facts:
                unique_facts[f["fact"]] = f
                
        deduped_facts = list(unique_facts.values())

        documents = [f["fact"] for f in deduped_facts]
        metadatas = [{"source": str(f.get("source", "")), "confidence": float(f.get("confidence", 0.0))} for f in deduped_facts]
        
        # Create deterministic IDs based on the fact text to avoid cross-batch duplicates
        import hashlib
        ids = [hashlib.md5(f["fact"].encode()).hexdigest() for f in deduped_facts]

        embeddings = self._get_embeddings(documents)

        # Upsert into collection (handles deduplication based on ID)
        if self.collection:
            self.collection.upsert(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings
            )

    def search_facts(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant facts from the vector store."""
        if not self.collection:
            print("⚠️ Cannot search facts: No active session set.")
            return []
            
        query_embedding = self._get_embeddings([query])[0]
        
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            formatted_results = []
            if results and 'documents' in results and results['documents'] and len(results['documents'][0]) > 0:
                for i in range(len(results['documents'][0])):
                    formatted_results.append({
                        "fact": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i] if 'metadatas' in results and results['metadatas'] else {},
                        "distance": results['distances'][0][i] if 'distances' in results and results['distances'] else None
                    })
            return formatted_results
        except Exception as e:
            print(f"⚠️ Vector search failed (DB might be empty for this session): {e}")
            return []

vector_store = VectorStore()
