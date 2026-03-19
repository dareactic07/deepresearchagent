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
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        
        # Get or create collection
        self.collection_name = "research_facts"
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.embedding_model.encode(texts)
        return embeddings.tolist()

    def add_facts(self, facts: List[Dict[str, Any]]):
        """
        Store facts in the vector database.
        Expected fact format: {"fact": str, "source": str, "confidence": float}
        """
        if not facts:
            return

        documents = [f["fact"] for f in facts]
        metadatas = [{"source": str(f.get("source", "")), "confidence": float(f.get("confidence", 0.0))} for f in facts]
        
        # Create deterministic IDs based on the fact text to avoid duplicates
        import hashlib
        ids = [hashlib.md5(f["fact"].encode()).hexdigest() for f in facts]

        embeddings = self._get_embeddings(documents)

        # Upsert into collection (handles deduplication based on ID)
        self.collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings
        )

    def search_facts(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant facts from the vector store."""
        query_embedding = self._get_embeddings([query])[0]
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        formatted_results = []
        if results['documents'] and len(results['documents']) > 0:
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    "fact": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i] if 'distances' in results else None
                })
        return formatted_results

vector_store = VectorStore()
