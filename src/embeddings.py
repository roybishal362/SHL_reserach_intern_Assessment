"""
Embeddings generation and FAISS index creation for SHL assessments
"""

import json
import pickle
import os
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


class EmbeddingsManager:
    """Manages embeddings creation and FAISS index"""
    
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        """
        Initialize embeddings manager
        
        Args:
            model_name: HuggingFace model name for embeddings
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.assessments = []
        
    def load_assessments(self, json_path: str = 'data/shl_assessments.json'):
        """Load assessment data from JSON file"""
        with open(json_path, 'r', encoding='utf-8') as f:
            self.assessments = json.load(f)
        print(f"Loaded {len(self.assessments)} assessments")
        
    def create_assessment_text(self, assessment: Dict) -> str:
        """
        Create structured rich text representation of assessment for embedding
        """
        # Mapping my field names to System 2 style labels for better semantic capture
        name = assessment.get('assessment_name', '')
        desc = assessment.get('description', '')
        test_type = assessment.get('test_type', 'N/A')
        duration = assessment.get('duration_minutes', 'N/A')
        adaptive = assessment.get('adaptive_support', 'No')
        remote = assessment.get('remote_support', 'No')
        
        # System 2 style structured template
        # We repeat Name and Type for semantic weighting
        content = f"""
Assessment Name: {name}
Title: {name}
Description: {desc[:600]}
Type: {test_type}
Category: {test_type}
Duration: {duration} minutes
Remote Testing Support: {remote}
Adaptive Assessment (IRT): {adaptive}
Keywords: {name}, {test_type}, talent assessment, SHL catalog
"""
        return content.strip()
    
    def create_embeddings(self) -> np.ndarray:
        """
        Generate embeddings for all assessments
        
        Returns:
            Numpy array of embeddings
        """
        print("Creating embeddings for assessments...")
        texts = [self.create_assessment_text(a) for a in self.assessments]
        
        # Generate embeddings in batches for efficiency
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalization
        )
        
        print(f"Generated embeddings shape: {embeddings.shape}")
        return embeddings
    
    def build_faiss_index(self, embeddings: np.ndarray):
        """
        Build FAISS index for fast similarity search
        
        Args:
            embeddings: Numpy array of embeddings
        """
        print("Building FAISS index...")
        
        # Use IndexFlatIP for inner product (works well with normalized embeddings)
        # This is equivalent to cosine similarity when embeddings are normalized
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # Add embeddings to index
        self.index.add(embeddings.astype('float32'))
        
        print(f"FAISS index built with {self.index.ntotal} vectors")
    
    def save_index(self, index_dir: str = 'data/faiss_index'):
        """
        Save FAISS index and metadata
        
        Args:
            index_dir: Directory to save index and metadata
        """
        os.makedirs(index_dir, exist_ok=True)
        
        # Save FAISS index
        index_path = os.path.join(index_dir, 'index.faiss')
        faiss.write_index(self.index, index_path)
        print(f"Saved FAISS index to {index_path}")
        
        # Save metadata (assessments data)
        metadata_path = os.path.join(index_dir, 'metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.assessments, f)
        print(f"Saved metadata to {metadata_path}")
        
        # Save model info
        info_path = os.path.join(index_dir, 'info.json')
        info = {
            'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
            'dimension': self.dimension,
            'num_assessments': len(self.assessments),
            'index_type': 'IndexFlatIP'
        }
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        print(f"Saved index info to {info_path}")
    
    def test_search(self, query: str, top_k: int = 5):
        """
        Test search functionality
        
        Args:
            query: Test query
            top_k: Number of results to return
        """
        print(f"\nTesting search with query: '{query}'")
        
        # Encode query
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        print(f"\nTop {top_k} results:")
        for i, (idx, score) in enumerate(zip(indices[0], scores[0]), 1):
            assessment = self.assessments[idx]
            print(f"\n{i}. {assessment['assessment_name']}")
            print(f"   Score: {score:.4f}")
            print(f"   Type: {assessment['test_type']}")
            print(f"   URL: {assessment['url']}")


def main():
    """Main execution function"""
    # Initialize manager
    manager = EmbeddingsManager()
    
    # Load assessments
    manager.load_assessments()
    
    # Create embeddings
    embeddings = manager.create_embeddings()
    
    # Build FAISS index
    manager.build_faiss_index(embeddings)
    
    # Save everything
    manager.save_index()
    
    # Test with sample queries
    print("\n" + "="*60)
    print("TESTING SEARCH FUNCTIONALITY")
    print("="*60)
    
    test_queries = [
        "Java developer assessment",
        "Sales representative test",
        "Leadership skills evaluation",
        "Python programming test",
        "Personality assessment"
    ]
    
    for query in test_queries:
        manager.test_search(query, top_k=3)
        print("\n" + "-"*60)
    
    print("\n✓ Embeddings and FAISS index created successfully!")


if __name__ == "__main__":
    main()
