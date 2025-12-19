"""
Retrieval module with Hybrid Search (Semantic + BM25) and diversity constraints
"""

import json
import pickle
import os
from typing import List, Dict, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from rank_bm25 import BM25Okapi
import re


class RetrievalSystem:
    """Handles hybrid search (Semantic + BM25) and diversity-based re-ranking"""
    
    def __init__(self, index_dir: str = 'data/faiss_index'):
        """
        Initialize retrieval system
        
        Args:
            index_dir: Directory containing FAISS index and metadata
        """
        self.index_dir = index_dir
        self.model = None
        self.index = None
        self.assessments = []
        self.bm25 = None
        self.load_index()
        self.init_bm25()
        
    def load_index(self):
        """Load FAISS index, metadata, and model"""
        # Load info
        info_path = os.path.join(self.index_dir, 'info.json')
        with open(info_path, 'r') as f:
            info = json.load(f)
        
        print(f"Loading embedding model: {info['model_name']}")
        try:
            self.model = SentenceTransformer(info['model_name'])
        except:
            print("Warning: Could not load embedding model. Semantic search will fail.")
        
        # Load FAISS index
        try:
            index_path = os.path.join(self.index_dir, 'index.faiss')
            self.index = faiss.read_index(index_path)
            print(f"Loaded FAISS index with {self.index.ntotal} vectors")
        except:
            print("Warning: Could not load FAISS index.")
        
        # Load metadata
        try:
            metadata_path = os.path.join(self.index_dir, 'metadata.pkl')
            with open(metadata_path, 'rb') as f:
                self.assessments = pickle.load(f)
            print(f"Loaded {len(self.assessments)} assessment metadata")
        except:
            print("Warning: Could not load metadata.")
            self.assessments = []

    def init_bm25(self):
        """Initialize BM25 index"""
        print("Initializing BM25 index...")
        corpus = []
        for assessment in self.assessments:
            # Combine fields for keyword search
            text = f"{assessment['assessment_name']} {assessment.get('description', '')} {assessment.get('test_type', '')}"
            tokenized_text = self._tokenize(text)
            corpus.append(tokenized_text)
        
        if corpus:
            self.bm25 = BM25Okapi(corpus)
            print("BM25 index initialized")
        else:
            print("Warning: No corpus for BM25")

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer for BM25"""
        # Lowercase and split by non-alphanumeric characters
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def search_semantic(self, query: str, top_k: int = 50) -> Dict[str, float]:
        """Perform semantic search and return {url: score} dict"""
        if not self.model or not self.index:
            return {}
            
        # Encode query
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Store results as dict for easy merging
        results = {}
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.assessments):
                url = self.assessments[idx]['url']
                results[url] = float(score)
        
        return results

    def search_bm25(self, query: str, top_k: int = 50) -> Dict[str, float]:
        """Perform BM25 search and return {url: score} dict"""
        if not self.bm25:
            return {}
            
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top K indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = {}
        for idx in top_indices:
            score = scores[idx]
            if score > 0: # Only include relevant results
                url = self.assessments[idx]['url']
                results[url] = float(score)
                
        return results

    def hybrid_search(self, query: str, top_k: int = 30, alpha: float = 0.5) -> List[Tuple[Dict, float]]:
        """
        Perform Hybrid Search (Semantic + BM25)
        
        Args:
            query: Search query
            top_k: Number of results to return
            alpha: Weight for semantic score (0.0 to 1.0). 1.0 = Pure Semantic, 0.0 = Pure BM25
            
        Returns:
            List of (assessment, score) tuples
        """
        # Get raw results
        semantic_results = self.search_semantic(query, top_k=top_k*2)
        bm25_results = self.search_bm25(query, top_k=top_k*2)
        
        # Find all unique URLs
        all_urls = set(semantic_results.keys()) | set(bm25_results.keys())
        
        # Normalize BM25 scores (Semantic cosine sim is already -1 to 1, usually 0-1)
        if bm25_results:
            bm25_vals = list(bm25_results.values())
            min_bm25 = min(bm25_vals)
            max_bm25 = max(bm25_vals)
            range_bm25 = max_bm25 - min_bm25 if max_bm25 > min_bm25 else 1.0
        else:
            min_bm25, range_bm25 = 0, 1
            
        # Combine scores (Reciprocal Rank Fusion or Linear Combination)
        # Using Dictionary-based fusion for now
        final_scores = []
        url_to_assessment = {a['url']: a for a in self.assessments}
        
        for url in all_urls:
            # Semantic score (default 0)
            sem_score = semantic_results.get(url, 0.0)
            
            # Text score (normalized to 0-1)
            raw_bm25 = bm25_results.get(url, 0.0)
            text_score = (raw_bm25 - min_bm25) / range_bm25 if range_bm25 > 0 else 0.0
            
            # Weighted Sum
            final_score = (alpha * sem_score) + ((1 - alpha) * text_score)
            
            if url in url_to_assessment:
                final_scores.append((url_to_assessment[url], final_score))
        
        # Sort by final score
        final_scores.sort(key=lambda x: x[1], reverse=True)
        
        return final_scores[:top_k]
    
    def apply_diversity_constraint(self, results: List[Tuple[Dict, float]], 
                                   final_k: int = 10,
                                   max_per_category: int = 4) -> List[Tuple[Dict, float]]:
        """Re-rank results to ensure diversity"""
        category_counts = defaultdict(int)
        diverse_results = []
        remaining_results = []
        
        for assessment, score in results:
            test_type = assessment.get('test_type', 'General Assessment')
            
            if category_counts[test_type] < max_per_category:
                diverse_results.append((assessment, score))
                category_counts[test_type] += 1
            else:
                remaining_results.append((assessment, score))
            
            if len(diverse_results) >= final_k:
                break
        
        if len(diverse_results) < final_k:
            count = final_k - len(diverse_results)
            diverse_results.extend(remaining_results[:count])
        
        return diverse_results[:final_k]
    
    def filter_by_duration(self, results: List[Tuple[Dict, float]], 
                           max_duration: int = None,
                           min_duration: int = None) -> List[Tuple[Dict, float]]:
        """Filter results by duration constraints"""
        if max_duration is None and min_duration is None:
            return results
        
        filtered = []
        for assessment, score in results:
            duration = assessment.get('duration_minutes')
            if duration is None: # Include unknowns
                filtered.append((assessment, score))
                continue
            
            if max_duration and duration > max_duration: continue
            if min_duration and duration < min_duration: continue
            
            filtered.append((assessment, score))
        
        return filtered
    
    def retrieve(self, query: str, top_k: int = 10, 
                 max_duration: int = None,
                 apply_diversity: bool = True) -> List[Dict]:
        """Main retrieval method with hybrid search"""
        
        # Use Hybrid Search instead of just Semantic
        # Alpha=0.6 favors semantic slightly, but lets strong keyword matches shine
        candidates = self.hybrid_search(query, top_k=min(top_k * 4, 60), alpha=0.6)
        
        if max_duration:
            candidates = self.filter_by_duration(candidates, max_duration=max_duration)
        
        if apply_diversity:
            results = self.apply_diversity_constraint(candidates, final_k=top_k)
        else:
            results = candidates[:top_k]
        
        formatted_results = []
        for assessment, score in results:
            result = assessment.copy()
            result['relevance_score'] = round(score, 4)
            formatted_results.append(result)
        
        return formatted_results
    
    # Backwards compatibility wrapper for simple semantic search
    def search(self, query: str, top_k: int = 30) -> List[Tuple[Dict, float]]:
        """Deprecated: Use hybrid_search instead"""
        return self.hybrid_search(query, top_k=top_k, alpha=1.0)


def test_retrieval():
    """Test retrieval system"""
    print("Testing Hybrid Search System...")
    retriever = RetrievalSystem()
    
    test_cases = [
        {'query': 'Java developer with 3 years experience', 'top_k': 5, 'max_duration': None},
        {'query': 'Entry level sales representative', 'top_k': 5, 'max_duration': 60}
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test['query']}")
        results = retriever.retrieve(**test)
        for j, res in enumerate(results, 1):
            print(f"{j}. {res['assessment_name']} (Score: {res['relevance_score']})")


if __name__ == "__main__":
    test_retrieval()
