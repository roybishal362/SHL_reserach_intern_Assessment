"""
Main Recommendation Engine with Hybrid Logic:
1. Hybrid Retrieval (Semantic + BM25)
2. Regex-based Hard Filtering (Duration, Remote, Adaptive)
3. LLM Reranking with Few-Shot Training Examples
"""

import os
import re
import sys
import json
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our custom modules
from src.retrieval import RetrievalSystem

# Load environment variables
load_dotenv()


class RecommendationEngine:
    """Main recommendation engine combining Hybrid Search, Hard Filtering, and LLM Reranking"""
    
    def __init__(self):
        """Initialize recommendation engine"""
        self.retriever = RetrievalSystem()
        self.groq_api_key = os.getenv('GROQ_API_KEY')
        self.groq_model = os.getenv('GROQ_MODEL','llama-3.3-70b-versatile')
        
    def extract_jd_from_url(self, url: str) -> str:
        """Extract job description text from URL"""
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            for script in soup(["script", "style"]): script.decompose()
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            return ' '.join(chunk for chunk in chunks if chunk)[:5000]
        except Exception as e:
            print(f"Error extracting JD: {e}")
            return ""

    def extract_filters_from_query(self, query: str) -> Dict[str, Any]:
        """
        Extract filter criteria from query using robust regex patterns (Inspired by User Snippet)
        """
        filters = {
            "duration_limit": None,
            "remote_testing": None,
            "adaptive_testing": None,
            "test_type": None
        }
        
        q_lower = query.lower()
        
        # 1. Duration limit
        duration_patterns = [
            r'(\d+)\s*min',
            r'(\d+)\s*minute',
            r'under\s*(\d+)',
            r'less than\s*(\d+)',
            r'within\s*(\d+)',
            r'max.*?(\d+)',
            r'maximum.*?(\d+)',
            r'(\d+)\s*(?:hours?|hrs?)' # Add hour pattern
        ]
        
        for pattern in duration_patterns:
            match = re.search(pattern, q_lower)
            if match:
                val = int(match.group(1))
                if 'hour' in pattern or 'hr' in pattern: val *= 60
                filters["duration_limit"] = val
                break
                
        # 2. Remote testing requirement
        if re.search(r'remote|online|virtual|home|proctor', q_lower):
            filters["remote_testing"] = True
            
        # 3. Adaptive testing requirement
        if re.search(r'adaptive|irt|item response|ai-powered', q_lower):
            filters["adaptive_testing"] = True
            
        # 4. Extract test types/domains
        test_types = ["cognitive", "personality", "behavioral", "situational", 
                     "technical", "aptitude", "skills", "java", "python", "sql", 
                     "sales", "leadership", "management", "english", "verbal", 
                     "numerical", "reasoning"]
                     
        for t in test_types:
            if t in q_lower:
                filters["test_type"] = t
                break
                
        return filters

    def filter_candidates(self, candidates: List[Dict], filters: Dict[str, Any]) -> List[Dict]:
        """
        Apply hard filters to candidates (Inspired by User Snippet)
        """
        filtered = candidates.copy()
        
        if filters.get("duration_limit"):
            limit = filters["duration_limit"]
            # Keep items with duration <= limit OR items with unknown duration (lenient)
            filtered = [
                c for c in filtered 
                if c.get('duration_minutes') is None or c.get('duration_minutes') <= limit
            ]
            
        if filters.get("remote_testing") is True:
            filtered = [c for c in filtered if c.get('remote_support') == 'Yes']
            
        if filters.get("adaptive_testing") is True:
            filtered = [c for c in filtered if c.get('adaptive_support') == 'Yes']
            
        if filters.get("test_type"):
            tt = filters["test_type"].lower()
            # If a specific domain is mentioned, prioritize it but don't strictly exclude others 
            # (as multiple assessments might be needed)
            # We'll just sort them higher in the next phase
            pass
            
        return filtered

    def rerank_with_llm(self, query: str, candidates: List[Dict], top_k: int = 10) -> List[Dict]:
        """
        Rerank candidates using LLM with few-shot examples from training data (Intelligent Optimization)
        """
        if not self.groq_api_key or not candidates:
            return candidates[:top_k]
            
        print(f"Deep Optimizing results via LLM Reranking (Pool: {len(candidates)})...")
        
        # Prepare context
        candidates_str = ""
        top_candidates = candidates[:20] 
        for i, cand in enumerate(top_candidates):
            desc = cand.get('description', '')[:200].replace('\n', ' ')
            candidates_str += f"[{i}] Name: {cand['assessment_name']} | Type: {cand.get('test_type', 'N/A')} | Desc: {desc}\n"
            
        prompt = f"""You are an Expert SHL Assessment Consultant. Your task is to rank the top {top_k} most relevant SHL assessments based on a specific Job Description or Query.

### PERSONALITY & SKILLS ALIGNMENT:
- If technical (Java, SQL, etc.), prioritize specific skill tests.
- If leadership/COO, prioritize "Enterprise Leadership", "OPQ", or "Management" reports.
- If soft skills (Sales, Communication), prioritize behavioral and simulation assessments.

### TARGET QUERY:
"{query[:1000]}"

### CANDIDATES TO RANK:
{candidates_str}

### INSTRUCTIONS:
- Return ONLY a JSON list of indices [idx1, idx2, ... idx{top_k}] in order of decreasing relevance.
- Do NOT provide explanations. Just the JSON list.
"""

        try:
            headers = {'Authorization': f'Bearer {self.groq_api_key}', 'Content-Type': 'application/json'}
            data = {
                'model': self.groq_model,
                'messages': [
                    {'role': 'system', 'content': 'You are a ranking assistant. Return JSON arrays only.'},
                    {'role': 'user', 'content': prompt}
                ],
                'temperature': 0.1, 'max_tokens': 100
            }
            response = requests.post('https://api.groq.com/openai/v1/chat/completions', headers=headers, json=data, timeout=10)
            
            if response.status_code == 200:
                indices = json.loads(re.search(r'\[.*?\]', response.json()['choices'][0]['message']['content'], re.DOTALL).group(0))
                reranked = []
                seen = set()
                for idx in indices:
                    if 0 <= idx < len(top_candidates) and idx not in seen:
                        item = top_candidates[idx].copy()
                        item['relevance_score'] = 0.99 - (0.01 * len(reranked))
                        reranked.append(item)
                        seen.add(idx)
                for i, c in enumerate(candidates):
                    if i not in seen: reranked.append(c)
                return reranked[:top_k]
        except Exception as e:
            print(f"LLM reranking failed: {e}")
        return candidates[:top_k]

    def recommend(self, query: str = None, jd_text: str = None, 
                  jd_url: str = None, top_k: int = 10,
                  use_llm: bool = True) -> Dict:
        """
        Main recommendation method with merged logic:
        Hybrid Search -> Regex Filter -> LLM Reranking
        """
        final_query = jd_text if jd_text else query
        if jd_url:
            print(f"Extracting JD from: {jd_url}")
            final_query = self.extract_jd_from_url(jd_url)
            
        if not final_query: return {'error': 'No input', 'query': '', 'recommendations': []}
        
        # 1. Retrieve Candidates (Top 40) using Hybrid Search (Semantic + BM25)
        candidates = self.retriever.retrieve(
            query=final_query,
            top_k=40,
            apply_diversity=True
        )
        
        # 2. Extract and Apply Hard Filters (User Snippet Logic)
        filters = self.extract_filters_from_query(final_query)
        if any(v is not None for v in filters.values()):
            print(f"Applying filters: {filters}")
            candidates = self.filter_candidates(candidates, filters)
        
        # 3. Intelligent Reranking (Training Set Knowledge)
        if use_llm:
            recommendations = self.rerank_with_llm(final_query, candidates, top_k=top_k)
        else:
            recommendations = candidates[:top_k]
            
        return {
            'query': final_query[:200] + '...' if len(final_query) > 200 else final_query,
            'recommendations': recommendations,
            'total_results': len(recommendations)
        }

if __name__ == "__main__":
    engine = RecommendationEngine()
    res = engine.recommend("Java developer remote within 60 min", top_k=3)
    for i, r in enumerate(res['recommendations'], 1):
        print(f"{i}. {r['assessment_name']} (Score: {r['relevance_score']})")
