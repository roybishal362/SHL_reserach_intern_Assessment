"""
Evaluation script to calculate Mean Recall@10 on training set
"""

import pandas as pd
import json
from typing import List, Dict
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.recommender import RecommendationEngine


def load_training_data() -> Dict[str,List[str]]:
    """
    Load training queries and their ground truth assessments
    
    Returns:
        Dictionary mapping queries to list of ground truth URLs
    """
    df = pd.read_excel('Gen_AI Dataset.xlsx')
    
    # Get unique queries
    unique_queries = df['Query'].unique().tolist()
    
    # Split into training (first 10) and test (remaining)
    training_queries = unique_queries[:10]
    
    ground_truth = {}
    for query in training_queries:
        query_rows = df[df['Query'] == query]
        urls = query_rows['Assessment_url'].tolist()
        ground_truth[query] = urls
    
    return ground_truth


def get_slug(url: str) -> str:
    """Extract slug from URL for comparison"""
    if pd.isna(url): return ""
    return url.rstrip('/').split('/')[-1].lower()


def calculate_recall_at_k(predictions: List[str], ground_truth: List[str], k: int = 10) -> float:
    """
    Calculate Recall@K using slug matching
    
    Args:
        predictions: List of predicted URLs (top K)
        ground_truth: List of ground truth URLs
        k: K value
        
    Returns:
        Recall@K score
    """
    pred_slugs = set(get_slug(url) for url in predictions[:k])
    gt_slugs = set(get_slug(url) for url in ground_truth)
    
    intersection = pred_slugs.intersection(gt_slugs)
    
    if len(gt_slugs) == 0:
        return 0.0
    
    recall = len(intersection) / len(gt_slugs)
    # Debug print if recall is low but we expected matches
    if recall == 0 and len(gt_slugs) > 0:
        # Check against non-normalized to see if we missed any
        pass
        
    return recall


def evaluate_system(top_k: int = 10):
    """
    Evaluate recommendation system on training set
    
    Args:
        top_k: Number of recommendations to evaluate
    """
    print("="*80)
    print("EVALUATION ON TRAINING SET")
    print("="*80)
    
    # Initialize recommendation engine
    print("\nInitializing recommendation engine...")
    engine = RecommendationEngine()
    
    # Load ground truth
    print("Loading ground truth data...")
    ground_truth = load_training_data()
    print(f"Loaded {len(ground_truth)} training queries\n")
    
    # Evaluate each query
    results = []
    
    for i, (query, gt_urls) in enumerate(ground_truth.items(), 1):
        print(f"\n{'='*80}")
        print(f"Query {i}/{len(ground_truth)}")
        print(f"{'='*80}")
        print(f"Query: {query[:100]}{'...' if len(query) > 100 else ''}")
        print(f"Ground truth assessments: {len(gt_urls)}")
        
        # Get recommendations using intelligent optimization
        result = engine.recommend(query=query, top_k=top_k, use_llm=True)
        
        if 'error' in result:
            print(f"ERROR: {result['error']}")
            results.append({
                'query': query,
                'recall@10': 0.0,
                'precision@10': 0.0,
                'found': 0,
                'total_gt': len(gt_urls)
            })
            continue
        
        recommendations = result['recommendations']
        pred_urls = [rec['url'] for rec in recommendations]
        
        # Calculate recall
        recall = calculate_recall_at_k(pred_urls, gt_urls, k=top_k)
        
        # Calculate how many GT URLs were found (using slugs)
        pred_slugs = set(get_slug(url) for url in pred_urls)
        gt_slugs = set(get_slug(url) for url in gt_urls)
        found_slugs = pred_slugs.intersection(gt_slugs)
        num_found = len(found_slugs)
        
        # Precision = found / top_k
        precision = num_found / top_k
        
        print(f"\nResults:")
        print(f"  Recall@{top_k}: {recall:.4f} ({num_found}/{len(gt_urls)})")
        print(f"  Precision@{top_k}: {precision:.4f}")
        
        if num_found > 0:
            print(f"\n  Found matches:")
            for slug in found_slugs:
                print(f"    ✓ {slug}")
        
        results.append({
            'query': query[:80] + '...' if len(query) > 80 else query,
            'recall@10': recall,
            'precision@10': precision,
            'found': num_found,
            'total_gt': len(gt_urls)
        })
    
    # Calculate summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}\n")
    
    mean_recall = sum(r['recall@10'] for r in results) / len(results)
    mean_precision = sum(r['precision@10'] for r in results) / len(results)
    total_found = sum(r['found'] for r in results)
    total_gt = sum(r['total_gt'] for r in results)
    
    print(f"Mean Recall@{top_k}: {mean_recall:.4f}")
    print(f"Mean Precision@{top_k}: {mean_precision:.4f}")
    print(f"Total found: {total_found}/{total_gt} ({total_found/total_gt*100:.1f}%)")
    
    # Create detailed results DataFrame
    df_results = pd.DataFrame(results)
    
    print(f"\n{'='*80}")
    print("PER-QUERY RESULTS")
    print(f"{'='*80}\n")
    print(df_results.to_string(index=False))
    
    # Save results
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    results_path = f'evaluation_results_{timestamp}.json'
    
    summary = {
        'mean_recall@10': mean_recall,
        'mean_precision@10': mean_precision,
        'total_found': total_found,
        'total_ground_truth': total_gt,
        'num_queries': len(results),
        'per_query_results': results
    }
    
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Evaluation results saved to {results_path}")
    
    return mean_recall, results


if __name__ == "__main__":
    mean_recall, results = evaluate_system(top_k=10)
    
    print(f"\n{'='*80}")
    if mean_recall >= 0.6:
        print("✓ SUCCESS: Mean Recall@10 >= 0.6")
    elif mean_recall >= 0.5:
        print("✓ GOOD: Mean Recall@10 >= 0.5")
    else:
        print("⚠ WARNING: Mean Recall@10 < 0.5 - May need improvement")
    print(f"{'='*80}")
