"""
Generate predictions for test set in required CSV format
"""

import pandas as pd
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.recommender import RecommendationEngine


def generate_predictions(output_path: str = 'predictions_test_set.csv', top_k: int = 10):
    """
    Generate predictions for test set queries
    """
    print("="*80)
    print("GENERATING PREDICTIONS")
    print("="*80)
    
    # Initialize recommendation engine
    print("\nInitializing recommendation engine...")
    engine = RecommendationEngine()
    
    # Load dataset
    print("Loading dataset...")
    if not os.path.exists('Gen_AI Dataset.xlsx'):
        print("ERROR: Gen_AI Dataset.xlsx not found!")
        return
        
    df = pd.read_excel('Gen_AI Dataset.xlsx')
    
    # Get unique queries
    unique_queries = df['Query'].unique().tolist()
    print(f"Total unique queries in dataset: {len(unique_queries)}")
    
    # ROBUST LOGIC: 
    # Usually test queries are from index 10 onwards.
    # If the file only has 10, then we predict for all 10.
    if len(unique_queries) > 10:
        test_queries = unique_queries[10:]
        print(f"Found {len(test_queries)} test queries (starting from index 10)")
    else:
        test_queries = unique_queries
        print(f"Only {len(test_queries)} unique queries found. Using ALL of them for predictions.")
    
    # Generate predictions
    predictions_data = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nProcessing {i}/{len(test_queries)}: {query[:70]}...")
        
        # Get recommendations with optimized engine
        # We enable LLM for maximum quality as requested
        result = engine.recommend(query=query, top_k=top_k, use_llm=True)
        
        recommendations = result.get('recommendations', [])
        print(f"  -> Got {len(recommendations)} recommendations")
        
        # Add to predictions list with cleaned/truncated query for readability
        clean_q = " ".join(query.split())[:150] + ("..." if len(query) > 150 else "")
        for rec in recommendations:
            predictions_data.append({
                'Query': clean_q,
                'Assessment_url': rec['url']
            })
    
    # Create and Save
    if not predictions_data:
        print("\nERROR: No predictions were generated!")
        return

    predictions_df = pd.DataFrame(predictions_data)
    
    # Use utf-8-sig for Excel compatibility
    predictions_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"\n{'='*80}")
    print("SUCCESS: PREDICTIONS SAVED")
    print(f"{'='*80}")
    print(f"Total Rows: {len(predictions_df)}")
    print(f"Destination: {os.path.abspath(output_path)}")
    print(f"File Size: {os.path.getsize(output_path)} bytes")
    print(f"\nPreview of first 5 rows:")
    print(predictions_df.head())
    
    return predictions_df


if __name__ == "__main__":
    generate_predictions()
