"""
Merge scraped data from multiple sources and remove duplicates
"""

import json
import pandas as pd
from collections import defaultdict

def merge_assessment_data():
    """Merge all scraped assessment data"""
    
    # Try to load all possible sources
    all_assessments = []
    sources = {}
    
    # Source 1: Current scraped data
    try:
        with open('data/shl_assessments.json', 'r', encoding='utf-8') as f:
            current = json.load(f)
            all_assessments.extend(current)
            sources['current_scrape'] = len(current)
            print(f"✓ Loaded {len(current)} from current scrape")
    except:
        print("✗ No current scrape data found")
    
    # Source 2: Dataset URLs (backup)
    try:
        df = pd.read_excel('Gen_AI Dataset.xlsx')
        dataset_urls = df['Assessment_url'].unique().tolist()
        sources['dataset_urls'] = len(dataset_urls)
        print(f"✓ Found {len(dataset_urls)} unique URLs in dataset")
        
        # For URLs not already scraped, create basic entries
        existing_urls = {a['url'] for a in all_assessments}
        for url in dataset_urls:
            if url not in existing_urls:
                # Create basic assessment entry
                slug = url.rstrip('/').split('/')[-1]
                name = slug.replace('-', ' ').replace('_', ' ').title()
                all_assessments.append({
                    'assessment_name': name,
                    'url': url,
                    'description': 'SHL talent assessment',
                    'duration_minutes': None,
                    'test_type': 'General Assessment',
                    'adaptive_support': 'Unknown',
                    'remote_support': 'Yes'
                })
    except Exception as e:
        print(f"✗ Could not load dataset: {e}")
    
    # Remove duplicates based on URL
    seen_urls = set()
    unique_assessments = []
    
    for assessment in all_assessments:
        url = assessment['url']
        if url not in seen_urls:
            seen_urls.add(url)
            unique_assessments.append(assessment)
    
    print(f"\n{'='*60}")
    print("MERGE RESULTS")
    print(f"{'='*60}")
    print(f"Total unique assessments: {len(unique_assessments)}")
    print(f"Sources used:")
    for source, count in sources.items():
        print(f"  - {source}: {count}")
    
    # Save merged data
    with open('data/shl_assessments.json', 'w', encoding='utf-8') as f:
        json.dump(unique_assessments, f, indent=2, ensure_ascii=False)
    
    df = pd.DataFrame(unique_assessments)
    df.to_csv('data/shl_assessments.csv', index=False, encoding='utf-8')
    
    print(f"\n✓ Saved {len(unique_assessments)} assessments")
    
    # Check against requirement
    print(f"\n{'='*60}")
    if len(unique_assessments) >= 377:
        print(f"✓ SUCCESS: {len(unique_assessments)} assessments (meets 377 requirement)")
    else:
        gap = 377 - len(unique_assessments)
        print(f"⚠ GAP: {len(unique_assessments)} assessments (need {gap} more)")
        print(f"\nPercentage of requirement: {len(unique_assessments)/377*100:.1f}%")
    print(f"{'='*60}")
    
    return unique_assessments

if __name__ == "__main__":
    assessments = merge_assessment_data()
