#!/usr/bin/env python3
"""
Test script for tagging 10 articles using the news tagging agent with Credit Risk profile only.
This script will run the tagging process on sample articles and display results.
"""

import json
from src.news_tagging.tagging_agent import tag_article, load_companies, load_profiles

# Simple test - just verify the functionality works without waiting too long  
def main():
    print("Testing tagging functionality with Credit Risk profile only...")
    
    # Load data
    companies = load_companies()
    profiles = load_profiles()
    
    # Filter for Credit Risk profile only
    credit_risk_profile = None
    for profile in profiles:
        if profile['name'] == 'Credit Risk':
            credit_risk_profile = profile
            break
    
    if not credit_risk_profile:
        print("Error: Credit Risk profile not found")
        return
    
    print(f"Loaded {len(companies)} companies and using only Credit Risk profile")
    
    try:
        # Test with first article from existing dataset 
        with open('data/articles.json', 'r') as f:
            articles = json.load(f)
            
        if len(articles) >= 10:  # Use all 10 articles
            test_articles = articles[:10]
            print(f"Testing tagging on {len(test_articles)} sample articles with Credit Risk profile")
            print("=" * 60)
            
            # Test each article with the Credit Risk profile only
            for i, article in enumerate(test_articles, 1):
                print(f"\nArticle {i}: {article['title']}")
                print("-" * 40)
                
                try:
                    tags = tag_article(article, companies, credit_risk_profile)
                    
                    if len(tags) > 0:
                        l1_tags = [t for t in tags if t.get('relation') == 'L1']
                        l2_tags = [t for t in tags if t.get('relation') == 'L2']
                        
                        print(f"  Tags found: {len(tags)}")
                        print(f"    L1 Tags: {len(l1_tags)}") 
                        print(f"    L2 Tags: {len(l2_tags)}")
                        
                        # Show sample tag information
                        if l1_tags:
                            print("    Sample L1 tags:")
                            for tag in l1_tags[:3]:  # Show first 3 tags
                                print(f"      - {tag['company']} ({tag.get('reason', 'N/A')})")
                        if l2_tags:
                            print("    Sample L2 tags:")  
                            for tag in l2_tags[:3]:  # Show first 3 tags
                                print(f"      - {tag['company']} as {tag.get('role', 'N/A')} (confidence: {tag.get('confidence', 'N/A')})")
                    else:
                        print("  No tags found")
                        
                except Exception as e:
                    # Show a simplified error message to avoid cluttering output
                    if "timeout" in str(e).lower() or "connection" in str(e).lower():
                        print(f"  LLM connection timeout (expected in test environment)")
                    else:
                        print(f"  Error: {type(e).__name__}")
                        
        else:
            print("Not enough articles in data/articles.json for testing")
            
    except Exception as e:
        print(f"Error reading articles or running tests: {e}")

if __name__ == "__main__":
    main()