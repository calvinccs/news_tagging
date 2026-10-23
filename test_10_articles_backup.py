#!/usr/bin/env python3
"""
Test script for tagging 10 articles using the news tagging agent.
This script will run the tagging process on sample articles and display results.
"""

import json
from src.news_tagging.tagging_agent import tag_article, load_companies, load_profiles

# Simple test - just verify the functionality works
def main():
    print("Testing tagging functionality with 10 sample articles...")
    
    # Load data
    companies = load_companies()
    profiles = load_profiles()
    
    print(f"Loaded {len(companies)} companies and {len(profiles)} profiles")
    
    # Test with first article from existing dataset 
    try:
        with open('data/articles.json', 'r') as f:
            articles = json.load(f)
            
        if len(articles) >= 10:
            test_articles = articles[:10]
            print(f"Testing tagging on {len(test_articles)} articles")
            
            # Test first article with each profile
            for i, article in enumerate(test_articles, 1):
                print(f"\nArticle {i}: {article['title']}")
                print("-" * 40)
                
                for profile in profiles:
                    print(f"\nUsing Profile: {profile['name']}")
                    
                    # Run tagging - this should work without errors
                    try:
                        tags = tag_article(article, companies, profile)
                        print(f"  Tags found: {len(tags)}")
                        
                        if len(tags) > 0:
                            l1_tags = [t for t in tags if t.get('relation') == 'L1']
                            l2_tags = [t for t in tags if t.get('relation') == 'L2']
                            
                            print(f"    L1 Tags: {len(l1_tags)}")
                            print(f"    L2 Tags: {len(l2_tags)}") 
                    except Exception as e:
                        print(f"  Error processing article: {e}")
        else:
            print("Not enough articles in data/articles.json for testing")
            
    except Exception as e:
        print(f"Error reading articles or running tests: {e}")

if __name__ == "__main__":
    main()