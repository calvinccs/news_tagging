#!/usr/bin/env python3
"""
Unit test for the news tagging functionality.
Tests that the basic functions can be imported and work correctly without LLM connection.
"""

import json
from src.news_tagging.tagging_agent import load_companies, load_profiles

def test_data_loading():
    """Test that data loading works properly"""
    print("Testing data loading functions...")
    
    try:
        companies = load_companies()
        profiles = load_profiles()
        
        print(f"✓ Loaded {len(companies)} companies")
        print(f"✓ Loaded {len(profiles)} profiles")
        
        if len(companies) > 0 and len(profiles) > 0:
            print("✓ Data loading successful")
            return True
        else:
            print("✗ No data loaded")
            return False
            
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return False

def test_article_structure():
    """Test that the articles have proper structure"""
    try:
        with open('data/articles.json', 'r') as f:
            articles = json.load(f)
            
        if len(articles) > 0:
            article = articles[0]
            required_fields = ['id', 'title', 'content', 'source', 'published_date']
            
            for field in required_fields:
                if field not in article:
                    print(f"✗ Missing field '{field}' in article")
                    return False
                    
            print("✓ Article structure test passed")
            return True
        else:
            print("✗ No articles found")
            return False
            
    except Exception as e:
        print(f"✗ Error testing article structure: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("Running Unit Tests for News Tagging System")
    print("=" * 50)
    
    success = True
    
    # Test data loading
    if not test_data_loading():
        success = False
        
    # Test article structure  
    if not test_article_structure():
        success = False
        
    print("\n" + "=" * 50)
    if success:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
    print("=" * 50)

if __name__ == "__main__":
    main()