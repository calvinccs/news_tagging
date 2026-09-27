"""Main package for news-tagging."""

def main() -> None:
    print("Hello from news-tagging!")
    
# Import the tagging agent so it can be used as a module
from .tagging_agent import tag_article

__all__ = ['tag_article']
