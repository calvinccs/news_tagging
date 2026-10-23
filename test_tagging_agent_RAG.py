#!/usr/bin/env python3
"""
Test script for the RAG tagging agent.
Runs the full pipeline on a few sample articles and prints retrieval
diagnostics plus L1/L2 tags.
"""

import traceback

# Adjust this import if the file lives somewhere else.
import src.news_tagging_scale_up_analysis.tagging_agent_RAG as rag

NUM_ARTICLES = 3

# Temporary override while debugging: disable the similarity floors so
# retrieval is top-k only. Set back to None to use the file's own values.
DEBUG_MIN_SIMILARITY = 0.0


def main():
    print(f"Testing RAG tagging with {NUM_ARTICLES} sample articles...")

    if DEBUG_MIN_SIMILARITY is not None:
        rag.RAG_MIN_SIMILARITY = DEBUG_MIN_SIMILARITY
        rag.L2_MIN_SIMILARITY = DEBUG_MIN_SIMILARITY

    # ---- Load data -------------------------------------------------
    companies = rag.load_json(rag.COMPANIES_FILE)
    articles = rag.load_json(rag.ARTICLES_FILE)
    print(f"Loaded {len(companies)} companies and {len(articles)} articles")

    # ---- Field sanity checks --------------------------------------
    print("\nCompany keys:", list(companies[0].keys()))
    print("Article keys:", list(articles[0].keys()))

    name_field = rag.COMPANY_NAME_FIELD
    if name_field not in companies[0]:
        print(f"\nERROR: companies have no '{name_field}' field. "
              f"Set COMPANY_NAME_FIELD in tagging_agent_RAG.py to one of "
              f"the keys above.")
        return

    # ---- Build embeddings and index --------------------------------
    # If you changed the company file, delete the cached .npy and
    # .faiss files first so these are rebuilt.
    embeddings = rag.build_company_embeddings(companies)
    index = rag.build_faiss_index(embeddings)
    print(f"Index size: {index.ntotal}")

    test_articles = articles[:NUM_ARTICLES]
    print("=" * 60)

    # ---- Run each article -----------------------------------------
    for i, article in enumerate(test_articles, 1):
        print(f"\nArticle {i}: {article['title']}")
        print("-" * 40)

        # Retrieval diagnostics: top 5 candidates and similarity scores
        text = f"{article['title']}\n{article[rag.ARTICLE_CONTENT_FIELD]}"
        cands = rag.retrieve_companies(
            text, companies, index, top_k=5, min_similarity=0.0
        )
        print("Top retrieval candidates:")
        for c in cands:
            print(f"  {c['_similarity']:.3f}  "
                  f"{c.get(name_field)!r} | {c.get('sector')}")

        # Full pipeline
        try:
            result = rag.tag_article_rag(
                article, companies, index, profile=""
            )
            l1, l2 = result["l1"], result["l2"]

            if l1 or l2:
                print(f"\n  Tags found: {len(l1) + len(l2)}")
                print(f"    L1 Tags: {len(l1)}")
                for t in l1:
                    print(f"      {t['company']} ({t['reason']})")
                print(f"    L2 Tags: {len(l2)}")
                for t in l2:
                    print(f"      {t['company']} - {t['role']} "
                          f"({t['confidence']})")
            else:
                print("\n  No tags found")

        except Exception:
            print("\n  ERROR during tagging:")
            traceback.print_exc()


if __name__ == "__main__":
    main()