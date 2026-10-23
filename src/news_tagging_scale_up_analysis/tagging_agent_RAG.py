"""
RAG version of the news tagging agent.

Architecture
------------
L1:
    Article -> embedding -> FAISS company retrieval -> LLM tagging

L2:
    L1 company -> embedding -> FAISS company retrieval -> LLM
    relationship classification

The embedding layer is deliberately separated from the LLM layer so that
the local BGE model can later be replaced by an HSBC RAG Studio embedding API.

Dependencies
------------
    pip install sentence-transformers faiss-cpu instructor pydantic openai

Expected company fields
-----------------------
    company_name
    sector
    industry
    description

Expected article fields
----------------------
    title
    content

Adjust COMPANY_NAME_FIELD / ARTICLE_CONTENT_FIELD if your JSON uses
different field names.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal
from datetime import datetime, timezone

import faiss
import numpy as np
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer


# ============================================================
# Configuration
# ============================================================

LM_STUDIO_BASE_URL = "http://127.0.0.1:7749/v1"
LLM_MODEL = "gemma-4-26b-a4b-it-qat"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

COMPANIES_FILE = "data/companies.json"
ARTICLES_FILE = "data/articles.json"

COMPANY_NAME_FIELD = "name"
ARTICLE_CONTENT_FIELD = "content"

# L1 retrieval controls
RAG_TOP_K = 50
RAG_MIN_SIMILARITY = 0
RAG_MAX_LLM_CALLS = 25

# L2 retrieval controls
L2_TOP_K = 50
L2_MIN_SIMILARITY = 0.45
L2_MAX_LLM_CALLS = 25

# Cache company embeddings so they are generated only once.
COMPANY_EMBEDDINGS_FILE = "company_embeddings_bge_small.npy"

# Optional cache for the FAISS index.
FAISS_INDEX_FILE = "company_index_bge_small.faiss"


# ============================================================
# Pydantic output schemas
# ============================================================

class L1Tag(BaseModel):
    company: str
    reason: Literal["mentioned", "sector_impact"]


class L1Result(BaseModel):
    tags: list[L1Tag] = Field(default_factory=list)


class L2Prediction(BaseModel):
    company: str
    role: Literal["customer", "supplier"]
    confidence: Literal["high", "medium", "low"]
    rationale: str


class L2Result(BaseModel):
    relationships: list[L2Prediction] = Field(default_factory=list)


# ============================================================
# Clients
# ============================================================

llm_client = instructor.from_openai(
    OpenAI(
        base_url=LM_STUDIO_BASE_URL,
        api_key="lm-studio",
        ),
    mode=instructor.Mode.MD_JSON,
)

embedding_model = SentenceTransformer(EMBEDDING_MODEL)


# ============================================================
# Company text / embeddings
# ============================================================

def company_to_text(company: dict) -> str:
    """Create the text representation used for company embeddings."""

    name = company.get(COMPANY_NAME_FIELD, "")
    sector = company.get("sector", "")
    industry = company.get("industry", "")
    description = company.get("description", "")

    return (
        f"{name}. "
        f"Sector: {sector}. "
        f"Industry: {industry}. "
        f"{description}"
    )


def build_company_embeddings(
    companies: list[dict],
    embeddings_file: str = COMPANY_EMBEDDINGS_FILE,
) -> np.ndarray:
    """
    Generate or load company embeddings.

    IMPORTANT:
    Company embeddings and article/company-query embeddings must use the
    same embedding model.
    """

    path = Path(embeddings_file)

    if path.exists():
        embeddings = np.load(path)

        if len(embeddings) != len(companies):
            raise ValueError(
                f"Cached embeddings contain {len(embeddings)} companies, "
                f"but dataset contains {len(companies)}."
            )

        print(f"Loaded cached company embeddings: {path}")
        return embeddings.astype("float32")

    print(f"Generating embeddings for {len(companies):,} companies...")

    texts = [company_to_text(company) for company in companies]

    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    embeddings = np.asarray(embeddings, dtype="float32")

    np.save(path, embeddings)

    print(f"Saved company embeddings: {path}")

    return embeddings


def build_faiss_index(
    company_embeddings: np.ndarray,
    index_file: str = FAISS_INDEX_FILE,
):
    """Build or load a cosine-similarity FAISS index."""

    path = Path(index_file)

    if path.exists():
        index = faiss.read_index(str(path))

        if index.ntotal != len(company_embeddings):
            raise ValueError(
                f"FAISS index contains {index.ntotal:,} vectors, "
                f"but dataset contains {len(company_embeddings):,} companies."
            )

        print(f"Loaded FAISS index: {path}")
        return index

    embeddings = np.asarray(company_embeddings, dtype="float32")

    # Embeddings are normalized, so inner product == cosine similarity.
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(path))

    print(f"Created FAISS index: {path}")

    return index


# ============================================================
# Retrieval
# ============================================================

def embed_text(text: str) -> np.ndarray:
    """Embed one query and return a normalized float32 vector."""

    vector = embedding_model.encode(
        [text],
        normalize_embeddings=True,
    )

    return np.asarray(vector, dtype="float32")


def retrieve_companies(
    query_text: str,
    companies: list[dict],
    index,
    top_k: int = 50,
    min_similarity: float = 0.50,
    exclude_names: set[str] | None = None,
    max_results: int | None = None,
) -> list[dict]:
    """
    Retrieve candidate companies using semantic similarity.

    Similarity is used for ranking/candidate generation, not as a final
    relevance decision.
    """

    exclude_names = exclude_names or set()

    query_embedding = embed_text(query_text)

    search_k = min(
        top_k + len(exclude_names) + 20,
        len(companies),
    )

    scores, indices = index.search(query_embedding, search_k)

    candidates = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        score = float(score)

        if score < min_similarity:
            continue

        company = companies[int(idx)]

        name = company.get(COMPANY_NAME_FIELD, "")

        if name in exclude_names:
            continue

        result = company.copy()
        result["_similarity"] = score
        result["_retrieval_rank"] = len(candidates) + 1

        candidates.append(result)

        if max_results is not None and len(candidates) >= max_results:
            break

        if max_results is None and len(candidates) >= top_k:
            break

    return candidates


# ============================================================
# L1 tagging
# ============================================================

def detect_l1(
    article: dict,
    companies: list[dict],
    profile: str = "",
) -> list[L1Tag]:
    """
    Existing L1 LLM classifier.

    Only candidates returned by RAG are passed to the LLM.
    """

    article_text = (
        f"Title: {article.get('title', '')}\n\n"
        f"Content: {article.get(ARTICLE_CONTENT_FIELD, '')}"
    )

    company_lines = "\n".join(
        f"- {c.get(COMPANY_NAME_FIELD, '')}: "
        f"{c.get('sector', '')} / "
        f"{c.get('industry', '')} / "
        f"{c.get('description', '')}"
        for c in companies
    )

    prompt = f"""
Identify which companies from the list below are DIRECTLY relevant to this
article.

Definitions:
- "mentioned": the company is explicitly named or referred to by an
  unambiguous nickname.
- "sector_impact": the company is not named, but the article describes an
  event that directly and centrally affects the company's own core
  operations.

Do NOT tag indirect or downstream exposure. For example, if an article is
about an oil disruption, do not tag an airline merely because fuel prices
could rise. Indirect exposure is handled separately.

Only tag a company when you are confident.

Article:
{article_text}

Companies:
{company_lines}

Additional profile instructions:
{profile}
"""

    result = llm_client.chat.completions.create(
        model=LLM_MODEL,
        response_model=L1Result,
        messages=[
            {
                "role": "system",
                "content": "You are a financial news company-impact classifier.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    valid_names = {
        c.get(COMPANY_NAME_FIELD, "")
        for c in companies
    }

    return [
        tag for tag in result.tags
        if tag.company in valid_names
    ]


# ============================================================
# L2 relationship classification
# ============================================================

def predict_relationships(
    subject_company: str,
    candidates: list[dict],
    article: dict,
) -> list[L2Prediction]:
    """
    Determine whether retrieved companies are plausible customers/suppliers
    of the L1 company.

    IMPORTANT:
    Retrieval finds the semantic business neighbourhood. The LLM decides
    whether a real customer/supplier relationship is plausible.
    """

    subject = next(
        (
            c for c in candidates
            if c.get(COMPANY_NAME_FIELD) == subject_company
        ),
        None,
    )

    if subject is None:
        # Subject may not be included in the candidate list because L2
        # candidates have already been filtered. This is handled by passing
        # the subject separately in the L2 wrapper below.
        raise ValueError(
            "Subject company must be supplied separately to the L2 classifier."
        )

    return []


def classify_l2_relationship(
    subject_company: dict,
    candidates: list[dict],
    article: dict,
) -> list[L2Prediction]:
    """LLM classifier for retrieved L2 candidates."""

    article_text = (
        f"Title: {article.get('title', '')}\n\n"
        f"Content: {article.get(ARTICLE_CONTENT_FIELD, '')}"
    )

    subject_text = company_to_text(subject_company)

    company_lines = "\n".join(
        f"- {c.get(COMPANY_NAME_FIELD, '')}: "
        f"{c.get('sector', '')} / "
        f"{c.get('industry', '')} / "
        f"{c.get('description', '')}"
        for c in candidates
    )

    prompt = f"""
Based on the article and the subject company description, identify plausible
CUSTOMERS or SUPPLIERS among the candidate companies.

A relationship must represent a plausible real business relationship.
Do NOT classify competitors merely because they operate in the same industry.

If you are not confident that a customer/supplier relationship exists,
omit the company.

Subject company:
{subject_text}

Article:
{article_text}

Candidate companies:
{company_lines}
"""

    result = llm_client.chat.completions.create(
        model=LLM_MODEL,
        response_model=L2Result,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a financial company relationship classifier. "
                    "Be conservative and only identify plausible "
                    "customer/supplier relationships."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    valid_names = {
        c.get(COMPANY_NAME_FIELD, "")
        for c in candidates
    }

    return [
        prediction
        for prediction in result.relationships
        if prediction.company in valid_names
    ]


# ============================================================
# RAG tagging pipeline
# ============================================================

def tag_article_rag(
    article: dict,
    companies: list[dict],
    company_index,
    profile: str = "",
) -> dict:
    """
    Complete RAG tagging pipeline.

    L1:
        article -> company retrieval -> LLM

    L2:
        L1 company -> company retrieval -> LLM
    """

    article_text = (
        f"{article.get('title', '')}\n"
        f"{article.get(ARTICLE_CONTENT_FIELD, '')}"
    )

    # --------------------------------------------------------
    # L1 retrieval
    # --------------------------------------------------------

    l1_candidates = retrieve_companies(
        article_text,
        companies,
        company_index,
        top_k=RAG_TOP_K,
        min_similarity=RAG_MIN_SIMILARITY,
        max_results=RAG_MAX_LLM_CALLS,
    )

    print(
        f"L1 retrieval: {len(l1_candidates)} candidates "
        f"(top similarity: "
        f"{l1_candidates[0]['_similarity']:.4f}"
        f")"
        if l1_candidates
        else "L1 retrieval: 0 candidates"
    )

    l1_tags = detect_l1(
        article,
        l1_candidates,
        profile,
    )

    l1_company_names = {
        tag.company
        for tag in l1_tags
    }

    # --------------------------------------------------------
    # L2 retrieval + classification
    # --------------------------------------------------------

    all_l2_predictions = []

    company_by_name = {
        c.get(COMPANY_NAME_FIELD): c
        for c in companies
    }

    for l1_tag in l1_tags:
        subject = company_by_name.get(l1_tag.company)

        if subject is None:
            continue

        subject_text = company_to_text(subject)

        # Exclude ALL L1 companies, not just the current subject.
        l2_candidates = retrieve_companies(
            subject_text,
            companies,
            company_index,
            top_k=L2_TOP_K,
            min_similarity=L2_MIN_SIMILARITY,
            exclude_names=l1_company_names,
            max_results=L2_MAX_LLM_CALLS,
        )

        print(
            f"L2 retrieval for {l1_tag.company}: "
            f"{len(l2_candidates)} candidates"
        )

        predictions = classify_l2_relationship(
            subject,
            l2_candidates,
            article,
        )

        all_l2_predictions.extend(predictions)

    # --------------------------------------------------------
    # Deduplicate L2
    # --------------------------------------------------------

    confidence_rank = {
        "high": 3,
        "medium": 2,
        "low": 1,
    }

    l2_by_company = {}

    for prediction in all_l2_predictions:
        # Never return an L1 company as L2.
        if prediction.company in l1_company_names:
            continue

        existing = l2_by_company.get(prediction.company)

        if (
            existing is None
            or confidence_rank[prediction.confidence]
            > confidence_rank[existing.confidence]
        ):
            l2_by_company[prediction.company] = prediction

    return {
        "article": article,
        "l1": [tag.model_dump() for tag in l1_tags],
        "l2": [
            prediction.model_dump()
            for prediction in l2_by_company.values()
        ],
        "rag_metadata": {
            "l1_candidates": len(l1_candidates),
            "l1_candidate_names": [
                c.get(COMPANY_NAME_FIELD)
                for c in l1_candidates
            ],
            "l1_similarities": [
                c.get("_similarity")
                for c in l1_candidates
            ],
        },
    }


# ============================================================
# Saving results
# ============================================================

def save_json(data, path: str):
    """Save data as UTF-8 JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Saved: {output_path}")


def run_batch(
    articles: list[dict],
    companies: list[dict],
    company_index,
    profile: str = "",
    output_dir: str = "results",
):
    """
    Run RAG tagging over a batch of articles and save results.

    Files created:
        results/rag_results.json
            Final L1/L2 tags for every article.

        results/rag_retrieval_results.json
            L1 retrieval candidates, ranks and similarity scores.

        results/rag_run_metadata.json
            Basic run configuration and timing metadata.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = []
    retrieval_results = []

    started_at = datetime.now(timezone.utc)

    for i, article in enumerate(articles, start=1):

        # article_id = article.get("article_id", f"art_{i:03d}")
        article_id = article["id"] 

        print(
            f"\n{'=' * 70}\n"
            f"Article {i}/{len(articles)}: {article_id}\n"
            f"{'=' * 70}"
        )

        result = tag_article_rag(
            article,
            companies,
            company_index,
            profile=profile,
        )

        results.append({
            "article_id": article_id,
            "l1": result["l1"],
            "l2": result["l2"],
        })

        retrieval_results.append({
            "article_id": article_id,
            "l1_candidates": result["rag_metadata"]["l1_candidates"],
            "candidates": [
                {
                    "company": name,
                    "similarity": similarity,
                }
                for name, similarity in zip(
                    result["rag_metadata"]["l1_candidate_names"],
                    result["rag_metadata"]["l1_similarities"],
                )
            ],
        })

        print(
            f"L1 tags: {len(result['l1'])} | "
            f"L2 tags: {len(result['l2'])}"
        )

    finished_at = datetime.now(timezone.utc)

    save_json(
        results,
        output_path / "rag_results.json",
    )

    save_json(
        retrieval_results,
        output_path / "rag_retrieval_results.json",
    )

    save_json(
        {
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "articles": len(articles),
            "companies": len(companies),
            "embedding_model": EMBEDDING_MODEL,
            "llm_model": LLM_MODEL,
            "l1_top_k": RAG_TOP_K,
            "l1_min_similarity": RAG_MIN_SIMILARITY,
            "l1_max_llm_calls": RAG_MAX_LLM_CALLS,
            "l2_top_k": L2_TOP_K,
            "l2_min_similarity": L2_MIN_SIMILARITY,
            "l2_max_llm_calls": L2_MAX_LLM_CALLS,
        },
        output_path / "rag_run_metadata.json",
    )

    return results


# ============================================================
# Data loading
# ============================================================

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# Example main
# ============================================================

if __name__ == "__main__":

    companies = load_json(COMPANIES_FILE)
    articles = load_json(ARTICLES_FILE)

    print(f"Companies: {len(companies):,}")
    print(f"Articles:  {len(articles):,}")

    company_embeddings = build_company_embeddings(companies)
    company_index = build_faiss_index(company_embeddings)

    # Run the full article set and save results.
    run_batch(
        articles=articles,
        companies=companies,
        company_index=company_index,
        profile="",
        output_dir="results",
    )
