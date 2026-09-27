"""Tagging agent that uses local LLM to tag articles with companies and their relationships."""

from typing import List, Dict, Any, Literal
import json
import yaml
from pydantic import BaseModel
import instructor
import openai
from openai import OpenAI

# Define the Pydantic models for schema enforcement
class L1Tag(BaseModel):
    company: str
    reason: Literal["mentioned", "sector_impact"]

class L1Result(BaseModel):
    tags: List[L1Tag]

class L2Prediction(BaseModel):
    company: str
    role: Literal["customer", "supplier"]
    confidence: Literal["high", "medium", "low"]
    rationale: str

# def detect_l1(article, companies, profile) -> List[Dict[str, Any]]:
#     """Detect L1 tags using the local LLM with schema enforcement."""
    
#     # Prepare the prompt for LLM
#     prompt = f"""
# You are a news tagging assistant that identifies companies mentioned in articles.
# Only return JSON with a list of company names that appear directly in the text or are clearly affected by industry trends.

# Article Title: {article['title']}
# Article Content: {article['content']}

# Company List (only use these for tagging):
# """
    
#     # Add company information to prompt, but excluding suppliers/customers
#     companies_info = []
#     for company in companies:
#         info = f"- {company['name']} ({company['sector']}): {company['description']}"
#         companies_info.append(info)
    
#     prompt += "\n".join(companies_info) + "\n\n"
    
#     # Add profile instruction
#     prompt += f"Profile Instruction: {profile['instruction']}\n"

#     try:
#         # Initialize instructor-wrapped client
#         client = openai.OpenAI(
#             base_url="http://127.0.0.1:7749/v1",
#             api_key="lm-studio"
#         )
        
#         # Use instructor to wrap the client for schema enforcement 
#         client = instructor.from_openai(client, mode=instructor.Mode.MD_JSON)
        
#         # Create the LLM call with schema enforcement 
#         response = client.chat.completions.create(
#             model="gemma-4-26b-a4b-it-qat",  # As specified in original requirements
#             messages=[
#                 {"role": "user", "content": prompt}
#             ],
#             response_model=L1Result,
#             temperature=0.1,
#             max_tokens=5000
#         )
        
#         l1_tags = response.tags
        
#         # Convert to the expected format with company name and reason 
#         result = []
#         for tag in l1_tags:
#             result.append({
#                 "company": tag.company,
#                 "relation": "L1",
#                 "profile_used": profile["name"],
#                 "reason": tag.reason
#             })
            
#         return result
        
#     except Exception as e:
#         # Fallback if LLM call fails
#         print(f"LLM call failed: {e}")
#         return []

def detect_l1(article, companies, profile) -> List[Dict[str, Any]]:
    """Detect L1 tags using the local LLM with schema enforcement."""

    companies_info = []
    for company in companies:
        info = f"- {company['name']} ({company['sector']}): {company['description']}"
        companies_info.append(info)

    prompt = f"""
You are a news tagging assistant. Identify which companies from the list below
are DIRECTLY relevant to this article.

Article Title: {article['title']}
Article Content: {article['content']}

Company List (only use these for tagging):
{chr(10).join(companies_info)}

For each company, use one of two tags:

1. "mentioned" — the company (or an unambiguous name/nickname for it) is
   explicitly named in the article text.

2. "sector_impact" — the company is NOT named, but the article describes
   an event that DIRECTLY and CENTRALLY affects that company's OWN core
   business operations (e.g. a war disrupting oil tanker shipping directly
   affects oil & gas companies' own extraction/shipping operations).

IMPORTANT: Do NOT use "sector_impact" for a company that is only
indirectly or financially exposed through a cost, demand, or supply
relationship with a directly-affected company. Example: in an oil
shipping disruption story, oil & gas companies ARE "sector_impact"
(their own operations are hit) — but airlines are NOT "sector_impact"
just because they'll face higher fuel costs. That kind of downstream
exposure is handled by a separate step; do not tag it here.

Only tag a company if you are confident it clearly fits one of the two
definitions above. When in doubt, do not tag it.

{profile['instruction']}
"""

    try:
        client = openai.OpenAI(base_url="http://127.0.0.1:7749/v1", api_key="lm-studio")
        client = instructor.from_openai(client, mode=instructor.Mode.JSON_SCHEMA)

        response = client.chat.completions.create(
            model="gemma-4-26b-a4b-it-qat",
            messages=[{"role": "user", "content": prompt}],
            response_model=L1Result,
            temperature=0.1,
            max_tokens=5000
        )

        return [
            {"company": tag.company, "relation": "L1",
             "profile_used": profile["name"], "reason": tag.reason}
            for tag in response.tags
        ]
    except Exception as e:
        print(f"LLM call failed: {e}")
        return []

# def predict_relationships(subject_company, companies, article) -> List[Dict[str, Any]]:
#     """Predict supplier/customer relationships for an L1 company."""
    
#     # Find the subject company data
#     company_data = next((c for c in companies if c['name'] == subject_company), None)
#     if not company_data:
#         return []

#     # Prepare prompt to predict relationship 
#     prompt = f"""
# Based on the article content and the company description, identify potential suppliers or customers of {subject_company}.

# Article Content: {article['content']}
# Company Description: {company_data['description']}

# Available Companies (ONLY select from this list):
# """
#     for c in companies:
#         prompt += f"- {c['name']}\n"
    
#     prompt += "\nReturn JSON with a list of companies that are likely customers or suppliers. Do not return descriptions, only company names from the list provided."

#     try:
#         # Initialize instructor-wrapped client
#         client = openai.OpenAI(
#             base_url="http://127.0.0.1:7749/v1",
#             api_key="lm-studio"
#         )
        
#         # Use instructor to wrap the client for schema enforcement 
#         client = instructor.from_openai(client, mode=instructor.Mode.MD_JSON)

#         # Create the LLM call with schema enforcement 
#         response = client.chat.completions.create(
#             model="gemma-4-26b-a4b-it-qat",  # As specified in original requirements
#             messages=[
#                 {"role": "user", "content": prompt}
#             ],
#             response_model=List[L2Prediction],
#             temperature=0.1,
#             max_tokens=5000
#         )

#         l2_predictions = response
        
#         # Convert to expected format and VALIDATE against company list
#         result = []
#         company_names = {c['name'] for c in companies}
        
#         for prediction in l2_predictions:
#             if prediction.company in company_names:
#                 result.append({
#                     "company": prediction.company,
#                     "relation": "L2",
#                     "profile_used": "",  # This will be set later when we combine results
#                     "role": prediction.role,
#                     "confidence": prediction.confidence,
#                     "rationale": prediction.rationale
#                 })

#         return result

#     except Exception as e:
#         print(f"Relationship prediction failed: {e}")
#         return []

def predict_relationships(subject_company, companies, article) -> List[Dict[str, Any]]:
    """Predict supplier/customer relationships for an L1 company."""
    company_data = next((c for c in companies if c['name'] == subject_company), None)
    if not company_data:
        return []

    candidates_info = []
    for c in companies:
        if c['name'] == subject_company:
            continue  # exclude self
        candidates_info.append(f"- {c['name']} ({c['sector']}): {c['description']}")

    prompt = f"""
Based on the article content and {subject_company}'s business description,
identify which OTHER companies from the list below are plausible CUSTOMERS
or SUPPLIERS of {subject_company} — i.e., companies that would realistically
buy from or sell to {subject_company} as part of normal business operations.

Article Content: {article['content']}
{subject_company} Description: {company_data['description']}

Candidate Companies (ONLY select from this list):
{chr(10).join(candidates_info)}

IMPORTANT:
- Only tag a REAL business relationship (one company's product/service is
  something the other actually buys or sells), based on what each company's
  description says it does.
- Do NOT tag companies that do the SAME kind of business as {subject_company}
  (competitors) — a competitor is not a customer or supplier, even if they
  operate in a related industry.
- If you are not confident a company is a plausible customer or supplier,
  do not include it.

Return JSON with a list of {{company, role ("customer" or "supplier"),
confidence, rationale}}.
"""

    try:
        client = openai.OpenAI(base_url="http://127.0.0.1:7749/v1", api_key="lm-studio")
        client = instructor.from_openai(client, mode=instructor.Mode.JSON_SCHEMA)

        response = client.chat.completions.create(
            model="gemma-4-26b-a4b-it-qat",
            messages=[{"role": "user", "content": prompt}],
            response_model=List[L2Prediction],
            temperature=0.1,
            max_tokens=5000
        )

        company_names = {c['name'] for c in companies} - {subject_company}
        return [
            {"company": p.company, "relation": "L2", "profile_used": "",
             "role": p.role, "confidence": p.confidence, "rationale": p.rationale}
            for p in response if p.company in company_names
        ]
    except Exception as e:
        print(f"Relationship prediction failed: {e}")
        return []

# def tag_article(article, companies, profile) -> List[Dict[str, Any]]:
#     """Main tagging function that combines L1 and L2 tags."""
    
#     # Step 1: Detect L1 tags
#     l1_tags = detect_l1(article, companies, profile)
    
#     # Step 2: For each L1 tag, predict relationships (L2) 
#     all_tags = []
    
#     # Add the L1 tags first  
#     for tag in l1_tags:
#         all_tags.append(tag)
        
#     # Add L2 tags by predicting supplier/customer relationships
#     for l1_tag in l1_tags:
#         company_name = l1_tag['company']
#         l2_predictions = predict_relationships(company_name, companies, article)
        
#         # Add each L2 prediction with proper relation type and profile info  
#         for pred in l2_predictions:
#             pred["profile_used"] = profile["name"]
#             all_tags.append(pred)
    
#     return all_tags

CONFIDENCE_RANK = {"high": 3, "medium": 2, "low": 1}

def tag_article(article, companies, profile) -> List[Dict[str, Any]]:
    """Main tagging function that combines L1 and L2 tags."""
    l1_tags = detect_l1(article, companies, profile)

    all_tags = list(l1_tags)
    l1_company_names = {t["company"] for t in l1_tags}

    l2_by_company: Dict[str, Dict[str, Any]] = {}

    for l1_tag in l1_tags:
        predictions = predict_relationships(l1_tag["company"], companies, article)
        for pred in predictions:
            name = pred["company"]

            # Skip anything already tagged L1 — don't downgrade it to L2 too
            if name in l1_company_names:
                continue

            pred["profile_used"] = profile["name"]

            existing = l2_by_company.get(name)
            if existing is None:
                l2_by_company[name] = pred
            else:
                # Keep whichever has higher confidence; merge rationale for context
                if CONFIDENCE_RANK.get(pred["confidence"], 0) > CONFIDENCE_RANK.get(existing["confidence"], 0):
                    pred["rationale"] = f"{pred['rationale']} (also suggested via {existing['rationale']})"
                    l2_by_company[name] = pred
                else:
                    existing["rationale"] = f"{existing['rationale']} (also suggested via {pred['rationale']})"

    all_tags.extend(l2_by_company.values())
    return all_tags


def load_companies() -> List[Dict[str, Any]]:
    """Load company data from JSON file."""
    with open('data/companies.json', 'r') as f:
        return json.load(f)

def load_profiles() -> List[Dict[str, Any]]:
    """Load profile data from YAML files."""
    profiles = []
    import os
    for filename in os.listdir('profiles/'):
        if filename.endswith('.yaml'):
            with open(os.path.join('profiles/', filename), 'r') as f:
                profile = yaml.safe_load(f)
                profile['name'] = profile['name']  # Ensure name is set
                profiles.append(profile)
    
    return profiles

if __name__ == "__main__":
    print("Tagging agent module loaded")

if __name__ == "__main__":
    print("Tagging agent module loaded")
