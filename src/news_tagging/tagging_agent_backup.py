"""Tagging agent that uses local LLM to tag articles with companies and their relationships."""

from typing import List, Dict, Any, Literal
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

def detect_l1(article, companies, profile) -> List[Dict[str, Any]]:
    """Detect L1 tags using the local LLM with schema enforcement."""
    
    # Prepare the prompt for LLM
    prompt = f"""
You are a news tagging assistant that identifies companies mentioned in articles.
Only return JSON with a list of company names that appear directly in the text or are clearly affected by industry trends.

Article Title: {article['title']}
Article Content: {article['content']}

Company List (only use these for tagging):
"""
    
    # Add company information to prompt, but excluding suppliers/customers
    companies_info = []
    for company in companies:
        info = f"- {company['name']} ({company['sector']}): {company['description']}"
        companies_info.append(info)
    
    prompt += "\n".join(companies_info) + "\n\n"
    
    # Add profile instruction
    prompt += f"Profile Instruction: {profile['instruction']}\n"

    try:
        # Initialize instructor-wrapped client
        client = openai.OpenAI(
            base_url="http://127.0.0.1:7749/v1",
            api_key="lm-studio"
        )
        
        # Use instructor to wrap the client for schema enforcement 
        client = instructor.from_openai(client, mode=instructor.Mode.MD_JSON)
        
        # Create the LLM call with schema enforcement 
        response = client.chat.completions.create(
            model="gemma-4-26b-a4b-it-qat",  # As specified in original requirements
            messages=[
                {"role": "user", "content": prompt}
            ],
            response_model=L1Result,
            temperature=0.1,
            max_tokens=5000
        )
        
        l1_tags = response.tags
        
        # Convert to the expected format with company name and reason 
        result = []
        for tag in l1_tags:
            result.append({
                "company": tag.company,
                "relation": "L1",
                "profile_used": profile["name"],
                "reason": tag.reason
            })
            
        return result
        
    except Exception as e:
        # Fallback if LLM call fails
        print(f"LLM call failed: {e}")
        return []

def predict_relationships(subject_company, companies, article) -> List[Dict[str, Any]]:
    """Predict supplier/customer relationships for an L1 company."""
    
    # Find the subject company data
    company_data = next((c for c in companies if c['name'] == subject_company), None)
    if not company_data:
        return []

    # Prepare prompt to predict relationship 
    prompt = f"""
Based on the article content and the company description, identify potential suppliers or customers of {subject_company}.

Article Content: {article['content']}
Company Description: {company_data['description']}

Available Companies (ONLY select from this list):
"""
    for c in companies:
        prompt += f"- {c['name']}\n"
    
    prompt += "\nReturn JSON with a list of companies that are likely customers or suppliers. Do not return descriptions, only company names from the list provided."

    try:
        # Initialize instructor-wrapped client
        client = openai.OpenAI(
            base_url="http://127.0.0.1:7749/v1",
            api_key="lm-studio"
        )
        
        # Use instructor to wrap the client for schema enforcement 
        client = instructor.from_openai(client, mode=instructor.Mode.MD_JSON)

        # Create the LLM call with schema enforcement 
        response = client.chat.completions.create(
            model="gemma-4-26b-a4b-it-qat",  # As specified in original requirements
            messages=[
                {"role": "user", "content": prompt}
            ],
            response_model=List[L2Prediction],
            temperature=0.1,
            max_tokens=5000
        )

        l2_predictions = response
        
        # Convert to expected format and VALIDATE against company list
        result = []
        company_names = {c['name'] for c in companies}
        
        for prediction in l2_predictions:
            if prediction.company in company_names:
                result.append({
                    "company": prediction.company,
                    "relation": "L2",
                    "profile_used": "",  # This will be set later when we combine results
                    "role": prediction.role,
                    "confidence": prediction.confidence,
                    "rationale": prediction.rationale
                })

        return result

    except Exception as e:
        print(f"Relationship prediction failed: {e}")
        return []

def tag_article(article, companies, profile) -> List[Dict[str, Any]]:
    """Main tagging function that combines L1 and L2 tags."""
    
    # Step 1: Detect L1 tags
    l1_tags = detect_l1(article, companies, profile)
    
    # Step 2: For each L1 tag, predict relationships (L2) 
    all_tags = []
    
    # Add the L1 tags first  
    for tag in l1_tags:
        all_tags.append(tag)
        
    # Add L2 tags by predicting supplier/customer relationships
    for l1_tag in l1_tags:
        company_name = l1_tag['company']
        l2_predictions = predict_relationships(company_name, companies, article)
        
        # Add each L2 prediction with proper relation type and profile info  
        for pred in l2_predictions:
            pred["profile_used"] = profile["name"]
            all_tags.append(pred)
    
    return all_tags

if __name__ == "__main__":
    print("Tagging agent module loaded")
