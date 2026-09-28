# News-to-Company Tagging System — Spec (v1)

## Goal
An agentic system tags synthetic news articles to S&P 500 companies — 
both directly mentioned companies and related companies (suppliers/customers)
— using a configurable tagging profile. A Streamlit dashboard displays tagged
articles and collects reviewer feedback on tag correctness.

Origin: inspired by a gap in Moody's News Edge, which tags only the directly
mentioned company, not related companies that may be materially affected.

## Data Flow
1. **Company list**: a curated subset (~30-50) of S&P 500 companies, static file.
2. **Article generator**: separate model (ChatGPT/Claude), given the company
   list, produces plausible synthetic articles. Outputs article content ONLY —
   no tags, no labels.
3. **Tagging agent**: local LLM (Gemma 4 26B-A4B via LM Studio) reads an
   untagged article + an active tagging profile, and outputs:
   - directly mentioned companies (from the company list)
   - related companies (via relationship prediction), marked as such, not
     conflated with direct mentions
4. **Storage**: SQLite. Stores articles, tags, and feedback.
5. **Dashboard (Streamlit)**: lists tagged articles, filter by company,
   shows which profile produced the tags, and lets a reviewer mark each tag
   correct/incorrect (feedback).

## Data Model

### Article
```json
{
  "id": "string",
  "title": "string",
  "content": "string",
  "source": "string",
  "published_date": "ISO8601"
}
```

### Tag (agent output)
```json
{
  "article_id": "string",
  "company": "string",
  "relation": "L1 | L2",
  "profile_used": "string"
}
```

### Feedback
```json
{
  "tag_id": "string",
  "reviewer_role": "string",
  "correct": true,
  "note": "string (optional)"
}
```
## Data Flow
1. **Company list**: a curated subset (~30-50) of S&P 500 companies, static file.
2. **Relationship map**: static supplier/customer map for ~4-6 of those companies.
3. **Article generator**: separate model (ChatGPT/Claude), given the company
   list, produces plausible synthetic articles. Outputs article content ONLY —
   no tags, no labels.
4. **Tagging agent**: local LLM (Gemma 4 26B-A4B via LM Studio) reads an
   untagged article + an active tagging profile, and outputs:
   - directly mentioned companies (from the company list)
   - related companies (via the relationship map), marked as such, not
     conflated with direct mentions
5. **Storage**: SQLite. Stores articles, tags, and feedback.
6. **Dashboard (Streamlit)**: lists tagged articles, filter by company,
   shows which profile produced the tags, and lets a reviewer mark each tag
   correct/incorrect (feedback).

## Data Model

### Article
```json
{
  "id": "string",
  "title": "string",
  "content": "string",
  "source": "string",
  "published_date": "ISO8601"
}
```

### Tag (agent output)
```json
{
  "article_id": "string",
  "company": "string",
  "relation": "L1 | L2 | L3",
  "profile_used": "string"
}
```

### Feedback
```json
{
  "tag_id": "string",
  "reviewer_role": "string",
  "correct": true,
  "note": "string (optional)"
}
```
## Tag Relation Levels
- **L1 — Direct exposure**: company is directly mentioned, OR clearly
  sector-impacted even without being named (e.g. a geopolitical shipping
  disruption story implicitly affects oil companies).
- **L2 — First-degree relationship**: a direct supplier or customer of an
  L1 company, predicted by the LLM based on business logic from company 
  descriptions.

## Components (v1)
- `data/companies.json` — curated company subset  
- `data/articles/` — synthetic untagged article fixtures
- `profiles/` — tagging profile configs (e.g. `credit_risk.yaml`,
  `sector_relevance.yaml`)
## Tagging Pipeline (src/news_tagging/tagging_agent.py)
Two-step LLM pipeline, plain sequential Python function calls (no agent
framework — no LangGraph/CrewAI — since the pipeline is linear with no
branching or tool use):

1. **detect_l1(article, companies, profile)**: one LLM call. Identifies
   companies with "L1" relevance — either explicitly named in the article,
   or (if no company is named) companies whose OWN sector is directly and
   centrally affected by the event described (e.g. an oil shipping
   disruption directly affects oil & gas companies, but NOT companies
   merely financially exposed via a cost/demand relationship — that's
   handled separately in step 2). Internally the LLM returns a "reason"
   (mentioned / sector_impact) but both surface as the same "L1" tag.

2. **predict_relationships(subject_company, companies, article)**: one LLM
   call PER L1 company. Given that company's description, predicts which
   OTHER tracked companies are plausible customers or suppliers — based on
   business logic from the descriptions, not a lookup table. Each
   prediction includes: role (customer/supplier), confidence
   (high/medium/low), and rationale (free text explaining why). The prompt
   explicitly excludes direct competitors and requires a DIRECT
   relationship only (not a multi-hop chain through an intermediate
   product/material).

3. **tag_article(article, companies, profile)**: orchestrates both steps,
   then deduplicates L2 predictions across multiple L1 sources (if two
   L1 companies both predict the same L2 company, keep the
   higher-confidence one and merge rationale), and excludes any company
   from L2 that's already tagged L1.

L3 (second-degree relationships — suppliers of suppliers) is deliberately
NOT implemented. Document this: it's mechanically simple to add (just
recurse predict_relationships one more hop), but was excluded because
stacking two layers of unconfirmed LLM prediction compounds unreliability
significantly — better resolved with real user feedback data first.

## Tagging profiles (profiles/)
credit_risk.yaml and sector_relevance.yaml both exist, but ONLY
credit_risk.yaml has been used/tested in this version. Document that the
system is designed to support different tagging lenses depending on the
end user/department (credit risk, sector relevance, etc. — this is
client/use-case dependent), but this version only exercises one profile.
- `src/news_tagging/tagging_agent.py` — calls the local LLM, applies active 
  profile, returns tags
- `dashboard/app.py` — single Streamlit app: view, filter, feedback
## Tag Relation Levels
- **L1 — Direct exposure**: company is directly mentioned, OR clearly
  sector-impacted even without being named (e.g. a geopolitical shipping
  disruption story implicitly affects oil companies).
- **L2 — First-degree relationship**: a direct supplier or customer of an
  L1 company, via the static relationship map.
- **L3 — Second-degree relationship**: a supplier-of-a-supplier or
  customer-of-a-customer, one hop beyond L2 — same relationship graph,
  traversed one level further.
  **Documented here for completeness; NOT implemented in v1.** The v1
  relationship map only encodes first-degree (L2) links, so L3 has no
  data to traverse yet.


## Components (v1 — nothing else)
- `data/companies.json` — curated company subset
- `data/relationships.json` — static supplier/customer map
- `data/articles/` — synthetic untagged article fixtures
- `profiles/` — tagging profile configs (e.g. `credit_risk.yaml`,
  `sector_relevance.yaml`)
- `agent/` — calls the local LLM, applies active profile, returns tags
- `storage/` — SQLite read/write
- `dashboard/` — single Streamlit app: view, filter, feedback

## Model Roles
## Deployment
Local only. README covers `uv` setup and `streamlit run` instructions.
No public hosting for v1.

## Explicit Out of Scope for v1
- No in-app accuracy/eval scoring (any eval is a separate, offline, manual
  script — never a feature of the running app)
- No real CI/CD retraining loop — feedback is captured and stored, but
  nothing automatically retrains or updates tagging behavior  
- No full S&P 500 coverage — curated subset only
- No multi-language support
- No auth / multi-user support / role-based access
- No model-comparison tooling
- No "Future Enhancements" section — if it's not in this doc, it's not built

## Future Work (not part of current implementation)
- RAG / retrieval-based candidate narrowing: the current design embeds
  the full company list in every LLM prompt, which works at ~30 companies
  but does not scale to a real-world database (potentially 3000+
  companies) — a RAG-based approach is planned as a SEPARATE follow-up
  project ("news tagging with RAG") to compare side by side, not part of
  this codebase.
- **Tagging agent**: Gemma 4 26B-A4B (LM Studio, non-reasoning/direct mode)
- **Article generator**: ChatGPT or Claude (external, ad hoc — not part of
  the running app)
- **Dev/coding agent**: Qwen3 Coder 30B via Cline (implementation only)

## Deployment
Local only. README covers `uv` setup and `streamlit run` instructions.
No public hosting for v1.

## Explicit Out of Scope for v1
- No in-app accuracy/eval scoring (any eval is a separate, offline, manual
  script — never a feature of the running app)
- No real CI/CD retraining loop — feedback is captured and stored, but
  nothing automatically retrains or updates tagging behavior
- No full S&P 500 coverage — curated subset only
- No full supply-chain data — a handful of hand-picked relationships only
- No manual tag-correction UI beyond simple correct/incorrect feedback
- No tag hierarchies/taxonomies
- No CLI tool
- No multi-language support
- No auth / multi-user support / role-based access
- No model-comparison tooling
- No "Future Enhancements" section — if it's not in this doc, it's not built