# News Tagging System

A system for categorizing and organizing news articles through automated and manual tagging.

## Project Structure

```
news-tagging/
├── src/news_tagging/          # Main source code
├── frontend/                  # Dashboard UI files (salvaged pieces)
├── dashboard/                 # Additional dashboard components  
├── data/                      # Raw data and fixtures
├── fixtures/                  # Tagged data exports (salvaged pieces)
├── SPEC.md                    # Project specification
├── pyproject.toml             # Project configuration
└── README.md                  # This file
```

## Setup

### Prerequisites
- Python 3.12+
- LM Studio with Gemma 4 26B-A4B model running on http://127.0.0.1:7749/v1

### Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install in development mode with dev dependencies
pip install -e .[dev]
```

## Development

### Running Tests
```bash
pytest
```

### Code Formatting  
```bash
black .
```

### Linting
```bash
flake8 src/
```

### Type Checking
```bash
mypy src/
```

## Usage

The main CLI command is:
```bash
news-tagging
```

For batch tagging of all articles:
1. Run the tagging script to process all synthetic articles with your chosen profile
2. The results will be stored in `data/batch_results.json` and `data/feedback.json`
3. View tagged results in the Streamlit dashboard: 
   ```bash
   streamlit run dashboard/app.py
   ```

## Design Decisions

1. **Predicted vs Static Relationships**: Relationships are predicted by the LLM rather than using a static relationship map because:
   - Static relationships quickly become stale and unrealistic for banks to maintain
   - LLM-based prediction allows for dynamic, context-aware relationships based on company descriptions
   - This approach is more scalable and adaptable to changing business conditions

2. **Reasoning Mode Disabled**: The reasoning mode is disabled in the Gemma 4 model because:
   - It was consuming the full token budget on internal reasoning and returning empty responses 
   - While this causes some precision loss on edge cases, it provides a large speed gain (~5 min -> under 1 min per article)
   - The tradeoff is acceptable for the improved performance

3. **RAG Out of Scope**: Retrieval-Augmented Generation (RAG) is not included because:
   - Current design embeds full company list in every LLM prompt, which works well at ~30 companies 
   - This doesn't scale to real-world databases with potentially 3000+ companies
   - A RAG-based approach is planned as a separate follow-up project ("news tagging with RAG")

4. **Moody's News Edge Motivation**: The system was inspired by Moody's News Edge which only tags the company directly mentioned in an article, not related companies (suppliers/customers) that may also be materially affected. This project extends beyond that to capture both direct and indirect impacts.