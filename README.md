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

### Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

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

See the SPEC.md file for detailed usage instructions.
