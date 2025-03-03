# Research Paper Analysis Pipeline

This Python application collects, filters, and scores academic papers from arXiv and Semantic Scholar. It uses OpenAI's o1-mini model, specifically designed for specialized reasoning tasks, to perform in-depth analysis of papers across multiple dimensions and generate comprehensive summaries with AI-focused insights.

## Features

- Fetches papers from specific arXiv categories
- Enriches paper data with citation information from Semantic Scholar
- Filters papers based on citation count
- Scores papers using OpenAI models on six dimensions:
  - Novelty: Innovation and originality
  - Impact: Academic and scientific significance
  - Technical Rigor: Methodology and validation
  - Clarity: Presentation and reproducibility
  - Business Value: Commercial potential
  - AI Relevance: Applicability to AI systems
- Computes citation impact using velocity and recency
- Generates detailed summaries for top-ranked papers
- Outputs results in structured JSON format

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the project root with your API keys:
```
OPENAI_API_KEY=your_openai_key_here
SEMANTICSCHOLAR_API_KEY=your_s2_key_here
```

3. Configure the pipeline in `config.py`:
- Adjust arXiv categories
- Set lookback period
- Configure minimum citation threshold
- Adjust model settings
- Modify scoring weights

## Usage

Run the pipeline:
```bash
python main.py
```

The script will:
1. Fetch papers from arXiv
2. Get citation data from Semantic Scholar
3. Filter and score papers
4. Generate summaries for top papers
5. Save results to `research_output.json`

## Configuration

Key settings in `config.py`:
- `arxiv_categories`: List of arXiv category codes
- `lookback_days`: How far back to search
- `min_citations`: Minimum citation threshold
- `analysis_model`: Uses o1-mini model for specialized reasoning and scoring
- `summary_model`: Uses o1-mini model for comprehensive summarization
- `max_output_tokens`: Configured for o1-mini's 65,536 token output capacity
- `score_weights`: Weights for each scoring dimension
- `citation_settings`: Parameters for citation impact:
  - `max_bonus`: Maximum citation score bonus
  - `recency_weight`: Weight for recent citations
  - `citation_velocity_threshold`: Citations/month threshold

## Analysis Tools

The pipeline includes a score analysis script:
```bash
python analyze_scores.py
```

This generates:
- Score distribution visualizations
- Correlation analysis between dimensions
- Detailed statistics on paper rankings
- Visual exports:
  - `score_distribution.png`: Box plots of score distributions
  - `score_correlation.png`: Correlation matrix heatmap

## Output Format

The `research_output.json` file contains:
- Metadata about the analysis run
- List of all analyzed papers with:
  - Title and authors
  - arXiv ID and PDF link
  - Citation metrics (count, velocity, impact)
  - Scores across all dimensions
  - Detailed explanations for each score
  - Key insights:
    - Paper significance
    - AI applications
    - Limitations and challenges
- Detailed summaries for top papers including:
  - Core innovations
  - Technical details
  - AI integration potential
  - Real-world applications
  - Research impact
  - Future directions

## Error Handling

The pipeline includes:
- Retry logic for API calls
- Rate limiting to avoid API throttling
- Validation of configuration
- Graceful handling of missing data

## Dependencies

Core functionality:
- `feedparser`: For parsing arXiv API responses
- `requests`: For API calls
- `openai`: For paper analysis
- `python-dotenv`: For environment variables
- `tenacity`: For retry logic
- `aiohttp`: For async operations

Analysis tools:
- `pandas`: For data manipulation
- `matplotlib`: For plotting
- `seaborn`: For statistical visualizations
