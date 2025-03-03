"""
Configuration settings for the research pipeline.
Contains API keys, model settings, and filtering parameters.
"""

import os
from typing import Dict, Any, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

CONFIG: Dict[str, Any] = {
    # ArXiv settings
    "arxiv_categories": [
        "cs.AI",   # Artificial Intelligence
        "cs.CL",   # Computation and Language
        "cs.CV",   # Computer Vision
        "cs.LG",   # Machine Learning
        "cs.HC",   # Human-Computer Interaction
        "cs.IR",   # Information Retrieval
        "cs.SE",   # Software Engineering
        "cs.CY",   # Computers and Society
        "cs.RO",   # Robotics
        "cs.ET",   # Emerging Technologies
        "cs.SI",   # Social and Information Networks
    ],
    "lookback_days": 45,
    
    # Filtering settings
    "min_citations": 0,  # Temporarily disable citation filtering
    "max_papers": 250,
    
    # API credentials
    "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
    "semanticscholar_api_key": os.getenv("SEMANTICSCHOLAR_API_KEY", ""),
    
    # Email settings
    "email": {
        "smtp_server": os.getenv("SMTP_SERVER", "smtp.gmail.com"),
        "smtp_port": int(os.getenv("SMTP_PORT", "587")),
        "sender_email": os.getenv("SENDER_EMAIL", ""),
        "sender_password": os.getenv("SENDER_PASSWORD", ""),  # App password for Gmail
        "recipient_emails": os.getenv("RECIPIENT_EMAILS", ""),
    },
    
    # OpenAI model settings
    "analysis_model": "o3-mini-2025-01-31",  # Fast and affordable reasoning model for specialized tasks
    "summary_model": "o3-mini-2025-01-31",   # Using same model for consistency
    "max_output_tokens": 32768,              # o1-mini supports up to 65,536 tokens output
    # Note: o1-mini model only supports default temperature (1.0)
    
    # API rate limiting
    "api_delay_seconds": 1,
    
    # Output settings
    "top_papers_count": 15,
    
    # Scoring weights
    "score_weights": {
        "novelty": 0.15,
        "impact": 0.25,
        "technical_rigor": 0.10,
        "clarity": 0.10,
        "business": 0.30,
        "ai_relevance": 0.10
    },
    
    # Citation impact settings
    "citation_settings": {
        "max_bonus": 0.3,  # Maximum citation bonus (30%)
        "recency_weight": 0.6,  # Weight given to recent citations
        "citation_velocity_threshold": 10  # Citations per month threshold for bonus
    },
    
    # Author reputation settings
    "author_settings": {
        "max_bonus": 0.2,  # Maximum author reputation bonus (20%)
        "h_index_weight": 0.6,  # Weight given to h-index
        "citation_weight": 0.4,  # Weight given to citation count
        "h_index_threshold": 20,  # h-index threshold for maximum bonus
        "citation_threshold": 1000,  # Citation count threshold for maximum bonus
        "min_authors_required": 1  # Minimum number of authors required to calculate reputation
    },
    
    # Novelty detection settings
    "novelty_settings": {
        "enabled": True,  # Enable novelty detection
        "reference_lookback_days": 180,  # 6 months of reference papers
        "reference_paper_limit": 1000,  # Max reference papers to use
        "initial_paper_limit": 2000,  # Initial papers to fetch from ArXiv
        "final_paper_limit": 250,  # Papers to pass to full analysis
        "novelty_weight": 0.7,  # Weight for novelty in pre-filtering
        "author_weight": 0.3,  # Weight for author reputation in pre-filtering
        "min_novelty_score": 0.3,  # Minimum novelty score to consider
        "cache_dir": "cache",  # Directory to store embedding cache
        "max_workers": 8,  # Maximum number of parallel workers for embedding fetching
        "max_concurrent_requests": 5  # Maximum number of concurrent async requests
    },
    
    # Performance settings
    "performance": {
        "parallel_processing": True,  # Enable parallel processing
        "max_workers": 8,  # Maximum number of parallel workers
        "use_async": True,  # Use async/await for concurrent processing
        "max_concurrent_requests": 5,  # Maximum number of concurrent requests
        "batch_size": 20  # Batch size for API requests
    }
}

def validate_config() -> None:
    """Validate the configuration settings."""
    required_keys = [
        "openai_api_key"  # Only require OpenAI key for now
    ]
    
    for key in required_keys:
        if not CONFIG.get(key):
            raise ValueError(f"Missing required configuration: {key}")
            
    if not isinstance(CONFIG["arxiv_categories"], list):
        raise ValueError("arxiv_categories must be a list")
        
    if not isinstance(CONFIG["lookback_days"], int) or CONFIG["lookback_days"] <= 0:
        raise ValueError("lookback_days must be a positive integer")
        
    weights = CONFIG["score_weights"]
    if sum(weights.values()) != 1.0:
        raise ValueError("Score weights must sum to 1.0")
