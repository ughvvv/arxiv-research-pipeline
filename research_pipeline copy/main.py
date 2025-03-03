"""
Main script for the research paper analysis pipeline.
"""

import json
import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Dict, Any
from datetime import datetime, timedelta
from config import CONFIG, validate_config
from arxiv_client import ArxivClient
from semanticscholar_client import SemanticScholarClient
from openai_utils import OpenAIAnalyzer
from novelty_detector import NoveltyDetector

def filter_papers(papers: List[Dict[str, Any]], min_citations: int) -> List[Dict[str, Any]]:
    """Filter papers based on citation count if available."""
    filtered_papers = []
    for paper in papers:
        # If citation data is available and min_citations > 0, use it for filtering
        if min_citations > 0 and "citation_count" in paper:
            if paper["citation_count"] >= min_citations:
                filtered_papers.append(paper)
        else:
            # If no citation requirement or no citation data, keep the paper
            filtered_papers.append(paper)
    return filtered_papers

def compute_citation_impact(citation_count: int, publication_date: str, citation_settings: Dict[str, float]) -> float:
    """
    Compute citation impact score considering recency and velocity.
    
    Args:
        citation_count: Number of citations
        publication_date: Paper publication date (ISO format)
        citation_settings: Dictionary containing citation impact parameters
    
    Returns:
        Citation impact score between 0.0 and max_bonus
    """
    if not citation_count:
        return 0.0
        
    try:
        pub_date = datetime.fromisoformat(publication_date)
        months_since_pub = (datetime.now() - pub_date).days / 30.0
        
        if months_since_pub <= 0:
            return 0.0
            
        # Calculate citation velocity (citations per month)
        citation_velocity = citation_count / months_since_pub
        
        # Apply recency weight - more recent papers get more credit for their citations
        recency_factor = max(0.0, 1.0 - (months_since_pub / 24.0))  # Linear decay over 2 years
        velocity_factor = min(1.0, citation_velocity / citation_settings["citation_velocity_threshold"])
        
        # Combine factors with recency weight
        impact_score = (
            citation_settings["recency_weight"] * recency_factor * velocity_factor +
            (1 - citation_settings["recency_weight"]) * velocity_factor
        )
        
        # Scale to max bonus
        return impact_score * citation_settings["max_bonus"]
        
    except (ValueError, TypeError):
        print(f"Warning: Invalid publication date format: {publication_date}")
        return 0.0

def compute_author_reputation(authors_data: List[Dict[str, Any]], author_settings: Dict[str, float]) -> float:
    """
    Compute author reputation score based on author metrics.
    
    Args:
        authors_data: List of author data dictionaries
        author_settings: Dictionary containing author reputation parameters
    
    Returns:
        Author reputation score between 0.0 and max_bonus
    """
    if not authors_data or len(authors_data) < author_settings.get("min_authors_required", 1):
        return 0.0
    
    # Calculate average h-index and citation count
    avg_h_index = sum(author.get("h_index", 0) for author in authors_data) / len(authors_data)
    avg_citation_count = sum(author.get("citation_count", 0) for author in authors_data) / len(authors_data)
    
    # Normalize metrics
    h_index_factor = min(1.0, avg_h_index / author_settings["h_index_threshold"])
    citation_factor = min(1.0, avg_citation_count / author_settings["citation_threshold"])
    
    # Combine factors with weights
    reputation_score = (
        author_settings["h_index_weight"] * h_index_factor +
        author_settings["citation_weight"] * citation_factor
    )
    
    # Scale to max bonus
    return reputation_score * author_settings["max_bonus"]

def compute_final_score(
    scores: Dict[str, float],
    weights: Dict[str, float],
    citation_impact: float = 0.0,
    author_reputation: float = 0.0
) -> float:
    """
    Compute weighted average of paper scores with citation impact and author reputation.
    
    Args:
        scores: Dictionary of component scores
        weights: Dictionary of component weights
        citation_impact: Impact score based on citations and recency
        author_reputation: Reputation score based on author metrics
    """
    base_score = sum(scores[key] * weights[key] for key in weights)
    return base_score * (1.0 + citation_impact + author_reputation)

def main():
    """Run the main paper analysis pipeline."""
    # Validate configuration
    validate_config()
    
    # Get novelty and performance settings
    novelty_settings = CONFIG.get("novelty_settings", {})
    novelty_enabled = novelty_settings.get("enabled", False)
    
    # Get performance settings
    performance_settings = CONFIG.get("performance", {})
    parallel_processing = performance_settings.get("parallel_processing", False)
    max_workers = performance_settings.get("max_workers", 5)
    use_async = performance_settings.get("use_async", False)
    max_concurrent_requests = performance_settings.get("max_concurrent_requests", 5)
    
    # Initialize clients
    arxiv_client = ArxivClient(
        categories=CONFIG["arxiv_categories"],
        lookback_days=CONFIG["lookback_days"]
    )
    
    openai_analyzer = OpenAIAnalyzer(
        api_key=CONFIG["openai_api_key"],
        analysis_model=CONFIG["analysis_model"],
        summary_model=CONFIG["summary_model"]
    )
    
    # Initialize Semantic Scholar client only if API key is provided
    s2_client = None
    novelty_detector = None
    if CONFIG.get("semanticscholar_api_key"):
        s2_client = SemanticScholarClient(
            api_key=CONFIG["semanticscholar_api_key"],
            delay_seconds=CONFIG["api_delay_seconds"]
        )
        
        # Initialize novelty detector if enabled
        if novelty_enabled:
            # Add performance settings to novelty config
            novelty_config = novelty_settings.copy()
            if parallel_processing:
                novelty_config["max_workers"] = max_workers
                novelty_config["max_concurrent_requests"] = max_concurrent_requests
                
            novelty_detector = NoveltyDetector(
                s2_client=s2_client,
                cache_dir=novelty_settings.get("cache_dir", "cache"),
                config=novelty_config
            )
    
    # Determine how many papers to fetch based on novelty settings
    max_papers = CONFIG["max_papers"]
    if novelty_enabled and novelty_detector:
        max_papers = novelty_settings.get("initial_paper_limit", 2000)
        print(f"Novelty detection enabled, fetching {max_papers} papers for pre-filtering")
    
    # Fetch papers from ArXiv
    print("Fetching papers from ArXiv...")
    papers = arxiv_client.fetch_papers(max_results=max_papers)
    print(f"Found {len(papers)} papers")
    
    # Try to get citation data if Semantic Scholar client is available
    if s2_client:
        print("\nAttempting to fetch citation data from Semantic Scholar...")
        arxiv_ids = [paper["arxiv_id"] for paper in papers]
        s2_results = s2_client.fetch_papers_batch(arxiv_ids)
        
        # Update papers with citation data where available
        for paper, s2_data in zip(papers, s2_results):
            if s2_data:
                # First, save the ArXiv publication date if available
                arxiv_date = None
                if "published" in paper:
                    # Save the ArXiv publication date
                    arxiv_date = paper["published"]
                    
                # Update paper with Semantic Scholar data
                paper.update(s2_data)
                
                # Restore the ArXiv publication date if it was available
                if arxiv_date:
                    paper["publication_date"] = arxiv_date
                    
                # Extract Semantic Scholar ID if available
                if "url" in paper and "semanticscholar.org/paper/" in paper["url"]:
                    paper["s2_id"] = paper["url"].split("semanticscholar.org/paper/")[1].split("/")[0]
    else:
        print("\nSkipping citation data (Semantic Scholar API key not configured)")
    
    # Apply novelty-based filtering if enabled
    if novelty_enabled and novelty_detector and s2_client:
        print("\nApplying novelty-based filtering...")
        
        # Build reference corpus if needed
        if not novelty_detector.reference_embeddings:
            print("Building reference corpus...")
            # For simplicity, use the same papers as reference corpus
            # In a production system, you would fetch a separate set of reference papers
            reference_count = novelty_detector.build_reference_corpus(papers)
            print(f"Built reference corpus with {reference_count} papers")
        
        # Fetch author information for papers
        print("\nFetching author information from Semantic Scholar...")
        for i, paper in enumerate(papers, 1):
            if "author_ids" in paper and paper["author_ids"]:
                print(f"Fetching author data for paper {i}/{len(papers)}")
                author_data = s2_client.fetch_authors_batch(paper["author_ids"])
                paper["author_data"] = [data for data in author_data if data is not None]
                
                # Compute author reputation
                if paper["author_data"]:
                    paper["author_reputation"] = compute_author_reputation(
                        paper["author_data"],
                        CONFIG["author_settings"]
                    )
                else:
                    paper["author_reputation"] = 0.0
            else:
                paper["author_data"] = []
                paper["author_reputation"] = 0.0
        
        # Select papers based on novelty and author reputation
        selected_papers = novelty_detector.select_papers_combined_score(
            papers,
            limit=novelty_settings.get("final_paper_limit", CONFIG["max_papers"]),
            novelty_weight=novelty_settings.get("novelty_weight", 0.7),
            author_weight=novelty_settings.get("author_weight", 0.3)
        )
        
        print(f"\nSelected {len(selected_papers)} papers based on novelty and author reputation")
        papers = selected_papers
    else:
        # Traditional filtering based on citation count
        papers = filter_papers(papers, CONFIG["min_citations"])
        print(f"\nFiltered to {len(papers)} papers based on citation count")
        
        # Fetch author information if Semantic Scholar client is available
        if s2_client:
            print("\nFetching author information from Semantic Scholar...")
            for i, paper in enumerate(papers, 1):
                if "author_ids" in paper and paper["author_ids"]:
                    print(f"Fetching author data for paper {i}/{len(papers)}")
                    author_data = s2_client.fetch_authors_batch(paper["author_ids"])
                    paper["author_data"] = [data for data in author_data if data is not None]
                else:
                    paper["author_data"] = []
    
    if not papers:
        print("No papers met the filtering criteria. Try adjusting the parameters in config.py")
        return
    
    # Score and analyze papers
    print("\nAnalyzing papers with OpenAI...")
    all_papers_output = []  # Store all papers' analysis
    
    for i, paper in enumerate(papers, 1):
        print(f"Analyzing paper {i}/{len(papers)}")
        
        # Get scores, explanations, and key insights
        scores, explanations = openai_analyzer.score_paper(paper["title"], paper["abstract"])
        paper["scores"] = scores
        paper["score_explanations"] = explanations
        
        # Compute citation impact if data is available
        citation_impact = 0.0
        if "citation_count" in paper and "publication_date" in paper:
            citation_impact = compute_citation_impact(
                paper["citation_count"],
                paper["publication_date"],
                CONFIG["citation_settings"]
            )
        paper["citation_impact"] = citation_impact
        
        # Compute author reputation if data is available
        author_reputation = 0.0
        if "author_data" in paper and paper["author_data"]:
            author_reputation = compute_author_reputation(
                paper["author_data"],
                CONFIG["author_settings"]
            )
        paper["author_reputation"] = author_reputation
        
        # Compute final score
        paper["final_score"] = compute_final_score(
            scores,
            CONFIG["score_weights"],
            citation_impact,
            author_reputation
        )
        
        # Store full analysis
        all_papers_output.append({
            "title": paper["title"],
            "arxiv_id": paper["arxiv_id"],
            "authors": paper["authors"],
            "pdf_url": paper["pdf_url"],
            "citation_count": paper.get("citation_count", 0),
            "publication_date": paper.get("publication_date", ""),
            "citation_impact": paper.get("citation_impact", 0.0),
            "author_reputation": paper.get("author_reputation", 0.0),
            "scores": scores,
            "score_explanations": explanations,
            "key_insights": paper.get("key_insights", {}),
            "final_score": paper["final_score"]
        })
    
    # Sort by final score
    papers.sort(key=lambda x: x["final_score"], reverse=True)
    top_papers = papers[:CONFIG["top_papers_count"]]
    
    # Generate summaries for top papers
    print(f"\nGenerating summaries for top {len(top_papers)} papers...")
    for i, paper in enumerate(top_papers, 1):
        print(f"Generating summary for paper {i}/{len(top_papers)}")
        paper["summary"] = openai_analyzer.generate_summary(
            paper["title"],
            paper["abstract"]
        )
    
    # Save results
    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "arxiv_categories": CONFIG["arxiv_categories"],
            "lookback_days": CONFIG["lookback_days"],
            "total_papers_found": len(papers),
            "citation_data_available": s2_client is not None,
            "author_data_available": s2_client is not None,
            "novelty_detection_enabled": novelty_enabled and novelty_detector is not None
        },
        "all_papers": all_papers_output,  # Store all papers' analysis
        "top_papers": [{
            "title": paper["title"],
            "arxiv_id": paper["arxiv_id"],
            "authors": paper["authors"],
            "pdf_url": paper["pdf_url"],
            "citation_count": paper.get("citation_count", 0),
            "publication_date": paper.get("publication_date", ""),
            "citation_impact": paper.get("citation_impact", 0.0),
            "author_reputation": paper.get("author_reputation", 0.0),
            "novelty_score": paper.get("novelty_score", 0.0),
            "combined_score": paper.get("combined_score", 0.0),
            "scores": paper["scores"],
            "score_explanations": paper["score_explanations"],
            "key_insights": paper.get("key_insights", {}),
            "final_score": paper["final_score"],
            "summary": paper["summary"]
        } for paper in top_papers]
    }
    
    # Write to file
    output_file = "research_output.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nAnalysis complete! Results saved to {output_file}")
    print(f"Processed {len(papers)} papers, saved all analyses and top {len(top_papers)} detailed summaries")
    
    # Format and send email
    send_email_report(output["top_papers"])

def format_email_content(top_papers: List[Dict[str, Any]]) -> tuple[str, str]:
    """Format the top papers into both HTML and plain text email content."""
    # Plain text version
    plain_content = "Top Research Papers Analysis - Business Impact Focus\n\n"
    plain_content += "=" * 80 + "\n\n"
    
    # Generate date strings
    current_date = datetime.now().strftime('%B %d, %Y at %I:%M %p')
    current_month = datetime.now().strftime('%B')
    current_year = datetime.now().strftime('%Y')
    
    # HTML version with improved mobile-friendly styling
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Research Pipeline: Business Impact Analysis</title>
    <style>
        /* Mobile-first responsive styles */
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            margin: 0;
            padding: 10px;
            font-size: 16px;
            background-color: #f5f7fa;
        }}
        .container {{
            max-width: 600px;
            margin: 0 auto;
            width: 100%;
        }}
        .header {{
            text-align: center;
            margin: 0 auto 30px;
            padding: 25px;
            background: #1a5f7a;
            color: white;
            border-radius: 12px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 22px;
            line-height: 1.3;
            font-weight: bold;
        }}
        .header p {{
            margin: 15px 0 0;
            font-size: 16px;
        }}
        .header-stats {{
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-top: 20px;
            flex-wrap: wrap;
        }}
        .stat-item {{
            background: rgba(255,255,255,0.1);
            padding: 12px 15px;
            border-radius: 8px;
            text-align: center;
            flex: 1 1 calc(33% - 10px);
            min-width: 80px;
        }}
        .stat-value {{
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .stat-label {{
            font-size: 12px;
            text-transform: uppercase;
        }}
        .toc {{
            background: white;
            padding: 20px;
            border-radius: 12px;
            margin: 0 auto 30px;
            border: 1px solid #e2e8f0;
        }}
        .toc h2 {{
            color: #1a5f7a;
            margin: 0 0 20px;
            text-align: center;
            font-size: 20px;
            font-weight: bold;
        }}
        .toc-item {{
            margin-bottom: 15px;
            padding-bottom: 15px;
            border-bottom: 1px solid #f0f0f0;
        }}
        .toc-item:last-child {{
            margin-bottom: 0;
            padding-bottom: 0;
            border-bottom: none;
        }}
        .toc-number {{
            display: inline-block;
            background: #1a5f7a;
            color: white;
            width: 24px;
            height: 24px;
            border-radius: 50%;
            text-align: center;
            line-height: 24px;
            font-weight: bold;
            font-size: 14px;
            margin-right: 10px;
        }}
        .toc-title {{
            color: #1a5f7a;
            text-decoration: none;
            font-weight: bold;
            font-size: 16px;
        }}
        .paper {{
            margin: 0 auto 30px;
            padding: 20px;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            background-color: #fff;
        }}
        .paper-title {{
            color: #1a5f7a;
            font-size: 18px;
            margin: 0 0 20px;
            font-weight: bold;
            line-height: 1.4;
            padding-bottom: 15px;
            border-bottom: 2px solid #e2e8f0;
        }}
        .paper-number {{
            display: inline-block;
            background: #1a5f7a;
            color: white;
            width: 28px;
            height: 28px;
            border-radius: 50%;
            text-align: center;
            line-height: 28px;
            font-weight: bold;
            font-size: 16px;
            margin-right: 10px;
        }}
        .quick-take {{
            background: #f0f9ff;
            border-radius: 8px;
            padding: 15px;
            margin: 20px 0;
            border: 1px solid #90caf9;
        }}
        .quick-take-header {{
            color: #1a5f7a;
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 15px;
            text-align: center;
            text-transform: uppercase;
        }}
        .quick-take-grid {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 10px;
        }}
        .quick-take-item {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            border: 1px solid #90caf9;
        }}
        .quick-take-label {{
            font-size: 14px;
            color: #4a5568;
            margin-bottom: 5px;
            font-weight: 500;
            text-transform: uppercase;
        }}
        .quick-take-value {{
            font-size: 20px;
            font-weight: bold;
            color: #1a5f7a;
        }}
        .metadata {{
            margin: 20px 0;
            background: #f8fafc;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
        }}
        .metadata-item {{
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px;
            background: white;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
            margin-bottom: 10px;
        }}
        .metadata-item:last-child {{
            margin-bottom: 0;
        }}
        .metadata-icon {{
            width: 30px;
            height: 30px;
            flex-shrink: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #f0f9ff;
            border-radius: 6px;
            color: #1a5f7a;
            font-size: 16px;
            border: 1px solid #e2e8f0;
        }}
        .metadata-content {{
            flex: 1;
        }}
        .metadata strong {{
            display: block;
            color: #2d3748;
            margin-bottom: 3px;
            font-size: 12px;
            text-transform: uppercase;
        }}
        .metadata a {{
            color: #1a5f7a;
            text-decoration: none;
            word-break: break-all;
            font-weight: 500;
        }}
        .summary {{
            background-color: #fff;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
            border: 1px solid #e2e8f0;
        }}
        .summary-section {{
            margin-bottom: 20px;
            padding: 15px;
            background: #f8fafc;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
        }}
        .summary-section:last-child {{
            margin-bottom: 0;
        }}
        .summary-header {{
            color: #1a5f7a;
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #e2e8f0;
        }}
        .summary-content {{
            color: #4a5568;
            line-height: 1.6;
            font-size: 14px;
        }}
        .summary-content p {{
            margin: 0 0 15px;
        }}
        .summary-content p:last-child {{
            margin-bottom: 0;
        }}
        .scores-table {{
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            margin: 20px 0;
            background: #fff;
            border-radius: 8px;
            overflow: hidden;
            font-size: 14px;
            border: 1px solid #e2e8f0;
        }}
        .scores-table caption {{
            padding: 8px;
            font-weight: bold;
            text-align: left;
        }}
        .scores-table td {{
            padding: 12px;
            border-bottom: 1px solid #e2e8f0;
        }}
        .scores-table tr:first-child td {{
            background: #1a5f7a;
            color: white;
            font-weight: bold;
            text-transform: uppercase;
        }}
        .scores-table td:first-child {{
            font-weight: 600;
            color: #2d3748;
            border-right: 1px solid #e2e8f0;
        }}
        .scores-table td:last-child {{
            text-align: right;
            color: #1a5f7a;
            font-weight: bold;
        }}
        .scores-table tr:last-child td {{
            border-bottom: none;
        }}
        
        /* Larger screen improvements */
        @media (min-width: 768px) {{
            .container {{
                max-width: 800px;
            }}
            .header {{
                padding: 35px;
                margin-bottom: 40px;
            }}
            .header h1 {{
                font-size: 28px;
            }}
            .quick-take-grid {{
                grid-template-columns: repeat(3, 1fr);
            }}
            .metadata {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                gap: 15px;
                padding: 20px;
            }}
            .metadata-item {{
                margin-bottom: 0;
            }}
            .paper {{
                padding: 30px;
            }}
            .paper-title {{
                font-size: 22px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Research Pipeline: Business Impact Analysis</h1>
            <p>Generated on {current_date}</p>
            <div class="header-stats">
                <div class="stat-item">
                    <div class="stat-value">{len(top_papers)}</div>
                    <div class="stat-label">Papers Analyzed</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{current_month}</div>
                    <div class="stat-label">Month</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{current_year}</div>
                    <div class="stat-label">Year</div>
                </div>
            </div>
        </div>

        <div class="toc">
            <h2>Table of Contents</h2>
"""

    # Add TOC entries for each paper
    for i, paper in enumerate(top_papers, 1):
        html_content += f"""
            <div class="toc-item">
                <div>
                    <span class="toc-number">{i}</span>
                    <a href="#paper-{i}" class="toc-title">{paper['title']}</a>
                </div>
                <div style="margin-top: 10px; margin-left: 34px;">
                    <a href="#quick-take-{i}" style="display: inline-block; margin-right: 10px; margin-bottom: 5px; color: #4a5568; text-decoration: none; font-size: 14px; background: white; padding: 5px 10px; border-radius: 6px; border: 1px solid #e2e8f0;">Quick Take</a>
                    <a href="#metadata-{i}" style="display: inline-block; margin-right: 10px; margin-bottom: 5px; color: #4a5568; text-decoration: none; font-size: 14px; background: white; padding: 5px 10px; border-radius: 6px; border: 1px solid #e2e8f0;">Details</a>
                    <a href="#summary-{i}" style="display: inline-block; margin-right: 10px; margin-bottom: 5px; color: #4a5568; text-decoration: none; font-size: 14px; background: white; padding: 5px 10px; border-radius: 6px; border: 1px solid #e2e8f0;">Summary</a>
                </div>
            </div>
"""
    
    html_content += """
            <p style="text-align: center; color: #718096; font-size: 14px; margin-top: 20px; font-style: italic;">(Tap section names to navigate)</p>
        </div>
"""
    
    for i, paper in enumerate(top_papers, 1):
        # Plain text formatting
        plain_content += f"{i}. {paper['title']}\n"
        plain_content += "-" * 80 + "\n"
        plain_content += f"Business Score: {paper['scores']['business']:.2f}\n"
        plain_content += f"Market Impact: {paper['scores']['impact']:.2f}\n"
        plain_content += f"Implementation Complexity: {paper['scores']['technical_rigor']:.2f}\n"
        plain_content += f"Authors: {', '.join(paper['authors'])}\n"
        plain_content += f"ArXiv URL: {paper['pdf_url']}\n"
        if paper.get('citation_count'):
            plain_content += f"Citations: {paper['citation_count']}\n"
        if paper.get('author_reputation', 0.0) > 0.0:
            plain_content += f"Author Reputation: {paper.get('author_reputation', 0.0):.2f}\n"
        if paper.get('novelty_score', 0.0) > 0.0:
            plain_content += f"Novelty Score: {paper.get('novelty_score', 0.0):.2f}\n"
        plain_content += f"Final Score: {paper['final_score']:.2f}\n\n"
        plain_content += "Summary:\n"
        plain_content += f"{paper['summary']}\n\n"
        plain_content += "Business Assessment:\n"
        for score_name, score_value in paper['scores'].items():
            plain_content += f"- {score_name.replace('_', ' ').title()}: {score_value:.2f}\n"
        plain_content += "\n" + "=" * 80 + "\n\n"
        
        # HTML formatting
        # Process the summary text with improved section handling
        sections = paper['summary'].split('# ')
        processed_sections = []
        
        for section in sections:
            if section.strip():
                # Split into title and content
                parts = section.split('\n', 1)
                if len(parts) > 1:
                    title, content = parts
                    # Process the content
                    content = (content.strip()
                        .replace('\n- ', '\n• ')  # Convert markdown lists to bullet points
                        .replace('\n\n', '</p><p>')
                        .replace('\n', '<br>')
                        .replace('**', '<strong>')
                        .replace('</strong>:', '</strong>')
                        .replace('_', ' '))
                    
                    processed_sections.append(f'''
                        <div class="summary-section">
                            <div class="summary-header">{title.strip()}</div>
                            <div class="summary-content">
                                <p>{content}</p>
                            </div>
                        </div>
                    ''')
        
        summary_html = ''.join(processed_sections)
        
        # Enhanced metadata with icons
        citation_html = f'''
            <div class="metadata-item">
                <div class="metadata-icon">📊</div>
                <div class="metadata-content">
                    <strong>Citations</strong>
                    {paper['citation_count']}
                </div>
            </div>
        ''' if paper.get('citation_count') else ''
        
        author_reputation_html = f'''
            <div class="metadata-item">
                <div class="metadata-icon">👨‍🎓</div>
                <div class="metadata-content">
                    <strong>Author Reputation</strong>
                    {paper.get('author_reputation', 0.0):.2f}
                </div>
            </div>
        ''' if paper.get('author_reputation', 0.0) > 0.0 else ''
        
        novelty_html = f'''
            <div class="metadata-item">
                <div class="metadata-icon">💡</div>
                <div class="metadata-content">
                    <strong>Novelty Score</strong>
                    {paper.get('novelty_score', 0.0):.2f}
                </div>
            </div>
        ''' if paper.get('novelty_score', 0.0) > 0.0 else ''
        
        # Generate paper HTML with section IDs and semantic elements
        paper_html = f"""
        <article id="paper-{i}" class="paper">
            <h2 class="paper-title"><span class="paper-number">{i}</span> {paper['title']}</h2>
            
            <section id="quick-take-{i}" class="quick-take">
                <h3 class="quick-take-header">Quick Business Take</h3>
                <div class="quick-take-grid">
                    <div class="quick-take-item">
                        <div class="quick-take-label">Business Value</div>
                        <div class="quick-take-value">{paper['scores']['business']:.2f}</div>
                    </div>
                    <div class="quick-take-item">
                        <div class="quick-take-label">Market Impact</div>
                        <div class="quick-take-value">{paper['scores']['impact']:.2f}</div>
                    </div>
                    <div class="quick-take-item">
                        <div class="quick-take-label">Implementation</div>
                        <div class="quick-take-value">{paper['scores']['technical_rigor']:.2f}</div>
                    </div>
                </div>
            </section>
            
            <section id="metadata-{i}" class="metadata">
                <div class="metadata-item">
                    <div class="metadata-icon">👥</div>
                    <div class="metadata-content">
                        <strong>Authors</strong>
                        {', '.join(paper['authors'])}
                    </div>
                </div>
                <div class="metadata-item">
                    <div class="metadata-icon">📄</div>
                    <div class="metadata-content">
                        <strong>ArXiv</strong>
                        <a href="{paper['pdf_url']}" target="_blank" rel="noopener">View Paper</a>
                    </div>
                </div>
                {citation_html}
                {author_reputation_html}
                {novelty_html}
                <div class="metadata-item">
                    <div class="metadata-icon">⭐</div>
                    <div class="metadata-content">
                        <strong>Final Score</strong>
                        <span style="color: #1a5f7a; font-weight: bold;">{paper['final_score']:.2f}</span>
                    </div>
                </div>
            </section>
            
            <section id="summary-{i}" class="summary">
                {summary_html}
            </section>
            
            <section id="assessment-{i}">
                <table class="scores-table">
                    <caption>Business Value Assessment</caption>
                    <tbody>
"""
        
        # Order scores to highlight business metrics first
        score_order = ['business', 'impact', 'technical_rigor', 'ai_relevance', 'novelty', 'clarity']
        for score_name in score_order:
            if score_name in paper['scores']:
                score_value = paper['scores'][score_name]
                is_business = score_name == 'business'
                bg_color = '#1a5f7a' if is_business else 'white'
                text_color = 'white' if is_business else '#2d3748'
                value_color = 'white' if is_business else '#1a5f7a'
                
                paper_html += f"""
                        <tr style="background: {bg_color};">
                            <td style="color: {text_color};">{score_name.replace('_', ' ').title()}</td>
                            <td style="color: {value_color};">{score_value:.2f}</td>
                        </tr>
"""
        
        paper_html += """
                    </tbody>
                </table>
            </section>
        </article>
"""
        
        html_content += paper_html
    
    html_content += """
    </div>
</body>
</html>
"""
    
    return plain_content, html_content

def send_email_report(top_papers: List[Dict[str, Any]]) -> None:
    """Send email report with top papers analysis."""
    email_config = CONFIG["email"]
    
    required_fields = [
        "smtp_server",
        "smtp_port",
        "sender_email",
        "sender_password",
        "recipient_emails"  # Changed to support multiple recipients
    ]
    
    if not all(email_config.get(field) for field in required_fields):
        print("Email configuration incomplete. Skipping email report.")
        return
    
    try:
        msg = MIMEMultipart('alternative')
        msg['From'] = email_config["sender_email"]
        # Handle multiple recipients
        recipient_list = email_config["recipient_emails"].split(',')
        msg['To'] = ', '.join(recipient_list)
        msg['Subject'] = f"Research Pipeline: Top {len(top_papers)} Papers Analysis - {datetime.now().strftime('%Y-%m-%d')}"
        
        # Create both plain text and HTML versions
        plain_content, html_content = format_email_content(top_papers)
        
        # Attach both versions
        part1 = MIMEText(plain_content, 'plain')
        part2 = MIMEText(html_content, 'html')
        msg.attach(part1)
        msg.attach(part2)
        
        with smtplib.SMTP(email_config["smtp_server"], int(email_config["smtp_port"])) as server:
            server.starttls()
            server.login(email_config["sender_email"], email_config["sender_password"])
            server.send_message(msg)
        
        print(f"\nEmail report sent successfully to {len(recipient_list)} recipients")
    except Exception as e:
        print(f"\nError sending email: {str(e)}")

if __name__ == "__main__":
    main()
