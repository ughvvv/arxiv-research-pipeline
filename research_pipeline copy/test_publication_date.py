"""
Test script to verify the publication date fix.
"""

from datetime import datetime
from arxiv_client import ArxivClient
from semanticscholar_client import SemanticScholarClient
from config import CONFIG
import json

def main():
    # Initialize clients
    arxiv_client = ArxivClient(
        categories=CONFIG["arxiv_categories"][:2],  # Just use first 2 categories for testing
        lookback_days=10  # Use a shorter lookback period for testing
    )
    
    # Initialize Semantic Scholar client if API key is provided
    s2_client = None
    if CONFIG.get("semanticscholar_api_key"):
        s2_client = SemanticScholarClient(
            api_key=CONFIG["semanticscholar_api_key"],
            delay_seconds=CONFIG["api_delay_seconds"]
        )
    else:
        print("Semantic Scholar API key not configured. Skipping citation data.")
        return
    
    # Fetch a few papers from ArXiv
    print("Fetching papers from ArXiv...")
    papers = arxiv_client.fetch_papers(max_results=5)  # Just get 5 papers for testing
    print(f"Found {len(papers)} papers")
    
    # Print original publication dates from ArXiv
    print("\nOriginal ArXiv publication dates:")
    for i, paper in enumerate(papers, 1):
        print(f"{i}. {paper['title']}")
        print(f"   ArXiv published: {paper.get('published', 'N/A')}")
    
    # Try to get citation data from Semantic Scholar
    print("\nFetching citation data from Semantic Scholar...")
    arxiv_ids = [paper["arxiv_id"] for paper in papers]
    s2_results = s2_client.fetch_papers_batch(arxiv_ids)
    
    # Print Semantic Scholar publication dates
    print("\nSemantic Scholar publication dates:")
    for i, (paper, s2_data) in enumerate(zip(papers, s2_results), 1):
        if s2_data:
            print(f"{i}. {paper['title']}")
            print(f"   S2 publication_date: {s2_data.get('publication_date', 'N/A')}")
    
    # Update papers with citation data, preserving ArXiv dates
    print("\nUpdating papers with citation data, preserving ArXiv dates...")
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
    
    # Print final publication dates
    print("\nFinal publication dates after update:")
    for i, paper in enumerate(papers, 1):
        print(f"{i}. {paper['title']}")
        print(f"   Final publication_date: {paper.get('publication_date', 'N/A')}")
        print(f"   Citation count: {paper.get('citation_count', 'N/A')}")

if __name__ == "__main__":
    main()
