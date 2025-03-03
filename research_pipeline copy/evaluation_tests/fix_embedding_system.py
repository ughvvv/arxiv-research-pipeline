#!/usr/bin/env python3
"""
This script diagnoses and fixes issues with the embedding system.
It attempts to fetch embeddings for recent papers, diagnoses any issues,
and updates the embedding cache with the retrieved embeddings.
"""

import os
import pickle
import json
import time
from typing import Dict, List, Optional, Any
import matplotlib.pyplot as plt
from arxiv_client import ArxivClient
from semanticscholar_client import SemanticScholarClient
from novelty_detector import NoveltyDetector
from config import CONFIG

def diagnose_embedding_system():
    """Diagnose issues with the embedding system."""
    print("Diagnosing embedding system...")
    
    # Check if cache directory exists
    cache_dir = CONFIG["novelty_settings"]["cache_dir"]
    if not os.path.exists(cache_dir):
        print(f"Cache directory '{cache_dir}' does not exist. Creating it...")
        os.makedirs(cache_dir, exist_ok=True)
    
    # Check if embedding cache exists
    cache_file = os.path.join(cache_dir, "embedding_cache.pkl")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                embedding_cache = pickle.load(f)
            print(f"Embedding cache exists with {len(embedding_cache)} entries.")
        except Exception as e:
            print(f"Error loading embedding cache: {str(e)}")
            embedding_cache = {}
    else:
        print("Embedding cache does not exist.")
        embedding_cache = {}
    
    return embedding_cache

def fetch_recent_papers(max_results=20):
    """Fetch recent papers from ArXiv."""
    print(f"Fetching {max_results} recent papers from ArXiv...")
    client = ArxivClient(categories=CONFIG["arxiv_categories"], lookback_days=7)
    papers = client.fetch_papers(max_results=max_results)
    print(f"Fetched {len(papers)} papers.")
    return papers

def extract_s2_ids(papers):
    """Extract Semantic Scholar IDs from papers."""
    s2_ids = []
    for paper in papers:
        arxiv_id = paper.get("arxiv_id")
        # For ArXiv papers, we need to use the correct format for the Semantic Scholar API
        # The API accepts ArXiv IDs in the format "ARXIV:{arxiv_id}" (uppercase 'ARXIV')
        s2_id = f"ARXIV:{arxiv_id}"
        s2_ids.append((arxiv_id, s2_id))
    
    print(f"Extracted {len(s2_ids)} Semantic Scholar IDs.")
    return s2_ids

def fetch_embeddings(s2_ids, s2_client):
    """Fetch embeddings for papers."""
    print(f"Fetching embeddings for {len(s2_ids)} papers...")
    
    # Extract just the S2 IDs (not the ArXiv IDs)
    ids_only = [s2_id for _, s2_id in s2_ids]
    
    # Fetch embeddings in batches to avoid rate limiting
    batch_size = 5
    all_embeddings = {}
    
    for i in range(0, len(ids_only), batch_size):
        batch = ids_only[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(ids_only) + batch_size - 1)//batch_size}...")
        
        # Fetch embeddings for batch
        batch_embeddings = s2_client.fetch_paper_embeddings_batch(batch)
        
        # Add to all embeddings
        all_embeddings.update(batch_embeddings)
        
        # Sleep to avoid rate limiting - increased to respect API limits
        time.sleep(5)
    
    # Count successful fetches
    successful = sum(1 for emb in all_embeddings.values() if emb is not None)
    print(f"Successfully fetched {successful}/{len(s2_ids)} embeddings.")
    
    return all_embeddings

def update_embedding_cache(embedding_cache, all_embeddings, s2_ids):
    """Update embedding cache with new embeddings."""
    # Create a mapping from S2 ID to ArXiv ID
    id_mapping = {s2_id: arxiv_id for arxiv_id, s2_id in s2_ids}
    
    # Update cache
    updated = 0
    for s2_id, embedding in all_embeddings.items():
        if embedding is not None:
            embedding_cache[s2_id] = embedding
            updated += 1
    
    print(f"Updated embedding cache with {updated} new embeddings.")
    return embedding_cache

def save_embedding_cache(embedding_cache, cache_dir):
    """Save embedding cache to disk."""
    cache_file = os.path.join(cache_dir, "embedding_cache.pkl")
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(embedding_cache, f)
        print(f"Saved embedding cache with {len(embedding_cache)} entries to {cache_file}.")
    except Exception as e:
        print(f"Error saving embedding cache: {str(e)}")

def compute_novelty_scores(papers, embedding_cache, nd):
    """Compute novelty scores for papers."""
    print("Computing novelty scores...")
    
    # Build reference corpus if needed
    if not nd.reference_embeddings:
        print("Building reference corpus...")
        # Use the embedding cache as reference corpus
        nd.reference_corpus = embedding_cache
        nd.reference_embeddings = list(embedding_cache.values())
        print(f"Built reference corpus with {len(nd.reference_embeddings)} embeddings.")
    
    # Compute novelty scores
    novelty_scores = {}
    for paper in papers:
        arxiv_id = paper.get("arxiv_id")
        
        # Use the ArXiv ID format for Semantic Scholar (uppercase 'ARXIV')
        s2_id = f"ARXIV:{arxiv_id}"
        
        # Get embedding
        embedding = embedding_cache.get(s2_id)
        if embedding is None:
            print(f"No embedding found for paper: {arxiv_id}")
            continue
        
        # Compute novelty score
        novelty_score = nd.compute_novelty_score(embedding)
        novelty_scores[arxiv_id] = novelty_score
        print(f"Paper {arxiv_id}: Novelty score = {novelty_score:.3f}")
    
    return novelty_scores

def plot_novelty_distribution(novelty_scores):
    """Plot distribution of novelty scores."""
    if not novelty_scores:
        print("No novelty scores to plot.")
        return
    
    scores = list(novelty_scores.values())
    plt.hist(scores, bins=10, color="skyblue", edgecolor="black")
    plt.title("Distribution of Novelty Scores")
    plt.xlabel("Novelty Score")
    plt.ylabel("Frequency")
    plt.savefig("evaluation_tests/fixed_novelty_distribution.png")
    print(f"Saved novelty score distribution to evaluation_tests/fixed_novelty_distribution.png.")

def main():
    # Initialize Semantic Scholar client
    s2_client = SemanticScholarClient(
        api_key=CONFIG["semanticscholar_api_key"],
        delay_seconds=CONFIG["api_delay_seconds"]
    )
    
    # Initialize NoveltyDetector
    nd = NoveltyDetector(
        s2_client=s2_client,
        cache_dir=CONFIG["novelty_settings"]["cache_dir"],
        config=CONFIG["novelty_settings"]
    )
    
    # Diagnose embedding system
    embedding_cache = diagnose_embedding_system()
    
    # Fetch recent papers
    papers = fetch_recent_papers(max_results=20)
    
    # Extract Semantic Scholar IDs
    s2_ids = extract_s2_ids(papers)
    
    # Fetch embeddings
    all_embeddings = fetch_embeddings(s2_ids, s2_client)
    
    # Update embedding cache
    embedding_cache = update_embedding_cache(embedding_cache, all_embeddings, s2_ids)
    
    # Save embedding cache
    save_embedding_cache(embedding_cache, CONFIG["novelty_settings"]["cache_dir"])
    
    # Compute novelty scores
    novelty_scores = compute_novelty_scores(papers, embedding_cache, nd)
    
    # Plot novelty distribution
    plot_novelty_distribution(novelty_scores)
    
    print("Embedding system diagnosis and fix complete.")

if __name__ == "__main__":
    main()
