#!/usr/bin/env python3
"""
This script fetches a sample of recent ArXiv papers (from the last 7 days),
computes their novelty scores using the NoveltyDetector, and (optionally)
groups the results by paper category.
It prints summary statistics and saves category-specific novelty distributions.
"""

import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from arxiv_client import ArxivClient
from novelty_detector import NoveltyDetector
from semanticscholar_client import SemanticScholarClient
from config import CONFIG

def fetch_recent_papers():
    # Override lookback_days to 7 for this test
    client = ArxivClient(categories=CONFIG["arxiv_categories"], lookback_days=7)
    papers = client.fetch_papers(max_results=50)
    print(f"Fetched {len(papers)} papers from the last 7 days.")
    return papers

def compute_novelty_for_papers(papers, nd):
    # Assumes each paper has a unique identifier (arxiv_id) and its embedding is cached
    paper_novelty = []
    for paper in papers:
        paper_id = paper.get("arxiv_id")
        # Retrieve embedding from cache if available
        embedding = nd.get_paper_embedding(paper_id)
        if embedding is None:
            # If not cached, skip computation (or we could fetch it via API)
            continue
        novelty_score = nd.compute_novelty_score(embedding)
        paper_novelty.append((paper, novelty_score))
    return paper_novelty

def group_by_category(paper_novelty):
    groups = {}
    for paper, novelty in paper_novelty:
        # Assume each paper has a "categories" field as a list
        categories = paper.get("categories", ["Unknown"])
        # For simplicity, use the first category as the primary category
        primary = categories[0] if categories else "Unknown"
        groups.setdefault(primary, []).append(novelty)
    return groups

def plot_category_distributions(groups):
    for category, scores in groups.items():
        scores = np.array(scores)
        plt.hist(scores, bins=10, color="lightgreen", edgecolor="black")
        plt.title(f"Novelty Score Distribution for {category}")
        plt.xlabel("Novelty Score")
        plt.ylabel("Frequency")
        filename = f"evaluation_tests/novelty_distribution_{category.replace(' ', '_')}.png"
        plt.savefig(filename)
        print(f"Saved histogram for category '{category}' as {filename}")
        plt.clf()

def main():
    # Initialize Semantic Scholar client and NoveltyDetector
    s2_client = SemanticScholarClient(api_key=CONFIG["semanticscholar_api_key"],
                                      delay_seconds=CONFIG["api_delay_seconds"])
    nd = NoveltyDetector(s2_client, cache_dir=CONFIG["novelty_settings"]["cache_dir"],
                         config=CONFIG["novelty_settings"])
    
    papers = fetch_recent_papers()
    if not papers:
        print("No papers fetched.")
        return

    paper_novelty = compute_novelty_for_papers(papers, nd)
    if not paper_novelty:
        print("No novelty scores computed. Ensure embeddings are available.")
        return

    # Overall statistics
    all_scores = np.array([score for (_, score) in paper_novelty])
    print(f"Average novelty score: {np.mean(all_scores):.3f}")
    print(f"Median novelty score: {np.median(all_scores):.3f}")

    plt.hist(all_scores, bins=15, color="skyblue", edgecolor="black")
    plt.title("Overall Novelty Score Distribution (Recent Papers)")
    plt.xlabel("Novelty Score")
    plt.ylabel("Frequency")
    overall_filename = "evaluation_tests/novelty_score_distribution_overall.png"
    plt.savefig(overall_filename)
    print(f"Saved overall novelty distribution as {overall_filename}")
    plt.clf()

    # Group by category and plot distributions
    groups = group_by_category(paper_novelty)
    for category, scores in groups.items():
        print(f"Category: {category}, Count: {len(scores)}, Average novelty: {np.mean(scores):.3f}")
    plot_category_distributions(groups)

if __name__ == "__main__":
    main()
