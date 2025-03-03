#!/usr/bin/env python3
"""
This script evaluates the distribution of novelty scores derived from cached paper embeddings.
It loads embeddings from the cache, computes novelty scores using the NoveltyDetector,
and plots a histogram showing the distribution.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from novelty_detector import NoveltyDetector
from semanticscholar_client import SemanticScholarClient
from config import CONFIG

def main():
    cache_dir = CONFIG["novelty_settings"]["cache_dir"]
    cache_file = os.path.join(cache_dir, "embedding_cache.pkl")
    if not os.path.exists(cache_file):
        print("No embedding cache found at", cache_file)
        return

    with open(cache_file, "rb") as f:
        embedding_cache = pickle.load(f)

    # Initialize a dummy Semantic Scholar client (used by the NoveltyDetector)
    dummy_client = SemanticScholarClient(api_key=CONFIG["semanticscholar_api_key"],
                                           delay_seconds=CONFIG["api_delay_seconds"])
    nd = NoveltyDetector(dummy_client, cache_dir=cache_dir, config=CONFIG["novelty_settings"])

    novelty_scores = []
    for paper_id, embedding in embedding_cache.items():
        score = nd.compute_novelty_score(embedding)
        novelty_scores.append(score)

    novelty_scores = np.array(novelty_scores)
    print(f"Computed novelty scores for {len(novelty_scores)} embeddings.")

    plt.hist(novelty_scores, bins=20, color="skyblue", edgecolor="black")
    plt.title("Distribution of Novelty Scores")
    plt.xlabel("Novelty Score")
    plt.ylabel("Frequency")
    plt.savefig("evaluation_tests/novelty_score_distribution.png")
    plt.show()

if __name__ == "__main__":
    main()
