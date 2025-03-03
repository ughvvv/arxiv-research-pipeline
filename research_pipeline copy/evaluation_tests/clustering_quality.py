#!/usr/bin/env python3
"""
This script evaluates the clustering quality of paper embeddings.
It loads embeddings from the persistent cache, applies KMeans clustering for a range 
of cluster counts, and computes the silhouette score for each clustering configuration.
The results are plotted to help determine optimal clustering parameters.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from config import CONFIG

def main():
    cache_dir = CONFIG["novelty_settings"]["cache_dir"]
    cache_file = os.path.join(cache_dir, "embedding_cache.pkl")
    if not os.path.exists(cache_file):
        print("No embedding cache found at", cache_file)
        return

    with open(cache_file, "rb") as f:
        embedding_cache = pickle.load(f)

    embeddings = np.array(list(embedding_cache.values()))
    if embeddings.size == 0:
        print("No embeddings available.")
        return

    range_n_clusters = range(2, 11)
    silhouette_scores = []

    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, cluster_labels)
        silhouette_scores.append(score)
        print(f"n_clusters = {n_clusters}, silhouette score = {score:.3f}")

    plt.plot(list(range_n_clusters), silhouette_scores, marker="o")
    plt.title("Silhouette Score vs Number of Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.savefig("evaluation_tests/silhouette_scores.png")
    plt.show()

if __name__ == "__main__":
    main()
