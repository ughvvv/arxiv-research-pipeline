#!/usr/bin/env python3
"""
This script performs a ground truth validation by comparing the computed novelty scores
with a sample of top papers extracted from the research output.
It prints details (title, novelty score, final score) for the top 5 papers in terms of novelty.
"""

import json
from config import CONFIG

def main():
    output_file = "research_output.json"
    
    try:
        with open(output_file, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading research output: {str(e)}")
        return

    top_papers = data.get("top_papers", [])
    if not top_papers:
        print("No top papers found in the research output.")
        return

    # Sort papers by novelty_score (if available) in descending order
    top_by_novelty = sorted(top_papers, key=lambda p: p.get("novelty_score", 0), reverse=True)

    print("Top 5 papers by novelty score:")
    for paper in top_by_novelty[:5]:
        print("Title:", paper.get("title", "N/A"))
        print("Novelty Score:", paper.get("novelty_score", "N/A"))
        print("Final Score:", paper.get("final_score", "N/A"))
        print("------------------------------------------------")

if __name__ == "__main__":
    main()
