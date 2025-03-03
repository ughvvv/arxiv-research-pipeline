"""
Analyze and visualize research paper scores and rankings.
"""

import json
import sys
from typing import Dict, Any, List
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def load_research_output(filepath: str = "research_output.json") -> Dict[str, Any]:
    """Load the research output JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {filepath}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {filepath}")
        sys.exit(1)

def analyze_scores(data: Dict[str, Any]) -> None:
    """Analyze and visualize paper scores."""
    papers = pd.DataFrame(data["all_papers"])
    
    # Extract scores into separate columns
    for score_type in ["novelty", "impact", "technical_rigor", "clarity", "business", "ai_relevance"]:
        papers[score_type] = papers["scores"].apply(lambda x: x.get(score_type, 0))
    
    # Create score distribution plot
    plt.figure(figsize=(12, 6))
    score_cols = ["novelty", "impact", "technical_rigor", "clarity", "business", "ai_relevance"]
    sns.boxplot(data=papers[score_cols])
    plt.title("Distribution of Paper Scores by Category")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("score_distribution.png")
    plt.close()
    
    # Create correlation heatmap
    plt.figure(figsize=(10, 8))
    correlation = papers[score_cols + ["final_score", "citation_impact"]].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
    plt.title("Score Correlation Matrix")
    plt.tight_layout()
    plt.savefig("score_correlation.png")
    plt.close()
    
    # Print summary statistics
    print("\nScore Summary Statistics:")
    print(papers[score_cols + ["final_score", "citation_impact"]].describe())
    
    # Analyze top papers
    top_papers = pd.DataFrame(data["top_papers"])
    print("\nTop Papers Analysis:")
    for _, paper in top_papers.iterrows():
        print(f"\nTitle: {paper['title']}")
        print("Scores:")
        for score_type, score in paper["scores"].items():
            print(f"  {score_type}: {score:.3f}")
        print(f"Citation Impact: {paper['citation_impact']:.3f}")
        print(f"Final Score: {paper['final_score']:.3f}")
        
        if "key_insights" in paper:
            print("\nKey Insights:")
            for category, insights in paper["key_insights"].items():
                print(f"\n{category.title()}:")
                print(insights)
        print("\n" + "="*80)

def main():
    """Main analysis function."""
    print("Loading research output...")
    data = load_research_output()
    
    print("\nAnalyzing scores and generating visualizations...")
    analyze_scores(data)
    
    print("\nAnalysis complete! Generated:")
    print("- score_distribution.png: Box plot of score distributions")
    print("- score_correlation.png: Correlation matrix of different scores")
    print("\nCheck the terminal output above for detailed statistics and top paper analysis.")

if __name__ == "__main__":
    main()
