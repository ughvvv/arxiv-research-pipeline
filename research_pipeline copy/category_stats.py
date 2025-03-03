"""
Script to analyze ArXiv paper publication statistics by category.
"""

import sys
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from config import CONFIG
from arxiv_client import ArxivClient

def main():
    # Get papers from the last 30 days
    lookback_days = 30
    categories = CONFIG["arxiv_categories"]
    
    print(f"Fetching papers from the last {lookback_days} days for {len(categories)} categories...")
    client = ArxivClient(categories=categories, lookback_days=lookback_days)
    papers = client.fetch_papers(max_results=1000)  # Fetch more papers for better statistics
    
    if not papers:
        print("No papers found.")
        return
    
    print(f"Found {len(papers)} papers.")
    
    # Count papers by category
    category_counts = Counter()
    for paper in papers:
        for category in paper.get("categories", []):
            category_counts[category] += 1
    
    # Count papers by date (for trend analysis)
    date_counts = defaultdict(int)
    category_by_date = defaultdict(lambda: defaultdict(int))
    
    for paper in papers:
        if "published" in paper:
            # Extract date part only
            date_str = paper["published"].split("T")[0]
            date_counts[date_str] += 1
            
            # Count by category and date
            for category in paper.get("categories", []):
                category_by_date[date_str][category] += 1
    
    # Display results
    print("\nPaper counts by category:")
    print("-" * 40)
    
    # Get category descriptions from CONFIG
    category_descriptions = {}
    for cat in CONFIG["arxiv_categories"]:
        # Extract description if available (after # comment)
        parts = cat.split("#")
        if len(parts) > 1:
            category_descriptions[parts[0].strip()] = parts[1].strip()
    
    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        # Get the full name if it's one of our tracked categories
        category_name = category
        if category in category_descriptions:
            category_name = f"{category} ({category_descriptions[category]})"
        
        print(f"{category_name}: {count}")
    
    # Display daily counts for the last 30 days
    print("\nDaily paper counts (last 30 days):")
    print("-" * 40)
    
    # Sort dates
    sorted_dates = sorted(date_counts.keys())
    for date_str in sorted_dates:
        print(f"{date_str}: {date_counts[date_str]} papers")
    
    # Create visualizations
    try:
        # Plot category distribution
        plt.figure(figsize=(12, 6))
        
        # Get top 10 categories by count
        top_categories = [cat for cat, _ in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:10]]
        top_counts = [category_counts[cat] for cat in top_categories]
        
        # Create labels with descriptions where available
        labels = []
        for cat in top_categories:
            if cat in category_descriptions:
                labels.append(f"{cat}\n({category_descriptions[cat]})")
            else:
                labels.append(cat)
        
        plt.bar(range(len(top_categories)), top_counts)
        plt.xticks(range(len(top_categories)), labels, rotation=45, ha='right')
        plt.title('Top 10 ArXiv Categories (Last 30 Days)')
        plt.tight_layout()
        plt.savefig('category_distribution.png')
        print("\nSaved category distribution chart to 'category_distribution.png'")
        
        # Plot daily trend
        plt.figure(figsize=(12, 6))
        plt.plot(sorted_dates, [date_counts[date] for date in sorted_dates])
        plt.xticks(rotation=45, ha='right')
        plt.title('Daily ArXiv Paper Submissions (Last 30 Days)')
        plt.tight_layout()
        plt.savefig('daily_trend.png')
        print("Saved daily trend chart to 'daily_trend.png'")
        
        # Plot top 5 categories over time
        plt.figure(figsize=(14, 8))
        top5_categories = [cat for cat, _ in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:5]]
        
        for category in top5_categories:
            # Get counts for each date
            category_trend = [category_by_date[date].get(category, 0) for date in sorted_dates]
            plt.plot(sorted_dates, category_trend, marker='o', linestyle='-', label=category)
        
        plt.legend()
        plt.xticks(rotation=45, ha='right')
        plt.title('Top 5 Categories Over Time (Last 30 Days)')
        plt.tight_layout()
        plt.savefig('category_trends.png')
        print("Saved category trends chart to 'category_trends.png'")
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        print("Note: Matplotlib is required for visualizations. Install with 'pip install matplotlib'")

if __name__ == "__main__":
    main()
