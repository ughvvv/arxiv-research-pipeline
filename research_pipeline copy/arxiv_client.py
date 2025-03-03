"""
ArXiv API client for fetching research papers.
Uses requests to fetch and parse ArXiv's API responses.
"""

import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import List, Dict, Any
import time
from urllib.parse import urlencode

class ArxivClient:
    """Client for interacting with the ArXiv API."""
    
    BASE_URL = "http://export.arxiv.org/api/query?"
    NAMESPACE = {'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'}
    
    def __init__(self, categories: List[str], lookback_days: int):
        """
        Initialize the ArXiv client.
        
        Args:
            categories: List of ArXiv category codes (e.g., ["cs.LG", "cs.AI"])
            lookback_days: Number of days to look back for papers
        """
        self.categories = categories
        self.lookback_days = lookback_days
    
    def _build_query(self) -> str:
        """Build the ArXiv API query string."""
        cat_query = " OR ".join(f"cat:{cat}" for cat in self.categories)
        date_limit = datetime.now() - timedelta(days=self.lookback_days)
        date_query = f"submittedDate:[{date_limit.strftime('%Y%m%d')}0000 TO 99991231235959]"
        
        return f"({cat_query}) AND {date_query}"
    
    def _parse_entry(self, entry: ET.Element) -> Dict[str, Any]:
        """Parse an entry from the ArXiv API response."""
        # Extract authors
        authors = []
        for author in entry.findall('atom:author', self.NAMESPACE):
            name = author.find('atom:name', self.NAMESPACE)
            if name is not None:
                authors.append(name.text)
        
        # Extract PDF link
        pdf_url = None
        for link in entry.findall('atom:link', self.NAMESPACE):
            if link.get('title') == 'pdf':
                pdf_url = link.get('href')
                break
        
        # Extract categories
        categories = []
        for category in entry.findall('arxiv:primary_category', self.NAMESPACE):
            categories.append(category.get('term'))
        
        # Get ID and clean it
        arxiv_id = entry.find('atom:id', self.NAMESPACE).text
        arxiv_id = arxiv_id.split('/')[-1]
        if 'v' in arxiv_id:
            arxiv_id = arxiv_id.split('v')[0]
        
        return {
            "arxiv_id": arxiv_id,
            "title": entry.find('atom:title', self.NAMESPACE).text.strip(),
            "abstract": entry.find('atom:summary', self.NAMESPACE).text.strip(),
            "published": entry.find('atom:published', self.NAMESPACE).text,
            "authors": authors,
            "pdf_url": pdf_url,
            "categories": categories
        }
    
    def fetch_papers(self, max_results: int = 100) -> List[Dict[str, Any]]:
        """
        Fetch papers from ArXiv.
        
        Args:
            max_results: Maximum number of results to return
            
        Returns:
            List of paper dictionaries containing metadata
        """
        query_params = {
            "search_query": self._build_query(),
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending"
        }
        
        url = self.BASE_URL + urlencode(query_params)
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.content)
            papers = []
            
            # Process each entry
            for entry in root.findall('atom:entry', self.NAMESPACE):
                try:
                    paper = self._parse_entry(entry)
                    papers.append(paper)
                except (AttributeError, TypeError) as e:
                    print(f"Error parsing entry: {e}")
                    continue
                
                time.sleep(1)  # Rate limiting
            
            return papers
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching papers: {e}")
            return []
        except ET.ParseError as e:
            print(f"Error parsing XML response: {e}")
            return []
