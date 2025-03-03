"""
Novelty detection module for research papers using embedding similarity.
"""

import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pickle
import math
from semanticscholar_client import SemanticScholarClient

class NoveltyDetector:
    """
    Detects novel papers based on embedding similarity.
    Uses Semantic Scholar embeddings to calculate novelty scores.
    """
    
    def __init__(
        self,
        s2_client: SemanticScholarClient,
        cache_dir: str = "cache",
        config: Dict[str, Any] = None
    ):
        """
        Initialize the novelty detector.
        
        Args:
            s2_client: Semantic Scholar client for fetching embeddings
            cache_dir: Directory to store embedding cache
            config: Configuration dictionary with novelty settings
        """
        self.s2_client = s2_client
        self.cache_dir = cache_dir
        self.config = config or {}
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize embedding cache
        self.embedding_cache = self._load_embedding_cache()
        
        # Initialize reference corpus
        self.reference_corpus = {}
        self.reference_embeddings = []
    
    def _load_embedding_cache(self) -> Dict[str, List[float]]:
        """Load embedding cache from disk."""
        cache_path = os.path.join(self.cache_dir, "embedding_cache.pkl")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading embedding cache: {str(e)}")
        return {}
    
    def _save_embedding_cache(self):
        """Save embedding cache to disk."""
        cache_path = os.path.join(self.cache_dir, "embedding_cache.pkl")
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(self.embedding_cache, f)
        except Exception as e:
            print(f"Error saving embedding cache: {str(e)}")
    
    def get_paper_embedding(self, paper_id: str) -> Optional[List[float]]:
        """
        Get embedding for a paper, using cache if available.
        
        Args:
            paper_id: Semantic Scholar paper ID
            
        Returns:
            Paper embedding vector or None if not available
        """
        # Check cache first
        if paper_id in self.embedding_cache:
            return self.embedding_cache[paper_id]
        
        # Fetch from API if not in cache
        embedding = self.s2_client.fetch_paper_embedding(paper_id)
        
        # Update cache if embedding was found
        if embedding:
            self.embedding_cache[paper_id] = embedding
            # Periodically save cache
            if len(self.embedding_cache) % 10 == 0:
                self._save_embedding_cache()
        
        return embedding
    
    def get_paper_embeddings_batch(
        self,
        paper_ids: List[str],
        max_workers: int = 5
    ) -> Dict[str, Optional[List[float]]]:
        """
        Get embeddings for multiple papers, using cache where available.
        
        Args:
            paper_ids: List of Semantic Scholar paper IDs
            max_workers: Maximum number of parallel workers
            
        Returns:
            Dictionary mapping paper IDs to embeddings
        """
        results = {}
        missing_ids = []
        
        # Check cache first
        for paper_id in paper_ids:
            if paper_id in self.embedding_cache:
                results[paper_id] = self.embedding_cache[paper_id]
            else:
                missing_ids.append(paper_id)
        
        # Fetch missing embeddings from API
        if missing_ids:
            print(f"Fetching {len(missing_ids)} embeddings from Semantic Scholar API")
            
            # Use parallel processing for larger batches
            if len(missing_ids) > 10:
                api_results = self.s2_client.fetch_paper_embeddings_batch(missing_ids, max_workers=max_workers)
            else:
                api_results = self.s2_client.fetch_paper_embeddings_batch(missing_ids)
            
            # Update cache with new embeddings
            for paper_id, embedding in api_results.items():
                if embedding:
                    self.embedding_cache[paper_id] = embedding
                results[paper_id] = embedding
            
            # Save updated cache
            self._save_embedding_cache()
        
        return results
    
    async def get_paper_embeddings_async(
        self,
        paper_ids: List[str],
        max_concurrent: int = 5
    ) -> Dict[str, Optional[List[float]]]:
        """
        Get embeddings for multiple papers asynchronously, using cache where available.
        
        Args:
            paper_ids: List of Semantic Scholar paper IDs
            max_concurrent: Maximum number of concurrent requests
            
        Returns:
            Dictionary mapping paper IDs to embeddings
        """
        results = {}
        missing_ids = []
        
        # Check cache first
        for paper_id in paper_ids:
            if paper_id in self.embedding_cache:
                results[paper_id] = self.embedding_cache[paper_id]
            else:
                missing_ids.append(paper_id)
        
        # Fetch missing embeddings from API
        if missing_ids:
            print(f"Fetching {len(missing_ids)} embeddings asynchronously from Semantic Scholar API")
            
            # Use async processing
            api_results = await self.s2_client.fetch_paper_embeddings_async(missing_ids, max_concurrent=max_concurrent)
            
            # Update cache with new embeddings
            for paper_id, embedding in api_results.items():
                if embedding:
                    self.embedding_cache[paper_id] = embedding
                results[paper_id] = embedding
            
            # Save updated cache
            self._save_embedding_cache()
        
        return results
    
    def build_reference_corpus(
        self,
        reference_papers: List[Dict[str, Any]]
    ) -> int:
        """
        Build reference corpus from a list of papers.
        
        Args:
            reference_papers: List of paper dictionaries with paper IDs
            
        Returns:
            Number of papers with embeddings in the reference corpus
        """
        # Extract paper IDs
        paper_ids = []
        for paper in reference_papers:
            if "s2_id" in paper:
                paper_ids.append(paper["s2_id"])
            elif "url" in paper and "semanticscholar.org/paper/" in paper["url"]:
                # Extract ID from URL
                paper_id = paper["url"].split("semanticscholar.org/paper/")[1].split("/")[0]
                paper_ids.append(paper_id)
        
        # Get performance settings from config
        max_workers = self.config.get("max_workers", 5)
        
        # Get embeddings for reference papers using parallel processing
        embeddings = self.get_paper_embeddings_batch(paper_ids, max_workers=max_workers)
        
        # Build reference corpus
        self.reference_corpus = {
            paper_id: embedding for paper_id, embedding in embeddings.items()
            if embedding is not None
        }
        
        # Extract embeddings for faster access
        self.reference_embeddings = list(self.reference_corpus.values())
        
        return len(self.reference_embeddings)
    
    def compute_novelty_score(
        self,
        paper_embedding: List[float]
    ) -> float:
        """
        Compute novelty score for a paper based on reference corpus.
        
        Args:
            paper_embedding: Embedding vector for the paper
            
        Returns:
            Novelty score between 0 and 1 (higher means more novel)
        """
        if not self.reference_embeddings:
            return 1.0  # Maximum novelty if no reference papers
        
        # Calculate similarities with reference papers
        similarities = [
            self._compute_similarity(paper_embedding, ref_embedding)
            for ref_embedding in self.reference_embeddings
        ]
        
        # Get average similarity (lower means more novel)
        avg_similarity = sum(similarities) / len(similarities)
        
        # Convert to novelty score (higher means more novel)
        novelty_score = 1.0 - avg_similarity
        
        return novelty_score
    
    def _compute_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score between 0 and 1
        """
        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        
        # Calculate magnitudes
        magnitude1 = math.sqrt(sum(a * a for a in embedding1))
        magnitude2 = math.sqrt(sum(b * b for b in embedding2))
        
        # Calculate cosine similarity
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def compute_novelty_scores_batch(
        self,
        papers: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Compute novelty scores for multiple papers.
        
        Args:
            papers: List of paper dictionaries
            
        Returns:
            Dictionary mapping paper IDs to novelty scores
        """
        novelty_scores = {}
        
        # Get paper IDs and extract Semantic Scholar IDs
        for paper in papers:
            paper_id = paper.get("arxiv_id")
            
            # Skip papers without IDs
            if not paper_id:
                continue
            
            # Get Semantic Scholar ID if available
            s2_id = paper.get("s2_id")
            if not s2_id and "url" in paper and "semanticscholar.org/paper/" in paper["url"]:
                # Extract ID from URL
                s2_id = paper["url"].split("semanticscholar.org/paper/")[1].split("/")[0]
            
            # Skip papers without Semantic Scholar ID
            if not s2_id:
                print(f"No Semantic Scholar ID for paper: {paper_id}")
                novelty_scores[paper_id] = 0.0
                continue
            
            # Get embedding for paper
            embedding = self.get_paper_embedding(s2_id)
            
            # Skip papers without embeddings
            if not embedding:
                print(f"No embedding available for paper: {paper_id}")
                novelty_scores[paper_id] = 0.0
                continue
            
            # Compute novelty score
            novelty_score = self.compute_novelty_score(embedding)
            novelty_scores[paper_id] = novelty_score
        
        return novelty_scores
    
    def select_novel_papers(
        self,
        papers: List[Dict[str, Any]],
        limit: int = 250,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Select the most novel papers from a list.
        
        Args:
            papers: List of paper dictionaries
            limit: Maximum number of papers to select
            min_score: Minimum novelty score to consider
            
        Returns:
            List of selected papers with novelty scores
        """
        # Compute novelty scores
        novelty_scores = self.compute_novelty_scores_batch(papers)
        
        # Add novelty scores to papers
        for paper in papers:
            paper_id = paper.get("arxiv_id")
            if paper_id in novelty_scores:
                paper["novelty_score"] = novelty_scores[paper_id]
            else:
                paper["novelty_score"] = 0.0
        
        # Filter papers by minimum score
        filtered_papers = [
            paper for paper in papers
            if paper.get("novelty_score", 0.0) >= min_score
        ]
        
        # Sort papers by novelty score (descending)
        sorted_papers = sorted(
            filtered_papers,
            key=lambda p: p.get("novelty_score", 0.0),
            reverse=True
        )
        
        # Select top papers
        selected_papers = sorted_papers[:limit]
        
        return selected_papers
    
    def select_papers_combined_score(
        self,
        papers: List[Dict[str, Any]],
        limit: int = 250,
        novelty_weight: float = 0.7,
        author_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Select papers based on a combination of novelty and author reputation.
        
        Args:
            papers: List of paper dictionaries
            limit: Maximum number of papers to select
            novelty_weight: Weight for novelty score
            author_weight: Weight for author reputation score
            
        Returns:
            List of selected papers with combined scores
        """
        # Compute novelty scores
        novelty_scores = self.compute_novelty_scores_batch(papers)
        
        # Add novelty scores to papers
        for paper in papers:
            paper_id = paper.get("arxiv_id")
            if paper_id in novelty_scores:
                paper["novelty_score"] = novelty_scores[paper_id]
            else:
                paper["novelty_score"] = 0.0
        
        # Compute combined scores
        for paper in papers:
            novelty_score = paper.get("novelty_score", 0.0)
            author_score = paper.get("author_reputation", 0.0)
            
            # Normalize author score to 0-1 range if needed
            if author_score > 1.0:
                author_score = min(1.0, author_score / 5.0)  # Assuming max author score is 5.0
            
            # Compute combined score
            combined_score = (
                novelty_weight * novelty_score +
                author_weight * author_score
            )
            
            paper["combined_score"] = combined_score
        
        # Sort papers by combined score (descending)
        sorted_papers = sorted(
            papers,
            key=lambda p: p.get("combined_score", 0.0),
            reverse=True
        )
        
        # Select top papers
        selected_papers = sorted_papers[:limit]
        
        return selected_papers
