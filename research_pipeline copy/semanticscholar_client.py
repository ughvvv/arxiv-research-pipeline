"""
Semantic Scholar API client for fetching paper metadata, citations, and author information.
Implements rate limiting, exponential backoff, and batch processing.
"""

import requests
import time
from typing import Dict, Any, Optional, List
from tenacity import retry, stop_after_attempt, wait_exponential
import asyncio
from aiohttp import ClientSession, TCPConnector
import math
from concurrent.futures import ThreadPoolExecutor
import os

class SemanticScholarClient:
    """Client for interacting with the Semantic Scholar API."""
    
    PAPER_URL_V1 = "https://api.semanticscholar.org/v1/paper/arXiv:{}"
    PAPER_URL_V2 = "https://api.semanticscholar.org/v2/paper/arXiv:{}"
    PAPER_EMBEDDING_URL_V2 = "https://api.semanticscholar.org/v2/paper/{}/embedding"
    PAPER_BATCH_URL_V1 = "https://api.semanticscholar.org/graph/v1/paper/batch"
    AUTHOR_URL_V2 = "https://api.semanticscholar.org/v2/author/{}"
    AUTHOR_SEARCH_URL_V2 = "https://api.semanticscholar.org/v2/author/search"
    BATCH_SIZE = 100  # Process papers in batches of 100 (API supports up to 500)
    
    def __init__(self, api_key: str, delay_seconds: int = 1):
        """
        Initialize the Semantic Scholar client.
        
        Args:
            api_key: Semantic Scholar API key
            delay_seconds: Minimum delay between API calls for rate limiting
        """
        self.api_key = api_key
        self.delay_seconds = delay_seconds
        self.session = requests.Session()
        self.session.headers.update({
            "x-api-key": api_key,
            "User-Agent": "Research Pipeline/1.0"
        })
        self.last_request_time = 0
    
    def _enforce_rate_limit(self):
        """Enforce rate limiting with dynamic delay."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.delay_seconds:
            sleep_time = self.delay_seconds - time_since_last
            time.sleep(sleep_time)
        self.last_request_time = time.time()
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        reraise=True
    )
    def fetch_paper_info(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch paper information from Semantic Scholar.
        
        Args:
            arxiv_id: ArXiv ID of the paper
            
        Returns:
            Dictionary containing paper metadata or None if not found
        """
        self._enforce_rate_limit()
        
        # Remove version number from arxiv_id if present
        base_id = arxiv_id.split('v')[0]
        url = self.PAPER_URL_V1.format(base_id)
        
        try:
            response = self.session.get(url)
            
            if response.status_code == 404:
                print(f"Paper not found: {arxiv_id}")
                return None
                
            if response.status_code == 403:
                print(f"Access forbidden. Response: {response.text}")
                return None
                
            response.raise_for_status()
            data = response.json()
            
            # Extract relevant information
            year = data.get("year")
            # Create ISO format date (using middle of year if only year available)
            publication_date = f"{year}-06-15" if year else None
            
            # Extract author IDs if available
            author_ids = []
            if "authors" in data:
                for author in data.get("authors", []):
                    if "authorId" in author:
                        author_ids.append(author["authorId"])
            
            paper_info = {
                "citation_count": data.get("citationCount", 0),
                "reference_count": len(data.get("references", [])),
                "year": year,
                "publication_date": publication_date,
                "fields_of_study": data.get("fieldsOfStudy", []),
                "venue": data.get("venue", ""),
                "url": data.get("url", ""),
                "influential_citation_count": data.get("influentialCitationCount", 0),
                "author_ids": author_ids
            }
            
            return paper_info
            
        except requests.exceptions.RequestException as e:
            if response.status_code == 429:  # Rate limit exceeded
                print("Rate limit exceeded, backing off...")
                raise e  # This will trigger the retry mechanism
            print(f"Error fetching {arxiv_id}: {str(e)}")
            if hasattr(response, 'text'):
                print(f"Response text: {response.text}")
            return None

    def fetch_papers_batch(
        self,
        arxiv_ids: List[str]
    ) -> List[Optional[Dict[str, Any]]]:
        """
        Fetch multiple papers sequentially with rate limiting.
        
        Args:
            arxiv_ids: List of ArXiv IDs to fetch
            
        Returns:
            List of paper metadata dictionaries (None for failed fetches)
        """
        results = []
        total_papers = len(arxiv_ids)
        
        for i, arxiv_id in enumerate(arxiv_ids, 1):
            print(f"Processing paper {i}/{total_papers}")
            result = self.fetch_paper_info(arxiv_id)
            results.append(result)
        
        return results
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        reraise=True
    )
    def fetch_author_info(self, author_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch author information from Semantic Scholar.
        
        Args:
            author_id: Semantic Scholar author ID
            
        Returns:
            Dictionary containing author metrics or None if not found
        """
        self._enforce_rate_limit()
        
        url = self.AUTHOR_URL_V2.format(author_id)
        
        try:
            response = self.session.get(url)
            
            if response.status_code == 404:
                print(f"Author not found: {author_id}")
                return None
                
            if response.status_code == 403:
                print(f"Access forbidden. Response: {response.text}")
                return None
                
            response.raise_for_status()
            data = response.json()
            
            # Extract relevant information
            author_info = {
                "author_id": author_id,
                "name": data.get("name", ""),
                "h_index": data.get("hIndex", 0),
                "citation_count": data.get("citationCount", 0),
                "paper_count": data.get("paperCount", 0),
                "influential_citation_count": data.get("influentialCitationCount", 0)
            }
            
            return author_info
            
        except requests.exceptions.RequestException as e:
            if response.status_code == 429:  # Rate limit exceeded
                print("Rate limit exceeded, backing off...")
                raise e  # This will trigger the retry mechanism
            print(f"Error fetching author {author_id}: {str(e)}")
            if hasattr(response, 'text'):
                print(f"Response text: {response.text}")
            return None
    
    def fetch_authors_batch(
        self,
        author_ids: List[str]
    ) -> List[Optional[Dict[str, Any]]]:
        """
        Fetch multiple authors sequentially with rate limiting.
        
        Args:
            author_ids: List of Semantic Scholar author IDs to fetch
            
        Returns:
            List of author metadata dictionaries (None for failed fetches)
        """
        results = []
        total_authors = len(author_ids)
        
        for i, author_id in enumerate(author_ids, 1):
            print(f"Processing author {i}/{total_authors}")
            result = self.fetch_author_info(author_id)
            results.append(result)
        
        return results
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True
    )
    def search_author_by_name(self, name: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for authors by name.
        
        Args:
            name: Author name to search for
            limit: Maximum number of results to return
            
        Returns:
            List of author metadata dictionaries
        """
        self._enforce_rate_limit()
        
        params = {
            "query": name,
            "limit": limit
        }
        
        try:
            response = self.session.get(self.AUTHOR_SEARCH_URL_V2, params=params)
            
            if response.status_code == 404:
                print(f"No authors found for: {name}")
                return []
                
            if response.status_code == 403:
                print(f"Access forbidden. Response: {response.text}")
                return []
                
            response.raise_for_status()
            data = response.json()
            
            authors = []
            for author in data.get("data", []):
                authors.append({
                    "author_id": author.get("authorId", ""),
                    "name": author.get("name", ""),
                    "h_index": author.get("hIndex", 0),
                    "citation_count": author.get("citationCount", 0),
                    "paper_count": author.get("paperCount", 0)
                })
            
            return authors
            
        except requests.exceptions.RequestException as e:
            if response.status_code == 429:  # Rate limit exceeded
                print("Rate limit exceeded, backing off...")
                raise e  # This will trigger the retry mechanism
            print(f"Error searching for author {name}: {str(e)}")
            if hasattr(response, 'text'):
                print(f"Response text: {response.text}")
            return []
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        reraise=True
    )
    def fetch_paper_embedding(self, paper_id: str) -> Optional[List[float]]:
        """
        Fetch paper embedding from Semantic Scholar.
        
        Args:
            paper_id: Semantic Scholar paper ID
            
        Returns:
            List of embedding values or None if not found
        """
        self._enforce_rate_limit()
        
        url = self.PAPER_EMBEDDING_URL_V2.format(paper_id)
        
        try:
            response = self.session.get(url)
            
            if response.status_code == 404:
                print(f"Paper embedding not found: {paper_id}")
                return None
                
            if response.status_code == 403:
                print(f"Access forbidden. Response: {response.text}")
                return None
                
            response.raise_for_status()
            data = response.json()
            
            # Extract embedding
            embedding = data.get("embedding")
            if not embedding:
                print(f"No embedding available for paper: {paper_id}")
                return None
                
            return embedding
            
        except requests.exceptions.RequestException as e:
            if response.status_code == 429:  # Rate limit exceeded
                print("Rate limit exceeded, backing off...")
                raise e  # This will trigger the retry mechanism
            print(f"Error fetching embedding for paper {paper_id}: {str(e)}")
            if hasattr(response, 'text'):
                print(f"Response text: {response.text}")
            return None
    
    def fetch_paper_embeddings_batch(
        self,
        paper_ids: List[str],
        max_workers: int = 5
    ) -> Dict[str, Optional[List[float]]]:
        """
        Fetch embeddings for multiple papers using the batch API endpoint.
        
        Args:
            paper_ids: List of Semantic Scholar paper IDs to fetch
            max_workers: Maximum number of parallel workers (used as fallback)
            
        Returns:
            Dictionary mapping paper IDs to their embeddings (None for failed fetches)
        """
        # Use the batch API for efficiency (up to 500 papers at once)
        if len(paper_ids) <= 500:
            return self._fetch_paper_embeddings_batch_api(paper_ids)
        else:
            # For very large batches, split into chunks of 500
            results = {}
            for i in range(0, len(paper_ids), 500):
                chunk = paper_ids[i:i+500]
                print(f"Processing batch {i//500 + 1}/{(len(paper_ids) + 499)//500}")
                chunk_results = self._fetch_paper_embeddings_batch_api(chunk)
                results.update(chunk_results)
                # Sleep to avoid rate limiting between large batches
                time.sleep(self.delay_seconds * 5)  # Increased delay to respect API limits
            return results
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        reraise=True
    )
    def _fetch_paper_embeddings_batch_api(self, paper_ids: List[str]) -> Dict[str, Optional[List[float]]]:
        """
        Fetch embeddings for multiple papers using the batch API endpoint.
        
        Args:
            paper_ids: List of Semantic Scholar paper IDs to fetch (max 500)
            
        Returns:
            Dictionary mapping paper IDs to their embeddings
        """
        self._enforce_rate_limit()
        
        # Skip if no paper IDs
        if not paper_ids:
            print("No paper IDs provided to batch API.")
            return {}
            
        # Log the paper IDs we're requesting
        print(f"Requesting embeddings for paper IDs: {paper_ids[:5]}...")
        if len(paper_ids) > 5:
            print(f"...and {len(paper_ids) - 5} more")
        
        # Prepare results dictionary
        results = {paper_id: None for paper_id in paper_ids}
        
        try:
            # Use the batch API with embedding.specter_v2 field
            params = {
                'fields': 'embedding.specter_v2'  # Request the newer embedding version
            }
            
            # Make the batch request
            response = self.session.post(
                self.PAPER_BATCH_URL_V1,
                params=params,
                json={"ids": paper_ids}
            )
            
            if response.status_code == 429:
                print("Rate limit exceeded, backing off...")
                raise requests.exceptions.RequestException("Rate limit exceeded")
                
            response.raise_for_status()
            data = response.json()
            
            # Process the response
            for paper in data:
                paper_id = paper.get('paperId')
                if paper_id and 'embedding' in paper and paper['embedding'] is not None:
                    # Extract the vector from the embedding object
                    if 'vector' in paper['embedding']:
                        results[paper_id] = paper['embedding']['vector']
                    else:
                        print(f"No vector found in embedding for paper: {paper_id}")
                        print(f"Embedding data: {paper['embedding']}")
                    
            # Count successful fetches
            successful = sum(1 for emb in results.values() if emb is not None)
            print(f"Successfully fetched {successful}/{len(paper_ids)} embeddings via batch API.")
            
            return results
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching embeddings batch: {str(e)}")
            if 'response' in locals() and hasattr(response, 'text'):
                print(f"Response text: {response.text}")
            
            # Fall back to parallel processing if batch API fails
            print("Falling back to parallel processing...")
            return self._fetch_paper_embeddings_parallel(paper_ids, max_workers=5)
    
    def _fetch_paper_embeddings_parallel(
        self,
        paper_ids: List[str],
        max_workers: int = 5
    ) -> Dict[str, Optional[List[float]]]:
        """
        Fetch embeddings for multiple papers using parallel processing.
        
        Args:
            paper_ids: List of Semantic Scholar paper IDs to fetch
            max_workers: Maximum number of parallel workers
            
        Returns:
            Dictionary mapping paper IDs to their embeddings
        """
        results = {}
        total_papers = len(paper_ids)
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_id = {
                executor.submit(self.fetch_paper_embedding, paper_id): paper_id
                for paper_id in paper_ids
            }
            
            # Process results as they complete
            for i, future in enumerate(future_to_id, 1):
                paper_id = future_to_id[future]
                try:
                    print(f"Processing paper embedding {i}/{total_papers}")
                    embedding = future.result()
                    results[paper_id] = embedding
                except Exception as e:
                    print(f"Error processing paper {paper_id}: {str(e)}")
                    results[paper_id] = None
        
        return results
    
    async def fetch_paper_embedding_async(self, paper_id: str, session: ClientSession) -> Optional[List[float]]:
        """
        Fetch paper embedding asynchronously from Semantic Scholar.
        
        Args:
            paper_id: Semantic Scholar paper ID
            session: aiohttp ClientSession
            
        Returns:
            List of embedding values or None if not found
        """
        url = self.PAPER_EMBEDDING_URL_V2.format(paper_id)
        headers = {
            "x-api-key": self.api_key,
            "User-Agent": "Research Pipeline/1.0"
        }
        
        try:
            # Add delay for rate limiting
            await asyncio.sleep(self.delay_seconds)
            
            async with session.get(url, headers=headers) as response:
                if response.status == 404:
                    print(f"Paper embedding not found: {paper_id}")
                    return None
                    
                if response.status == 403:
                    print(f"Access forbidden for paper: {paper_id}")
                    return None
                
                if response.status == 429:
                    print("Rate limit exceeded, backing off...")
                    await asyncio.sleep(10)  # Back off for 10 seconds
                    return None
                
                response.raise_for_status()
                data = await response.json()
                
                # Extract embedding
                embedding = data.get("embedding")
                if not embedding:
                    print(f"No embedding available for paper: {paper_id}")
                    return None
                    
                return embedding
                
        except Exception as e:
            print(f"Error fetching embedding for paper {paper_id}: {str(e)}")
            return None
    
    async def fetch_paper_embeddings_async(
        self,
        paper_ids: List[str],
        max_concurrent: int = 5
    ) -> Dict[str, Optional[List[float]]]:
        """
        Fetch embeddings for multiple papers asynchronously.
        
        Args:
            paper_ids: List of Semantic Scholar paper IDs to fetch
            max_concurrent: Maximum number of concurrent requests
            
        Returns:
            Dictionary mapping paper IDs to their embeddings
        """
        results = {}
        
        # Create a semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async with ClientSession(connector=TCPConnector(ssl=False)) as session:
            async def fetch_with_semaphore(paper_id):
                async with semaphore:
                    return paper_id, await self.fetch_paper_embedding_async(paper_id, session)
            
            # Create tasks for all paper IDs
            tasks = [fetch_with_semaphore(paper_id) for paper_id in paper_ids]
            
            # Process results as they complete
            for i, task in enumerate(asyncio.as_completed(tasks), 1):
                paper_id, embedding = await task
                print(f"Processed paper embedding {i}/{len(paper_ids)}")
                results[paper_id] = embedding
        
        return results
    
    def compute_embedding_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
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
    
    def compute_novelty_score(
        self, 
        paper_embedding: List[float], 
        reference_embeddings: List[List[float]]
    ) -> float:
        """
        Compute novelty score based on embedding similarity.
        
        Args:
            paper_embedding: Embedding of the paper to evaluate
            reference_embeddings: List of embeddings for reference papers
            
        Returns:
            Novelty score between 0 and 1 (higher means more novel)
        """
        if not reference_embeddings:
            return 1.0  # Maximum novelty if no reference papers
            
        # Calculate similarities with all reference papers
        similarities = [
            self.compute_embedding_similarity(paper_embedding, ref_embedding)
            for ref_embedding in reference_embeddings
        ]
        
        # Average similarity (lower means more novel)
        avg_similarity = sum(similarities) / len(similarities)
        
        # Convert to novelty score (higher means more novel)
        novelty_score = 1.0 - avg_similarity
        
        return novelty_score
