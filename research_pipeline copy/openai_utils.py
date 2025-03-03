"""
OpenAI utilities for analyzing and scoring papers.
"""

from typing import Dict, Any, List, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import OpenAI
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from config import CONFIG  # Import CONFIG from config module

class OpenAIAnalyzer:
    """Class for analyzing papers using OpenAI's API."""
    
    def __init__(self, api_key: str, analysis_model: str, summary_model: str):
        """
        Initialize the OpenAI analyzer.
        
        Args:
            api_key: OpenAI API key
            analysis_model: Model to use for analysis (e.g. gpt-4o-2024-08-06)
            summary_model: Model to use for summaries (e.g. gpt-4o-mini-2024-07-18)
        """
        if not api_key or api_key == "your_openai_key_here":
            raise ValueError(
                "Invalid OpenAI API key. Please set OPENAI_API_KEY in your .env file. "
                "You can get your API key from: https://platform.openai.com/api-keys"
            )
            
        self.client = OpenAI(api_key=api_key)
        self.analysis_model = analysis_model
        self.summary_model = summary_model
        
        # Model settings
        self.max_tokens = CONFIG.get("max_output_tokens", 16384)
    
    def _validate_scores(self, scores: Dict[str, float]) -> bool:
        """Validate that scores are within expected ranges and different."""
        if not all(isinstance(v, (int, float)) for v in scores.values()):
            return False
        if not all(0 <= v <= 10 for v in scores.values()):
            return False
        # Check if at least two scores are different
        unique_scores = set(scores.values())
        return len(unique_scores) > 1
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def score_paper(self, title: str, abstract: str) -> Tuple[Dict[str, float], Dict[str, str]]:
        """
        Score a paper based on its title and abstract.
        
        Args:
            title: Paper title
            abstract: Paper abstract
            
        Returns:
            Tuple of (scores, explanations) where scores is a dictionary with normalized scores
            and explanations contains the reasoning for each score
        """
        try:
            prompt = f"""
            You are a business-focused research analyst evaluating papers for their practical value and real-world impact. Based on the title and abstract below, score and explain the paper on six dimensions:

            Title: {title}
            Abstract: {abstract}

            For each dimension below:
            1. Provide a score from 1-10 (use decimals for precision)
            2. Provide a clear, practical explanation for your score

            Please format your response exactly as follows:

            SCORES:
            Novelty: [score]/10
            Impact: [score]/10
            Technical Rigor: [score]/10
            Clarity: [score]/10
            Business: [score]/10
            AI Relevance: [score]/10

            EXPLANATIONS:
            Novelty: [explanation]
            Impact: [explanation]
            Technical Rigor: [explanation]
            Clarity: [explanation]
            Business: [explanation]
            AI Relevance: [explanation]

            KEY INSIGHTS:
            Market Opportunity:
            - [bullet points about commercial potential]

            Implementation Path:
            - [bullet points about how to apply this]

            Risk Factors:
            - [bullet points about challenges]

            Scoring Criteria:
            - Novelty: How different is this from existing market solutions? Consider practical differentiation, unique selling points, and competitive advantages.
            - Impact: What is the potential market impact? Consider market size, revenue potential, cost savings, and efficiency gains.
            - Technical Rigor: How ready is this for real-world use? Consider implementation complexity, resource requirements, and scalability.
            - Clarity: How easily can this be explained to stakeholders? Consider communication clarity, documentation quality, and ease of understanding.
            - Business: What is the commercial viability? Consider time-to-market, implementation costs, ROI potential, and industry adoption readiness.
            - AI Relevance: How applicable is this to current AI systems? Consider integration ease, compatibility with existing tools, and practical AI use cases.
            """

            response = self.client.chat.completions.create(
                model=self.analysis_model,
                messages=[{"role": "user", "content": prompt}]
            )

            content = response.choices[0].message.content
            
            # Parse scores
            scores = {}
            explanations = {}
            key_insights = {
                "market_opportunity": "",
                "implementation_path": "",
                "risk_factors": ""
            }
            
            # Split content into sections
            sections = content.split("\n\n")
            current_section = None
            
            for section in sections:
                if section.strip().startswith("SCORES:"):
                    lines = section.strip().split("\n")[1:]  # Skip the "SCORES:" header
                    for line in lines:
                        if ":" in line:
                            dimension, score = line.split(":", 1)
                            dimension = dimension.strip().lower().replace(" ", "_")
                            try:
                                # Extract numeric score before "/10"
                                score = float(score.strip().split("/")[0])
                                scores[dimension] = score
                            except (ValueError, IndexError):
                                continue
                
                elif section.strip().startswith("EXPLANATIONS:"):
                    lines = section.strip().split("\n")[1:]  # Skip the "EXPLANATIONS:" header
                    for line in lines:
                        if ":" in line:
                            dimension, explanation = line.split(":", 1)
                            dimension = dimension.strip().lower().replace(" ", "_")
                            explanations[dimension] = explanation.strip()
                
                elif section.strip().startswith("KEY INSIGHTS:"):
                    current_insight = None
                    lines = section.strip().split("\n")[1:]  # Skip the "KEY INSIGHTS:" header
                    for line in lines:
                        if line.strip().startswith("Market Opportunity:"):
                            current_insight = "market_opportunity"
                        elif line.strip().startswith("Implementation Path:"):
                            current_insight = "implementation_path"
                        elif line.strip().startswith("Risk Factors:"):
                            current_insight = "risk_factors"
                        elif line.strip().startswith("-") and current_insight:
                            key_insights[current_insight] += line.strip() + "\n"
            
            # Validate scores
            if not self._validate_scores(scores):
                print(f"Warning: Invalid or identical scores for paper: {title}")
                print(f"Scores: {scores}")
                return self._generate_fallback_scores(), {}
            
            # Normalize scores to 0-1 range
            normalized_scores = {
                key: value / 10.0 for key, value in scores.items()
            }
            
            return normalized_scores, explanations
            
        except Exception as e:
            if "authentication" in str(e).lower():
                print("\nError: Invalid OpenAI API key")
                print("Please check your .env file and ensure OPENAI_API_KEY is set correctly")
                print("You can get your API key from: https://platform.openai.com/api-keys")
                raise e
            print(f"\nError scoring paper: {str(e)}")
            return self._generate_fallback_scores(), {}
    
    def _generate_fallback_scores(self) -> Dict[str, float]:
        """Generate fallback scores when scoring fails."""
        return {
            "novelty": 0.5,
            "impact": 0.5,
            "technical_rigor": 0.5,
            "clarity": 0.5,
            "business": 0.5,
            "ai_relevance": 0.5
        }
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def generate_summary(self, title: str, abstract: str) -> str:
        """
        Generate a detailed summary of the paper.
        
        Args:
            title: Paper title
            abstract: Paper abstract
            
        Returns:
            Structured summary with key points and analysis
        """
        try:
            prompt = f"""
            You are a business-focused research analyst helping companies understand and apply cutting-edge research. Based on the title and abstract below, provide a practical analysis focusing on real-world applications:

            Title: {title}
            Abstract: {abstract}

            Structure your response in these three sections:

            # What's New
            - Explain the innovation in simple, clear terms
            - Focus on what makes this different from existing solutions
            - Highlight the key problem it solves
            - Use analogies or examples to make it understandable

            # Real World Uses
            - List 3-5 specific, concrete ways this could be used today
            - Include examples from different industries
            - Explain how existing businesses could implement this
            - Highlight any immediate practical benefits
            - Consider cost/benefit and implementation complexity

            # Business Impact
            - Identify which industries could benefit most
            - Estimate time-to-market for practical applications
            - List potential challenges or barriers to adoption
            - Suggest ways companies could start experimenting with this
            - Include rough cost implications (high/medium/low)

            Keep the language simple and practical. Focus on business value rather than technical details. Use bullet points for clarity.
            """

            response = self.client.chat.completions.create(
                model=self.summary_model,
                messages=[{"role": "user", "content": prompt}]
            )

            return response.choices[0].message.content.strip()
            
        except Exception as e:
            if "authentication" in str(e).lower():
                print("\nError: Invalid OpenAI API key")
                print("Please check your .env file and ensure OPENAI_API_KEY is set correctly")
                print("You can get your API key from: https://platform.openai.com/api-keys")
                raise e
            print(f"\nError generating summary: {str(e)}")
            return "Summary generation failed due to an error."
