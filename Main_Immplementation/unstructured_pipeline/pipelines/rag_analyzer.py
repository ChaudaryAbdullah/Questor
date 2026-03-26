import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List

try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    from utils.logger import Logger
except ImportError:
    from utils import Logger

class RagAnalyzer:
    def __init__(self):
        self.logger = Logger.get_logger(self.__class__.__name__)
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or api_key == "your_groq_api_key_here":
            self.logger.warning("GROQ_API_KEY not set. RagAnalyzer will be disabled.")
            self.client = None
        else:
            if Groq is None:
                self.logger.warning("groq library not installed. RagAnalyzer will be disabled. Install with pip install groq.")
                self.client = None
            else:
                self.client = Groq(api_key=api_key)
        
        # Load fraud patterns YAML
        self.patterns = {}
        yaml_path = Path(__file__).parent.parent / "fraud_patterns.yaml"
        if yaml_path.exists():
            with open(yaml_path, "r", encoding="utf-8") as f:
                self.patterns = yaml.safe_load(f)
        else:
            self.logger.warning(f"fraud_patterns.yaml not found at {yaml_path}")
                
    def analyze_document(self, content: str) -> List[Dict[str, Any]]:
        """
        Send document content to Groq API to analyze for fraud patterns based on YAML definition.
        Returns a list of rag_analysis dictionaries.
        """
        if not self.client:
            self.logger.warning("RagAnalyzer client not initialized, skipping RAG analysis.")
            return []
            
        # Due to context window limits, we might need to truncate the content for the API.
        max_chars = 12000 
        if len(content) > max_chars:
            content = content[:max_chars] + "... [TRUNCATED]"
            
        prompt = f"""
You are an expert forensic accountant and financial risk analyzer.
Given the following financial document text and the defined fraud patterns below, analyze the document for any indications of these fraud patterns.
Your output must be ONLY a valid JSON array of objects representing each identified risk/pattern. For each pattern found, output a JSON object with:
  "query": the name of the fraud pattern checked
  "fraud_indicators": an array of strings outlining the specific indicators found in the text
  "metadata": an object containing "risk_level" (must be one of: CRITICAL, HIGH, MEDIUM, LOW, or MINIMAL) and "sec_rule" if applicable.

If no fraud indicators are found, return a JSON object indicating low risk:
  [{{"query": "Overall Assessment", "fraud_indicators": [], "metadata": {{"risk_level": "MINIMAL"}}}}]

Fraud Patterns to look for:
{json.dumps(self.patterns.get('fraud_patterns', {}), indent=2)}

Document Text:
{content}
"""
        
        try:
            self.logger.info("Calling Groq API for RAG analysis...")
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a specialized JSON-only output agent. Never output text outside the JSON array."},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.1-8b-instant",
                temperature=0.1,
                max_tokens=2048
            )
            
            response_content = chat_completion.choices[0].message.content.strip()
            
            # Extract JSON from potential markdown blocks
            if response_content.startswith("```json"):
                response_content = response_content[7:]
            if response_content.startswith("```"):
                response_content = response_content[3:]
            if response_content.endswith("```"):
                response_content = response_content[:-3]
                
            response_content = response_content.strip()
            
            results = json.loads(response_content)
            if isinstance(results, dict):
                results = [results]
                
            self.logger.info(f"RAG analysis completed. Found {len(results)} indicators.")
            return results
                
        except Exception as e:
            self.logger.error(f"Failed to perform RAG analysis: {str(e)}")
            return []
