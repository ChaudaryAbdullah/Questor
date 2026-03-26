import sys
import os
import json
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipelines.rag_analyzer import RagAnalyzer

def main():
    print("Testing RagAnalyzer...")
    analyzer = RagAnalyzer()
    
    if not analyzer.client:
        print("Failed to initialize client. Check GROQ_API_KEY")
        return
        
    sample_text = """
    In Q4 2025, our subsidiary ITC Ventures reported significant off-balance sheet growth, 
    but this was not fully consolidated in Note 5 of our latest SEC filings due to complex 
    ownership structures managed by our CFO's private holding company. Furthermore, we recognized 
    $50M in advance revenue from a related party supplier, creating a temporary circular transaction cycle.
    """
    
    print("Running document analysis...")
    results = analyzer.analyze_document(sample_text)
    
    print(json.dumps(results, indent=2))
    
if __name__ == "__main__":
    main()
