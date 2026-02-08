#!/usr/bin/env python
# File: examples/risk_scoring_example.py
"""
Example script demonstrating risk scoring and output formatting features
"""
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.risk_scorer import RiskScorer
from pipelines.output_formatter import OutputFormatter


def example_1_basic_risk_scoring():
    """Example 1: Basic risk scoring for a single document"""
    print("=" * 80)
    print("EXAMPLE 1: Basic Risk Scoring")
    print("=" * 80)
    
    # Initialize risk scorer
    scorer = RiskScorer()
    
    # Sample document with fraud indicators
    document = {
        'doc_id': 'FRAUD_2020_001',
        'content': '''
        The company engaged in fictitious revenue recognition through 
        special purpose entities. Management concealed off-balance sheet 
        liabilities totaling $500 million. Round-trip transactions were 
        used to manipulate earnings. The restatement revealed material 
        weaknesses in internal controls.
        ''',
        'label': 'Fraud',
        'company_id': 'COMP_001'
    }
    
    # Sample entities (would normally come from NER)
    entities = {
        'FRAUD_INDICATOR': [
            {'text': 'fictitious', 'label': 'FRAUD_INDICATOR'},
            {'text': 'concealed', 'label': 'FRAUD_INDICATOR'},
            {'text': 'manipulation', 'label': 'FRAUD_INDICATOR'}
        ],
        'FINANCIAL_TERM': [
            {'text': 'special purpose entities', 'label': 'FINANCIAL_TERM'},
            {'text': 'off-balance sheet', 'label': 'FINANCIAL_TERM'},
            {'text': 'round-trip', 'label': 'FINANCIAL_TERM'}
        ],
        'MONEY': [
            {'text': '$500 million', 'label': 'MONEY'}
        ]
    }
    
    # Sample relationships
    relationships = [
        {
            'subject': 'company',
            'predicate': 'concealed',
            'object': 'liabilities',
            'relation_type': 'CONCEALED'
        },
        {
            'subject': 'management',
            'predicate': 'manipulated',
            'object': 'earnings',
            'relation_type': 'FRAUD_ACTION'
        }
    ]
    
    # Calculate risk score
    risk_data = scorer.calculate_document_risk(
        document=document,
        entities=entities,
        relationships=relationships
    )
    
    # Display results
    print(f"\nDocument: {risk_data['doc_id']}")
    print(f"Overall Risk Score: {risk_data['overall_risk_score']:.2f}")
    print(f"Risk Level: {risk_data['risk_level']}")
    print(f"\nComponent Scores:")
    for component, score in risk_data['component_scores'].items():
        print(f"  {component}: {score:.2f}")
    print(f"\nRisk Factors:")
    for factor in risk_data['risk_factors']:
        print(f"  - {factor}")
    print()


def example_2_output_formatting():
    """Example 2: Formatting output for multiagent system"""
    print("=" * 80)
    print("EXAMPLE 2: Output Formatting for Multiagent System")
    print("=" * 80)
    
    # Initialize components
    scorer = RiskScorer()
    formatter = OutputFormatter()
    
    # Sample document
    document = {
        'doc_id': 'UNKNOWN_2019_002',
        'content': 'Company disclosed derivative transactions with related parties...',
        'label': 'Unknown',
        'company_id': 'COMP_002',
        'file_path': 'data/sample.txt'
    }
    
    entities = {
        'FINANCIAL_TERM': [
            {'text': 'derivative', 'label': 'FINANCIAL_TERM'},
            {'text': 'related parties', 'label': 'FINANCIAL_TERM'}
        ]
    }
    
    relationships = []
    
    chunks = [
        {'chunk_id': 'chunk_001', 'doc_id': 'UNKNOWN_2019_002', 'text': '...'},
        {'chunk_id': 'chunk_002', 'doc_id': 'UNKNOWN_2019_002', 'text': '...'}
    ]
    
    # Calculate risk
    risk_data = scorer.calculate_document_risk(
        document=document,
        entities=entities,
        relationships=relationships
    )
    
    # Format for multiagent system
    formatted_output = formatter.format_for_multiagent(
        document=document,
        risk_data=risk_data,
        entities=entities,
        relationships=relationships,
        chunks=chunks
    )
    
    # Display formatted structure
    print(f"\nFormatted Output Structure:")
    print(f"  Document ID: {formatted_output['document_id']}")
    print(f"  Risk Score: {formatted_output['risk_assessment']['overall_score']:.2f}")
    print(f"  Risk Level: {formatted_output['risk_assessment']['risk_level']}")
    print(f"  Requires Investigation: {formatted_output['risk_assessment']['requires_investigation']}")
    print(f"\n  Recommended Agents:")
    for agent in formatted_output['agent_routing']['recommended_agents']:
        print(f"    - {agent}")
    print(f"  Priority: {formatted_output['agent_routing']['priority']}")
    print(f"\n  Retrieval References:")
    print(f"    Vector DB Chunks: {len(formatted_output['retrieval_references']['vector_db_chunks'])}")
    print(f"    Knowledge Graph Nodes: {len(formatted_output['retrieval_references']['knowledge_graph_nodes'])}")
    print()


def example_3_batch_processing():
    """Example 3: Batch processing with summary"""
    print("=" * 80)
    print("EXAMPLE 3: Batch Processing with Summary")
    print("=" * 80)
    
    # Initialize components
    scorer = RiskScorer()
    formatter = OutputFormatter()
    
    # Sample batch of documents
    documents = [
        {
            'doc_id': 'DOC_001',
            'content': 'Fictitious revenue and concealed liabilities...',
            'label': 'Fraud'
        },
        {
            'doc_id': 'DOC_002',
            'content': 'Standard financial disclosure...',
            'label': 'NonFraud'
        },
        {
            'doc_id': 'DOC_003',
            'content': 'Special purpose entity with round-trip transactions...',
            'label': 'Unknown'
        }
    ]
    
    # Process each document
    formatted_docs = []
    for doc in documents:
        # Simplified - normally would have real entities/relationships
        entities = {'FRAUD_INDICATOR': []} if 'Fictitious' in doc['content'] else {}
        relationships = []
        
        risk_data = scorer.calculate_document_risk(
            document=doc,
            entities=entities,
            relationships=relationships
        )
        
        formatted = formatter.format_for_multiagent(
            document=doc,
            risk_data=risk_data,
            entities=entities,
            relationships=relationships,
            chunks=[]
        )
        
        formatted_docs.append({
            'document': doc,
            'risk_data': risk_data,
            'entities': entities,
            'relationships': relationships,
            'chunks': []
        })
    
    # Create batch output
    batch_output = formatter.format_batch_for_multiagent(formatted_docs)
    
    # Display summary
    print(f"\nBatch Summary:")
    summary = batch_output['summary_statistics']
    print(f"  Total Documents: {summary['total_documents']}")
    print(f"  Average Risk Score: {summary['average_risk_score']:.2f}")
    print(f"  Max Risk Score: {summary['max_risk_score']:.2f}")
    print(f"  Min Risk Score: {summary['min_risk_score']:.2f}")
    print(f"\n  Risk Distribution:")
    for level, count in summary['risk_level_distribution'].items():
        print(f"    {level}: {count}")
    print(f"\n  High-Risk Documents: {summary['high_risk_count']}")
    
    # Display high-risk documents
    if batch_output['high_risk_documents']:
        print(f"\n  High-Risk Document Details:")
        for doc in batch_output['high_risk_documents']:
            print(f"    - {doc['document_id']}: {doc['risk_score']:.2f} ({doc['risk_level']})")
    print()


def example_4_custom_risk_analysis():
    """Example 4: Custom risk analysis scenarios"""
    print("=" * 80)
    print("EXAMPLE 4: Custom Risk Analysis Scenarios")
    print("=" * 80)
    
    scorer = RiskScorer()
    
    # Scenario 1: High fraud indicators
    print("\n[Scenario 1: High Fraud Indicators]")
    doc1 = {
        'doc_id': 'SCENARIO_1',
        'content': 'Fabricated financial statements with material misstatement and concealment of liabilities'
    }
    entities1 = {
        'FRAUD_INDICATOR': [
            {'text': 'fabricated', 'label': 'FRAUD_INDICATOR'},
            {'text': 'misstatement', 'label': 'FRAUD_INDICATOR'},
            {'text': 'concealment', 'label': 'FRAUD_INDICATOR'}
        ]
    }
    risk1 = scorer.calculate_document_risk(doc1, entities1, [])
    print(f"Risk Score: {risk1['overall_risk_score']:.2f} ({risk1['risk_level']})")
    
    # Scenario 2: Complex financial structures
    print("\n[Scenario 2: Complex Financial Structures]")
    doc2 = {
        'doc_id': 'SCENARIO_2',
        'content': 'Multiple subsidiaries, joint ventures, and special purpose entities with derivative transactions'
    }
    entities2 = {
        'FINANCIAL_TERM': [
            {'text': 'subsidiaries', 'label': 'FINANCIAL_TERM'},
            {'text': 'joint ventures', 'label': 'FINANCIAL_TERM'},
            {'text': 'special purpose entities', 'label': 'FINANCIAL_TERM'},
            {'text': 'derivative', 'label': 'FINANCIAL_TERM'}
        ],
        'ORG': [
            {'text': 'Company A', 'label': 'ORG'},
            {'text': 'Company B', 'label': 'ORG'},
            {'text': 'Company C', 'label': 'ORG'}
        ]
    }
    risk2 = scorer.calculate_document_risk(doc2, entities2, [])
    print(f"Risk Score: {risk2['overall_risk_score']:.2f} ({risk2['risk_level']})")
    
    # Scenario 3: Large financial amounts
    print("\n[Scenario 3: Large Financial Amounts]")
    doc3 = {
        'doc_id': 'SCENARIO_3',
        'content': 'Transactions totaling $500 million, $250 million, and $100 million were recorded'
    }
    entities3 = {
        'MONEY': [
            {'text': '$500 million', 'label': 'MONEY'},
            {'text': '$250 million', 'label': 'MONEY'},
            {'text': '$100 million', 'label': 'MONEY'}
        ]
    }
    risk3 = scorer.calculate_document_risk(doc3, entities3, [])
    print(f"Risk Score: {risk3['overall_risk_score']:.2f} ({risk3['risk_level']})")
    
    # Scenario 4: Suspicious relationships
    print("\n[Scenario 4: Suspicious Relationships]")
    doc4 = {
        'doc_id': 'SCENARIO_4',
        'content': 'Company concealed assets and transferred funds to offshore entities'
    }
    relationships4 = [
        {'subject': 'Company', 'predicate': 'concealed', 'object': 'assets', 'relation_type': 'CONCEALED'},
        {'subject': 'Company', 'predicate': 'transferred', 'object': 'funds', 'relation_type': 'TRANSFERRED'}
    ]
    risk4 = scorer.calculate_document_risk(doc4, {}, relationships4)
    print(f"Risk Score: {risk4['overall_risk_score']:.2f} ({risk4['risk_level']})")
    print()


def main():
    """Run all examples"""
    print("\n" + "=" * 80)
    print("RISK SCORING AND OUTPUT FORMATTING EXAMPLES")
    print("=" * 80 + "\n")
    
    try:
        example_1_basic_risk_scoring()
        example_2_output_formatting()
        example_3_batch_processing()
        example_4_custom_risk_analysis()
        
        print("=" * 80)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError running examples: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
