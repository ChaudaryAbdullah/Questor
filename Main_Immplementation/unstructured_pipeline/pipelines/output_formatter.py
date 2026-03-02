# File: pipelines/output_formatter.py
"""
Output Formatter for Multiagent System Integration
Formats processed data for consumption by downstream multiagent systems
"""
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
from datetime import datetime

try:
    from utils import Config, Logger
except ImportError:
    from utils.config import Config
    from utils.logger import Logger


class OutputFormatter:
    """
    Formats pipeline output for multiagent system consumption
    
    Provides structured data including:
    1. Document metadata and content
    2. Risk scores and assessments
    3. Extracted entities and relationships
    4. Vector embeddings references
    5. Knowledge graph node references
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize output formatter
        
        Args:
            output_dir: Directory to save formatted outputs (optional)
        """
        self.logger = Logger.get_logger(self.__class__.__name__)
        self.output_dir = output_dir or Config.DATA_DIR.parent / 'output'
        self.output_dir.mkdir(exist_ok=True)
        self.logger.info(f"Output formatter initialized. Output dir: {self.output_dir}")
    
    def format_for_multiagent(
        self,
        document: Dict[str, Any],
        risk_data: Dict[str, Any],
        entities: Optional[Dict[str, List[Dict]]] = None,
        relationships: Optional[List[Dict]] = None,
        chunks: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Format a single document's data for multiagent system
        
        Args:
            document: Original document data
            risk_data: Risk scoring results
            entities: Extracted entities
            relationships: Extracted relationships
            chunks: Text chunks with metadata
            
        Returns:
            Formatted dictionary ready for multiagent consumption
        """
        doc_id = document.get('doc_id', 'unknown')
        
        formatted_output = {
            # Document identification
            'document_id': doc_id,
            'timestamp': datetime.now().isoformat(),
            'source_file': str(document.get('file_path', '')),
            
            # Document metadata
            'metadata': {
                'label': document.get('label', 'unknown'),
                'company_id': document.get('company_id'),
                'date': document.get('date'),
                'content_length': len(document.get('content', '')),
                'chunk_count': len(chunks) if chunks else 0
            },
            
            # Risk assessment
            'risk_assessment': {
                'overall_score': risk_data.get('overall_risk_score', 0.0),
                'risk_level': risk_data.get('risk_level', 'UNKNOWN'),
                'component_scores': risk_data.get('component_scores', {}),
                'risk_factors': risk_data.get('risk_factors', []),
                'requires_investigation': risk_data.get('overall_risk_score', 0) >= 60
            },
            
            # Extracted intelligence
            'extracted_data': {
                'entities': self._format_entities(entities),
                'relationships': self._format_relationships(relationships),
                'entity_summary': self._create_entity_summary(entities)
            },
            
            # References for retrieval
            'retrieval_references': {
                'vector_db_chunks': [chunk.get('chunk_id') for chunk in chunks] if chunks else [],
                'knowledge_graph_nodes': self._extract_graph_node_ids(entities),
                'embedding_model': Config.EMBEDDING_MODEL
            },
            
            # Multiagent routing hints
            'agent_routing': self._determine_agent_routing(risk_data, entities, relationships)
        }
        
        return formatted_output
    
    def format_batch_for_multiagent(
        self,
        documents_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Format multiple documents for multiagent system
        
        Args:
            documents_data: List of document data dictionaries
            
        Returns:
            Batch formatted output with summary statistics
        """
        formatted_documents = []
        
        for doc_data in documents_data:
            try:
                formatted_doc = self.format_for_multiagent(
                    document=doc_data.get('document', {}),
                    risk_data=doc_data.get('risk_data', {}),
                    entities=doc_data.get('entities'),
                    relationships=doc_data.get('relationships'),
                    chunks=doc_data.get('chunks')
                )
                formatted_documents.append(formatted_doc)
            except Exception as e:
                self.logger.error(f"Failed to format document: {str(e)}")
                continue
        
        # Create batch summary
        batch_output = {
            'batch_metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_documents': len(formatted_documents),
                'processing_pipeline': 'unstructured_pipeline',
                'version': '1.0.0'
            },
            'summary_statistics': self._create_batch_summary(formatted_documents),
            'documents': formatted_documents,
            'high_risk_documents': self._filter_high_risk_documents(formatted_documents)
        }
        
        return batch_output
    
    def _format_entities(
        self,
        entities: Optional[Dict[str, List[Dict]]]
    ) -> Dict[str, List[Dict]]:
        """Format entities for multiagent consumption"""
        if not entities:
            return {}
        
        formatted_entities = {}
        
        for entity_type, entity_list in entities.items():
            formatted_entities[entity_type] = [
                {
                    'text': ent.get('text', ''),
                    'label': ent.get('label', entity_type),
                    'position': {
                        'start': ent.get('start', 0),
                        'end': ent.get('end', 0)
                    }
                }
                for ent in entity_list
            ]
        
        return formatted_entities
    
    def _format_relationships(
        self,
        relationships: Optional[List[Dict]]
    ) -> List[Dict]:
        """Format relationships for multiagent consumption"""
        if not relationships:
            return []
        
        formatted_relationships = [
            {
                'subject': rel.get('subject', ''),
                'predicate': rel.get('predicate', ''),
                'object': rel.get('object', ''),
                'type': rel.get('relation_type', 'RELATED_TO')
            }
            for rel in relationships
        ]
        
        return formatted_relationships
    
    def _create_entity_summary(
        self,
        entities: Optional[Dict[str, List[Dict]]]
    ) -> Dict[str, Any]:
        """Create summary statistics for entities"""
        if not entities:
            return {
                'total_entities': 0,
                'entity_types': {},
                'key_entities': []
            }
        
        entity_counts = {
            entity_type: len(entity_list)
            for entity_type, entity_list in entities.items()
        }
        
        # Extract key entities (most important ones)
        key_entities = []
        
        # Prioritize fraud indicators and financial terms
        priority_types = ['FRAUD_INDICATOR', 'FINANCIAL_TERM', 'ORG', 'MONEY']
        for entity_type in priority_types:
            if entity_type in entities:
                key_entities.extend([
                    {'type': entity_type, 'text': ent.get('text', '')}
                    for ent in entities[entity_type][:5]  # Top 5 of each type
                ])
        
        return {
            'total_entities': sum(entity_counts.values()),
            'entity_types': entity_counts,
            'key_entities': key_entities[:10]  # Top 10 overall
        }
    
    def _extract_graph_node_ids(
        self,
        entities: Optional[Dict[str, List[Dict]]]
    ) -> List[str]:
        """Extract knowledge graph node identifiers"""
        if not entities:
            return []
        
        node_ids = []
        
        # Create node IDs from entities (simplified)
        for entity_type, entity_list in entities.items():
            for ent in entity_list:
                # Create a unique node ID
                node_id = f"{entity_type}:{ent.get('text', '').replace(' ', '_')}"
                node_ids.append(node_id)
        
        return list(set(node_ids))  # Remove duplicates
    
    def _determine_agent_routing(
        self,
        risk_data: Dict[str, Any],
        entities: Optional[Dict],
        relationships: Optional[List[Dict]]
    ) -> Dict[str, Any]:
        """
        Determine which agents should process this document
        
        Returns routing hints for multiagent system
        """
        routing = {
            'recommended_agents': [],
            'priority': 'normal',
            'processing_hints': []
        }
        
        risk_score = risk_data.get('overall_risk_score', 0)
        
        # Determine priority based on risk
        if risk_score >= 80:
            routing['priority'] = 'critical'
            routing['recommended_agents'].append('fraud_investigation_agent')
        elif risk_score >= 60:
            routing['priority'] = 'high'
            routing['recommended_agents'].append('risk_assessment_agent')
        else:
            routing['priority'] = 'normal'
        
        # NLP Disclosure Agent - for complex text analysis
        if entities and len(entities.get('FINANCIAL_TERM', [])) > 5:
            routing['recommended_agents'].append('nlp_disclosure_agent')
            routing['processing_hints'].append('complex_financial_language')
        
        # Graph Linkage Agent - for relationship analysis
        if relationships and len(relationships) > 3:
            routing['recommended_agents'].append('graph_linkage_agent')
            routing['processing_hints'].append('complex_entity_relationships')
        
        # Fraud indicators present
        if entities and 'FRAUD_INDICATOR' in entities:
            routing['recommended_agents'].append('fraud_detection_agent')
            routing['processing_hints'].append('fraud_indicators_present')
        
        # Default agent if none specified
        if not routing['recommended_agents']:
            routing['recommended_agents'].append('general_analysis_agent')
        
        return routing
    
    def _create_batch_summary(
        self,
        formatted_documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create summary statistics for a batch of documents"""
        if not formatted_documents:
            return {}
        
        risk_scores = [
            doc.get('risk_assessment', {}).get('overall_score', 0)
            for doc in formatted_documents
        ]
        
        risk_levels = [
            doc.get('risk_assessment', {}).get('risk_level', 'UNKNOWN')
            for doc in formatted_documents
        ]
        
        # Count risk levels
        risk_level_counts = {}
        for level in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'MINIMAL']:
            risk_level_counts[level] = risk_levels.count(level)
        
        return {
            'total_documents': len(formatted_documents),
            'average_risk_score': round(sum(risk_scores) / len(risk_scores), 2) if risk_scores else 0,
            'max_risk_score': max(risk_scores) if risk_scores else 0,
            'min_risk_score': min(risk_scores) if risk_scores else 0,
            'risk_level_distribution': risk_level_counts,
            'high_risk_count': risk_level_counts.get('CRITICAL', 0) + risk_level_counts.get('HIGH', 0)
        }
    
    def _filter_high_risk_documents(
        self,
        formatted_documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter and return only high-risk documents"""
        high_risk_docs = [
            {
                'document_id': doc.get('document_id'),
                'risk_score': doc.get('risk_assessment', {}).get('overall_score', 0),
                'risk_level': doc.get('risk_assessment', {}).get('risk_level'),
                'risk_factors': doc.get('risk_assessment', {}).get('risk_factors', [])
            }
            for doc in formatted_documents
            if doc.get('risk_assessment', {}).get('overall_score', 0) >= 60
        ]
        
        # Sort by risk score (descending)
        high_risk_docs.sort(key=lambda x: x['risk_score'], reverse=True)
        
        return high_risk_docs
    
    def save_to_json(
        self,
        data: Dict[str, Any],
        filename: str
    ) -> Path:
        """
        Save formatted data to JSON file
        
        Args:
            data: Formatted data dictionary
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Saved output to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to save output: {str(e)}")
            raise
    
    def save_batch_output(
        self,
        batch_data: Dict[str, Any],
        batch_name: Optional[str] = None
    ) -> Path:
        """
        Save batch output with timestamp
        
        Args:
            batch_data: Batch formatted data
            batch_name: Optional batch identifier
            
        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        batch_id = batch_name or 'batch'
        filename = f"{batch_id}_{timestamp}.json"
        
        return self.save_to_json(batch_data, filename)
    
    def create_summary_report(
        self,
        batch_data: Dict[str, Any]
    ) -> str:
        """
        Create human-readable summary report
        
        Args:
            batch_data: Batch formatted data
            
        Returns:
            Formatted text report
        """
        summary = batch_data.get('summary_statistics', {})
        high_risk = batch_data.get('high_risk_documents', [])
        
        report = f"""
========================================
FRAUD DETECTION PIPELINE - SUMMARY REPORT
========================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERALL STATISTICS
------------------
Total Documents Processed: {summary.get('total_documents', 0)}
Average Risk Score: {summary.get('average_risk_score', 0):.2f}
Maximum Risk Score: {summary.get('max_risk_score', 0):.2f}
Minimum Risk Score: {summary.get('min_risk_score', 0):.2f}

RISK DISTRIBUTION
-----------------
"""
        
        risk_dist = summary.get('risk_level_distribution', {})
        for level in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'MINIMAL']:
            count = risk_dist.get(level, 0)
            report += f"{level:12s}: {count:4d} documents\n"
        
        report += f"\nHIGH-RISK DOCUMENTS: {len(high_risk)}\n"
        report += "=" * 40 + "\n"
        
        if high_risk:
            report += "\nTOP HIGH-RISK DOCUMENTS:\n"
            report += "-" * 40 + "\n"
            
            for i, doc in enumerate(high_risk[:10], 1):
                report += f"\n{i}. Document: {doc.get('document_id')}\n"
                report += f"   Risk Score: {doc.get('risk_score'):.2f} ({doc.get('risk_level')})\n"
                report += f"   Risk Factors:\n"
                for factor in doc.get('risk_factors', [])[:3]:
                    report += f"   - {factor}\n"
        
        report += "\n" + "=" * 40 + "\n"
        report += "END OF REPORT\n"
        report += "=" * 40 + "\n"
        
        return report
