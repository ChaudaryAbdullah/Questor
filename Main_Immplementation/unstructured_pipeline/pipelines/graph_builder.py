
# File: pipelines/graph_builder.py
"""
Knowledge Graph Construction
"""
from typing import List, Dict, Any
from tqdm import tqdm

from databases import GraphDatabase
from utils import Logger
from utils.exceptions import GraphDBError


class GraphBuilder:
    """Builds knowledge graph from extracted entities and relationships"""
    
    def __init__(self):
        self.logger = Logger.get_logger(self.__class__.__name__)
        self.graph_db = GraphDatabase()
    
    def build_graph_from_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """
        Build knowledge graph from processed documents
        
        Args:
            documents: List of documents with entities and relationships
            
        Returns:
            Statistics about graph construction
        """
        self.logger.info(f"Building knowledge graph from {len(documents)} documents...")
        
        # Create indexes for performance
        self.graph_db.create_indexes()
        
        stats = {
            'entities_added': 0,
            'relationships_added': 0,
            'documents_processed': 0
        }
        
        for doc in tqdm(documents, desc="Building graph"):
            try:
                # Add document node
                self._add_document_node(doc)
                
                # Add entities
                entities_added = self._add_entities_to_graph(doc)
                stats['entities_added'] += entities_added
                
                # Add relationships
                relationships_added = self._add_relationships_to_graph(doc)
                stats['relationships_added'] += relationships_added
                
                stats['documents_processed'] += 1
                
            except Exception as e:
                self.logger.warning(f"Failed to process doc {doc.get('doc_id')}: {str(e)}")
                continue
        
        self.logger.info(f"Graph construction complete: {stats}")
        return stats
    
    def _add_document_node(self, doc: Dict[str, Any]):
        """Add document as a node in the graph"""
        properties = {
            'doc_id': doc['doc_id'],
            'file_name': doc.get('file_name', ''),
            'label': doc.get('label', 'unknown'),
            'company_id': doc.get('company_id', ''),
            'date': doc.get('date', '')
        }
        
        self.graph_db.add_entity('Document', doc['doc_id'], properties)
    
    def _add_entities_to_graph(self, doc: Dict[str, Any]) -> int:
        """Add extracted entities to the graph"""
        entities = doc.get('entities', {})
        count = 0
        
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                entity_name = entity['text']
                
                # Map spaCy labels to our entity types
                graph_entity_type = self._map_entity_type(entity_type)
                
                properties = {
                    'name': entity_name,
                    'type': entity_type,
                    'source_doc': doc['doc_id']
                }
                
                try:
                    self.graph_db.add_entity(graph_entity_type, entity_name, properties)
                    
                    # Link entity to document
                    self.graph_db.add_relationship(
                        doc['doc_id'], 'Document',
                        entity_name, graph_entity_type,
                        'MENTIONS'
                    )
                    
                    count += 1
                except Exception as e:
                    self.logger.debug(f"Failed to add entity {entity_name}: {str(e)}")
        
        return count
    
    def _add_relationships_to_graph(self, doc: Dict[str, Any]) -> int:
        """Add extracted relationships to the graph"""
        relationships = doc.get('relationships', [])
        count = 0
        
        for rel in relationships:
            try:
                subject = rel['subject']
                obj = rel['object']
                rel_type = rel['relation_type']
                
                # Try to find entity types
                subject_type = 'Entity'
                object_type = 'Entity'
                
                self.graph_db.add_relationship(
                    subject, subject_type,
                    obj, object_type,
                    rel_type,
                    {'predicate': rel.get('predicate', '')}
                )
                
                count += 1
            except Exception as e:
                self.logger.debug(f"Failed to add relationship: {str(e)}")
        
        return count
    
    def _map_entity_type(self, spacy_label: str) -> str:
        """Map spaCy entity labels to graph node types"""
        mapping = {
            'PERSON': 'Person',
            'ORG': 'Company',
            'MONEY': 'MonetaryAmount',
            'DATE': 'Date',
            'GPE': 'Location',
            'FINANCIAL_TERM': 'FinancialTerm',
            'FRAUD_INDICATOR': 'FraudIndicator'
        }
        
        return mapping.get(spacy_label, 'Entity')
    
    def close(self):
        """Close graph database connection"""
        self.graph_db.close()



