"""
Knowledge Graph Builder Module
Constructs property graphs from extracted entities using Neo4j.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import os

from ..utils.logger import get_graph_logger
from ..utils.config_manager import get_config


@dataclass
class GraphNode:
    """Represents a graph node."""
    id: str
    labels: List[str]
    properties: Dict[str, Any]


@dataclass
class GraphRelationship:
    """Represents a graph relationship."""
    id: str
    type: str
    source_id: str
    target_id: str
    properties: Dict[str, Any]


class KnowledgeGraphBuilder:
    """
    Builds knowledge graphs from extracted entities using Neo4j.
    Falls back to NetworkX if Neo4j unavailable.
    """
    
    def __init__(self, use_neo4j: bool = True):
        """Initialize graph builder."""
        self.logger = get_graph_logger()
        self.use_neo4j = use_neo4j
        
        # Load config
        try:
            config = get_config()
            self.graph_config = config.graph_db_config
            self.neo4j_uri = config.neo4j_uri
            self.neo4j_user, self.neo4j_password = config.neo4j_credentials
        except Exception:
            self.graph_config = {}
            self.neo4j_uri = "bolt://localhost:7687"
            self.neo4j_user = "neo4j"
            self.neo4j_password = ""
        
        self._driver = None
        self._nx_graph = None
        self._initialized = False
        
        self.logger.info("KnowledgeGraphBuilder initialized")
    
    def _init_neo4j(self) -> None:
        """Initialize Neo4j driver."""
        if self._initialized:
            return
        
        try:
            from neo4j import GraphDatabase
            
            if not self.neo4j_password:
                raise ValueError("Neo4j password not set")
            
            self._driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password)
            )
            
            # Test connection
            self._driver.verify_connectivity()
            self._initialized = True
            self.logger.info(f"Neo4j connected: {self.neo4j_uri}")
            
        except Exception as e:
            self.logger.warning(f"Neo4j unavailable: {e}. Using NetworkX fallback.")
            self.use_neo4j = False
            self._init_networkx()
    
    def _init_networkx(self) -> None:
        """Initialize NetworkX graph."""
        try:
            import networkx as nx
            self._nx_graph = nx.MultiDiGraph()
            self._initialized = True
            self.logger.info("NetworkX graph initialized")
        except ImportError:
            self.logger.error("NetworkX not installed")
            raise
    
    def create_nodes(self, entities: List[Dict]) -> List[str]:
        """Create nodes from entities."""
        if self.use_neo4j:
            return self._create_nodes_neo4j(entities)
        else:
            return self._create_nodes_nx(entities)
    
    def _create_nodes_neo4j(self, entities: List[Dict]) -> List[str]:
        """Create nodes in Neo4j."""
        self._init_neo4j()
        node_ids = []
        
        with self._driver.session() as session:
            for entity in entities:
                entity_id = entity.get("id", "")
                entity_type = entity.get("entity_type", "UNKNOWN")
                properties = {k: v for k, v in entity.items() 
                            if k not in ["id", "entity_type"] and isinstance(v, (str, int, float, bool))}
                
                query = f"""
                MERGE (n:{entity_type} {{id: $id}})
                SET n += $properties
                RETURN n.id as id
                """
                
                result = session.run(query, id=entity_id, properties=properties)
                record = result.single()
                if record:
                    node_ids.append(record["id"])
        
        self.logger.debug(f"Created {len(node_ids)} nodes in Neo4j")
        return node_ids
    
    def _create_nodes_nx(self, entities: List[Dict]) -> List[str]:
        """Create nodes in NetworkX."""
        if self._nx_graph is None:
            self._init_networkx()
        
        node_ids = []
        for entity in entities:
            entity_id = entity.get("id", "")
            self._nx_graph.add_node(entity_id, **entity)
            node_ids.append(entity_id)
        
        return node_ids
    
    def create_relationships(self, relationships: List[Dict]) -> List[str]:
        """Create relationships."""
        if self.use_neo4j:
            return self._create_relationships_neo4j(relationships)
        else:
            return self._create_relationships_nx(relationships)
    
    def _create_relationships_neo4j(self, relationships: List[Dict]) -> List[str]:
        """Create relationships in Neo4j."""
        self._init_neo4j()
        rel_ids = []
        
        with self._driver.session() as session:
            for rel in relationships:
                source_id = rel.get("source_entity_id", "")
                target_id = rel.get("target_entity_id", "")
                rel_type = rel.get("relationship_type", "RELATED_TO")
                properties = rel.get("properties", {})
                
                query = f"""
                MATCH (a {{id: $source_id}})
                MATCH (b {{id: $target_id}})
                MERGE (a)-[r:{rel_type}]->(b)
                SET r += $properties
                RETURN id(r) as rel_id
                """
                
                result = session.run(query, source_id=source_id, target_id=target_id, properties=properties)
                record = result.single()
                if record:
                    rel_ids.append(str(record["rel_id"]))
        
        self.logger.debug(f"Created {len(rel_ids)} relationships in Neo4j")
        return rel_ids
    
    def _create_relationships_nx(self, relationships: List[Dict]) -> List[str]:
        """Create relationships in NetworkX."""
        if self._nx_graph is None:
            self._init_networkx()
        
        rel_ids = []
        for i, rel in enumerate(relationships):
            source_id = rel.get("source_entity_id", "")
            target_id = rel.get("target_entity_id", "")
            rel_type = rel.get("relationship_type", "RELATED_TO")
            
            self._nx_graph.add_edge(source_id, target_id, type=rel_type, **rel)
            rel_ids.append(f"rel_{i}")
        
        return rel_ids
    
    def search_nodes(self, keyword: str, limit: int = 10) -> List[Dict]:
        """Search for nodes by keyword."""
        if self.use_neo4j:
            return self._search_nodes_neo4j(keyword, limit)
        else:
            return self._search_nodes_nx(keyword, limit)
    
    def _search_nodes_neo4j(self, keyword: str, limit: int) -> List[Dict]:
        """Search nodes in Neo4j."""
        self._init_neo4j()
        
        with self._driver.session() as session:
            query = """
            MATCH (n)
            WHERE any(prop in keys(n) WHERE toString(n[prop]) CONTAINS $keyword)
            RETURN n
            LIMIT $limit
            """
            
            result = session.run(query, keyword=keyword, limit=limit)
            return [dict(record["n"]) for record in result]
    
    def _search_nodes_nx(self, keyword: str, limit: int) -> List[Dict]:
        """Search nodes in NetworkX."""
        if self._nx_graph is None:
            return []
        
        results = []
        keyword_lower = keyword.lower()
        
        for node_id, data in self._nx_graph.nodes(data=True):
            for value in data.values():
                if isinstance(value, str) and keyword_lower in value.lower():
                    results.append({"id": node_id, **data})
                    break
            
            if len(results) >= limit:
                break
        
        return results
    
    def detect_cycles(self) -> List[List[str]]:
        """Detect circular patterns (fraud indicator)."""
        if self.use_neo4j:
            return self._detect_cycles_neo4j()
        else:
            return self._detect_cycles_nx()
    
    def _detect_cycles_neo4j(self) -> List[List[str]]:
        """Detect cycles in Neo4j."""
        self._init_neo4j()
        
        with self._driver.session() as session:
            query = """
            MATCH path = (n)-[*2..5]->(n)
            RETURN [node in nodes(path) | node.id] as cycle
            LIMIT 100
            """
            
            result = session.run(query)
            return [record["cycle"] for record in result]
    
    def _detect_cycles_nx(self) -> List[List[str]]:
        """Detect cycles in NetworkX."""
        if self._nx_graph is None:
            return []
        
        try:
            import networkx as nx
            cycles = list(nx.simple_cycles(self._nx_graph))
            return cycles[:100]
        except Exception:
            return []
    
    def export_graph(self, format: str = "graphml", output_path: str = "graph.graphml") -> str:
        """Export graph to file."""
        if not self.use_neo4j and self._nx_graph:
            import networkx as nx
            
            if format == "graphml":
                nx.write_graphml(self._nx_graph, output_path)
            elif format == "gexf":
                nx.write_gexf(self._nx_graph, output_path)
            
            self.logger.info(f"Graph exported to {output_path}")
            return output_path
        
        return ""
    
    def close(self) -> None:
        """Close connections."""
        if self._driver:
            self._driver.close()
            self.logger.info("Neo4j connection closed")
