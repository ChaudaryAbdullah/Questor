
# File: databases/graph_db.py
"""
Graph Database Interface using Neo4j
"""
from neo4j import GraphDatabase as Neo4jGraphDatabase
from typing import List, Dict, Any, Optional
from contextlib import contextmanager

from utils import Config, Logger
from utils.exceptions import GraphDBError


class GraphDatabase:
    """Interface for Neo4j graph database operations"""
    
    def __init__(self):
        self.logger = Logger.get_logger(self.__class__.__name__)
        self.driver = None
        self._connect()
    
    def _connect(self):
        """Establish connection to Neo4j database"""
        try:
            self.logger.info("Connecting to Neo4j database...")
            
            self.driver = Neo4jGraphDatabase.driver(
                Config.NEO4J_URI,
                auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD)
            )
            
            # Test connection
            self.driver.verify_connectivity()
            
            self.logger.info("Successfully connected to Neo4j database")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Neo4j: {str(e)}")
            raise GraphDBError(f"Database connection failed: {str(e)}")
    
    @contextmanager
    def session(self):
        """Context manager for database sessions"""
        session = self.driver.session()
        try:
            yield session
        finally:
            session.close()
    
    def create_indexes(self):
        """Create necessary indexes for performance"""
        indexes = [
            "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX company_name IF NOT EXISTS FOR (c:Company) ON (c.name)",
            "CREATE INDEX person_name IF NOT EXISTS FOR (p:Person) ON (p.name)",
            "CREATE INDEX document_id IF NOT EXISTS FOR (d:Document) ON (d.doc_id)"
        ]
        
        try:
            with self.session() as session:
                for index in indexes:
                    session.run(index)
            self.logger.info("Database indexes created successfully")
        except Exception as e:
            self.logger.error(f"Failed to create indexes: {str(e)}")
            raise GraphDBError(f"Index creation failed: {str(e)}")
    
    def add_entity(
        self,
        entity_type: str,
        name: str,
        properties: Dict[str, Any]
    ) -> bool:
        """
        Add an entity node to the graph
        
        Args:
            entity_type: Type of entity (Company, Person, etc.)
            name: Entity name
            properties: Additional properties
            
        Returns:
            bool: Success status
        """
        query = f"""
        MERGE (e:{entity_type} {{name: $name}})
        SET e += $properties
        RETURN e
        """
        
        try:
            with self.session() as session:
                result = session.run(query, name=name, properties=properties)
                return result.single() is not None
        except Exception as e:
            self.logger.error(f"Failed to add entity: {str(e)}")
            raise GraphDBError(f"Failed to add entity: {str(e)}")
    
    def add_relationship(
        self,
        from_entity: str,
        from_type: str,
        to_entity: str,
        to_type: str,
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Create a relationship between two entities
        
        Args:
            from_entity: Source entity name
            from_type: Source entity type
            to_entity: Target entity name
            to_type: Target entity type
            relationship_type: Type of relationship
            properties: Relationship properties
            
        Returns:
            bool: Success status
        """
        props = properties or {}
        
        query = f"""
        MATCH (from:{from_type} {{name: $from_name}})
        MATCH (to:{to_type} {{name: $to_name}})
        MERGE (from)-[r:{relationship_type}]->(to)
        SET r += $properties
        RETURN r
        """
        
        try:
            with self.session() as session:
                result = session.run(
                    query,
                    from_name=from_entity,
                    to_name=to_entity,
                    properties=props
                )
                return result.single() is not None
        except Exception as e:
            self.logger.error(f"Failed to add relationship: {str(e)}")
            raise GraphDBError(f"Failed to add relationship: {str(e)}")
    
    def query_graph(self, cypher_query: str, parameters: Optional[Dict] = None) -> List[Dict]:
        """
        Execute a custom Cypher query
        
        Args:
            cypher_query: Cypher query string
            parameters: Query parameters
            
        Returns:
            List of result dictionaries
        """
        params = parameters or {}
        
        try:
            with self.session() as session:
                result = session.run(cypher_query, params)
                return [record.data() for record in result]
        except Exception as e:
            self.logger.error(f"Query execution failed: {str(e)}")
            raise GraphDBError(f"Query execution failed: {str(e)}")
    
    def find_hidden_subsidiaries(self, company_name: str) -> List[Dict]:
        """
        Find subsidiaries of a company (as per your fraud detection use case)
        
        Args:
            company_name: Name of the company
            
        Returns:
            List of subsidiary information
        """
        query = """
        MATCH (c:Company {name: $company_name})-[:OWNS]->(s:Subsidiary)
        RETURN s.name as subsidiary_name, s.disclosed as disclosed
        """
        
        return self.query_graph(query, {"company_name": company_name})
    
    def clear_database(self):
        """Clear all nodes and relationships"""
        query = "MATCH (n) DETACH DELETE n"
        
        try:
            with self.session() as session:
                session.run(query)
            self.logger.warning("Graph database has been cleared")
        except Exception as e:
            self.logger.error(f"Failed to clear database: {str(e)}")
            raise GraphDBError(f"Failed to clear database: {str(e)}")
    
    def close(self):
        """Close the database connection"""
        if self.driver:
            self.driver.close()
            self.logger.info("Neo4j connection closed")

