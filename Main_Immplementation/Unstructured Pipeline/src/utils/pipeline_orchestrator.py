"""
Pipeline Orchestrator
Main orchestration module that coordinates all pipeline components.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict

from ..preprocessing import DocumentPreprocessor, ProcessedDocument
from ..nlp import EntityExtractor, ExtractionResult
from ..nlp.embedding_generator import EmbeddingGenerator
from ..retrieval.vector_store_manager import VectorStoreManager
from ..retrieval.rag_engine import RAGEngine
from ..graph.graph_builder import KnowledgeGraphBuilder
from ..validators.fraud_patterns import FraudPatternDetector, FraudFinding
from ..utils.logger import get_logger
from ..utils.config_manager import get_config


@dataclass
class PipelineResult:
    """Complete pipeline execution result."""
    document_id: str
    document_path: str
    company_name: Optional[str]
    fiscal_year: Optional[str]
    
    # Processing results
    preprocessing_time: float
    entity_extraction_time: float
    embedding_time: float
    graph_construction_time: float
    fraud_analysis_time: float
    total_time: float
    
    # Counts
    chunks_created: int
    entities_extracted: int
    relationships_extracted: int
    fraud_findings_count: int
    
    # Fraud findings
    fraud_findings: List[Dict]
    rag_analysis: List[Dict]
    
    # Metadata
    timestamp: str
    success: bool
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def save(self, output_dir: Path) -> None:
        """Save results to JSON."""
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{self.document_id}_results.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)


class PipelineOrchestrator:
    """
    Main pipeline orchestrator.
    Coordinates document processing, entity extraction, graph construction, and fraud detection.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize pipeline orchestrator."""
        self.logger = get_logger("pipeline_orchestrator")
        
        # Load configuration
        self.config = get_config(config_path)
        self.config.ensure_directories()
        
        # Initialize components
        self.logger.info("Initializing pipeline components...")
        
        # Get entity extraction config
        entity_config = self.config.entity_extraction_config
        use_llm = entity_config.get("use_llm", True)
        use_spacy = entity_config.get("use_spacy_fallback", True)
        
        self.preprocessor = DocumentPreprocessor()
        self.entity_extractor = EntityExtractor(
            use_llm=use_llm,
            use_spacy_fallback=use_spacy
        )
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = VectorStoreManager()
        self.graph_builder = KnowledgeGraphBuilder()
        self.fraud_detector = FraudPatternDetector()
        self.rag_engine = RAGEngine(
            vector_store=self.vector_store,
            embedding_generator=self.embedding_generator,
            graph_builder=self.graph_builder
        )
        
        self.logger.info("Pipeline orchestrator initialized successfully")
    
    def process_document(
        self,
        file_path: str,
        document_id: Optional[str] = None
    ) -> PipelineResult:
        """
        Process a single document through the entire pipeline.
        
        Args:
            file_path: Path to the document
            document_id: Optional document identifier
        
        Returns:
            PipelineResult object
        """
        start_time = time.time()
        
        if document_id is None:
            document_id = Path(file_path).stem
        
        self.logger.log_pipeline_start(file_path)
        
        try:
            # Phase 1: Preprocessing
            self.logger.info("Phase 1: Document Preprocessing")
            preproc_start = time.time()
            processed_doc = self.preprocessor.process(file_path)
            preproc_time = time.time() - preproc_start
            
            # Phase 2: Entity Extraction
            self.logger.info("Phase 2: Entity Extraction")
            extraction_start = time.time()
            chunks_dict = [chunk.to_dict() for chunk in processed_doc.chunks]
            extraction_result = self.entity_extractor.extract_from_chunks(
                chunks_dict, document_id
            )
            extraction_time = time.time() - extraction_start
            
            # Phase 3: Generate Embeddings
            self.logger.info("Phase 3: Generating Embeddings")
            embed_start = time.time()
            chunks_with_embeddings = self.embedding_generator.generate_for_chunks(chunks_dict)
            embed_time = time.time() - embed_start
            
            # Phase 4: Store in Vector Database
            self.logger.info("Phase 4: Storing in Vector Database")
            self.vector_store.add_chunks(chunks_with_embeddings)
            
            # Phase 5: Build Knowledge Graph
            self.logger.info("Phase 5: Building Knowledge Graph")
            graph_start = time.time()
            entities_dict = [e.to_dict() for e in extraction_result.entities]
            relationships_dict = [r.to_dict() for r in extraction_result.relationships]
            
            self.graph_builder.create_nodes(entities_dict)
            self.graph_builder.create_relationships(relationships_dict)
            graph_time = time.time() - graph_start
            
            # Phase 6: Fraud Detection
            self.logger.info("Phase 6: Fraud Pattern Detection")
            fraud_start = time.time()
            
            sections = [s.section_type for s in processed_doc.sections]
            fraud_findings = self.fraud_detector.detect_all_patterns(
                entities_dict,
                relationships_dict,
                sections,
                self.graph_builder
            )
            
            # Run RAG analysis
            rag_responses = self.rag_engine.run_fraud_analysis()
            
            fraud_time = time.time() - fraud_start
            
            # Calculate total time
            total_time = time.time() - start_time
            
            # Create result
            result = PipelineResult(
                document_id=document_id,
                document_path=file_path,
                company_name=processed_doc.company_name,
                fiscal_year=processed_doc.fiscal_year,
                preprocessing_time=preproc_time,
                entity_extraction_time=extraction_time,
                embedding_time=embed_time,
                graph_construction_time=graph_time,
                fraud_analysis_time=fraud_time,
                total_time=total_time,
                chunks_created=len(processed_doc.chunks),
                entities_extracted=len(extraction_result.entities),
                relationships_extracted=len(extraction_result.relationships),
                fraud_findings_count=len(fraud_findings),
                fraud_findings=[asdict(f) for f in fraud_findings],
                rag_analysis=[asdict(r) for r in rag_responses],
                timestamp=datetime.now().isoformat(),
                success=True
            )
            
            self.logger.log_pipeline_end(file_path, True, total_time)
            
            # Save results
            output_dir = Path(self.config.get("data.reports_dir", "data/reports"))
            result.save(output_dir)
            
            # Print summary
            self._print_summary(result)
            
            return result
            
        except Exception as e:
            total_time = time.time() - start_time
            self.logger.exception(f"Pipeline failed: {e}")
            self.logger.log_pipeline_end(file_path, False, total_time)
            
            return PipelineResult(
                document_id=document_id,
                document_path=file_path,
                company_name=None,
                fiscal_year=None,
                preprocessing_time=0,
                entity_extraction_time=0,
                embedding_time=0,
                graph_construction_time=0,
                fraud_analysis_time=0,
                total_time=total_time,
                chunks_created=0,
                entities_extracted=0,
                relationships_extracted=0,
                fraud_findings_count=0,
                fraud_findings=[],
                rag_analysis=[],
                timestamp=datetime.now().isoformat(),
                success=False,
                error_message=str(e)
            )
    
    def _print_summary(self, result: PipelineResult) -> None:
        """Print pipeline execution summary."""
        print("\n" + "=" * 80)
        print("FRAUD DETECTION PIPELINE - EXECUTION SUMMARY")
        print("=" * 80)
        print(f"Document: {result.document_path}")
        print(f"Company: {result.company_name or 'Unknown'}")
        print(f"Fiscal Year: {result.fiscal_year or 'Unknown'}")
        print(f"\nProcessing Statistics:")
        print(f"  - Chunks Created: {result.chunks_created}")
        print(f"  - Entities Extracted: {result.entities_extracted}")
        print(f"  - Relationships Found: {result.relationships_extracted}")
        print(f"  - Fraud Findings: {result.fraud_findings_count}")
        print(f"\nTiming:")
        print(f"  - Preprocessing: {result.preprocessing_time:.2f}s")
        print(f"  - Entity Extraction: {result.entity_extraction_time:.2f}s")
        print(f"  - Embeddings: {result.embedding_time:.2f}s")
        print(f"  - Graph Construction: {result.graph_construction_time:.2f}s")
        print(f"  - Fraud Analysis: {result.fraud_analysis_time:.2f}s")
        print(f"  - Total Time: {result.total_time:.2f}s")
        
        if result.fraud_findings_count > 0:
            print(f"\nFRAUD FINDINGS ({result.fraud_findings_count}):")
            for finding in result.fraud_findings[:5]:
                print(f"  [{finding['severity']}] {finding['pattern_name']}")
                print(f"      {finding['description']}")
        
        print("=" * 80 + "\n")
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        if self.graph_builder:
            self.graph_builder.close()
        self.logger.info("Pipeline cleanup completed")


def main():
    """Main entry point for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fraud Detection Pipeline")
    parser.add_argument("--input", required=True, help="Input document path")
    parser.add_argument("--output", default="data/reports", help="Output directory")
    parser.add_argument("--config", help="Configuration file path")
    
    args = parser.parse_args()
    
    # Initialize and run pipeline
    orchestrator = PipelineOrchestrator(config_path=args.config)
    
    try:
        result = orchestrator.process_document(args.input)
        
        if result.success:
            print(f"\nResults saved to: {args.output}")
            exit(0)
        else:
            print(f"\nPipeline failed: {result.error_message}")
            exit(1)
    finally:
        orchestrator.cleanup()


if __name__ == "__main__":
    main()
