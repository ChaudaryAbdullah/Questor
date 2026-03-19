#!/usr/bin/env python3
"""
Test script to run entity extraction only (no embeddings, no graph, no RAG).
Useful for testing spaCy-only mode without network dependencies.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from src.preprocessing import DocumentPreprocessor
from src.nlp import EntityExtractor
from src.utils.logger import get_logger
from src.utils.config_manager import get_config
import json

load_dotenv()

logger = get_logger(__name__)


def main():
    """Run entity extraction test."""
    
    # Find input file
    input_dir = Path("Input")
    input_files = list(input_dir.glob("*.txt"))
    
    if not input_files:
        logger.error("❌ No .txt files found in Input directory")
        return 1
    
    input_file = input_files[0]
    logger.info(f"📄 Processing file: {input_file}")
    
    # Load config
    config = get_config()
    entity_config = config.entity_extraction_config
    
    logger.info(f"🔧 Entity Extraction Config:")
    logger.info(f"   - use_llm: {entity_config.get('use_llm')}")
    logger.info(f"   - use_spacy_fallback: {entity_config.get('use_spacy_fallback')}")
    
    try:
        # Step 1: Preprocess document
        logger.info("\n⚙️  Step 1: Preprocessing document...")
        preprocessor = DocumentPreprocessor()
        processed_doc = preprocessor.process(str(input_file))
        logger.info(f"   ✓ Created {len(processed_doc.chunks)} chunks from {len(processed_doc.sections)} sections")
        
        # Step 2: Extract entities
        logger.info("\n⚙️  Step 2: Extracting entities...")
        extractor = EntityExtractor(
            use_llm=entity_config.get('use_llm', False),
            use_spacy_fallback=entity_config.get('use_spacy_fallback', True)
        )
        
        # Convert chunks to dict format
        chunks_dict = [chunk.to_dict() for chunk in processed_doc.chunks]
        
        # Extract from first 10 chunks only (for speed)
        test_chunks = chunks_dict[:10]  
        logger.info(f"   Testing with first {len(test_chunks)} chunks...")
        
        result = extractor.extract_from_chunks(test_chunks, input_file.stem)
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ ENTITY EXTRACTION COMPLETED!")
        logger.info("=" * 60)
        logger.info(f"📊 Total entities: {len(result.entities)}")
        logger.info(f"🔗 Total relationships: {len(result.relationships)}")
        logger.info(f"⏱️  Extraction time: {result.extraction_time:.2f}s")
        logger.info(f"🛠️  Method used: {result.method_used}")
        logger.info("=" * 60)
        
        # Show entity breakdown
        entity_types = {}
        for entity in result.entities:
            entity_types[entity.entity_type] = entity_types.get(entity.entity_type, 0) + 1
        
        logger.info("\n📊 Entities by type:")
        for etype, count in sorted(entity_types.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"   {etype}: {count}")
        
        # Save results
        output_dir = Path("data/test_results")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{input_file.stem}_entities.json"
        with open(output_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        logger.info(f"\n✅ Results saved to: {output_file}")
        
        # Show sample entities
        logger.info("\n📋 Sample entities (first 10):")
        for i, entity in enumerate(result.entities[:10], 1):
            logger.info(f"   {i}. [{entity.entity_type}] {entity.text} (confidence: {entity.confidence:.2f})")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
