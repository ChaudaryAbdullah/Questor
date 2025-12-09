
# File: main.py
"""
Main entry point for the fraud detection system
"""
import argparse
from pathlib import Path

from pipelines.unstructured_pipeline import UnstructuredPipeline
from utils import Config, Logger


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Financial Fraud Detection - Unstructured Data Pipeline'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of documents to process (default: all)'
    )
    parser.add_argument(
        '--skip-embeddings',
        action='store_true',
        help='Skip embedding generation'
    )
    parser.add_argument(
        '--skip-graph',
        action='store_true',
        help='Skip knowledge graph construction'
    )
    parser.add_argument(
        '--reset',
        action='store_true',
        help='Reset pipeline (delete all data)'
    )
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show pipeline status'
    )
    parser.add_argument(
        '--query',
        type=str,
        help='Query the vector database'
    )
    
    args = parser.parse_args()
    
    # Create necessary directories
    Config.create_directories()
    
    # Initialize logger
    logger = Logger.get_logger('Main')
    
    # Initialize pipeline
    pipeline = UnstructuredPipeline()
    
    try:
        if args.reset:
            logger.warning("Resetting pipeline...")
            pipeline.reset_pipeline(confirm=True)
            return
        
        if args.status:
            status = pipeline.get_pipeline_status()
            logger.info("Pipeline Status:")
            for key, value in status.items():
                logger.info(f"  {key}: {value}")
            return
        
        if args.query:
            logger.info(f"Querying vector database: {args.query}")
            results = pipeline.query_vector_db(args.query, n_results=5)
            logger.info(f"Found {len(results['ids'][0])} results")
            for i, (doc_id, distance) in enumerate(zip(results['ids'][0], results['distances'][0])):
                logger.info(f"  {i+1}. {doc_id} (distance: {distance:.4f})")
            return
        
        # Run the main pipeline
        logger.info("Starting pipeline execution...")
        stats = pipeline.run(
            limit=args.limit,
            skip_embeddings=args.skip_embeddings,
            skip_graph=args.skip_graph
        )
        
        if stats['success']:
            logger.info("\n" + "=" * 80)
            logger.info("FINAL STATISTICS")
            logger.info("=" * 80)
            logger.info(f"Documents processed: {stats['documents_processed']}")
            logger.info(f"Chunks created: {stats['chunks_created']}")
            logger.info(f"Elapsed time: {stats['elapsed_time']:.2f} seconds")
            
            if 'embedding_stats' in stats and stats['embedding_stats'].get('status') != 'skipped':
                logger.info(f"Embeddings generated: {stats['embedding_stats']['chunks_embedded']}")
            
            if 'graph_stats' in stats and stats['graph_stats'].get('status') != 'skipped':
                logger.info(f"Graph entities: {stats['graph_stats']['entities_added']}")
                logger.info(f"Graph relationships: {stats['graph_stats']['relationships_added']}")
        else:
            logger.error(f"Pipeline failed: {stats.get('error', 'Unknown error')}")
    
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()