# File: main_optimized.py
"""
Memory-optimized entry point for the fraud detection system
"""
import argparse
from pathlib import Path
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Direct imports to avoid __init__.py issues
from pipelines.unstructured_pipeline import UnstructuredPipelineOptimized

# Check if psutil is available (for memory monitoring)
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not installed. Memory monitoring disabled.")
    print("Install with: pip install psutil")

# Import utils
try:
    from utils import Config, Logger
except ImportError:
    from utils.config import Config
    from utils.logger import Logger


def get_memory_info():
    """Get current memory usage information"""
    if not PSUTIL_AVAILABLE:
        return {'rss_mb': 0, 'vms_mb': 0}
    
    try:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return {
            'rss_mb': mem_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': mem_info.vms / 1024 / 1024,  # Virtual Memory Size
        }
    except Exception:
        return {'rss_mb': 0, 'vms_mb': 0}


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Financial Fraud Detection - Memory-Optimized Unstructured Data Pipeline'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of documents to process (default: all)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
        help='Number of documents to process per batch (default: 10)'
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
    parser.add_argument(
        '--memory-monitor',
        action='store_true',
        help='Enable memory monitoring'
    )
    parser.add_argument(
        '--export-output',
        type=str,
        default=None,
        help='Export formatted output for multiagent system (provide batch name)'
    )
    parser.add_argument(
        '--risk-summary',
        action='store_true',
        help='Show risk score summary after processing'
    )
    parser.add_argument(
        '--disable-risk-scoring',
        action='store_true',
        help='Disable risk scoring (faster processing)'
    )
    parser.add_argument(
        '--input-file',
        type=str,
        default=None,
        help='Process a specific file (e.g., "full-submission.txt") instead of all files'
    )
    
    args = parser.parse_args()
    
    # Create necessary directories
    Config.create_directories()
    
    # Initialize logger
    logger = Logger.get_logger('MainOptimized')
    
    # Log initial memory
    if args.memory_monitor:
        mem_info = get_memory_info()
        logger.info(f"Initial memory - RSS: {mem_info['rss_mb']:.2f} MB, VMS: {mem_info['vms_mb']:.2f} MB")
    
    # Initialize pipeline with specified batch size and risk scoring
    enable_risk_scoring = not args.disable_risk_scoring
    pipeline = UnstructuredPipelineOptimized(
        batch_size=args.batch_size,
        enable_risk_scoring=enable_risk_scoring,
        enable_output_formatting=enable_risk_scoring  # Enable output formatting if risk scoring is enabled
    )
    
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
        
        logger.info("Starting optimized pipeline execution...")
        logger.info(f"Configuration:")
        logger.info(f"  - Batch size: {args.batch_size}")
        logger.info(f"  - Document limit: {args.limit or 'All'}")
        logger.info(f"  - Input file: {args.input_file or 'All matching files'}")
        logger.info(f"  - Skip embeddings: {args.skip_embeddings}")
        logger.info(f"  - Skip graph: {args.skip_graph}")
        logger.info(f"  - Risk scoring: {'Enabled' if enable_risk_scoring else 'Disabled'}")
        
        stats = pipeline.run(
            limit=args.limit,
            skip_embeddings=args.skip_embeddings,
            skip_graph=args.skip_graph,
            process_batch_size=args.batch_size,
            input_file=args.input_file
        )
        
        if stats['success']:
            logger.info("\n" + "=" * 80)
            logger.info("FINAL STATISTICS")
            logger.info("=" * 80)
            logger.info(f"Documents processed: {stats['documents_processed']}")
            logger.info(f"Chunks created: {stats['chunks_created']}")
            logger.info(f"Entities extracted: {stats.get('entities_extracted', 0)}")
            logger.info(f"Graph nodes added: {stats.get('graph_nodes_added', 0)}")
            logger.info(f"Graph relationships added: {stats.get('graph_relationships_added', 0)}")
            logger.info(f"Elapsed time: {stats['elapsed_time']:.2f} seconds")
            
            if args.memory_monitor:
                mem_info = get_memory_info()
                logger.info(f"Final memory - RSS: {mem_info['rss_mb']:.2f} MB, VMS: {mem_info['vms_mb']:.2f} MB")
            
            # Show risk summary if requested
            if args.risk_summary and enable_risk_scoring:
                logger.info("\n" + "=" * 80)
                logger.info("RISK SCORE SUMMARY")
                logger.info("=" * 80)
                risk_summary = pipeline.get_risk_summary()
                logger.info(f"Total documents: {risk_summary.get('total_documents', 0)}")
                logger.info(f"Average risk score: {risk_summary.get('average_risk_score', 0):.2f}")
                logger.info(f"Max risk score: {risk_summary.get('max_risk_score', 0):.2f}")
                logger.info(f"Min risk score: {risk_summary.get('min_risk_score', 0):.2f}")
                logger.info("\nRisk Level Distribution:")
                for level, count in risk_summary.get('risk_level_distribution', {}).items():
                    logger.info(f"  {level}: {count}")
                logger.info(f"\nHigh-risk documents: {risk_summary.get('high_risk_count', 0)}")
                if risk_summary.get('high_risk_documents'):
                    logger.info("\nTop 5 High-Risk Documents:")
                    for i, doc in enumerate(risk_summary['high_risk_documents'][:5], 1):
                        logger.info(f"  {i}. {doc['doc_id']}: {doc['risk_score']:.2f} ({doc['risk_level']})")
            
            # Export formatted output if requested
            if args.export_output and enable_risk_scoring:
                logger.info("\n" + "=" * 80)
                logger.info("EXPORTING FORMATTED OUTPUT")
                logger.info("=" * 80)
                output_path = pipeline.export_formatted_outputs(batch_name=args.export_output)
                if output_path:
                    logger.info(f"Output exported successfully to: {output_path}")
                else:
                    logger.warning("Failed to export output")
        else:
            logger.error(f"Pipeline failed: {stats.get('error', 'Unknown error')}")
    
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
    finally:
        pipeline.close()
        
        if args.memory_monitor:
            mem_info = get_memory_info()
            logger.info(f"Cleanup memory - RSS: {mem_info['rss_mb']:.2f} MB, VMS: {mem_info['vms_mb']:.2f} MB")


if __name__ == "__main__":
    main()