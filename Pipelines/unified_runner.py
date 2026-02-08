#!/usr/bin/env python3
"""
Unified Pipeline Runner
Main orchestrator for running structured and unstructured fraud detection pipelines
"""
import argparse
import sys
import os
import subprocess
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import time

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shared.utils import load_config, setup_logging, ensure_directory, format_time
from score_combiner import ScoreCombiner
from agents.orchestrator import AgentOrchestrator, load_agent_config


class UnifiedPipelineRunner:
    """
    Orchestrates execution of structured and unstructured pipelines
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize unified pipeline runner
        
        Args:
            config: Configuration dictionary (loads from config.yaml if None)
        """
        self.config = config or load_config()
        self.logger = setup_logging(
            level=self.config.get('logging', {}).get('level', 'INFO')
        )
        
        self.base_dir = Path(__file__).parent
        self.output_config = self.config.get('output', {})
        
        # Ensure output directories exist
        for dir_key in ['structured_dir', 'unstructured_dir', 'combined_dir', 'multiagent_dir']:
            ensure_directory(self.output_config.get(dir_key, f'output/{dir_key}'))
        
        self.logger.info("Unified Pipeline Runner initialized")
    
    def run(
        self,
        pipeline: str = 'both',
        structured_args: Optional[List[str]] = None,
        unstructured_args: Optional[List[str]] = None,
        combine_scores: bool = True,
        batch_name: Optional[str] = None,
        enable_agents: bool = True,
        enabled_agents: Optional[List[str]] = None,
        disabled_agents: Optional[List[str]] = None,
        agent_config_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run the specified pipeline(s)
        
        Args:
            pipeline: Which pipeline to run ('structured', 'unstructured', or 'both')
            structured_args: Additional arguments for structured pipeline
            unstructured_args: Additional arguments for unstructured pipeline
            combine_scores: Whether to combine scores after running both
            batch_name: Name for the batch (auto-generated if None)
        
        Returns:
            Dictionary with execution results
        """
        start_time = time.time()
        
        if batch_name is None:
            batch_name = f"fraud_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info("=" * 80)
        self.logger.info("UNIFIED FRAUD DETECTION PIPELINE")
        self.logger.info("=" * 80)
        self.logger.info(f"Mode: {pipeline.upper()}")
        self.logger.info(f"Batch: {batch_name}")
        self.logger.info("")
        
        results = {
            'batch_name': batch_name,
            'timestamp': datetime.now().isoformat(),
            'pipeline_mode': pipeline,
            'success': False,
            'structured_output': None,
            'unstructured_output': None,
            'combined_output': None,
            'agent_results': None
        }
        
        try:
            # Run structured pipeline
            if pipeline in ['structured', 'both']:
                self.logger.info("🔷 Running STRUCTURED pipeline...")
                results['structured_output'] = self._run_structured_pipeline(
                    structured_args or [],
                    batch_name
                )
                if results['structured_output']:
                    self.logger.info(f"✓ Structured pipeline completed: {results['structured_output']}")
                else:
                    self.logger.warning("⚠ Structured pipeline produced no output")
            
            # Run unstructured pipeline
            if pipeline in ['unstructured', 'both']:
                self.logger.info("\n🔶 Running UNSTRUCTURED pipeline...")
                results['unstructured_output'] = self._run_unstructured_pipeline(
                    unstructured_args or [],
                    batch_name
                )
                if results['unstructured_output']:
                    self.logger.info(f"✓ Unstructured pipeline completed: {results['unstructured_output']}")
                else:
                    self.logger.warning("⚠ Unstructured pipeline produced no output")
            
            # Run agents if enabled
            if enable_agents and results['structured_output']:
                self.logger.info("\n🤖 Running fraud detection agents...")
                results['agent_results'] = self._run_agents(
                    results['structured_output'],
                    enabled_agents=enabled_agents,
                    disabled_agents=disabled_agents,
                    agent_config_path=agent_config_path
                )
            
            # Combine scores if pipelines ran
            if combine_scores and pipeline == 'both':
                if results['structured_output'] or results['unstructured_output']:
                    self.logger.info("\n🔗 Combining pipeline outputs...")
                    results['combined_output'] = self._combine_outputs(
                        results['structured_output'],
                        results['unstructured_output'],
                        batch_name,
                        agent_results=results.get('agent_results')
                    )
                    if results['combined_output']:
                        self.logger.info(f"✓ Combined output created: {results['combined_output']}")
            
            results['success'] = True
            results['elapsed_time'] = time.time() - start_time
            
            # Display summary
            self._display_summary(results)
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            results['error'] = str(e)
            results['elapsed_time'] = time.time() - start_time
        
        return results
    
    def _run_structured_pipeline(
        self,
        additional_args: List[str],
        batch_name: str
    ) -> Optional[str]:
        """
        Run the structured pipeline
        
        Returns:
            Path to output file if successful, None otherwise
        """
        # Get entry point from config
        entry_point = self.config['pipelines']['structured']['entry_point']
        script_path = self.base_dir / entry_point
        
        if not script_path.exists():
            self.logger.error(f"Structured pipeline script not found: {script_path}")
            return None
        
        # Prepare command
        # Note: run_inference.py processes JSON files from Input directory
        # We'll run it and capture its output
        
        cmd = [sys.executable, str(script_path)] + additional_args
        
        self.logger.debug(f"Executing: {' '.join(cmd)}")
        
        try:
            # Run the pipeline
            result = subprocess.run(
                cmd,
                cwd=script_path.parent,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            # Log output
            if result.stdout:
                for line in result.stdout.split('\n'):
                    if line.strip():
                        self.logger.info(f"  {line}")
            
            if result.stderr and result.returncode != 0:
                for line in result.stderr.split('\n'):
                    if line.strip():
                        self.logger.error(f"  {line}")
            
            if result.returncode != 0:
                self.logger.error(f"Structured pipeline failed with exit code {result.returncode}")
                return None
            
            # Look for output files
            # Note: structured pipeline saves to its own Output directory
            output_dir = self.base_dir / 'stuctured_pipeline' / 'Output'
            output_files = list(output_dir.glob('*_results.json'))
            
            if output_files:
                # Return the most recent file
                latest = max(output_files, key=lambda p: p.stat().st_mtime)
                return str(latest)
            else:
                self.logger.warning("No JSON output found from structured pipeline")
                return None
                
        except subprocess.TimeoutExpired:
            self.logger.error("Structured pipeline execution timed out")
            return None
        except Exception as e:
            self.logger.error(f"Error running structured pipeline: {e}")
            return None
    
    def _run_unstructured_pipeline(
        self,
        additional_args: List[str],
        batch_name: str
    ) -> Optional[str]:
        """
        Run the unstructured pipeline
        
        Returns:
            Path to output file if successful, None otherwise
        """
        # Get entry point from config
        entry_point = self.config['pipelines']['unstructured']['entry_point']
        script_path = self.base_dir / entry_point
        
        if not script_path.exists():
            self.logger.error(f"Unstructured pipeline script not found: {script_path}")
            return None
        
        # Prepare command with export flag
        cmd = [
            sys.executable,
            str(script_path),
            '--export-output', batch_name
        ] + additional_args
        
        self.logger.debug(f"Executing: {' '.join(cmd)}")
        
        try:
            # Run the pipeline
            result = subprocess.run(
                cmd,
                cwd=script_path.parent,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            # Log output
            if result.stdout:
                for line in result.stdout.split('\n'):
                    if line.strip():
                        self.logger.info(f"  {line}")
            
            if result.stderr and result.returncode != 0:
                for line in result.stderr.split('\n'):
                    if line.strip():
                        self.logger.error(f"  {line}")
            
            if result.returncode != 0:
                self.logger.error(f"Unstructured pipeline failed with exit code {result.returncode}")
                return None
            
            # Look for output files
            output_dir = self.base_dir / 'unstructured_pipeline' / 'output'
            output_files = list(output_dir.glob(f'{batch_name}*.json'))
            
            if output_files:
                # Return the most recent file
                latest = max(output_files, key=lambda p: p.stat().st_mtime)
                return str(latest)
            else:
                self.logger.warning("No JSON output found from unstructured pipeline")
                return None
                
        except subprocess.TimeoutExpired:
            self.logger.error("Unstructured pipeline execution timed out")
            return None
        except Exception as e:
            self.logger.error(f"Error running unstructured pipeline: {e}")
            return None
    
    def _combine_outputs(
        self,
        structured_output: Optional[str],
        unstructured_output: Optional[str],
        batch_name: str,
        agent_results: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Combine outputs from both pipelines and agents
        
        Returns:
            Path to combined output file
        """
        try:
            combiner = ScoreCombiner(self.config)
            combined_path = combiner.combine_batch(
                structured_output=structured_output,
                unstructured_output=unstructured_output,
                batch_name=batch_name,
                agent_results=agent_results
            )
            return combined_path
        except Exception as e:
            self.logger.error(f"Failed to combine outputs: {e}")
            return None
    
    def _run_agents(
        self,
        structured_output: str,
        enabled_agents: Optional[List[str]] = None,
        disabled_agents: Optional[List[str]] = None,
        agent_config_path: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Run fraud detection agents on structured data
        
        Returns:
            Dictionary with agent results and combined score
        """
        try:
            # Load agent config
            agent_config = load_agent_config(agent_config_path)
            
            # Initialize orchestrator
            orchestrator = AgentOrchestrator(agent_config)
            
            # Load structured data
            import json
            with open(structured_output, 'r') as f:
                structured_data = json.load(f)
            
            # Run agents
            agent_results = orchestrator.run_agents(
                structured_data,
                enabled_agents=enabled_agents,
                disabled_agents=disabled_agents
            )
            
            # Log agent results
            for agent_name, result in agent_results.items():
                if result.success:
                    self.logger.info(f"  ✓ {agent_name}: score={result.score:.2f}, confidence={result.confidence:.2f}")
                    if result.findings:
                        for finding in result.findings[:2]:  # Show first 2 findings
                            self.logger.info(f"    - {finding}")
                else:
                    self.logger.warning(f"  ✗ {agent_name}: {result.error}")
            
            # Calculate combined agent score
            combined_results = orchestrator.calculate_combined_score(agent_results)
            
            # Add individual results to output
            combined_results['individual_results'] = {
                name: result.to_dict() for name, result in agent_results.items()
            }
            
            return combined_results
            
        except Exception as e:
            self.logger.error(f"Failed to run agents: {e}", exc_info=True)
            return None
    
    def _display_summary(self, results: Dict[str, Any]):
        """Display execution summary"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("EXECUTION SUMMARY")
        self.logger.info("=" * 80)
        self.logger.info(f"Status: {'✅ SUCCESS' if results['success'] else '❌ FAILED'}")
        self.logger.info(f"Elapsed time: {format_time(results['elapsed_time'])}")
        self.logger.info(f"Batch name: {results['batch_name']}")
        
        if results.get('structured_output'):
            self.logger.info(f"\n📊 Structured output: {results['structured_output']}")
        
        if results.get('unstructured_output'):
            self.logger.info(f"📄 Unstructured output: {results['unstructured_output']}")
        
        if results.get('combined_output'):
            self.logger.info(f"🔗 Combined output: {results['combined_output']}")
            self._display_combined_stats(results['combined_output'])
        
        self.logger.info("=" * 80)
    
    def _display_combined_stats(self, combined_path: str):
        """Display statistics from combined output"""
        try:
            with open(combined_path, 'r') as f:
                data = json.load(f)
            
            summary = data.get('summary_statistics', {})
            
            if summary:
                self.logger.info("\n📈 Combined Statistics:")
                self.logger.info(f"  Total records: {summary.get('total_records', 0)}")
                self.logger.info(f"  Fraud predictions: {summary.get('fraud_predictions', 0)} "
                               f"({summary.get('fraud_percentage', 0):.1f}%)")
                self.logger.info(f"  Average risk: {summary.get('average_risk_score', 0):.2f}")
                self.logger.info(f"  Critical cases: {summary.get('critical_count', 0)}")
                self.logger.info(f"  High risk cases: {summary.get('high_risk_count', 0)}")
        except Exception as e:
            self.logger.debug(f"Could not load combined stats: {e}")


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Unified Fraud Detection Pipeline Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run both pipelines and combine results
  python unified_runner.py --pipeline both --limit 10
  
  # Run only structured pipeline
  python unified_runner.py --pipeline structured --input Input/
  
  # Run only unstructured pipeline with limited documents
  python unified_runner.py --pipeline unstructured --limit 5
        """
    )
    
    parser.add_argument(
        '--pipeline',
        type=str,
        choices=['structured', 'unstructured', 'both'],
        default='both',
        help='Which pipeline to run (default: both)'
    )
    
    parser.add_argument(
        '--batch-name',
        type=str,
        help='Name for this batch (auto-generated if not provided)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--no-combine',
        action='store_true',
        help='Skip score combination even when running both pipelines'
    )
    
    # Structured pipeline arguments
    parser.add_argument(
        '--input',
        type=str,
        help='[Structured] Input directory for JSON files'
    )
    
    # Unstructured pipeline arguments
    parser.add_argument(
        '--limit',
        type=int,
        help='[Unstructured] Limit number of documents to process'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        help='[Unstructured] Batch size for processing'
    )
    
    # Agent control arguments
    parser.add_argument(
        '--no-agents',
        action='store_true',
        help='Disable all fraud detection agents'
    )
    
    parser.add_argument(
        '--enable-agents',
        type=str,
        help='Comma-separated list of agents to enable (e.g., benfords_law,beneish_mscore)'
    )
    
    parser.add_argument(
        '--disable-agent',
        type=str,
        action='append',
        dest='disabled_agents',
        help='Disable specific agent (can be used multiple times)'
    )
    
    parser.add_argument(
        '--agent-config',
        type=str,
        help='Path to agent configuration file'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config) if args.config else load_config()
    
    # Prepare pipeline-specific arguments
    structured_args = []
    if args.input:
        structured_args.append(args.input)
    
    unstructured_args = []
    if args.limit:
        unstructured_args.extend(['--limit', str(args.limit)])
    if args.batch_size:
        unstructured_args.extend(['--batch-size', str(args.batch_size)])
    
    # Parse agent arguments
    enable_agents = not args.no_agents
    enabled_agents = None
    if args.enable_agents:
        enabled_agents = [a.strip() for a in args.enable_agents.split(',')]
    disabled_agents = args.disabled_agents or []
    
    # Run pipeline
    runner = UnifiedPipelineRunner(config)
    results = runner.run(
        pipeline=args.pipeline,
        structured_args=structured_args if structured_args else None,
        unstructured_args=unstructured_args if unstructured_args else None,
        combine_scores=not args.no_combine,
        batch_name=args.batch_name,
        enable_agents=enable_agents,
        enabled_agents=enabled_agents,
        disabled_agents=disabled_agents,
        agent_config_path=args.agent_config
    )
    
    # Exit with appropriate code
    sys.exit(0 if results['success'] else 1)


if __name__ == '__main__':
    main()
