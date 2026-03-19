#!/usr/bin/env python3
"""
Unified Pipeline Runner
Orchestrates structured pipeline, unstructured pipeline, and agents
"""
import sys
import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Add paths
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

# Add the reworked Unstructured Pipeline directory to path
UNSTRUCTURED_PIPELINE_DIR = BASE_DIR / 'Unstructured Pipeline'
sys.path.insert(0, str(UNSTRUCTURED_PIPELINE_DIR))

# Import components
from agents.orchestrator import AgentOrchestrator, load_agent_config
from score_combiner import ScoreCombiner
from shared.utils import setup_logging, ensure_directory

# Import the reworked unstructured pipeline runner
try:
    from run_pipeline import run_pipeline as run_unstructured_pipeline_fn
except ImportError as _e:
    run_unstructured_pipeline_fn = None
    logging.getLogger(__name__).warning(
        f"Could not import run_pipeline from Unstructured Pipeline: {_e}"
    )

# Import structured pipeline - add to path first
sys.path.insert(0, str(BASE_DIR / 'stuctured_pipeline'))
from inference_pipeline import main as run_structured_inference


class UnifiedPipelineRunner:
    """
    Unified runner that orchestrates all pipelines and agents
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize unified runner"""
        self.config = config or {}
        self.logger = setup_logging(level=self.config.get('logging', {}).get('level', 'INFO'))
        
        # Initialize components
        self.agent_orchestrator = None
        self.score_combiner = ScoreCombiner(config)
        
        # Paths
        self.input_dir = BASE_DIR / 'Input'
        self.output_dir = ensure_directory(BASE_DIR / 'Output')
        
        self.logger.info("Unified Pipeline Runner initialized")
    
    def run(
        self,
        input_directory: Optional[Path] = None,
        enable_agents: bool = True,
        save_output: bool = True
    ) -> Dict[str, Any]:
        """
        Run the complete unified pipeline
        
        Args:
            input_directory: Directory containing input files (uses default if None)
            enable_agents: Whether to run agent orchestrator
            save_output: Whether to save combined output
            
        Returns:
            Dictionary with results and output path
        """
        self.logger.info("=" * 80)
        self.logger.info("STARTING UNIFIED PIPELINE")
        self.logger.info("=" * 80)
        
        input_dir = input_directory or self.input_dir
        
        try:
            # Step 1: Find JSON files (structured) and txt file (unstructured)
            self.logger.info("\n[1/4] Scanning input directory...")
            json_files = list(input_dir.glob("*.json"))
            txt_files  = list(input_dir.glob("*.txt"))

            if not json_files:
                self.logger.error("No .json files found in input directory")
                return {'success': False, 'error': 'No JSON input files found'}

            self.logger.info(f"✓ Found {len(json_files)} JSON file(s) and {len(txt_files)} TXT file(s)")

            # The single txt file used by the unstructured pipeline (if any)
            txt_file = txt_files[0] if txt_files else None
            if txt_file:
                self.logger.info(f"  TXT for unstructured pipeline: {txt_file.name}")
            else:
                self.logger.warning("  No .txt file found – unstructured pipeline will be skipped")

            # Process each JSON file (structured + agents + score combination)
            all_results = []

            for file_path in json_files:
                filename = file_path.name
                self.logger.info(f"\n{'=' * 80}")
                self.logger.info(f"Processing: {filename}")
                self.logger.info(f"{'=' * 80}")

                result = self._process_single_file(file_path, txt_file, enable_agents)

                if result.get('success'):
                    all_results.append(result)
                else:
                    self.logger.error(f"Failed to process {filename}: {result.get('error')}")
            
            # Save combined results
            if save_output and all_results:
                output_path = self._save_results(all_results)
                self.logger.info(f"\n✓ Saved combined results to: {output_path}")
            else:
                output_path = None
            
            self.logger.info("\n" + "=" * 80)
            self.logger.info("UNIFIED PIPELINE COMPLETED")
            self.logger.info(f"Processed  {len(all_results)}/{len(json_files)} files successfully")
            self.logger.info("=" * 80)
            
            return {
                'success': True,
                'files_processed': len(all_results),
                'total_files': len(json_files),
                'output_path': str(output_path) if output_path else None,
                'results': all_results
            }
            
        except Exception as e:
            self.logger.error(f"Unified pipeline failed: {str(e)}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }
    
    def _process_single_file(
        self,
        file_path: Path,
        txt_file: Optional[Path],
        enable_agents: bool
    ) -> Dict[str, Any]:
        """
        Process a single JSON input file through all pipelines.

        Args:
            file_path:   Path to input JSON file (structured pipeline)
            txt_file:    Path to the .txt file for the unstructured pipeline (or None)
            enable_agents: Whether to run agent orchestrator

        Returns:
            Combined results for this file
        """
        cik = file_path.stem  # Use filename stem as identifier
        try:
            # Step 2: Run structured pipeline
            self.logger.info("\n[2/4] Running structured pipeline...")
            structured_result = self._run_structured_pipeline(file_path)

            if not structured_result.get('success'):
                self.logger.warning(f"Structured pipeline failed: {structured_result.get('error')}")
                structured_result = None
            else:
                self.logger.info(f"✓ Structured risk score: {structured_result.get('risk_score', 0):.4f}")

            # Step 3: Run reworked unstructured pipeline on the .txt file
            self.logger.info("\n[3/4] Running unstructured pipeline...")
            unstructured_result = self._run_unstructured_pipeline(txt_file)

            if not unstructured_result or not unstructured_result.get('success'):
                err = unstructured_result.get('error_message') if unstructured_result else 'No .txt file'
                self.logger.warning(f"Unstructured pipeline failed/skipped: {err}")
                unstructured_result = None
            else:
                ra = unstructured_result.get('risk_assessment', {})
                self.logger.info(
                    f"✓ Unstructured risk score: {ra.get('overall_risk_score', 0):.1f} "
                    f"({ra.get('risk_level', 'N/A')})  "
                    f"| Findings: {unstructured_result.get('fraud_findings_count', 0)}"
                )

            # Step 4: Run agents (if enabled)
            agent_results = None
            if enable_agents and structured_result:
                self.logger.info("\n[4/4] Running agent orchestrator...")
                agent_results = self._run_agents(structured_result)

                if agent_results and agent_results.get('agents_succeeded', 0) > 0:
                    self.logger.info(f"✓ Agents succeeded: {agent_results.get('agents_succeeded', 0)}")
                    self.logger.info(f"  Combined agent score: {agent_results.get('combined_score', 0):.2f}")
                else:
                    self.logger.warning("No agents succeeded")
                    agent_results = None

            # Combine scores
            self.logger.info("\nCombining scores...")
            combined_result = self._combine_scores(
                structured_result,
                unstructured_result,
                agent_results,
                cik,
                file_path.name
            )

            self.logger.info(
                f"✓ Final combined risk score: "
                f"{combined_result.get('combined_risk', {}).get('overall_risk_score', 0):.2f}"
            )

            return {
                'success': True,
                'cik': cik,
                'filename': file_path.name,
                'structured': structured_result,
                'unstructured': unstructured_result,
                'agents': agent_results,
                'combined': combined_result
            }

        except Exception as e:
            self.logger.error(f"Failed to process file: {str(e)}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'cik': cik,
                'filename': file_path.name
            }
    
    def _run_structured_pipeline(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Run structured pipeline on JSON file"""
        try:
            # Save current directory
            current_dir = os.getcwd()
            
            try:
                # Change to structured pipeline directory
                structured_dir = BASE_DIR / 'stuctured_pipeline'
                os.chdir(structured_dir)
                
                # Convert file path to relative from structured_pipeline directory
                relative_path = os.path.relpath(file_path, structured_dir)
                
                # Run inference
                result = run_structured_inference(relative_path)
                return result
                
            finally:
                # Restore directory
                os.chdir(current_dir)
                
        except Exception as e:
            self.logger.error(f"Structured pipeline error: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _run_unstructured_pipeline(self, txt_file: Optional[Path]) -> Optional[Dict[str, Any]]:
        """
        Run the reworked unstructured pipeline on a .txt document.

        Args:
            txt_file: Path to the .txt file in Main_Immplementation/Input/, or None.

        Returns:
            Result dict from run_pipeline(), or None if unavailable.
        """
        if txt_file is None:
            self.logger.warning("No .txt file provided – skipping unstructured pipeline")
            return None

        if run_unstructured_pipeline_fn is None:
            self.logger.error("run_pipeline could not be imported from Unstructured Pipeline")
            return {'success': False, 'error_message': 'run_pipeline not importable'}

        try:
            self.logger.info(f"  Processing: {txt_file.name}")
            result = run_unstructured_pipeline_fn(str(txt_file))
            return result
        except Exception as e:
            self.logger.error(f"Unstructured pipeline error: {str(e)}")
            return {'success': False, 'error_message': str(e)}
    
    def _run_agents(self, structured_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Run agent orchestrator.

        Agents need access to the raw financial statements (balance_sheet,
        income_statement, cash_flow) — NOT just the model prediction results.
        We load the original input JSON using the 'input_file' field that the
        structured pipeline embeds in its output, and merge it before passing
        data to the agents.
        """
        try:
            if not self.agent_orchestrator:
                agent_config = load_agent_config()
                self.agent_orchestrator = AgentOrchestrator(agent_config)

            # ------------------------------------------------------------------
            # KEY FIX: enrich agent_data with the raw input JSON so that
            # financial agents (Altman Z-Score, Cash Flow/Earnings, Debt Anomaly,
            # Related Party, Expense Padding) have the balance_sheet /
            # income_statement / cash_flow fields they need.
            # Without this, agents only see model prediction fields (risk_score,
            # individual_model_results, etc.) and return "Data not applicable".
            # ------------------------------------------------------------------
            agent_data = dict(structured_data)  # start with a copy

            input_filename = structured_data.get('input_file', '')
            if input_filename:
                input_candidates = [
                    self.input_dir / input_filename,
                    BASE_DIR / 'stuctured_pipeline' / 'Input' / input_filename,
                    Path(input_filename),
                ]
                for candidate in input_candidates:
                    if candidate.exists():
                        try:
                            with open(candidate, 'r') as f:
                                raw_financial = json.load(f)
                            agent_data.update(raw_financial)
                            self.logger.info(
                                f"  ℹ Agents enriched with raw financial data from: {candidate.name}"
                            )
                        except Exception as load_err:
                            self.logger.warning(
                                f"  ⚠ Could not load raw input JSON '{candidate}': {load_err}"
                            )
                        break
                else:
                    self.logger.warning(
                        f"  ⚠ Raw input JSON '{input_filename}' not found in known locations — "
                        f"financial agents may report 'Data not applicable'"
                    )
            else:
                self.logger.warning(
                    "  ⚠ Structured result has no 'input_file' field — "
                    "financial agents may report 'Data not applicable'"
                )

            # Run agents on the enriched data dict
            agent_results = self.agent_orchestrator.run_agents(agent_data)

            # Calculate combined score
            combined = self.agent_orchestrator.calculate_combined_score(agent_results)

            # Add individual results
            combined['individual_results'] = {
                name: {
                    'success': result.success,
                    'score': result.score,
                    'confidence': result.confidence,
                    'findings': result.findings,
                    'metrics': result.metrics
                }
                for name, result in agent_results.items()
            }

            return combined

        except Exception as e:
            self.logger.error(f"Agent orchestrator error: {str(e)}")
            return None
    
    def _combine_scores(
        self,
        structured_result: Optional[Dict],
        unstructured_result: Optional[Dict],
        agent_results: Optional[Dict],
        cik: str,
        filename: str
    ) -> Dict[str, Any]:
        """Combine all scores using ScoreCombiner"""
        
        # Extract risk assessments
        from shared.output_schema import RiskAssessment
        
        structured_risk = None
        if structured_result:
            structured_risk = self.score_combiner._extract_risk_assessment(
                structured_result, 'structured'
            )
        
        unstructured_risk = None
        if unstructured_result and unstructured_result.get('success'):
            # The new unstructured pipeline returns a risk_assessment dict directly
            if unstructured_result.get('risk_assessment'):
                unstructured_risk = self.score_combiner._extract_risk_assessment(
                    unstructured_result, 'unstructured'
                )
            elif unstructured_result.get('formatted_output'):
                # Legacy formatted_output path (kept for compatibility)
                unstructured_risk = self.score_combiner._extract_risk_assessment(
                    unstructured_result['formatted_output'], 'unstructured'
                )
        
        # Combine with agents
        combined_risk = self.score_combiner.combine_with_agents(
            structured_risk,
            unstructured_risk,
            agent_results
        )
        
        return {
            'cik': cik,
            'filename': filename,
            'structured_risk': structured_risk.to_dict() if structured_risk else None,
            'unstructured_risk': unstructured_risk.to_dict() if unstructured_risk else None,
            'agent_results': agent_results,
            'combined_risk': combined_risk.to_dict()
        }
    
    def _save_results(self, results: List[Dict[str, Any]]) -> Path:
        """Save combined results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"unified_results_{timestamp}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': timestamp,
                'total_files': len(results),
                'results': results
            }, f, indent=2, default=str)
        
        return output_path
    
    def close(self):
        """Close all pipeline connections"""
        self.logger.info("Unified runner closed")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run unified fraud detection pipeline')
    parser.add_argument(
        '--input-dir',
        type=str,
        help='Input directory containing JSON files (default: ./Input)'
    )
    parser.add_argument(
        '--no-agents',
        action='store_true',
        help='Disable agent orchestrator'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save output file'
    )
    
    args = parser.parse_args()
    
    # Create runner
    runner = UnifiedPipelineRunner()
    
    try:
        # Run pipeline
        result = runner.run(
            input_directory=Path(args.input_dir) if args.input_dir else None,
            enable_agents=not args.no_agents,
            save_output=not args.no_save
        )
        
        if result['success']:
            print(f"\n{'=' * 80}")
            print("✅ PIPELINE COMPLETED SUCCESSFULLY")
            print(f"{'=' * 80}")
            print(f"Files processed: {result['files_processed']}/{result['total_files']}")
            if result.get('output_path'):
                print(f"Output saved to: {result['output_path']}")
        else:
            print(f"\n❌ PIPELINE FAILED: {result.get('error')}")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        runner.close()


if __name__ == "__main__":
    main()
