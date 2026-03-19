#!/usr/bin/env python3
"""
Unstructured fraud-detection pipeline runner.

Standalone usage:
    python run_pipeline.py              (reads from Main_Immplementation/Input/)

Programmatic usage (called by run_unified.py):
    from run_pipeline import run_pipeline
    result = run_pipeline("/path/to/file.txt")
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Resolve paths relative to THIS file so the script works regardless of cwd
# ---------------------------------------------------------------------------
PIPELINE_DIR = Path(__file__).parent.resolve()

# Load .env from the pipeline directory
load_dotenv(PIPELINE_DIR / ".env")

# Add src to path
sys.path.insert(0, str(PIPELINE_DIR))

from src.utils.pipeline_orchestrator import PipelineOrchestrator
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
# Input: shared Main_Immplementation/Input/
MAIN_INPUT_DIR = PIPELINE_DIR.parent / "Input"

# Own output: data/reports/ (set by PipelineOrchestrator internally)
PIPELINE_REPORTS_DIR = PIPELINE_DIR / "data" / "reports"


def _build_risk_assessment_from_result(result) -> dict:
    """
    Build a risk_assessment dict from a PipelineResult object.

    Strategy:
    - If there are explicit fraud_findings, score them by severity.
    - Otherwise, score from RAG analysis using the risk levels and
      fraud_indicators found in each RAG response.
    - Returns a dict compatible with ScoreCombiner._extract_risk_assessment().
    """
    # ---- Severity-weighted fraud findings ----
    fraud_score = 0.0
    severity_weights = {"CRITICAL": 30, "HIGH": 20, "MEDIUM": 10, "LOW": 5}

    if result.fraud_findings_count > 0:
        for ff in result.fraud_findings:
            sev = ff.get("severity", "MEDIUM")
            fraud_score += severity_weights.get(sev, 10)
        fraud_score = min(100.0, fraud_score)

    # ---- RAG-based scoring (when no explicit fraud findings) ----
    rag_indicators_total = 0
    rag_risk_levels = []
    rag_summary = []

    rag_level_weights = {"HIGH": 20, "MEDIUM": 10, "LOW": 5}

    for rag in result.rag_analysis:
        meta = rag.get("metadata", {})
        rl = meta.get("risk_level", "MEDIUM")
        rag_risk_levels.append(rl)
        indicators = rag.get("fraud_indicators", [])
        rag_indicators_total += len(indicators)

        # Build a summary entry
        if indicators:
            query = rag.get("query", "")
            rag_summary.append(f"[{rl}] {query}: {len(indicators)} indicators")

        # Add score from RAG only if fraud_findings_count is 0
        if result.fraud_findings_count == 0:
            fraud_score += rag_level_weights.get(rl, 10) * len(indicators) * 0.5

    if result.fraud_findings_count == 0 and rag_indicators_total > 0:
        fraud_score = min(100.0, fraud_score)

    # ---- Categorise ----
    risk_level = (
        "CRITICAL" if fraud_score >= 80
        else "HIGH"     if fraud_score >= 60
        else "MEDIUM"   if fraud_score >= 40
        else "LOW"      if fraud_score >= 20
        else "MINIMAL"
    )

    # ---- Risk factors ----
    risk_factors = []
    # From fraud findings
    for ff in result.fraud_findings[:3]:
        risk_factors.append(
            f"[{ff.get('severity','?')}] {ff.get('pattern_name','Unknown')}"
        )
    # From RAG summaries
    risk_factors.extend(rag_summary[:5])

    confidence = 0.8 if (result.fraud_findings_count > 0 or rag_indicators_total > 0) else 0.5

    return {
        "overall_risk_score": round(fraud_score, 2),
        "risk_level": risk_level,
        "component_scores": {
            "fraud_findings_count": result.fraud_findings_count,
            "entities_extracted": result.entities_extracted,
            "rag_queries_run": len(result.rag_analysis),
            "rag_indicators_total": rag_indicators_total,
        },
        "risk_factors": risk_factors,
        "confidence": confidence,
    }


def run_pipeline(input_file_path: str) -> dict:
    """
    Process a single .txt document through the unstructured fraud-detection
    pipeline and return structured results.

    This function is intended to be called from run_unified.py (step 3).

    Args:
        input_file_path: Absolute (or relative) path to the .txt input file.

    Returns:
        dict with keys:
            success              (bool)
            document_id          (str)
            company_name         (str | None)
            fiscal_year          (str | None)
            entities_extracted   (int)
            relationships_extracted (int)
            chunks_created       (int)
            fraud_findings_count (int)
            fraud_findings       (list[dict])
            rag_analysis         (list[dict])   – full RAG responses
            risk_assessment      (dict)          – for ScoreCombiner
            output_file          (str)           – path in data/reports/
            error_message        (str | None)
    """
    if not os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY") == "your_groq_api_key_here":
        logger.error("❌ GROQ_API_KEY not set in .env file!")
        return {"success": False, "error_message": "GROQ_API_KEY not configured"}

    input_file = Path(input_file_path).resolve()
    if not input_file.exists():
        return {"success": False, "error_message": f"Input file not found: {input_file}"}

    try:
        logger.info("🚀 Initializing unstructured pipeline...")
        orchestrator = PipelineOrchestrator()

        logger.info(f"⚙️  Processing document: {input_file}")
        result = orchestrator.process_document(str(input_file))

        # ------------------------------------------------------------------
        # The PipelineOrchestrator already saves the full report to data/reports/
        # as <document_id>_results.json — find it.
        # ------------------------------------------------------------------
        doc_id = result.document_id or input_file.stem
        report_path = PIPELINE_REPORTS_DIR / f"{doc_id}_results.json"

        if not report_path.exists():
            # Fallback: find the most recently written json in data/reports/
            candidates = sorted(PIPELINE_REPORTS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            report_path = candidates[0] if candidates else None

        logger.info(f"📁 Unstructured report: {report_path}")

        # ------------------------------------------------------------------
        # Build risk_assessment from the rich result data
        # ------------------------------------------------------------------
        risk_assessment = _build_risk_assessment_from_result(result)

        orchestrator.cleanup()

        return {
            "success": True,           # result succeeded if we got data, even partial
            "document_id": doc_id,
            "company_name": result.company_name,
            "fiscal_year": result.fiscal_year,
            "chunks_created": result.chunks_created,
            "entities_extracted": result.entities_extracted,
            "relationships_extracted": result.relationships_extracted,
            "fraud_findings_count": result.fraud_findings_count,
            "fraud_findings": result.fraud_findings,
            "rag_analysis": result.rag_analysis,
            "rag_queries_count": len(result.rag_analysis),
            "risk_assessment": risk_assessment,
            "output_file": str(report_path) if report_path else None,
            "timing": {
                "preprocessing_time": result.preprocessing_time,
                "entity_extraction_time": result.entity_extraction_time,
                "embedding_time": result.embedding_time,
                "graph_construction_time": result.graph_construction_time,
                "fraud_analysis_time": result.fraud_analysis_time,
                "total_time": result.total_time,
            },
            "error_message": result.error_message,
        }

    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}", exc_info=True)
        return {"success": False, "error_message": str(e)}


def main():
    """Standalone entry point – discovers the .txt file in Main_Immplementation/Input/."""

    if not os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY") == "your_groq_api_key_here":
        logger.error("❌ GROQ_API_KEY not set in .env file!")
        logger.error("Please edit .env and add your Groq API key")
        return 1

    # Prefer Main_Immplementation/Input/; fall back to own Input/
    input_dirs_to_try = [MAIN_INPUT_DIR, PIPELINE_DIR / "Input"]
    input_file = None

    for candidate_dir in input_dirs_to_try:
        if candidate_dir.exists():
            txt_files = list(candidate_dir.glob("*.txt"))
            if txt_files:
                input_file = txt_files[0]
                logger.info(f"📂 Using input directory: {candidate_dir}")
                break

    if input_file is None:
        logger.error(
            "❌ No .txt file found in any of: "
            + ", ".join(str(d) for d in input_dirs_to_try)
        )
        return 1

    logger.info(f"📄 Processing file: {input_file}")
    result = run_pipeline(str(input_file))

    logger.info("=" * 60)
    if result.get("success"):
        logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    else:
        logger.info(f"❌ PIPELINE FAILED: {result.get('error_message')}")
    logger.info("=" * 60)
    logger.info(f"📊 Entities extracted:  {result.get('entities_extracted', 0)}")
    logger.info(f"📄 Chunks created:      {result.get('chunks_created', 0)}")
    logger.info(f"🔍 RAG queries run:     {result.get('rag_queries_count', 0)}")
    logger.info(f"🚨 Fraud findings:      {result.get('fraud_findings_count', 0)}")
    ra = result.get("risk_assessment", {})
    logger.info(
        f"⚠️  Risk score:          {ra.get('overall_risk_score', 0):.1f} "
        f"({ra.get('risk_level', 'N/A')})"
    )
    logger.info(f"📁 Report saved to:    {result.get('output_file', 'N/A')}")
    logger.info("=" * 60)

    if result.get("rag_analysis"):
        logger.info("\n📋 RAG ANALYSIS SUMMARY:")
        for i, rag in enumerate(result["rag_analysis"][:3], 1):
            q = rag.get("query", "?")
            meta = rag.get("metadata", {})
            rl = meta.get("risk_level", "?")
            n_ind = len(rag.get("fraud_indicators", []))
            logger.info(f"  {i}. [{rl}] {q} → {n_ind} indicators")

    return 0 if result.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
