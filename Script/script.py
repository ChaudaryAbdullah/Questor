#!/usr/bin/env python3
"""
full_pipeline_llm_optimized.py - BATCH PROCESSING VERSION

Usage:
    python full_pipeline_llm_optimized.py            # auto-process .txt files
    python full_pipeline_llm_optimized.py /path/to/file.txt  # process single file
    python full_pipeline_llm_optimized.py /path/to/dir       # process directory

Key Features:
- BATCH MODE: Processes all tables in ONE LLM call (10-20x faster!)
- Saves to 'output/' folder automatically
- Smart filtering: skips small tables, limits processing
- Dual output: batch JSON + individual table JSONs

Performance Settings:
- BATCH_PROCESS_TABLES = True  (RECOMMENDED for speed)
- MAX_TABLES_FOR_LLM = -1 (process all, or set limit like 20)
- MIN_TABLE_LINES = 3 (skip tiny tables)
"""

import os
import re
import sys
import json
import textwrap
import subprocess
from glob import glob
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import joblib

from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ---------------------------
# Config
# ---------------------------
EMBED_MODEL = "all-MiniLM-L6-v2"
MODEL_FILE = "fraud_filter_model.pkl"
ENCODER_FILE = "label_encoder.pkl"
BATCH_SIZE = 4000       # lines per batch for streaming classification
EMBED_BATCH = 64        # batch size for embedding.encode()

# LLM Settings
USE_LLM = True         # Set to True if Ollama is installed
LLM_MODEL = "phi3"  # Your local Ollama model (mistral, phi3, gemma, etc.)
OUTPUT_DIR = "output"  # All output files go here

# Performance tuning - ADJUST THESE FOR YOUR NEEDS
BATCH_PROCESS_TABLES = True  # ⚡ Process all tables in ONE LLM call (MUCH FASTER!)
MAX_TABLES_FOR_LLM = -1      # How many tables to process (-1 = all, or set limit like 20)
MIN_TABLE_LINES = 3          # Skip tables with fewer lines than this
MAX_TABLE_LINES_FOR_LLM = 30 # Only send first N lines of each table to LLM
UNSTRUCTURED_SAMPLE_SIZE = 300  # Lines to sample from unstructured text
SKIP_ENTITY_EXTRACTION = False  # Set True to skip entity extraction (saves time)

# ---------------------------
# Utilities: cleaning / heuristics
# ---------------------------

def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"[INFO] Created output directory: {OUTPUT_DIR}")

def clean_html_line(line: str) -> str:
    """Remove HTML tags and common noisy entities, collapse whitespace."""
    if "<" in line and ">" in line:
        try:
            text = BeautifulSoup(line, "html.parser").get_text(" ", strip=True)
        except Exception:
            text = re.sub(r"<[^>]+>", " ", line)
    else:
        text = line
    # Remove repeated page markers like <page> or (Page 1)
    text = re.sub(r"\b(page|Page)\b[:\s]*\d+", "", text)
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\xa0", " ").strip()
    return text

def looks_like_table_line(line: str) -> bool:
    """
    Heuristic: table-like lines usually have both letters and digits and often
    multiple numeric tokens (e.g. 'Total assets 1,200,000 1,100,000')
    or separators like tabs, multiple spaces aligning columns.
    """
    if not line:
        return False
    has_digit = bool(re.search(r"\d", line))
    has_alpha = bool(re.search(r"[A-Za-z]", line))
    # columns separated by multiple spaces or tabs
    colsep = bool(re.search(r"\t| {2,}", line))
    multi_numbers = len(re.findall(r"[-+]?\d[\d,\.]*", line)) >= 1
    # treat lines with lists of numbers and some text as table lines
    return has_digit and has_alpha and (colsep or multi_numbers)

def extract_tables_from_lines(lines: List[str]) -> List[List[str]]:
    """
    Group consecutive table-like lines into table blocks and return list of tables.
    """
    tables = []
    current = []
    for line in lines:
        if looks_like_table_line(line):
            current.append(line)
        else:
            if current:
                tables.append(current)
                current = []
    if current:
        tables.append(current)
    return tables

# ---------------------------
# LLM refinement (Ollama with batch support)
# ---------------------------

def check_ollama_available() -> bool:
    """Check if Ollama is installed and available."""
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False

def check_model_available(model: str) -> bool:
    """Check if specific Ollama model is available."""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, timeout=5)
        if result.returncode == 0:
            output = result.stdout.decode("utf-8")
            return model in output
        return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False

def call_local_ollama(prompt: str, model: str = LLM_MODEL, timeout: int = 45) -> str:
    """
    Calls `ollama run <model>` with prompt on stdin and returns stdout.
    If Ollama not available or model not pulled, returns an informative string.
    """
    try:
        proc = subprocess.run(
            ["ollama", "run", model, "--verbose"],
            input=prompt.encode("utf-8"),
            capture_output=True,
            timeout=timeout
        )
        out = proc.stdout.decode("utf-8").strip()
        err = proc.stderr.decode("utf-8").strip()
        
        if proc.returncode != 0:
            print(f"[WARN] Ollama returned error code {proc.returncode}")
            if err:
                print(f"[WARN] Ollama stderr: {err}")
        
        if not out:
            if err:
                out = err
            else:
                out = json.dumps({
                    "error": "Empty response from Ollama",
                    "table_type": "Unknown",
                    "canonical_headers": [],
                    "rows": [],
                    "confidence": 0
                })
        
        return out
    except FileNotFoundError:
        return json.dumps({
            "error": "Ollama not found. Please install Ollama from https://ollama.ai",
            "table_type": "Unknown",
            "canonical_headers": [],
            "rows": [],
            "confidence": 0
        })
    except subprocess.TimeoutExpired:
        print(f"[WARN] Ollama timeout after {timeout}s - skipping this request")
        return json.dumps({
            "error": f"Ollama timeout after {timeout}s",
            "table_type": "Unknown",
            "canonical_headers": [],
            "rows": [],
            "confidence": 0
        })
    except Exception as e:
        return json.dumps({
            "error": f"Ollama error: {str(e)}",
            "table_type": "Unknown",
            "canonical_headers": [],
            "rows": [],
            "confidence": 0
        })

def refine_tables_batch_with_llm(tables_data: List[Tuple[int, List[str]]]) -> str:
    """
    ⚡ BATCH MODE: Process multiple tables in a single LLM call for efficiency.
    tables_data: List of (table_index, table_lines) tuples
    Returns JSON with results for all tables.
    
    This is 10-20x faster than processing tables individually!
    """
    if not USE_LLM or not tables_data:
        return json.dumps({"tables": [], "note": "LLM batch processing disabled or no tables"})
    
    # Build combined prompt with all tables
    tables_text = []
    for idx, lines in tables_data:
        table_sample = lines[:MAX_TABLE_LINES_FOR_LLM]
        tables_text.append(f"--- TABLE {idx} ---\n" + "\n".join(table_sample))
    
    combined = "\n\n".join(tables_text)
    
    prompt = textwrap.dedent(f"""
    You are a financial data assistant. Analyze ALL tables below and return ONLY valid JSON.
    
    Required format: 
    {{
      "tables": [
        {{
          "table_id": 1,
          "table_type": "Balance Sheet/Income Statement/Cash Flow/Other",
          "canonical_headers": ["col1", "col2"],
          "rows": [{{"col1": "val1", "col2": "val2"}}],
          "confidence": 0.8
        }},
        ...
      ]
    }}
    
    Tables to analyze:
    {combined}
    
    JSON:""")
    
    return call_local_ollama(prompt, model=LLM_MODEL, timeout=120)  # Longer timeout for batch


def refine_table_with_llm(table_lines: List[str]) -> str:
    """
    Single-table processing mode (slower but kept for backwards compatibility).
    For best performance, use BATCH_PROCESS_TABLES = True instead.
    """
    if not USE_LLM:
        return json.dumps({
            "note": "LLM refinement disabled",
            "table_type": "Unknown",
            "canonical_headers": [],
            "rows": [],
            "confidence": 0
        })
    
    # Limit table size for LLM processing
    table_sample = table_lines[:MAX_TABLE_LINES_FOR_LLM]
    
    prompt = textwrap.dedent(f"""
    You are a financial data assistant. Analyze this table and return ONLY valid JSON.
    Required format: {{"table_type": "Balance Sheet/Income Statement/Cash Flow/Other", "canonical_headers": ["col1", "col2"], "rows": [{{"col1": "val1", "col2": "val2"}}], "confidence": 0.8}}
    
    Table data:
    {chr(10).join(table_sample)}
    
    JSON:""")
    return call_local_ollama(prompt, model=LLM_MODEL)

def extract_entities_from_unstructured(lines: List[str], sample_size: int = UNSTRUCTURED_SAMPLE_SIZE) -> str:
    """
    Run LLM on a reasonable sample of unstructured text to extract entities.
    Returns LLM output (expected JSON).
    """
    if not USE_LLM or SKIP_ENTITY_EXTRACTION:
        return json.dumps({
            "note": "LLM entity extraction disabled",
            "CompanyNames": [],
            "Dates": [],
            "Amounts": [],
            "Identifiers": [],
            "Addresses": [],
            "OtherNotes": []
        })
    
    sample = "\n".join(lines[:sample_size])
    prompt = textwrap.dedent(f"""
    Extract entities from this financial text. Return ONLY valid JSON.
    Required format: {{"CompanyNames": [], "Dates": [], "Amounts": [], "Identifiers": [], "Addresses": [], "OtherNotes": []}}
    
    Text:
    {sample}
    
    JSON:""")
    return call_local_ollama(prompt, model=LLM_MODEL)

# ---------------------------
# Classifier wrapper
# ---------------------------

class FraudFilter:
    def __init__(self, embed_model=EMBED_MODEL):
        print(f"[INFO] Loading embedding model '{embed_model}' (this may take a few seconds)...")
        self.embedder = SentenceTransformer(embed_model)
        self.classifier = None
        self.encoder = None

    def train(self, texts: List[str], labels: List[str], save_path=MODEL_FILE, encoder_path=ENCODER_FILE):
        """
        Train classifier from scratch. texts: list[str], labels: list[str] (e.g. structured/unstructured/not useful)
        """
        print("[INFO] Generating embeddings for training data...")
        X = self.embedder.encode(texts, batch_size=EMBED_BATCH, convert_to_numpy=True)
        le = LabelEncoder()
        y = le.fit_transform(labels)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        print("[INFO] Classification report on held-out set:")
        print(classification_report(y_test, y_pred, target_names=le.classes_))

        self.classifier = clf
        self.encoder = le
        joblib.dump(clf, save_path)
        joblib.dump(le, encoder_path)
        print(f"[INFO] Saved classifier -> {save_path} and encoder -> {encoder_path}")

    def load(self, model_path=MODEL_FILE, encoder_path=ENCODER_FILE):
        if not os.path.exists(model_path) or not os.path.exists(encoder_path):
            raise FileNotFoundError("Model or encoder file not found. Train first or place them in working dir.")
        self.classifier = joblib.load(model_path)
        self.encoder = joblib.load(encoder_path)
        print(f"[INFO] Loaded classifier from {model_path} and encoder from {encoder_path}")

    def predict_batch(self, texts: List[str]) -> Tuple[List[str], List[float]]:
        """
        Predict labels for a list of texts. Returns (labels, confidences)
        """
        if self.classifier is None or self.encoder is None:
            raise RuntimeError("Classifier not loaded.")
        X = self.embedder.encode(texts, batch_size=EMBED_BATCH, convert_to_numpy=True)
        probs = self.classifier.predict_proba(X)
        idx = probs.argmax(axis=1)
        labels = self.encoder.inverse_transform(idx)
        confidences = probs.max(axis=1).tolist()
        return labels.tolist(), confidences

# ---------------------------
# Main streaming pipeline
# ---------------------------

def process_file(path: str, ff: FraudFilter, use_llm: bool = True, batch_size: int = BATCH_SIZE):
    """
    Main function: streams file lines in batches, classifies, and saves outputs.
    """
    print(f"\n{'='*80}")
    print(f"[INFO] Processing file: {path}")
    print(f"{'='*80}")
    base = os.path.splitext(os.path.basename(path))[0]
    
    # Ensure output directory exists
    ensure_output_dir()

    structured_lines = []
    unstructured_lines = []
    discarded_lines = []

    # Streaming read & classify in batches
    print("[INFO] Phase 1: Classifying lines...")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        buffer = []
        line_ct = 0
        for raw in f:
            line_ct += 1
            line = clean_html_line(raw)
            if not line:
                continue
            buffer.append(line)

            if len(buffer) >= batch_size:
                labels, confs = ff.predict_batch(buffer)
                for l, lab, conf in zip(buffer, labels, confs):
                    if lab == "structured":
                        structured_lines.append(l)
                    elif lab == "unstructured":
                        unstructured_lines.append(l)
                    else:
                        discarded_lines.append(l)
                buffer = []

        # leftover
        if buffer:
            labels, confs = ff.predict_batch(buffer)
            for l, lab, conf in zip(buffer, labels, confs):
                if lab == "structured":
                    structured_lines.append(l)
                elif lab == "unstructured":
                    unstructured_lines.append(l)
                else:
                    discarded_lines.append(l)

    print(f"[INFO] Lines processed: {line_ct}")
    print(f"[INFO] ├─ Structured: {len(structured_lines)}")
    print(f"[INFO] ├─ Unstructured: {len(unstructured_lines)}")
    print(f"[INFO] └─ Discarded: {len(discarded_lines)}")

    # Save to files in output directory
    print("\n[INFO] Phase 2: Saving classified text...")
    out_structured = os.path.join(OUTPUT_DIR, f"{base}_structured.txt")
    out_unstructured = os.path.join(OUTPUT_DIR, f"{base}_unstructured.txt")
    out_discarded = os.path.join(OUTPUT_DIR, f"{base}_discarded.txt")

    with open(out_structured, "w", encoding="utf-8") as f:
        f.write("\n".join(structured_lines))
    with open(out_unstructured, "w", encoding="utf-8") as f:
        f.write("\n".join(unstructured_lines))
    with open(out_discarded, "w", encoding="utf-8") as f:
        f.write("\n".join(discarded_lines))

    print(f"[INFO] ✓ Saved: {out_structured}")
    print(f"[INFO] ✓ Saved: {out_unstructured}")
    print(f"[INFO] ✓ Saved: {out_discarded}")

    # Extract tables from structured lines and save as CSVs
    print("\n[INFO] Phase 3: Extracting tables...")
    tables = extract_tables_from_lines(structured_lines)
    print(f"[INFO] Detected {len(tables)} table blocks in structured lines.")
    
    # Save all tables as CSV first
    for i, table in enumerate(tables, start=1):
        csv_path = os.path.join(OUTPUT_DIR, f"{base}_table_{i}.csv")
        df = pd.DataFrame({"raw_line": table})
        df.to_csv(csv_path, index=False)
        print(f"[INFO] ✓ Saved table {i} -> {csv_path} ({len(table)} lines)")
    
    # Process tables with LLM (batch or individual)
    if use_llm and tables:
        print(f"\n[INFO] Phase 4: LLM Analysis ({'BATCH' if BATCH_PROCESS_TABLES else 'INDIVIDUAL'} mode)...")
        
        # Filter tables based on criteria
        tables_to_process = []
        for i, table in enumerate(tables, start=1):
            if len(table) >= MIN_TABLE_LINES:
                if MAX_TABLES_FOR_LLM == -1 or len(tables_to_process) < MAX_TABLES_FOR_LLM:
                    tables_to_process.append((i, table))
        
        if tables_to_process:
            if BATCH_PROCESS_TABLES:
                # ⚡ BATCH MODE: Process all tables in one LLM call
                print(f"[INFO] Processing {len(tables_to_process)} tables with LLM in BATCH mode...")
                print(f"[INFO] (This will make ONE LLM call - much faster!)")
                
                batch_result = refine_tables_batch_with_llm(tables_to_process)
                
                # Save batch result
                batch_output_path = os.path.join(OUTPUT_DIR, f"{base}_tables_batch_refined.json")
                with open(batch_output_path, "w", encoding="utf-8") as fh:
                    fh.write(batch_result)
                print(f"[INFO] ✓ Batch LLM results saved -> {batch_output_path}")
                
                # Also save individual table results for convenience
                try:
                    batch_data = json.loads(batch_result)
                    if "tables" in batch_data:
                        for table_result in batch_data["tables"]:
                            table_id = table_result.get("table_id", 0)
                            if table_id > 0:
                                individual_path = os.path.join(OUTPUT_DIR, f"{base}_table_{table_id}_refined.json")
                                with open(individual_path, "w", encoding="utf-8") as fh:
                                    json.dump(table_result, fh, indent=2)
                        print(f"[INFO] ✓ Individual table JSONs also saved ({len(batch_data['tables'])} files)")
                except json.JSONDecodeError:
                    print(f"[WARN] Could not parse batch LLM output as JSON")
                    
            else:
                # INDIVIDUAL MODE: Process tables one by one (slower)
                print(f"[INFO] Processing {len(tables_to_process)} tables with LLM individually...")
                for i, table in tables_to_process:
                    print(f"[INFO] Refining table {i} with LLM...")
                    refined = refine_table_with_llm(table)
                    out_ref = os.path.join(OUTPUT_DIR, f"{base}_table_{i}_refined.json")
                    with open(out_ref, "w", encoding="utf-8") as fh:
                        fh.write(refined)
                    print(f"[INFO] ✓ Saved -> {out_ref}")
        
        # Report skipped tables
        skipped = len(tables) - len(tables_to_process)
        if skipped > 0:
            print(f"[INFO] Skipped {skipped} tables (too small: <{MIN_TABLE_LINES} lines or limit reached)")

    # Optional: run LLM on unstructured text (sample or full depending on size)
    if use_llm and unstructured_lines and not SKIP_ENTITY_EXTRACTION:
        print(f"\n[INFO] Phase 5: Entity Extraction...")
        print(f"[INFO] Extracting entities from unstructured text (sampling {min(UNSTRUCTURED_SAMPLE_SIZE, len(unstructured_lines))} lines)...")
        entities_out = extract_entities_from_unstructured(unstructured_lines, sample_size=UNSTRUCTURED_SAMPLE_SIZE)
        out_entities = os.path.join(OUTPUT_DIR, f"{base}_unstructured_entities.json")
        with open(out_entities, "w", encoding="utf-8") as fh:
            fh.write(entities_out)
        print(f"[INFO] ✓ Unstructured entities saved -> {out_entities}")
    elif SKIP_ENTITY_EXTRACTION:
        print(f"[INFO] Phase 5: Entity extraction disabled (SKIP_ENTITY_EXTRACTION=True)")

# ---------------------------
# Entry point: train/load and process files
# ---------------------------

def main():
    global USE_LLM
    
    print("="*80)
    print("Financial Document Processing Pipeline (Optimized with Batch LLM)")
    print("="*80)
    
    # Check if Ollama is available
    if USE_LLM:
        if not check_ollama_available():
            print("[WARN] Ollama not found. Disabling LLM features.")
            print("[INFO] To enable LLM features, install Ollama from https://ollama.ai")
            USE_LLM = False
        elif not check_model_available(LLM_MODEL):
            print(f"[WARN] Ollama model '{LLM_MODEL}' not found. Disabling LLM features.")
            print(f"[INFO] To enable LLM features, run: ollama pull {LLM_MODEL}")
            USE_LLM = False
        else:
            print(f"[INFO] ✓ Ollama found with model '{LLM_MODEL}'")
            print(f"[INFO] ✓ LLM Mode: {'BATCH (Fast!)' if BATCH_PROCESS_TABLES else 'INDIVIDUAL (Slower)'}")
            print(f"[INFO] ✓ Table limit: {'ALL' if MAX_TABLES_FOR_LLM == -1 else MAX_TABLES_FOR_LLM}")
    else:
        print("[INFO] LLM processing disabled (USE_LLM=False)")
    
    # Create filter and either load or train with tiny example dataset
    ff = FraudFilter()

    # If model exists, load it. Otherwise train a small starter classifier.
    if os.path.exists(MODEL_FILE) and os.path.exists(ENCODER_FILE):
        ff.load()
    else:
        # Minimal seed training data: PLEASE expand this with many more examples before production use.
        print("\n[INFO] No classifier found. Training starter model...")
        seed_texts = [
            # structured examples (tables / numeric rows)
            "Total assets 1,537,443 1,757,262 752,221",
            "Sales 1,452,169 1,037,972 230,899",
            "Income (loss) 610,160 76,443 (495,063)",
            "1st Quarter High Low 2016 2015",
            "Cash and cash equivalents 234,567 198,234",
            "Accounts receivable 456,789 398,456",
            "Property, plant and equipment 1,234,567 1,198,234",
            "Total liabilities 987,654 876,543",

            # unstructured examples (useful prose / legal / notes)
            "Original Sixteen to One Mine, Inc. (the Company) was incorporated in 1911 in California.",
            "For the twelve month period ended December 31, 2016, a total of 35 citations were issued.",
            "The Company lacks sufficient funds to implement long-term construction projects to increase mining efficiency.",
            "Management believes that the current economic conditions may impact future operations.",
            "The audit was conducted in accordance with generally accepted auditing standards.",
            "Risk factors include market volatility and regulatory changes.",

            # not useful (boilerplate / page markers / contact info)
            "For more information contact our PR team at pr@company.com",
            "This report contains 120 pages",
            "Page 5",
            "SECURITIES AND EXCHANGE COMMISSION WASHINGTON, D.C. 20549",
            "Table of Contents",
            "Forward-looking statements disclaimer",
            "Copyright 2023 All rights reserved"
        ]
        seed_labels = [
            "structured", "structured", "structured", "structured",
            "structured", "structured", "structured", "structured",
            "unstructured", "unstructured", "unstructured", "unstructured",
            "unstructured", "unstructured",
            "not useful", "not useful", "not useful", "not useful",
            "not useful", "not useful", "not useful"
        ]
        print("[INFO] Training with small seed dataset. Expand for production use.")
        ff.train(seed_texts, seed_labels)

    # Determine input files: either CLI-specified or all txt in various directories
    files_to_process = []
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if os.path.isfile(arg):
            files_to_process = [arg]
        elif os.path.isdir(arg):
            files_to_process = glob(os.path.join(arg, "*.txt"))
    else:
        # Check multiple possible locations
        search_dirs = ["./data", "/mnt/data", "."]
        for search_dir in search_dirs:
            if os.path.isdir(search_dir):
                files_to_process = glob(os.path.join(search_dir, "*.txt"))
                if files_to_process:
                    print(f"\n[INFO] Found {len(files_to_process)} .txt files in {search_dir}")
                    break

    if not files_to_process:
        print("\n[WARN] No .txt files found to process.")
        print("\n[INFO] Usage:")
        print("  python full_pipeline_llm_optimized.py                    # Process all .txt files")
        print("  python full_pipeline_llm_optimized.py /path/to/file.txt  # Process specific file")
        print("  python full_pipeline_llm_optimized.py /path/to/dir       # Process directory")
        return

    # Process each file
    print(f"\n[INFO] Processing {len(files_to_process)} file(s)...")
    for file_path in files_to_process:
        try:
            process_file(file_path, ff, use_llm=USE_LLM, batch_size=BATCH_SIZE)
            print(f"\n[SUCCESS] ✓ Completed processing {file_path}")
            print("-" * 80)
        except Exception as e:
            print(f"\n[ERROR] ✗ Failed to process {file_path}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*80}")
    print(f"[INFO] ✓ Processing complete! Processed {len(files_to_process)} file(s).")
    print(f"[INFO] ✓ All output files saved to: {OUTPUT_DIR}/")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()