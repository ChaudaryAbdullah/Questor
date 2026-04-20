#!/usr/bin/env python3
import os
import json
import glob
from pathlib import Path

def load_ground_truth(filepath):
    truth = {}
    if not os.path.exists(filepath):
        print(f"Warning: Ground truth file not found at {filepath}")
        return truth
        
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    for line in lines[2:]: # Skip header
        line = line.strip()
        if not line:
            continue
        parts = line.split('\t')
        if len(parts) >= 2:
            cik = parts[0].strip()
            status = parts[1].strip()
            if status == 'Fraudulent':
                truth[cik] = 1
            elif status == 'Non-Fraudulent':
                truth[cik] = 0
    return truth

def get_run_name(result_item):
    """Determine what components were active from the output flags."""
    struc = result_item.get('structured') is not None
    unstruc = result_item.get('unstructured') is not None
    agents = result_item.get('agents') is not None
    
    if struc and unstruc and agents:
        return "1. Baseline (All Components)"
    elif struc and unstruc and not agents:
        return "2. No Agents"
    elif not struc and unstruc and agents:
        return "3. No Structured Pipeline"
    elif struc and not unstruc and agents:
        return "4. No Unstructured Pipeline"
    elif not struc and not unstruc and agents:
        return "5. Agents Only"
    elif struc and not unstruc and not agents:
        return "6. Structured Only"
    elif not struc and unstruc and not agents:
        return "7. Unstructured Only"
    else:
        return "Unknown Config"

def evaluate():
    base_dir = Path(__file__).parent
    truth_file = base_dir / 'unstructured_pipeline' / 'CIK_fraud_status.txt'
    output_dir = base_dir / 'Output'
    
    ground_truth = load_ground_truth(truth_file)
    if not ground_truth:
        print("No ground truth labels loaded.")
        return
        
    json_files = glob.glob(str(output_dir / 'unified_results_*.json'))
    if not json_files:
        print(f"No result JSON files found in {output_dir}")
        return
        
    # Sort by creation time to roughly match the bash script execution order
    json_files.sort(key=os.path.getmtime)
    
    metrics = []
    
    for f in json_files:
        with open(f, 'r') as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                continue
                
        results = data.get('results', [])
        if not results:
            continue
            
        run_name = get_run_name(results[0])
        
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        total_evaled = 0
        
        for item in results:
            cik = str(item.get('cik'))
            if cik not in ground_truth:
                continue
                
            true_label = ground_truth[cik]
            
            # Extract combined score safely
            combined = item.get('combined', {})
            combined_risk = combined.get('combined_risk', {})
            score = combined_risk.get('overall_risk_score', 0)
            
            # Predict fraud if score >= 50
            pred_label = 1 if score >= 50 else 0
            
            if true_label == 1 and pred_label == 1:
                tp += 1
            elif true_label == 0 and pred_label == 0:
                tn += 1
            elif true_label == 0 and pred_label == 1:
                fp += 1
            elif true_label == 1 and pred_label == 0:
                fn += 1
            total_evaled += 1
            
        if total_evaled == 0:
            continue
            
        accuracy = (tp + tn) / total_evaled if total_evaled > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics.append({
            'Run': run_name,
            'Files Used': f"{total_evaled}/{len(results)}",
            'Accuracy': f"{accuracy*100:.1f}%",
            'Precision': f"{precision*100:.1f}%",
            'Recall': f"{recall*100:.1f}%",
            'F1 Score': f"{f1*100:.1f}%"
        })
        
    # Sort metrics based on name to get 1, 2, 3..
    metrics.sort(key=lambda x: x['Run'])

    # Print Table
    if not metrics:
        print("No evaluations could be made.")
        return

    print("="*85)
    print(f"{'Ablation Study Configuration':<40} | {'Docs':<8} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<8} | {'F1 Score':<8}")
    print("-" * 85)
    for m in metrics:
        print(f"{m['Run']:<40} | {m['Files Used']:<8} | {m['Accuracy']:<8} | {m['Precision']:<9} | {m['Recall']:<8} | {m['F1 Score']:<8}")
    print("="*85)

if __name__ == "__main__":
    evaluate()
