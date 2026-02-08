"""
Batch Inference Runner

Process all JSON files in the Input folder and generate predictions.
Creates aggregated summary report of all results.
"""

import os
import sys
import json
import glob
from datetime import datetime
from inference_pipeline import main as run_single_inference


INPUT_DIR = "Input"
OUTPUT_DIR = "Output"


def process_all_json_files(input_dir: str = INPUT_DIR):
    """Process all JSON files in the input directory."""
    # Find all JSON files
    json_pattern = os.path.join(input_dir, "*.json")
    json_files = glob.glob(json_pattern)
    
    if not json_files:
        print(f"❌ No JSON files found in {input_dir}")
        return
    
    print(f"🔍 Found {len(json_files)} JSON file(s) to process")
    print("="*60)
    
    # Process each file
    results = []
    successful = 0
    failed = 0
    
    for json_file in json_files:
        try:
            print(f"\n📄 Processing: {os.path.basename(json_file)}")
            result = run_single_inference(json_file)
            results.append(result)
            successful += 1
        except Exception as e:
            print(f"❌ Error processing {os.path.basename(json_file)}: {e}")
            failed += 1
            results.append({
                'input_file': os.path.basename(json_file),
                'error': str(e),
                'status': 'FAILED'
            })
    
    # Generate aggregated report
    print("\n" + "="*60)
    print("BATCH PROCESSING SUMMARY")
    print("="*60)
    print(f"Total files: {len(json_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if successful > 0:
        generate_batch_report(results)
    
    return results


def generate_batch_report(results: list):
    """Generate aggregated batch processing report."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(OUTPUT_DIR, f"batch_report_{timestamp}.txt")
    
    # Count risk levels
    risk_counts = {'CRITICAL': 0, 'HIGH': 0, 'MODERATE': 0, 'LOW': 0, 'MINIMAL': 0}
    fraud_count = 0
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("BATCH FRAUD DETECTION ANALYSIS REPORT\n")
        f.write("="*70 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total files processed: {len(results)}\n\n")
        
        # Summary statistics
        valid_results = [r for r in results if 'risk_score' in r]
        
        if valid_results:
            avg_risk = sum(r['risk_score'] for r in valid_results) / len(valid_results)
            max_risk = max(r['risk_score'] for r in valid_results)
            min_risk = min(r['risk_score'] for r in valid_results)
            
            for r in valid_results:
                if r.get('risk_level'):
                    risk_counts[r['risk_level']] += 1
                if r['overall_prediction'] == 'FRAUD':
                    fraud_count += 1
            
            f.write(f"Average Risk Score: {avg_risk:.6f}\n")
            f.write(f"Max Risk Score: {max_risk:.6f}\n")
            f.write(f"Min Risk Score: {min_risk:.6f}\n\n")
            
            f.write("Risk Level Distribution:\n")
            for level in ['CRITICAL', 'HIGH', 'MODERATE', 'LOW', 'MINIMAL']:
                count = risk_counts[level]
                pct = (count / len(valid_results) * 100) if valid_results else 0
                f.write(f"  {level:12} : {count:3} ({pct:5.1f}%)\n")
            
            f.write(f"\nFraud Cases Detected: {fraud_count}/{len(valid_results)} ")
            f.write(f"({fraud_count/len(valid_results)*100:.1f}%)\n\n")
        
        # Individual file results
        f.write("="*70 + "\n")
        f.write("INDIVIDUAL FILE RESULTS\n")
        f.write("="*70 + "\n\n")
        
        # Sort by risk score (highest first)
        valid_results_sorted = sorted(valid_results, key=lambda x: x.get('risk_score', 0), reverse=True)
        
        for result in valid_results_sorted:
            f.write(f"File: {result['input_file']}\n")
            f.write(f"  Risk Score: {result['risk_score']:.6f}\n")
            f.write(f"  Risk Level: {result['risk_level']}\n")
            f.write(f"  Prediction: {result['overall_prediction']}\n")
            f.write(f"  Models Flagging FRAUD: {result['fraud_model_count']}/{result['total_models']}\n")
            f.write(f"  ---\n\n")
        
        # Failed files
        failed_results = [r for r in results if 'error' in r]
        if failed_results:
            f.write("="*70 + "\n")
            f.write("FAILED FILES\n")
            f.write("="*70 + "\n\n")
            for result in failed_results:
                f.write(f"File: {result['input_file']}\n")
                f.write(f"  Error: {result.get('error', 'Unknown error')}\n\n")
    
    print(f"\n✅ Batch report saved to: {report_path}")
    print(f"\n📊 SUMMARY:")
    print(f"   Fraud cases detected: {fraud_count}/{len(valid_results)}")
    print(f"   Critical risk: {risk_counts['CRITICAL']}")
    print(f"   High risk: {risk_counts['HIGH']}")
    print(f"   Moderate risk: {risk_counts['MODERATE']}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
    else:
        input_dir = INPUT_DIR
    
    if not os.path.exists(input_dir):
        print(f"❌ Input directory not found: {input_dir}")
        sys.exit(1)
    
    process_all_json_files(input_dir)
