import os
from pathlib import Path

def full_diagnostic(base_dir):
    """Find EVERY file in the directory."""
    
    print(f"\n{'='*70}")
    print(f"FULL DIAGNOSTIC SCAN: {base_dir}")
    print(f"{'='*70}")
    
    # Method 1: os.walk (what your script uses)
    count_walk = 0
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if not f.startswith('.'):
                count_walk += 1
    
    print(f"\n1️⃣ os.walk count: {count_walk} files")
    
    # Method 2: Count manually per top-level folder
    print(f"\n2️⃣ Manual count per folder:")
    print("-"*70)
    
    base_path = Path(base_dir)
    total_manual = 0
    
    for item in sorted(base_path.iterdir()):
        if item.is_dir():
            # Count ALL files recursively
            file_list = list(item.rglob('*'))
            file_count = sum(1 for f in file_list if f.is_file() and not f.name.startswith('.'))
            total_manual += file_count
            print(f"   📁 {item.name}: {file_count} files")
            
            # If it's a dataset folder, show more detail
            if 'Fraudulent' in item.name or 'fraudulent' in item.name.lower():
                # Show subfolders
                for sub in sorted(item.iterdir())[:3]:
                    if sub.is_dir():
                        sub_count = sum(1 for f in sub.rglob('*') if f.is_file())
                        print(f"      └── {sub.name}: {sub_count} files")
                        
            elif 'Non Fraudulent' in item.name or 'non' in item.name.lower():
                # Show year folders
                for sub in sorted(item.iterdir()):
                    if sub.is_dir():
                        sub_count = sum(1 for f in sub.rglob('*') if f.is_file())
                        print(f"      └── {sub.name}: {sub_count} files")
    
    print("-"*70)
    print(f"   TOTAL (manual): {total_manual} files")
    
    # Method 3: Direct count of each dataset
    print(f"\n3️⃣ Direct dataset count:")
    print("-"*70)
    
    # Try different possible folder names
    fraud_names = ["Fraudulent Dataset", "Fraudulant Dataset", "fraudulent", "Fraudulent"]
    non_fraud_names = ["Non Fraudulent Dataset", "Non-Fraudulent Dataset", "non_fraudulent", "NonFraudulent"]
    
    fraud_path = None
    non_fraud_path = None
    
    for name in fraud_names:
        test_path = base_path / name
        if test_path.exists():
            fraud_path = test_path
            break
    
    for name in non_fraud_names:
        test_path = base_path / name
        if test_path.exists():
            non_fraud_path = test_path
            break
    
    fraud_count = 0
    non_fraud_count = 0
    
    if fraud_path:
        fraud_count = sum(1 for f in fraud_path.rglob('*') if f.is_file() and not f.name.startswith('.'))
        print(f"   ✓ Fraudulent ({fraud_path.name}): {fraud_count} files")
    else:
        print(f"   ❌ Fraudulent folder NOT FOUND")
    
    if non_fraud_path:
        non_fraud_count = sum(1 for f in non_fraud_path.rglob('*') if f.is_file() and not f.name.startswith('.'))
        print(f"   ✓ Non Fraudulent ({non_fraud_path.name}): {non_fraud_count} files")
    else:
        print(f"   ❌ Non Fraudulent folder NOT FOUND")
    
    print("-"*70)
    print(f"   TOTAL: {fraud_count + non_fraud_count} files")
    
    # Show ALL top-level contents
    print(f"\n4️⃣ ALL contents of '{base_dir}':")
    print("-"*70)
    for item in sorted(base_path.iterdir()):
        item_type = "📁" if item.is_dir() else "📄"
        print(f"   {item_type} {item.name}")
    
    # Check if data is in a different location
    print(f"\n5️⃣ Checking parent directory too:")
    print("-"*70)
    parent = base_path.parent
    for item in sorted(parent.iterdir()):
        if item.is_dir():
            item_count = sum(1 for f in item.rglob('*') if f.is_file())
            print(f"   📁 {item.name}: {item_count} files")

    return fraud_count, non_fraud_count

if __name__ == "__main__":
    # Run diagnostic
    full_diagnostic("Data")
    
    # Also try current directory
    print(f"\n\n{'='*70}")
    print("Also checking current directory contents:")
    print(f"{'='*70}")
    for item in sorted(Path('.').iterdir()):
        if item.is_dir() and not item.name.startswith('.'):
            count = sum(1 for f in item.rglob('*') if f.is_file())
            print(f"   📁 {item.name}: {count} files")