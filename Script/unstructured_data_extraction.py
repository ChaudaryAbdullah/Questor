import os
import re
from pathlib import Path
from bs4 import BeautifulSoup
from tqdm import tqdm

def extract_unstructured_data(file_path):
    """
    Extract unstructured text data from SEC filing documents.
    Removes structured tags and extracts human-readable text.
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Find all documents within the file
        documents = re.findall(r'<DOCUMENT>(.*?)</DOCUMENT>', content, re.DOTALL)
        
        unstructured_text = []
        
        # If no <DOCUMENT> tags found, process the raw content (for plain text files)
        if not documents:
            # Try to clean up any HTML/XML if present
            soup = BeautifulSoup(content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            # Remove excessive blank lines
            text = re.sub(r'\n\s*\n+', '\n\n', text)
            
            # Remove any remaining tags
            text = re.sub(r'<[^>]+>', '', text)
            
            if text.strip():
                return text.strip()
            return None
        
        # Process SEC-style documents with <DOCUMENT> tags
        for doc in documents:
            # Extract document type
            doc_type_match = re.search(r'<TYPE>(.*?)\n', doc)
            doc_type = doc_type_match.group(1).strip() if doc_type_match else 'UNKNOWN'
            
            # Extract text content
            text_match = re.search(r'<TEXT>(.*?)</TEXT>', doc, re.DOTALL)
            if text_match:
                text_content = text_match.group(1)
                
                # Parse HTML/SGML content
                soup = BeautifulSoup(text_content, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # Get text
                text = soup.get_text()
                
                # Clean up text
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = '\n'.join(chunk for chunk in chunks if chunk)
                
                # Remove excessive blank lines
                text = re.sub(r'\n\s*\n+', '\n\n', text)
                
                # Remove structured data markers
                text = re.sub(r'<[^>]+>', '', text)  # Remove any remaining tags
                
                if text.strip():
                    unstructured_text.append(f"{'='*70}\n")
                    unstructured_text.append(f"DOCUMENT TYPE: {doc_type}\n")
                    unstructured_text.append(f"{'='*70}\n\n")
                    unstructured_text.append(text)
                    unstructured_text.append(f"\n\n")
        
        return ''.join(unstructured_text) if unstructured_text else None
    
    except Exception as e:
        print(f"\nError processing {file_path}: {str(e)}")
        return None

def get_file_metadata(file_path, base_path):
    """
    Extract metadata from file path structure.
    """
    relative_path = file_path.relative_to(base_path)
    parts = relative_path.parts
    path_str = str(relative_path)
    
    metadata = {
        'category': None,
        'cik': None,
        'filing_type': None,
        'accession_number': None,
        'year': None,
        'date': None
    }
    
    # FIXED: Check 'Non Fraudulent' FIRST (because 'Fraudulent' is substring of 'Non Fraudulent')
    if 'Non Fraudulent Dataset' in parts or 'Non Fraudulent Dataset' in path_str:
        metadata['category'] = 'Non Fraudulent'
        # Structure: Non Fraudulent Dataset/Year/CIK_Date.txt
        if len(parts) >= 2:
            metadata['year'] = parts[1] if len(parts) > 1 and parts[1].isdigit() else None
            
            # Extract CIK and date from filename
            filename = file_path.stem  # filename without extension
            match = re.match(r'(\d+)_(\d{8})', filename)
            if match:
                metadata['cik'] = match.group(1)
                metadata['date'] = match.group(2)
    
    elif 'Fraudulent Dataset' in parts or 'Fraudulent Dataset' in path_str:
        metadata['category'] = 'Fraudulent'
        # Structure: Fraudulent Dataset/sec-edgar-filings/CIK/Filing Type/Accession Number/file
        if len(parts) >= 5:
            metadata['cik'] = parts[2] if len(parts) > 2 else None
            metadata['filing_type'] = parts[3] if len(parts) > 3 else None
            metadata['accession_number'] = parts[4] if len(parts) > 4 else None
    
    return metadata

def generate_unique_filename(metadata, original_filename, file_counter):
    """
    Generate a unique filename based on metadata.
    """
    if metadata['category'] == 'Non Fraudulent':
        # Format: NonFraud_CIK_Date_counter.txt
        if metadata['cik'] and metadata['date']:
            return f"NonFraud_{metadata['cik']}_{metadata['date']}_{file_counter}.txt"
        else:
            return f"NonFraud_{original_filename}_{file_counter}.txt"
    
    elif metadata['category'] == 'Fraudulent':
        # Format: Fraud_CIK_FilingType_Accession_counter.txt
        if metadata['cik'] and metadata['accession_number']:
            filing = metadata['filing_type'].replace('/', '-') if metadata['filing_type'] else 'UNKNOWN'
            accession = metadata['accession_number'].replace('/', '-')
            return f"Fraud_{metadata['cik']}_{filing}_{accession}_{file_counter}.txt"
        else:
            return f"Fraud_{original_filename}_{file_counter}.txt"
    
    else:
        return f"Unknown_{original_filename}_{file_counter}.txt"

def is_valid_file(file_path):
    """
    Check if file should be processed.
    """
    file_name = file_path.name
    
    # Skip hidden files and system files
    if file_name.startswith('.') or file_name in ['.DS_Store', '.gitkeep', 'desktop.ini']:
        return False
    
    # Skip directories
    if file_path.is_dir():
        return False
    
    # Accept files with .txt extension or no extension
    # Also accept .htm and .html files
    valid_extensions = ['.txt', '.htm', '.html', '']
    
    if file_path.suffix.lower() in valid_extensions:
        return True
    
    # Check if it's a text file by trying to read it
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            first_line = f.readline()
            # Check if it looks like a text file
            return len(first_line) > 0
    except:
        return False

def process_files(input_base_dir, output_base_dir):
    """
    Process all files in the input directory structure and save unstructured data
    in a flat structure (all files in one folder).
    """
    input_path = Path(input_base_dir)
    output_path = Path(output_base_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Counter for processed files
    stats = {
        'total_processed': 0,
        'total_errors': 0,
        'fraudulent': 0,
        'non_fraudulent': 0,
        'by_year': {},
        'by_filing_type': {}
    }
    
    file_counter = 0
    
    # Collect all files to process
    print("\n🔍 Scanning for files...")
    all_files = []
    
    for root, dirs, files in os.walk(input_path):
        root_path = Path(root)
        
        # Skip the unstructured_data directory itself
        if 'unstructured_data' in root_path.parts:
            continue
        
        for file in files:
            file_path = root_path / file
            
            # Check if file is valid for processing
            if is_valid_file(file_path):
                all_files.append(file_path)
    
    print(f"✓ Found {len(all_files)} files to process")
    
    # FIXED: Check 'Non Fraudulent' FIRST
    non_fraud_count = sum(1 for f in all_files if 'Non Fraudulent Dataset' in str(f))
    fraud_count = sum(1 for f in all_files if 'Fraudulent Dataset' in str(f) and 'Non Fraudulent Dataset' not in str(f))
    
    print(f"  • Fraudulent Dataset: {fraud_count} files")
    print(f"  • Non Fraudulent Dataset: {non_fraud_count} files")
    print("="*70)
    
    # Process files with progress bar
    for input_file_path in tqdm(all_files, desc="Extracting unstructured data", unit="file"):
        try:
            file_counter += 1
            
            # Get metadata
            metadata = get_file_metadata(input_file_path, input_path)
            
            # Get relative path for metadata
            relative_path = input_file_path.relative_to(input_path)
            
            # Generate unique filename
            original_filename = input_file_path.stem
            output_filename = generate_unique_filename(metadata, original_filename, file_counter)
            
            # Output file path (flat structure - all in one folder)
            output_file_path = output_path / output_filename
            
            # Extract unstructured data
            unstructured_data = extract_unstructured_data(input_file_path)
            
            if unstructured_data:
                # Save to output file with metadata header
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    # Write metadata header
                    f.write("="*70 + "\n")
                    f.write("FILE METADATA\n")
                    f.write("="*70 + "\n")
                    f.write(f"Source File: {relative_path}\n")
                    f.write(f"Category: {metadata['category']}\n")
                    
                    if metadata['cik']:
                        f.write(f"CIK: {metadata['cik']}\n")
                    if metadata['filing_type']:
                        f.write(f"Filing Type: {metadata['filing_type']}\n")
                    if metadata['accession_number']:
                        f.write(f"Accession Number: {metadata['accession_number']}\n")
                    if metadata['year']:
                        f.write(f"Year: {metadata['year']}\n")
                    if metadata['date']:
                        f.write(f"Date: {metadata['date']}\n")
                    
                    f.write("="*70 + "\n\n")
                    
                    # Write unstructured data
                    f.write(unstructured_data)
                
                # Update statistics
                stats['total_processed'] += 1
                
                if metadata['category'] == 'Fraudulent':
                    stats['fraudulent'] += 1
                elif metadata['category'] == 'Non Fraudulent':
                    stats['non_fraudulent'] += 1
                
                if metadata['year']:
                    stats['by_year'][metadata['year']] = stats['by_year'].get(metadata['year'], 0) + 1
                
                if metadata['filing_type']:
                    stats['by_filing_type'][metadata['filing_type']] = stats['by_filing_type'].get(metadata['filing_type'], 0) + 1
            else:
                stats['total_errors'] += 1
        
        except Exception as e:
            stats['total_errors'] += 1
            tqdm.write(f"\n✗ Error processing {input_file_path.name}: {str(e)}")
    
    # Print detailed summary
    print(f"\n{'='*70}")
    print("PROCESSING COMPLETE!")
    print(f"{'='*70}")
    print(f"\n📊 SUMMARY STATISTICS:")
    print(f"{'─'*70}")
    print(f"Total files processed: {stats['total_processed']}")
    print(f"Total errors: {stats['total_errors']}")
    print(f"\n📁 BY CATEGORY:")
    print(f"  • Fraudulent Dataset: {stats['fraudulent']} files")
    print(f"  • Non Fraudulent Dataset: {stats['non_fraudulent']} files")
    
    if stats['by_year']:
        print(f"\n📅 BY YEAR (Non Fraudulent):")
        for year in sorted(stats['by_year'].keys()):
            print(f"  • {year}: {stats['by_year'][year]} files")
    
    if stats['by_filing_type']:
        print(f"\n📄 BY FILING TYPE (Fraudulent):")
        for filing_type in sorted(stats['by_filing_type'].keys()):
            print(f"  • {filing_type}: {stats['by_filing_type'][filing_type]} files")
    
    print(f"\n💾 Output location: {output_path}")
    print(f"📝 All files saved in flat structure (no subfolders)")
    print(f"{'='*70}\n")

def preview_structure(base_dir):
    """
    Preview the directory structure before processing.
    """
    print("\n📂 DIRECTORY STRUCTURE PREVIEW:")
    print("="*70)
    
    base_path = Path(base_dir)
    file_count = {'fraudulent': 0, 'non_fraudulent': 0}
    
    # Count files in each category
    for root, dirs, files in os.walk(base_path):
        root_path = Path(root)
        if 'unstructured_data' in root_path.parts:
            continue
        
        for file in files:
            file_path = root_path / file
            
            # Use the same validation logic
            if is_valid_file(file_path):
                path_str = str(root_path)
                # FIXED: Check 'Non Fraudulent' FIRST
                if 'Non Fraudulent Dataset' in path_str:
                    file_count['non_fraudulent'] += 1
                elif 'Fraudulent Dataset' in path_str:
                    file_count['fraudulent'] += 1
    
    print(f"├── Fraudulent Dataset: {file_count['fraudulent']} files")
    print(f"│   └── Structure: sec-edgar-filings/[CIK]/[Filing Type]/[Accession]/full-submission.txt")
    print(f"│")
    print(f"└── Non Fraudulent Dataset: {file_count['non_fraudulent']} files")
    print(f"    └── Structure: [Year]/[CIK]_[Date].txt")
    print(f"\nTotal files: {file_count['fraudulent'] + file_count['non_fraudulent']}")
    print(f"\n⚠️  OUTPUT: All files will be saved in flat structure:")
    print(f"    Data/unstructured_data/")
    print(f"    ├── NonFraud_[CIK]_[Date]_[counter].txt")
    print(f"    ├── Fraud_[CIK]_[FilingType]_[Accession]_[counter].txt")
    print(f"    └── ...")
    print("="*70)
    
    # Debug: Show some sample non-fraudulent files if count is 0
    if file_count['non_fraudulent'] == 0:
        print("\n⚠️  WARNING: No Non Fraudulent files detected!")
        print("Checking directory structure...\n")
        
        non_fraud_path = base_path / "Non Fraudulent Dataset"
        if non_fraud_path.exists():
            print(f"✓ Found: {non_fraud_path}")
            # List subdirectories
            subdirs = [d for d in non_fraud_path.iterdir() if d.is_dir()]
            print(f"  Subdirectories: {[d.name for d in subdirs[:5]]}")
            
            # Check files in first subdirectory
            if subdirs:
                first_subdir = subdirs[0]
                files_in_subdir = list(first_subdir.iterdir())
                print(f"\n  Files in {first_subdir.name}/:")
                for f in files_in_subdir[:5]:
                    print(f"    • {f.name} (extension: '{f.suffix}')")
        else:
            print(f"✗ Directory not found: {non_fraud_path}")

def main():
    # Define paths
    input_base_dir = "Data"  # Your base directory
    output_base_dir = "Data/unstructured_data"  # Output directory
    
    # Check if input directory exists
    if not os.path.exists(input_base_dir):
        print(f"❌ Error: Input directory '{input_base_dir}' does not exist!")
        return
    
    # Preview structure
    preview_structure(input_base_dir)
    
    # Ask for confirmation
    print("\n" + "="*70)
    response = input("Do you want to proceed with extraction? (yes/no): ").lower()
    if response not in ['yes', 'y']:
        print("❌ Extraction cancelled.")
        return
    
    print("\n" + "="*70)
    print("🚀 Starting extraction of unstructured data...")
    print(f"📁 Input directory: {input_base_dir}")
    print(f"💾 Output directory: {output_base_dir} (FLAT STRUCTURE)")
    print("="*70)
    
    # Process all files
    process_files(input_base_dir, output_base_dir)
    
    print("✅ Extraction completed successfully!")
    print("✅ All files saved in flat structure without subfolders!")

if __name__ == "__main__":
    main()