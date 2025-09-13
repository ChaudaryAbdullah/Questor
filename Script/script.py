#!/usr/bin/env python3
"""
extract_financials.py
- Reads all .txt files under ./data
- Writes per-file outputs in ./output:
    structured_<base>.csv   -> key:value metadata
    unstructured_<base>.txt -> narrative text
    tables_<base>_<N>.csv   -> parsed financial tables
"""

import os
import re
import csv
from statistics import median, mean

DATA_FOLDER = "data"
OUTPUT_FOLDER = "output"

# Parameters
MIN_TABLE_ROWS = 2
BASE_GAP = 8
MATCH_THRESHOLD_MIN = 8

# --- Helpers ---

def is_noise(line):
    s = line.strip()
    if not s:
        return True
    if re.match(r"^<[^>]+>$", s):   # xml/html tags
        return True
    if re.match(r"^(-|=){3,}$", s): # separators
        return True
    return False

def looks_like_table_line(line):
    if not re.search(r"\d", line) and not line.strip().lower().startswith("year"):
        return False
    if re.search(r"\s{2,}", line):
        return True
    tokens = re.findall(r"[\d\$\.,\(\)]+", line)
    return len(tokens) >= 2

def is_table_title(line):
    kws = [
        "balance sheet",
        "statement of operations",
        "income statement",
        "statement of cash",
        "cash flows",
        "financial data"
    ]
    return any(k in line.lower() for k in kws)

_year_re = re.compile(r"^(19|20)\d{2}$")

def is_year_line(line):
    tokens = re.findall(r"\b\d{4}\b", line)
    return len([t for t in tokens if _year_re.match(t)]) >= 2

def token_spans(line):
    return [(m.group(0), m.start()) for m in re.finditer(r"\S+", line)]

def cluster_positions(starts):
    if not starts:
        return []
    starts_sorted = sorted(starts)
    if len(starts_sorted) < 2:
        return [starts_sorted[0]]
    diffs = [starts_sorted[i+1] - starts_sorted[i] for i in range(len(starts_sorted)-1)]
    med = int(median(diffs)) if diffs else BASE_GAP
    gap = max(BASE_GAP, med * 2)
    clusters, cur = [], [starts_sorted[0]]
    for s in starts_sorted[1:]:
        if s - cur[-1] <= gap:
            cur.append(s)
        else:
            clusters.append(cur)
            cur = [s]
    clusters.append(cur)
    return [int(round(mean(c))) for c in clusters]

# --- Table alignment ---

def align_table_block(block_lines):
    token_infos = []
    all_starts = []
    for L in block_lines:
        spans = token_spans(L)
        token_infos.append(spans)
        all_starts.extend(st for _, st in spans)

    if not all_starts:
        return None, None

    col_centers = cluster_positions(all_starts)
    if not col_centers:
        return None, None

    if len(col_centers) > 1:
        gaps = [col_centers[i+1] - col_centers[i] for i in range(len(col_centers)-1)]
        median_gap = int(median(gaps))
    else:
        median_gap = BASE_GAP
    match_threshold = max(MATCH_THRESHOLD_MIN, median_gap // 2 + 4)

    rows = []
    for spans in token_infos:
        row = [""] * len(col_centers)
        for text, st in spans:
            best_idx, best_dist = None, None
            for idx, c in enumerate(col_centers):
                d = abs(st - c)
                if best_dist is None or d < best_dist:
                    best_dist, best_idx = d, idx

            if best_dist is not None and best_dist <= match_threshold:
                token = text.strip()
                if row[best_idx]:
                    row[best_idx] += " " + token
                else:
                    row[best_idx] = token
            else:
                if st < col_centers[0]:
                    row[0] = (row[0] + " " + text).strip()
                elif st > col_centers[-1]:
                    row[-1] = (row[-1] + " " + text).strip()
                else:
                    row[best_idx] = (row[best_idx] + " " + text).strip()

        # Clean dashed rows (like ---- under headers)
        if all(re.match(r"^-+$", c) or c == "" for c in row):
            continue

        # If row has only years, treat as header row
        parts = [p for p in row if p.strip()]
        if parts and (parts[0].lower().startswith("year") or all(_year_re.match(p) for p in parts)):
            rows.append(parts)
        else:
            rows.append(row)

    # Detect header
    header_idx = None
    for i, r in enumerate(rows):
        if r and (r[0].lower().startswith("year") or any(_year_re.match(c) for c in r)):
            header_idx = i
            break

    if header_idx is not None:
        headers = rows[header_idx]
        data = [r + [""] * (len(headers) - len(r)) for r in rows[header_idx+1:]]
    else:
        max_cols = max(len(r) for r in rows)
        headers = [f"Col{c}" for c in range(max_cols)]
        data = [r + [""] * (max_cols - len(r)) for r in rows]

    return headers, data

# --- Extraction ---

def extract_key_values_and_unstructured(lines):
    kv, unstructured = [], []
    for L in lines:
        s = L.strip()
        if is_noise(s):
            continue
        if ":" in s and not s.startswith("<"):
            k, v = s.split(":", 1)
            k, v = k.strip(), v.strip()
            if 1 <= len(k) <= 60 and re.match(r"^[A-Za-z0-9 _\-\(\)]+$", k):
                kv.append((k, v))
                continue
        unstructured.append(L)
    return kv, unstructured

def find_financial_table_blocks(lines):
    blocks = []
    n = len(lines)
    i = 0
    while i < n:
        l = lines[i]
        if is_table_title(l) or is_year_line(l) or l.strip().lower().startswith("year"):
            title = l.strip() if is_table_title(l) else "FinancialTable"
            j = i
            block = []
            while j < n and (looks_like_table_line(lines[j]) or lines[j].strip()):
                if is_table_title(lines[j]) and j != i:
                    break
                block.append(lines[j].rstrip("\n"))
                j += 1
            if len(block) >= MIN_TABLE_ROWS:
                blocks.append((title, block))
            i = j
        else:
            i += 1
    return blocks

# --- File Processing ---

def process_file(path, outdir):
    base = os.path.splitext(os.path.basename(path))[0]
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw_lines = f.readlines()

    kvs, unstructured = extract_key_values_and_unstructured(raw_lines)
    table_blocks = find_financial_table_blocks(raw_lines)

    os.makedirs(outdir, exist_ok=True)

    # Structured key-values
    kv_csv = os.path.join(outdir, f"structured_{base}.csv")
    with open(kv_csv, "w", newline="", encoding="utf-8") as outf:
        writer = csv.writer(outf)
        writer.writerow(["Key", "Value"])
        for k, v in kvs:
            writer.writerow([k, v])

    # Unstructured
    un_file = os.path.join(outdir, f"unstructured_{base}.txt")
    with open(un_file, "w", encoding="utf-8") as outf:
        outf.writelines(unstructured)

    # Tables
    table_files = []
    for idx, (title, block_lines) in enumerate(table_blocks, start=1):
        headers, rows = align_table_block(block_lines)
        if headers and rows:
            tfile = os.path.join(outdir, f"tables_{base}_{idx}.csv")
            with open(tfile, "w", newline="", encoding="utf-8") as outf:
                writer = csv.writer(outf)
                if title:
                    writer.writerow([f"Table Title: {title}"])
                writer.writerow(headers)
                for r in rows:
                    writer.writerow(r)
            table_files.append(tfile)

    print(f"Processed {path}: {len(kvs)} metadata, {len(table_files)} tables, {len(unstructured)} unstructured lines.")
    return kv_csv, table_files, un_file

def process_all(data_folder=DATA_FOLDER, out_folder=OUTPUT_FOLDER):
    os.makedirs(out_folder, exist_ok=True)
    for fn in sorted(os.listdir(data_folder)):
        if fn.lower().endswith(".txt"):
            process_file(os.path.join(data_folder, fn), out_folder)

if __name__ == "__main__":
    process_all()
    print("Done.")
