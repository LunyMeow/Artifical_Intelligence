#!/usr/bin/env python3
"""
Export embeddings from SQLite DB to CSV format suitable for C++ CommandParamExtractor.

Usage:
    python3 export_embeddings_for_cpp.py --db embeddings.db --output embeddings.csv

Output format:
    word,dim_0,dim_1,dim_2,...,dim_49
    copy,0.123,-0.456,0.789,...
    file,0.234,0.567,-0.890,...
"""

import sqlite3
import argparse
import numpy as np
import sys


def export_embeddings(db_path, output_path):
    """
    Export embeddings from SQLite to CSV.
    
    Args:
        db_path: Path to embeddings.db
        output_path: Path to output CSV
    
    Returns:
        Tuple: (total_exported, skipped, errors)
    """
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
    except sqlite3.Error as e:
        print(f"[ERROR] Cannot open DB: {e}", file=sys.stderr)
        return 0, 0, 1

    print(f"[INFO] Reading embeddings from {db_path}...")

    # Get all embeddings
    try:
        cur.execute("SELECT word, vector FROM embeddings")
        rows = cur.fetchall()
    except sqlite3.Error as e:
        print(f"[ERROR] DB query failed: {e}", file=sys.stderr)
        conn.close()
        return 0, 0, 1

    if not rows:
        print("[WARNING] No embeddings found in DB!", file=sys.stderr)
        conn.close()
        return 0, 0, 0

    print(f"[INFO] Found {len(rows)} embedding entries")

    # Write CSV
    total_exported = 0
    skipped = 0
    errors = 0

    try:
        with open(output_path, 'w') as csvfile:
            # Write header
            header = "word"
            for i in range(50):
                header += f",dim_{i}"
            csvfile.write(header + "\n")

            for word, vector_str in rows:
                try:
                    # Parse vector (format depends on how it was stored)
                    # Try comma-separated first
                    if vector_str is None:
                        print(f"[WARNING] Null vector for word: {word}")
                        skipped += 1
                        continue

                    # Handle different vector storage formats
                    if isinstance(vector_str, bytes):
                        vector_str = vector_str.decode('utf-8')

                    if vector_str.startswith('['):
                        # NumPy array format [0.1, 0.2, ...]
                        vector_str = vector_str.strip('[]')

                    dims = [float(x.strip()) for x in vector_str.split(',')]

                    if len(dims) != 50:
                        print(f"[WARNING] Word '{word}': expected 50 dims, got {len(dims)}")
                        skipped += 1
                        continue

                    # Write to CSV
                    line = word
                    for dim in dims:
                        line += f",{dim:.6f}"
                    csvfile.write(line + "\n")

                    total_exported += 1

                    if total_exported % 50 == 0:
                        print(f"[PROGRESS] {total_exported} embeddings exported...", file=sys.stderr)

                except ValueError as e:
                    print(f"[ERROR] Word '{word}': Failed to parse vector: {e}")
                    errors += 1
                except Exception as e:
                    print(f"[ERROR] Unexpected error for word '{word}': {e}")
                    errors += 1

    except IOError as e:
        print(f"[ERROR] Cannot write to {output_path}: {e}", file=sys.stderr)
        conn.close()
        return 0, 0, 1

    conn.close()

    print(f"\n[SUMMARY]")
    print(f"  Total exported: {total_exported}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors: {errors}")
    print(f"  Output: {output_path}")

    return total_exported, skipped, errors


def main():
    parser = argparse.ArgumentParser(
        description="Export embeddings from SQLite to C++ CSV format"
    )
    parser.add_argument("--db", required=True, help="Path to embeddings.db")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    exported, skipped, errors = export_embeddings(args.db, args.output)

    if errors > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
