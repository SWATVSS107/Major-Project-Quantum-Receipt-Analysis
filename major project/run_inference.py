"""
run_inference.py - Batch invoice feature extraction using the high-accuracy pipeline.

Usage:
    python run_inference.py --input Dataset --output output/purchase_register.xlsx
    python run_inference.py --input Dataset --output output/purchase_register.xlsx --max_files 10
    python run_inference.py --input Dataset --output output/purchase_register.xlsx --no_ollama
    python run_inference.py --input "Dataset/01-1.pdf" --output output/single.xlsx
"""

import os
import sys
import json
import glob
import argparse
import traceback

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from src.invoice_pipeline.pipeline import process_invoice
from src.invoice_pipeline.export.excel_export import export_to_excel


def collect_files(input_path, max_files=None):
    """Collect all supported invoice files from a folder or single file."""
    supported_exts = (".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".tif")
    if os.path.isfile(input_path):
        return [input_path]
    elif os.path.isdir(input_path):
        all_files = []
        for f in sorted(os.listdir(input_path)):
            if f.lower().endswith(supported_exts):
                all_files.append(os.path.join(input_path, f))
        if max_files:
            all_files = all_files[:max_files]
        return all_files
    else:
        print(f"[ERROR] Input path not found: {input_path}")
        return []


def main():
    parser = argparse.ArgumentParser(description="Invoice AI - Batch Feature Extraction")
    parser.add_argument("--input", required=True, help="Input folder or single file path")
    parser.add_argument("--output", default="output/purchase_register.xlsx", help="Output Excel file path")
    parser.add_argument("--max_files", type=int, default=None, help="Max number of files to process")
    parser.add_argument("--no_ollama", action="store_true", help="Disable Ollama LLM verification (faster, less accurate)")
    parser.add_argument("--save_json", action="store_true", help="Also save results as JSON")
    args = parser.parse_args()

    use_ollama = not args.no_ollama
    files = collect_files(args.input, max_files=args.max_files)

    if not files:
        print("[ERROR] No invoice files found.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Invoice AI - High Accuracy Extraction Pipeline")
    print(f"{'='*60}")
    print(f"  Files to process : {len(files)}")
    print(f"  Ollama LLM       : {'ENABLED (Llama 3.2)' if use_ollama else 'DISABLED'}")
    print(f"  Output Excel     : {args.output}")
    print(f"{'='*60}\n")

    results = []
    success_count = 0
    fail_count = 0

    for i, file_path in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] Processing: {os.path.basename(file_path)}")
        try:
            result = process_invoice(file_path, use_ollama=use_ollama)
            results.append(result)
            success_count += 1
        except Exception as e:
            print(f"[ERROR] Failed to process {file_path}: {e}")
            traceback.print_exc()
            fail_count += 1

    # Save Excel
    if results:
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
        export_to_excel(results, args.output)

    # Optionally save JSON
    if args.save_json and results:
        json_path = os.path.splitext(args.output)[0] + ".json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[EXPORT] JSON saved: {json_path}")

    print(f"\n{'='*60}")
    print(f"  Batch Complete!")
    print(f"  Processed: {success_count} success / {fail_count} failed")
    print(f"  Excel: {args.output}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
