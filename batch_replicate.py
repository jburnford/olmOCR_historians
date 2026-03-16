#!/usr/bin/env python3
"""
Batch process PDFs through olmOCR on Replicate.

Usage:
    python batch_replicate.py ./input_pdfs/ ./output_md/

    # With a specific model version:
    python batch_replicate.py ./input_pdfs/ ./output_md/ --model username/olmocr-pdf

    # Force reprocessing of already-completed files:
    python batch_replicate.py ./input_pdfs/ ./output_md/ --overwrite

Requirements:
    pip install replicate

Set your API token:
    export REPLICATE_API_TOKEN=r8_...
"""

import argparse
import os
import sys
import time
from pathlib import Path

try:
    import replicate
except ImportError:
    print("Error: 'replicate' package not installed. Run: pip install replicate")
    sys.exit(1)


DEFAULT_MODEL = "jburnford/olmocr-pdf"


def process_pdf(pdf_path: Path, output_dir: Path, model: str, overwrite: bool) -> bool:
    """Process a single PDF through Replicate and save the markdown output."""
    output_file = output_dir / f"{pdf_path.stem}.md"

    if output_file.exists() and not overwrite:
        print(f"  Skipping (already exists): {output_file.name}")
        return True

    try:
        start = time.time()
        with open(pdf_path, "rb") as f:
            output = replicate.run(
                model,
                input={"pdf": f},
            )

        # Output is a file URL — download it
        if hasattr(output, "read"):
            md_content = output.read()
            if isinstance(md_content, bytes):
                md_content = md_content.decode("utf-8")
        elif isinstance(output, str):
            # Direct string output or URL
            if output.startswith("http"):
                import urllib.request
                with urllib.request.urlopen(output) as resp:
                    md_content = resp.read().decode("utf-8")
            else:
                md_content = output
        else:
            md_content = str(output)

        output_file.write_text(md_content, encoding="utf-8")
        elapsed = time.time() - start
        print(f"  Done ({elapsed:.1f}s) -> {output_file.name}")
        return True

    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Batch process PDFs through olmOCR on Replicate"
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing PDF files")
    parser.add_argument("output_dir", type=Path, help="Directory for markdown output")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Replicate model to use (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Reprocess PDFs even if output already exists"
    )
    args = parser.parse_args()

    if not os.environ.get("REPLICATE_API_TOKEN"):
        print("Error: REPLICATE_API_TOKEN environment variable not set.")
        print("  export REPLICATE_API_TOKEN=r8_...")
        sys.exit(1)

    if not args.input_dir.is_dir():
        print(f"Error: Input directory not found: {args.input_dir}")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(args.input_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDF files found in {args.input_dir}")
        sys.exit(0)

    print(f"Found {len(pdfs)} PDFs in {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model: {args.model}")
    print()

    succeeded = 0
    failed = 0

    for i, pdf in enumerate(pdfs, 1):
        print(f"[{i}/{len(pdfs)}] {pdf.name}")
        if process_pdf(pdf, args.output_dir, args.model, args.overwrite):
            succeeded += 1
        else:
            failed += 1

    print(f"\nComplete: {succeeded} succeeded, {failed} failed out of {len(pdfs)} PDFs")


if __name__ == "__main__":
    main()
