# olmOCR for Historians

Process historical PDF documents into markdown text using [olmOCR](https://github.com/allenai/olmocr) on [Replicate](https://replicate.com).

No GPU required. Pay-per-use (~$0.07 per 100 pages).

## Quick Start (Web UI)

1. Create a free account at [replicate.com](https://replicate.com)
2. Add a credit card in billing settings
3. Go to the model page: [jburnford/olmocr-pdf](https://replicate.com/jburnford/olmocr-pdf)
4. Upload a PDF and click **Run**
5. Download the resulting `.md` file

## Batch Processing

For processing many PDFs at once, use the batch script.

### Setup

```bash
pip install replicate
export REPLICATE_API_TOKEN=r8_your_token_here
```

### Usage

```bash
# Process all PDFs in a folder
python batch_replicate.py ./my_pdfs/ ./output/

# Resume after interruption (skips already-processed files)
python batch_replicate.py ./my_pdfs/ ./output/

# Force reprocessing
python batch_replicate.py ./my_pdfs/ ./output/ --overwrite
```

Output files are named to match input files: `document.pdf` → `document.md`

## Cost

Runs on NVIDIA L40S GPU at $3.51/hour. olmOCR processes roughly 1-2 pages per second, so:

| Document Size | Approximate Cost |
|---------------|-----------------|
| 10 pages      | ~$0.01          |
| 100 pages     | ~$0.07          |
| 1,000 pages   | ~$0.70          |

You only pay for actual processing time. No subscription or minimum.

## Model Details

- **Model**: [olmOCR-2-7B-1025-FP8](https://huggingface.co/allenai/olmOCR-2-7B-1025-FP8) by Allen AI
- **Optimized for**: Historical documents, tables, multi-column layouts
- **Output**: Clean markdown with preserved document structure
