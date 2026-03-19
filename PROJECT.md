# olmOCR for Historians — Project Documentation

## Goal

Provide historians with self-service access to olmOCR for processing historical PDF documents into clean markdown. Users pay per use via Replicate — no GPU required, no subscription, no infrastructure to manage.

**Model page**: [replicate.com/jburnford/olmocr-pdf](https://replicate.com/jburnford/olmocr-pdf)

## Architecture

```
                   ┌─────────────────────────────┐
                   │     Replicate (L40S GPU)     │
                   │                              │
 Upload PDF ──────>│  vLLM Server (subprocess)    │──────> Download .md
                   │    ↕                         │
                   │  olmOCR-7B-0225-preview       │
                   │  (Qwen2-VL fine-tune)        │
                   └─────────────────────────────┘
```

### How It Works

1. User uploads a PDF via the Replicate web UI or API
2. `predict.py` counts pages, renders each to a 1024px PNG
3. olmOCR builds prompts with anchor text (layout-aware text hints)
4. All pages are sent concurrently to a local vLLM server (8 workers)
5. vLLM batches inference on the GPU for throughput
6. Raw JSON responses are parsed to extract `natural_text`
7. Pages are assembled into a single .md file and returned

### Key Components

| File | Purpose |
|------|---------|
| `predict.py` | Cog predictor — vLLM server lifecycle, page rendering, OCR, markdown assembly |
| `cog.yaml` | Docker build config — CUDA 12.1, vLLM 0.6.6, olmocr 0.1.58 |
| `batch_replicate.py` | Client-side batch script for processing folders of PDFs |
| `.github/workflows/push-to-replicate.yml` | CI/CD — auto-builds and pushes to Replicate on git push |

## Deployment

### GitHub Actions CI/CD

Every push to `main` triggers a build via GitHub Actions:

1. `replicate/setup-cog@v2` installs the Cog CLI (v0.16.12)
2. `cog push r8.im/jburnford/olmocr-pdf` builds the Docker image and pushes to Replicate

### Required GitHub Secrets

| Secret | Source | Purpose |
|--------|--------|---------|
| `REPLICATE_CLI_TOKEN` | [replicate.com/auth/token](https://replicate.com/auth/token) | Authenticates `cog push` |

### Replicate Model Settings

- **Hardware**: NVIDIA L40S GPU (48GB VRAM)
- **Cost**: $0.000975/sec ($3.51/hr)

## Dependency Management

The container build is carefully ordered to avoid version conflicts:

```yaml
run:
  # 1. Install vLLM first — brings torch 2.5.1, numpy 1.x, transformers, pydantic v2
  - pip install vllm==0.6.6.post1

  # 2. Install olmocr code only — no deps to avoid pulling torch 2.10 / numpy 2.x
  - pip install --no-deps olmocr==0.1.58

  # 3. Add olmocr's lightweight deps not covered by vLLM
  - pip install pypdfium2 pypdf ftfy lingua-language-detector "numpy<2"
```

### Resolved Compatibility Issues

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| Pydantic v1/v2 conflict | Cog bundles pydantic v1, vLLM needs v2 | vLLM's `--ignore-installed` overwrites; pydantic v2 has v1 compat shim |
| Dual torch versions | olmocr → torch 2.10 in `/dep/`, vLLM → torch 2.5 in site-packages | Install vLLM first, olmocr with `--no-deps` |
| numpy mismatch | olmocr → numpy 2.x, cv2/vLLM need numpy 1.x | Pin `numpy<2` |
| rope_scaling conflict | Model config has both legacy `type` and modern `rope_type` fields | Patch config.json after weight download |
| Cog 0.17 auth change | CLI token vs API token | Use `REPLICATE_CLI_TOKEN` from replicate.com/auth/token |

## Model Details

- **Base model**: [allenai/olmOCR-7B-0225-preview](https://huggingface.co/allenai/olmOCR-7B-0225-preview)
- **Architecture**: Qwen2-VL-7B-Instruct fine-tuned on olmOCR-mix-0225 dataset
- **Weights**: Pre-cached on Replicate CDN (~17GB, downloads in ~14s)
- **Inference**: vLLM 0.6.6 with OpenAI-compatible API, bfloat16 precision
- **Page rendering**: poppler-utils (`pdfinfo`, `pdftoppm`) via olmocr's `render_pdf_to_base64png`
- **Prompt construction**: olmocr's anchor text extraction + `build_finetuning_prompt`

## Performance

| Metric | Value |
|--------|-------|
| Cold start | ~15s (weight download) + ~60s (vLLM model load) |
| Warm throughput | ~1-2 pages/sec (target, with vLLM batching) |
| Cost per 100 pages | ~$0.07 |
| Cost per 1000 pages | ~$0.70 |
| Max pages per run | ~2,500 (30-min Replicate timeout) |

## Usage

### Web UI (Single PDF)

1. Go to [replicate.com/jburnford/olmocr-pdf](https://replicate.com/jburnford/olmocr-pdf)
2. Upload PDF, click Run
3. Download the .md output

### Batch Processing (Many PDFs)

```bash
pip install replicate
export REPLICATE_API_TOKEN=r8_...

# Process a folder of PDFs
python batch_replicate.py ./input_pdfs/ ./output_md/

# Resume after interruption (skips completed files)
python batch_replicate.py ./input_pdfs/ ./output_md/

# Force reprocessing
python batch_replicate.py ./input_pdfs/ ./output_md/ --overwrite
```

Output files match input names: `document.pdf` → `document.md`

## Repository

- **GitHub**: [github.com/jburnford/olmOCR_historians](https://github.com/jburnford/olmOCR_historians)
- **Replicate**: [replicate.com/jburnford/olmocr-pdf](https://replicate.com/jburnford/olmocr-pdf)

## Future Improvements

- **Upgrade to newer olmOCR** (0.4.x+) with GRPO-trained model for better table/math handling
- **FP8 quantization** for faster inference and lower memory
- **SGLang backend** — reportedly 3.1x faster than vLLM for this model
- **Streaming output** — return pages as they complete
- **Cost tracking** in batch script — report per-PDF and total spend
