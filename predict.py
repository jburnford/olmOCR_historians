"""
Full-PDF olmOCR predictor for Replicate.

Processes an entire PDF document through olmOCR and returns a single
markdown file. Uses transformers with batched inference for speed.
"""

import base64
import json
import os
import subprocess
import tempfile
import time
from io import BytesIO
from pathlib import Path as PathLib

from cog import BasePredictor, Input, Path

MODEL_CACHE = "checkpoints"
MODEL_URL = "https://weights.replicate.delivery/default/allenai/olmOCR-7B-0225-preview/model.tar"
VISION_URL = "https://weights.replicate.delivery/default/qwen/Qwen2-VL-7B-Instruct/model.tar"

BATCH_SIZE = 4  # Pages processed simultaneously


def download_weights(url, dest):
    start = time.time()
    print("downloading url:", url)
    print("downloading to:", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took:", time.time() - start)


def get_page_count(pdf_path: str) -> int:
    """Get total page count from a PDF using pdfinfo."""
    result = subprocess.run(
        ["pdfinfo", pdf_path],
        capture_output=True, text=True, timeout=30,
    )
    for line in result.stdout.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":")[1].strip())
    raise ValueError(f"Could not determine page count for {pdf_path}")


def extract_text(raw: str) -> str:
    """Extract natural_text from olmOCR JSON response."""
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and "natural_text" in parsed:
            return parsed["natural_text"]
    except (json.JSONDecodeError, TypeError):
        pass
    return raw


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory."""
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        os.makedirs(MODEL_CACHE, exist_ok=True)

        if not os.path.exists(os.path.join(MODEL_CACHE, "olmOCR-7B-0225-preview")):
            download_weights(MODEL_URL, os.path.join(MODEL_CACHE, "olmOCR-7B-0225-preview"))

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            os.path.join(MODEL_CACHE, "olmOCR-7B-0225-preview"),
            torch_dtype=torch.bfloat16,
        ).eval().to(self.device)

        if not os.path.exists(os.path.join(MODEL_CACHE, "Qwen2-VL-7B-Instruct")):
            download_weights(VISION_URL, os.path.join(MODEL_CACHE, "Qwen2-VL-7B-Instruct"))

        self.processor = AutoProcessor.from_pretrained(
            os.path.join(MODEL_CACHE, "Qwen2-VL-7B-Instruct")
        )

    def _prepare_page(self, pdf_path: str, page_num: int):
        """Render a page and build the prompt. Returns (text_prompt, pil_image) or None on failure."""
        from PIL import Image
        from olmocr.data.renderpdf import render_pdf_to_base64png
        from olmocr.prompts import build_finetuning_prompt
        from olmocr.prompts.anchor import get_anchor_text

        try:
            image_base64 = render_pdf_to_base64png(
                pdf_path, page_num, target_longest_image_dim=1024
            )
            anchor_text = get_anchor_text(
                pdf_path, page_num,
                pdf_engine="pdfreport",
                target_length=4000,
            )
            prompt = build_finetuning_prompt(anchor_text)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}",
                            },
                        },
                    ],
                }
            ]

            text_prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            pil_image = Image.open(BytesIO(base64.b64decode(image_base64)))
            return text_prompt, pil_image
        except Exception as e:
            print(f"  Warning: Failed to prepare page {page_num}: {e}")
            return None

    def _generate_batch(self, text_prompts, images):
        """Run batched inference on multiple pages at once."""
        import torch

        inputs = self.processor(
            text=text_prompts,
            images=images,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                temperature=0.8,
                max_new_tokens=4096,
                num_return_sequences=1,
                do_sample=True,
            )

        # Decode each sequence in the batch
        prompt_length = inputs["input_ids"].shape[1]
        new_tokens = outputs[:, prompt_length:]
        decoded = self.processor.tokenizer.batch_decode(
            new_tokens, skip_special_tokens=True
        )
        return [extract_text(d) for d in decoded]

    def predict(
        self,
        pdf: Path = Input(description="PDF file to OCR"),
        max_pages: int = Input(
            description="Maximum pages to process (0 = all pages)",
            default=0,
            ge=0,
        ),
    ) -> Path:
        """Process an entire PDF and return a markdown file."""
        pdf_path = str(pdf)
        total_pages = get_page_count(pdf_path)

        if max_pages > 0:
            total_pages = min(total_pages, max_pages)

        print(f"Processing {total_pages} pages (batch size {BATCH_SIZE})...")
        start_time = time.time()

        # Prepare all pages (rendering + prompt building)
        prepared = []
        for page_num in range(1, total_pages + 1):
            result = self._prepare_page(pdf_path, page_num)
            prepared.append((page_num, result))

        # Process in batches
        all_results = [None] * total_pages
        batch_texts = []
        batch_images = []
        batch_indices = []

        for i, (page_num, prep) in enumerate(prepared):
            if prep is None:
                all_results[i] = f"[Page {page_num}: failed to process]"
                continue

            text_prompt, pil_image = prep
            batch_texts.append(text_prompt)
            batch_images.append(pil_image)
            batch_indices.append(i)

            if len(batch_texts) == BATCH_SIZE:
                print(f"  Batch: pages {[prepared[j][0] for j in batch_indices]}...")
                try:
                    results = self._generate_batch(batch_texts, batch_images)
                    for idx, result in zip(batch_indices, results):
                        all_results[idx] = result
                except Exception as e:
                    print(f"  Batch failed: {e}, falling back to single pages")
                    for idx, t, img in zip(batch_indices, batch_texts, batch_images):
                        try:
                            results = self._generate_batch([t], [img])
                            all_results[idx] = results[0]
                        except Exception as e2:
                            all_results[idx] = f"[Page {prepared[idx][0]}: {e2}]"
                batch_texts = []
                batch_images = []
                batch_indices = []

        # Process remaining pages
        if batch_texts:
            print(f"  Batch: pages {[prepared[j][0] for j in batch_indices]}...")
            try:
                results = self._generate_batch(batch_texts, batch_images)
                for idx, result in zip(batch_indices, results):
                    all_results[idx] = result
            except Exception as e:
                print(f"  Batch failed: {e}, falling back to single pages")
                for idx, t, img in zip(batch_indices, batch_texts, batch_images):
                    try:
                        results = self._generate_batch([t], [img])
                        all_results[idx] = results[0]
                    except Exception as e2:
                        all_results[idx] = f"[Page {prepared[idx][0]}: {e2}]"

        # Assemble markdown
        md_parts = [r.strip() if r else f"[Page {i+1}: no output]" for i, r in enumerate(all_results)]
        markdown = "\n\n".join(md_parts)

        elapsed = time.time() - start_time
        pages_per_sec = total_pages / elapsed if elapsed > 0 else 0
        print(f"Done. {total_pages} pages in {elapsed:.1f}s ({pages_per_sec:.2f} pages/sec)")

        output_path = PathLib(tempfile.mkdtemp()) / "output.md"
        output_path.write_text(markdown, encoding="utf-8")
        return Path(str(output_path))
