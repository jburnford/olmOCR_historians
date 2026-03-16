"""
Full-PDF olmOCR predictor for Replicate.

Processes an entire PDF document through olmOCR and returns a single
markdown file. Uses transformers for inference, processing pages sequentially.
"""

import base64
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

    def _process_page(self, pdf_path: str, page_num: int) -> str:
        """Process a single page and return extracted text."""
        import torch
        from PIL import Image
        from olmocr.data.renderpdf import render_pdf_to_base64png
        from olmocr.prompts import build_finetuning_prompt
        from olmocr.prompts.anchor import get_anchor_text

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

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        pil_image = Image.open(BytesIO(base64.b64decode(image_base64)))
        inputs = self.processor(
            text=[text], images=[pil_image],
            padding=True, return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                temperature=0.8,
                max_new_tokens=4096,
                num_return_sequences=1,
                do_sample=True,
            )

        prompt_length = inputs["input_ids"].shape[1]
        new_tokens = output[:, prompt_length:]
        decoded = self.processor.tokenizer.batch_decode(
            new_tokens, skip_special_tokens=True
        )
        return decoded[0] if decoded else ""

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

        print(f"Processing {total_pages} pages from {pdf_path}")

        md_parts = []
        for page_num in range(1, total_pages + 1):
            try:
                print(f"  Page {page_num}/{total_pages}...")
                text = self._process_page(pdf_path, page_num)
                md_parts.append(text.strip())
            except Exception as e:
                print(f"  Warning: Page {page_num} failed: {e}")
                md_parts.append(f"[Page {page_num}: failed to process]")

        markdown = "\n\n".join(md_parts)

        output_path = PathLib(tempfile.mkdtemp()) / "output.md"
        output_path.write_text(markdown, encoding="utf-8")

        print(f"Done. {total_pages} pages processed.")
        return Path(str(output_path))
