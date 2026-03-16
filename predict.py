"""
Full-PDF olmOCR predictor for Replicate.

Processes an entire PDF document through olmOCR and returns a single
markdown file. Uses vLLM for batched inference across all pages.
"""

import base64
import subprocess
import tempfile
from io import BytesIO
from pathlib import Path as PathLib

from cog import BasePredictor, Input, Path
from PIL import Image


MODEL_ID = "allenai/olmOCR-2-7B-1025-FP8"


def get_page_count(pdf_path: str) -> int:
    """Get total page count from a PDF using pdfinfo."""
    result = subprocess.run(
        ["pdfinfo", pdf_path],
        capture_output=True, text=True, timeout=30
    )
    for line in result.stdout.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":")[1].strip())
    raise ValueError(f"Could not determine page count for {pdf_path}")


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory via vLLM."""
        from vllm import LLM
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.llm = LLM(
            model=MODEL_ID,
            gpu_memory_utilization=0.9,
            max_model_len=8192,
            dtype="auto",
            trust_remote_code=True,
            limit_mm_per_prompt={"image": 1},
        )

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
        from olmocr.data.renderpdf import render_pdf_to_base64png
        from olmocr.prompts import build_finetuning_prompt
        from olmocr.prompts.anchor import get_anchor_text
        from vllm import SamplingParams

        pdf_path = str(pdf)
        total_pages = get_page_count(pdf_path)

        if max_pages > 0:
            total_pages = min(total_pages, max_pages)

        print(f"Processing {total_pages} pages from {pdf_path}")

        # Build prompts and images for all pages
        prompts = []
        images = []
        failed_pages = []

        for page_num in range(1, total_pages + 1):
            try:
                image_base64 = render_pdf_to_base64png(
                    pdf_path, page_num, target_longest_image_dim=1024
                )
                anchor_text = get_anchor_text(
                    pdf_path, page_num,
                    pdf_engine="pdfreport",
                    target_length=4000,
                )
                prompt_text = build_finetuning_prompt(anchor_text)

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}"
                                },
                            },
                        ],
                    }
                ]

                text_prompt = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                pil_image = Image.open(BytesIO(base64.b64decode(image_base64)))

                prompts.append(text_prompt)
                images.append(pil_image)

            except Exception as e:
                print(f"Warning: Failed to render page {page_num}: {e}")
                failed_pages.append(page_num)
                prompts.append(None)
                images.append(None)

        # Batch inference through vLLM (only valid pages)
        valid_indices = [i for i in range(len(prompts)) if prompts[i] is not None]
        valid_inputs = [
            {
                "prompt": prompts[i],
                "multi_modal_data": {"image": images[i]},
            }
            for i in valid_indices
        ]

        params = SamplingParams(
            temperature=0.8,
            max_tokens=4096,
            top_p=0.95,
        )

        print(f"Running inference on {len(valid_inputs)} pages...")
        outputs = self.llm.generate(valid_inputs, sampling_params=params)

        # Map results back to page order
        results = [None] * len(prompts)
        for idx, vllm_output in zip(valid_indices, outputs):
            results[idx] = vllm_output.outputs[0].text

        # Assemble markdown
        md_parts = []
        for i in range(len(results)):
            page_num = i + 1
            if results[i] is not None:
                md_parts.append(results[i].strip())
            else:
                md_parts.append(f"[Page {page_num}: failed to process]")

        markdown = "\n\n".join(md_parts)

        # Write output
        output_path = PathLib(tempfile.mkdtemp()) / "output.md"
        output_path.write_text(markdown, encoding="utf-8")

        print(f"Done. {len(valid_inputs)} pages processed, {len(failed_pages)} failed.")
        return Path(str(output_path))
