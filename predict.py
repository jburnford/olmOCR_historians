"""
Full-PDF olmOCR predictor for Replicate.

Processes an entire PDF document through olmOCR and returns a single
markdown file. Runs vLLM as a subprocess server for batched inference.
"""

import concurrent.futures
import json
import os
import subprocess
import tempfile
import time
import urllib.request
import urllib.error
from pathlib import Path as PathLib

from cog import BasePredictor, Input, Path

MODEL_CACHE = "checkpoints"
MODEL_URL = "https://weights.replicate.delivery/default/allenai/olmOCR-7B-0225-preview/model.tar"
VLLM_PORT = 8000
VLLM_URL = f"http://localhost:{VLLM_PORT}"
MODEL_PATH = os.path.join(MODEL_CACHE, "olmOCR-7B-0225-preview")


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
        """Download weights and start vLLM server."""
        os.makedirs(MODEL_CACHE, exist_ok=True)

        if not os.path.exists(MODEL_PATH):
            download_weights(MODEL_URL, MODEL_PATH)

        print("Starting vLLM server...")
        self.server_process = subprocess.Popen(
            [
                "python", "-m", "vllm.entrypoints.openai.api_server",
                "--model", MODEL_PATH,
                "--served-model-name", "olmocr",
                "--port", str(VLLM_PORT),
                "--gpu-memory-utilization", "0.9",
                "--max-model-len", "8192",
                "--dtype", "bfloat16",
                "--trust-remote-code",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        self._wait_for_server()
        print("vLLM server ready.")

    def _wait_for_server(self, timeout=600):
        """Poll until vLLM server is healthy."""
        start = time.time()
        while time.time() - start < timeout:
            try:
                req = urllib.request.urlopen(f"{VLLM_URL}/health", timeout=5)
                if req.status == 200:
                    return
            except (urllib.error.URLError, ConnectionError, OSError):
                pass
            if self.server_process.poll() is not None:
                stdout = self.server_process.stdout.read().decode() if self.server_process.stdout else ""
                raise RuntimeError(
                    f"vLLM server exited with code {self.server_process.returncode}:\n{stdout[-3000:]}"
                )
            time.sleep(3)
        raise RuntimeError("vLLM server failed to start within timeout")

    def _process_page(self, pdf_path: str, page_num: int) -> str:
        """Render a page and send it to the vLLM server for OCR."""
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

        request_data = json.dumps({
            "model": "olmocr",
            "messages": [
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
            ],
            "max_tokens": 4096,
            "temperature": 0.8,
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{VLLM_URL}/v1/chat/completions",
            data=request_data,
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req, timeout=300)
        result = json.loads(resp.read())
        raw = result["choices"][0]["message"]["content"]
        return extract_text(raw)

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

        print(f"Processing {total_pages} pages...")
        start_time = time.time()

        # Send all pages concurrently — vLLM batches them on the GPU
        results = [None] * total_pages
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = {}
            for page_num in range(1, total_pages + 1):
                future = executor.submit(self._process_page, pdf_path, page_num)
                futures[future] = page_num - 1

            done = 0
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                done += 1
                try:
                    results[idx] = future.result()
                    print(f"  [{done}/{total_pages}] Page {idx + 1} done")
                except Exception as e:
                    print(f"  [{done}/{total_pages}] Page {idx + 1} failed: {e}")
                    results[idx] = f"[Page {idx + 1}: failed to process]"

        md_parts = [
            r.strip() if r else f"[Page {i+1}: no output]"
            for i, r in enumerate(results)
        ]
        markdown = "\n\n".join(md_parts)

        total_elapsed = time.time() - start_time
        pps = total_pages / total_elapsed if total_elapsed > 0 else 0
        print(f"Done. {total_pages} pages in {total_elapsed:.1f}s ({pps:.2f} pages/sec)")

        output_path = PathLib(tempfile.mkdtemp()) / "output.md"
        output_path.write_text(markdown, encoding="utf-8")
        return Path(str(output_path))
