"""Production adapters using only public PaddleOCR/PaddleX model interfaces."""
from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
from typing import Any, Mapping

from .pipeline import (
    ParserConfig,
    RegionPipeline,
    PageParseResult,
    public_result_dict,
)
from .process_output import decode_process_output


def _set_model_environment(root: Path) -> None:
    # PaddleOCR-VL-1.6 本地 GPU 解码存在分配器无限等待报告；必须在导入 Paddle 前设置。
    # 用户仍可通过同名环境变量显式覆盖；这不是 OCR 识别参数或超时替代品。
    os.environ.setdefault("FLAGS_allocator_strategy", "naive_best_fit")
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(root / ".cache" / "paddlex"))
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    os.environ.setdefault("HF_HOME", str(root / ".cache" / "huggingface-paddleocr"))


def _print_worker_log(line: bytes | str) -> None:
    # 原生库可能绕过 redirect_stdout，直接往 fd 1 写中文 Windows 本地编码日志。
    # 只有诊断显示允许替换未知字节；JSON 协议和知识正文始终严格使用 UTF-8。
    print("[OCR worker] " + decode_process_output(line).rstrip(), file=sys.stderr, flush=True)


def _read_response(stream: Any, timeout: float) -> dict:
    responses: queue.Queue = queue.Queue(maxsize=1)

    def read() -> None:
        try:
            while True:
                raw_line = stream.readline()
                if not raw_line:
                    raise RuntimeError("PaddleOCR worker closed stdout; check worker logs")
                try:
                    line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                except UnicodeDecodeError as exc:
                    if raw_line.lstrip().startswith(b"{"):
                        raise RuntimeError(
                            "OCR worker JSON protocol is not valid UTF-8; "
                            "check PYTHONIOENCODING and the worker stdout configuration"
                        ) from exc
                    _print_worker_log(raw_line)
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    _print_worker_log(line)
                    continue
                if isinstance(payload, dict) and {"event", "outputs", "error"}.intersection(payload):
                    responses.put(payload)
                    return
                _print_worker_log(line)
        except Exception as exc:
            responses.put(exc)

    threading.Thread(target=read, daemon=True).start()
    try:
        response = responses.get(timeout=timeout)
    except queue.Empty as exc:
        raise TimeoutError("Timed out waiting for PaddleOCR worker") from exc
    if isinstance(response, Exception):
        raise response
    return response


class PaddleModelBackend:
    """Separate layout, VLM, and light-OCR models behind a small public adapter."""

    def __init__(
        self,
        root: str | Path,
        config: ParserConfig,
        *,
        layout_model_name: str = "PP-DocLayoutV3",
        layout_model_dir: str | Path | None = None,
        vlm_model_name: str = "PaddleOCR-VL-1.6-0.9B",
        vlm_model_dir: str | Path | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.config = config
        _set_model_environment(self.root)
        self.worker = None
        self.layout_model = None
        self.vlm_model = None
        try:
            import paddlex

            layout_dir = Path(layout_model_dir) if layout_model_dir else self.root / "models" / "PP-DocLayoutV3_safetensors"
            vlm_dir = Path(vlm_model_dir) if vlm_model_dir else Path("models") / "PaddleOCR-VL-1.6"
            layout_device = os.environ.get("PADDLEOCR_LAYOUT_DEVICE", os.environ.get("PADDLEOCR_VLM_DEVICE", "gpu:0"))
            vlm_device = os.environ.get("PADDLEOCR_VLM_DEVICE", "gpu:0")
            self.layout_model = paddlex.create_model(
                model_name=layout_model_name,
                model_dir=str(layout_dir),
                device=layout_device,
                engine=os.environ.get("PADDLEOCR_LAYOUT_ENGINE", "paddle_dynamic"),
                batch_size=1,
            )
            self.vlm_model = paddlex.create_model(
                model_name=vlm_model_name,
                model_dir=str(vlm_dir),
                device=vlm_device,
                engine=os.environ.get("PADDLEOCR_VLM_ENGINE", "paddle_dynamic"),
                batch_size=1,
            )
            self.worker = subprocess.Popen(
                [sys.executable, str(self.root / "scripts" / "light_ocr_worker.py")],
                cwd=self.root,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                # stdout 是二进制协议流：不能把混入的本地编码日志交给 UTF-8 文本包装器。
                bufsize=0,
                env=dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8"),
            )
            ready = _read_response(
                self.worker.stdout,
                float(os.environ.get("PADDLEOCR_WORKER_READY_TIMEOUT", "600")),
            )
            if ready.get("event") != "READY":
                raise RuntimeError(f"OCR worker did not become ready: {ready}")
            print(
                f"[OCR worker] READY in {ready.get('load_seconds', 0):.2f}s "
                f"({ready.get('inference_engine')}, {ready.get('device')})",
                file=sys.stderr,
                flush=True,
            )
        except BaseException:
            self.close()
            raise

    def detect_layout(self, image: Any) -> Mapping[str, Any]:
        results = list(self.layout_model.predict(image))
        if len(results) != 1:
            raise RuntimeError(f"Expected one layout result, got {len(results)}")
        data = dict(public_result_dict(results[0]))
        # DetResult may retain the full page array for visualization.  It is not
        # layout metadata and must never be expanded into layout.json.
        data.pop("input_img", None)
        data.pop("input_image", None)
        return data

    def recognize_vlm(self, groups: dict[tuple[int, int], dict]) -> None:
        for (min_pixels, max_pixels), group in groups.items():
            items = [
                {"image": image, "query": query}
                for image, query in zip(group["images"], group["queries"])
            ]
            has_spotting = any(query == "Spotting:" for query in group["queries"])
            results = list(self.vlm_model.predict(
                items,
                use_cache=True,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                max_new_tokens=self.config.max_new_tokens,
                skip_special_tokens=not has_spotting,
            ))
            if len(results) != len(items):
                raise RuntimeError("Public VLM result count mismatch")
            group["vlm_results"] = [public_result_dict(result) for result in results]

    def recognize_text(self, images: list[Any]) -> list[dict[str, Any]]:
        import numpy as np

        with tempfile.TemporaryDirectory(prefix="light-ocr-") as temp_dir:
            paths = []
            for index, image in enumerate(images):
                path = Path(temp_dir) / f"{index:04d}.npy"
                np.save(path, image, allow_pickle=False)
                paths.append(str(path))
            if self.worker is None or self.worker.poll() is not None:
                raise RuntimeError("PaddleOCR worker is not running")
            self.worker.stdin.write((json.dumps({"images": paths}) + "\n").encode("utf-8"))
            self.worker.stdin.flush()
            response = _read_response(
                self.worker.stdout,
                float(os.environ.get("PADDLEOCR_WORKER_INFER_TIMEOUT", "600")),
            )
        if "error" in response:
            raise RuntimeError(response["error"])
        outputs = response.get("outputs")
        if not isinstance(outputs, list) or len(outputs) != len(images):
            raise RuntimeError("Light OCR result count mismatch")
        return outputs

    def close(self) -> None:
        worker = getattr(self, "worker", None)
        if worker is not None:
            if worker.poll() is None:
                try:
                    worker.stdin.write(b'{"stop":true}\n')
                    worker.stdin.flush()
                    worker.wait(timeout=10)
                except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                    worker.kill()
                    worker.wait()
            for stream in (worker.stdin, worker.stdout):
                if stream:
                    try:
                        stream.close()
                    except OSError:
                        pass
        self.worker = None
        self.layout_model = None
        self.vlm_model = None


class PageRuntime:
    """Resident, serial runtime shared by all files in a folder batch."""

    def __init__(self, root: str | Path, config: ParserConfig) -> None:
        self.backend = PaddleModelBackend(root, config)
        self.parser = RegionPipeline(self.backend, config)
        self.lock = threading.Lock()

    def parse_page(self, *args, **kwargs) -> PageParseResult:
        with self.lock:
            return self.parser.parse_image(*args, **kwargs)

    def close(self) -> None:
        self.backend.close()
